// Target: does a single-pass (online) softmax pay on the v79?
//
// The kernel of proposal 0060 makes three passes over the row: the maximum, then the exponential
// with its sum into a pad, then the normalization of the pad. The online form merges the first
// two by carrying a running maximum and rescaling the running sum by exp(m_old - m_new) whenever
// the maximum moves. It is the form flash attention uses, where the blocks stream from memory
// and a second visit is not affordable.
//
// This program measures the three quantities that decide the question, without editing the
// kernel: the whole softmax, the maximum pass on its own (which is the most the online form can
// ever save), and an online implementation written here against the same exponential.
//
// The structural point the numbers should confirm: the three-pass form evaluates the exponential
// exactly once for each element and keeps the result. The online form cannot, because the final
// maximum is unknown until the row ends, thus its second pass must either evaluate the
// exponential again or rescale every stored block by exp(m_block - M), and both cost one
// exponential for each vector. The online form therefore trades a cheap load-and-max pass for an
// expensive one.
//
// Arguments: --nsm 512 --iters 12 --dist 0
//   --dist 0 a realistic attention-logit row (normal, sigma 4, one peak), 1 a uniform sweep,
//            2 a row whose maximum arrives last, which is the worst case for the running maximum.
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "hvx-norm.h"
#include "softmax-ops.c"

// SM_X and the grouped exponential exist only in the proposed build. Without the proposal this
// program still compiles, and its online and max-pass cases then use the same group width.
#ifndef SM_X
#define SM_X 4
#endif

// Without the proposal there is no grouped exponential, thus the whole comparison needs it and
// this program measures nothing in a checkout build. Naming LAB_TARGETS=smx keeps that build
// out of the way of the other agents.
#ifndef LAB_PROPOSED
int main(void) {
    lab_init();
    printf("lab: smx this program needs the proposal 0060, which holds the grouped exponential\n");
    return 0;
}
#else
#include "hvx-exp-rr.h"

#define TARGET "smx"

// The scalar reference, in float64.
static void ref_softmax(const float * x, float * y, uint32_t n) {
    double m = (double) x[0];
    for (uint32_t i = 1; i < n; i++) {
        if ((double) x[i] > m) { m = (double) x[i]; }
    }
    double s = 0.0;
    for (uint32_t i = 0; i < n; i++) { s += exp((double) x[i] - m); }
    const double inv = s > 0.0 ? 1.0 / s : 1.0;
    for (uint32_t i = 0; i < n; i++) { y[i] = (float) (exp((double) x[i] - m) * inv); }
}

static double nmse(const float * got, const float * ref, size_t n) {
    double se = 0.0, sr = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double) got[i] - (double) ref[i];
        se += d * d; sr += (double) ref[i] * (double) ref[i];
    }
    return sr > 0.0 ? se / sr : 0.0;
}

// Pass 1 on its own: the maximum of the row, in the same shape the kernel uses. The difference
// between the whole softmax and the two passes that follow is what the online form could save.
static float max_pass_only(const uint8_t * restrict src, int num_elems) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    const int step_of_1 = num_elems >> 5;
    const int nq = step_of_1 & ~(SM_X - 1);
    HVX_Vector mx[SM_X];
    for (int r = 0; r < SM_X; r++) { mx[r] = hvx_vec_splat_f32(((const float *) src)[0]); }
    for (int i = 0; i < nq; i += SM_X) {
        for (int r = 0; r < SM_X; r++) { mx[r] = Q6_Vsf_vmax_VsfVsf(mx[r], v_src[i + r]); }
    }
    for (int i = nq; i < step_of_1; i++) { mx[0] = Q6_Vsf_vmax_VsfVsf(mx[0], v_src[i]); }
    for (int r = 1; r < SM_X; r++) { mx[0] = Q6_Vsf_vmax_VsfVsf(mx[0], mx[r]); }
    float out[32] __attribute__((aligned(128)));
    *(HVX_Vector *) out = hvx_vec_reduce_max_f32(mx[0]);
    return out[0];
}

// The online form: one pass carrying a running maximum and a rescaled running sum, then a second
// pass that evaluates the exponential again against the final maximum. Two passes over the row
// and two evaluations of the exponential for each element.
static void softmax_online(const uint8_t * restrict src, uint8_t * restrict dst, int num_elems) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst = (HVX_Vector *) dst;
    const int step_of_1 = num_elems >> 5;
    const int nq = step_of_1 & ~(SM_X - 1);

    HVX_Vector mx[SM_X], sm[SM_X];
    for (int r = 0; r < SM_X; r++) {
        mx[r] = hvx_vec_splat_f32(((const float *) src)[0]);
        sm[r] = Q6_V_vzero();
    }
    // pass 1: the running maximum and the sum rescaled to it
    for (int i = 0; i < nq; i += SM_X) {
        HVX_Vector mn[SM_X], d[SM_X], e[SM_X], c[SM_X], ce[SM_X];
        for (int r = 0; r < SM_X; r++) { mn[r] = Q6_Vsf_vmax_VsfVsf(mx[r], v_src[i + r]); }
        for (int r = 0; r < SM_X; r++) { c[r] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(mx[r], mn[r])); }
        for (int r = 0; r < SM_X; r++) { d[r] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v_src[i + r], mn[r])); }
        hvx_vec_exp_f32_xn(c, ce);      // the rescale factor, one exponential for each vector
        hvx_vec_exp_f32_xn(d, e);       // the term itself
        for (int r = 0; r < SM_X; r++) {
            sm[r] = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(sm[r]), ce[r]), e[r]);
        }
        for (int r = 0; r < SM_X; r++) { mx[r] = mn[r]; }
    }
    // fold the streams: rescale each to the common maximum
    HVX_Vector m_all = mx[0];
    for (int r = 1; r < SM_X; r++) { m_all = Q6_Vsf_vmax_VsfVsf(m_all, mx[r]); }
    HVX_Vector max_vec = hvx_vec_reduce_max_f32(m_all);
    HVX_Vector acc = Q6_V_vzero();
    for (int r = 0; r < SM_X; r++) {
        HVX_Vector cr[SM_X], ce[SM_X];
        for (int q = 0; q < SM_X; q++) { cr[q] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(mx[r], max_vec)); }
        hvx_vec_exp_f32_xn(cr, ce);
        acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(sm[r]), ce[0])));
    }
    HVX_Vector sum_vec = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(acc));
    HVX_Vector scale = hvx_vec_inverse_f32(sum_vec);

    // pass 2: the exponential again, against the final maximum
    for (int i = 0; i < nq; i += SM_X) {
        HVX_Vector d[SM_X], e[SM_X];
        for (int r = 0; r < SM_X; r++) { d[r] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v_src[i + r], max_vec)); }
        hvx_vec_exp_f32_xn(d, e);
        for (int r = 0; r < SM_X; r++) {
            v_dst[i + r] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(e[r], scale));
        }
    }
}

#define TIME(label, lanes, elems, call)                                              \
    do {                                                                              \
        uint64_t best = UINT64_MAX;                                                   \
        for (uint32_t it = 0; it < iters; it++) {                                     \
            LAB_BARRIER();                                                            \
            const uint64_t t0 = lab_cycles();                                         \
            call;                                                                     \
            const uint64_t t1 = lab_cycles();                                         \
            LAB_BARRIER();                                                            \
            if (t1 - t0 < best) { best = t1 - t0; }                                   \
        }                                                                             \
        printf("lab: %s %-22s cycles %7llu   per_vec %7.2f\n", TARGET, label,          \
               (unsigned long long) best, (double) best / ((double) (elems) / (lanes))); \
    } while (0)

int main(int argc, char ** argv) {
    const uint32_t nsm   = (uint32_t) lab_arg_long(argc, argv, "--nsm", 512);
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 12);
    const int      dist  = (int) lab_arg_long(argc, argv, "--dist", 0);

    lab_init();
#ifdef LAB_PROPOSED
    printf("lab: %s build = proposed\n", TARGET);
#else
    printf("lab: %s build = checkout\n", TARGET);
#endif

    float * xs  = lab_ddr_alloc((size_t) nsm * sizeof(float) + 256, 128);
    float * ys  = lab_ddr_alloc((size_t) nsm * sizeof(float) + 256, 128);
    float * yo  = lab_ddr_alloc((size_t) nsm * sizeof(float) + 256, 128);
    float * rs  = lab_ddr_alloc((size_t) nsm * sizeof(float) + 256, 128);
    float * pad = (float *) lab_vtcm_alloc((size_t) nsm * sizeof(float) + 256, 128);

    // The input distribution decides how often the running maximum moves.
    const char * dname = "normal sigma 4, one peak";
    lab_fill_f32(xs, nsm, -12.0f, 12.0f);
    if (dist == 0) {
        for (uint32_t i = 0; i < nsm; i++) { xs[i] = xs[i] * 0.33f; }
        xs[nsm / 3] = 11.0f;
    } else if (dist == 1) {
        dname = "uniform sweep";
        for (uint32_t i = 0; i < nsm; i++) { xs[i] = -12.0f + 24.0f * (float) i / (float) (nsm - 1); }
    } else {
        dname = "rising, the maximum arrives last";
        for (uint32_t i = 0; i < nsm; i++) { xs[i] = 0.02f * (float) i; }
    }
    printf("lab: %s nsm = %u  iters = %u  distribution = %s\n", TARGET, nsm, iters, dname);

    TIME("softmax_3pass", 32, nsm, hvx_fast_softmax_f32((const uint8_t *) xs, (uint8_t *) ys, (uint8_t *) pad, (int) nsm));
    ref_softmax(xs, rs, nsm);
    printf("lab: %s softmax_3pass nmse = %.4g\n", TARGET, nmse(ys, rs, nsm));

    volatile float sink = 0.0f;
    TIME("max_pass_only", 32, nsm, sink = max_pass_only((const uint8_t *) xs, (int) nsm));
    (void) sink;

    TIME("softmax_online", 32, nsm, softmax_online((const uint8_t *) xs, (uint8_t *) yo, (int) nsm));
    printf("lab: %s softmax_online nmse = %.4g\n", TARGET, nmse(yo, rs, nsm));

#ifdef LAB_PROPOSED
    // The second range-reduction stage, against the exponential the row actually uses.
    {
        static float xe[128] __attribute__((aligned(128)));
        static float ye[128] __attribute__((aligned(128)));
        double worst_old = 0.0, worst_new = 0.0;
        for (int blk = 0; blk < 64; blk++) {
            for (int i = 0; i < 128; i++) {
                xe[i] = -87.0f + 174.0f * (float) (blk * 128 + i) / (64.0f * 128.0f - 1.0f);
            }
            HVX_Vector a[4], o[4];
            for (int r = 0; r < 4; r++) { a[r] = *(HVX_Vector *) (xe + 32 * r); }
            hvx_vec_exp_f32_xn(a, o);
            for (int r = 0; r < 4; r++) { *(HVX_Vector *) (ye + 32 * r) = o[r]; }
            for (int i = 0; i < 128; i++) {
                const double ref = exp((double) xe[i]);
                const double e = ref > 0.0 ? fabs((double) ye[i] - ref) / ref : 0.0;
                if (e > worst_old) { worst_old = e; }
            }
            for (int r = 0; r < 4; r++) { a[r] = *(HVX_Vector *) (xe + 32 * r); }
            hvx_vec_exp_rr_f32_xn(a, o, 4);
            for (int r = 0; r < 4; r++) { *(HVX_Vector *) (ye + 32 * r) = o[r]; }
            for (int i = 0; i < 128; i++) {
                const double ref = exp((double) xe[i]);
                const double e = ref > 0.0 ? fabs((double) ye[i] - ref) / ref : 0.0;
                if (e > worst_new) { worst_new = e; }
            }
        }
        printf("lab: %s exp relative error over [-87, 87]: sibling %.3e, with the table %.3e\n",
               TARGET, worst_old, worst_new);

        // Every result must stay live or the compiler removes the loop: the first form of this
        // measurement fed back only o[0] and reported 1 cycle for the whole loop. Each variant
        // feeds all four vectors back and then stores them, thus both carry the same dependence.
        HVX_Vector a[4], o[4];
        for (int r = 0; r < 4; r++) { a[r] = *(HVX_Vector *) (xe + 32 * r); }
        TIME("exp_sibling_x4", 32, 128 * 40,
             for (int q = 0; q < 40; q++) { hvx_vec_exp_f32_xn(a, o); for (int r = 0; r < 4; r++) { a[r] = o[r]; } });
        for (int r = 0; r < 4; r++) { *(HVX_Vector *) (ye + 32 * r) = a[r]; }
        for (int r = 0; r < 4; r++) { a[r] = *(HVX_Vector *) (xe + 32 * r); }
        TIME("exp_table_x4", 32, 128 * 40,
             for (int q = 0; q < 40; q++) { hvx_vec_exp_rr_f32_xn(a, o, 4); for (int r = 0; r < 4; r++) { a[r] = o[r]; } });
        for (int r = 0; r < 4; r++) { *(HVX_Vector *) (ye + 32 * r) = a[r]; }
        printf("lab: %s sink %.3g\n", TARGET, (double) ye[0] + (double) ye[64]);

        // The throughput form, which is what the row loop of the softmax actually does: many
        // independent vectors, no value carried from one group to the next. The feedback form
        // above measures the length of the dependent chain instead, and the two answer
        // different questions. Fewer multiplies can only show here.
        {
            enum { NV = 64 };
            static float  big[NV * 32] __attribute__((aligned(128)));
            static float  obig[NV * 32] __attribute__((aligned(128)));
            for (int i = 0; i < NV * 32; i++) {
                big[i] = -20.0f + 40.0f * (float) i / (float) (NV * 32 - 1);
            }
            const HVX_Vector * vin = (const HVX_Vector *) big;
            HVX_Vector * vout = (HVX_Vector *) obig;
            TIME("exp_sibling_stream", 32, NV * 32,
                 for (int g = 0; g < NV; g += 4) { hvx_vec_exp_f32_xn(vin + g, vout + g); });
            TIME("exp_table_stream", 32, NV * 32,
                 for (int g = 0; g < NV; g += 4) { hvx_vec_exp_rr_f32_xn(vin + g, vout + g, 4); });
            double w_old = 0.0, w_new = 0.0;
            for (int g = 0; g < NV; g += 4) { hvx_vec_exp_f32_xn(vin + g, vout + g); }
            for (int i = 0; i < NV * 32; i++) {
                const double ref = exp((double) big[i]);
                const double e = fabs((double) obig[i] - ref) / ref;
                if (e > w_old) { w_old = e; }
            }
            for (int g = 0; g < NV; g += 4) { hvx_vec_exp_rr_f32_xn(vin + g, vout + g, 4); }
            for (int i = 0; i < NV * 32; i++) {
                const double ref = exp((double) big[i]);
                const double e = fabs((double) obig[i] - ref) / ref;
                if (e > w_new) { w_new = e; }
            }
            printf("lab: %s stream relative error over [-20, 20]: sibling %.3e, table %.3e\n",
                   TARGET, w_old, w_new);
        }
    }
#endif

    return 0;
}
#endif /* LAB_PROPOSED */
