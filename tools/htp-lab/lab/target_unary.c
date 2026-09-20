// Target 8: the softplus op of unary-ops.c, the only scalar element loop of that file.
//
// The op of the checkout evaluates logf(1.0f + expf(x)) one element at a time, at both of its
// sites (softplus_f32 and tile_unary_softplus_f32). On the phone it measures 22.3 ms of the
// 536 ms of DSP op time of a 512-token prefill of the 4B Q8_0, at 2.24 instructions per packet
// with CU_BUSY at 70.9 %, which is the signature of scalar float work.
//
// The program times that scalar loop and, with the proposal applied (LAB_PROPOSED), the two
// vector paths of hvx-softplus.h against it. Each path is checked against a float64 reference.
// The report gives the cycles for each vector of 32 f32 elements, thus the paths compare
// directly whatever the row length.
//
// The program then sweeps the element count, because one element count is not sufficient. The
// loop body of hvx-softplus.h does the full groups of four vectors first and then one tail
// block for the rest. The compiler inlines the four-vector routine one time for each block, and
// it can give the two copies different code. An element count that is a multiple of 128 runs
// the first block only. The model calls the op with ne0 = ssm_dt_rank, which is 16 for the 2B
// and 32 for the 4B, thus the device runs the tail block and no other block. On 2026-09-20 a
// seed of the wrong type in the Horner accumulator gave -1.43e36 in the tail block and the
// correct result in the main block, and a measurement at 1024 elements saw nothing.
//
// The sweep also makes sure that no path writes an element after the row. The tail block writes
// a part of a vector, thus a wrong mask corrupts the next row.
//
// The input range comes from the model: the softplus input is alpha + ssm_dt, and the ssm_dt
// bias of the 2B holds values from -12.3 to +10.3. Thus the default range of 12 is applicable.
//
// The program stops with the status 1 when one path is outside its tolerance at one count.
//
// Arguments: --n 2048 --rows 8 --iters 3 --range 12
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "hvx-utils.h"

#ifdef LAB_PROPOSED
#include "hvx-softplus.h"
#endif

#define TARGET "unary"

// The largest element count of the sweep, and the number of guard elements after the row
#define LAB_SWEEP_MAXN 512
#define LAB_SWEEP_GUARD 32

// The bit pattern of each guard element and of each element of the row before a call
#define LAB_SWEEP_FILL 0x7f8abcde

// The limits of the f32 paths. The worst measured error of the vector path is 4.8e-7 absolute
// and 6.6e-7 relative over x in [-12, 12], thus these limits give a margin of 20.
#define LAB_F32_ABS_TOL 1.0e-5
#define LAB_F32_REL_TOL 1.0e-5

// The limit of the int16 path. The header of hvx-softplus.h documents an unbounded relative
// error for x below -8, thus this path takes a check of the absolute error only.
#define LAB_I16_ABS_TOL 1.0e-3

typedef void (*lab_row_fn)(const float * restrict, float * restrict, uint32_t);

// The scalar loop of the checkout, both sites hold it verbatim.
static void softplus_scalar(const float * restrict src, float * restrict dst, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const float x = src[i];
        dst[i] = (x > 20.0f) ? x : logf(1.0f + expf(x));
    }
}

// The reference, in float64 and with no cancellation.
static double softplus_ref(double x) {
    return (x > 0.0 ? x : 0.0) + log1p(exp(-fabs(x)));
}

// Time one row function over the rows, and report its error against the reference. O(rows * n).
static void measure(const char * name, lab_row_fn fn,
                    const float * src, float * dst, const double * ref,
                    uint32_t n, uint32_t rows, uint32_t iters) {
    fn(src, dst, n);                                   // the warm-up run fills the caches

    uint64_t best = UINT64_MAX;
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        for (uint32_t r = 0; r < rows; r++) {
            fn(src + (size_t) r * n, dst + (size_t) r * n, n);
        }
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best) {
            best = t1 - t0;
        }
    }

    // the error of the first row, which every row repeats
    fn(src, dst, n);
    double se = 0.0, sr = 0.0, worst_abs = 0.0, worst_rel = 0.0, worst_rel_tail = 0.0;
    double at_abs = 0.0, at_rel = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        const double d = (double) dst[i] - ref[i];
        se += d * d;
        sr += ref[i] * ref[i];
        const double a = fabs(d);
        const double rel = a / ref[i];
        if (a > worst_abs)   { worst_abs = a;  at_abs = (double) src[i]; }
        if (rel > worst_rel) { worst_rel = rel; at_rel = (double) src[i]; }
        if (src[i] < -8.0f && rel > worst_rel_tail) { worst_rel_tail = rel; }
    }

    const double vectors = (double) rows * n / VLEN_FP32;
    printf("lab: %s %-14s cyc/vec %9.2f  cyc/elem %7.3f  nmse %9.2e  max abs %9.2e (x %+.3f)"
           "  max rel %9.2e (x %+.3f)  max rel x<-8 %9.2e\n",
           TARGET, name, (double) best / vectors, (double) best / ((double) rows * n),
           sr > 0.0 ? se / sr : 0.0, worst_abs, at_abs, worst_rel, at_rel, worst_rel_tail);
}

#ifdef LAB_PROPOSED
static void softplus_vec_f32(const float * restrict src, float * restrict dst, uint32_t n) {
    hvx_softplus_f32_aa((uint8_t *) dst, (const uint8_t *) src, n);
}

static void softplus_vec_i16(const float * restrict src, float * restrict dst, uint32_t n) {
    hvx_softplus_i16_f32_aa((uint8_t *) dst, (const uint8_t *) src, n);
}
#endif

// The result of one path at one element count
struct sweep_result {
    double   max_abs;       // the largest absolute error
    double   max_rel;       // the largest relative error
    double   at;            // the input of the largest absolute error
    uint32_t overwrite;     // the number of guard elements that the path wrote
    uint32_t bad;           // the number of elements outside the tolerance
};

// Run one path at one element count and compare it with the float64 reference. O(n).
//
// The function fills the row and the guard elements after the row with one pattern, thus a
// write after the row is visible. The source row holds LAB_SWEEP_GUARD more elements than the
// count, because the tail block reads one full vector.
//
// Args:
//     fn: The row function
//     src: The source row, 128-byte aligned
//     dst: The destination row, 128-byte aligned, with LAB_SWEEP_GUARD elements after the row
//     n: The number of elements
//     abs_tol: The limit of the absolute error
//     rel_tol: The limit of the relative error, or 0 for a check of the absolute error only
//
// Returns:
//     The errors, the number of elements outside the tolerance, and the number of guard
//     elements that the path wrote
static struct sweep_result sweep_one(lab_row_fn fn, const float * src, float * dst,
                                     uint32_t n, double abs_tol, double rel_tol) {
    struct sweep_result r = { 0.0, 0.0, 0.0, 0, 0 };
    const uint32_t fill = LAB_SWEEP_FILL;

    for (uint32_t i = 0; i < n + LAB_SWEEP_GUARD; i++) {
        memcpy(&dst[i], &fill, sizeof(fill));
    }

    fn(src, dst, n);

    for (uint32_t i = 0; i < n; i++) {
        const double ref = softplus_ref((double) src[i]);
        const double a   = fabs((double) dst[i] - ref);
        const double rel = (ref > 0.0) ? a / ref : 0.0;
        if (a > r.max_abs) { r.max_abs = a; r.at = (double) src[i]; }
        if (rel > r.max_rel) { r.max_rel = rel; }
        if (a > abs_tol && (rel_tol == 0.0 || rel > rel_tol)) {
            r.bad++;
        }
    }
    for (uint32_t i = n; i < n + LAB_SWEEP_GUARD; i++) {
        uint32_t got;
        memcpy(&got, &dst[i], sizeof(got));
        if (got != fill) {
            r.overwrite++;
        }
    }
    return r;
}

// Sweep the element count for each path. O(sum of the counts).
//
// Args:
//     src: The source row, 128-byte aligned, with LAB_SWEEP_MAXN + LAB_SWEEP_GUARD elements
//     dst: The destination row, of the same dimension
//
// Returns:
//     The number of (path, count) pairs that failed
static uint32_t sweep(const float * src, float * dst) {
    // The model uses 16 (the 2B) and 32 (the 4B). The counts 128 and 256 are multiples of four
    // vectors, thus they run the main block of the loop body and no tail block.
    static const uint32_t counts[] = { 1,  2,  3,  8,  15, 16,  17,  31,  32,  33, 48,
                                       63, 64, 65, 96, 127, 128, 129, 160, 255, 256 };
    const uint32_t ncount = (uint32_t) (sizeof(counts) / sizeof(counts[0]));
    uint32_t fail = 0;

    printf("lab: %s sweep: the count, then the largest absolute error of each path\n", TARGET);
    for (uint32_t k = 0; k < ncount; k++) {
        const uint32_t n = counts[k];
        const struct sweep_result s = sweep_one(softplus_scalar, src, dst, n,
                                                LAB_F32_ABS_TOL, LAB_F32_REL_TOL);
        uint32_t bad = s.bad + s.overwrite;
#ifdef LAB_PROPOSED
        const struct sweep_result v = sweep_one(softplus_vec_f32, src, dst, n,
                                                LAB_F32_ABS_TOL, LAB_F32_REL_TOL);
        const struct sweep_result q = sweep_one(softplus_vec_i16, src, dst, n,
                                                LAB_I16_ABS_TOL, 0.0);
        bad += v.bad + v.overwrite + q.bad + q.overwrite;
        printf("lab: %s sweep n = %4u  scalar %9.2e  vector_f32 %9.2e (x %+.3f)  vector_i16 %9.2e"
               "  bad %u/%u/%u  after the row %u/%u/%u  %s\n",
               TARGET, n, s.max_abs, v.max_abs, v.at, q.max_abs,
               (unsigned) s.bad, (unsigned) v.bad, (unsigned) q.bad,
               (unsigned) s.overwrite, (unsigned) v.overwrite, (unsigned) q.overwrite,
               bad ? "FAIL" : "ok");
#else
        printf("lab: %s sweep n = %4u  scalar %9.2e  bad %u  after the row %u  %s\n",
               TARGET, n, s.max_abs, (unsigned) s.bad, (unsigned) s.overwrite,
               bad ? "FAIL" : "ok");
#endif
        if (bad) {
            fail++;
        }
    }
    printf("lab: %s sweep: %u of %u counts failed\n", TARGET, (unsigned) fail, (unsigned) ncount);
    return fail;
}

int main(int argc, char ** argv) {
    const uint32_t n     = (uint32_t) lab_arg_long(argc, argv, "--n", 2048);
    const uint32_t rows  = (uint32_t) lab_arg_long(argc, argv, "--rows", 8);
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 3);
    const float    range = (float) lab_arg_long(argc, argv, "--range", 12);

    lab_init();

    const size_t total = (size_t) rows * n;
    float *  src = lab_ddr_alloc(total * sizeof(float) + 256, 128);
    float *  dst = lab_ddr_alloc(total * sizeof(float) + 256, 128);
    double * ref = lab_ddr_alloc((size_t) n * sizeof(double), 128);

    lab_fill_f32(src, n, -range, range);
    // the first lanes cover the ends of the range, which the random fill reaches rarely
    const float edges[8] = { 0.0f, -1e-4f, 1e-4f, -25.0f, 25.0f, -8.5f, 8.5f, -0.5f };
    for (uint32_t i = 0; i < 8 && i < n; i++) {
        src[i] = edges[i];
    }
    for (uint32_t r = 1; r < rows; r++) {
        memcpy(src + (size_t) r * n, src, (size_t) n * sizeof(float));
    }
    for (uint32_t i = 0; i < n; i++) {
        ref[i] = softplus_ref((double) src[i]);
    }

    printf("lab: %s n = %u rows = %u range = %.1f\n", TARGET, n, rows, (double) range);
    measure("scalar", softplus_scalar, src, dst, ref, n, rows, iters);
#ifdef LAB_PROPOSED
    measure("vector_f32", softplus_vec_f32, src, dst, ref, n, rows, iters);
    measure("vector_i16", softplus_vec_i16, src, dst, ref, n, rows, iters);
#endif

    // The sweep of the element count. The buffers are separate, because the sweep writes guard
    // elements after each row.
    const size_t sweep_n = LAB_SWEEP_MAXN + LAB_SWEEP_GUARD;
    float * ssrc = lab_ddr_alloc(sweep_n * sizeof(float), 128);
    float * sdst = lab_ddr_alloc(sweep_n * sizeof(float), 128);
    lab_fill_f32(ssrc, sweep_n, -range, range);
    const float sweep_edges[8] = { 0.0f, -1e-4f, 1e-4f, -25.0f, 25.0f, -8.5f, 8.5f, -0.5f };
    for (uint32_t i = 0; i < 8; i++) {
        ssrc[i] = sweep_edges[i];
    }
    const uint32_t fail = sweep(ssrc, sdst);

    return fail ? 1 : 0;
}
