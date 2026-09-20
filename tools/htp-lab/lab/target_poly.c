// Target: the piecewise int16 evaluators of hvx-poly-i16.h, which gen/poly_i16.py generates.
//
// Each evaluator gives a bounded smooth function of |x| from the exponent field and the mantissa
// of an f16 input, with the coefficients read by vlut16 and a Horner loop in int16. The reason is
// the packet rules of the v79: a packet holds 1 qfloat multiply and 2 int16 multiplies, and a
// result is ready 2 packets after its producer, thus a qfloat chain costs about 4 cycles for each
// of its steps where 4 independent int16 chains cost about 1.
//
// The program does two things. It compares each evaluator against libm over a sweep, which is the
// same check that gen/poly_i16.py --check does in Python and thus proves that the C path and the
// Python model agree. And it times each evaluator at 1, 2 and 4 vectors per call, which gives the
// cycles per vector that a caller must plan with.
//
// Arguments: --iters 64 --lo -18 --hi 18
#include "lab.h"

#include <stdio.h>
#include <math.h>

#include "hvx-utils.h"
#include "hvx-poly-i16.h"

#define TARGET "poly"
#define N_VEC 64

static float    in_f[4][N_VEC] __attribute__((aligned(128)));
static __fp16   in_h[4][N_VEC] __attribute__((aligned(128)));
static int16_t  out_i[4][N_VEC] __attribute__((aligned(128)));

// One evaluator under test: its routine, its scale, and the reference it must match.
struct poly_case {
    const char * name;
    void      (*fn)(const HVX_Vector * a, HVX_Vector * out, const int n);
    int         scale;
    double    (*ref)(double u);
    const char * note;
};

static double ref_silu_bump(double u)   { return u / (1.0 + exp(u)); }
static double ref_sigmoid_drop(double u) { return 0.5 - 1.0 / (1.0 + exp(u)); }
static double ref_tanh_half(double u)   { return tanh(u); }
static double ref_exp_drop(double u)    { return 1.0 - exp(-u); }

static void call_silu_bump(const HVX_Vector * a, HVX_Vector * o, const int n) {
    if (n == 1) { hvx_poly_silu_bump(a, o, 1); } else if (n == 2) { hvx_poly_silu_bump(a, o, 2); } else { hvx_poly_silu_bump(a, o, 4); }
}
static void call_sigmoid_drop(const HVX_Vector * a, HVX_Vector * o, const int n) {
    if (n == 1) { hvx_poly_sigmoid_drop(a, o, 1); } else if (n == 2) { hvx_poly_sigmoid_drop(a, o, 2); } else { hvx_poly_sigmoid_drop(a, o, 4); }
}
static void call_tanh_half(const HVX_Vector * a, HVX_Vector * o, const int n) {
    if (n == 1) { hvx_poly_tanh_half(a, o, 1); } else if (n == 2) { hvx_poly_tanh_half(a, o, 2); } else { hvx_poly_tanh_half(a, o, 4); }
}
static void call_exp_drop(const HVX_Vector * a, HVX_Vector * o, const int n) {
    if (n == 1) { hvx_poly_exp_drop(a, o, 1); } else if (n == 2) { hvx_poly_exp_drop(a, o, 2); } else { hvx_poly_exp_drop(a, o, 4); }
}

static const struct poly_case cases[] = {
    { "silu_bump",    call_silu_bump,    HVX_POLY_SILU_BUMP_SCALE,    ref_silu_bump,    "u/(1+e^u), the bump of SiLU" },
    { "sigmoid_drop", call_sigmoid_drop, HVX_POLY_SIGMOID_DROP_SCALE, ref_sigmoid_drop, "0.5 - 1/(1+e^u)" },
    { "tanh_half",    call_tanh_half,    HVX_POLY_TANH_HALF_SCALE,    ref_tanh_half,    "tanh(u), u >= 0" },
    { "exp_drop",     call_exp_drop,     HVX_POLY_EXP_DROP_SCALE,     ref_exp_drop,     "1 - e^-u, u >= 0" },
};

// The baseline: the f32 chain of the backend. hvx_vec_fast_sigmoid_f32 is about 25 dependent
// qfloat operations over 32 f32 lanes, thus the comparison must be per lane and not per vector.
static void bench_f32_chain(uint32_t iters) {
    static float in[4][32] __attribute__((aligned(128)));
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 32; i++) {
            in[r][i] = (float) (-4.0 + 0.25 * i + r);
        }
    }
    const HVX_Vector one     = hvx_vec_splat_f32(1.f);
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.f);
    const HVX_Vector min_exp = hvx_vec_splat_f32(-87.f);
    for (int n = 1; n <= 4; n *= 2) {
        HVX_Vector a[4], o[4];
        for (int r = 0; r < 4; r++) {
            a[r] = *(HVX_Vector *) in[r];
        }
        uint64_t best = UINT64_MAX;
        for (int rep = 0; rep < 3; rep++) {
            LAB_BARRIER();
            const uint64_t t0 = lab_cycles();
            for (uint32_t it = 0; it < iters; it++) {
                for (int r = 0; r < n; r++) {
                    o[r] = hvx_vec_fast_sigmoid_f32_guard(a[r], one, max_exp, min_exp);
                }
                a[0] = Q6_V_vor_VV(o[0], a[1]);
            }
            const uint64_t t1 = lab_cycles();
            LAB_BARRIER();
            if (t1 - t0 < best) {
                best = t1 - t0;
            }
        }
        printf("lab: %s f32_sigmoid   n=%d  %6.1f cycles per call, %6.2f cycles per vector,"
               " %6.3f cycles per lane (32 lanes each)\n", TARGET, n,
               (double) best / iters, (double) best / iters / n, (double) best / iters / n / 32.0);
    }
}

int main(int argc, char ** argv) {
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 64);
    const double   lo    = (double) lab_arg_long(argc, argv, "--lo", -18);
    const double   hi    = (double) lab_arg_long(argc, argv, "--hi", 18);

    lab_init();

    bench_f32_chain(iters);

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const struct poly_case * c = &cases[ci];

        // the accuracy sweep, every value of the range through the 4-vector form
        double s2 = 0.0, worst = 0.0, wx = 0.0, wr = 0.0, wg = 0.0;
        long   cnt = 0;
        for (uint32_t step = 0; step < 256; step++) {
            for (int r = 0; r < 4; r++) {
                for (int i = 0; i < N_VEC; i++) {
                    const double t = (double) ((step * 4 + r) * N_VEC + i) / (256.0 * 4 * N_VEC - 1);
                    in_h[r][i] = (__fp16) (lo + (hi - lo) * t);
                }
            }
            HVX_Vector a[4], o[4];
            for (int r = 0; r < 4; r++) {
                a[r] = *(HVX_Vector *) in_h[r];
            }
            c->fn(a, o, 4);
            for (int r = 0; r < 4; r++) {
                *(HVX_Vector *) out_i[r] = o[r];
            }
            for (int r = 0; r < 4; r++) {
                for (int i = 0; i < N_VEC; i++) {
                    const double u   = fabs((double) in_h[r][i]);
                    const double ref = c->ref(u);
                    const double got = (double) out_i[r][i] / (double) (1 << c->scale);
                    const double e   = fabs(got - ref);
                    s2 += e * e;
                    cnt++;
                    if (e > worst) { worst = e; wx = (double) in_h[r][i]; wr = ref; wg = got; }
                }
            }
        }
        printf("lab: %s %-13s rms %.3e worst %.3e at x = %g (ref %g got %g)  %s\n", TARGET, c->name,
               sqrt(s2 / (double) cnt), worst, wx, wr, wg, c->note);

        // the cost at 1, 2 and 4 vectors per call
        for (int n = 1; n <= 4; n *= 2) {
            HVX_Vector a[4], o[4];
            for (int r = 0; r < 4; r++) {
                for (int i = 0; i < N_VEC; i++) {
                    in_f[r][i] = (float) (0.1 + 0.02 * i + r);
                    in_h[r][i] = (__fp16) in_f[r][i];
                }
                a[r] = *(HVX_Vector *) in_h[r];
            }
            c->fn(a, o, n);                              // the warm-up fills the caches
            uint64_t best = UINT64_MAX;
            for (int rep = 0; rep < 3; rep++) {
                LAB_BARRIER();
                const uint64_t t0 = lab_cycles();
                for (uint32_t it = 0; it < iters; it++) {
                    c->fn(a, o, n);
                    a[0] = Q6_V_vor_VV(o[0], a[1]);      // a dependence, thus the loop is not removed
                }
                const uint64_t t1 = lab_cycles();
                LAB_BARRIER();
                if (t1 - t0 < best) {
                    best = t1 - t0;
                }
            }
            printf("lab: %s %-13s n=%d  %6.1f cycles per call, %6.2f cycles per vector,"
                   " %6.3f cycles per lane (64 lanes each)\n", TARGET, c->name, n,
                   (double) best / iters, (double) best / iters / n, (double) best / iters / n / 64.0);
        }
    }
    return 0;
}
