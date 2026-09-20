// Target 6: the accuracy of the f16 math helpers of the HTP kernels against libm.
//
// An f16 kernel is only as good as hvx_vec_exp2_f16, hvx_vec_inverse_f16 and
// hvx_vec_fast_sigmoid_f16. No model of ours runs them today, because every activation is f32,
// thus nothing has measured them. This target sweeps an input range and reports the worst relative
// error of each helper, and the worst points. Run it with MODE=functional, the timing is not the point.
//
// Arguments: --lo -12 --hi 12
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef LAB_PROPOSED
#include "gdn-conv-ops.c"   // the proposal: gdn_conv_silu_t_f16 and gdn_conv_silu2_f16
#else
#include "hvx-utils.h"
#endif

#define TARGET "f16math"
#define N      (64 * 128)

static uint16_t in_h[N]  __attribute__((aligned(128)));
static uint16_t out_h[N] __attribute__((aligned(128)));
static double   x_d[N];

typedef HVX_Vector (*vec_fn)(HVX_Vector);

static HVX_Vector fn_exp2(HVX_Vector v)    { return hvx_vec_exp2_f16(v); }
static HVX_Vector fn_inverse(HVX_Vector v) { return hvx_vec_inverse_f16(v); }
static HVX_Vector fn_sigmoid(HVX_Vector v) { return hvx_vec_fast_sigmoid_f16(v); }

static double ref_exp2(double x)    { return pow(2.0, x); }
static double ref_inverse(double x) { return 1.0 / x; }
static double ref_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// Sweep [lo, hi], run fn over 64-lane vectors, compare with ref. The reference input is the f16
// value that the kernel really saw, thus the input rounding is not counted as an error.
static void sweep(const char * name, vec_fn fn, double (*ref)(double), double lo, double hi) {
    for (int i = 0; i < N; i++) {
        const float x = (float) (lo + (hi - lo) * i / (N - 1));
        in_h[i] = lab_f32_to_hf(x);
        x_d[i]  = (double) lab_hf_to_f32(in_h[i]);
    }
    for (int i = 0; i < N; i += 64) {
        *(HVX_Vector *) (out_h + i) = fn(*(HVX_Vector *) (in_h + i));
    }
    double worst = 0.0, sum2 = 0.0;
    int    wi = 0, n = 0;
    for (int i = 0; i < N; i++) {
        const double r = ref(x_d[i]);
        if (!(fabs(r) > 1e-4 && fabs(r) < 6e4)) {
            continue;  // outside the range where f16 holds the result with full precision
        }
        const double g = (double) lab_hf_to_f32(out_h[i]);
        const double e = fabs(g - r) / fabs(r);
        sum2 += e * e; n++;
        if (e > worst) { worst = e; wi = i; }
    }
    printf("lab: %s %-8s range [%g, %g]: rms rel err %.3g  worst rel err %.3g at x = %g (ref %g got %g)\n",
           TARGET, name, lo, hi, sqrt(sum2 / (n ? n : 1)), worst, x_d[wi], ref(x_d[wi]),
           (double) lab_hf_to_f32(out_h[wi]));
}

#ifdef LAB_PROPOSED
static HVX_Vector fn_silu_t(HVX_Vector v) { return gdn_conv_silu_t_f16(v); }
static double ref_silu_t(double x) { const double e = exp(-fabs(x)); return e / (1.0 + e); }

// the conversions and the f32 SiLU of the proposal, on f32 inputs
static float in_f[N] __attribute__((aligned(128)));
static float out_f[N] __attribute__((aligned(128)));
static void sweep_f32(double lo, double hi) {
    for (int i = 0; i < N; i++) in_f[i] = (float) (lo + (hi - lo) * i / (N - 1));
    // 1. the round trip f32 -> f16 -> f32
    double w_rt = 0.0;
    for (int i = 0; i < N; i += 64) {
        const HVX_Vector     a = *(HVX_Vector *) (in_f + i), b = *(HVX_Vector *) (in_f + i + 32);
        const HVX_VectorPair r = hvx_vec_f16_to_f32(hvx_vec_f32_to_f16(a, b));
        *(HVX_Vector *) (out_f + i) = Q6_V_lo_W(r); *(HVX_Vector *) (out_f + i + 32) = Q6_V_hi_W(r);
    }
    int wi = 0;
    for (int i = 0; i < N; i++) {
        const double e = fabs((double) out_f[i] - (double) in_f[i]) / (fabs((double) in_f[i]) + 1e-3);
        if (e > w_rt) { w_rt = e; wi = i; }
    }
    printf("lab: %s roundtrip f32->f16->f32 [%g, %g]: worst rel err %.3g at x = %g (got %g)\n", TARGET, lo, hi,
           w_rt, (double) in_f[wi], (double) out_f[wi]);
    // 1a. over the whole sweep: the bits of the hardware conversion against the software conversion,
    //     and t from the hardware bits (through memory), from the software bits, and with no store between.
    {
        static uint16_t hw[64] __attribute__((aligned(128)));
        static uint16_t sw[64] __attribute__((aligned(128)));
        static uint16_t th[64] __attribute__((aligned(128)));
        static uint16_t ts[64] __attribute__((aligned(128)));
        int n_bits = 0, n_t = 0, shown = 0;
        for (int v = 0; v < N; v += 64) {
            *(HVX_Vector *) hw = hvx_vec_f32_to_f16(*(HVX_Vector *) (in_f + v), *(HVX_Vector *) (in_f + v + 32));
            for (int i = 0; i < 64; i++) sw[i] = lab_f32_to_hf(in_f[v + i]);
            *(HVX_Vector *) th = gdn_conv_silu_t_f16(*(HVX_Vector *) hw);
            *(HVX_Vector *) ts = gdn_conv_silu_t_f16(*(HVX_Vector *) sw);
            for (int i = 0; i < 64; i++) {
                n_bits += hw[i] != sw[i];
                const double x = (double) in_f[v + i], e0 = exp(-fabs(x)), r = e0 / (1.0 + e0);
                const double eh = fabs((double) lab_hf_to_f32(th[i]) - r) / r;
                if (r > 1e-3 && eh > 0.02) {
                    n_t++;
                    if (shown++ < 6) {
                        printf("lab: %s BAD x % .5f hw 0x%04x sw 0x%04x  t_hw %.5f t_sw %.5f ref %.5f  (vec %d lane %d)\n", TARGET,
                               x, hw[i], sw[i], (double) lab_hf_to_f32(th[i]), (double) lab_hf_to_f32(ts[i]), r, v / 64, i);
                    }
                }
            }
        }
        printf("lab: %s whole sweep [%g, %g]: %d lanes differ in bits, %d lanes have t off by more than 2 %%\n", TARGET, lo, hi, n_bits, n_t);
    }
    // 1b. t through the f32 path: convert, t in f16, convert back. With the clamp and without it.
    for (int clamp = 0; clamp < 2; clamp++) {
        const HVX_Vector lim_hi = hvx_vec_splat_f32(32.0f), lim_lo = hvx_vec_splat_f32(-32.0f);
        for (int i = 0; i < N; i += 64) {
            HVX_Vector a = *(HVX_Vector *) (in_f + i), b = *(HVX_Vector *) (in_f + i + 32);
            if (clamp) {
                a = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(a, lim_lo), lim_hi);
                b = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(b, lim_lo), lim_hi);
            }
            const HVX_VectorPair r = hvx_vec_f16_to_f32(gdn_conv_silu_t_f16(hvx_vec_f32_to_f16(a, b)));
            *(HVX_Vector *) (out_f + i) = Q6_V_lo_W(r); *(HVX_Vector *) (out_f + i + 32) = Q6_V_hi_W(r);
        }
        double w = 0.0; int wj = 0;
        for (int i = 0; i < N; i++) {
            const double x = (double) in_f[i], e0 = exp(-fabs(x)), r = e0 / (1.0 + e0);
            const double e = fabs((double) out_f[i] - r) / r;
            if (r > 1e-4 && e > w) { w = e; wj = i; }
        }
        const double xw = (double) in_f[wj], ew = exp(-fabs(xw));
        printf("lab: %s t via f32 path, clamp %d [%g, %g]: worst rel err %.3g at x = %g (ref %g got %g)\n", TARGET,
               clamp, lo, hi, w, xw, ew / (1.0 + ew), (double) out_f[wj]);
    }
    // 2. the SiLU of two blocks
    double w_s = 0.0, s2 = 0.0; wi = 0;
    for (int i = 0; i < N; i += 64) {
        HVX_Vector ya, yb;
        gdn_conv_silu2_f16(*(HVX_Vector *) (in_f + i), *(HVX_Vector *) (in_f + i + 32), &ya, &yb);
        *(HVX_Vector *) (out_f + i) = ya; *(HVX_Vector *) (out_f + i + 32) = yb;
    }
    for (int i = 0; i < N; i++) {
        const double x = (double) in_f[i], r = x / (1.0 + exp(-x));
        const double e = fabs((double) out_f[i] - r) / (fabs(r) + 1e-3);
        s2 += e * e;
        if (e > w_s) { w_s = e; wi = i; }
    }
    printf("lab: %s silu2_f16 [%g, %g]: rms rel err %.3g worst %.3g at x = %g (ref %g got %g)\n", TARGET, lo, hi,
           sqrt(s2 / N), w_s, (double) in_f[wi], (double) in_f[wi] / (1.0 + exp(-(double) in_f[wi])), (double) out_f[wi]);
}
#endif

#ifdef LAB_PROPOSED
// The full map of vlut16 in the 128-byte mode, measured and not assumed. The table holds
// 1000 + its halfword index, thus an output names the halfword that it read.
//   1. The position map: one matching byte at the position p, all other bytes do not match.
//      The output halfword q that is not zero gives q = f(p).
//   2. The value map: all 128 bytes have the value v, for each v from 0 to 255 and each Rt from
//      0 to 31. The output gives the halfword for (Rt, v), or no match.
// The program then does a check of two closed forms against all measured points and prints the result.
static void probe_vlut16(void) {
    static int16_t  tbl[64]  __attribute__((aligned(128)));
    static uint8_t  idx[128] __attribute__((aligned(128)));
    static int16_t  out[128] __attribute__((aligned(128)));
    for (int k = 0; k < 64; k++) tbl[k] = (int16_t) (1000 + k);

    int pos_split = 1, pos_ident = 1;
    for (int p = 0; p < 128; p++) {
        for (int k = 0; k < 128; k++) idx[k] = 0xff;
        idx[p] = 5;
        const HVX_VectorPair r = Q6_Wh_vlut16_VbVhR(*(HVX_Vector *) idx, *(HVX_Vector *) tbl, 0);
        *(HVX_Vector *) out = Q6_V_lo_W(r); *(HVX_Vector *) (out + 64) = Q6_V_hi_W(r);
        int q = -1, n = 0;
        for (int k = 0; k < 128; k++) if (out[k] != 0) { q = k; n++; }
        if (n != 1) { printf("lab: %s vlut16 position %d: %d outputs are not zero\n", TARGET, p, n); pos_split = pos_ident = 0; continue; }
        if (q != p) pos_ident = 0;
        if (q != ((p & 1) ? 64 + p / 2 : p / 2)) pos_split = 0;
    }
    printf("lab: %s vlut16 position map: identity %s, lo = even bytes and hi = odd bytes %s\n", TARGET,
           pos_ident ? "HOLDS" : "fails", pos_split ? "HOLDS" : "fails");

    int rule = 1, n_match = 0, ff_miss = 1;
    for (int rt = 0; rt < 32; rt++) {
        for (int v = 0; v < 256; v++) {
            for (int k = 0; k < 128; k++) idx[k] = (uint8_t) v;
            const HVX_VectorPair r = Q6_Wh_vlut16_VbVhR(*(HVX_Vector *) idx, *(HVX_Vector *) tbl, rt);
            *(HVX_Vector *) out = Q6_V_lo_W(r);
            const int hw = out[0] ? out[0] - 1000 : -1;
            // the documented form: the high nibble of the byte is equal to Rt & 15, the entry is the
            // word (byte % 32), and the bit 1 of Rt selects the odd halfword of that word
            const int want = ((v & 0xf0) == ((rt & 0xf) << 4)) ? 2 * (v % 32) + ((rt >> 1) & 1) : -1;
            if (hw != want) {
                if (rule) printf("lab: %s vlut16 value map differs first at Rt %d byte %d: halfword %d, the form gives %d\n", TARGET, rt, v, hw, want);
                rule = 0;
            }
            n_match += hw >= 0;
            if (rt == 0 && v >= 0xf0 && hw >= 0) ff_miss = 0;
        }
    }
    printf("lab: %s vlut16 value map: the documented form %s for all 32 x 256 points (%d matches). With Rt 0 the bytes 0xf0 to 0xff give 0: %s\n",
           TARGET, rule ? "HOLDS" : "fails", n_match, ff_miss ? "yes" : "NO");
    if (!rule) {
        for (int rt = 0; rt < 4; rt++) {
            printf("lab: %s vlut16 Rt %d:", TARGET, rt);
            for (int v = 0; v < 256; v++) {
                for (int k = 0; k < 128; k++) idx[k] = (uint8_t) v;
                const HVX_VectorPair r = Q6_Wh_vlut16_VbVhR(*(HVX_Vector *) idx, *(HVX_Vector *) tbl, rt);
                *(HVX_Vector *) out = Q6_V_lo_W(r);
                if (out[0]) printf(" %d>%d", v, out[0] - 1000);
            }
            printf("\n");
        }
    }
}

// h of the int16 routine against the exact bump, for n vectors in one call
static void sweep_silu_i16(int n, double lo, double hi) {
    static __fp16   in[4][64]  __attribute__((aligned(128)));
    static int16_t  out[4][64] __attribute__((aligned(128)));
    double worst = 0.0, s2 = 0.0; long cnt = 0; double wx = 0.0, wr = 0.0, wg = 0.0;
    for (int step = 0; step < 64; step++) {
        for (int r = 0; r < n; r++) {
            for (int i = 0; i < 64; i++) {
                in[r][i] = (__fp16) (lo + (hi - lo) * (double) ((step * 4 + r) * 64 + i) / (64.0 * 4 * 64 - 1));
            }
        }
        HVX_Vector a[4], h[4];
        for (int r = 0; r < n; r++) a[r] = *(HVX_Vector *) in[r];
        if (n == 1) hvx_silu_h_i16(a, h, 1); else if (n == 2) hvx_silu_h_i16(a, h, 2); else hvx_silu_h_i16(a, h, 4);
        for (int r = 0; r < n; r++) *(HVX_Vector *) out[r] = h[r];
        for (int r = 0; r < n; r++) {
            for (int i = 0; i < 64; i++) {
                const double u   = fabs((double) in[r][i]);
                const double ref = u / (1.0 + exp(u));
                const double got = (double) out[r][i] / 65536.0;
                const double e   = fabs(got - ref);
                s2 += e * e; cnt++;
                if (e > worst) { worst = e; wx = (double) in[r][i]; wr = ref; wg = got; }
            }
        }
    }
    printf("lab: %s silu_h_i16 n=%d [%g, %g]: rms abs err %.3g worst %.3g at x = %g (ref %g got %g)\n", TARGET, n, lo, hi,
           sqrt(s2 / (double) cnt), worst, wx, wr, wg);
}
#endif

int main(int argc, char ** argv) {
    const double lo = (double) lab_arg_long(argc, argv, "--lo", -12);
    const double hi = (double) lab_arg_long(argc, argv, "--hi", 12);
    lab_init();
    sweep("exp2",    fn_exp2,    ref_exp2,    -12.0, 0.0);
    sweep("exp2",    fn_exp2,    ref_exp2,    -1.0,  0.0);
    sweep("inverse", fn_inverse, ref_inverse, 1.0,   2.0);
    sweep("inverse", fn_inverse, ref_inverse, 0.01,  100.0);
    sweep("sigmoid", fn_sigmoid, ref_sigmoid, lo,    hi);
    sweep("sigmoid", fn_sigmoid, ref_sigmoid, -2.0,  2.0);
#ifdef LAB_PROPOSED
    probe_vlut16();
    sweep_silu_i16(1, -18.0, 18.0);
    sweep_silu_i16(2, -18.0, 18.0);
    sweep_silu_i16(4, -18.0, 18.0);
    sweep_silu_i16(4, -0.01, 0.01);
    sweep("silu_t",  fn_silu_t,  ref_silu_t,  -8.0,  8.0);
    sweep("silu_t",  fn_silu_t,  ref_silu_t,  -2.0,  2.0);
    sweep_f32(-8.0, 8.0);
    sweep_f32(-2.0, 2.0);
#endif
    return 0;
}
