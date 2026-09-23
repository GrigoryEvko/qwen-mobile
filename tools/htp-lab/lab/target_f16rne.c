// Target: the f32 to f16 conversions of the HVX against the conversion of the CPU backend.
//
// The reference is the conversion of the CPU backend: IEEE round to nearest even, the sign of a
// zero kept, +-Inf kept, and a NaN gives a NaN. The forms:
//   (a) hvx_vec_f32_to_f16 of hvx-base.h (the tree under test)
//   (b) the multiply form: Q6_Vhf_equals_Wqf32 of vmpy(x, 1.0) (v79 and later, else (a))
//   (c) the integer form hvx_exact_sf_to_hf of hvx-exact.h, packed with vpacke
// The main sweep covers each exponent from the f32 subnormals to past the f16 range, each 10-bit
// mantissa head, and the tail bits of a tie and its neighbors, with the two signs (862208
// values). A list of 25 special values follows, with the output of each form. Then the cycles of
// one unaligned copy of a row: the copy of hvx-copy.h with (a), a plain loop with (b), (c) and (a).
//
// The known limits of the qf path before v79, measured with this target on the v75 simulator: the
// qf32 to hf conversion of (a) differs from the CPU conversion in 80888 of the 862208 sweep values
// (the sweep holds many ties, thus the rate is not that of random data) and in 7 of the 25 special
// values: -0 gives +0, +-Inf gives a NaN (0x7fff), -65520 gives -65504 and not -Inf, +-FLT_MAX
// gives a NaN, and 2^-25 gives 0x0001 and not 0. The form (c) gives the CPU bits for each value on
// v75, v79 and v81, but it costs 3 times the plain copy (288 against 96 cycles for 512 values on
// v75), thus the series does not use it. On v79 the add of 0 in (a) gives +0 for -0, and (b) keeps
// -0. On v81, (a) gives the CPU bits.
//
// Arguments: --iters 50 --sweep 1 (0 skips the sweep)
#include "lab.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "hvx-utils.h"
#include "hvx-copy.h"
#include "hvx-exact.h"

#define TARGET "f16rne"

static float    src[64] __attribute__((aligned(128)));
static uint16_t out[3][64] __attribute__((aligned(128)));

// The f16 bits of an f32 value as the CPU backend converts it, in integer arithmetic.
static __attribute__((noinline)) uint16_t ref_f16(uint32_t w) {
    const uint32_t sign = (w >> 16) & 0x8000u;
    const uint32_t a    = w & 0x7fffffffu;
    if (a > 0x7f800000u) {
        return (uint16_t) (sign | 0x7e00u);
    }
    if (a >= 0x477ff000u) {
        return (uint16_t) (sign | 0x7c00u);
    }
    if (a >= 0x38800000u) {
        const uint32_t v   = a - 0x38000000u;
        const uint32_t odd = (v >> 13) & 1u;
        return (uint16_t) (sign | ((v + 0x0fffu + odd) >> 13));
    }
    const uint32_t e  = a >> 23;
    const uint32_t m  = (a & 0x007fffffu) | (e ? 0x00800000u : 0u);
    const uint32_t sh = 126u - (e ? e : 1u);
    if (sh > 24u) {
        return (uint16_t) sign;
    }
    const uint32_t q    = m >> sh;
    const uint32_t rest = m & ((1u << sh) - 1u);
    const uint32_t half = 1u << (sh - 1u);
    const uint32_t up   = rest > half || (rest == half && (q & 1u));
    return (uint16_t) (sign | (q + up));
}

static int same(uint16_t got, uint16_t want) {
    const int nan_w = (want & 0x7c00u) == 0x7c00u && (want & 0x3ffu);
    const int nan_g = (got & 0x7c00u) == 0x7c00u && (got & 0x3ffu);
    return nan_w ? nan_g : got == want;
}

static inline HVX_Vector form_mpy(HVX_Vector v0, HVX_Vector v1) {
#if __HVX_ARCH__ >= 79
    const HVX_Vector one = hvx_vec_splat_f32(1.0f);
    return Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vmpy_VsfVsf(v1, one),
                                                               Q6_Vqf32_vmpy_VsfVsf(v0, one))));
#else
    return hvx_vec_f32_to_f16(v0, v1);
#endif
}

static inline HVX_Vector form_exact(HVX_Vector v0, HVX_Vector v1) {
    return Q6_Vh_vpacke_VwVw(hvx_exact_sf_to_hf(v1), hvx_exact_sf_to_hf(v0));
}

static void convert(const uint32_t * words) {
    memcpy(src, words, sizeof(src));
    const HVX_Vector v0 = *(HVX_Vector *) src, v1 = *(HVX_Vector *) (src + 32);
    *(HVX_Vector *) out[0] = hvx_vec_f32_to_f16(v0, v1);
    *(HVX_Vector *) out[1] = form_mpy(v0, v1);
    *(HVX_Vector *) out[2] = form_exact(v0, v1);
}

static void __attribute__((noinline)) copy_a(uint8_t * dst, const uint8_t * s, uint32_t n) {
    hvx_copy_f16_f32_uu(dst, s, n);
}

static void __attribute__((noinline)) copy_c(uint8_t * dst, const uint8_t * s, uint32_t n) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst;
    const HVX_UVector * restrict vsrc = (const HVX_UVector *) s;
    const uint32_t nvec = n / 64, nloe = n % 64;
    uint32_t i = 0;
    for (; i < nvec; i++) {
        vdst[i] = form_exact(vsrc[i * 2 + 0], vsrc[i * 2 + 1]);
    }
    if (nloe) {
        hvx_vec_store_u((void *) &vdst[i], nloe * 2, form_exact(vsrc[i * 2 + 0], vsrc[i * 2 + 1]));
    }
}

static void __attribute__((noinline)) copy_b(uint8_t * dst, const uint8_t * s, uint32_t n) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst;
    const HVX_UVector * restrict vsrc = (const HVX_UVector *) s;
    const uint32_t nvec = n / 64, nloe = n % 64;
    uint32_t i = 0;
    for (; i < nvec; i++) {
        vdst[i] = form_mpy(vsrc[i * 2 + 0], vsrc[i * 2 + 1]);
    }
    if (nloe) {
        hvx_vec_store_u((void *) &vdst[i], nloe * 2, form_mpy(vsrc[i * 2 + 0], vsrc[i * 2 + 1]));
    }
}

static void __attribute__((noinline)) copy_d(uint8_t * dst, const uint8_t * s, uint32_t n) {
    HVX_UVector * restrict vdst = (HVX_UVector *) dst;
    const HVX_UVector * restrict vsrc = (const HVX_UVector *) s;
    const uint32_t nvec = n / 64, nloe = n % 64;
    uint32_t i = 0;
    for (; i < nvec; i++) {
        vdst[i] = hvx_vec_f32_to_f16(vsrc[i * 2 + 0], vsrc[i * 2 + 1]);
    }
    if (nloe) {
        hvx_vec_store_u((void *) &vdst[i], nloe * 2, hvx_vec_f32_to_f16(vsrc[i * 2 + 0], vsrc[i * 2 + 1]));
    }
}

int main(int argc, char ** argv) {
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 50);
    const uint32_t emax  = lab_arg_long(argc, argv, "--sweep", 1) ? 145 : 0;
    lab_init();
    static const char * names[3] = { "hvx_vec_f32_to_f16", "vmpy form", "exact form" };

    static const uint32_t tails[] = { 0x0000, 0x0001, 0x0fff, 0x1000, 0x1001, 0x1fff, 0x0800, 0x17ff };
    long n = 0, bad[3] = { 0, 0, 0 };
    int  fill = 0;
    uint32_t words[64];
    for (uint32_t e = 0; e <= emax && emax; e++) {
        for (uint32_t head = 0; head < 1024; head += (e < 96 ? 37 : 1)) {
            for (uint32_t t = 0; t < sizeof(tails) / sizeof(tails[0]); t++) {
                for (uint32_t s = 0; s < 2; s++) {
                    words[fill++] = (s << 31) | (e << 23) | (head << 13) | tails[t];
                    if (fill < 64) {
                        continue;
                    }
                    convert(words);
                    for (int i = 0; i < 64; i++) {
                        const uint16_t r = ref_f16(words[i]);
                        for (int f = 0; f < 3; f++) {
                            bad[f] += !same(out[f][i], r);
                        }
                    }
                    n += 64;
                    fill = 0;
                }
            }
        }
    }
    printf("lab: %s sweep of %ld values, different from the CPU conversion: %s %ld, %s %ld, %s %ld\n", TARGET, n,
           names[0], bad[0], names[1], bad[1], names[2], bad[2]);

    // the special values
    static const uint32_t specials[] = {
        0x00000000, 0x80000000,             // +0, -0
        0x7f800000, 0xff800000,             // +Inf, -Inf
        0x7fc00000, 0xffc00000, 0x7f800001, // quiet NaN, negative quiet NaN, signaling NaN
        0x33000000, 0xb3000000,             // +-2^-25, half the smallest f16 subnormal (a tie, gives 0)
        0x33000001, 0x32ffffff,             // just above and below 2^-25
        0x33800000, 0x33c00000,             // 2^-24 (the smallest subnormal), 1.5 * 2^-24 (a tie)
        0x00000001, 0x80000001,             // the smallest f32 subnormal
        0x477fe000, 0x477fefff,             // 65504, the largest value that stays 65504
        0x477ff000, 0xc77ff000,             // +-65520, a tie that goes to Inf
        0x477ff001, 0x7f7fffff, 0xff7fffff, // above 65520, +-FLT_MAX
        0x38800000, 0x387fe000, 0x387ff000, // 2^-14, below it, the tie between the subnormals
    };
    const size_t ns = sizeof(specials) / sizeof(specials[0]);
    for (size_t i = 0; i < 64; i++) {
        words[i] = specials[i % ns];
    }
    convert(words);
    long sbad[3] = { 0, 0, 0 };
    for (size_t i = 0; i < ns; i++) {
        const uint16_t r = ref_f16(specials[i]);
        printf("lab: %s special 0x%08x: cpu 0x%04x  a 0x%04x  b 0x%04x  c 0x%04x%s\n", TARGET, specials[i], r, out[0][i],
               out[1][i], out[2][i],
               (!same(out[0][i], r) || !same(out[1][i], r) || !same(out[2][i], r)) ? "  <- differs" : "");
        for (int f = 0; f < 3; f++) {
            sbad[f] += !same(out[f][i], r);
        }
    }
    printf("lab: %s special values different from the CPU conversion: %s %ld, %s %ld, %s %ld\n", TARGET, names[0],
           sbad[0], names[1], sbad[1], names[2], sbad[2]);

    static const uint32_t lens[] = { 256, 512, 2560 };
    static float    row[2560] __attribute__((aligned(128)));
    static uint16_t dst[2560 + 64] __attribute__((aligned(128)));
    lab_fill_f32(row, 2560, -4.0f, 4.0f);
    for (size_t k = 0; k < sizeof(lens) / sizeof(lens[0]); k++) {
        uint64_t best[4] = { UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX };
        for (int form = 0; form < 4; form++) {
            for (int rep = 0; rep < 3; rep++) {
                LAB_BARRIER();
                const uint64_t t0 = lab_cycles();
                for (uint32_t it = 0; it < iters; it++) {
                    if (form == 0) {
                        copy_a((uint8_t *) dst + 2, (const uint8_t *) row + 4, lens[k]);
                    } else if (form == 1) {
                        copy_b((uint8_t *) dst + 2, (const uint8_t *) row + 4, lens[k]);
                    } else if (form == 2) {
                        copy_c((uint8_t *) dst + 2, (const uint8_t *) row + 4, lens[k]);
                    } else {
                        copy_d((uint8_t *) dst + 2, (const uint8_t *) row + 4, lens[k]);
                    }
                    LAB_BARRIER();
                }
                const uint64_t t1 = lab_cycles();
                best[form] = t1 - t0 < best[form] ? t1 - t0 : best[form];
            }
        }
        printf("lab: %s cycles of one unaligned copy of %4u values: %s %.1f, %s %.1f, %s %.1f, plain loop with (a) %.1f\n",
               TARGET, (unsigned) lens[k], names[0], (double) best[0] / iters, names[1], (double) best[1] / iters,
               names[2], (double) best[2] / iters, (double) best[3] / iters);
    }
    return 0;
}
