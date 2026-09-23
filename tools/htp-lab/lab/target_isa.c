// Target: the HVX instruction census. Refer to tools/htp-lab/isa/gen_census.py.
//
// The program makes the input corpus, writes it to the run directory (corpus_<stream>.bin), runs
// each op of isa_kernels.c on it, writes each output (out_<op>.bin) and prints one line for each
// op: "lab: isa <op> <hash>", or "lab: isa <op> n/a" if the op is not available on this version.
// tools/htp-lab/isa/compare.py compares the outputs of the versions.
//
// The corpus is the same on each version, because only integer code makes it: the float bit
// patterns come from integer fields, and the program never uses a float instruction for them.
// Run it in functional mode:
//   ARCH=v75 MODE=functional LAB_TARGETS=isa LAB_OUT=tools/htp-lab/out-isa TAG=c PROPOSALS=none \
//       tools/htp-lab/run.sh run isa
// Arguments: --op N runs only op N. --write 0 writes no files (the hash lines only).
// --bench N runs the cost bench instead (use MODE=timing): for each op, isa_bench on N trips of
// ISA_BENCH_UNROLL independent iterations, and one line "lab: isa-bench <op> <cycles per iteration>
// <output vectors per iteration>". The inputs are ordinary values: f32, f16 and bf16 in +-[1, 2),
// their canonical qf encodings, random integers.
#include "lab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../isa/isa_kernels.c"

// ---------------------------------------------------------------------------------------------
// Random numbers: splitmix64 with a fixed seed for each stream
// ---------------------------------------------------------------------------------------------

static uint64_t g_rng;

static void rng_seed(uint64_t s) {
    g_rng = s * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
}

static uint64_t rng64(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static uint32_t rng32(void) {
    return (uint32_t) (rng64() >> 32);
}

// A random integer in [0, n)
static uint32_t rng_below(uint32_t n) {
    return (uint32_t) (((uint64_t) rng32() * n) >> 32);
}

// A random integer in [lo, hi]
static int rng_range(int lo, int hi) {
    return lo + (int) rng_below((uint32_t) (hi - lo + 1));
}

// ---------------------------------------------------------------------------------------------
// IEEE binary formats, built from integer fields only
// ---------------------------------------------------------------------------------------------

typedef struct {
    int mb;    // the bits of the stored mantissa
    int eb;    // the bits of the exponent
    int bias;
} fmt_t;

static const fmt_t FSF = { 23, 8, 127 };
static const fmt_t FHF = { 10, 5, 15 };
static const fmt_t FBF = { 7, 8, 127 };

static uint32_t fp_signbit(const fmt_t * f) { return 1u << (f->mb + f->eb); }
static uint32_t fp_emask(const fmt_t * f) { return (1u << f->eb) - 1; }
static uint32_t fp_mmask(const fmt_t * f) { return (1u << f->mb) - 1; }
static int      fp_emin(const fmt_t * f) { return 1 - f->bias; }                          // unbiased
static int      fp_emax(const fmt_t * f) { return (int) fp_emask(f) - 1 - f->bias; }      // unbiased

static int bitlen64(uint64_t m) {
    return m ? 64 - __builtin_clzll(m) : 0;
}

// Rounds (-1)^sign * m * 2^e2 to the format with round to nearest even, with subnormals and
// overflow to infinity. Integer code only. O(1).
static uint32_t fp_make(const fmt_t * f, int sign, uint64_t m, int e2) {
    const uint32_t sbit = sign ? fp_signbit(f) : 0;
    if (m == 0) {
        return sbit;
    }
    const int p = f->mb + 1;
    const int n = bitlen64(m);
    const int e = e2 + n - 1;
    const int q = (e < fp_emin(f) ? fp_emin(f) : e) - (p - 1);  // the exponent of the last kept bit
    const int shift = q - e2;
    uint64_t r;
    int      qq = q;
    if (shift <= 0) {
        r = m << (-shift);  // exact: m has at most p bits here
    } else if (shift > n) {
        r = 0;              // less than half of the smallest step
    } else {
        const uint64_t rem  = shift == 64 ? m : (m & ((1ull << shift) - 1));
        const uint64_t half = 1ull << (shift - 1);
        r = shift == 64 ? 0 : (m >> shift);
        if (rem > half || (rem == half && (r & 1))) {
            r++;
        }
    }
    if (r >> p) {
        r >>= 1;
        qq++;
    }
    if (r == 0) {
        return sbit;
    }
    if (r >> (p - 1)) {
        const int eres = qq + p - 1;
        if (eres > fp_emax(f)) {
            return sbit | (fp_emask(f) << f->mb);
        }
        return sbit | ((uint32_t) (eres + f->bias) << f->mb) | ((uint32_t) r & fp_mmask(f));
    }
    return sbit | (uint32_t) r;  // subnormal
}

// Splits a finite value into sign, integer significand and exponent: x = (-1)^s * m * 2^e2.
// Returns 0 for infinity and NaN.
static int fp_parts(const fmt_t * f, uint32_t x, int * s, uint64_t * m, int * e2) {
    *s = (int) ((x >> (f->mb + f->eb)) & 1);
    const uint32_t e  = (x >> f->mb) & fp_emask(f);
    const uint32_t fr = x & fp_mmask(f);
    if (e == fp_emask(f)) {
        return 0;
    }
    if (e == 0) {
        *m  = fr;
        *e2 = fp_emin(f) - f->mb;
    } else {
        *m  = fr | (1ull << f->mb);
        *e2 = (int) e - f->bias - f->mb;
    }
    return 1;
}

// A random normal value with the unbiased exponent e (clamped to the normal range) and a random sign
static uint32_t fp_rand_normal(const fmt_t * f, int e) {
    if (e < fp_emin(f)) e = fp_emin(f);
    if (e > fp_emax(f)) e = fp_emax(f);
    const uint32_t s = rng32() & 1;
    return (s ? fp_signbit(f) : 0) | ((uint32_t) (e + f->bias) << f->mb) | (rng32() & fp_mmask(f));
}

// The negated product, rounded to the format, or a random value when an input is not finite
static uint32_t fp_neg_product(const fmt_t * f, uint32_t a, uint32_t b) {
    int sa, sb, ea, eb;
    uint64_t ma, mb;
    if (!fp_parts(f, a, &sa, &ma, &ea) || !fp_parts(f, b, &sb, &mb, &eb)) {
        return fp_rand_normal(f, 0);
    }
    return fp_make(f, !(sa ^ sb), ma * mb, ea + eb);
}

// A random odd integer with exactly `bits` bits
static uint64_t rand_odd_bits(int bits) {
    if (bits <= 1) return 1;
    return (1ull << (bits - 1)) | (rng64() & ((1ull << (bits - 1)) - 1)) | 1;
}

// ---------------------------------------------------------------------------------------------
// The structured pairs
// ---------------------------------------------------------------------------------------------

enum {
    K_SPECIAL  = 0,  // sf only: the cross product of the special values
    K_SWEEP    = 1,  // sf only: each exponent and sign with structured and random mantissas
    K_ALL      = 2,  // hf, bf: every encoding in stream A, a partner in B
    K_CANCEL   = 3,  // b is close to -a
    K_EXPDIFF  = 4,  // the exponents differ by 0 to 40
    K_HALFPROD = 5,  // a * b is exactly halfway between two neighbors of the format
    K_HALFSUM  = 6,  // a + b is exactly halfway between two neighbors of the format
    K_EDGE     = 7,  // products and sums at the overflow and the underflow limits
    K_RANDBITS = 8,  // random encodings
    K_RANDSIM  = 9,  // random values with exponents near each other
};

static const uint8_t k_block_kinds[8] = { K_CANCEL, K_EXPDIFF, K_HALFPROD, K_HALFSUM, K_EDGE, K_RANDBITS, K_RANDSIM, K_RANDSIM };

// 64 special sf values: zeros, subnormals, limits, infinities, NaNs with payloads, halfway points
// of the conversions to hf, bf and integers
static const uint32_t k_spec_sf[64] = {
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x00400000, 0x80400000, 0x007fffff, 0x807fffff,
    0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
    0x7fc00000, 0xffc00000, 0x7fc12345, 0x7f800001, 0x7fa00000, 0xff800001, 0x3f000000, 0x40000000,
    0x3fc00000, 0x40400000, 0x3f800001, 0x3f7fffff, 0x4b800000, 0x477fe000, 0x477ff000, 0x477fefff,
    0x33800000, 0x33000000, 0x33000001, 0x38800000, 0x387fe000, 0x7f000000, 0x40490fdb, 0xc0490fdb,
    0x3dcccccd, 0x3eaaaaab, 0x4f000000, 0xcf000000, 0x4effffff, 0x46ffff00, 0x47000000, 0xc7000080,
    0x40200000, 0xc0200000, 0x3effffff, 0x3f808000, 0x3f818000, 0x3f801000, 0x3f803000, 0x4b000001,
    0x4affffff, 0x3f7f8000, 0x00000003, 0x80000002, 0x0f800000, 0x70000000, 0x7f7fff80, 0x7f7f8000,
};

// 32 special hf values
static const uint16_t k_spec_hf[32] = {
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0200, 0x0400, 0x8400, 0x3c00, 0xbc00, 0x7bff,
    0xfbff, 0x7c00, 0xfc00, 0x7e00, 0xfe00, 0x7d00, 0x7c01, 0xfc01, 0x3800, 0x4000, 0x3e00, 0x4100,
    0xc100, 0x3c01, 0x3bff, 0x7800, 0x77ff, 0x3555, 0x4248, 0x0003,
};

// The structured mantissas of the sf exponent sweep: 0, 1, all ones, x.5, and the halfway points
// of the conversion to hf (odd and even) and to bf (odd and even)
static const uint32_t k_sweep_mant[8] = { 0, 1, 0x7fffff, 0x400000, 0x1000, 0x3000, 0x8000, 0x18000 };

static uint32_t fp_special(const fmt_t * f, uint32_t k) {
    if (f == &FSF) return k_spec_sf[k & 63];
    if (f == &FHF) return k_spec_hf[k & 31];
    return k_spec_sf[k & 63] >> 16;  // bf: the upper half of the sf value
}

// The partner b of a value a: equal, opposite, same exponent, random bits, near exponent,
// special, one, or zero
static uint32_t partner(const fmt_t * f, uint32_t a) {
    const uint32_t width_mask = (fp_signbit(f) << 1) - 1;
    const int      ea = (int) ((a >> f->mb) & fp_emask(f)) - f->bias;
    switch (rng_below(8)) {
        case 0: return a;
        case 1: return a ^ fp_signbit(f);
        case 2: return fp_rand_normal(f, ea);
        case 3: return rng32() & width_mask;
        case 4: return fp_rand_normal(f, ea + rng_range(-2, 2));
        case 5: return fp_special(f, rng32());
        case 6: return (uint32_t) f->bias << f->mb;  // 1.0
        default: return (rng32() & 1) ? fp_signbit(f) : 0;
    }
}

// One structured triple (a, b, c) of the kind. c is -(a * b) (rounded) or a random value near a.
static void gen_pair(const fmt_t * f, int kind, uint32_t * a, uint32_t * b, uint32_t * c) {
    const int p = f->mb + 1;
    const int emin = fp_emin(f), emax = fp_emax(f);
    const uint32_t width_mask = (fp_signbit(f) << 1) - 1;
    switch (kind) {
        case K_CANCEL: {
            *a = fp_rand_normal(f, rng_range(emin + p, emax - 2));
            if (rng_below(4) == 0) {
                int s, e2;
                uint64_t m;
                fp_parts(f, *a, &s, &m, &e2);
                const int j = rng_range(1, p);
                *b = fp_make(f, !s, m + (m >> j), e2);
            } else {
                const int k = rng_range(-8, 8);
                *b = (*a ^ fp_signbit(f)) + (uint32_t) k;
            }
            break;
        }
        case K_EXPDIFF: {
            const int d = (int) rng_below(41);
            const int ea = rng_range(emin + 1, emax - 1);
            *a = fp_rand_normal(f, ea);
            const uint64_t mb = (1ull << f->mb) | (rng64() & fp_mmask(f));
            *b = fp_make(f, (int) (rng32() & 1), mb, ea - d - f->mb);
            if (rng32() & 1) {
                const uint32_t t = *a; *a = *b; *b = t;
            }
            break;
        }
        case K_HALFPROD: {
            // odd significands whose product has exactly p + 1 bits: the dropped bit is the
            // last one, thus the exact product is a tie
            uint64_t ma = 1, mbv = 1;
            int la = 2, lb = 2;
            for (int tries = 0; tries < 32; tries++) {
                la  = rng_range(2, p - 1);
                lb  = p + 2 - la;
                ma  = rand_odd_bits(la);
                mbv = rand_odd_bits(lb);
                if (bitlen64(ma * mbv) == p + 1) break;
            }
            int eprod;
            const uint32_t sel = rng_below(8);
            if (sel == 0)      eprod = rng_range(emin - p, emin + 1);   // near the subnormal range
            else if (sel == 1) eprod = rng_range(emax - 1, emax + 1);   // near the overflow
            else               eprod = rng_range(emin + 2, emax - 2);
            int ea = eprod / 2 + rng_range(-3, 3);
            if (ea < emin) ea = emin;
            if (ea > emax) ea = emax;
            int eb = eprod - ea;
            if (eb < emin) eb = emin;
            if (eb > emax) eb = emax;
            *a = fp_make(f, (int) (rng32() & 1), ma, ea - (la - 1));
            *b = fp_make(f, (int) (rng32() & 1), mbv, eb - (lb - 1));
            break;
        }
        case K_HALFSUM: {
            const int ea = rng_range(emin + p + 2, emax - 1);
            *a = fp_rand_normal(f, ea);
            const uint32_t k = rng_below(8);
            *b = fp_make(f, (int) (rng32() & 1), 2 * k + 1, ea - f->mb - 1);
            if (rng32() & 1) {
                const uint32_t t = *a; *a = *b; *b = t;
            }
            break;
        }
        case K_EDGE: {
            switch (rng_below(4)) {
                case 0: {  // the product overflows or nearly overflows
                    const int ea = emax / 2 + rng_range(-2, 2);
                    *a = fp_rand_normal(f, ea);
                    *b = fp_rand_normal(f, emax - ea + rng_range(-1, 1));
                    break;
                }
                case 1: {  // the sum overflows or nearly overflows
                    const uint32_t s = rng32() & 1;
                    *a = fp_rand_normal(f, emax) & ~fp_signbit(f);
                    *b = fp_rand_normal(f, emax - (int) rng_below((uint32_t) p + 2)) & ~fp_signbit(f);
                    if (s) { *a |= fp_signbit(f); *b |= fp_signbit(f); }
                    break;
                }
                case 2: {  // the product is subnormal or less
                    const int ea = emin / 2 + rng_range(-2, 2);
                    *a = fp_rand_normal(f, ea);
                    *b = fp_rand_normal(f, emin - ea - (int) rng_below((uint32_t) p + 3));
                    break;
                }
                default: {  // subnormals and the smallest normals
                    *a = (rng32() & (fp_signbit(f) | fp_mmask(f))) | ((rng32() & 1) ? (1u << f->mb) : 0);
                    *b = (rng32() & (fp_signbit(f) | fp_mmask(f))) | ((rng32() & 1) ? (1u << f->mb) : 0);
                    break;
                }
            }
            break;
        }
        case K_RANDBITS:
            *a = rng32() & width_mask;
            *b = rng32() & width_mask;
            *c = rng32() & width_mask;
            return;
        default: {  // K_RANDSIM
            const int range = f == &FHF ? 10 : 20;
            const int ea = rng_range(-range, range);
            *a = fp_rand_normal(f, ea);
            *b = fp_rand_normal(f, ea + rng_range(-3, 3));
            break;
        }
    }
    *c = (rng32() & 1) ? fp_neg_product(f, *a, *b) : fp_rand_normal(f, rng_range(-8, 8));
}

// ---------------------------------------------------------------------------------------------
// The streams
// ---------------------------------------------------------------------------------------------

#define LANES_32 (ISA_STREAM_BYTES / 4)
#define LANES_16 (ISA_STREAM_BYTES / 2)

static uint8_t * g_stream[ISA_S_COUNT];
static uint8_t   g_kind_sf[LANES_32];
static uint8_t   g_kind_hf[LANES_16];

// sf: lanes [0, 4096) are the special cross product, [4096, 20480) the exponent sweep, the
// remaining lanes are blocks of 8192 lanes with the kinds of k_block_kinds.
static void build_sf(void) {
    uint32_t * A = (uint32_t *) g_stream[ISA_S_SF_A];
    uint32_t * B = (uint32_t *) g_stream[ISA_S_SF_B];
    uint32_t * C = (uint32_t *) g_stream[ISA_S_SF_C];
    rng_seed(1);
    for (uint32_t L = 0; L < LANES_32; L++) {
        if (L < 4096) {
            A[L] = k_spec_sf[L >> 6];
            B[L] = k_spec_sf[L & 63];
            C[L] = k_spec_sf[(L * 37 + 11) & 63];
            g_kind_sf[L] = K_SPECIAL;
        } else if (L < 20480) {
            const uint32_t j = L - 4096;
            const uint32_t e = j >> 6, s = (j >> 5) & 1, k = j & 31;
            const uint32_t m = k < 8 ? k_sweep_mant[k] : (rng32() & 0x7fffff);
            A[L] = (s << 31) | (e << 23) | m;
            B[L] = partner(&FSF, A[L]);
            C[L] = (rng32() & 1) ? fp_neg_product(&FSF, A[L], B[L]) : fp_rand_normal(&FSF, rng_range(-8, 8));
            g_kind_sf[L] = K_SWEEP;
        } else {
            const int kind = k_block_kinds[(L >> 13) & 7];
            gen_pair(&FSF, kind, &A[L], &B[L], &C[L]);
            g_kind_sf[L] = (uint8_t) kind;
        }
    }
}

// hf and bf: lanes [0, 65536) hold every encoding in A with a partner in B, the remaining lanes
// are blocks of 8192 lanes with the kinds of k_block_kinds.
static void build_16(const fmt_t * f, int sa, int sb, int sc, uint64_t seed, uint8_t * kinds) {
    uint16_t * A = (uint16_t *) g_stream[sa];
    uint16_t * B = (uint16_t *) g_stream[sb];
    uint16_t * C = (uint16_t *) g_stream[sc];
    rng_seed(seed);
    for (uint32_t L = 0; L < LANES_16; L++) {
        uint32_t a, b, c;
        int kind;
        if (L < 65536) {
            a = L;
            b = partner(f, a);
            c = (rng32() & 1) ? fp_neg_product(f, a, b) : fp_rand_normal(f, rng_range(-4, 4));
            kind = K_ALL;
        } else {
            kind = k_block_kinds[(L >> 13) & 7];
            gen_pair(f, kind, &a, &b, &c);
        }
        A[L] = (uint16_t) a;
        B[L] = (uint16_t) b;
        C[L] = (uint16_t) c;
        if (kinds) kinds[L] = (uint8_t) kind;
    }
}

// sf_c gets two cross-stream structures after hf_a and hf_b exist:
//   1. For the vector-pair accumulators (Wsf += hf * hf): in the iterations i with i % 4 == 0,
//      lane j of the pair is -(hf_a * hf_b) of hf lane 2j (low vector) or 2(j - 32) + 1 (high
//      vector). This assumes that the low vector holds the products of the even hf lanes.
//   2. For vdmpyacc (sf += hf * hf + hf * hf): in the iterations i with i % 8 == 3, sf lane j is
//      -(the exact sum of the two products of hf lanes 2j and 2j + 1), rounded to sf.
static void build_sf_c_cross(void) {
    uint32_t * C = (uint32_t *) g_stream[ISA_S_SF_C];
    const uint16_t * HA = (const uint16_t *) g_stream[ISA_S_HF_A];
    const uint16_t * HB = (const uint16_t *) g_stream[ISA_S_HF_B];
    for (uint32_t L = 0; L < LANES_32; L++) {
        if (((L >> 6) & 3) == 0) {
            const uint32_t i = L >> 6, j = L & 63;
            const uint32_t h = 64 * i + (j < 32 ? 2 * j : 2 * (j - 32) + 1);
            int sa, sb, ea, eb;
            uint64_t ma, mb;
            if (fp_parts(&FHF, HA[h], &sa, &ma, &ea) && fp_parts(&FHF, HB[h], &sb, &mb, &eb)) {
                C[L] = fp_make(&FSF, !(sa ^ sb), ma * mb, ea + eb);
            }
        } else if (((L >> 5) & 7) == 3) {
            const uint32_t i = L >> 5, j = L & 31;
            const uint32_t h0 = 64 * i + 2 * j, h1 = h0 + 1;
            int s0, s1, t0, t1, e0, e1, f0, f1;
            uint64_t m0, m1, n0, n1;
            if (fp_parts(&FHF, HA[h0], &s0, &m0, &e0) && fp_parts(&FHF, HB[h0], &t0, &n0, &f0) &&
                fp_parts(&FHF, HA[h1], &s1, &m1, &e1) && fp_parts(&FHF, HB[h1], &t1, &n1, &f1)) {
                int64_t  p0 = (int64_t) (m0 * n0), p1 = (int64_t) (m1 * n1);
                int      x0 = e0 + f0, x1 = e1 + f1;
                if (s0 ^ t0) p0 = -p0;
                if (s1 ^ t1) p1 = -p1;
                // align to the smaller exponent, at most 38 bits of shift (the products have 22 bits)
                if (x0 - x1 > 38) { p1 = 0; x1 = x0; }
                if (x1 - x0 > 38) { p0 = 0; x0 = x1; }
                const int xm = x0 < x1 ? x0 : x1;
                const int64_t sum = (p0 << (x0 - xm)) + (p1 << (x1 - xm));
                C[L] = fp_make(&FSF, sum > 0, (uint64_t) (sum < 0 ? -sum : sum), xm);
            }
        }
    }
}

// qf32 and qf16 encoders (integer only). The value of a qf32 word is (2M + 1) * 2^(E - 150) with
// M = bits 31:8 (24-bit two's complement) and E = bits 7:0. The value of a qf16 word is
// (2M + 1) * 2^(E - 25) with M = bits 15:5 (11-bit two's complement) and E = bits 4:0.
//   canonical: M = s >> 1 (s = the IEEE significand with the hidden bit), negated as ~M for a
//     negative value, E = the IEEE biased exponent (1 for a subnormal). The value is s | 1, thus
//     an even s moves up by one IEEE unit. This is the form that the hardware makes from IEEE.
//     Zero becomes the word 0. Infinity and NaN keep their exponent field (E = all ones).
//   exact: s = odd * 2^t gives M = (odd - 1) / 2 and E = e + t, thus the value is exact. When
//     E does not fit, the encoder falls back to the canonical form.
static uint32_t qf_encode(const fmt_t * f, uint32_t x, int exact) {
    const int      mbits = f->mb + 1;               // the bits of M: 24 (qf32) or 11 (qf16)
    const uint32_t mmask = (1u << mbits) - 1;
    const uint32_t emask = fp_emask(f);
    const uint32_t sign  = (x >> (f->mb + f->eb)) & 1;
    uint32_t       e     = (x >> f->mb) & emask;
    const uint32_t fr    = x & fp_mmask(f);
    if (e == 0 && fr == 0) {
        return 0;
    }
    uint32_t s = e ? (fr | (1u << f->mb)) : fr;
    if (e == 0) e = 1;
    uint32_t M, E;
    const int t = __builtin_ctz(s);
    if (exact && e + (uint32_t) t <= emask && e != emask) {
        M = (s >> t) >> 1;
        E = e + (uint32_t) t;
    } else {
        M = s >> 1;
        E = e;
    }
    if (sign) M = ~M;
    return ((M & mmask) << f->eb) | E;
}

static const uint32_t k_qf32_e[11] = { 0, 1, 2, 126, 127, 128, 149, 150, 253, 254, 255 };
static const uint32_t k_qf32_m[9]  = { 0, 1, 0xffffff, 0x400000, 0x3fffff, 0x7fffff, 0x800000, 0xc00000, 0xbfffff };
static const uint32_t k_qf16_e[11] = { 0, 1, 2, 14, 15, 16, 24, 25, 29, 30, 31 };
static const uint32_t k_qf16_m[9]  = { 0, 1, 0x7ff, 0x200, 0x1ff, 0x3ff, 0x400, 0x600, 0x5ff };

static uint32_t qf32_edge(void) {
    const uint32_t mi = rng_below(10);
    const uint32_t m = mi < 9 ? k_qf32_m[mi] : (rng32() & 0xffffff);
    return (m << 8) | k_qf32_e[rng_below(11)];
}

static uint32_t qf16_edge(void) {
    const uint32_t mi = rng_below(10);
    const uint32_t m = mi < 9 ? k_qf16_m[mi] : (rng32() & 0x7ff);
    return (m << 5) | k_qf16_e[rng_below(11)];
}

// A qf32 stream from an sf stream: random words in the K_RANDBITS blocks, edge words in half of
// each K_EDGE block, the exact encoding in each fourth lane, the canonical encoding elsewhere.
static void build_qf32(int dst, int src, uint64_t seed) {
    uint32_t *       Q = (uint32_t *) g_stream[dst];
    const uint32_t * S = (const uint32_t *) g_stream[src];
    rng_seed(seed);
    for (uint32_t L = 0; L < LANES_32; L++) {
        const int kind = g_kind_sf[L];
        if (kind == K_RANDBITS) Q[L] = rng32();
        else if (kind == K_EDGE && ((L >> 12) & 1)) Q[L] = qf32_edge();
        else Q[L] = qf_encode(&FSF, S[L], (L & 3) == 3);
    }
}

// A qf16 stream from an hf stream, with the rules of build_qf32. With `all` set, lanes [0, 65536)
// hold every qf16 word instead.
static void build_qf16(int dst, int src, uint64_t seed, int all) {
    uint16_t *       Q = (uint16_t *) g_stream[dst];
    const uint16_t * S = (const uint16_t *) g_stream[src];
    rng_seed(seed);
    for (uint32_t L = 0; L < LANES_16; L++) {
        const int kind = g_kind_hf[L];
        if (all && L < 65536) Q[L] = (uint16_t) L;
        else if (kind == K_RANDBITS) Q[L] = (uint16_t) rng32();
        else if (kind == K_EDGE && ((L >> 12) & 1)) Q[L] = (uint16_t) qf16_edge();
        else Q[L] = (uint16_t) qf_encode(&FHF, S[L], (L & 3) == 3);
    }
}

// f8: bytes [0, 65536) of a and b are every pair of f8 encodings, the remaining bytes are random
static void build_f8(void) {
    uint8_t * A = g_stream[ISA_S_F8_A];
    uint8_t * B = g_stream[ISA_S_F8_B];
    uint8_t * C = g_stream[ISA_S_F8_C];
    rng_seed(7);
    for (uint32_t L = 0; L < ISA_STREAM_BYTES; L++) {
        const uint32_t r = rng32();
        A[L] = L < 65536 ? (uint8_t) L : (uint8_t) r;
        B[L] = L < 65536 ? (uint8_t) (L >> 8) : (uint8_t) (r >> 8);
        C[L] = (uint8_t) (r >> 16);
    }
}

// Integers: vector v has the kind v % 4: random bytes, bytes from the edge set, halfwords from
// the edge set, or words from the edge set.
static const uint8_t  k_edge_b[8] = { 0x00, 0x01, 0x7f, 0x80, 0x81, 0xfe, 0xff, 0x40 };
static const uint16_t k_edge_h[8] = { 0x0000, 0x0001, 0x7fff, 0x8000, 0x8001, 0xffff, 0x4000, 0xc000 };
static const uint32_t k_edge_w[8] = { 0x00000000, 0x00000001, 0x7fffffff, 0x80000000, 0x80000001, 0xffffffff, 0x40000000, 0x00008000 };

static void build_int(int dst, uint64_t seed) {
    uint8_t * P = g_stream[dst];
    rng_seed(seed);
    for (uint32_t v = 0; v < ISA_STREAM_VECTORS; v++) {
        uint8_t * vec = P + (size_t) v * ISA_VEC_BYTES;
        for (int k = 0; k < ISA_VEC_BYTES; k += 4) {
            const uint32_t r = rng32();
            uint32_t w;
            switch (v & 3) {
                case 0: w = r; break;
                case 1: w = k_edge_b[r & 7] | (k_edge_b[(r >> 3) & 7] << 8) | (k_edge_b[(r >> 6) & 7] << 16) |
                            ((uint32_t) k_edge_b[(r >> 9) & 7] << 24); break;
                case 2: w = k_edge_h[r & 7] | ((uint32_t) k_edge_h[(r >> 3) & 7] << 16); break;
                default: w = k_edge_w[r & 7] ^ ((r >> 29) == 0 ? (r & 0xff) : 0); break;
            }
            memcpy(vec + k, &w, 4);
        }
    }
}

// seq16: halfword L is L for L < 65536 (every value), random after
static void build_seq16(void) {
    uint16_t * P = (uint16_t *) g_stream[ISA_S_SEQ16];
    rng_seed(11);
    for (uint32_t L = 0; L < LANES_16; L++) {
        P[L] = L < 65536 ? (uint16_t) L : (uint16_t) rng32();
    }
}

// wconv: int32 values for the conversion to sf: random, small (exact), ties of the 24-bit rounding
// with odd and even kept parts, ties +-1, powers of two +-1, and the limits
static void build_wconv(void) {
    uint32_t * P = (uint32_t *) g_stream[ISA_S_WCONV];
    rng_seed(13);
    for (uint32_t L = 0; L < LANES_32; L++) {
        const uint32_t r = rng32();
        uint32_t w;
        switch (L & 7) {
            case 0: w = r; break;
            case 1: w = (uint32_t) ((int32_t) (r & 0x01ffffff) - 0x01000000); break;
            case 2:
            case 3: {
                const int k = rng_range(1, 7);
                w = (((r & 0xffffff) | 0x800000) << k) | (1u << (k - 1));
                if (rng32() & 1) w = (uint32_t) -(int32_t) (w & 0x7fffffff);
                break;
            }
            case 4: {
                const int k = rng_range(1, 7);
                w = ((((r & 0xffffff) | 0x800000) << k) | (1u << (k - 1))) + (uint32_t) rng_range(-1, 1);
                break;
            }
            case 5: w = (1u << rng_below(31)) + (uint32_t) rng_range(-1, 1); break;
            case 6: w = k_edge_w[r & 7]; break;
            default: w = (uint32_t) -(int32_t) (1u << rng_below(31)) + (uint32_t) rng_range(-1, 1); break;
        }
        P[L] = w;
    }
}

static void build_corpus(void) {
    for (int s = 1; s < ISA_S_COUNT; s++) {
        g_stream[s] = (uint8_t *) lab_ddr_alloc(ISA_STREAM_BYTES, 128);
    }
    g_stream[ISA_S_NONE] = g_stream[ISA_S_INT_A];  // never read: its bytes per iteration are 0
    build_sf();
    build_16(&FHF, ISA_S_HF_A, ISA_S_HF_B, ISA_S_HF_C, 2, g_kind_hf);
    build_16(&FBF, ISA_S_BF_A, ISA_S_BF_B, ISA_S_BF_C, 3, NULL);
    build_sf_c_cross();
    build_qf32(ISA_S_QF32_A, ISA_S_SF_A, 4);
    build_qf32(ISA_S_QF32_B, ISA_S_SF_B, 5);
    build_qf32(ISA_S_QF32_C, ISA_S_SF_C, 6);
    build_qf16(ISA_S_QF16_A, ISA_S_HF_A, 8, 1);
    build_qf16(ISA_S_QF16_B, ISA_S_HF_B, 9, 0);
    build_qf16(ISA_S_QF16_C, ISA_S_HF_C, 10, 0);
    build_f8();
    build_int(ISA_S_INT_A, 21);
    build_int(ISA_S_INT_B, 22);
    build_int(ISA_S_INT_C, 23);
    build_seq16();
    build_wconv();
}

// ---------------------------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------------------------

static int write_file(const char * path, const char * magic, uint32_t id, const char * name, uint32_t n_vectors,
                      uint32_t bytes_per_vector, const void * data, uint32_t hash) {
    isa_file_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, magic, 4);
    h.version          = ISA_FILE_VERSION;
    h.id               = id;
    h.n_vectors        = n_vectors;
    h.bytes_per_vector = bytes_per_vector;
    h.arch             = __HVX_ARCH__;
    h.hash             = hash;
    h.source           = 0;
    strncpy(h.name, name, sizeof(h.name) - 1);
    FILE * f = fopen(path, "wb");
    if (!f) {
        printf("lab: error: cannot open %s for writing\n", path);
        return -1;
    }
    const size_t bytes = (size_t) n_vectors * bytes_per_vector;
    const int ok = fwrite(&h, sizeof(h), 1, f) == 1 && fwrite(data, 1, bytes, f) == bytes;
    fclose(f);
    if (!ok) {
        printf("lab: error: the write of %s failed\n", path);
        return -1;
    }
    return 0;
}

// The cost bench. One input pool for each element type (ISA_BENCH_UNROLL * n * 256 bytes), thus the
// timing of an op does not include the corpus. The first call warms the caches, the second is timed.
// O(ops * n).
static void bench(int n, long only) {
    const size_t bytes = (size_t) ISA_BENCH_UNROLL * n * 256;
    uint8_t * pool[ISA_T_COUNT];
    rng_seed(99);
    for (int t = 0; t < ISA_T_COUNT; t++) {
        pool[t] = (uint8_t *) lab_ddr_alloc(bytes, 128);
        uint32_t * w = (uint32_t *) pool[t];
        uint16_t * h = (uint16_t *) pool[t];
        for (size_t i = 0; i < bytes / 4; i++) {
            const uint32_t r = rng32(), r2 = rng32();
            switch (t) {
                case ISA_T_SF: w[i] = (r & 0x80000000u) | 0x3f800000u | (r2 & 0x7fffff); break;
                case ISA_T_QF32: w[i] = qf_encode(&FSF, (r & 0x80000000u) | 0x3f800000u | (r2 & 0x7fffff), 0); break;
                case ISA_T_HF:
                    h[2 * i] = (uint16_t) ((r & 0x8000) | 0x3c00 | (r2 & 0x3ff));
                    h[2 * i + 1] = (uint16_t) (((r >> 16) & 0x8000) | 0x3c00 | ((r2 >> 16) & 0x3ff));
                    break;
                case ISA_T_QF16:
                    h[2 * i] = (uint16_t) qf_encode(&FHF, (r & 0x8000) | 0x3c00 | (r2 & 0x3ff), 0);
                    h[2 * i + 1] = (uint16_t) qf_encode(&FHF, ((r >> 16) & 0x8000) | 0x3c00 | ((r2 >> 16) & 0x3ff), 0);
                    break;
                case ISA_T_BF:
                    h[2 * i] = (uint16_t) ((r & 0x8000) | 0x3f80 | (r2 & 0x7f));
                    h[2 * i + 1] = (uint16_t) (((r >> 16) & 0x8000) | 0x3f80 | ((r2 >> 16) & 0x7f));
                    break;
                default: w[i] = r; break;
            }
        }
    }
    uint8_t * out = (uint8_t *) lab_ddr_alloc(bytes, 128);
    for (int k = 0; k < ISA_N_OPS; k++) {
        if (only >= 0 && k != only) continue;
        const isa_op_desc * d = &isa_ops[k];
        const void * in[3];
        for (int j = 0; j < 3; j++) in[j] = pool[d->in_type[j]];
        if (isa_bench(k, in[0], in[1], in[2], out, n) != 0) {
            printf("lab: isa-bench %s n/a\n", d->name);
            continue;
        }
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        LAB_BARRIER();
        isa_bench(k, in[0], in[1], in[2], out, n);
        LAB_BARRIER();
        const uint64_t t1 = lab_cycles();
        const double per_iter = (double) (t1 - t0) / (double) (ISA_BENCH_UNROLL * n);
        printf("lab: isa-bench %s %.2f %d\n", d->name, per_iter, d->out_bytes / 128);
    }
}

int main(int argc, char ** argv) {
    const long only  = lab_arg_long(argc, argv, "--op", -1);
    const long write = lab_arg_long(argc, argv, "--write", 1);
    const long nbench = lab_arg_long(argc, argv, "--bench", 0);
    lab_init();
    if (nbench > 0) {
        printf("lab: isa-bench arch %d trips %ld unroll %d\n", __HVX_ARCH__, nbench, ISA_BENCH_UNROLL);
        bench((int) nbench, only);
        return 0;
    }
    printf("lab: isa arch %d ops %d streams %d vectors %d\n", __HVX_ARCH__, ISA_N_OPS, ISA_S_COUNT, ISA_STREAM_VECTORS);
    if (sizeof(isa_file_header) != 128) {
        printf("lab: error: isa_file_header has %lu bytes, not 128\n", (unsigned long) sizeof(isa_file_header));
        return 1;
    }

    build_corpus();
    char path[160];
    for (int s = 1; s < ISA_S_COUNT; s++) {
        const uint32_t h = isa_hash(g_stream[s], ISA_STREAM_BYTES);
        printf("lab: isa corpus %s %08lx\n", isa_stream_names[s], (unsigned long) h);
        if (write) {
            snprintf(path, sizeof(path), "corpus_%s.bin", isa_stream_names[s]);
            write_file(path, ISA_FILE_MAGIC_CORPUS, (uint32_t) s, isa_stream_names[s], ISA_STREAM_VECTORS, ISA_VEC_BYTES,
                       g_stream[s], h);
        }
    }

    size_t max_out = 0;
    for (int k = 0; k < ISA_N_OPS; k++) {
        const size_t b = (size_t) isa_ops[k].n_vectors * isa_ops[k].out_bytes;
        if (b > max_out) max_out = b;
    }
    uint8_t * out = (uint8_t *) lab_ddr_alloc(max_out, 128);

    int n_run = 0, n_na = 0;
    for (int k = 0; k < ISA_N_OPS; k++) {
        if (only >= 0 && k != only) continue;
        const isa_op_desc * d = &isa_ops[k];
        memset(out, 0, (size_t) d->n_vectors * d->out_bytes);
        const int rc = isa_run(k, g_stream[d->in_stream[0]], g_stream[d->in_stream[1]], g_stream[d->in_stream[2]], out,
                               d->n_vectors);
        if (rc != 0) {
            printf("lab: isa %s n/a\n", d->name);
            n_na++;
            continue;
        }
        const size_t   bytes = (size_t) d->n_vectors * d->out_bytes;
        const uint32_t h     = isa_hash(out, bytes);
        printf("lab: isa %s %08lx\n", d->name, (unsigned long) h);
        if (write) {
            snprintf(path, sizeof(path), "out_%s.bin", d->name);
            write_file(path, ISA_FILE_MAGIC_OUTPUT, (uint32_t) k, d->name, d->n_vectors, d->out_bytes, out, h);
        }
        n_run++;
    }
    printf("lab: isa done: %d ops run, %d not available\n", n_run, n_na);
    return 0;
}
