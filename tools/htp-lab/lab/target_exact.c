// Target: the integer-only IEEE routines of hvx-exact.h.
//
// The program makes the inputs with integer code only, thus the input bytes are the same on each
// Hexagon version. It runs each routine and writes the inputs and the outputs to files in the run
// directory, and tools/htp-lab/exact/check.py compares them with an exact reference on the host.
// It also prints one FNV-1a checksum per routine, thus the four versions compare from the stdout.
// With --timing 1 it measures the cycles per vector of each routine and of the qf sequence that the
// kernels use today, over data in the L2.
//
// Arguments: --vectors 8192 --timing 0
#include "lab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx-base.h"
#include "hvx-exact.h"

#define TARGET "exact"

enum { CASE_RANDOM, CASE_SHORT, CASE_NEAR, CASE_CANCEL, CASE_SPECIAL, CASE_COUNT };

static const uint32_t k_special[] = {
    0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00800000, 0x80800000, 0x3f800000, 0xbf800000,
    0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xffc00123, 0x33800000,
};

// A random f32 bit pattern with a normal exponent in [e_lo, e_hi].
static uint32_t rand_f32(uint32_t e_lo, uint32_t e_hi) {
    const uint32_t s = lab_rand_u32() & 0x80000000u;
    const uint32_t e = e_lo + lab_rand_u32() % (e_hi - e_lo + 1);
    const uint32_t m = lab_rand_u32() & 0x007fffffu;
    return s | (e << 23) | m;
}

// Keep only the top k bits of the significand: short significands give exact products and ties.
static uint32_t shorten(uint32_t x) {
    const uint32_t k = lab_rand_u32() % 24;
    const uint32_t mask = k == 0 ? 0 : (0x007fffffu >> (23 - k)) << (23 - k);
    return (x & 0xff800000u) | (x & mask);
}

// Fill n pairs for a binary routine. The cases mix random values, short significands, close
// exponents, cancellations and special values.
static void fill_pairs(uint32_t * a, uint32_t * b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t x = rand_f32(1, 254);
        uint32_t y = rand_f32(1, 254);
        switch (lab_rand_u32() % CASE_COUNT) {
            case CASE_RANDOM:
                break;
            case CASE_SHORT:
                x = shorten(rand_f32(64, 190));
                y = shorten(rand_f32(64, 190));
                break;
            case CASE_NEAR: {
                const uint32_t ex = 40 + lab_rand_u32() % 170;
                const uint32_t d  = lab_rand_u32() % 32;
                x = (x & 0x807fffffu) | (ex << 23);
                y = (y & 0x807fffffu) | ((ex - d) << 23);
                if (lab_rand_u32() & 1) {
                    y = shorten(y);
                }
                break;
            }
            case CASE_CANCEL:
                y = (x ^ 0x80000000u) ^ (lab_rand_u32() & ((1u << (lab_rand_u32() % 24)) - 1));
                break;
            case CASE_SPECIAL:
                x = k_special[lab_rand_u32() % (sizeof(k_special) / sizeof(k_special[0]))];
                if (lab_rand_u32() & 1) {
                    y = k_special[lab_rand_u32() % (sizeof(k_special) / sizeof(k_special[0]))];
                }
                break;
        }
        if (lab_rand_u32() & 1) {
            a[i] = x;
            b[i] = y;
        } else {
            a[i] = y;
            b[i] = x;
        }
    }
}

// Fill n f32 values for the conversion to f16: mostly the exponents around the f16 range, with
// ties at bit 12 and the special values.
static void fill_sf_for_hf(uint32_t * a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t x;
        switch (lab_rand_u32() % 4) {
            case 0:
                x = rand_f32(95, 145);
                break;
            case 1:
                x = (rand_f32(95, 145) & 0xffffe000u) | 0x00001000u;
                break;
            case 2:
                x = rand_f32(1, 254);
                break;
            default:
                x = k_special[lab_rand_u32() % (sizeof(k_special) / sizeof(k_special[0]))];
                break;
        }
        a[i] = x;
    }
}

static uint32_t fnv1a(const uint32_t * p, size_t n) {
    uint32_t h = 0x811c9dc5u;
    for (size_t i = 0; i < n; i++) {
        for (int k = 0; k < 4; k++) {
            h = (h ^ ((p[i] >> (8 * k)) & 0xffu)) * 0x01000193u;
        }
    }
    return h;
}

static void write_file(const char * name, const void * p, size_t bytes) {
    FILE * f = fopen(name, "wb");
    if (f == NULL) {
        printf("lab: error: cannot open %s for writing\n", name);
        exit(2);
    }
    if (fwrite(p, 1, bytes, f) != bytes) {
        printf("lab: error: short write to %s\n", name);
        exit(2);
    }
    fclose(f);
}

typedef HVX_Vector (*un_fn)(HVX_Vector);
typedef HVX_Vector (*bin_fn)(HVX_Vector, HVX_Vector);

static void run_un(un_fn fn, const uint32_t * in, uint32_t * out, size_t n_vec) {
    for (size_t v = 0; v < n_vec; v++) {
        hvx_vmem(out + 32 * v) = fn(hvx_vmem(in + 32 * v));
    }
}

static void run_bin(bin_fn fn, const uint32_t * a, const uint32_t * b, uint32_t * out, size_t n_vec) {
    for (size_t v = 0; v < n_vec; v++) {
        hvx_vmem(out + 32 * v) = fn(hvx_vmem(a + 32 * v), hvx_vmem(b + 32 * v));
    }
}

// The qf sequences of the kernels of today, for the timing comparison only.
static HVX_Vector qf_mul(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}

static HVX_Vector qf_add(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

// The timing loops call the routine inline on 4 independent vectors per iteration, thus the
// scheduler can fill the packets as a kernel does. n_vec must be a multiple of 4.
#define DEF_TIME_BIN(tag, fn)                                                                            \
    static void time_bin_##tag(const uint32_t * a, const uint32_t * b, uint32_t * out, size_t n_vec) { \
        for (size_t v = 0; v < n_vec; v += 4) {                                                        \
            const HVX_Vector r0 = fn(hvx_vmem(a + 32 * v), hvx_vmem(b + 32 * v));                      \
            const HVX_Vector r1 = fn(hvx_vmem(a + 32 * v + 32), hvx_vmem(b + 32 * v + 32));            \
            const HVX_Vector r2 = fn(hvx_vmem(a + 32 * v + 64), hvx_vmem(b + 32 * v + 64));            \
            const HVX_Vector r3 = fn(hvx_vmem(a + 32 * v + 96), hvx_vmem(b + 32 * v + 96));            \
            hvx_vmem(out + 32 * v)      = r0;                                                           \
            hvx_vmem(out + 32 * v + 32) = r1;                                                           \
            hvx_vmem(out + 32 * v + 64) = r2;                                                           \
            hvx_vmem(out + 32 * v + 96) = r3;                                                           \
        }                                                                                              \
    }
#define DEF_TIME_UN(tag, fn)                                                            \
    static void time_un_##tag(const uint32_t * a, uint32_t * out, size_t n_vec) {      \
        for (size_t v = 0; v < n_vec; v += 4) {                                       \
            const HVX_Vector r0 = fn(hvx_vmem(a + 32 * v));                           \
            const HVX_Vector r1 = fn(hvx_vmem(a + 32 * v + 32));                      \
            const HVX_Vector r2 = fn(hvx_vmem(a + 32 * v + 64));                      \
            const HVX_Vector r3 = fn(hvx_vmem(a + 32 * v + 96));                      \
            hvx_vmem(out + 32 * v)      = r0;                                          \
            hvx_vmem(out + 32 * v + 32) = r1;                                          \
            hvx_vmem(out + 32 * v + 64) = r2;                                          \
            hvx_vmem(out + 32 * v + 96) = r3;                                          \
        }                                                                             \
    }

DEF_TIME_BIN(mul_exact, hvx_exact_sf_mul)
DEF_TIME_BIN(add_exact, hvx_exact_sf_add)
DEF_TIME_BIN(mul_qf, qf_mul)
DEF_TIME_BIN(add_qf, qf_add)
DEF_TIME_UN(hf2sf, hvx_exact_hf_to_sf)
DEF_TIME_UN(sf2hf, hvx_exact_sf_to_hf)

#define TIME(name, call)                                                           \
    do {                                                                           \
        call; /* warm the L2 */                                                    \
        LAB_BARRIER();                                                             \
        const uint64_t t0 = lab_cycles();                                          \
        call;                                                                      \
        LAB_BARRIER();                                                             \
        const uint64_t t1 = lab_cycles();                                          \
        lab_report(TARGET, name, (double) (t1 - t0) / (double) nt, "cycles/vector"); \
    } while (0)

int main(int argc, char ** argv) {
    lab_init();
    const size_t n_vec  = (size_t) lab_arg_long(argc, argv, "--vectors", 8192);
    const int    timing = (int) lab_arg_long(argc, argv, "--timing", 0);
    const size_t n      = 32 * n_vec;

    uint32_t * a   = lab_ddr_alloc(4 * n, 128);
    uint32_t * b   = lab_ddr_alloc(4 * n, 128);
    uint32_t * out = lab_ddr_alloc(4 * n, 128);

    if (timing) {
        const size_t nt = (n_vec < 256 ? n_vec : 256) & ~(size_t) 3;
        fill_pairs(a, b, 32 * nt);
        TIME("sf_mul exact", time_bin_mul_exact(a, b, out, nt));
        TIME("sf_add exact", time_bin_add_exact(a, b, out, nt));
        TIME("sf_mul qf", time_bin_mul_qf(a, b, out, nt));
        TIME("sf_add qf", time_bin_add_qf(a, b, out, nt));
        for (size_t i = 0; i < 32 * nt; i++) {
            a[i] &= 0xffffu;
        }
        TIME("hf_to_sf exact", time_un_hf2sf(a, out, nt));
        fill_sf_for_hf(a, 32 * nt);
        TIME("sf_to_hf exact", time_un_sf2hf(a, out, nt));
        return 0;
    }

    // The binary routines
    fill_pairs(a, b, n);
    write_file("mul_a.bin", a, 4 * n);
    write_file("mul_b.bin", b, 4 * n);
    run_bin(hvx_exact_sf_mul, a, b, out, n_vec);
    write_file("mul_out.bin", out, 4 * n);
    printf("lab: exact sf_mul fnv %08lx\n", (unsigned long) fnv1a(out, n));

    fill_pairs(a, b, n);
    write_file("add_a.bin", a, 4 * n);
    write_file("add_b.bin", b, 4 * n);
    run_bin(hvx_exact_sf_add, a, b, out, n_vec);
    write_file("add_out.bin", out, 4 * n);
    printf("lab: exact sf_add fnv %08lx\n", (unsigned long) fnv1a(out, n));

    // f16 to f32: every f16 value once (2048 vectors)
    const size_t n_hf = 65536;
    for (size_t i = 0; i < n_hf; i++) {
        a[i] = (uint32_t) i;
    }
    run_un(hvx_exact_hf_to_sf, a, out, n_hf / 32);
    write_file("hf2sf_out.bin", out, 4 * n_hf);
    printf("lab: exact hf_to_sf fnv %08lx\n", (unsigned long) fnv1a(out, n_hf));

    // f32 to f16
    fill_sf_for_hf(a, n);
    write_file("sf2hf_in.bin", a, 4 * n);
    run_un(hvx_exact_sf_to_hf, a, out, n_vec);
    write_file("sf2hf_out.bin", out, 4 * n);
    printf("lab: exact sf_to_hf fnv %08lx\n", (unsigned long) fnv1a(out, n));
    return 0;
}
