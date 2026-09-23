// Target: the Q8_0 activation quantizer that gives the bits of the CPU reference (htp/hvx-q8-ref.h).
//
// The program makes the rows with integer code only, thus the input bytes are the same on each
// Hexagon version. It quantizes each row and writes the inputs and the outputs to files in the run
// directory, and tools/htp-lab/exact/check_q8.py compares them with quantize_row_q8_0_ref on the
// host. It prints one FNV-1a checksum for each output, thus the versions compare from the stdout.
//
// Modes:
//   default       rows of 4096, 2688 and 128 values: x_<k>.bin, tiles_<k>.bin, compact_<k>.bin,
//                 sums_<k>.bin and scales_<k>.bin (the f16 scales of the flat layout)
//   --scales 1    d and id for amax values: every significand at one exponent (--all 1) or a
//                 sample, plus each exponent: am.bin, d.bin, id.bin
//   --timing 1    cycles for each row of the tiled, compact and flat quantizers, k 2560 and 9728
//
// Arguments: --rows 48 --scales 0 --all 0 --timing 0
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include "lab.h"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_compute_res.h>

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"
#include "hvx-arith.h"
#include "hvx-reduce.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "matmul-ops.h"
#include "htp-vtcm.h"

static const uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = { 0 };
static const uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[128] = { 0 };
static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[128] = { 0 };

#include "hvx-mm-kernels-tiled.h"
#include "hvx-mm-kernels-flat.h"
#include "hvx-q8-ref.h"

#define TARGET "q8oracle"

enum { CASE_SPREAD, CASE_OUTLIER, CASE_TIE, CASE_SHORT, CASE_EQUAL, CASE_ZERO, CASE_TINY, CASE_WIDE, CASE_COUNT };

// A random f32 bit pattern with a biased exponent in [e_lo, e_hi] and a random sign.
static uint32_t rand_f32(uint32_t e_lo, uint32_t e_hi) {
    const uint32_t s = lab_rand_u32() & 0x80000000u;
    const uint32_t e = e_lo + lab_rand_u32() % (e_hi - e_lo + 1);
    return s | (e << 23) | (lab_rand_u32() & 0x007fffffu);
}

// The f32 bits of n * 2^k for |n| < 2^24, with integer code only.
static uint32_t int_times_pow2(int32_t n, int32_t k) {
    if (n == 0) {
        return 0;
    }
    const uint32_t s = n < 0 ? 0x80000000u : 0;
    uint32_t       m = (uint32_t) (n < 0 ? -n : n);
    int32_t        e = 150 + k;
    while (m < 0x00800000u) {
        m <<= 1;
        e -= 1;
    }
    return s | ((uint32_t) e << 23) | (m & 0x007fffffu);
}

// Fill one block of 32 values. The cases cover the ranges of the model activations, the exact ties
// of x * id and their neighbors, short significands, and the edges of the limits.
static void fill_block(uint32_t * x) {
    const uint32_t base = 110 + lab_rand_u32() % 30;
    switch (lab_rand_u32() % CASE_COUNT) {
        case CASE_SPREAD:
            for (int j = 0; j < 32; j++) {
                x[j] = rand_f32(base - 8, base);
            }
            break;
        case CASE_OUTLIER:
            for (int j = 0; j < 32; j++) {
                x[j] = rand_f32(base - 8, base);
            }
            x[lab_rand_u32() % 32] = rand_f32(base + 5, base + 9);
            break;
        case CASE_TIE: {
            // amax = 127 * 2^k gives d = 2^k and id = 2^-k exactly, thus (n + 0.5) * 2^k gives the
            // tie n + 0.5, and a change of one ulp gives each neighbor of the tie.
            const int32_t k = (int32_t) (lab_rand_u32() % 32) - 20;
            x[0] = int_times_pow2((lab_rand_u32() & 1) ? 127 : -127, k);
            for (int j = 1; j < 32; j++) {
                const int32_t n = (int32_t) (lab_rand_u32() % 253) - 126;
                uint32_t      v = int_times_pow2(2 * n + 1, k - 1);
                const uint32_t r = lab_rand_u32() % 4;
                if (r == 1 && (v & 0x7fffffffu) != 0) {
                    v += 1;
                } else if (r == 2 && (v & 0x007fffffu) != 0) {
                    v -= 1;
                }
                x[j] = v;
            }
            break;
        }
        case CASE_SHORT:
            for (int j = 0; j < 32; j++) {
                const uint32_t keep = lab_rand_u32() % 12;
                x[j] = rand_f32(base - 6, base) & ~((0x007fffffu >> keep) & 0x007fffffu);
            }
            break;
        case CASE_EQUAL: {
            const uint32_t v = rand_f32(base - 4, base);
            for (int j = 0; j < 32; j++) {
                x[j] = (lab_rand_u32() & 1) ? v : (v ^ 0x80000000u);
            }
            break;
        }
        case CASE_ZERO:
            for (int j = 0; j < 32; j++) {
                x[j] = lab_rand_u32() & 0x80000000u;
            }
            break;
        case CASE_TINY:
            // amax below 2^-119, where the limits of hvx-q8-ref.h apply.
            for (int j = 0; j < 32; j++) {
                x[j] = (lab_rand_u32() % 3 == 0) ? (lab_rand_u32() & 0x807fffffu) : rand_f32(1, 7);
            }
            break;
        default:
            // The whole exponent range that the limits permit: amax from 2^-119 to 2^20.
            for (int j = 0; j < 32; j++) {
                x[j] = rand_f32(9, 147);
            }
            break;
    }
}

static uint32_t fnv1a(const void * p, size_t bytes) {
    const uint8_t * b = p;
    uint32_t        h = 0x811c9dc5u;
    for (size_t i = 0; i < bytes; i++) {
        h = (h ^ b[i]) * 0x01000193u;
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

// Quantize rows of k values and write the five files for k.
static void run_rows(uint32_t k, uint32_t n_rows) {
    const uint32_t n_blk  = k / 32;
    uint32_t *     x      = lab_ddr_alloc((size_t) n_rows * k * 4, 128);
    uint8_t *      tiles  = lab_ddr_alloc((size_t) n_rows * n_blk * 1152, 128);
    uint8_t *      comp   = lab_ddr_alloc((size_t) n_rows * k, 128);
    int32_t *      sums   = lab_ddr_alloc((size_t) n_rows * n_blk * 4, 128);
    __fp16 *       scales = lab_ddr_alloc((size_t) n_rows * n_blk * 2, 128);

    for (uint32_t b = 0; b < n_rows * n_blk; b++) {
        fill_block(x + 32 * b);
    }
    for (uint32_t r = 0; r < n_rows; r++) {
        quantize_groups_f32_q8_0_ref((const float *) (x + (size_t) r * k), tiles + (size_t) r * n_blk * 1152,
                                     comp + (size_t) r * k, sums + (size_t) r * n_blk, scales + (size_t) r * n_blk,
                                     k / 128);
    }
    char name[64];
    snprintf(name, sizeof(name), "scales_%u.bin", (unsigned) k);
    write_file(name, scales, (size_t) n_rows * n_blk * 2);
    snprintf(name, sizeof(name), "x_%u.bin", (unsigned) k);
    write_file(name, x, (size_t) n_rows * k * 4);
    snprintf(name, sizeof(name), "tiles_%u.bin", (unsigned) k);
    write_file(name, tiles, (size_t) n_rows * n_blk * 1152);
    snprintf(name, sizeof(name), "compact_%u.bin", (unsigned) k);
    write_file(name, comp, (size_t) n_rows * k);
    snprintf(name, sizeof(name), "sums_%u.bin", (unsigned) k);
    write_file(name, sums, (size_t) n_rows * n_blk * 4);
    printf("lab: q8oracle k %u rows %u fnv tiles %08lx compact %08lx sums %08lx\n", (unsigned) k, (unsigned) n_rows,
           (unsigned long) fnv1a(tiles, (size_t) n_rows * n_blk * 1152), (unsigned long) fnv1a(comp, (size_t) n_rows * k),
           (unsigned long) fnv1a(sums, (size_t) n_rows * n_blk * 4));
}

// The amax value of lane i of the scales corpus: with all set, every significand at the exponent
// 127, else 65536 random ones. Then 64 values for each exponent from 1 to 254, and zero.
static uint32_t scales_am(size_t i, size_t n_m, int all) {
    if (i < n_m) {
        return (127u << 23) | (all ? (uint32_t) i : (lab_rand_u32() & 0x007fffffu));
    }
    i -= n_m;
    if (i < 254 * 64) {
        const uint32_t e = 1 + (uint32_t) (i / 64);
        const size_t   j = i % 64;
        return (e << 23) | (j < 2 ? (j ? 0x007fffffu : 0) : (lab_rand_u32() & 0x007fffffu));
    }
    return 0;
}

static FILE * open_out(const char * name) {
    FILE * f = fopen(name, "wb");
    if (f == NULL) {
        printf("lab: error: cannot open %s for writing\n", name);
        exit(2);
    }
    return f;
}

// d and id for the scales corpus, in chunks of 2^20 lanes, because the lab DDR heap is small.
static void run_scales(int all) {
    const size_t n_m   = all ? (1u << 23) : 65536;
    const size_t n     = n_m + 254 * 64 + 32;
    const size_t chunk = 1u << 20;
    uint32_t *   am    = lab_ddr_alloc(chunk * 4, 128);
    uint32_t *   d     = lab_ddr_alloc(chunk * 4, 128);
    uint32_t *   id    = lab_ddr_alloc(chunk * 4, 128);
    FILE *       f_am  = open_out("am.bin");
    FILE *       f_d   = open_out("d.bin");
    FILE *       f_id  = open_out("id.bin");
    uint32_t     h_d = 0, h_id = 0;

    for (size_t c0 = 0; c0 < n; c0 += chunk) {
        const size_t nc = (n - c0 < chunk) ? n - c0 : chunk;
        for (size_t i = 0; i < nc; i++) {
            am[i] = scales_am(c0 + i, n_m, all);
        }
        for (size_t v = 0; v < nc / 32; v++) {
            HVX_Vector vd, vid;
            q8r_scales(hvx_vmem(am + 32 * v), &vd, &vid);
            hvx_vmem(d + 32 * v)  = vd;
            hvx_vmem(id + 32 * v) = vid;
        }
        if (fwrite(am, 4, nc, f_am) != nc || fwrite(d, 4, nc, f_d) != nc || fwrite(id, 4, nc, f_id) != nc) {
            printf("lab: error: short write of the scales files\n");
            exit(2);
        }
        h_d  ^= fnv1a(d, nc * 4);
        h_id ^= fnv1a(id, nc * 4);
    }
    fclose(f_am);
    fclose(f_d);
    fclose(f_id);
    printf("lab: q8oracle scales %zu fnv d %08lx id %08lx\n", n, (unsigned long) h_d, (unsigned long) h_id);
}

#define TIME(name, k, call)                                                              \
    do {                                                                                 \
        call; /* warm the L2 */                                                          \
        LAB_BARRIER();                                                                   \
        const uint64_t t0 = lab_cycles();                                                \
        for (int it = 0; it < 8; it++) {                                                 \
            call;                                                                        \
        }                                                                                \
        LAB_BARRIER();                                                                   \
        const uint64_t t1 = lab_cycles();                                                \
        char key[64];                                                                    \
        snprintf(key, sizeof(key), "%s k=%u", name, (unsigned) (k));                     \
        lab_report(TARGET, key, (double) (t1 - t0) / 8.0, "cycles/row");                \
    } while (0)

static __attribute__((noinline)) void quant_tiled(float * x, uint8_t * y, uint32_t k) {
    quantize_row_f32_q8_0_tiled(x, y, k);
}

static __attribute__((noinline)) void quant_compact(float * x, uint8_t * y, uint8_t * c, int32_t * s, uint32_t k) {
    quantize_row_f32_q8_0_tiled_compact(x, y, c, s, k);
}

static __attribute__((noinline)) void quant_flat(float * x, uint8_t * y, uint32_t k) {
    quantize_row_f32_q8_0_flat(x, y, k);
}

// The cycles for each row of the three quantizers that the matmul kernels call.
static void run_timing(void) {
    static const uint32_t ks[] = { 2560, 9728 };
    for (size_t t = 0; t < sizeof(ks) / sizeof(ks[0]); t++) {
        const uint32_t k = ks[t];
        uint32_t *     x = lab_ddr_alloc((size_t) k * 4, 128);
        uint8_t *      y = lab_ddr_alloc((size_t) k / 32 * 1152, 128);
        uint8_t *      c = lab_ddr_alloc(k, 128);
        int32_t *      s = lab_ddr_alloc((size_t) k / 32 * 4, 128);
        for (uint32_t b = 0; b < k / 32; b++) {
            fill_block(x + 32 * b);
        }
        TIME("tiled", k, quant_tiled((float *) x, y, k));
        TIME("compact", k, quant_compact((float *) x, y, c, s, k));
        TIME("flat", k, quant_flat((float *) x, y, k));
    }
}

int main(int argc, char ** argv) {
    lab_init();
    const uint32_t n_rows = (uint32_t) lab_arg_long(argc, argv, "--rows", 48);
    const int      scales = (int) lab_arg_long(argc, argv, "--scales", 0);
    const int      all    = (int) lab_arg_long(argc, argv, "--all", 0);
    const int      timing = (int) lab_arg_long(argc, argv, "--timing", 0);

    if (timing) {
        run_timing();
        return 0;
    }
    if (scales) {
        run_scales(all);
        return 0;
    }
    run_rows(4096, n_rows);
    run_rows(2688, n_rows);
    run_rows(128, 4 * n_rows);
    return 0;
}
