// Target 4b: the Q8_0 tiled dots with the Q8 activation, for 1 to 4 activation rows.
//
// The program is the Q8_0 twin of target_q4.c. The two product models are Q8_0, thus this target
// gives the numbers of the verify step of speculative decoding. The weight tiles and the tiled
// activation are in VTCM, as after the DMA and the quantization task of the operator. The compact
// activation is in DDR, because the compact dots read it with scalar loads.
//
// Arguments: --k 2048 --rows 4 --ct 4 --iters 5
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include "lab.h"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_compute_res.h>

#include <math.h>
#include <string.h>
#include <stdatomic.h>
#include <stdio.h>

#include "hex-dma.h"
#include "hvx-utils.h"
#include "hvx-dump.h"
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

#define KERNEL_WRAP __attribute__((noinline))

static KERNEL_WRAP void kernel_q8_0_32x1(const uint32_t n, float * s, const void * vx, const void * vy, uint32_t valid_rows) {
    tiled_vec_dot_q8_0_32x1(n, s, vx, vy, valid_rows, NULL);
}

static KERNEL_WRAP void kernel_q8_0_32x2(const uint32_t n, float * s0, float * s1, const void * vx,
                                         const void * vy0, const void * vy1, uint32_t valid_rows) {
    tiled_vec_dot_q8_0_32x2(n, s0, s1, vx, vy0, vy1, valid_rows, NULL, NULL);
}

static KERNEL_WRAP void kernel_quantize_q8_0(float * x, uint8_t * y, uint32_t k) {
    quantize_row_f32_q8_0_tiled(x, y, k);
}

#ifdef HTP_MM_HAVE_MULTIROW
static KERNEL_WRAP void kernel_q8_0_32x1c(const uint32_t n, float * s0, const void * vx, const void * vy0,
                                          const uint32_t * ya0, const int32_t * yb0, uint32_t valid_rows) {
    tiled_vec_dot_q8_0_32x1c(n, s0, vx, vy0, ya0, yb0, valid_rows, NULL);
}

static KERNEL_WRAP void kernel_q8_0_32x2c(const uint32_t n, float * s0, float * s1, const void * vx,
                                          const void * vy0, const void * vy1,
                                          const uint32_t * ya0, const uint32_t * ya1,
                                          const int32_t * yb0, const int32_t * yb1, uint32_t valid_rows) {
    tiled_vec_dot_q8_0_32x2c(n, s0, s1, vx, vy0, vy1, ya0, ya1, yb0, yb1, valid_rows, NULL, NULL);
}

static KERNEL_WRAP void kernel_q8_0_32x3c(const uint32_t n, float * s0, float * s1, float * s2, const void * vx,
                                          const void * vy0, const void * vy1, const void * vy2,
                                          const uint32_t * ya0, const uint32_t * ya1, const uint32_t * ya2,
                                          const int32_t * yb0, const int32_t * yb1, const int32_t * yb2,
                                          uint32_t valid_rows) {
    tiled_vec_dot_q8_0_32x3c(n, s0, s1, s2, vx, vy0, vy1, vy2, ya0, ya1, ya2, yb0, yb1, yb2, valid_rows, NULL, NULL, NULL);
}

static KERNEL_WRAP void kernel_q8_0_32x4c(const uint32_t n, float * s0, float * s1, float * s2, float * s3,
                                          const void * vx, const void * vy0, const void * vy1, const void * vy2,
                                          const void * vy3, const uint32_t * ya0, const uint32_t * ya1,
                                          const uint32_t * ya2, const uint32_t * ya3,
                                          const int32_t * yb0, const int32_t * yb1, const int32_t * yb2,
                                          const int32_t * yb3, uint32_t valid_rows) {
    tiled_vec_dot_q8_0_32x4c(n, s0, s1, s2, s3, vx, vy0, vy1, vy2, vy3, ya0, ya1, ya2, ya3,
                             yb0, yb1, yb2, yb3, valid_rows, NULL, NULL, NULL, NULL);
}

static KERNEL_WRAP void kernel_quantize_q8_0_compact(float * x, uint8_t * y, uint8_t * y_compact, int32_t * y_sum, uint32_t k) {
    quantize_row_f32_q8_0_tiled_compact(x, y, y_compact, y_sum, k);
}
#endif

#define TARGET "q8"
#define Q8_W_TILE     HTP_MM_WEIGHT_TILE_SIZE_Q8_0          // 1088
#define Q8_W_TILE_ALN HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q8_0  // 1152
#define Q8_A_TILE     HTP_MM_ACT_TILE_SIZE_Q8_0             // 1152
#define MAX_ROWS      4

// One plain Q8_0 block: the scale and 32 quants
struct q8_block {
    uint16_t d;
    int8_t   q[32];
};

// Builds one 1088-byte weight tile from 32 rows of plain blocks (the repack of ggml-hexagon.cpp)
static void repack_tile(uint8_t * tile, const struct q8_block * rows, uint32_t n_k_blocks, uint32_t kt) {
    for (uint32_t cp = 0; cp < 16; cp++) {
        for (uint32_t row = 0; row < 32; row++) {
            const struct q8_block * b = &rows[row * n_k_blocks + kt];
            tile[cp * 64 + 2 * row + 0] = (uint8_t) b->q[2 * cp + 0];
            tile[cp * 64 + 2 * row + 1] = (uint8_t) b->q[2 * cp + 1];
        }
    }
    uint16_t * scales = (uint16_t *) (tile + 1024);
    for (uint32_t row = 0; row < 32; row++) {
        scales[row] = rows[row * n_k_blocks + kt].d;
    }
}

// Reads the quants and the scale of k-tile kt back from a tiled Q8 activation row (a DDR copy)
static void read_act_tile(const uint8_t * y_q, uint32_t kt, int8_t * q, float * d) {
    const uint8_t * t = y_q + kt * Q8_A_TILE;
    for (uint32_t j = 0; j < 8; j++) {
        memcpy(q + 4 * j, t + j * 128, 4);
    }
    uint16_t h;
    memcpy(&h, t + 8 * 128, 2);
    *d = lab_hf_to_f32(h);
}

static void decode_act_row(const uint8_t * y_q, uint32_t n_k_tiles, int8_t * q, float * d) {
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        read_act_tile(y_q, kt, q + kt * 32, d + kt);
    }
}

// The scalar reference in the arithmetic order of the 32x1 dot
static float ref_dot(const struct q8_block * wrow, const int8_t * qa, const float * da, uint32_t n_k_tiles) {
    // hexagon-clang makes one fused multiply-add of acc += x * y by default. The HVX dot and the
    // x86 oracle round the product and the sum apart, thus the reference must do the same.
#pragma clang fp contract(off)
    float acc = 0.0f;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const int8_t * q = qa + kt * 32;
        int32_t sum = 0;
        for (uint32_t k = 0; k < 32; k++) {
            sum += (int32_t) wrow[kt].q[k] * (int32_t) q[k];
        }
        acc += (float) sum * (lab_hf_to_f32(wrow[kt].d) * da[kt]);
    }
    return acc;
}

static uint64_t g_best;

static void timing_reset(void) {
    g_best = UINT64_MAX;
}

static void timing_add(uint64_t d) {
    if (d < g_best) {
        g_best = d;
    }
}

static size_t report_rows(const char * variant, uint32_t n, double tiles, float ** out, float ** ref, uint32_t n_w_rows) {
    char key[64];
    snprintf(key, sizeof(key), "rows%u_%s_cycles_per_tile", n, variant);
    lab_report(TARGET, key, (double) g_best / tiles, "cycles");
    snprintf(key, sizeof(key), "rows%u_%s_weight_bytes_per_cycle", n, variant);
    lab_report(TARGET, key, (double) Q8_W_TILE * tiles / (double) g_best, "B/cycle");
    size_t bad = 0, bits = 0;
    for (uint32_t r = 0; r < n; r++) {
        snprintf(key, sizeof(key), "rows%u_%s_out%u", n, variant, r);
        bad += lab_compare_f32(key, out[r], ref[r], n_w_rows, 1e-2f, 1e-4f);
        for (uint32_t i = 0; i < n_w_rows; i++) {
            bits += memcmp(&out[r][i], &ref[r][i], sizeof(float)) != 0;
        }
    }
    snprintf(key, sizeof(key), "rows%u_%s_bits_different", n, variant);
    lab_report(TARGET, key, (double) bits, "of the outputs");
    return bad;
}

int main(int argc, char ** argv) {
    const uint32_t k      = (uint32_t) lab_arg_long(argc, argv, "--k", 2048);
    const uint32_t n_rows = (uint32_t) lab_arg_long(argc, argv, "--rows", 4);
    const uint32_t n_ct   = (uint32_t) lab_arg_long(argc, argv, "--ct", 4);
    const uint32_t iters  = (uint32_t) lab_arg_long(argc, argv, "--iters", 5);

    if (k % 128 != 0 || n_rows < 1 || n_rows > MAX_ROWS || n_ct < 1) {
        printf("lab: error: k must be a multiple of 128, rows 1..4, ct >= 1\n");
        return 2;
    }
    lab_init();

    const uint32_t n_k_tiles = k / 32;
    const uint32_t n_w_rows  = n_ct * 32;
    const double   tiles     = (double) n_ct * n_k_tiles;

    struct q8_block * wq = lab_ddr_alloc((size_t) n_w_rows * n_k_tiles * sizeof(struct q8_block), 128);
    for (uint32_t i = 0; i < n_w_rows * n_k_tiles; i++) {
        wq[i].d = lab_f32_to_hf(lab_rand_f32(0.005f, 0.05f));
        for (uint32_t j = 0; j < 32; j++) {
            wq[i].q[j] = (int8_t) (lab_rand_u32() & 0xFF);
        }
    }

    uint8_t * wt = lab_vtcm_alloc((size_t) n_ct * n_k_tiles * Q8_W_TILE_ALN, 128);
    memset(wt, 0, (size_t) n_ct * n_k_tiles * Q8_W_TILE_ALN);
    for (uint32_t ct = 0; ct < n_ct; ct++) {
        for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
            repack_tile(wt + ((size_t) ct * n_k_tiles + kt) * Q8_W_TILE_ALN, wq + (size_t) ct * 32 * n_k_tiles, n_k_tiles, kt);
        }
    }

    const size_t act_row_size = htp_mm_q8_0_tiled_row_size(k);
    float *   act_f32[MAX_ROWS];
    uint8_t * act_q8[MAX_ROWS];
    uint8_t * act_ddr[MAX_ROWS];
    for (uint32_t r = 0; r < n_rows; r++) {
        act_f32[r] = lab_ddr_alloc(k * sizeof(float), 128);
        act_q8[r]  = lab_vtcm_alloc(act_row_size, 128);
        act_ddr[r] = lab_ddr_alloc(act_row_size, 128);
        lab_fill_f32(act_f32[r], k, -1.0f, 1.0f);
        kernel_quantize_q8_0(act_f32[r], act_q8[r], k);
        memcpy(act_ddr[r], act_q8[r], act_row_size);
    }
    lab_report(TARGET, "k", k, "");

    float * out[MAX_ROWS];
    float * ref[MAX_ROWS];
    int8_t * qa = lab_ddr_alloc((size_t) n_k_tiles * 32, 128);
    float *  da = lab_ddr_alloc((size_t) n_k_tiles * sizeof(float), 128);
    for (uint32_t r = 0; r < n_rows; r++) {
        out[r] = lab_ddr_alloc(n_w_rows * sizeof(float) + 128, 128);
        ref[r] = lab_ddr_alloc(n_w_rows * sizeof(float), 128);
        decode_act_row(act_ddr[r], n_k_tiles, qa, da);
        for (uint32_t wr = 0; wr < n_w_rows; wr++) {
            ref[r][wr] = ref_dot(wq + (size_t) wr * n_k_tiles, qa, da, n_k_tiles);
        }
    }

    size_t bad = 0;

    // each column of the 32x2 dot against the 32x1 dot of the same row, bit for bit
    if (n_rows >= 2) {
        float * o1 = lab_ddr_alloc(n_w_rows * sizeof(float) + 128, 128);
        float * o2[2];
        o2[0] = lab_ddr_alloc(n_w_rows * sizeof(float) + 128, 128);
        o2[1] = lab_ddr_alloc(n_w_rows * sizeof(float) + 128, 128);
        for (uint32_t ct = 0; ct < n_ct; ct++) {
            const uint8_t * w_tile = wt + (size_t) ct * n_k_tiles * Q8_W_TILE_ALN;
            kernel_q8_0_32x2(k, o2[0] + ct * 32, o2[1] + ct * 32, w_tile, act_q8[0], act_q8[1], 32);
        }
        size_t diff = 0;
        for (uint32_t r = 0; r < 2; r++) {
            for (uint32_t ct = 0; ct < n_ct; ct++) {
                const uint8_t * w_tile = wt + (size_t) ct * n_k_tiles * Q8_W_TILE_ALN;
                kernel_q8_0_32x1(k, o1 + ct * 32, w_tile, act_q8[r], 32);
            }
            for (uint32_t i = 0; i < n_w_rows; i++) {
                diff += memcmp(&o1[i], &o2[r][i], sizeof(float)) != 0;
            }
        }
        lab_report(TARGET, "x2_against_x1_bits_different", (double) diff, "of the outputs");
    }

    // the sequence of the operator of the checkout: the 32x2 dot per row pair, the 32x1 dot last
    for (uint32_t n = 1; n <= n_rows; n++) {
        timing_reset();
        for (uint32_t it = 0; it < iters; it++) {
            LAB_BARRIER();
            const uint64_t t0 = lab_cycles();
            for (uint32_t ct = 0; ct < n_ct; ct++) {
                const uint8_t * w_tile = wt + (size_t) ct * n_k_tiles * Q8_W_TILE_ALN;
                uint32_t r = 0;
                for (; r + 1 < n; r += 2) {
                    kernel_q8_0_32x2(k, out[r] + ct * 32, out[r + 1] + ct * 32, w_tile, act_q8[r], act_q8[r + 1], 32);
                }
                for (; r < n; r++) {
                    kernel_q8_0_32x1(k, out[r] + ct * 32, w_tile, act_q8[r], 32);
                }
            }
            const uint64_t t1 = lab_cycles();
            LAB_BARRIER();
            timing_add(t1 - t0);
        }
        bad += report_rows("base", n, tiles, out, ref, n_w_rows);
    }

#ifdef HTP_MM_HAVE_MULTIROW
    uint8_t * act_c[MAX_ROWS];
    int32_t * act_s[MAX_ROWS];
    for (uint32_t r = 0; r < n_rows; r++) {
        act_c[r] = lab_ddr_alloc(k, 128);
        act_s[r] = lab_ddr_alloc((size_t) n_k_tiles * sizeof(int32_t), 128);
    }
    timing_reset();
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        for (uint32_t r = 0; r < n_rows; r++) {
            kernel_quantize_q8_0_compact(act_f32[r], act_q8[r], act_c[r], act_s[r], k);
        }
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        timing_add(t1 - t0);
    }
    lab_report(TARGET, "quant_compact_cycles_per_128", (double) g_best / ((double) n_rows * k / 128), "cycles");

    size_t bad_compact = 0;
    for (uint32_t r = 0; r < n_rows; r++) {
        memcpy(act_ddr[r], act_q8[r], act_row_size);
        for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
            int8_t qt[32];
            float  dt;
            read_act_tile(act_ddr[r], kt, qt, &dt);
            if (memcmp(qt, act_c[r] + kt * 32, 32) != 0) {
                bad_compact++;
            }
            int32_t sum = 0;
            for (uint32_t j = 0; j < 32; j++) {
                sum += (int32_t) qt[j];
            }
            if (act_s[r][kt] != sum) {
                bad_compact++;
            }
        }
    }
    lab_report(TARGET, "compact_tiles_different", (double) bad_compact, "");
    bad += bad_compact;

    for (uint32_t n = 1; n <= n_rows; n++) {
        timing_reset();
        for (uint32_t it = 0; it < iters; it++) {
            LAB_BARRIER();
            const uint64_t t0 = lab_cycles();
            for (uint32_t ct = 0; ct < n_ct; ct++) {
                const uint8_t * w_tile = wt + (size_t) ct * n_k_tiles * Q8_W_TILE_ALN;
                const uint32_t * ya0 = (const uint32_t *) act_c[0];
                const uint32_t * ya1 = (const uint32_t *) act_c[n > 1 ? 1 : 0];
                const uint32_t * ya2 = (const uint32_t *) act_c[n > 2 ? 2 : 0];
                const uint32_t * ya3 = (const uint32_t *) act_c[n > 3 ? 3 : 0];
                switch (n) {
                    case 1:
                        kernel_q8_0_32x1c(k, out[0] + ct * 32, w_tile, act_q8[0], ya0, act_s[0], 32);
                        break;
                    case 2:
                        kernel_q8_0_32x2c(k, out[0] + ct * 32, out[1] + ct * 32, w_tile, act_q8[0], act_q8[1],
                                          ya0, ya1, act_s[0], act_s[1], 32);
                        break;
                    case 3:
                        kernel_q8_0_32x3c(k, out[0] + ct * 32, out[1] + ct * 32, out[2] + ct * 32, w_tile,
                                          act_q8[0], act_q8[1], act_q8[2], ya0, ya1, ya2,
                                          act_s[0], act_s[1], act_s[2], 32);
                        break;
                    default:
                        kernel_q8_0_32x4c(k, out[0] + ct * 32, out[1] + ct * 32, out[2] + ct * 32, out[3] + ct * 32,
                                          w_tile, act_q8[0], act_q8[1], act_q8[2], act_q8[3], ya0, ya1, ya2, ya3,
                                          act_s[0], act_s[1], act_s[2], act_s[3], 32);
                        break;
                }
            }
            const uint64_t t1 = lab_cycles();
            LAB_BARRIER();
            timing_add(t1 - t0);
        }
        bad += report_rows("compact", n, tiles, out, ref, n_w_rows);
    }
#endif

    lab_report(TARGET, "mismatches", (double) bad, "");
    return bad ? 1 : 0;
}
