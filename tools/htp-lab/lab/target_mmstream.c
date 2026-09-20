// Target: the DDR weight stream of the decode matvec, against the speed of light of this core.
//
// Decode reads the whole weight set once per token and nothing else, thus its rate is the rate at
// which this kernel can pull bytes from DDR. The question this target answers is whether the
// 51 to 55 GB/s that the phone measures is the memory system or the kernel.
//
// Every buffer here is at least 4 MB. Below about 1 MB the whole working set sits in the L2 and a
// DDR number is indistinguishable from a VTCM number, thus a small buffer measures nothing.
//
// The cases:
//   read          sequential vector loads with a cheap integer accumulate: the read ceiling
//   read_l2f<N>   the same with an l2fetch N vectors ahead
//   dot_prod      tiled_vec_dot_q8_0_32x1 of the checkout, weights in DDR, activation in VTCM
//   dot_lane      the same arithmetic with two changes, described below
//
// dot_lane changes the weight tile layout and the activation form. The tile of the checkout holds
// the 4 k-values of a row at the byte offsets 2*row, 2*row+1, 64+2*row, 64+2*row+1, thus they do
// not share a 32-bit lane and the kernel spends a vror and a vshuff on every vector to bring them
// together. Writing the repack so that vector g holds the 4 k-values of row r at the bytes 4*r..4*r+3
// removes both permutes. The activation of a k-tile is 32 bytes, and the tile of the checkout
// replicates it over 1024 bytes so that vrmpy can take it as a vector; Q6_Vw_vrmpyacc_VwVubRb takes
// those 4 bytes from a scalar register instead, which removes 8 of the 16 loads of a k-tile.
// That form reads the vector as unsigned, thus the tile holds w + 128 and the kernel subtracts
// 128 * sum(activation) at the end. The activation sum per k-tile is what the compact quantizer of
// the multirow path already computes.
//
// Arguments: --mb 16 --k 2560 --iters 3
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
#include "hex-utils.h"
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

#define TARGET   "mmstream"
#define W_TILE   1152          // HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q8_0
#define A_TILE   1152          // HTP_MM_ACT_TILE_SIZE_Q8_0
#define MHZ      2112.0

struct q8_block {
    uint16_t d;
    int8_t   q[32];
};

// The weight tile of the checkout: byte cp*64 + 2*row + j holds the k-value 2*cp+j of the row.
static void repack_prod(uint8_t * tile, const struct q8_block * rows, uint32_t n_k_blocks, uint32_t kt) {
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

// The lane-aligned tile: vector g, lane row, bytes 4*row..4*row+3 hold the k-values 4g..4g+3,
// biased by 128 so the unsigned vrmpy form reads them.
static void repack_lane(uint8_t * tile, const struct q8_block * rows, uint32_t n_k_blocks, uint32_t kt) {
    for (uint32_t g = 0; g < 8; g++) {
        for (uint32_t row = 0; row < 32; row++) {
            const struct q8_block * b = &rows[row * n_k_blocks + kt];
            for (uint32_t t = 0; t < 4; t++) {
                tile[g * 128 + 4 * row + t] = (uint8_t) ((int32_t) b->q[4 * g + t] + 128);
            }
        }
    }
    uint16_t * scales = (uint16_t *) (tile + 1024);
    for (uint32_t row = 0; row < 32; row++) {
        scales[row] = rows[row * n_k_blocks + kt].d;
    }
}

// The lane-aligned dot of one column tile. a32[kt*8 + g] holds the 4 activation bytes of k 4g..4g+3,
// a_sum[kt] their signed sum and a_d[kt] the activation scale.
static __attribute__((noinline)) void dot_lane_32x1(uint32_t n_k_tiles, float * restrict s,
                                                    const uint8_t * restrict tile,
                                                    const uint32_t * restrict a32,
                                                    const int32_t * restrict a_sum,
                                                    const __fp16 * restrict a_d) {
    HVX_Vector v_acc_f = Q6_V_vzero();
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict w = (const HVX_Vector *) (tile + (size_t) kt * W_TILE);
        const uint32_t * restrict a = a32 + (size_t) kt * 8;

        HVX_Vector v_s = Q6_V_vzero();
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            v_s = Q6_Vw_vrmpyacc_VwVubRb(v_s, w[g], (int32_t) a[g]);
        }
        // the bias correction: the tile holds w + 128, thus subtract 128 * sum(activation)
        v_s = Q6_Vw_vsub_VwVw(v_s, Q6_V_vsplat_R(128 * a_sum[kt]));

        HVX_Vector v_sf   = Q6_Vsf_equals_Vw(v_s);
        HVX_Vector v_comb = hvx_vec_mul_f16_f16_to_f32_lower32(w[8], Q6_Vh_vsplat_R(*(const uint16_t *) &a_d[kt]));
        v_acc_f = hvx_vec_add_f32_f32(v_acc_f, hvx_vec_mul_f32_f32(v_sf, v_comb));
    }
    hvx_vec_store_u(s, 32 * sizeof(float), v_acc_f);
}

// The read ceiling: sequential vector loads with an integer accumulate, which is 4 per packet and
// thus never the limit. dist is the l2fetch distance in vectors, 0 for none.
static __attribute__((noinline)) HVX_Vector stream_read(const uint8_t * restrict p, size_t n_vec, uint32_t dist) {
    HVX_Vector acc = Q6_V_vzero();
    const HVX_Vector * restrict v = (const HVX_Vector *) p;
    for (size_t i = 0; i < n_vec; i++) {
        if (dist && (i % 8) == 0 && i + dist < n_vec) {
            hex_l2fetch_block(v + i + dist, 8 * 128);
        }
        acc = Q6_V_vor_VV(acc, v[i]);
    }
    return acc;
}

static uint64_t g_best;
static void t_reset(void) { g_best = UINT64_MAX; }
static void t_add(uint64_t d) { if (d < g_best) { g_best = d; } }

static void report_bw(const char * name, double bytes, double mb) {
    const double cyc = (double) g_best;
    lab_report(TARGET, name, bytes / cyc, "B/cycle");
    char key[96];
    snprintf(key, sizeof(key), "%s_GBps", name);
    lab_report(TARGET, key, bytes / cyc * MHZ / 1000.0, "GB/s");
    snprintf(key, sizeof(key), "%s_buffer_MB", name);
    lab_report(TARGET, key, mb, "MB");
}

int main(int argc, char ** argv) {
    const uint32_t mb    = (uint32_t) lab_arg_long(argc, argv, "--mb", 16);
    const uint32_t k     = (uint32_t) lab_arg_long(argc, argv, "--k", 2560);
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 3);

    lab_init();

    const uint32_t n_k_tiles = k / 32;
    const size_t   ct_bytes  = (size_t) n_k_tiles * W_TILE;
    uint32_t       n_ct      = (uint32_t) (((size_t) mb << 20) / ct_bytes);
    if (n_ct < 1) { n_ct = 1; }
    const size_t   w_bytes   = (size_t) n_ct * ct_bytes;
    const double   w_mb      = (double) w_bytes / (1024.0 * 1024.0);

    lab_report(TARGET, "k", k, "");
    lab_report(TARGET, "out_rows", n_ct * 32, "");
    lab_report(TARGET, "weight_MB", w_mb, "MB");

    // the plain blocks, then the two tiled copies, both in DDR
    const uint32_t n_w_rows = n_ct * 32;
    struct q8_block * wq = lab_ddr_alloc((size_t) n_w_rows * n_k_tiles * sizeof(struct q8_block), 128);
    for (size_t i = 0; i < (size_t) n_w_rows * n_k_tiles; i++) {
        wq[i].d = lab_f32_to_hf(lab_rand_f32(0.005f, 0.05f));
        for (uint32_t j = 0; j < 32; j++) { wq[i].q[j] = (int8_t) (lab_rand_u32() & 0xFF); }
    }
    uint8_t * w_prod = lab_ddr_alloc(w_bytes, 128);
    uint8_t * w_lane = lab_ddr_alloc(w_bytes, 128);
    // The exact repack is a scalar byte loop, thus building it over the whole buffer would dominate
    // the simulation. The first n_ref_ct column tiles are exact and carry the correctness check, and
    // the rest are a copy of them: the timing does not depend on the content.
    const uint32_t n_ref_ct = n_ct < 2 ? n_ct : 2;
    for (uint32_t ct = 0; ct < n_ref_ct; ct++) {
        for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
            const size_t off = ((size_t) ct * n_k_tiles + kt) * W_TILE;
            repack_prod(w_prod + off, wq + (size_t) ct * 32 * n_k_tiles, n_k_tiles, kt);
            repack_lane(w_lane + off, wq + (size_t) ct * 32 * n_k_tiles, n_k_tiles, kt);
        }
    }
    for (uint32_t ct = n_ref_ct; ct < n_ct; ct++) {
        const uint32_t src = ct % n_ref_ct;
        memcpy(w_prod + (size_t) ct * ct_bytes, w_prod + (size_t) src * ct_bytes, ct_bytes);
        memcpy(w_lane + (size_t) ct * ct_bytes, w_lane + (size_t) src * ct_bytes, ct_bytes);
    }
    const uint32_t n_ref_rows = n_ref_ct * 32;

    // the activation: the tiled form in VTCM for the kernel of the checkout, and the 32-byte form
    const size_t act_row = htp_mm_q8_0_tiled_row_size(k);
    float *   a_f32 = lab_ddr_alloc(k * sizeof(float), 128);
    uint8_t * a_q   = lab_vtcm_alloc(act_row, 128);
    lab_fill_f32(a_f32, k, -1.0f, 1.0f);
    quantize_row_f32_q8_0_tiled(a_f32, a_q, k);

    uint32_t * a32   = lab_vtcm_alloc((size_t) n_k_tiles * 8 * sizeof(uint32_t), 128);
    int32_t *  a_sum = lab_vtcm_alloc((size_t) n_k_tiles * sizeof(int32_t), 128);
    __fp16 *   a_d   = lab_vtcm_alloc((size_t) n_k_tiles * sizeof(__fp16), 128);
    int8_t     qa[32];
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const uint8_t * t = a_q + (size_t) kt * A_TILE;
        for (uint32_t j = 0; j < 8; j++) { memcpy(qa + 4 * j, t + j * 128, 4); }
        uint16_t h; memcpy(&h, t + 8 * 128, 2);
        memcpy(&a_d[kt], &h, 2);
        int32_t s = 0;
        for (uint32_t j = 0; j < 32; j++) { s += qa[j]; }
        a_sum[kt] = s;
        memcpy(&a32[(size_t) kt * 8], qa, 32);
    }

    // the float64 reference of every output row
    float * ref = lab_ddr_alloc((size_t) n_w_rows * sizeof(float), 128);
    for (uint32_t wr = 0; wr < n_ref_rows; wr++) {
        double acc = 0.0;
        for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
            const struct q8_block * b = &wq[(size_t) wr * n_k_tiles + kt];
            const int8_t * q = (const int8_t *) &a32[(size_t) kt * 8];
            int32_t sum = 0;
            for (uint32_t j = 0; j < 32; j++) { sum += (int32_t) b->q[j] * (int32_t) q[j]; }
            acc += (double) sum * ((double) lab_hf_to_f32(b->d) * (double) lab_hf_to_f32(*(const uint16_t *) &a_d[kt]));
        }
        ref[wr] = (float) acc;
    }

    const size_t n_vec = w_bytes / 128;
    HVX_Vector sink = Q6_V_vzero();

    // 1. the read ceiling, with and without prefetch
    const uint32_t dists[] = { 0, 16, 64, 256 };
    for (size_t di = 0; di < sizeof(dists) / sizeof(dists[0]); di++) {
        t_reset();
        for (uint32_t it = 0; it < iters; it++) {
            LAB_BARRIER();
            const uint64_t t0 = lab_cycles();
            sink = Q6_V_vor_VV(sink, stream_read(w_prod, n_vec, dists[di]));
            const uint64_t t1 = lab_cycles();
            LAB_BARRIER();
            t_add(t1 - t0);
        }
        char key[64];
        snprintf(key, sizeof(key), "read_l2f%u", dists[di]);
        report_bw(key, (double) w_bytes, w_mb);
    }

    // 2. the kernel of the checkout, weights streamed from DDR
    float * out = lab_ddr_alloc((size_t) n_w_rows * sizeof(float) + 128, 128);
    t_reset();
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        for (uint32_t ct = 0; ct < n_ct; ct++) {
            tiled_vec_dot_q8_0_32x1(k, out + ct * 32, w_prod + (size_t) ct * ct_bytes, a_q, 32, NULL);
        }
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        t_add(t1 - t0);
    }
    report_bw("dot_prod", (double) w_bytes, w_mb);
    lab_report(TARGET, "dot_prod_cycles_per_ktile", (double) g_best / ((double) n_ct * n_k_tiles), "cycles");
    size_t bad_prod = lab_compare_f32("dot_prod", out, ref, n_ref_rows, 1e-2f, 2e-3f);

    // 3. the lane-aligned kernel
    float * out2 = lab_ddr_alloc((size_t) n_w_rows * sizeof(float) + 128, 128);
    t_reset();
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        for (uint32_t ct = 0; ct < n_ct; ct++) {
            dot_lane_32x1(n_k_tiles, out2 + ct * 32, w_lane + (size_t) ct * ct_bytes, a32, a_sum, a_d);
        }
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        t_add(t1 - t0);
    }
    report_bw("dot_lane", (double) w_bytes, w_mb);
    lab_report(TARGET, "dot_lane_cycles_per_ktile", (double) g_best / ((double) n_ct * n_k_tiles), "cycles");
    size_t bad_lane = lab_compare_f32("dot_lane", out2, ref, n_ref_rows, 1e-2f, 2e-3f);

    // the normalized mean squared error of each against the float64 reference
    double se_p = 0.0, se_l = 0.0, sr = 0.0;
    for (uint32_t i = 0; i < n_ref_rows; i++) {
        const double r = ref[i];
        se_p += ((double) out[i] - r) * ((double) out[i] - r);
        se_l += ((double) out2[i] - r) * ((double) out2[i] - r);
        sr += r * r;
    }
    lab_report(TARGET, "nmse_dot_prod", sr > 0 ? se_p / sr : 0.0, "");
    lab_report(TARGET, "nmse_dot_lane", sr > 0 ? se_l / sr : 0.0, "");
    lab_report(TARGET, "mismatches", (double) (bad_prod + bad_lane), "");
    lab_report(TARGET, "sink", (double) Q6_R_vextract_VR(sink, 0), "");
    return 0;
}
