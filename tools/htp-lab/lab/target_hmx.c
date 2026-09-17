// Target 3: the HMX tile MAC rate, F16 against int8, for one 32x32 tile stream.
//
// The F16 path is core_dot_chunk_fp16 of hmx-mm-kernels-tiled.h (the prefill GEMM of the backend),
// with the tile layouts of that file: element (i, j) of a 32x32 tile at byte (i/2)*128 + j*4 +
// (i%2)*2. The activation tile holds rows x k, the weight tile k x columns, the output rows x
// columns. The program compares the F16 result against a scalar reference.
//
// The int8 path is the same instruction stream with unsigned 8-bit activations and signed 8-bit
// weights (activation.ub, weight.b, mxclracc, and the .ub output conversion). The program measures
// its cycles only: the tile layout and the output conversion of the int8 mode are not documented
// in the SDK, thus the int8 result is not compared against a reference.
//
// In this SDK (Hexagon Tools 19.0.07) the simulator runs the HMX instructions in functional mode
// (SSR bit 26 enables them) but the timing model does not retire them: a read of the HMX result
// waits forever. The run script gives a cycle limit to the timing run, and the functional run
// gives the correctness result.
//
// Arguments: --dot_tiles 64 --col_tiles 4 --iters 8
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
#include "hmx-mm-kernels-tiled.h"

#define TARGET "hmx"
#define TILE_BYTES 2048
#define TILE_ELMS  1024

typedef __fp16 hf;

// The byte offset of element (i, j) in a 32x32 half tile
static inline size_t tile_off(uint32_t i, uint32_t j) {
    return (size_t) (i / 2) * 128 + (size_t) j * 4 + (i % 2) * 2;
}

static uint64_t time_f16(hf * out, const hf * act, const hf * wgt, const hf * scales,
                         uint32_t n_col_tiles, uint32_t n_dot_tiles, uint32_t iters) {
    uint64_t best = UINT64_MAX;
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        core_dot_chunk_fp16(out, act, wgt, scales, 1, n_col_tiles, n_dot_tiles);
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best) {
            best = t1 - t0;
        }
    }
    return best;
}

// The int8 stream: the same shape as core_dot_chunk_fp16 with 8-bit operands. An int8 tile of
// 32 rows by 32 columns is 1024 bytes. The range field of mxmem holds 16 bits, thus the stream
// covers at most 64 tiles per instruction, and a longer stream needs more instructions.
#define INT8_TILE_BYTES 1024
#define INT8_MAX_TILES  64

static uint64_t time_int8(uint8_t * out, const uint8_t * act, const int8_t * wgt, const uint8_t * bias,
                          uint32_t n_col_tiles, uint32_t n_dot_tiles, uint32_t iters) {
    uint64_t best = UINT64_MAX;
    const size_t dot_stride = (size_t) n_dot_tiles * INT8_TILE_BYTES;
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        __asm__ volatile("bias = mxmem2(%0)\n" : : "r"(bias));
        const int8_t * col_base = wgt;
        uint8_t * out_tile = out;
        for (uint32_t c = 0; c < n_col_tiles; c++) {
            __asm__ volatile("mxclracc\n");
            const uint8_t * a = act;
            const int8_t *  w = col_base;
            uint32_t left = n_dot_tiles;
            while (left > 0) {
                const uint32_t chunk = left > INT8_MAX_TILES ? INT8_MAX_TILES : left;
                const uint32_t range = INT8_TILE_BYTES * chunk - 1;
                __asm__ volatile("{\n activation.ub = mxmem(%1, %0):deep\n weight.b = mxmem(%2, %0)\n}\n"
                                 : : "r"(range), "r"(a), "r"(w));
                a += (size_t) chunk * INT8_TILE_BYTES;
                w += (size_t) chunk * INT8_TILE_BYTES;
                left -= chunk;
            }
            __asm__ volatile("mxmem(%0, %1):after.ub = acc\n" : : "r"(out_tile), "r"(0) : "memory");
            col_base += dot_stride;
            out_tile += INT8_TILE_BYTES;
        }
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best) {
            best = t1 - t0;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    const uint32_t n_dot_tiles = (uint32_t) lab_arg_long(argc, argv, "--dot_tiles", 64);
    const uint32_t n_col_tiles = (uint32_t) lab_arg_long(argc, argv, "--col_tiles", 4);
    const uint32_t iters       = (uint32_t) lab_arg_long(argc, argv, "--iters", 8);
    const uint32_t k           = n_dot_tiles * 32;

    lab_init();

    hf * act    = lab_vtcm_alloc((size_t) n_dot_tiles * TILE_BYTES, TILE_BYTES);
    hf * wgt    = lab_vtcm_alloc((size_t) n_col_tiles * n_dot_tiles * TILE_BYTES, TILE_BYTES);
    hf * out    = lab_vtcm_alloc((size_t) n_col_tiles * TILE_BYTES, TILE_BYTES);
    hf * scales = lab_vtcm_alloc(256, 256);

    float * act_f = lab_ddr_alloc((size_t) 32 * k * sizeof(float), 128);
    float * wgt_f = lab_ddr_alloc((size_t) k * n_col_tiles * 32 * sizeof(float), 128);
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t kk = 0; kk < k; kk++) {
            const float v = lab_rand_f32(-1.0f, 1.0f);
            act_f[r * k + kk] = lab_hf_to_f32(lab_f32_to_hf(v));
            *(hf *) ((uint8_t *) act + (size_t) (kk / 32) * TILE_BYTES + tile_off(r, kk % 32)) = (hf) act_f[r * k + kk];
        }
    }
    for (uint32_t c = 0; c < n_col_tiles * 32; c++) {
        for (uint32_t kk = 0; kk < k; kk++) {
            const float v = lab_rand_f32(-1.0f, 1.0f);
            wgt_f[kk * n_col_tiles * 32 + c] = lab_hf_to_f32(lab_f32_to_hf(v));
            const size_t tile = (size_t) (c / 32) * n_dot_tiles + (kk / 32);
            *(hf *) ((uint8_t *) wgt + tile * TILE_BYTES + tile_off(kk % 32, c % 32)) = (hf) v;
        }
    }
    hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
    memset(out, 0, (size_t) n_col_tiles * TILE_BYTES);

    const uint64_t f16_cycles = time_f16(out, act, wgt, scales, n_col_tiles, n_dot_tiles, iters);

    // the reference of the F16 GEMM in f32
    float * got = lab_ddr_alloc((size_t) 32 * n_col_tiles * 32 * sizeof(float), 128);
    float * ref = lab_ddr_alloc((size_t) 32 * n_col_tiles * 32 * sizeof(float), 128);
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < n_col_tiles * 32; c++) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < k; kk++) {
                acc += act_f[r * k + kk] * wgt_f[kk * n_col_tiles * 32 + c];
            }
            ref[r * n_col_tiles * 32 + c] = acc;
            const hf o = *(const hf *) ((const uint8_t *) out + (size_t) (c / 32) * TILE_BYTES + tile_off(r, c % 32));
            got[r * n_col_tiles * 32 + c] = (float) o;
        }
    }
    const size_t bad = lab_compare_f32("f16_gemm", got, ref, (size_t) 32 * n_col_tiles * 32, 0.25f, 1e-2f);

    const double tiles = (double) n_col_tiles * n_dot_tiles;
    lab_report(TARGET, "k", k, "");
    lab_report(TARGET, "f16_cycles_per_tile", (double) f16_cycles / tiles, "cycles");
    lab_report(TARGET, "f16_macs_per_cycle", 32768.0 * tiles / (double) f16_cycles, "MAC/cycle");

    // the int8 stream on the same buffers with random bytes
    uint8_t * act8  = lab_vtcm_alloc((size_t) n_dot_tiles * INT8_TILE_BYTES, TILE_BYTES);
    int8_t *  wgt8  = lab_vtcm_alloc((size_t) n_col_tiles * n_dot_tiles * INT8_TILE_BYTES, TILE_BYTES);
    uint8_t * out8  = lab_vtcm_alloc((size_t) n_col_tiles * INT8_TILE_BYTES, TILE_BYTES);
    uint8_t * bias8 = lab_vtcm_alloc(256, 256);
    lab_fill_u8(act8, (size_t) n_dot_tiles * INT8_TILE_BYTES);
    lab_fill_u8((uint8_t *) wgt8, (size_t) n_col_tiles * n_dot_tiles * INT8_TILE_BYTES);
    memset(bias8, 0, 256);
    memset(out8, 0, (size_t) n_col_tiles * INT8_TILE_BYTES);

    const uint64_t int8_cycles = time_int8(out8, act8, wgt8, bias8, n_col_tiles, n_dot_tiles, iters);
    lab_report(TARGET, "int8_cycles_per_1kb_tile", (double) int8_cycles / tiles, "cycles");
    lab_report(TARGET, "int8_over_f16_cycle_ratio", (double) int8_cycles / (double) f16_cycles, "");
    printf("lab: int8 output bytes: %02x %02x %02x %02x %02x %02x %02x %02x (conversion not validated)\n",
           out8[0], out8[1], out8[2], out8[3], out8[4], out8[5], out8[6], out8[7]);
    lab_report(TARGET, "mismatches", (double) bad, "");
    return bad ? 1 : 0;
}
