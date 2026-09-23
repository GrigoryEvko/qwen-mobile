// Target fadq: the preparation of a Q8_0 KV block for the HMX flash attention (flash-attn-ops.c).
//
// At decode the HMX flash attention spends its time in the HVX preparation of each KV block: the
// DMA puts the rows of the block into VTCM, then fa_k_interleave_thread and fa_v_interleave_thread
// write the rows into the HMX tiles. For one KV block of n rows of 256 values on one HVX thread,
// this target measures:
//   f16      the F16 path: hmx_interleave_rows_to_tiles (K) and hmx_interleave_cols_to_tiles (V)
//   q8ref    the Q8_0 path for a head dimension that is not a multiple of 64:
//            hvx_dequantize_row_q8_0_f16 on each row in place, then the two F16 interleaves
//   q8fused  the Q8_0 path for a multiple of 64: fa_q8_0_rows_to_tiles and fa_q8_0_cols_to_tiles,
//            which convert two blocks (64 values) for each vector straight into the tiles
// and compares every K and V tile value of q8fused with q8ref. The program includes the kernel file,
// thus it calls the functions of the kernel and not copies of them.
//
// --convert-only 1 times the conversion alone (hvx_dequantize_row_q8_0_f16 against
// hvx_dequantize_pair_q8_0_f16), because the timing model of the simulator does not complete a
// run with vscatter (the K tiles). The tile check then runs in functional mode (MODE=functional).
// --exhaustive 1 also compares the two conversions for every quant (-128..127) and every finite
// F16 scale, 16.3 million values. The only permitted differences are signed zeros of a product
// with a negative or a zero scale, which no Q8_0 quantizer writes.
//
// Arguments: --rows 256 --iters 5 --exhaustive 0 --convert-only 0
// Results of 2026-09-23 (v79, PROFILE=release): 656 against 66 cycles for each row of 256 values
// (--convert-only 1 --rows 64); 0 different tile values for 1, 63 and 64 rows; the exhaustive
// check gives 31999 differences, all signed zeros.
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include "lab.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

// The kernel file needs the DMA queue of dma-queue.h, and the standalone runtime of the simulator
// does not give the user DMA engine to the thread (refer to target_fa.c). This target calls no
// function that moves data with the DMA, thus a copy-at-once queue is sufficient for the build.
#define HTP_DMA_H

typedef struct {
    void *       dst;
    const void * src;
} dma_ptr;

#define LAB_DMA_CAPACITY 64

typedef struct dma_queue_s {
    void *   dst[LAB_DMA_CAPACITY];
    uint32_t push_idx;
    uint32_t pop_idx;
} dma_queue;
typedef dma_queue * dma_queue_t;

static inline dma_ptr dma_make_ptr(void * dst, const void * src) {
    dma_ptr p = { dst, src };
    return p;
}

static inline bool dma_queue_push(dma_queue * q, dma_ptr p, size_t dst_stride, size_t src_stride, size_t row_size,
                                  size_t nrows) {
    for (size_t r = 0; r < nrows; r++) {
        memcpy((uint8_t *) p.dst + r * dst_stride, (const uint8_t *) p.src + r * src_stride, row_size);
    }
    q->dst[q->push_idx++ & (LAB_DMA_CAPACITY - 1)] = p.dst;
    return true;
}

static inline dma_ptr dma_queue_pop(dma_queue * q) {
    dma_ptr p = { NULL, NULL };
    if (q->pop_idx != q->push_idx) {
        p.dst = q->dst[q->pop_idx++ & (LAB_DMA_CAPACITY - 1)];
    }
    return p;
}

#define DMA_CACHE_MAX_SIZE 128

typedef struct {
    uint8_t * base;
    uint32_t  line_size;
    uint32_t  capacity;
} dma_cache;

static inline void dma_cache_init(dma_cache * c, uint8_t * base, uint32_t line_size, uint32_t capacity) {
    c->base      = base;
    c->line_size = line_size;
    c->capacity  = capacity;
}

static inline bool dma_cache_push(dma_queue * q, dma_cache * c, const uint8_t * src, uint32_t dst_stride,
                                  uint32_t src_stride, uint32_t row_size, uint32_t nrows) {
    return dma_queue_push(q, dma_make_ptr(c->base, src), dst_stride, src_stride, row_size, nrows);
}

#include "flash-attn-ops.c"

#define TARGET     "fadq"
#define DIM        256                    // DK = DV of Qwen3.5
#define ROW_Q8     (DIM / 32 * 34)        // 272 bytes of one Q8_0 row
#define ROW_STRIDE 512                    // the VTCM row stride of the kernel (size_k_row_padded)

// ---------------------------------------------------------------------------------------------
// The measured calls. noinline keeps each path a separate function in the profile.

#define KERNEL __attribute__((noinline))

// A load of the last tiles waits for the scatters of the call, as the HMX loads of the kernel do.
static inline void drain(const __fp16 * kt, const __fp16 * vt) {
    volatile HVX_Vector sink;
    sink = *(const HVX_Vector *) kt;
    sink = *(const HVX_Vector *) vt;
    (void) sink;
}

static KERNEL void run_f16(__fp16 * kt, __fp16 * vt, const __fp16 * rows, uint32_t n, uint32_t n_row_tiles) {
    hmx_interleave_rows_to_tiles(kt, rows, n, DIM, ROW_STRIDE / 2, 0, n);
    hmx_interleave_cols_to_tiles(vt, rows, n, DIM, ROW_STRIDE / 2, n_row_tiles, 0, n);
    drain(kt, vt);
}

// The dequantization in place: the K rows and the V rows of a block are in their own VTCM buffers.
static KERNEL void run_q8ref(__fp16 * kt, __fp16 * vt, uint8_t * krows, uint8_t * vrows, uint32_t n,
                             uint32_t n_row_tiles) {
    for (uint32_t r = 0; r < n; ++r) {
        __fp16 * row = (__fp16 *) (krows + (size_t) r * ROW_STRIDE);
        hvx_dequantize_row_q8_0_f16(row, row, DIM);
    }
    hmx_interleave_rows_to_tiles(kt, (const __fp16 *) krows, n, DIM, ROW_STRIDE / 2, 0, n);
    for (uint32_t r = 0; r < n; ++r) {
        __fp16 * row = (__fp16 *) (vrows + (size_t) r * ROW_STRIDE);
        hvx_dequantize_row_q8_0_f16(row, row, DIM);
    }
    hmx_interleave_cols_to_tiles(vt, (const __fp16 *) vrows, n, DIM, ROW_STRIDE / 2, n_row_tiles, 0, n);
    drain(kt, vt);
}

static KERNEL void run_q8fused(__fp16 * kt, __fp16 * vt, const uint8_t * rows, uint32_t n, uint32_t n_row_tiles) {
    fa_q8_0_rows_to_tiles(kt, rows, n, DIM, ROW_STRIDE, 0, n);
    fa_q8_0_cols_to_tiles(vt, rows, n, DIM, ROW_STRIDE, n_row_tiles, 0, n);
    drain(kt, vt);
}

static KERNEL void run_convert_ref(uint8_t * rows, uint32_t n) {
    for (uint32_t r = 0; r < n; ++r) {
        __fp16 * row = (__fp16 *) (rows + (size_t) r * ROW_STRIDE);
        hvx_dequantize_row_q8_0_f16(row, row, DIM);
    }
}

static KERNEL void run_convert_new(HVX_Vector * restrict out, const uint8_t * restrict rows, uint32_t n) {
    for (uint32_t r = 0; r < n; ++r) {
        const uint8_t * p = rows + (size_t) r * ROW_STRIDE;
#pragma clang loop unroll_count(4)
        for (uint32_t i = 0; i < DIM / 64; ++i) {
            out[r * (DIM / 64) + i] = hvx_dequantize_pair_q8_0_f16(p + 2 * sizeof(block_q8_0) * i);
        }
    }
}

// ---------------------------------------------------------------------------------------------

// Writes one Q8_0 row of 8 blocks: random quants, and scales in the range of a KV cache.
static void fill_q8_row(uint8_t * row) {
    for (uint32_t b = 0; b < DIM / 32; ++b) {
        uint8_t *      blk = row + b * sizeof(block_q8_0);
        const uint16_t d   = lab_f32_to_hf(lab_rand_f32(0.0005f, 0.5f));
        memcpy(blk, &d, 2);
        for (uint32_t j = 0; j < 32; ++j) {
            blk[2 + j] = (uint8_t) (lab_rand_u32() & 0xFF);
        }
    }
}

// The exhaustive check of the element conversion: for each finite F16 scale, one row whose 8
// blocks hold that scale and whose 256 quants are -128..127, converted by both functions. A pair
// of zeros of different sign counts apart from the other differences, and the other differences
// count by the class of the scale (zero or subnormal, normal). O(63488 x 256).
static size_t g_zero_sign, g_sub, g_norm;

static size_t exhaustive(uint8_t * row_a, uint8_t * row_b, __fp16 * out_b) {
    size_t bad = 0, shown = 0;
    for (uint32_t s = 0; s < 65536; ++s) {
        if ((s & 0x7C00) == 0x7C00) {
            continue;  // Inf and NaN are not scales of a KV cache
        }
        for (uint32_t b = 0; b < DIM / 32; ++b) {
            uint8_t *      blk = row_a + b * sizeof(block_q8_0);
            const uint16_t d   = (uint16_t) s;
            memcpy(blk, &d, 2);
            for (uint32_t j = 0; j < 32; ++j) {
                blk[2 + j] = (uint8_t) (int8_t) (b * 32 + j - 128);
            }
        }
        memcpy(row_b, row_a, ROW_Q8);
        hvx_dequantize_row_q8_0_f16((__fp16 *) row_a, row_a, DIM);
        for (uint32_t i = 0; i < DIM / 64; ++i) {
            hvx_vmem(out_b + 64 * i) = hvx_dequantize_pair_q8_0_f16(row_b + 2 * sizeof(block_q8_0) * i);
        }
        for (uint32_t e = 0; e < DIM; ++e) {
            uint16_t x, y;
            memcpy(&x, (const uint16_t *) row_a + e, 2);
            memcpy(&y, (const uint16_t *) out_b + e, 2);
            if (x == y) {
                continue;
            }
            ++bad;
            if (((x | y) & 0x7FFF) == 0) {
                ++g_zero_sign;
                continue;
            }
            if ((s & 0x7C00) == 0) {
                ++g_sub;
            } else {
                ++g_norm;
            }
            if (shown < 12) {
                printf("lab: fadq exhaustive mismatch scale 0x%04x quant %d: ref 0x%04x fused 0x%04x\n", s,
                       (int) e - 128, x, y);
                ++shown;
            }
        }
    }
    return bad;
}

int main(int argc, char ** argv) {
    const uint32_t n_rows = (uint32_t) lab_arg_long(argc, argv, "--rows", 256);
    const uint32_t iters  = (uint32_t) lab_arg_long(argc, argv, "--iters", 5);
    const long     exh    = lab_arg_long(argc, argv, "--exhaustive", 0);
    const long     conv   = lab_arg_long(argc, argv, "--convert-only", 0);
    if (n_rows < 1 || n_rows > 1024) {
        printf("lab: error: --rows must be 1..1024\n");
        return 2;
    }
    lab_init();

    // The tiles cover whole 32-row tiles, as the VTCM plan of the kernel does.
    const uint32_t n_row_tiles = (n_rows + 31) / 32;
    const size_t   tile_bytes  = (size_t) n_row_tiles * (DIM / 32) * HMX_FP16_TILE_SIZE;

    uint8_t * q8      = lab_vtcm_alloc((size_t) n_rows * ROW_STRIDE, 128);  // the Q8_0 rows after the DMA
    uint8_t * work_k  = lab_vtcm_alloc((size_t) n_rows * ROW_STRIDE, 128);  // the in-place copies of q8ref
    uint8_t * work_v  = lab_vtcm_alloc((size_t) n_rows * ROW_STRIDE, 128);
    __fp16 *  f16rows = lab_vtcm_alloc((size_t) n_rows * ROW_STRIDE, 128);  // F16 rows of the same size
    __fp16 *  kt_ref  = lab_vtcm_alloc(tile_bytes, 2048);
    __fp16 *  vt_ref  = lab_vtcm_alloc(tile_bytes, 2048);
    __fp16 *  kt_new  = lab_vtcm_alloc(tile_bytes, 2048);
    __fp16 *  vt_new  = lab_vtcm_alloc(tile_bytes, 2048);

    for (uint32_t r = 0; r < n_rows; ++r) {
        fill_q8_row(q8 + (size_t) r * ROW_STRIDE);
        for (uint32_t e = 0; e < DIM; ++e) {
            f16rows[(size_t) r * (ROW_STRIDE / 2) + e] = (__fp16) lab_rand_f32(-1.0f, 1.0f);
        }
    }

    if (conv) {
        HVX_Vector * out       = lab_vtcm_alloc((size_t) n_rows * ROW_STRIDE, 128);
        uint64_t     best_cref = UINT64_MAX, best_cnew = UINT64_MAX;
        for (uint32_t it = 0; it < iters; ++it) {
            memcpy(work_k, q8, (size_t) n_rows * ROW_STRIDE);
            LAB_BARRIER();
            uint64_t t0 = lab_cycles();
            run_convert_ref(work_k, n_rows);
            uint64_t t1 = lab_cycles();
            best_cref   = t1 - t0 < best_cref ? t1 - t0 : best_cref;
            t0          = lab_cycles();
            run_convert_new(out, q8, n_rows);
            t1        = lab_cycles();
            best_cnew = t1 - t0 < best_cnew ? t1 - t0 : best_cnew;
        }
        size_t bad = 0;
        for (uint32_t r = 0; r < n_rows; ++r) {
            bad += memcmp(work_k + (size_t) r * ROW_STRIDE, (const uint8_t *) out + (size_t) r * ROW_STRIDE,
                          DIM * 2) != 0;
        }
        lab_report(TARGET, "rows", n_rows, "");
        lab_report(TARGET, "convert_ref_cycles_per_row", (double) best_cref / n_rows, "cycles");
        lab_report(TARGET, "convert_new_cycles_per_row", (double) best_cnew / n_rows, "cycles");
        lab_report(TARGET, "convert_rows_different", (double) bad, "");
        return 0;
    }

    uint64_t best_f16 = UINT64_MAX, best_ref = UINT64_MAX, best_new = UINT64_MAX;
    for (uint32_t it = 0; it < iters; ++it) {
        uint64_t t0 = lab_cycles();
        run_f16(kt_ref, vt_ref, f16rows, n_rows, n_row_tiles);
        uint64_t t1 = lab_cycles();
        best_f16    = t1 - t0 < best_f16 ? t1 - t0 : best_f16;

        // q8ref works in place, thus it gets a fresh copy of the rows each time.
        memcpy(work_k, q8, (size_t) n_rows * ROW_STRIDE);
        memcpy(work_v, q8, (size_t) n_rows * ROW_STRIDE);
        memset(kt_ref, 0, tile_bytes);
        memset(vt_ref, 0, tile_bytes);
        LAB_BARRIER();
        t0 = lab_cycles();
        run_q8ref(kt_ref, vt_ref, work_k, work_v, n_rows, n_row_tiles);
        t1       = lab_cycles();
        best_ref = t1 - t0 < best_ref ? t1 - t0 : best_ref;

        memset(kt_new, 0, tile_bytes);
        memset(vt_new, 0, tile_bytes);
        LAB_BARRIER();
        t0 = lab_cycles();
        run_q8fused(kt_new, vt_new, q8, n_rows, n_row_tiles);
        t1       = lab_cycles();
        best_new = t1 - t0 < best_new ? t1 - t0 : best_new;
    }

    // vscatter is asynchronous: the loads of drain() order it, as the kernel does before the HMX.
    size_t bad_k = 0, bad_v = 0;
    for (size_t i = 0; i < tile_bytes / 2; ++i) {
        uint16_t a, b, c, d;
        memcpy(&a, (const uint16_t *) kt_ref + i, 2);
        memcpy(&b, (const uint16_t *) kt_new + i, 2);
        memcpy(&c, (const uint16_t *) vt_ref + i, 2);
        memcpy(&d, (const uint16_t *) vt_new + i, 2);
        bad_k += a != b;
        bad_v += c != d;
    }

    lab_report(TARGET, "rows", n_rows, "");
    lab_report(TARGET, "f16_cycles_per_row", (double) best_f16 / n_rows, "cycles");
    lab_report(TARGET, "q8ref_cycles_per_row", (double) best_ref / n_rows, "cycles");
    lab_report(TARGET, "q8fused_cycles_per_row", (double) best_new / n_rows, "cycles");
    lab_report(TARGET, "q8ref_over_f16", (double) best_ref / best_f16, "x");
    lab_report(TARGET, "q8fused_over_f16", (double) best_new / best_f16, "x");
    lab_report(TARGET, "k_tile_values_different", (double) bad_k, "");
    lab_report(TARGET, "v_tile_values_different", (double) bad_v, "");

    if (exh) {
        uint8_t *    row_a = lab_vtcm_alloc(ROW_STRIDE, 128);
        uint8_t *    row_b = lab_vtcm_alloc(ROW_STRIDE, 128);
        __fp16 *     out_b = lab_vtcm_alloc(ROW_STRIDE, 128);
        const size_t bad   = exhaustive(row_a, row_b, out_b);
        lab_report(TARGET, "exhaustive_values_different", (double) bad, "");
        lab_report(TARGET, "exhaustive_zero_sign_only", (double) g_zero_sign, "");
        lab_report(TARGET, "exhaustive_subnormal_or_zero_scale", (double) g_sub, "");
        lab_report(TARGET, "exhaustive_normal_scale", (double) g_norm, "");
    }
    return 0;
}
