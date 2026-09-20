// Target: the data-movement kernels (cpy, concat, rope) and the two primitives they rest on.
//
// These kernels do layout work, thus the v79 limits that bind them are the permute width (1 per
// packet), the memory width (1 load and 1 store per packet) and the cost of a gather. The
// program measures each primitive on its own, so the floor of each kernel can be derived rather
// than guessed:
//
//   copy        the load and store floor of a pure vector copy
//   gatherT     a 32x32 f32 transpose through Q6_vgather_ARMVw, the way concat-ops.c does it
//   shuffT      the same transpose through the vshuff butterfly network
//   scalarT     the same transpose element by element, the way the cpy reshape path does it
//               when the rows are not contiguous (cpy-ops.c:167)
//   rope        hvx_rope_f32_aa, the interleaved rope, which holds 1 vdeal and 1 vshuff
//   rope_neox   hvx_rope_neox_f32_aa, the deinterleaved rope, which holds no permute
//
// Every transpose gives the same result, thus the program compares all three against one scalar
// reference with tolerance 0: a move must be exact.
//
// Arguments: --iters 8 --ne 128 --nrows 64
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "rope-ops.c"

#define TARGET "move"

// ---------------------------------------------------------------- the transpose primitives

// The 32x32 f32 transpose of concat-ops.c: one gather for each output row, with 32 offsets of
// the source row stride. The gather reads the VTCM, thus this is its best case.
static void transpose_gather(HVX_Vector * restrict dst, const HVX_Vector * restrict src,
                             HVX_Vector voff, uint32_t mu) {
    for (uint32_t i = 0; i < 32; i++) {
        const size_t rt = (size_t) ((const uint8_t *) src + i * sizeof(float));
        Q6_vgather_ARMVw(&dst[i], rt, mu, voff);
    }
}

// A shuffle butterfly was tried here and rejected. A 32 x 32 f32 transpose needs all 32 vectors
// live at once, which is the whole register file, thus every one of the 5 stages spills to
// memory: measured 1356 cycles against 150 for the gather, and the result was wrong as well.
// The gather unit is the right tool for a transpose on this part.

// The transpose of the cpy reshape path: one element for each iteration, with the index update
// that cpy-ops.c does. O(n^2) elements, no vector unit at all.
static void transpose_scalar(float * restrict dst, const float * restrict src) {
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            dst[c * 32 + r] = src[r * 32 + c];
        }
    }
}

// The realistic transposed copy of one 32x32 f32 tile, the shape that ggml_cont(ggml_transpose)
// asks for. The source rows are contiguous in DDR at a row stride, thus the tile is staged into
// the VTCM where the gather unit can address it, transposed by 32 independent gathers, and the
// output rows written back contiguously. Every gather is independent, thus the unit pipelines.
static void transpose_tile_gather(float * restrict dst, uint32_t dst_stride,
                                  const float * restrict src, uint32_t src_stride,
                                  HVX_Vector * restrict stage, HVX_Vector * restrict gtile,
                                  HVX_Vector voff, uint32_t mu) {
    for (uint32_t r = 0; r < 32; r++) {
        stage[r] = hvx_vmemu(src + (size_t) r * src_stride);
    }
    for (uint32_t c = 0; c < 32; c++) {
        Q6_vgather_ARMVw(&gtile[c], (size_t) ((const uint8_t *) stage + c * sizeof(float)), mu, voff);
    }
    for (uint32_t c = 0; c < 32; c++) {
        hvx_vmemu(dst + (size_t) c * dst_stride) = gtile[c];
    }
}

// The same tile the way cpy-ops.c does it today when the rows are not contiguous: one element
// for each iteration of the loop.
static void transpose_tile_scalar(float * restrict dst, uint32_t dst_stride,
                                  const float * restrict src, uint32_t src_stride) {
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            dst[(size_t) c * dst_stride + r] = src[(size_t) r * src_stride + c];
        }
    }
}

// ---------------------------------------------------------------- the reference

static void ref_transpose(float * dst, const float * src) {
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            dst[c * 32 + r] = src[r * 32 + c];
        }
    }
}

// Both rope kernels read the theta cache the same way: 32 cosines then 32 sines for each block
// of 64 elements. Only the DATA layout differs, thus only hvx_rope_f32_aa needs the permutes.
static void fill_theta_basic(float * t, uint32_t ne) {
    for (uint32_t b = 0; b * 64 < ne; b++) {
        for (uint32_t k = 0; k < 32; k++) {
            const double a = (double) (b * 32 + k) * 0.0137;
            t[b * 64 + k]      = (float) cos(a);
            t[b * 64 + 32 + k] = (float) sin(a);
        }
    }
}

// the deinterleaved layout of hvx_rope_neox_f32_aa: 32 cosines, then 32 sines, per block
static void fill_theta_neox(float * t, uint32_t ne) {
    const uint32_t he = ne / 2;
    for (uint32_t b = 0; b * 32 < he; b++) {
        for (uint32_t k = 0; k < 32; k++) {
            const double a = (double) (b * 32 + k) * 0.0137;
            t[b * 64 + k]      = (float) cos(a);
            t[b * 64 + 32 + k] = (float) sin(a);
        }
    }
}

// The interleaved rope: the pair i sits at src[2i] and src[2i+1], and its angle is the lane
// (i % 32) of the block (i / 32).
static void ref_rope_basic(float * dst, const float * src, uint32_t ne, const float * t) {
    for (uint32_t i = 0; i < ne / 2; i++) {
        const uint32_t b = i / 32, k = i % 32;
        const double c = t[b * 64 + k], s = t[b * 64 + 32 + k];
        const double x0 = src[2 * i + 0], x1 = src[2 * i + 1];
        dst[2 * i + 0] = (float) (x0 * c - x1 * s);
        dst[2 * i + 1] = (float) (x0 * s + x1 * c);
    }
}

static void ref_rope_neox(float * dst, const float * src, uint32_t ne, const float * t) {
    const uint32_t he = ne / 2;
    for (uint32_t i = 0; i < he; i++) {
        const uint32_t b = i / 32, k = i % 32;
        const double c = t[b * 64 + k], s = t[b * 64 + 32 + k];
        const double x0 = src[i], x1 = src[he + i];
        dst[i]      = (float) (x0 * c - x1 * s);
        dst[he + i] = (float) (x0 * s + x1 * c);
    }
}

int main(int argc, char ** argv) {
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 8);
    const uint32_t ne    = (uint32_t) lab_arg_long(argc, argv, "--ne", 128);
    const uint32_t nrows = (uint32_t) lab_arg_long(argc, argv, "--nrows", 64);

    lab_init();

    // the transpose block: 32 rows of 32 f32, in the VTCM where the gather needs it
    HVX_Vector * vt_src = lab_vtcm_alloc(32 * 128 + 256, 128);
    HVX_Vector * vt_dst = lab_vtcm_alloc(32 * 128 + 256, 128);
    float *      t_ref  = lab_ddr_alloc(32 * 32 * sizeof(float), 128);
    float *      t_got  = lab_ddr_alloc(32 * 32 * sizeof(float), 128);

    lab_fill_f32((float *) vt_src, 32 * 32, -2.0f, 2.0f);
    ref_transpose(t_ref, (const float *) vt_src);

    int32_t offsets[32] __attribute__((aligned(128)));
    for (int k = 0; k < 32; k++) {
        offsets[k] = k * 128;          // the source row stride, as concat-ops.c builds it
    }
    const HVX_Vector voff = *(HVX_Vector *) offsets;
    const uint32_t   mu   = 32 * 128;  // the span the gather may address

    uint64_t best_g = UINT64_MAX, best_c = UINT64_MAX, best_cp = UINT64_MAX;

    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        uint64_t t0 = lab_cycles();
        transpose_gather(vt_dst, vt_src, voff, mu);
        uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best_g) best_g = t1 - t0;
    }
    memcpy(t_got, vt_dst, 32 * 32 * sizeof(float));
    const size_t bad_g = lab_compare_f32("gatherT", t_got, t_ref, 32 * 32, 0.0f, 0.0f);

    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        uint64_t t0 = lab_cycles();
        transpose_scalar((float *) vt_dst, (const float *) vt_src);
        uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best_c) best_c = t1 - t0;
    }
    memcpy(t_got, vt_dst, 32 * 32 * sizeof(float));
    const size_t bad_c = lab_compare_f32("scalarT", t_got, t_ref, 32 * 32, 0.0f, 0.0f);

    // the plain vector copy of the same 1024 elements, the floor of a move
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        uint64_t t0 = lab_cycles();
        hvx_copy_uu((uint8_t *) vt_dst, (const uint8_t *) vt_src, 32 * 32, sizeof(float));
        uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best_cp) best_cp = t1 - t0;
    }

    lab_report(TARGET, "copy_1024_cycles",    (double) best_cp, "cycles");
    lab_report(TARGET, "copy_per_element",    (double) best_cp / 1024.0, "cycles");
    lab_report(TARGET, "gatherT_cycles",      (double) best_g, "cycles");
    lab_report(TARGET, "gatherT_per_gather",  (double) best_g / 32.0, "cycles");
    lab_report(TARGET, "scalarT_cycles",      (double) best_c, "cycles");
    lab_report(TARGET, "scalarT_per_element", (double) best_c / 1024.0, "cycles");
    lab_report(TARGET, "scalarT_over_gatherT", (double) best_c / (double) best_g, "x");
    lab_report(TARGET, "transpose_mismatches", (double) (bad_g + bad_c), "");

    // ------------------------------------------------------------ the tile pipeline
    // A 512 x 512 f32 transpose in 32x32 tiles: the shape of ggml_cont(ggml_transpose(kb)) of
    // the gated delta net, where kb is [CS, CS, n_chunks, H_v] with CS = 64.
    const uint32_t TN = (uint32_t) lab_arg_long(argc, argv, "--tn", 256);
    const uint32_t tiles = TN / 32;
    float * m_src = lab_ddr_alloc((size_t) TN * TN * sizeof(float) + 256, 128);
    float * m_dst = lab_ddr_alloc((size_t) TN * TN * sizeof(float) + 256, 128);
    float * m_ref = lab_ddr_alloc((size_t) TN * TN * sizeof(float), 128);
    HVX_Vector * stage = lab_vtcm_alloc(32 * 128 + 256, 128);
    HVX_Vector * gtile = lab_vtcm_alloc(32 * 128 + 256, 128);

    lab_fill_f32(m_src, (size_t) TN * TN, -2.0f, 2.0f);
    for (uint32_t r = 0; r < TN; r++) {
        for (uint32_t c = 0; c < TN; c++) {
            m_ref[(size_t) c * TN + r] = m_src[(size_t) r * TN + c];
        }
    }

    uint64_t best_tg = UINT64_MAX, best_ts = UINT64_MAX;
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        uint64_t t0 = lab_cycles();
        for (uint32_t tr = 0; tr < tiles; tr++) {
            for (uint32_t tc = 0; tc < tiles; tc++) {
                transpose_tile_gather(m_dst + (size_t) tc * 32 * TN + tr * 32, TN,
                                      m_src + (size_t) tr * 32 * TN + tc * 32, TN,
                                      stage, gtile, voff, mu);
            }
        }
        uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best_tg) best_tg = t1 - t0;
    }
    const size_t bad_tg = lab_compare_f32("tileT_gather", m_dst, m_ref, (size_t) TN * TN, 0.0f, 0.0f);

    memset(m_dst, 0, (size_t) TN * TN * sizeof(float));
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        uint64_t t0 = lab_cycles();
        for (uint32_t tr = 0; tr < tiles; tr++) {
            for (uint32_t tc = 0; tc < tiles; tc++) {
                transpose_tile_scalar(m_dst + (size_t) tc * 32 * TN + tr * 32, TN,
                                      m_src + (size_t) tr * 32 * TN + tc * 32, TN);
            }
        }
        uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best_ts) best_ts = t1 - t0;
    }
    const size_t bad_ts = lab_compare_f32("tileT_scalar", m_dst, m_ref, (size_t) TN * TN, 0.0f, 0.0f);

    const double m_elems = (double) TN * TN;
    lab_report(TARGET, "tileT_n", TN, "");
    lab_report(TARGET, "tileT_gather_cycles",   (double) best_tg, "cycles");
    lab_report(TARGET, "tileT_gather_per_elem", (double) best_tg / m_elems, "cycles");
    lab_report(TARGET, "tileT_scalar_cycles",   (double) best_ts, "cycles");
    lab_report(TARGET, "tileT_scalar_per_elem", (double) best_ts / m_elems, "cycles");
    lab_report(TARGET, "tileT_speedup",         (double) best_ts / (double) best_tg, "x");
    lab_report(TARGET, "tileT_mismatches",      (double) (bad_tg + bad_ts), "");

    // ------------------------------------------------------------ rope
    const uint32_t he = ne / 2;
    float * r_src = lab_ddr_alloc((size_t) nrows * ne * sizeof(float) + 256, 128);
    float * r_dst = lab_ddr_alloc((size_t) nrows * ne * sizeof(float) + 256, 128);
    float * r_ref = lab_ddr_alloc((size_t) ne * sizeof(float), 128);
    float * th_b  = lab_ddr_alloc((size_t) ne * sizeof(float) + 256, 128);
    float * th_n  = lab_ddr_alloc((size_t) (he + 31) / 32 * 64 * sizeof(float) + 256, 128);

    lab_fill_f32(r_src, (size_t) nrows * ne, -2.0f, 2.0f);
    fill_theta_basic(th_b, ne);
    fill_theta_neox(th_n, ne);

    uint64_t best_rb = UINT64_MAX, best_rn = UINT64_MAX;
    for (uint32_t it = 0; it < iters; it++) {
        memcpy(r_dst, r_src, (size_t) nrows * ne * sizeof(float));
        LAB_BARRIER();
        uint64_t t0 = lab_cycles();
        for (uint32_t r = 0; r < nrows; r++) {
            hvx_rope_f32_aa(r_dst + (size_t) r * ne, r_dst + (size_t) r * ne, ne, th_b);
        }
        uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best_rb) best_rb = t1 - t0;
    }
    ref_rope_basic(r_ref, r_src, ne, th_b);
    const size_t bad_rb = lab_compare_f32("rope_basic", r_dst, r_ref, ne, 1e-5f, 1e-4f);

    for (uint32_t it = 0; it < iters; it++) {
        memcpy(r_dst, r_src, (size_t) nrows * ne * sizeof(float));
        LAB_BARRIER();
        uint64_t t0 = lab_cycles();
        for (uint32_t r = 0; r < nrows; r++) {
            hvx_rope_neox_f32_aa(r_dst + (size_t) r * ne, r_dst + (size_t) r * ne, ne, th_n);
        }
        uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best_rn) best_rn = t1 - t0;
    }
    ref_rope_neox(r_ref, r_src, ne, th_n);
    const size_t bad_rn = lab_compare_f32("rope_neox", r_dst, r_ref, ne, 1e-5f, 1e-4f);

    const double rope_elems = (double) nrows * ne;
    lab_report(TARGET, "rope_ne", ne, "");
    lab_report(TARGET, "rope_nrows", nrows, "");
    lab_report(TARGET, "rope_basic_cycles",   (double) best_rb, "cycles");
    lab_report(TARGET, "rope_basic_per_elem", (double) best_rb / rope_elems, "cycles");
    lab_report(TARGET, "rope_neox_cycles",    (double) best_rn, "cycles");
    lab_report(TARGET, "rope_neox_per_elem",  (double) best_rn / rope_elems, "cycles");
    lab_report(TARGET, "rope_mismatches", (double) (bad_rb + bad_rn), "");

    return (bad_g + bad_c + bad_tg + bad_ts + bad_rb + bad_rn) ? 1 : 0;
}
