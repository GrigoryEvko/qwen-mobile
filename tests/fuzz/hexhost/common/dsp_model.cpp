// The model of the DSP side of the Hexagon backend for the hexhost fuzz harness.
// Refer to dsp_model.h. Each check names the file and the line of the DSP source
// that it models.

#include "dsp_model.h"

#include "ggml.h"

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

#include "htp-ops.h"
#include "hex-common.h"
#include "hex-fastdiv.h"
#include "matmul-ops.h"
#include "flash-attn-ops.h"
#include "unary-ops.h"
#include "get-rows-ops.h"
#include "set-rows-ops.h"
#include "rope-ops.h"

namespace fakedsp {

namespace {

// Gives a verdict with a failed status and a reason.
op_verdict fail(uint32_t status, const char * fmt, ...) __attribute__((format(printf, 2, 3)));
op_verdict fail(uint32_t status, const char * fmt, ...) {
    op_verdict v;
    v.status = status;
    char    buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    v.why = buf;
    return v;
}

// Gives a verdict of an op that returns OK and computes a wrong result or nothing.
op_verdict silent(const char * why) {
    op_verdict v;
    v.silent = true;
    v.why    = why;
    return v;
}

// Gives the verdict with the class of a known defect.
op_verdict tagged(op_verdict v, const char * tag) {
    v.tag = tag;
    return v;
}

// Gives a verdict of an op that writes VTCM beyond the reservation with no check.
op_verdict overflow(uint64_t need, uint64_t have, const char * what) {
    op_verdict v;
    v.vtcm_overflow = true;
    char buf[256];
    snprintf(buf, sizeof(buf), "%s needs %" PRIu64 " bytes of VTCM, the DSP has %" PRIu64 " and does not check", what,
             need, have);
    v.why = buf;
    return v;
}

// Gives the verdict of a VTCM layout of `total` bytes that is larger than the VTCM of the DSP. The
// DSP computes the layout with a 32-bit size_t. A layout of 4 GB or more thus gets a wrapped size on
// the DSP, which can fit, and the op then writes beyond VTCM. Such a layout gets the tag
// "vtcm-size-wrap". A smaller layout gets the refusal of the DSP with `status`.
op_verdict layout_too_large(uint64_t total, uint64_t vtcm, uint32_t status, const char * what) {
    if (total > UINT32_MAX) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: layout %" PRIu64 " > VTCM %" PRIu64 ": the DSP computes the layout with a "
                 "32-bit size_t, thus it gets a wrapped size and can write beyond VTCM", what, total, vtcm);
        op_verdict v;
        v.vtcm_overflow = true;
        v.why           = buf;
        return tagged(v, "vtcm-size-wrap");
    }
    return fail(status, "%s: layout %" PRIu64 " > VTCM %" PRIu64, what, total, vtcm);
}

bool aligned128(uint64_t a) {
    return (a & 127) == 0;
}

bool is_repacked(uint32_t t) {
    return t == HTP_TYPE_Q4_0 || t == HTP_TYPE_Q4_1 || t == HTP_TYPE_Q8_0 || t == HTP_TYPE_IQ4_NL ||
           t == HTP_TYPE_MXFP4 || t == HTP_TYPE_Q6_K || t == HTP_TYPE_Q4_K;
}

// htp_ops_context_set_n_threads (htp-ctx.h:131)
bool valid_n_threads(const dsp_ctx & ctx, int64_t n) {
    return n >= 1 && (uint64_t) n <= ctx.n_threads;
}

// hmx_mm_2d_f32 before its loops (matmul-ops.c:2648-2696). Gives OK or the reason of the ret -1.
op_verdict hmx_2d(const dsp_ctx & ctx, const htp_mm_kernel_params & k, uint64_t dst, uint64_t act, uint32_t kk,
                  uint32_t n, uint32_t wtype, int aligned_tile_size) {
    if (kk % 32 != 0 || n % 32 != 0) {
        return fail(HTP_STATUS_INTERNAL_ERR, "hmx-2d: k %u or n %u is not a multiple of 32", kk, n);
    }
    if (!aligned128(dst) || !aligned128(act)) {
        return tagged(fail(HTP_STATUS_INTERNAL_ERR, "hmx-2d: dst 0x%" PRIx64 " or activation 0x%" PRIx64
                           " is not 128-byte aligned", dst, act), "hmx-align");
    }
    if (htp_mm_get_tiled_row_stride((int) wtype, kk) == 0) {
        return fail(HTP_STATUS_INTERNAL_ERR, "hmx-2d: no row stride for type %u", wtype);
    }
    if (k.m_chunk <= 0 || k.n_chunk <= 0) {
        return fail(HTP_STATUS_INTERNAL_ERR, "hmx-2d: m_chunk %d n_chunk %d (a division by zero on the DSP)", k.m_chunk,
                    k.n_chunk);
    }
    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, (int) wtype, kk, (size_t) k.m_chunk, (size_t) k.n_chunk, 1,
                                 false, k.pipeline != 0, (uint32_t) k.n_act_threads, (uint32_t) aligned_tile_size);
    if (L.total_bytes > ctx.vtcm_size) {
        return layout_too_large(L.total_bytes, ctx.vtcm_size, HTP_STATUS_INTERNAL_ERR, "hmx-2d");
    }
    return op_verdict();
}

// The tile sizes of the kernel params must be those of the weight type (matmul-ops.c:2678-2680).
op_verdict hmx_tiles(const htp_mm_kernel_params & k, uint32_t wtype) {
    const bool quant = wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32;
    if (quant && (k.tile_size != (int32_t) htp_mm_get_weight_tile_size((int) wtype) ||
                  k.aligned_tile_size != (int32_t) htp_mm_get_weight_aligned_tile_size((int) wtype))) {
        return silent("hmx: the tile sizes of the kernel params do not match the weight type");
    }
    return op_verdict();
}

// hmx_mm_f16_f32_batched (matmul-ops.c:3223-3265) and its simple loop (3202-3221)
op_verdict hmx_batched(const dsp_ctx & ctx, const htp_mm_kernel_params & k, const tensor_ref & src0,
                       const tensor_ref & src1, const tensor_ref & dst) {
    const uint32_t kk = src0.ne[0], n = src0.ne[1];
    const uint32_t act_stride = src1.nb[1] / 4, weight_stride = src0.nb[1] / 2, dst_stride = dst.nb[1] / 4;
    if (act_stride < kk || weight_stride < kk || dst_stride < n) {
        return fail(HTP_STATUS_INTERNAL_ERR, "hmx-batched: a stride is below its row (act %u w %u dst %u, k %u n %u)",
                    act_stride, weight_stride, dst_stride, kk, n);
    }
    if (src0.ne[2] == 0 || src0.ne[3] == 0 || src1.ne[2] == 0 || src1.ne[3] == 0) {
        return fail(HTP_STATUS_INTERNAL_ERR, "hmx-batched: a zero batch dimension");
    }
    if (src1.ne[2] % src0.ne[2] != 0 || src1.ne[3] % src0.ne[3] != 0) {
        return fail(HTP_STATUS_INTERNAL_ERR, "hmx-batched: the batch dimensions do not broadcast");
    }
    if (kk % 32 != 0 || n % 32 != 0) {
        return fail(HTP_STATUS_INTERNAL_ERR, "hmx-batched: k %u or n %u is not a multiple of 32", kk, n);
    }
    if (!aligned128(dst.addr) || !aligned128(src1.addr)) {
        return tagged(fail(HTP_STATUS_INTERNAL_ERR, "hmx-batched: dst or activation is not 128-byte aligned"), "hmx-align");
    }
    const uint32_t group = src1.ne[2] / src0.ne[2];
    if (group > 1 && (uint64_t) (uint32_t) k.vtcm_size <= ctx.vtcm_size) {
        struct htp_mm_hmx_vtcm_layout L;
        htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, HTP_TYPE_F16, kk, (size_t) k.m_chunk,
                                     (size_t) k.n_chunk, group, act_stride > kk, false, (uint32_t) k.n_act_threads, 0);
        if (L.total_bytes <= ctx.vtcm_size) {
            return op_verdict();
        }
    }
    for (uint32_t b3 = 0; b3 < src1.ne[3]; b3++) {
        for (uint32_t b2 = 0; b2 < src1.ne[2]; b2++) {
            const uint64_t d = dst.addr + (uint64_t) b2 * dst.nb[2] + (uint64_t) b3 * dst.nb[3];
            const uint64_t a = src1.addr + (uint64_t) b2 * src1.nb[2] + (uint64_t) b3 * src1.nb[3];
            op_verdict     v = hmx_2d(ctx, k, d, a, kk, n, HTP_TYPE_F16, 0);
            if (v.status != HTP_STATUS_OK) {
                v.why = "hmx-batched-simple: " + v.why;
                return v;
            }
        }
    }
    return op_verdict();
}

// hvx_mm_matmul (matmul-ops.c:1502-1772) before the job run
op_verdict hvx_mm(const dsp_ctx & ctx, const htp_mm_kernel_params & k, const op_record & op) {
    const tensor_ref & src0 = op.src[0];
    const tensor_ref & src1 = op.src[1];
    const tensor_ref & src2 = op.src[2];
    const tensor_ref & dst  = op.dst[0];
    const uint32_t     wt   = src0.type;
    const bool         rp   = is_repacked(wt);
    const uint32_t     ne10 = src1.ne[0];
    const uint32_t     rows = src1.ne[1] * src1.ne[2] * src1.ne[3];

    size_t src1_row_size = src1.nb[1];
    switch (k.kernel_type) {
        case HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            if (wt != HTP_TYPE_F16) {
                return silent("hvx-mm: the F16 VTCM kernel with a weight that is not F16");
            }
            src1_row_size = hex_round_up(ne10 * 2, 128);
            break;
        case HTP_MM_KERNEL_HVX_F16_F32_DDR:
        case HTP_MM_KERNEL_HVX_F16_F16_DDR:
            if (wt != HTP_TYPE_F16) {
                return silent("hvx-mm: an F16 DDR kernel with a weight that is not F16");
            }
            break;
        case HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            if (wt != HTP_TYPE_F32) {
                return silent("hvx-mm: the F32 VTCM kernel with a weight that is not F32");
            }
            src1_row_size = hex_round_up(ne10 * 4, 128);
            break;
        case HTP_MM_KERNEL_HVX_F32_F32_DDR:
            if (wt != HTP_TYPE_F32) {
                return silent("hvx-mm: the F32 DDR kernel with a weight that is not F32");
            }
            break;
        case HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT:
            if (!rp) {
                return fail(HTP_STATUS_NO_SUPPORT, "hvx-mm: the flat quant kernel with weight type %u", wt);
            }
            src1_row_size = (wt == HTP_TYPE_Q4_1 || wt == HTP_TYPE_Q4_K) ? htp_mm_q8_1_flat_row_size(ne10)
                                                                         : htp_mm_q8_0_flat_row_size(ne10);
            break;
        default:
            if (!rp) {
                return fail(HTP_STATUS_NO_SUPPORT, "hvx-mm: kernel %d with weight type %u (hvx_mm_init_vec_dot fails)",
                            k.kernel_type, wt);
            }
            src1_row_size = (wt == HTP_TYPE_Q4_1 || wt == HTP_TYPE_Q4_K) ? htp_mm_q8_1_tiled_row_size(ne10)
                                                                         : htp_mm_q8_0_tiled_row_size(ne10);
            break;
    }
    if (k.n_prefetch <= 0) {
        return silent("hvx-mm: n_prefetch is not positive, the weight buffer of the layout is empty");
    }
    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, k.kernel_type, (int) wt, ne10, rows, (uint32_t) k.n_threads, dst.nb[1],
                                 src0.nb[1], src1_row_size, src2.present ? src2.nb[1] : 0, (uint32_t) k.n_prefetch, false,
                                 false);
    if (L.total_bytes > ctx.vtcm_size) {
        char what[32];
        snprintf(what, sizeof(what), "hvx-mm (kernel %d)", k.kernel_type);
        return layout_too_large(L.total_bytes, ctx.vtcm_size, HTP_STATUS_VTCM_TOO_SMALL, what);
    }
    return op_verdict();
}

// The shape relation of a matmul and of its fused add. A dst that is smaller
// than the product, or a src2 that is not a row or the full product, gives
// writes or reads outside the tensors on the DSP (no DSP check).
op_verdict mm_shapes(const op_record & op, bool with_src2) {
    const tensor_ref & src0 = op.src[0];
    const tensor_ref & src1 = op.src[1];
    const tensor_ref & dst  = op.dst[0];
    const bool         rp   = (src0.flags & HTP_TENSOR_REPACK) != 0;
    if ((rp ? dst.ne[0] > src0.ne[0 + 1] : dst.ne[0] != src0.ne[1]) || dst.ne[1] != src1.ne[1] ||
        dst.ne[2] != src1.ne[2] || dst.ne[3] != src1.ne[3]) {
        return tagged(silent("mm: the dst shape is not the shape of the product"), "mm-add-shape");
    }
    if (with_src2) {
        const tensor_ref & src2 = op.src[2];
        if (!src2.present) {
            return fail(HTP_STATUS_INVAL_PARAMS, "mm-add: no src2");
        }
        if (src2.ne[0] != dst.ne[0] || (src2.ne[1] != 1 && src2.ne[1] != dst.ne[1]) ||
            (src2.ne[2] != 1 && src2.ne[2] != dst.ne[2]) || (src2.ne[3] != 1 && src2.ne[3] != dst.ne[3])) {
            return tagged(silent("mm-add: src2 is not a row of the product or the product shape (reads outside src2)"),
                          "mm-add-src2");
        }
    }
    return op_verdict();
}

// op_matmul (matmul-ops.c:3733) with MUL_MAT and MUL_MAT_ADD
op_verdict model_matmul(const dsp_ctx & ctx, const op_record & op) {
    const auto & k = *(const htp_mm_kernel_params *) op.kparams;
    if (!op.src[0].present || !op.src[1].present || !op.dst[0].present) {
        return fail(HTP_STATUS_INVAL_PARAMS, "mm: a tensor is missing");
    }
    if (!valid_n_threads(ctx, k.n_threads)) {
        return fail(HTP_STATUS_INVAL_PARAMS, "mm: n_threads %d is outside [1, %u]", k.n_threads, ctx.n_threads);
    }
    if (k.n_hmx && (k.n_act_threads <= 0 || k.n_act_threads > k.n_threads)) {
        return fail(HTP_STATUS_INVAL_PARAMS, "mm: n_act_threads %d is outside [1, %d]", k.n_act_threads, k.n_threads);
    }
    op_verdict s = mm_shapes(op, op.opcode == HTP_OP_MUL_MAT_ADD);
    if (s.silent || s.status != HTP_STATUS_OK) {
        return s;
    }
    if (k.n_hmx) {
        if (!ctx.n_hmx) {
            return fail(HTP_STATUS_INTERNAL_ERR, "mm: an HMX kernel on a session with HMX off");
        }
        if (k.kernel_type == HTP_MM_KERNEL_HMX_F16_BATCHED) {
            return hmx_batched(ctx, k, op.src[0], op.src[1], op.dst[0]);
        }
        if (k.kernel_type != HTP_MM_KERNEL_HMX_2D) {
            return silent("mm: n_hmx is set but the kernel type is not an HMX kernel");
        }
        op_verdict t = hmx_tiles(k, op.src[0].type);
        if (t.silent) {
            return t;
        }
        return hmx_2d(ctx, k, op.dst[0].addr, op.src[1].addr, op.src[0].ne[0], op.src[0].ne[1], op.src[0].type,
                      k.aligned_tile_size);
    }
    return hvx_mm(ctx, k, op);
}

// op_matmul_nx (matmul-ops.c:4297) and hmx_mm_nx_2d_f32 (2886)
op_verdict model_matmul_nx(const dsp_ctx & ctx, const op_record & op) {
    const auto & k = *(const htp_mm_kernel_params *) op.kparams;
    if (!valid_n_threads(ctx, k.n_threads)) {
        return fail(HTP_STATUS_INVAL_PARAMS, "mm-nx: n_threads %d is outside [1, %u]", k.n_threads, ctx.n_threads);
    }
    if (k.n_hmx && (k.n_act_threads <= 0 || k.n_act_threads > k.n_threads)) {
        return fail(HTP_STATUS_INVAL_PARAMS, "mm-nx: n_act_threads %d", k.n_act_threads);
    }
    const uint32_t nw = (uint32_t) k.n_weights;
    if (nw == 0 || nw > HTP_OP_MAX_OUTPUTS || !op.src[nw].present) {
        return fail(HTP_STATUS_INVAL_PARAMS, "mm-nx: n_weights %u", nw);
    }
    for (uint32_t p = 0; p < nw; p++) {
        if (!op.src[p].present || !op.dst[p].present) {
            return silent("mm-nx: a weight or a dst of the n_weights is missing");
        }
        if (op.src[p].type != op.src[0].type || op.src[p].ne[0] != op.src[0].ne[0]) {
            return silent("mm-nx: the weights differ in type or in k");
        }
    }
    const tensor_ref & act = op.src[nw];
    const uint32_t     wt  = op.src[0].type;
    if (k.n_hmx) {
        if (!ctx.n_hmx) {
            return fail(HTP_STATUS_INTERNAL_ERR, "mm-nx: an HMX kernel on a session with HMX off");
        }
        const uint32_t kk = act.ne[0];
        if (kk % 32 != 0) {
            return fail(HTP_STATUS_NO_SUPPORT, "mm-nx-hmx: k %u is not a multiple of 32", kk);
        }
        if (!aligned128(act.addr)) {
            return tagged(fail(HTP_STATUS_NO_SUPPORT, "mm-nx-hmx: the activation 0x%" PRIx64 " is not 128-byte aligned",
                               act.addr), "hmx-align");
        }
        if (htp_mm_get_tiled_row_stride((int) wt, kk) == 0) {
            return fail(HTP_STATUS_NO_SUPPORT, "mm-nx-hmx: no row stride for type %u", wt);
        }
        op_verdict t = hmx_tiles(k, wt);
        if (t.silent) {
            return t;
        }
        struct htp_mm_hmx_vtcm_layout L;
        htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, (int) wt, kk, (size_t) k.m_chunk, (size_t) k.n_chunk, 1,
                                     false, k.pipeline != 0, (uint32_t) k.n_act_threads, (uint32_t) k.aligned_tile_size);
        if (L.total_bytes > ctx.vtcm_size) {
            return layout_too_large(L.total_bytes, ctx.vtcm_size, HTP_STATUS_VTCM_TOO_SMALL, "mm-nx-hmx");
        }
        return op_verdict();
    }
    if (!is_repacked(wt)) {
        return fail(HTP_STATUS_NO_SUPPORT, "mm-nx: weight type %u (hvx_mm_init_vec_dot fails)", wt);
    }
    if (wt == HTP_TYPE_Q6_K) {
        return silent("mm-nx: Q6_K takes the F16/F32 job hvx_mm_nx_2d (is_repacked of op_matmul_nx excludes Q6_K)");
    }
    const uint32_t rows = act.ne[1] * act.ne[2] * act.ne[3];
    size_t         src1_row_size;
    if (k.kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
        src1_row_size = (wt == HTP_TYPE_Q4_1 || wt == HTP_TYPE_Q4_K) ? htp_mm_q8_1_flat_row_size(act.ne[0])
                                                                     : htp_mm_q8_0_flat_row_size(act.ne[0]);
    } else {
        src1_row_size = (wt == HTP_TYPE_Q4_1 || wt == HTP_TYPE_Q4_K) ? htp_mm_q8_1_tiled_row_size(act.ne[0])
                                                                     : htp_mm_q8_0_tiled_row_size(act.ne[0]);
    }
    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, k.kernel_type, (int) wt, act.ne[0], rows, (uint32_t) k.n_threads, 0,
                                 op.src[0].nb[1], src1_row_size, 0, (uint32_t) k.n_prefetch, false, true);
    if (L.total_bytes > ctx.vtcm_size) {
        return layout_too_large(L.total_bytes, ctx.vtcm_size, HTP_STATUS_VTCM_TOO_SMALL, "mm-nx");
    }
    return op_verdict();
}

// op_matmul_id (matmul-ops.c:4092) and op_matmul_id_nx (4208) before the scan of the ids
op_verdict model_matmul_id(const dsp_ctx & ctx, const op_record & op, bool nx) {
    const auto & k = *(const htp_mm_kernel_params *) op.kparams;
    if (!valid_n_threads(ctx, k.n_threads)) {
        return fail(HTP_STATUS_INVAL_PARAMS, "mm-id: n_threads %d is outside [1, %u]", k.n_threads, ctx.n_threads);
    }
    if (k.n_hmx && (k.n_act_threads <= 0 || k.n_act_threads > k.n_threads)) {
        return fail(HTP_STATUS_INVAL_PARAMS, "mm-id: n_act_threads %d", k.n_act_threads);
    }
    const uint32_t nw = nx ? (uint32_t) k.n_weights : 1;
    if (nw == 0 || nw > HTP_OP_MAX_OUTPUTS || !op.src[nw].present || !op.src[nw + 1].present) {
        return fail(HTP_STATUS_INVAL_PARAMS, "mm-id: n_weights %u or the activation or the ids missing", nw);
    }
    const tensor_ref & src0 = op.src[0];
    const tensor_ref & act  = op.src[nw];
    if (k.n_hmx) {
        if (!ctx.n_hmx) {
            return fail(HTP_STATUS_INTERNAL_ERR, "mm-id: an HMX kernel on a session with HMX off");
        }
        // hmx_mm_id_2d_f32 (matmul-ops.c:3500-3506), with the base pointers of each weight
        for (uint32_t p = 0; p < nw; p++) {
            const tensor_ref & w = op.src[p];
            const tensor_ref & d = op.dst[p];
            if (!w.present || !d.present) {
                continue;
            }
            if (w.ne[0] % 32 != 0 || w.ne[1] % 32 != 0) {
                return fail(HTP_STATUS_NO_SUPPORT, "mm-id-hmx: k %u or n %u is not a multiple of 32", w.ne[0], w.ne[1]);
            }
            if (!aligned128(d.addr) || !aligned128(act.addr)) {
                return tagged(fail(HTP_STATUS_NO_SUPPORT, "mm-id-hmx: dst or activation is not 128-byte aligned"), "hmx-align");
            }
            if (htp_mm_get_tiled_row_stride((int) w.type, w.ne[0]) == 0) {
                return fail(HTP_STATUS_NO_SUPPORT, "mm-id-hmx: no row stride for type %u", w.type);
            }
        }
        return op_verdict();
    }
    if (!is_repacked(src0.type)) {
        return fail(HTP_STATUS_NO_SUPPORT, "mm-id: weight type %u (hvx_mm_init_vec_dot fails)", src0.type);
    }
    const uint32_t rows = act.ne[1] * act.ne[2] * act.ne[3];
    const size_t   src1_row_size = (src0.type == HTP_TYPE_Q4_1 || src0.type == HTP_TYPE_Q4_K)
                                       ? htp_mm_q8_1_tiled_row_size(act.ne[0])
                                       : htp_mm_q8_0_tiled_row_size(act.ne[0]);
    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, k.kernel_type, (int) src0.type, act.ne[0], rows, (uint32_t) k.n_threads, 0,
                                 src0.nb[1], src1_row_size, 0, (uint32_t) k.n_prefetch, true, false);
    if (L.total_bytes > ctx.vtcm_size) {
        return layout_too_large(L.total_bytes, ctx.vtcm_size, HTP_STATUS_VTCM_TOO_SMALL, "mm-id");
    }
    return op_verdict();
}

bool unary_has_tiled_task(uint32_t op) {
    switch (op) {
        case HTP_OP_NORM:
        case HTP_OP_RMS_NORM:
        case HTP_OP_RMS_NORM_MUL:
        case HTP_OP_L2_NORM:
            return false;
        default:
            return true;
    }
}

// op_unary and execute_op_unary (unary-ops.c:1140-1375) with the BLOCK rule of the tasks (unary-ops.c:716-748)
op_verdict model_unary(const dsp_ctx & ctx, const op_record & op) {
    const tensor_ref & src0 = op.src[0];
    const tensor_ref & dst  = op.dst[0];
    if (!src0.present || !dst.present) {
        return fail(HTP_STATUS_INVAL_PARAMS, "unary: a tensor is missing");
    }
    if (src0.type != HTP_TYPE_F32 && src0.type != HTP_TYPE_F16) {
        return fail(HTP_STATUS_NO_SUPPORT, "unary: src0 type %u", src0.type);
    }
    const bool f16 = src0.type == HTP_TYPE_F16;
    if (f16) {
        switch (op.opcode) {
            case HTP_OP_NORM:
            case HTP_OP_RMS_NORM:
            case HTP_OP_SCALE:
            case HTP_OP_CLAMP:
            case HTP_OP_SQR:
            case HTP_OP_SQRT:
            case HTP_OP_L2_NORM:
            case HTP_OP_UNARY_ABS:
            case HTP_OP_UNARY_LOG:
                break;
            default:
                return fail(HTP_STATUS_NO_SUPPORT, "unary: op %u is not supported for F16", op.opcode);
        }
    }
    const auto & k = *(const htp_unary_kernel_params *) op.kparams;
    if (!valid_n_threads(ctx, k.n_threads)) {
        return fail(HTP_STATUS_INVAL_PARAMS, "unary: n_threads %u is outside [1, %u]", k.n_threads, ctx.n_threads);
    }
    const uint32_t nrows = src0.ne[1] * src0.ne[2] * src0.ne[3];
    if (nrows == 0) {
        return op_verdict();
    }
    if (ctx.vtcm_size < (uint64_t) k.vtcm_size) {
        return fail(HTP_STATUS_VTCM_TOO_SMALL, "unary: kparams vtcm_size %u > VTCM %" PRIu64, k.vtcm_size, ctx.vtcm_size);
    }
    if (k.col_tile) {
        if (f16 || !unary_has_tiled_task(op.opcode)) {
            return f16 ? silent("unary: the F16 op takes an F32 tiled task")
                       : fail(HTP_STATUS_NO_SUPPORT, "unary: op %u has no tiled task (task function NULL)", op.opcode);
        }
        return op_verdict();
    }
    const bool rms_mul     = op.opcode == HTP_OP_RMS_NORM_MUL;
    const bool src0_contig = src0.nb[2] == (uint64_t) src0.ne[1] * src0.nb[1] && src0.nb[3] == (uint64_t) src0.ne[2] * src0.nb[2];
    const bool dst_contig  = dst.nb[2] == (uint64_t) dst.ne[1] * dst.nb[1] && dst.nb[3] == (uint64_t) dst.ne[2] * dst.nb[2];
    bool       clip        = false;
    if (rms_mul) {
        const tensor_ref & src1 = op.src[1];
        if (!src1.present) {
            return fail(HTP_STATUS_INVAL_PARAMS, "rms-norm-mul: no weight");
        }
        const bool src1_contig = src1.nb[2] == (uint64_t) src0.ne[1] * src1.nb[1] && src1.nb[3] == (uint64_t) src0.ne[2] * src1.nb[2];
        clip = !k.broadcast_weight && !src1_contig;
    }
    const uint32_t b0    = (src0_contig && !clip) ? k.block : (k.block < src0.ne[1] ? k.block : src0.ne[1]);
    const uint32_t b1    = (dst_contig && !clip) ? k.block : (k.block < dst.ne[1] ? k.block : dst.ne[1]);
    const uint32_t block = b0 < b1 ? b0 : b1;
    if (block == 0) {
        return tagged(silent("unary: BLOCK is 0, the task returns at once and the op returns OK with no output"),
                      "unary-block0");
    }
    return op_verdict();
}

// op_binary and execute_op_binary (binary-ops.c:753-917), with the thread count of the context
op_verdict model_binary(const dsp_ctx & ctx, const op_record & op) {
    const tensor_ref & src0 = op.src[0];
    const tensor_ref & src1 = op.src[1];
    const tensor_ref & dst  = op.dst[0];
    if (!src0.present || !src1.present || !dst.present) {
        return fail(HTP_STATUS_INVAL_PARAMS, "binary: a tensor is missing");
    }
    if (src1.nb[1] < src1.nb[0]) {
        return fail(HTP_STATUS_NO_SUPPORT, "binary: src1 is permuted");
    }
    if (src0.type != HTP_TYPE_F32 && src0.type != HTP_TYPE_F16) {
        return fail(HTP_STATUS_NO_SUPPORT, "binary: src0 type %u", src0.type);
    }
    const uint32_t nrows = src0.ne[1] * src0.ne[2] * src0.ne[3];
    if (nrows == 0) {
        return op_verdict();
    }
    const size_t es = src0.type == HTP_TYPE_F32 ? 4 : 2;
    const size_t r0 = src0.ne[0] * es, r1 = src1.ne[0] * es, rd = dst.ne[0] * es;
    const size_t a0 = hex_round_up((uint32_t) r0, 128), a1 = hex_round_up((uint32_t) r1, 128),
                 ad = hex_round_up((uint32_t) rd, 128);
    const bool is_add_id  = op.opcode == HTP_OP_ADD_ID;
    const bool is_scalar  = !is_add_id && src1.ne[0] == 1;
    const bool transposed = src0.nb[1] < r0 || src1.nb[1] < r1 || dst.nb[1] < rd;
    const bool same_shape = !is_add_id && !is_scalar && !transposed && src1.ne[0] == src0.ne[0] &&
                            src0.ne[0] % 128 == 0 && (src1.ne[1] == src0.ne[1] || src1.ne[1] == 1) &&
                            (src1.ne[2] == src0.ne[2] || src1.ne[2] == 1) && (src1.ne[3] == src0.ne[3] || src1.ne[3] == 1);
    const bool row_bcast = same_shape && src1.ne[1] == 1 && src1.ne[2] == 1 && src1.ne[3] == 1;
    const size_t total   = same_shape ? 2 * (a0 + a1 + ad) : 2 * (a0 + ad);
    const uint32_t n     = ctx.n_threads;
    size_t rows_per_buffer = ctx.vtcm_size / (n * total);
    if (row_bcast) {
        if (ctx.vtcm_size < a1) {
            return tagged(fail(HTP_STATUS_VTCM_TOO_SMALL, "binary: the broadcast row does not fit"), "vtcm-binary");
        }
        rows_per_buffer = (ctx.vtcm_size - a1) / (n * total);
    }
    if (rows_per_buffer < 1) {
        return tagged(fail(HTP_STATUS_VTCM_TOO_SMALL, "binary: no row of %zu bytes fits for %u threads", total, n),
                      "vtcm-binary");
    }
    return op_verdict();
}

// op_activations (act-ops.c:443-522)
op_verdict model_glu(const dsp_ctx & ctx, const op_record & op) {
    const tensor_ref & src0 = op.src[0];
    const tensor_ref & dst  = op.dst[0];
    if (!src0.present || !dst.present || src0.type != HTP_TYPE_F32) {
        return fail(HTP_STATUS_NO_SUPPORT, "glu: src0 missing or not F32");
    }
    if ((uint64_t) dst.ne[0] * 4 != dst.nb[1]) {
        return fail(HTP_STATUS_NO_SUPPORT, "glu: the dst rows are not contiguous");
    }
    if (src0.ne[1] * src0.ne[2] * src0.ne[3] == 0) {
        return op_verdict();
    }
    const size_t a = hex_round_up(dst.ne[0] * 4, 128);
    const size_t per_row = 3 * a;
    if (ctx.vtcm_size / (ctx.n_threads * per_row) == 0) {
        return tagged(fail(HTP_STATUS_VTCM_TOO_SMALL, "glu: no row of %zu bytes fits for %u threads", per_row,
                           ctx.n_threads), "vtcm-glu");
    }
    return op_verdict();
}

// op_softmax (softmax-ops.c:503-579)
op_verdict model_softmax(const dsp_ctx & ctx, const op_record & op) {
    const tensor_ref & src0 = op.src[0];
    const tensor_ref & dst  = op.dst[0];
    if (!src0.present || src0.type != HTP_TYPE_F32) {
        return fail(HTP_STATUS_NO_SUPPORT, "softmax: src0 not F32");
    }
    if (src0.ne[1] * src0.ne[2] * src0.ne[3] == 0) {
        return op_verdict();
    }
    const uint64_t s0 = hex_round_up(4 * src0.nb[1], 128);
    const uint64_t sd = hex_round_up(4 * dst.nb[1], 128);
    const uint64_t need = (2 * s0 + sd) * ctx.n_threads;
    if (ctx.vtcm_size < need) {
        return tagged(fail(HTP_STATUS_VTCM_TOO_SMALL, "softmax: scratch pads %" PRIu64 " > VTCM %" PRIu64, need,
                           ctx.vtcm_size), "vtcm-softmax");
    }
    return op_verdict();
}

// op_rope (rope-ops.c:693-803)
op_verdict model_rope(const dsp_ctx & ctx, const op_record & op) {
    if (!op.src[0].present || op.src[0].type != HTP_TYPE_F32) {
        return fail(HTP_STATUS_NO_SUPPORT, "rope: src0 not F32");
    }
    const auto & k = *(const htp_rope_kernel_params *) op.kparams;
    if (!valid_n_threads(ctx, k.n_threads)) {
        return fail(HTP_STATUS_INVAL_PARAMS, "rope: n_threads %u is outside [1, %u]", k.n_threads, ctx.n_threads);
    }
    if (ctx.vtcm_size < k.vtcm_size) {
        return tagged(overflow(k.vtcm_size, ctx.vtcm_size, "rope (an assert only)"), "vtcm-rope");
    }
    return op_verdict();
}

// op_flash_attn_ext (flash-attn-ops.c:2365-2487) and hmx_flash_attn_ext (1771-1889)
op_verdict model_fa(const dsp_ctx & ctx, const op_record & op) {
    const tensor_ref & q = op.src[0];
    const tensor_ref & kt = op.src[1];
    const tensor_ref & v = op.src[2];
    if (!q.present || !kt.present || !v.present) {
        return fail(HTP_STATUS_INVAL_PARAMS, "fa: q, k or v missing");
    }
    if ((q.type != HTP_TYPE_F16 && q.type != HTP_TYPE_F32) || (kt.type != HTP_TYPE_F16 && kt.type != HTP_TYPE_Q8_0) ||
        (v.type != HTP_TYPE_F16 && v.type != HTP_TYPE_Q8_0)) {
        return fail(HTP_STATUS_NO_SUPPORT, "fa: types q %u k %u v %u", q.type, kt.type, v.type);
    }
    const auto & k = *(const htp_fa_kernel_params *) op.kparams;
    if (k.kernel_type == HTP_FA_KERNEL_UNSUPPORTED) {
        return fail(HTP_STATUS_NO_SUPPORT, "fa: kernel type UNSUPPORTED");
    }
    if (!valid_n_threads(ctx, k.n_threads)) {
        return fail(HTP_STATUS_INVAL_PARAMS, "fa: n_threads %u is outside [1, %u]", k.n_threads, ctx.n_threads);
    }
    if (k.kernel_type == HTP_FA_KERNEL_HMX) {
        if (!ctx.n_hmx) {
            return fail(HTP_STATUS_NO_SUPPORT, "fa-hmx: HMX off");
        }
        if (q.ne[0] % 32 != 0 || v.ne[0] % 32 != 0) {
            return fail(HTP_STATUS_NO_SUPPORT, "fa-hmx: DK %u or DV %u is not a multiple of 32", q.ne[0], v.ne[0]);
        }
        if (k.Br == 0 || k.Bc == 0 || k.G == 0) {
            return fail(HTP_STATUS_INTERNAL_ERR, "fa-hmx: Br %u Bc %u G %u (a division by zero on the DSP)", k.Br, k.Bc, k.G);
        }
        struct hmx_fa_vtcm_layout L;
        hmx_fa_vtcm_layout_build(&L, k.G, q.ne[0], v.ne[0], k.Br, k.Bc, k.n_threads, k.u.hmx.pipeline != 0, k.is_q_fp32 != 0);
        if (L.total_bytes > ctx.vtcm_size) {
            return layout_too_large(L.total_bytes, ctx.vtcm_size, HTP_STATUS_VTCM_TOO_SMALL, "fa-hmx");
        }
        return op_verdict();
    }
    if (q.ne[2] > 512) {
        return fail(HTP_STATUS_NO_SUPPORT, "fa-hvx: %u heads, the DSP takes 512", q.ne[2]);
    }
    const uint64_t n  = k.n_threads;
    const uint64_t kb = (uint64_t) k.u.hvx.size_k_row_padded * 64;
    const uint64_t vb = (uint64_t) k.u.hvx.size_v_row_padded * 64;
    const uint64_t mb = hex_round_up(64 * 2, 128);
    const bool     has_mask = op.src[3].present;
    const uint64_t acc = hex_round_up(v.ne[0] * 4, 128);
    const uint64_t need = k.u.hvx.size_q_row_padded * n + kb * 2 * n + vb * 2 * n + (has_mask ? mb * 128 * n : 0) + acc * n;
    if (need > ctx.vtcm_size) {
        return tagged(fail(HTP_STATUS_VTCM_TOO_SMALL, "fa-hvx: scratch pads %" PRIu64 " > VTCM %" PRIu64, need,
                           ctx.vtcm_size), "vtcm-fa");
    }
    return op_verdict();
}

// op_get_rows (get-rows-ops.c:211-292). The DSP has no VTCM check.
op_verdict model_get_rows(const dsp_ctx & ctx, const op_record & op) {
    const tensor_ref & src0 = op.src[0];
    const tensor_ref & src1 = op.src[1];
    const tensor_ref & dst  = op.dst[0];
    if (src0.type != HTP_TYPE_F32 && src0.type != HTP_TYPE_F16 && src0.type != HTP_TYPE_Q8_0) {
        return fail(HTP_STATUS_NO_SUPPORT, "get-rows: src0 type %u", src0.type);
    }
    if (dst.type != HTP_TYPE_F32) {
        return fail(HTP_STATUS_NO_SUPPORT, "get-rows: dst type %u", dst.type);
    }
    if (src1.type != HTP_TYPE_I32 && src1.type != HTP_TYPE_I64) {
        return fail(HTP_STATUS_NO_SUPPORT, "get-rows: index type %u", src1.type);
    }
    const auto & k = *(const htp_get_rows_kernel_params *) op.kparams;
    if (k.total_tasks == 0) {
        return op_verdict();
    }
    if (!valid_n_threads(ctx, k.n_threads)) {
        return fail(HTP_STATUS_INVAL_PARAMS, "get-rows: n_threads %d is outside [1, %u]", k.n_threads, ctx.n_threads);
    }
    if (!k.use_dma) {
        struct htp_get_rows_vtcm_layout L;
        htp_get_rows_vtcm_layout_build(&L, (int) src0.type, src0.ne[0], (uint32_t) k.n_threads);
        if (L.total_bytes > ctx.vtcm_size) {
            return tagged(overflow(L.total_bytes, ctx.vtcm_size, "get-rows"), "vtcm-rows");
        }
    }
    return op_verdict();
}

// op_set_rows (set-rows-ops.c:187-260). The DSP has no VTCM check.
op_verdict model_set_rows(const dsp_ctx & ctx, const op_record & op) {
    const tensor_ref & src0 = op.src[0];
    const tensor_ref & dst  = op.dst[0];
    if (src0.type != HTP_TYPE_F32) {
        return fail(HTP_STATUS_NO_SUPPORT, "set-rows: src0 type %u", src0.type);
    }
    if (dst.type != HTP_TYPE_F32 && dst.type != HTP_TYPE_F16 && dst.type != HTP_TYPE_Q8_0) {
        return fail(HTP_STATUS_NO_SUPPORT, "set-rows: dst type %u", dst.type);
    }
    const auto & k = *(const htp_set_rows_kernel_params *) op.kparams;
    if (k.total_tasks == 0) {
        return op_verdict();
    }
    if (!valid_n_threads(ctx, k.n_threads)) {
        return fail(HTP_STATUS_INVAL_PARAMS, "set-rows: n_threads %d is outside [1, %u]", k.n_threads, ctx.n_threads);
    }
    struct htp_set_rows_vtcm_layout L;
    htp_set_rows_vtcm_layout_build(&L, (int) dst.type, src0.ne[0], (uint32_t) k.n_threads);
    if (L.total_bytes > ctx.vtcm_size) {
        return tagged(overflow(L.total_bytes, ctx.vtcm_size, "set-rows"), "vtcm-rows");
    }
    return op_verdict();
}

// op_cpy and exec_cpy (cpy-ops.c:411-567)
op_verdict model_cpy(const dsp_ctx & ctx, const op_record & op) {
    const tensor_ref & src0 = op.src[0];
    const tensor_ref & dst  = op.dst[0];
    if (!src0.present || !dst.present) {
        return fail(HTP_STATUS_INVAL_PARAMS, "cpy: a tensor is missing");
    }
    uint32_t ts0, tsd;
    switch (src0.type) {
        case HTP_TYPE_F32: ts0 = 4; break;
        case HTP_TYPE_F16: ts0 = 2; break;
        default: return fail(HTP_STATUS_NO_SUPPORT, "cpy: src0 type %u", src0.type);
    }
    switch (dst.type) {
        case HTP_TYPE_F32: tsd = 4; break;
        case HTP_TYPE_F16: tsd = 2; break;
        default: return fail(HTP_STATUS_NO_SUPPORT, "cpy: dst type %u", dst.type);
    }
    const bool same_type  = src0.type == dst.type;
    const bool transposed = src0.nb[0] > src0.nb[1] || dst.nb[0] > dst.nb[1] || src0.nb[0] != ts0 || dst.nb[0] != tsd ||
                            src0.nb[1] < src0.ne[0] * ts0 || dst.nb[1] < dst.ne[0] * tsd;
    const bool same_shape = !transposed && src0.ne[0] == dst.ne[0] && src0.ne[1] == dst.ne[1] &&
                            src0.ne[2] == dst.ne[2] && src0.ne[3] == dst.ne[3];
    (void) ctx;
    if (same_shape || same_type) {
        return op_verdict();
    }
    return tagged(fail(HTP_STATUS_NO_SUPPORT, "cpy: a type conversion with a change of shape or stride"), "cpy-layout");
}

// op_concat (concat-ops.c:233-328)
op_verdict model_concat(const dsp_ctx & ctx, const op_record & op) {
    const tensor_ref & src0 = op.src[0];
    const tensor_ref & src1 = op.src[1];
    const tensor_ref & dst  = op.dst[0];
    if (!src0.present || !src1.present || !dst.present) {
        return fail(HTP_STATUS_INVAL_PARAMS, "concat: a tensor is missing");
    }
    const int      dim = op.params[0];
    const bool     is_2d = dst.ne[2] == 1 && dst.ne[3] == 1;
    const uint32_t ts = (dst.type == HTP_TYPE_F32 || dst.type == HTP_TYPE_I32) ? 4 : 2;
    if (dim == 0 && is_2d && src1.nb[0] > src1.nb[1] && !(src0.nb[0] > src0.nb[1]) && dst.ne[1] > 0) {
        const uint32_t block_i = ts == 4 ? 32 : 64;
        const uint64_t s1p     = hex_round_up(src1.ne[0], block_i);
        const uint64_t row     = hex_round_up((uint32_t) ((src0.ne[0] + s1p) * ts), 128);
        const uint64_t sp0     = block_i * row;
        const uint64_t sp1     = s1p * block_i * ts + block_i * 128;
        const uint64_t need    = (sp0 + sp1) * ctx.n_threads;
        if (need > ctx.vtcm_size) {
            return tagged(fail(HTP_STATUS_VTCM_TOO_SMALL, "concat-2d-transposed: scratch pads %" PRIu64 " > VTCM %" PRIu64,
                               need, ctx.vtcm_size), "vtcm-concat");
        }
    }
    return op_verdict();
}

// op_gdn_conv_step (gdn-conv-ops.c:433-519) and op_gdn_conv_chunk (905-1008)
op_verdict model_gdn_conv(const dsp_ctx & ctx, const op_record & op, bool chunk) {
    const tensor_ref & states = op.src[0];
    const tensor_ref & s_copy = op.src[1];
    const tensor_ref & x      = op.src[2];
    const tensor_ref & w      = op.src[3];
    const tensor_ref & slot   = op.src[4];
    const tensor_ref & y      = op.dst[0];
    if (!states.present || !s_copy.present || !x.present || !w.present || !slot.present || !y.present) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-conv: a tensor is missing");
    }
    if (states.type != HTP_TYPE_F32 || x.type != HTP_TYPE_F32 || w.type != HTP_TYPE_F32 || slot.type != HTP_TYPE_F32 ||
        y.type != HTP_TYPE_F32 || s_copy.type != HTP_TYPE_I32) {
        return fail(HTP_STATUS_NO_SUPPORT, "gdn-conv: a type");
    }
    const uint32_t d_conv = w.ne[0];
    const uint32_t n_ch   = w.ne[1];
    if (d_conv != 4) {
        return fail(HTP_STATUS_NO_SUPPORT, "gdn-conv: d_conv %u", d_conv);
    }
    const uint32_t T = x.ne[0];
    if (chunk) {
        if (n_ch == 0 || n_ch % 32 != 0 || T < 4) {
            return fail(HTP_STATUS_INVAL_PARAMS, "gdn-conv-chunk: n_ch %u or T %u", n_ch, T);
        }
    } else if (n_ch == 0) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-conv: no channels");
    }
    const uint32_t row = (d_conv - 1) * n_ch;
    if (states.ne[0] != row || states.nb[0] != 4 || states.nb[1] != row * 4) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-conv: the state table shape");
    }
    if (chunk) {
        if (x.ne[1] != n_ch || x.nb[1] != 4 || x.nb[0] % 4 != 0) {
            return fail(HTP_STATUS_INVAL_PARAMS, "gdn-conv-chunk: the projection shape");
        }
    } else if (x.ne[0] != 1 || x.ne[1] != n_ch || x.nb[1] != 4) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-conv: the column shape");
    }
    if (w.nb[0] != 4 || w.nb[1] != d_conv * 4) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-conv: the weight strides");
    }
    if (slot.ne[0] != row || slot.nb[0] != 4) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-conv: the slot shape");
    }
    const uint32_t n_slots = slot.ne[1] > 0 ? slot.ne[1] : 1;
    if (n_slots > 1 && slot.nb[1] < row * 4) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-conv: %u rollback slots %u bytes apart, a slot is %u bytes", n_slots,
                    slot.nb[1], row * 4);
    }
    if (chunk) {
        if (y.ne[0] != n_ch || y.ne[1] != T || y.nb[0] != 4 || y.nb[1] % 4 != 0) {
            return fail(HTP_STATUS_INVAL_PARAMS, "gdn-conv-chunk: the output shape");
        }
    } else if (y.ne[0] != n_ch || y.nb[0] != 4) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-conv: the output shape");
    }
    const int32_t src_idx = op.kparams[0];
    if (op.kparams[1] != 1 || src_idx < 0 || (uint32_t) src_idx >= states.ne[1]) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-conv: slot index %d, valid flag %d, %u slots", src_idx, op.kparams[1],
                    states.ne[1]);
    }
    if (chunk) {
        const uint32_t per_thread = (n_ch + ctx.n_threads - 1) / ctx.n_threads;
        const uint32_t ch         = hex_round_up(per_thread, 32);
        const uint64_t need       = (uint64_t) hex_round_up(11 * ch * 4 + 512, 128) * ctx.n_threads;
        if (ctx.vtcm_size < need) {
            return fail(HTP_STATUS_VTCM_TOO_SMALL, "gdn-conv-chunk: VTCM %" PRIu64 " < %" PRIu64, ctx.vtcm_size, need);
        }
    }
    return op_verdict();
}

// op_gdn_state_step (gated-delta-net-ops.c:1136-1239)
op_verdict model_gdn_state(const dsp_ctx & ctx, const op_record & op) {
    for (int i = 0; i < 8; i++) {
        if (!op.src[i].present) {
            return fail(HTP_STATUS_INVAL_PARAMS, "gdn-state: input %d is missing", i);
        }
    }
    if (!op.dst[0].present) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-state: the output is missing");
    }
    const tensor_ref & q = op.src[0];
    const tensor_ref & k = op.src[1];
    const tensor_ref & v = op.src[2];
    const tensor_ref & g = op.src[3];
    const tensor_ref & beta = op.src[4];
    const tensor_ref & states = op.src[5];
    const tensor_ref & s_copy = op.src[6];
    const tensor_ref & slot = op.src[7];
    const tensor_ref & dst = op.dst[0];
    for (int i = 0; i < 8; i++) {
        if (i != 6 && op.src[i].type != HTP_TYPE_F32) {
            return fail(HTP_STATUS_NO_SUPPORT, "gdn-state: input %d is not F32", i);
        }
    }
    if (s_copy.type != HTP_TYPE_I32 || dst.type != HTP_TYPE_F32) {
        return fail(HTP_STATUS_NO_SUPPORT, "gdn-state: s_copy or dst type");
    }
    const uint32_t S_v = v.ne[0], H = v.ne[1], K = (uint32_t) op.params[0];
    if (S_v == 0 || S_v > 128 || H == 0 || v.ne[2] != 1 || v.ne[3] != 1 || K < 1) {
        return fail(HTP_STATUS_NO_SUPPORT, "gdn-state: S_v %u H %u tokens %u seqs %u K %u", S_v, H, v.ne[2], v.ne[3], K);
    }
    if ((g.ne[0] != 1 && g.ne[0] != S_v) || beta.ne[0] != 1) {
        return fail(HTP_STATUS_NO_SUPPORT, "gdn-state: the gate shape");
    }
    if (q.ne[0] != S_v || k.ne[0] != S_v || q.ne[1] == 0 || k.ne[1] == 0 || q.ne[2] != 1 || k.ne[2] != 1 ||
        q.ne[3] != 1 || k.ne[3] != 1) {
        return fail(HTP_STATUS_NO_SUPPORT, "gdn-state: the q or k shape");
    }
    const uint32_t D = S_v * S_v * H;
    if (states.ne[0] != D || states.nb[0] != 4 || states.nb[1] != D * 4) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-state: the state table shape");
    }
    if (slot.ne[0] != D || slot.nb[0] != 4) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-state: the slot shape");
    }
    if (dst.ne[0] != S_v * H || dst.ne[1] < 1 || dst.nb[0] != 4) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-state: the output shape");
    }
    const int32_t src_idx = op.kparams[0];
    if (op.kparams[1] != 1 || src_idx < 0 || (uint32_t) src_idx >= states.ne[1]) {
        return fail(HTP_STATUS_INVAL_PARAMS, "gdn-state: slot index %d, valid flag %d", src_idx, op.kparams[1]);
    }
    const uint64_t sa = ((uint64_t) S_v * S_v * 4 + 127) & ~(uint64_t) 127;
    if (ctx.vtcm_size < 2 * sa * ctx.n_threads) {
        return fail(HTP_STATUS_VTCM_TOO_SMALL, "gdn-state: VTCM %" PRIu64 " < %" PRIu64, ctx.vtcm_size,
                    2 * sa * ctx.n_threads);
    }
    return op_verdict();
}

// op_gated_delta_net (gated-delta-net-ops.c:994-1102) on the sequential path
op_verdict model_gdn(const dsp_ctx & ctx, const op_record & op) {
    for (int i = 0; i < 6; i++) {
        if (!op.src[i].present || op.src[i].type != HTP_TYPE_F32) {
            return fail(HTP_STATUS_NO_SUPPORT, "gdn: input %d missing or not F32", i);
        }
    }
    const tensor_ref & q = op.src[0];
    const tensor_ref & k = op.src[1];
    const tensor_ref & v = op.src[2];
    const tensor_ref & g = op.src[3];
    const tensor_ref & beta = op.src[4];
    const tensor_ref & state = op.src[5];
    const tensor_ref & dst = op.dst[0];
    const uint32_t S_v = v.ne[0], H = v.ne[1], T = v.ne[2], ns = v.ne[3], K = (uint32_t) op.params[0];
    if (S_v == 0 || S_v > 128 || H == 0 || T == 0 || ns == 0) {
        return fail(HTP_STATUS_NO_SUPPORT, "gdn: S_v %u H %u T %u seqs %u", S_v, H, T, ns);
    }
    if ((g.ne[0] != 1 && g.ne[0] != S_v) || beta.ne[0] != 1) {
        return fail(HTP_STATUS_NO_SUPPORT, "gdn: the gate shape");
    }
    if (q.ne[0] != S_v || k.ne[0] != S_v || q.ne[1] == 0 || k.ne[1] == 0 || q.ne[2] != T || k.ne[2] != T ||
        q.ne[3] == 0 || k.ne[3] == 0 || ns % q.ne[3] != 0 || ns % k.ne[3] != 0) {
        return fail(HTP_STATUS_NO_SUPPORT, "gdn: the q or k shape");
    }
    if (state.ne[0] != S_v || state.ne[1] != S_v || state.ne[2] != H || state.ne[3] != ns) {
        return fail(HTP_STATUS_NO_SUPPORT, "gdn: the state shape %ux%ux%ux%u", state.ne[0], state.ne[1], state.ne[2],
                    state.ne[3]);
    }
    if (dst.ne[0] != S_v * H || dst.ne[1] != T * ns + S_v * ns * K) {
        return fail(HTP_STATUS_NO_SUPPORT, "gdn: the output shape");
    }
    const uint64_t sa = ((uint64_t) S_v * S_v * 4 + 127) & ~(uint64_t) 127;
    if (ctx.vtcm_size < 2 * sa * ctx.n_threads) {
        return tagged(overflow(2 * sa * ctx.n_threads, ctx.vtcm_size, "gated-delta-net (an assert only)"), "vtcm-gdn");
    }
    return op_verdict();
}

} // namespace

const char * opcode_name(uint32_t opcode) {
    switch (opcode) {
        case HTP_OP_MUL: return "MUL";
        case HTP_OP_ADD: return "ADD";
        case HTP_OP_SUB: return "SUB";
        case HTP_OP_DIV: return "DIV";
        case HTP_OP_MUL_MAT: return "MUL_MAT";
        case HTP_OP_MUL_MAT_ID: return "MUL_MAT_ID";
        case HTP_OP_MUL_MAT_NX: return "MUL_MAT_NX";
        case HTP_OP_MUL_MAT_ID_NX: return "MUL_MAT_ID_NX";
        case HTP_OP_MUL_MAT_ADD: return "MUL_MAT_ADD";
        case HTP_OP_RMS_NORM: return "RMS_NORM";
        case HTP_OP_RMS_NORM_MUL: return "RMS_NORM_MUL";
        case HTP_OP_UNARY_SILU: return "SILU";
        case HTP_OP_UNARY_SIGMOID: return "SIGMOID";
        case HTP_OP_UNARY_SOFTPLUS: return "SOFTPLUS";
        case HTP_OP_UNARY_EXP: return "EXP";
        case HTP_OP_UNARY_NEG: return "NEG";
        case HTP_OP_GLU_SWIGLU: return "SWIGLU";
        case HTP_OP_SOFTMAX: return "SOFTMAX";
        case HTP_OP_ROPE: return "ROPE";
        case HTP_OP_FLASH_ATTN_EXT: return "FLASH_ATTN_EXT";
        case HTP_OP_SET_ROWS: return "SET_ROWS";
        case HTP_OP_GET_ROWS: return "GET_ROWS";
        case HTP_OP_SCALE: return "SCALE";
        case HTP_OP_CPY: return "CPY";
        case HTP_OP_CPY_FENCE: return "CPY_FENCE";
        case HTP_OP_SSM_CONV: return "SSM_CONV";
        case HTP_OP_L2_NORM: return "L2_NORM";
        case HTP_OP_GATED_DELTA_NET: return "GATED_DELTA_NET";
        case HTP_OP_CONCAT: return "CONCAT";
        case HTP_OP_FENCE: return "FENCE";
        case HTP_OP_GDN_CONV_STEP: return "GDN_CONV_STEP";
        case HTP_OP_GDN_STATE_STEP: return "GDN_STATE_STEP";
        case HTP_OP_GDN_CONV_CHUNK: return "GDN_CONV_CHUNK";
        default: return "OTHER";
    }
}

std::string verdict_id(const op_verdict & v, const char * op_name) {
    if (!v.tag.empty()) {
        return "dsp-" + v.tag;
    }
    if (v.status != HTP_STATUS_OK) {
        return std::string("dsp-refuse-") + op_name;
    }
    if (v.silent) {
        return std::string("dsp-silent-") + op_name;
    }
    return std::string("dsp-vtcm-overflow-") + op_name;
}

op_verdict model_op(const dsp_ctx & ctx, const op_record & op) {
    switch (op.opcode) {
        case HTP_OP_MUL_MAT:
        case HTP_OP_MUL_MAT_ADD:
            return model_matmul(ctx, op);
        case HTP_OP_MUL_MAT_NX:
            return model_matmul_nx(ctx, op);
        case HTP_OP_MUL_MAT_ID:
            return model_matmul_id(ctx, op, false);
        case HTP_OP_MUL_MAT_ID_NX:
            return model_matmul_id(ctx, op, true);
        case HTP_OP_MUL:
        case HTP_OP_ADD:
        case HTP_OP_SUB:
        case HTP_OP_DIV:
        case HTP_OP_ADD_ID:
            return model_binary(ctx, op);
        case HTP_OP_NORM:
        case HTP_OP_RMS_NORM:
        case HTP_OP_RMS_NORM_MUL:
        case HTP_OP_SCALE:
        case HTP_OP_CLAMP:
        case HTP_OP_LEAKY_RELU:
        case HTP_OP_SQR:
        case HTP_OP_SQRT:
        case HTP_OP_UNARY_SOFTPLUS:
        case HTP_OP_UNARY_SIGMOID:
        case HTP_OP_UNARY_SILU:
        case HTP_OP_UNARY_GELU:
        case HTP_OP_UNARY_GELU_QUICK:
        case HTP_OP_UNARY_NEG:
        case HTP_OP_UNARY_EXP:
        case HTP_OP_UNARY_TANH:
        case HTP_OP_UNARY_ABS:
        case HTP_OP_UNARY_LOG:
        case HTP_OP_UNARY_RELU:
        case HTP_OP_L2_NORM:
        case HTP_OP_TRI:
            return model_unary(ctx, op);
        case HTP_OP_GLU_SWIGLU:
        case HTP_OP_GLU_SWIGLU_OAI:
        case HTP_OP_GLU_SWIGLU_CLAMP:
        case HTP_OP_GLU_GEGLU:
            return model_glu(ctx, op);
        case HTP_OP_SOFTMAX:
            return model_softmax(ctx, op);
        case HTP_OP_ROPE:
            return model_rope(ctx, op);
        case HTP_OP_FLASH_ATTN_EXT:
            return model_fa(ctx, op);
        case HTP_OP_GET_ROWS:
            return model_get_rows(ctx, op);
        case HTP_OP_SET_ROWS:
            return model_set_rows(ctx, op);
        case HTP_OP_CPY:
        case HTP_OP_CPY_FENCE:
            return model_cpy(ctx, op);
        case HTP_OP_CONCAT:
            return model_concat(ctx, op);
        case HTP_OP_GDN_CONV_STEP:
            return model_gdn_conv(ctx, op, false);
        case HTP_OP_GDN_CONV_CHUNK:
            return model_gdn_conv(ctx, op, true);
        case HTP_OP_GDN_STATE_STEP:
            return model_gdn_state(ctx, op);
        case HTP_OP_GATED_DELTA_NET:
            return model_gdn(ctx, op);
        case HTP_OP_INVALID:
            return fail(HTP_STATUS_NO_SUPPORT, "unknown op (main.c:905)");
        default:
            if (op.opcode > HTP_OP_INVALID) {
                return fail(HTP_STATUS_NO_SUPPORT, "unknown op %u (main.c:905)", op.opcode);
            }
            return op_verdict();
    }
}

uint64_t tensor_extent(const tensor_ref & t) {
    if (!t.present) {
        return 0;
    }
    if (t.flags & HTP_TENSOR_REPACK) {
        return t.size;
    }
    for (int i = 0; i < 4; i++) {
        if (t.ne[i] == 0) {
            return 0;
        }
    }
    const int64_t blck = ggml_blck_size((enum ggml_type) t.type);
    uint64_t      nbytes;
    if (blck == 1) {
        nbytes = ggml_type_size((enum ggml_type) t.type);
        for (int i = 0; i < 4; i++) {
            nbytes += (uint64_t) (t.ne[i] - 1) * t.nb[i];
        }
    } else {
        nbytes = (uint64_t) t.ne[0] * t.nb[0] / blck;
        for (int i = 1; i < 4; i++) {
            nbytes += (uint64_t) (t.ne[i] - 1) * t.nb[i];
        }
    }
    return nbytes;
}

namespace {

// Gives the byte ranges that an op writes, as [addr, addr + size) pairs, for the touch mode.
void op_writes(const op_record & op, std::vector<std::pair<uint64_t, uint64_t>> & out) {
    out.clear();
    for (int i = 0; i < 4; i++) {
        const tensor_ref & d = op.dst[i];
        if (!d.present || (d.flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            continue;
        }
        uint64_t n = tensor_extent(d);
        if (op.opcode == HTP_OP_GDN_STATE_STEP && i == 0) {
            n = (uint64_t) d.ne[0] * 4;  // the attention row only
        }
        out.emplace_back(d.addr, n);
    }
}

// Writes seq and status into a fence slot, as htp_fence_write does (htp-fence.h:17).
void fence_write(uint64_t addr, uint32_t seq, uint32_t status) {
    uint32_t * f = (uint32_t *) (uintptr_t) addr;
    __atomic_store_n(&f[1], status, __ATOMIC_RELEASE);
    __atomic_store_n(&f[0], seq, __ATOMIC_RELEASE);
}

} // namespace

void process_batch(const dsp_ctx & ctx, const htp_opbatch_req & req, const struct dspqueue_buffer & dbuf,
                   batch_record & rec, htp_opbatch_rsp & rsp) {
    // The block size check of process_opbatch (main.c:1118). The real DSP drops the batch.
    const uint64_t b_size  = sizeof(htp_buf_desc) * (uint64_t) req.n_bufs;
    const uint64_t t_size  = sizeof(htp_tensor) * (uint64_t) req.n_tensors;
    const uint64_t o_size  = sizeof(htp_op_desc) * (uint64_t) req.n_ops;
    const uint64_t p_size  = sizeof(htp_prof_desc) * (uint64_t) req.n_ops;
    const uint64_t tr_size = (uint64_t) (HTP_MAX_NTHREADS + 1) * req.n_traces * sizeof(htp_trace_desc);
    const uint64_t need    = b_size + t_size + o_size + p_size + tr_size;
    if (dbuf.size < need || need > 0xffffffffull) {
        violation("desc-block-size", "the batch block is %u bytes, the descriptors need %" PRIu64
                  " (the DSP drops the batch and the host waits forever)", dbuf.size, need);
    }
    uint64_t base = 0, asize = 0;
    int      afd  = -1;
    if (!lookup_alloc((uint64_t) (uintptr_t) dbuf.ptr, &base, &asize, &afd) || (int) dbuf.fd != afd ||
        (uint64_t) (uintptr_t) dbuf.ptr + dbuf.size > base + asize || base + dbuf.offset != (uint64_t) (uintptr_t) dbuf.ptr) {
        violation("desc-block-range", "the batch block %p size %u fd %u offset %u is not inside its rpcmem allocation",
                  dbuf.ptr, dbuf.size, dbuf.fd, dbuf.offset);
        return;
    }
    if (req.n_bufs > HTP_OP_MAX_BUFS) {
        violation("desc-n-bufs", "the batch has %u buffers, the DSP maps %d", req.n_bufs, HTP_OP_MAX_BUFS);
    }

    const uint8_t *      m    = (const uint8_t *) dbuf.ptr;
    const htp_buf_desc * bufs = (const htp_buf_desc *) m;
    const htp_tensor *   tens = (const htp_tensor *) (m + b_size);
    const htp_op_desc *  ops  = (const htp_op_desc *) (m + b_size + t_size);
    htp_prof_desc *      pds  = (htp_prof_desc *) (m + b_size + t_size + o_size);

    // The buffers (prep_op_bufs, main.c:960). The sum of the buffer sizes must fit the VA budget of the DSP.
    uint64_t vmem = 0;
    for (uint32_t i = 0; i < req.n_bufs; i++) {
        uint64_t b = 0, s = 0;
        int      fd = -1;
        if (!lookup_alloc(bufs[i].base, &b, &s, &fd) || b != bufs[i].base || (uint32_t) fd != bufs[i].fd ||
            bufs[i].size > s) {
            violation("desc-buffer", "buffer #%u base 0x%" PRIx64 " size %" PRIu64 " fd %u is not an rpcmem allocation",
                      i, bufs[i].base, bufs[i].size, bufs[i].fd);
            return;
        }
        for (uint32_t j = 0; j < i; j++) {
            if (bufs[j].fd == bufs[i].fd) {
                violation("desc-buffer-dup", "buffers #%u and #%u have the same fd %u", j, i, bufs[i].fd);
            }
        }
        vmem += bufs[i].size;
    }
    if (ctx.max_vmem && vmem > ctx.max_vmem) {
        violation("desc-vmem", "the buffers of the batch need %" PRIu64 " bytes of DSP VA, the budget is %" PRIu64
                  " (prep_op_bufs aborts)", vmem, ctx.max_vmem);
    }

    // The tensors (prep_tensor, main.c:1018)
    std::vector<tensor_ref> refs(req.n_tensors);
    for (uint32_t i = 0; i < req.n_tensors; i++) {
        const htp_tensor & h = tens[i];
        if (h.bi >= req.n_bufs) {
            violation("desc-tensor-buffer", "tensor #%u has buffer index %u of %u", i, h.bi, req.n_bufs);
            return;
        }
        if ((uint64_t) h.data + h.size > bufs[h.bi].size) {
            violation("desc-tensor-range", "tensor #%u at offset %u size %u is outside its buffer of %" PRIu64 " bytes",
                      i, h.data, h.size, bufs[h.bi].size);
        }
        tensor_ref & r = refs[i];
        r.present = true;
        r.addr    = bufs[h.bi].base + h.data;
        r.size    = h.size;
        r.type    = h.type;
        r.flags   = h.flags;
        r.index   = (uint16_t) i;
        memcpy(r.ne, h.ne, sizeof(r.ne));
        memcpy(r.nb, h.nb, sizeof(r.nb));
        const uint64_t ext = tensor_extent(r);
        if (ext > h.size) {
            violation("desc-tensor-extent", "tensor #%u: the shape %ux%ux%ux%u with strides %u:%u:%u:%u spans %" PRIu64
                      " bytes, the descriptor declares %u", i, h.ne[0], h.ne[1], h.ne[2], h.ne[3], h.nb[0], h.nb[1],
                      h.nb[2], h.nb[3], ext, h.size);
        }
    }

    // The ops (proc_op_req, main.c:1040)
    std::vector<std::pair<uint64_t, uint64_t>> writes;
    for (uint32_t i = 0; i < req.n_ops; i++) {
        const htp_op_desc & o = ops[i];
        op_record           r;
        r.opcode = o.opcode;
        r.flags  = o.flags;
        memcpy(r.params, o.params, sizeof(r.params));
        memcpy(r.kparams, o.kernel_params, sizeof(r.kparams));
        for (int s = 0; s < HTP_OP_MAX_INPUTS; s++) {
            if (o.src[s] == 0xffff) {
                continue;
            }
            if (o.src[s] >= req.n_tensors) {
                violation("desc-op-src", "op #%u src %d has tensor index %u of %u", i, s, o.src[s], req.n_tensors);
                return;
            }
            r.src[s] = refs[o.src[s]];
        }
        for (int d = 0; d < HTP_OP_MAX_OUTPUTS; d++) {
            if (o.dst[d] == 0xffff) {
                continue;
            }
            if (o.dst[d] >= req.n_tensors) {
                violation("desc-op-dst", "op #%u dst %d has tensor index %u of %u", i, d, o.dst[d], req.n_tensors);
                return;
            }
            r.dst[d] = refs[o.dst[d]];
            if (r.dst[d].flags & HTP_TENSOR_WEIGHT) {
                violation("desc-op-writes-weight", "op #%u (%s) writes a weight tensor", i, opcode_name(o.opcode));
            }
        }

        const op_verdict  v = model_op(ctx, r);
        count(std::string("op ") + opcode_name(o.opcode));
        const std::string id = verdict_id(v, opcode_name(o.opcode));
        if (v.status != HTP_STATUS_OK) {
            // An ignored refusal keeps the op status OK, thus the host goes on as if the op ran
            r.note = v.why;
            violation(id.c_str(), "op #%u of %u (%s): the DSP returns status %u: %s", i, req.n_ops,
                      opcode_name(o.opcode), v.status, v.why.c_str());
        } else if (v.silent) {
            r.note = v.why;
            violation(id.c_str(), "op #%u of %u (%s): the DSP returns OK with a wrong or no result: %s", i, req.n_ops,
                      opcode_name(o.opcode), v.why.c_str());
        } else if (v.vtcm_overflow) {
            r.note = v.why;
            violation(id.c_str(), "op #%u of %u (%s): %s", i, req.n_ops, opcode_name(o.opcode), v.why.c_str());
        }

        if (r.status != HTP_STATUS_OK && rsp.status == HTP_STATUS_OK) {
            rsp.status     = r.status;
            rsp.err_op     = i;
            rsp.err_opcode = o.opcode;
        }

        // The memory traffic of the op: the reads of the inputs and the writes of the outputs
        if (ctx.touch) {
            volatile uint8_t sum = 0;
            for (int s = 0; s < HTP_OP_MAX_INPUTS; s++) {
                if (r.src[s].present) {
                    const uint8_t * p = (const uint8_t *) (uintptr_t) r.src[s].addr;
                    const uint64_t  n = tensor_extent(r.src[s]);
                    for (uint64_t b = 0; b < n; b += 64) {
                        sum = sum + p[b];
                    }
                }
            }
            (void) sum;
            op_writes(r, writes);
            for (auto & w : writes) {
                memset((void *) (uintptr_t) w.first, ctx.fill, w.second);
            }
        }

        // The fences (op_fence, main.c:711, and op_cpy, cpy-ops.c:549)
        if (o.opcode == HTP_OP_FENCE && r.src[0].present) {
            const uint32_t seq  = (uint32_t) o.params[0];
            const uint32_t mode = (uint32_t) o.params[1];
            if (mode == 1) {
                fence_write(r.src[0].addr, seq, rsp.status);
            } else {
                const uint32_t * f   = (const uint32_t *) (uintptr_t) r.src[0].addr;
                const uint32_t   cur = __atomic_load_n(&f[0], __ATOMIC_ACQUIRE);
                if ((int32_t) (cur - seq) < 0 && !getenv("HEXHOST_ASYNC_PEERS")) {
                    violation("fence-wait", "op #%u waits for fence seq 0x%x, the fence holds 0x%x and no peer signals it",
                              i, seq, cur);
                }
            }
        }
        if (o.opcode == HTP_OP_CPY_FENCE && r.src[1].present) {
            fence_write(r.src[1].addr, (uint32_t) o.params[0], rsp.status);
        }

        if (ctx.profiler) {
            memset(&pds[i], 0, sizeof(pds[i]));
            pds[i].opcode       = o.opcode;
            pds[i].usecs        = 1;
            pds[i].cycles_start = i * 10;
            pds[i].cycles_stop  = i * 10 + 5;
        }

        rec.ops.push_back(std::move(r));
    }
}

} // namespace fakedsp
