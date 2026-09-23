// fuzz_kparams: the op support checks, the kernel selection and the kernel
// params of the host, against the model of the DSP side checks.
//
// Each input gives a session (thread count, HMX, VTCM size, the switches of the
// backend) and a small graph of one to four nodes with the shapes of Qwen3.5
// and random shapes: one op, or a chain that the host fuses (RMS_NORM+MUL,
// MUL_MAT+ADD, MUL_MAT_NX, MUL_MAT_ID_NX). The tensors get fake addresses with
// a selected alignment, thus the shapes can be as large as the model. When
// supports_op accepts every node, the harness packs the graph with the real
// op batch of the host (pack_graph) and runs the model of the DSP checks on
// each packed op. The invariant: the host never packs an op that the DSP
// refuses, computes wrong, or runs outside its VTCM.

#include "fuzz_death.h"
#include "harness.h"
#include "hexhost.h"
#include "fake_dsp.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <fuzzer/FuzzedDataProvider.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {

// The dims of Qwen3.5 2B and 4B and of common models
const int64_t K_DIMS[] = { 32, 64, 96, 128, 256, 512, 1024, 2048, 2560, 3072, 4096, 5120, 6144, 9216, 12288, 16384 };
const int64_t N_DIMS[] = { 1, 16, 32, 48, 64, 96, 128, 256, 1024, 2048, 2560, 4096, 6144, 8192, 9216, 12288,
                           32768, 65536, 151936, 248320 };
const int64_t M_DIMS[] = { 1, 2, 3, 4, 5, 6, 8, 16, 31, 32, 33, 64, 100, 128, 256, 512, 1024, 2048 };
const int64_t ROW_DIMS[] = { 1, 7, 31, 32, 33, 64, 128, 256, 1000, 2048, 2560, 4096, 9216, 32768, 65536, 100000,
                             131072, 248320, 524288 };

// Picks a value from a list or a random value in [lo, hi].
int64_t pick(FuzzedDataProvider & fdp, const int64_t * list, size_t n, int64_t lo, int64_t hi) {
    if (fdp.ConsumeBool()) {
        return list[fdp.ConsumeIntegralInRange<size_t>(0, n - 1)];
    }
    return fdp.ConsumeIntegralInRange<int64_t>(lo, hi);
}

// The graph of one input and the resources that it holds
struct builder {
    FuzzedDataProvider &               fdp;
    hexhost::device *                  dev;
    ggml_context *                     ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;

    builder(FuzzedDataProvider & f, hexhost::device * d) : fdp(f), dev(d) {
        ggml_init_params p = { 64u << 20, nullptr, true };
        ctx                = ggml_init(p);
    }

    ~builder() {
        ggml_free(ctx);
        for (auto * b : bufs) {
            ggml_backend_buffer_free(b);
        }
    }

    // Gives a tensor its own small buffer and a fake address with a selected misalignment.
    void place(ggml_tensor * t, bool weight) {
        ggml_backend_buffer_t b = hexhost::fake_buffer_new(dev, 4096);
        bufs.push_back(b);
        static const size_t offs[] = { 0, 0, 0, 0, 0, 0, 4, 32, 64, 128, 256 };
        const size_t        off    = weight ? 0 : offs[fdp.ConsumeIntegralInRange<size_t>(0, 10)];
        t->data                    = (char *) ggml_backend_buffer_get_base(b) + off;
        hexhost::init_tensor(b, t, weight);
    }

    // Creates a leaf tensor with an address.
    ggml_tensor * leaf(ggml_type type, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1, bool weight = false) {
        ggml_tensor * t = nullptr;
        if (!harness::guarded([&] { t = ggml_new_tensor_4d(ctx, type, n0, n1, n2, n3); })) {
            return nullptr;
        }
        place(t, weight);
        return t;
    }

    // Gives the extras to the nodes of a graph (views and results) as the allocator does.
    void place_nodes(ggml_cgraph * gf) {
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if (n->extra) {
                continue;
            }
            if (n->view_src) {
                // ggml_backend_view_init: a view gets its address from its source
                n->data = (char *) n->view_src->data + n->view_offs;
                hexhost::init_tensor(n->view_src->buffer, n, false);
            } else {
                place(n, false);
            }
        }
    }
};

// Builds the nodes of one scenario. Gives the last nodes of the chains, or an
// empty list when the input does not give a valid ggml graph.
std::vector<ggml_tensor *> build(builder & b, int scenario) {
    FuzzedDataProvider & fdp = b.fdp;
    ggml_context *       ctx = b.ctx;
    std::vector<ggml_tensor *> out;
    bool ok = true;

    auto weight_type = [&]() {
        static const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0,
                                           GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
                                           GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 };
        return types[fdp.ConsumeIntegralInRange<size_t>(0, 10)];
    };
    auto k_for = [&](ggml_type t) {
        int64_t k = pick(fdp, K_DIMS, sizeof(K_DIMS) / 8, 1, 20000);
        const int64_t blk = (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K) ? 256 : ggml_blck_size(t);
        k = ((k + blk - 1) / blk) * blk;
        return k;
    };
    // An activation of k columns and m rows, as a leaf or as a view of a wider leaf
    auto activation = [&](int64_t k, int64_t m, int64_t m2 = 1, ggml_type type = GGML_TYPE_F32) -> ggml_tensor * {
        const int mode = fdp.ConsumeIntegralInRange<int>(0, 5);
        if (mode == 0) {
            const int64_t pad = fdp.ConsumeIntegralInRange<int64_t>(1, 64);
            ggml_tensor * base = b.leaf(type, k + pad, m, m2);
            if (!base) {
                return nullptr;
            }
            const size_t  col  = fdp.ConsumeIntegralInRange<size_t>(0, (size_t) pad) * ggml_type_size(type);
            ggml_tensor * v    = nullptr;
            if (!harness::guarded([&] { v = ggml_view_3d(ctx, base, k, m, m2, base->nb[1], base->nb[2], col); })) {
                return nullptr;
            }
            return v;
        }
        return b.leaf(type, k, m, m2);
    };

    switch (scenario) {
        case 0:    // MUL_MAT
        case 1:    // MUL_MAT + ADD
        case 2: {  // MUL_MAT_NX
            const ggml_type wt   = weight_type();
            const int64_t   k    = k_for(wt);
            const int64_t   n    = pick(fdp, N_DIMS, sizeof(N_DIMS) / 8, 1, 300000);
            const int64_t   m    = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 4096);
            const bool      bat  = (wt == GGML_TYPE_F16 || wt == GGML_TYPE_F32) && harness::rare(fdp, 4);
            const int64_t   w2   = bat ? fdp.ConsumeIntegralInRange<int64_t>(1, 8) : 1;
            const int64_t   r2   = bat ? fdp.ConsumeIntegralInRange<int64_t>(1, 4) : 1;
            const ggml_type at   = (wt != GGML_TYPE_F32 && harness::rare(fdp, 8)) ? GGML_TYPE_F16 : GGML_TYPE_F32;
            ggml_tensor *   w    = b.leaf(wt, k, n, w2, 1, true);
            ggml_tensor *   x    = activation(k, m, w2 * r2, at);
            if (!w || !x) {
                return {};
            }
            ggml_tensor * mm = nullptr;
            ok = harness::guarded([&] { mm = ggml_mul_mat(ctx, w, x); });
            if (!ok) {
                return {};
            }
            if (scenario == 0) {
                out.push_back(mm);
            } else if (scenario == 1) {
                const int form = fdp.ConsumeIntegralInRange<int>(0, 5);
                ggml_tensor * lhs = mm;
                if (form == 5) {
                    // the ADD reads the product through a view at offset 0 with fewer rows
                    const int64_t rows = fdp.ConsumeIntegralInRange<int64_t>(1, mm->ne[1]);
                    const int64_t cols = fdp.ConsumeIntegralInRange<int64_t>(1, mm->ne[0]);
                    ok = harness::guarded([&] { lhs = ggml_view_2d(ctx, mm, cols, rows, mm->nb[1], 0); });
                }
                ggml_tensor * bias = nullptr;
                switch (form) {
                    case 0: bias = b.leaf(GGML_TYPE_F32, lhs->ne[0]); break;
                    case 1: bias = b.leaf(GGML_TYPE_F32, lhs->ne[0], lhs->ne[1], lhs->ne[2]); break;
                    case 2: bias = b.leaf(GGML_TYPE_F32, 1); break;
                    case 3: bias = b.leaf(GGML_TYPE_F32, 1, lhs->ne[1], lhs->ne[2]); break;
                    default: bias = b.leaf(GGML_TYPE_F32, lhs->ne[0]); break;
                }
                if (!ok || !bias) {
                    return {};
                }
                ggml_tensor * add = nullptr;
                const bool swap = fdp.ConsumeBool() && ggml_can_repeat(lhs, bias);
                ok = harness::guarded([&] { add = swap ? ggml_add(ctx, bias, lhs) : ggml_add(ctx, lhs, bias); });
                if (!ok) {
                    return {};
                }
                out.push_back(add);
            } else {
                out.push_back(mm);
                const int n_more = fdp.ConsumeIntegralInRange<int>(1, 4);
                for (int j = 0; j < n_more; j++) {
                    const int64_t n2 = fdp.ConsumeBool() ? n : pick(fdp, N_DIMS, sizeof(N_DIMS) / 8, 1, 300000);
                    // A weight of another type needs a row of whole blocks of that type, else keep wt
                    ggml_type     t2 = harness::rare(fdp, 6) ? weight_type() : wt;
                    const int64_t b2 = (t2 == GGML_TYPE_Q4_K || t2 == GGML_TYPE_Q6_K) ? 256 : ggml_blck_size(t2);
                    if (k % b2 != 0) {
                        t2 = wt;
                    }
                    ggml_tensor * w2t = b.leaf(t2, k, n2, w2, 1, true);
                    ggml_tensor * mm2 = nullptr;
                    if (!w2t || !harness::guarded([&] { mm2 = ggml_mul_mat(ctx, w2t, x); })) {
                        return {};
                    }
                    out.push_back(mm2);
                }
            }
            break;
        }
        case 3: {  // MUL_MAT_ID, one or two with the same ids
            const ggml_type wt     = weight_type();
            const int64_t   k      = k_for(wt);
            const int64_t   n      = pick(fdp, N_DIMS, sizeof(N_DIMS) / 8, 32, 20000);
            const int64_t   n_exp  = fdp.ConsumeIntegralInRange<int64_t>(1, 64);
            const int64_t   n_used = fdp.ConsumeIntegralInRange<int64_t>(1, n_exp < 8 ? n_exp : 8);
            // Fewer tokens than experts gives ne12 / ne02 = 0 in the matmul params of the host
            const int64_t   tok    = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 512);
            ggml_tensor *   w      = b.leaf(wt, k, n, n_exp, 1, true);
            ggml_tensor *   x      = b.leaf(GGML_TYPE_F32, k, fdp.ConsumeBool() ? n_used : 1, tok);
            ggml_tensor *   ids    = b.leaf(GGML_TYPE_I32, n_used, tok);
            if (!w || !x || !ids) {
                return {};
            }
            ggml_tensor * mm = nullptr;
            if (!harness::guarded([&] { mm = ggml_mul_mat_id(ctx, w, x, ids); })) {
                return {};
            }
            out.push_back(mm);
            if (fdp.ConsumeBool()) {
                ggml_tensor * w1  = b.leaf(wt, k, fdp.ConsumeBool() ? n : n + 32, n_exp, 1, true);
                ggml_tensor * mm1 = nullptr;
                if (!w1 || !harness::guarded([&] { mm1 = ggml_mul_mat_id(ctx, w1, x, ids); })) {
                    return {};
                }
                out.push_back(mm1);
            }
            break;
        }
        case 4: {  // RMS_NORM (+ MUL)
            const int64_t ne0 = pick(fdp, ROW_DIMS, sizeof(ROW_DIMS) / 8, 1, 600000);
            const int64_t ne1 = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 64);
            const int64_t ne2 = fdp.ConsumeIntegralInRange<int64_t>(1, 4);
            ggml_tensor * x   = activation(ne0, ne1, ne2);
            if (!x) {
                return {};
            }
            ggml_tensor * r = nullptr;
            const int     op = fdp.ConsumeIntegralInRange<int>(0, 2);
            if (!harness::guarded([&] {
                    r = op == 0 ? ggml_rms_norm(ctx, x, 1e-6f) : op == 1 ? ggml_norm(ctx, x, 1e-5f) : ggml_l2_norm(ctx, x, 1e-6f);
                })) {
                return {};
            }
            if (op == 0 && fdp.ConsumeBool()) {
                ggml_tensor * w = fdp.ConsumeBool() ? b.leaf(GGML_TYPE_F32, ne0) : b.leaf(GGML_TYPE_F32, ne0, ne1, ne2);
                ggml_tensor * m = nullptr;
                if (!w || !harness::guarded([&] { m = fdp.ConsumeBool() ? ggml_mul(ctx, r, w) : ggml_mul(ctx, w, r); })) {
                    // ggml_mul(w, r) aborts when w does not hold r, which is not a valid graph
                    return {};
                }
                out.push_back(m);
            } else {
                out.push_back(r);
            }
            break;
        }
        case 5: {  // FLASH_ATTN_EXT
            static const int64_t HD[] = { 32, 64, 80, 96, 128, 192, 256, 512 };
            const int64_t dk   = HD[fdp.ConsumeIntegralInRange<size_t>(0, 7)];
            const int64_t dv   = fdp.ConsumeBool() ? dk : HD[fdp.ConsumeIntegralInRange<size_t>(0, 7)];
            const int64_t nkv  = fdp.ConsumeIntegralInRange<int64_t>(1, 16);
            const int64_t g    = fdp.ConsumeIntegralInRange<int64_t>(1, 8);
            const int64_t nq   = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 4096);
            const int64_t kv   = fdp.ConsumeIntegralInRange<int64_t>(1, 40000);
            const ggml_type qt = fdp.ConsumeBool() ? GGML_TYPE_F32 : GGML_TYPE_F16;
            const ggml_type kt = harness::rare(fdp, 4) ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
            if (kt == GGML_TYPE_Q8_0 && (dk % 32 || dv % 32)) {
                return {};
            }
            ggml_tensor * q = b.leaf(qt, dk, nq, nkv * g);
            ggml_tensor * k = b.leaf(kt, dk, kv, nkv);
            ggml_tensor * v = b.leaf(kt, dv, kv, nkv);
            ggml_tensor * mask = fdp.ConsumeBool() ? b.leaf(GGML_TYPE_F16, kv, nq + fdp.ConsumeIntegralInRange<int64_t>(0, 63)) : nullptr;
            if (!q || !k || !v) {
                return {};
            }
            ggml_tensor * fa = nullptr;
            if (!harness::guarded([&] { fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f); })) {
                return {};
            }
            out.push_back(fa);
            break;
        }
        case 6: {  // unary ops, F32 and F16
            const bool    f16 = harness::rare(fdp, 4);
            const int64_t ne0 = pick(fdp, ROW_DIMS, sizeof(ROW_DIMS) / 8, 1, 600000);
            const int64_t ne1 = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 64);
            ggml_tensor * x   = activation(ne0, ne1, 1, f16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
            if (!x) {
                return {};
            }
            ggml_tensor * r  = nullptr;
            const int     op = fdp.ConsumeIntegralInRange<int>(0, 11);
            if (!harness::guarded([&] {
                    switch (op) {
                        case 0: r = ggml_silu(ctx, x); break;
                        case 1: r = ggml_sigmoid(ctx, x); break;
                        case 2: r = ggml_softplus(ctx, x); break;
                        case 3: r = ggml_exp(ctx, x); break;
                        case 4: r = ggml_scale(ctx, x, 0.5f); break;
                        case 5: r = ggml_sqr(ctx, x); break;
                        case 6: r = ggml_sqrt(ctx, x); break;
                        case 7: r = ggml_neg(ctx, x); break;
                        case 8: r = ggml_abs(ctx, x); break;
                        case 9: r = ggml_rms_norm(ctx, x, 1e-6f); break;
                        case 10: r = ggml_l2_norm(ctx, x, 1e-6f); break;
                        default: r = ggml_clamp(ctx, x, -1.0f, 1.0f); break;
                    }
                })) {
                return {};
            }
            out.push_back(r);
            break;
        }
        case 7: {  // SWIGLU (split)
            const int64_t ne0 = pick(fdp, ROW_DIMS, sizeof(ROW_DIMS) / 8, 1, 300000);
            const int64_t ne1 = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 64);
            ggml_tensor * a   = activation(ne0, ne1);
            ggml_tensor * g   = activation(ne0, ne1);
            ggml_tensor * r   = nullptr;
            if (!a || !g || !harness::guarded([&] { r = ggml_swiglu_split(ctx, a, g); })) {
                return {};
            }
            out.push_back(r);
            break;
        }
        case 8: {  // SOFTMAX
            const int64_t ne0  = pick(fdp, ROW_DIMS, sizeof(ROW_DIMS) / 8, 1, 200000);
            const int64_t ne1  = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 64);
            const int64_t ne2  = fdp.ConsumeIntegralInRange<int64_t>(1, 8);
            ggml_tensor * x    = b.leaf(GGML_TYPE_F32, ne0, ne1, ne2);
            ggml_tensor * mask = fdp.ConsumeBool() ? b.leaf(GGML_TYPE_F32, ne0, ne1) : nullptr;
            ggml_tensor * r    = nullptr;
            if (!x || !harness::guarded([&] { r = ggml_soft_max_ext(ctx, x, mask, 0.5f, 0.0f); })) {
                return {};
            }
            out.push_back(r);
            break;
        }
        case 9: {  // ROPE
            static const int64_t HD[] = { 64, 80, 96, 128, 256, 512 };
            const int64_t hd   = HD[fdp.ConsumeIntegralInRange<size_t>(0, 5)];
            const int64_t nh   = fdp.ConsumeIntegralInRange<int64_t>(1, 64);
            const int64_t nt   = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 2048);
            const int     nd   = fdp.ConsumeBool() ? (int) hd : (int) (hd / 4) * 2;
            static const int modes[] = { GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX, GGML_ROPE_TYPE_IMROPE };
            const int     mode = modes[fdp.ConsumeIntegralInRange<size_t>(0, 2)];
            ggml_tensor * x    = activation(hd, nh, nt);
            ggml_tensor * pos  = b.leaf(GGML_TYPE_I32, mode == GGML_ROPE_TYPE_IMROPE ? nt * 4 : nt);
            ggml_tensor * r    = nullptr;
            if (!x || !pos) {
                return {};
            }
            if (mode == GGML_ROPE_TYPE_IMROPE) {
                int sections[4] = { 11, 11, 10, 0 };
                if (!harness::guarded([&] {
                        r = ggml_rope_multi(ctx, x, pos, nullptr, nd, sections, mode, 262144, 10000000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                    })) {
                    return {};
                }
            } else if (!harness::guarded([&] {
                           r = ggml_rope_ext(ctx, x, pos, nullptr, nd, mode, 262144, 10000000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                       })) {
                return {};
            }
            out.push_back(r);
            break;
        }
        case 10: {  // GET_ROWS
            static const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0 };
            const ggml_type t   = types[fdp.ConsumeIntegralInRange<size_t>(0, 2)];
            int64_t         ne0 = pick(fdp, ROW_DIMS, sizeof(ROW_DIMS) / 8, 1, 600000);
            if (t == GGML_TYPE_Q8_0) {
                ne0 = ((ne0 + 31) / 32) * 32;
            }
            const int64_t rows = fdp.ConsumeIntegralInRange<int64_t>(1, 300000);
            const int64_t nidx = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 512);
            ggml_tensor * tab  = b.leaf(t, ne0, rows, 1, 1, fdp.ConsumeBool());
            ggml_tensor * idx  = b.leaf(fdp.ConsumeBool() ? GGML_TYPE_I32 : GGML_TYPE_I64, nidx);
            ggml_tensor * r    = nullptr;
            if (!tab || !idx || !harness::guarded([&] { r = ggml_get_rows(ctx, tab, idx); })) {
                return {};
            }
            out.push_back(r);
            break;
        }
        case 11: {  // SET_ROWS
            static const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0 };
            const ggml_type t   = types[fdp.ConsumeIntegralInRange<size_t>(0, 2)];
            int64_t         ne0 = pick(fdp, ROW_DIMS, sizeof(ROW_DIMS) / 8, 1, 600000);
            if (t == GGML_TYPE_Q8_0) {
                ne0 = ((ne0 + 31) / 32) * 32;
            }
            const int64_t nrows = fdp.ConsumeIntegralInRange<int64_t>(1, 4096);
            const int64_t nset  = fdp.ConsumeIntegralInRange<int64_t>(1, 64);
            ggml_tensor * dst   = b.leaf(t, ne0, nrows);
            ggml_tensor * src   = b.leaf(GGML_TYPE_F32, ne0, nset);
            ggml_tensor * idx   = b.leaf(fdp.ConsumeBool() ? GGML_TYPE_I32 : GGML_TYPE_I64, nset);
            ggml_tensor * r     = nullptr;
            if (!dst || !src || !idx || !harness::guarded([&] { r = ggml_set_rows(ctx, dst, src, idx); })) {
                return {};
            }
            out.push_back(r);
            break;
        }
        case 12: {  // CPY and CONT
            const ggml_type st  = fdp.ConsumeBool() ? GGML_TYPE_F32 : GGML_TYPE_F16;
            const ggml_type dt  = fdp.ConsumeBool() ? GGML_TYPE_F32 : GGML_TYPE_F16;
            const int64_t   ne0 = pick(fdp, ROW_DIMS, sizeof(ROW_DIMS) / 8, 1, 70000);
            const int64_t   ne1 = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 256);
            ggml_tensor *   a   = b.leaf(st, ne0, ne1);
            if (!a) {
                return {};
            }
            ggml_tensor * src = a;
            if (fdp.ConsumeBool()) {
                harness::guarded([&] { src = ggml_transpose(ctx, a); });
            }
            ggml_tensor * r = nullptr;
            if (fdp.ConsumeBool()) {
                ggml_tensor * d = b.leaf(dt, src->ne[0], src->ne[1]);
                if (!d || !harness::guarded([&] { r = ggml_cpy(ctx, src, d); })) {
                    return {};
                }
            } else if (!harness::guarded([&] { r = ggml_cont(ctx, src); })) {
                return {};
            }
            out.push_back(r);
            break;
        }
        case 13: {  // CONCAT on dim 0 with a transposed second input (the conv chain of Qwen3.5)
            const int64_t n_ch = pick(fdp, N_DIMS, sizeof(N_DIMS) / 8, 1, 20000);
            const int64_t T    = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 16384);
            ggml_tensor * st   = b.leaf(GGML_TYPE_F32, 3, n_ch);
            ggml_tensor * p    = b.leaf(GGML_TYPE_F32, n_ch, T);
            ggml_tensor * xt   = nullptr;
            ggml_tensor * r    = nullptr;
            if (!st || !p || !harness::guarded([&] { xt = ggml_transpose(ctx, p); r = ggml_concat(ctx, st, xt, 0); })) {
                return {};
            }
            out.push_back(r);
            break;
        }
        case 14: {  // GATED_DELTA_NET
            const int64_t S_v = fdp.ConsumeIntegralInRange<int64_t>(1, 160);
            const int64_t H   = fdp.ConsumeIntegralInRange<int64_t>(1, 64);
            const int64_t T   = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 1024);
            const int64_t ns  = fdp.ConsumeIntegralInRange<int64_t>(1, 3);
            const int     K   = fdp.ConsumeIntegralInRange<int>(1, 5);
            ggml_tensor * q   = b.leaf(GGML_TYPE_F32, S_v, H, T, ns);
            ggml_tensor * k   = b.leaf(GGML_TYPE_F32, S_v, H, T, ns);
            ggml_tensor * v   = b.leaf(GGML_TYPE_F32, S_v, H, T, ns);
            ggml_tensor * g   = b.leaf(GGML_TYPE_F32, fdp.ConsumeBool() ? 1 : S_v, H, T, ns);
            ggml_tensor * be  = b.leaf(GGML_TYPE_F32, 1, H, T, ns);
            ggml_tensor * s   = b.leaf(GGML_TYPE_F32, S_v, S_v, H, ns);
            ggml_tensor * r   = nullptr;
            if (!q || !k || !v || !g || !be || !s || !harness::guarded([&] { r = ggml_gated_delta_net(ctx, q, k, v, g, be, s, K); })) {
                return {};
            }
            out.push_back(r);
            break;
        }
        case 15: {  // SSM_CONV
            const int64_t d_inner = pick(fdp, N_DIMS, sizeof(N_DIMS) / 8, 1, 20000);
            const int64_t d_conv  = fdp.ConsumeIntegralInRange<int64_t>(2, 8);
            const int64_t T       = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 1024);
            ggml_tensor * sx      = b.leaf(GGML_TYPE_F32, d_conv - 1 + T, d_inner, 1);
            ggml_tensor * c       = b.leaf(GGML_TYPE_F32, d_conv, d_inner);
            ggml_tensor * r       = nullptr;
            if (!sx || !c || !harness::guarded([&] { r = ggml_ssm_conv(ctx, sx, c); })) {
                return {};
            }
            out.push_back(r);
            break;
        }
        default: {  // binary ops with broadcast
            const bool    f16 = harness::rare(fdp, 4);
            const ggml_type t = f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
            const int64_t ne0 = pick(fdp, ROW_DIMS, sizeof(ROW_DIMS) / 8, 1, 300000);
            const int64_t ne1 = pick(fdp, M_DIMS, sizeof(M_DIMS) / 8, 1, 64);
            const int64_t ne2 = fdp.ConsumeIntegralInRange<int64_t>(1, 4);
            ggml_tensor * a   = activation(ne0, ne1, ne2, t);
            const int     bf  = fdp.ConsumeIntegralInRange<int>(0, 3);
            ggml_tensor * bb  = bf == 0 ? b.leaf(t, ne0, ne1, ne2) : bf == 1 ? b.leaf(t, ne0) : bf == 2 ? b.leaf(t, 1) : b.leaf(t, 1, ne1, ne2);
            ggml_tensor * r   = nullptr;
            const int     op  = fdp.ConsumeIntegralInRange<int>(0, 3);
            if (!a || !bb || !harness::guarded([&] {
                    r = op == 0 ? ggml_add(ctx, a, bb) : op == 1 ? ggml_mul(ctx, a, bb) : op == 2 ? ggml_sub(ctx, a, bb) : ggml_div(ctx, a, bb);
                })) {
                return {};
            }
            out.push_back(r);
            break;
        }
    }
    return out;
}

} // namespace

extern "C" int LLVMFuzzerInitialize(int * argc, char *** argv) {
    (void) argc;
    (void) argv;
    harness::init();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
    FuzzedDataProvider fdp(data, size);

    // The session: the hardware of the fake DSP and the switches of the backend
    fakedsp::config cfg;
    static const uint32_t threads[] = { 6, 6, 6, 8, 4, 2, 1, 10, 3, 5 };
    static const uint64_t vtcms[]   = { 8u << 20, 8u << 20, 8u << 20, 8u << 20, 4u << 20, 2u << 20, 1u << 20, 512u << 10, 256u << 10 };
    cfg.n_threads = threads[fdp.ConsumeIntegralInRange<size_t>(0, 9)];
    cfg.n_hmx     = harness::rare(fdp, 6) ? 0 : 1;
    cfg.vtcm_size = vtcms[fdp.ConsumeIntegralInRange<size_t>(0, 8)];
    fakedsp::configure(cfg);

    hexhost::options o;
    static const int fusions[] = { 1, 1, 1, 0, 4, 8, 16, 32, 8 | 16, 2 | 4 | 8 | 16 | 32 };
    o.opbatch    = 64;
    o.opqueue    = 1;
    o.nhmx       = harness::rare(fdp, 6) ? 0 : 1;
    o.mm_select  = fdp.ConsumeIntegralInRange<int>(1, 3);
    o.fa_select  = fdp.ConsumeIntegralInRange<int>(0, 2);
    o.multirow   = fdp.ConsumeBool() ? 1 : 0;
    o.opfusion   = fusions[fdp.ConsumeIntegralInRange<size_t>(0, 9)];
    o.gdn_chunk  = fdp.ConsumeBool() ? 1 : 0;
    o.nhvx       = harness::rare(fdp, 4) ? fdp.ConsumeIntegralInRange<size_t>(1, 12) : 0;
    hexhost::set_options(o);

    hexhost::device * dev = hexhost::device_new();
    if (!hexhost::device_open(dev)) {
        hexhost::device_free(dev);
        return 0;
    }
    {
        builder b(fdp, dev);
        const int scenario = fdp.ConsumeIntegralInRange<int>(0, 16);
        std::vector<ggml_tensor *> outs = build(b, scenario);
        if (!outs.empty()) {
            ggml_cgraph * gf = ggml_new_graph_custom(b.ctx, 128, false);
            for (auto * t : outs) {
                ggml_build_forward_expand(gf, t);
            }
            b.place_nodes(gf);

            bool all = true;
            for (int i = 0; i < ggml_graph_n_nodes(gf) && all; i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (ggml_op_is_empty(n->op)) {
                    continue;
                }
                all = hexhost::supports_op(dev, n);
            }
            if (all) {
                char ctx[64];
                snprintf(ctx, sizeof(ctx), "kparams scenario %d", scenario);
                for (const auto & p : hexhost::pack_graph(dev, gf)) {
                    harness::check_packed(dev, p, ctx);
                }
            }
        }
    }
    hexhost::device_free(dev);
    return 0;
}
