// The graph generator of the hexhost graph fuzzer. Refer to graphgen.h.
//
// The layer builders follow llama.cpp: build_layer_attn_linear of
// src/models/qwen35.cpp, build_conv_state and build_recurrent_attn of
// src/models/delta-net-base.cpp, build_rs of src/llama-graph.cpp, and
// build_layer_attn of qwen35.cpp with the KV cache writes of llama-kv-cache.

#include "graphgen.h"

#include "fake_dsp.h"
#include "harness.h"
#include "hexhost.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace graphgen {

namespace {

const ggml_type WTYPES[] = { GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1,
                             GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
                             GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 };

// Gives a weight type from the code that fits a row of k elements: the type of the code, else
// Q8_0 (blocks of 32), else F16.
ggml_type wtype_for(int code, int64_t k) {
    const ggml_type t = WTYPES[(unsigned) code % (sizeof(WTYPES) / sizeof(WTYPES[0]))];
    const int64_t   b = (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K) ? 256 : ggml_blck_size(t);
    if (k % b == 0) {
        return t;
    }
    return (k % 32 == 0) ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
}

template <typename T, size_t N> T pick(FuzzedDataProvider & fdp, const T (&list)[N]) {
    return list[fdp.ConsumeIntegralInRange<size_t>(0, N - 1)];
}

// Picks the dims and the mutations of the model.
void pick_params(FuzzedDataProvider & fdp, model_params & mp) {
    static const int64_t EMBD[] = { 32, 64, 96, 128, 256, 64 };
    static const int64_t FF[]   = { 64, 128, 256, 512 };
    static const int64_t SV[]   = { 8, 16, 32, 16 };
    static const int64_t HD[]   = { 32, 64 };
    static const int64_t VOC[]  = { 64, 256, 1000, 2048 };
    mp.n_embd  = pick(fdp, EMBD);
    mp.n_ff    = pick(fdp, FF);
    mp.S_v     = pick(fdp, SV);
    mp.H_k     = fdp.ConsumeIntegralInRange<int64_t>(1, 2);
    mp.H_v     = mp.H_k * fdp.ConsumeIntegralInRange<int64_t>(1, 2);
    mp.mem     = fdp.ConsumeIntegralInRange<int64_t>(1, 3);
    mp.K       = harness::rare(fdp, 3) ? fdp.ConsumeIntegralInRange<int64_t>(2, 4) : 1;
    mp.hd      = pick(fdp, HD);
    mp.n_kv    = fdp.ConsumeIntegralInRange<int64_t>(1, 2);
    mp.n_head  = mp.n_kv * fdp.ConsumeIntegralInRange<int64_t>(1, 2);
    // At least the largest token count (33), thus the rows of one SET_ROWS are distinct
    mp.kv_size = fdp.ConsumeBool() ? 64 : 128;
    mp.vocab   = pick(fdp, VOC);
    mp.n_layers = fdp.ConsumeIntegralInRange<int>(1, 3);
    for (int il = 0; il < mp.n_layers; il++) {
        layer_params & lp = mp.layers[il];
        const int      k  = fdp.ConsumeIntegralInRange<int>(0, 9);
        lp.kind           = k < 5 ? 0 : k < 8 ? 1 : 2;
        lp.d_conv         = harness::rare(fdp, 10) ? 3 : 4;
        lp.zero_state     = harness::rare(fdp, 6);
        lp.extra_rows     = harness::rare(fdp, 8);
        lp.reader         = harness::rare(fdp, 6) ? fdp.ConsumeIntegralInRange<int>(0, 5) : -1;
        lp.out_flag       = harness::rare(fdp, 10) ? fdp.ConsumeIntegralInRange<int>(0, 3) : -1;
        // A computed index is one form of the known finding gdn-slot-stale (the host reads the
        // index before the DSP computes it). HEXHOST_IGNORE=gdn-slot-stale keeps it off.
        lp.computed_idx   = harness::rare(fdp, 10) && !fakedsp::is_ignored("gdn-slot-stale");
        lp.slot_mode      = harness::rare(fdp, 8) ? fdp.ConsumeIntegralInRange<int>(1, 2) : 0;
        lp.idx_offset     = harness::rare(fdp, 8) ? 1 : 0;
        lp.swiglu         = fdp.ConsumeBool();
        lp.view_add       = harness::rare(fdp, 6);
        lp.view_reader    = lp.view_add && fdp.ConsumeBool();
        lp.keep_qkv       = !harness::rare(fdp, 3);
        for (int j = 0; j < 8; j++) {
            lp.wtype[j] = fdp.ConsumeIntegralInRange<int>(0, 11);
        }
    }
    mp.mtp        = harness::rare(fdp, 4);
    mp.out_ids    = harness::rare(fdp, 4);
    mp.head_wtype = fdp.ConsumeIntegralInRange<int>(0, 11);
}

// Fills the bytes of one weight with values that give finite results of a moderate size (the
// numeric mode): a vector (a norm or a bias) gets values near 1, a matrix gets values in
// [-1, 1] / sqrt(k), and a quantized type gets these values through ggml_quantize_chunk.
// O(number of elements).
void numeric_weight(ggml_tensor * t, std::mt19937 & rng, std::vector<uint8_t> & bytes) {
    const int64_t      n     = ggml_nelements(t);
    const bool         vec   = t->ne[1] == 1;
    const float        scale = vec ? 0.25f : 1.0f / sqrtf((float) t->ne[0]);
    // ssm_a is -exp(A_log) in the model: a negative gate makes the state decay
    const float        sign  = strcmp(ggml_get_name(t), "ssm_a") == 0 ? -1.0f : 1.0f;
    std::vector<float> f(n);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto & x : f) {
        x = vec ? sign * (1.0f + scale * u(rng)) : scale * u(rng);
    }
    if (t->type == GGML_TYPE_F32) {
        memcpy(bytes.data(), f.data(), bytes.size());
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) bytes.data(), n);
    } else {
        ggml_quantize_chunk(t->type, f.data(), bytes.data(), 0, n / t->ne[0], t->ne[0], nullptr);
    }
}

// Creates the weights and the caches, allocates them, and fills the weights.
bool make_tensors(FuzzedDataProvider & fdp, world & w) {
    const model_params & mp = w.mp;
    ggml_init_params     p  = { ggml_tensor_overhead() * 256, nullptr, true };
    w.ctx_w                 = ggml_init(p);
    w.ctx_c                 = ggml_init(p);

    auto wt = [&](ggml_type t, int64_t k, int64_t n) {
        ggml_tensor * x = ggml_new_tensor_2d(w.ctx_w, t, k, n);
        w.weights.push_back(x);
        return x;
    };
    auto w1 = [&](int64_t n) {
        ggml_tensor * x = ggml_new_tensor_1d(w.ctx_w, GGML_TYPE_F32, n);
        w.weights.push_back(x);
        return x;
    };

    w.lt.resize(mp.n_layers);
    for (int il = 0; il < mp.n_layers; il++) {
        const layer_params & lp = mp.layers[il];
        layer_tensors &      lt = w.lt[il];
        lt.attn_norm            = w1(mp.n_embd);
        if (lp.kind == 0) {
            const int64_t n_ch = 2 * mp.S_v * mp.H_k + mp.S_v * mp.H_v;
            const int64_t row  = (lp.d_conv - 1) * n_ch;
            const int64_t D    = mp.S_v * mp.S_v * mp.H_v;
            lt.wqkv     = wt(wtype_for(lp.wtype[0], mp.n_embd), mp.n_embd, n_ch);
            lt.wz       = wt(wtype_for(lp.wtype[1], mp.n_embd), mp.n_embd, mp.S_v * mp.H_v);
            lt.wbeta    = wt(wtype_for(lp.wtype[2], mp.n_embd), mp.n_embd, mp.H_v);
            lt.walpha   = wt(wtype_for(lp.wtype[3], mp.n_embd), mp.n_embd, mp.H_v);
            lt.dt       = w1(mp.H_v);
            lt.ssm_a    = w1(mp.H_v);
            ggml_set_name(lt.ssm_a, "ssm_a");
            lt.conv     = ggml_new_tensor_2d(w.ctx_w, GGML_TYPE_F32, lp.d_conv, n_ch);
            w.weights.push_back(lt.conv);
            lt.ssm_norm = w1(mp.S_v);
            lt.ssm_out  = wt(wtype_for(lp.wtype[4], mp.S_v * mp.H_v), mp.S_v * mp.H_v, mp.n_embd);
            lt.conv_cache = ggml_new_tensor_2d(w.ctx_c, GGML_TYPE_F32, row, mp.mem * mp.K);
            lt.ssm_cache  = ggml_new_tensor_2d(w.ctx_c, GGML_TYPE_F32, D, mp.mem * mp.K);
            ggml_set_name(lt.conv_cache, "conv_cache");
            ggml_set_name(lt.ssm_cache, "ssm_cache");
            w.caches.push_back(lt.conv_cache);
            w.caches.push_back(lt.ssm_cache);
        } else if (lp.kind == 1) {
            lt.wq     = wt(wtype_for(lp.wtype[0], mp.n_embd), mp.n_embd, mp.hd * 2 * mp.n_head);
            lt.wk     = wt(wtype_for(lp.wtype[1], mp.n_embd), mp.n_embd, mp.hd * mp.n_kv);
            lt.wv     = wt(wtype_for(lp.wtype[2], mp.n_embd), mp.n_embd, mp.hd * mp.n_kv);
            lt.wo     = wt(wtype_for(lp.wtype[3], mp.hd * mp.n_head), mp.hd * mp.n_head, mp.n_embd);
            lt.q_norm = w1(mp.hd);
            lt.k_norm = w1(mp.hd);
            lt.k_cache = ggml_new_tensor_2d(w.ctx_c, GGML_TYPE_F16, mp.hd * mp.n_kv, mp.kv_size);
            lt.v_cache = ggml_new_tensor_2d(w.ctx_c, GGML_TYPE_F16, mp.hd * mp.n_kv, mp.kv_size);
            ggml_set_name(lt.k_cache, "k_cache");
            ggml_set_name(lt.v_cache, "v_cache");
            w.caches.push_back(lt.k_cache);
            w.caches.push_back(lt.v_cache);
        }
        lt.ffn_norm = w1(mp.n_embd);
        lt.up       = wt(wtype_for(lp.wtype[5], mp.n_embd), mp.n_embd, mp.n_ff);
        lt.gate     = wt(wtype_for(lp.wtype[6], mp.n_embd), mp.n_embd, mp.n_ff);
        lt.down     = wt(wtype_for(lp.wtype[7], mp.n_ff), mp.n_ff, mp.n_embd);
    }
    w.out_norm = w1(mp.n_embd);
    w.w_out    = wt(wtype_for(mp.head_wtype, mp.n_embd), mp.n_embd, mp.vocab);
    if (mp.mtp) {
        w.mtp_proj = wt(wtype_for(mp.head_wtype + 1, 2 * mp.n_embd), 2 * mp.n_embd, mp.n_embd);
        w.mtp_norm = w1(mp.n_embd);
    }

    ggml_backend_buffer_type_t buft = hexhost::device_buft(w.dev);
    w.buf_w = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx_w, buft);
    w.buf_c = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx_c, buft);
    if (!w.buf_w || !w.buf_c) {
        return false;
    }
    ggml_backend_buffer_set_usage(w.buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // The weights: random bytes (or random values in the numeric mode), set in one piece or in two
    // pieces (the shadow buffer of the repack)
    std::mt19937 rng(fdp.ConsumeIntegral<uint32_t>());
    for (ggml_tensor * t : w.weights) {
        const size_t         n = ggml_nbytes(t);
        std::vector<uint8_t> bytes(n);
        if (w.numeric) {
            numeric_weight(t, rng, bytes);
        } else {
            for (auto & b : bytes) {
                b = (uint8_t) rng();
            }
            if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) {
                for (size_t i = 0; i < n; i += ggml_type_size(t->type)) {
                    bytes[i + ggml_type_size(t->type) - 1] &= 0x3f;  // finite values
                }
            }
        }
        const bool split = ggml_is_quantized(t->type) && t->type != GGML_TYPE_Q6_K && t->type != GGML_TYPE_Q4_K &&
                           harness::rare(fdp, 6);
        if (split) {
            const size_t row  = ggml_row_size(t->type, t->ne[0]);
            const size_t half = (t->ne[1] / 2) * row;
            ggml_backend_tensor_set(t, bytes.data() + half, half, n - half);
            ggml_backend_tensor_set(t, bytes.data(), 0, half);
        } else {
            ggml_backend_tensor_set(t, bytes.data(), 0, n);
        }
    }
    return true;
}

// The state of one graph build
struct gbuild {
    world &              w;
    graph_spec &         g;
    FuzzedDataProvider & fdp;
    ggml_context *       ctx;
    ggml_cgraph *        gf;
    int64_t              T;
    ggml_tensor *        s_copy  = nullptr;
    ggml_tensor *        pos     = nullptr;
    ggml_tensor *        mask    = nullptr;
    ggml_tensor *        k_idxs  = nullptr;
    int64_t              rs_head = 0;
    std::vector<ggml_tensor *> late;   // nodes to expand at the end (extra readers)

    // A new input tensor of the graph
    ggml_tensor * input(ggml_type type, int64_t n0, int64_t n1, input_kind kind, int32_t limit) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, type, n0, n1);
        ggml_set_input(t);
        input_spec s;
        s.t     = t;
        s.kind  = kind;
        s.limit = limit;
        g.inputs.push_back(s);
        return t;
    }

    // Marks a tensor as a graph output
    void output(ggml_tensor * t) {
        ggml_set_output(t);
        g.outputs.push_back(t);
    }

    // RMS_NORM and MUL by a weight row, the fusion candidate of the host
    ggml_tensor * norm(ggml_tensor * x, ggml_tensor * wn) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), wn);
    }

    // build_rs of llama-graph.cpp for one state table
    ggml_tensor * build_rs(const layer_params & lp, ggml_tensor * cache, int64_t size, ggml_tensor * main_idx,
                           ggml_tensor * extra_idx) {
        const int64_t rows   = cache->ne[1];
        ggml_tensor * states = ggml_reshape_2d(ctx, cache, size, rows);
        const int64_t rs_zero = lp.zero_state ? (rs_head % rows) : -1;
        ggml_tensor * sz = ggml_view_1d(ctx, states, size * (rs_zero >= 0), rs_zero >= 0 ? rs_zero * states->nb[1] : 0);
        ggml_build_forward_expand(gf, ggml_scale_inplace(ctx, sz, 0.0f));
        ggml_tensor * R = ggml_get_rows(ctx, states, main_idx);
        ggml_build_forward_expand(gf, R);
        const int64_t n_extra = extra_idx->ne[0];
        ggml_tensor * se = ggml_get_rows(ctx, states, extra_idx);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, se, ggml_view_2d(ctx, cache, size, n_extra, cache->nb[1], (rs_head + 1) * cache->nb[1])));
        return R;
    }

    // An extra reader of an intermediate, observable as a graph output
    void reader(ggml_tensor * t) {
        ggml_tensor * r = ggml_scale(ctx, t, 2.0f);
        output(r);
        late.push_back(r);
    }

    // build_layer_attn_linear of qwen35.cpp
    ggml_tensor * linear(int il, ggml_tensor * h) {
        const model_params & mp = w.mp;
        const layer_params & lp = mp.layers[il];
        const layer_tensors & lt = w.lt[il];
        const int64_t S_v = mp.S_v, H_k = mp.H_k, H_v = mp.H_v;
        const int64_t n_ch = 2 * S_v * H_k + S_v * H_v;
        const int64_t row  = (lp.d_conv - 1) * n_ch;
        const int64_t D    = S_v * S_v * H_v;
        const int64_t rows = mp.mem * mp.K;
        // The extra state needs a second row. rs_head + n_rs <= rows, thus each view is inside its cache
        const int64_t n_rs = (lp.extra_rows && rows >= 2) ? 2 : 1;
        rs_head            = std::max<int64_t>(0, std::min<int64_t>(rs_head, rows - n_rs));

        ggml_tensor * qkv  = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, lt.wqkv, h), n_ch, T, 1);
        if (lp.keep_qkv) {
            reader(qkv);
        }
        ggml_tensor * z    = ggml_mul_mat(ctx, lt.wz, h);
        ggml_tensor * beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, ggml_mul_mat(ctx, lt.wbeta, h), 1, H_v, T, 1));
        ggml_tensor * alpha = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, lt.walpha, h), H_v, T, 1);
        ggml_tensor * gate  = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, lt.dt)), lt.ssm_a);
        gate                = ggml_reshape_4d(ctx, gate, 1, H_v, T, 1);

        // the slot index: an input, or the result of an ARGSORT (a computed index)
        ggml_tensor * main_idx;
        ggml_tensor * extra_idx;
        if (lp.computed_idx) {
            ggml_tensor * keys = input(GGML_TYPE_F32, rows, 1, INPUT_FLOAT, 0);
            ggml_tensor * ord  = ggml_argsort(ctx, keys, GGML_SORT_ORDER_ASC);
            main_idx           = ggml_view_1d(ctx, ord, 1, 0);
            extra_idx          = ggml_view_1d(ctx, ord, n_rs - 1, ord->nb[0]);
        } else {
            if (!s_copy) {
                // three elements: an offset of one and two rows at most
                s_copy = input(GGML_TYPE_I32, 3, 1, INPUT_SLOT, (int32_t) rows);
                // The other form of gdn-slot-stale: ggml-alloc gives the s_copy bytes to a later
                // tensor before the fused op reads them. An output flag keeps the bytes.
                if (fakedsp::is_ignored("gdn-slot-stale")) {
                    ggml_set_output(s_copy);
                }
            }
            main_idx  = ggml_view_1d(ctx, s_copy, 1, lp.idx_offset * s_copy->nb[0]);
            extra_idx = ggml_view_1d(ctx, s_copy, std::min<int64_t>(n_rs - 1, s_copy->ne[0] - 1 - lp.idx_offset),
                                     (lp.idx_offset + 1) * s_copy->nb[0]);
        }

        // build_conv_state of delta-net-base.cpp
        ggml_tensor * R  = build_rs(lp, lt.conv_cache, row, main_idx, extra_idx);
        ggml_tensor * cs = ggml_reshape_3d(ctx, R, lp.d_conv - 1, n_ch, 1);
        ggml_tensor * xt = ggml_transpose(ctx, qkv);
        ggml_tensor * CI = ggml_concat(ctx, cs, xt, 0);
        const size_t  rsz = ggml_row_size(GGML_TYPE_F32, row);
        auto slot_off = [&](int64_t s_slot) -> size_t {
            size_t off = (size_t) (s_slot * mp.mem + rs_head) * rsz;
            if (lp.slot_mode == 2 && s_slot > 0) {
                off -= (rsz / 2) & ~(size_t) 3;   // two slots overlap
            }
            return off;
        };
        if (mp.K == 1) {
            ggml_tensor * last = ggml_view_3d(ctx, CI, lp.d_conv - 1, n_ch, 1, CI->nb[1], CI->nb[2], ggml_row_size(CI->type, T));
            ggml_tensor * upd  = ggml_view_2d(ctx, lt.conv_cache, row, 1, lt.conv_cache->nb[1], slot_off(0));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, last, upd));
        } else {
            for (int64_t t = 1; t <= mp.K; ++t) {
                const int64_t s_idx  = std::max<int64_t>(0, T - mp.K + t);
                int64_t       s_slot = mp.K - t;
                if (lp.slot_mode == 1 && s_slot == mp.K - 1) {
                    s_slot = mp.K - 2;   // two CPYs write one slot
                }
                ggml_tensor * last = ggml_view_3d(ctx, CI, lp.d_conv - 1, n_ch, 1, CI->nb[1], CI->nb[2], ggml_row_size(CI->type, s_idx));
                ggml_tensor * upd  = ggml_view_2d(ctx, lt.conv_cache, row, 1, lt.conv_cache->nb[1], slot_off(s_slot));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, last, upd));
            }
        }
        ggml_tensor * C = ggml_ssm_conv(ctx, CI, lt.conv);
        ggml_tensor * S = ggml_silu(ctx, C);

        const size_t  nb1_qkv = ggml_row_size(S->type, n_ch);
        ggml_tensor * q = ggml_view_4d(ctx, S, S_v, H_k, T, 1, ggml_row_size(S->type, S_v), nb1_qkv, nb1_qkv * T, 0);
        ggml_tensor * k = ggml_view_4d(ctx, S, S_v, H_k, T, 1, ggml_row_size(S->type, S_v), nb1_qkv, nb1_qkv * T,
                                       S_v * H_k * ggml_element_size(S));
        ggml_tensor * v = ggml_view_4d(ctx, S, S_v, H_v, T, 1, ggml_row_size(S->type, S_v), nb1_qkv, nb1_qkv * T,
                                       ggml_row_size(S->type, 2 * S_v * H_k));
        q = ggml_l2_norm(ctx, q, 1e-6f);
        k = ggml_l2_norm(ctx, k, 1e-6f);
        if (H_k != H_v) {
            q = ggml_repeat_4d(ctx, q, S_v, H_v, T, 1);
            k = ggml_repeat_4d(ctx, k, S_v, H_v, T, 1);
        }

        ggml_tensor * R2 = build_rs(lp, lt.ssm_cache, D, main_idx, extra_idx);
        ggml_tensor * st = ggml_reshape_4d(ctx, R2, S_v, S_v, H_v, 1);

        ggml_tensor * G;
        ggml_tensor * out;
        const size_t  head_bytes = ggml_row_size(GGML_TYPE_F32, S_v * H_v * T);
        if (mp.K == 1) {
            G   = ggml_gated_delta_net(ctx, q, k, v, gate, beta, st, 1);
            out = ggml_view_4d(ctx, G, S_v, H_v, T, 1, ggml_row_size(G->type, S_v), ggml_row_size(G->type, S_v * H_v),
                               ggml_row_size(G->type, S_v * H_v * T), 0);
            ggml_tensor * ns = ggml_view_4d(ctx, G, S_v, S_v, H_v, 1, ggml_row_size(G->type, S_v), ggml_row_size(G->type, S_v * S_v),
                                            ggml_row_size(G->type, D), head_bytes);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ns, ggml_view_2d(ctx, lt.ssm_cache, D, 1, lt.ssm_cache->nb[1],
                                                                          rs_head * ggml_row_size(GGML_TYPE_F32, D))));
        } else {
            G   = ggml_gated_delta_net(ctx, q, k, v, gate, beta, st, (int) mp.K);
            out = ggml_view_4d(ctx, G, S_v, H_v, T, 1, ggml_row_size(G->type, S_v), ggml_row_size(G->type, S_v * H_v),
                               ggml_row_size(G->type, S_v * H_v * T), 0);
            const int64_t nw  = std::min<int64_t>(T, mp.K);
            ggml_tensor * src = ggml_view_3d(ctx, G, D, 1, nw, ggml_row_size(G->type, D), ggml_row_size(G->type, D), head_bytes);
            ggml_tensor * dst = ggml_view_3d(ctx, lt.ssm_cache, D, 1, nw, lt.ssm_cache->nb[1],
                                             (size_t) mp.mem * ggml_row_size(GGML_TYPE_F32, D), rs_head * ggml_row_size(GGML_TYPE_F32, D));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
        }

        // the mutations of the chains
        switch (lp.reader) {
            case 0: reader(R); break;
            case 1: reader(CI); break;
            case 2: reader(C); break;
            case 3: reader(ggml_view_1d(ctx, G, S_v, head_bytes)); break;
            case 4: reader(out); break;
            case 5: reader(R2); break;
            default: break;
        }
        switch (lp.out_flag) {
            case 0: output(R); break;
            case 1: output(CI); break;
            case 2: output(C); break;
            case 3:
                // With K > 1 the op leaves the snapshot slots 1 to K - 1 of G unwritten, thus the
                // numeric compare of the phone does not take G as an output.
                if (!(w.numeric && mp.K > 1)) {
                    output(G);
                }
                break;
            default: break;
        }

        ggml_tensor * z2 = ggml_reshape_4d(ctx, z, S_v, H_v, T, 1);
        ggml_tensor * on = ggml_mul(ctx, norm(out, lt.ssm_norm), ggml_silu(ctx, z2));
        ggml_tensor * fo = ggml_reshape_3d(ctx, on, S_v * H_v, T, 1);
        return ggml_reshape_2d(ctx, ggml_mul_mat(ctx, lt.ssm_out, fo), mp.n_embd, T);
    }

    // build_layer_attn of qwen35.cpp with the KV cache writes and FLASH_ATTN_EXT
    ggml_tensor * attention(int il, ggml_tensor * h) {
        const model_params &  mp = w.mp;
        const layer_tensors & lt = w.lt[il];
        const int64_t hd = mp.hd, nh = mp.n_head, nkv = mp.n_kv;
        if (!pos) {
            pos = input(GGML_TYPE_I32, T, 1, INPUT_POS, 0);
        }
        if (!k_idxs) {
            k_idxs = input(harness::rare(fdp, 4) ? GGML_TYPE_I64 : GGML_TYPE_I32, T, 1, INPUT_ROWS,
                           (int32_t) mp.kv_size);
        }
        if (!mask) {
            mask = input(GGML_TYPE_F16, mp.kv_size, T, INPUT_MASK, 0);
        }
        ggml_tensor * qf = ggml_mul_mat(ctx, lt.wq, h);
        ggml_tensor * kc = ggml_mul_mat(ctx, lt.wk, h);
        ggml_tensor * vc = ggml_mul_mat(ctx, lt.wv, h);
        ggml_tensor * Q  = ggml_view_3d(ctx, qf, hd, nh, T, ggml_element_size(qf) * hd * 2, ggml_element_size(qf) * hd * 2 * nh, 0);
        Q                = norm(Q, lt.q_norm);
        ggml_tensor * K  = norm(ggml_reshape_3d(ctx, kc, hd, nkv, T), lt.k_norm);
        ggml_tensor * gt = ggml_view_3d(ctx, qf, hd, nh, T, ggml_element_size(qf) * hd * 2, ggml_element_size(qf) * hd * 2 * nh,
                                        ggml_element_size(qf) * hd);
        gt               = ggml_cont_2d(ctx, gt, hd * nh, T);
        ggml_tensor * V  = ggml_reshape_3d(ctx, vc, hd, nkv, T);
        Q = ggml_rope_ext(ctx, Q, pos, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, pos, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_build_forward_expand(gf, ggml_set_rows(ctx, lt.k_cache, ggml_reshape_2d(ctx, K, hd * nkv, T), k_idxs));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, lt.v_cache, ggml_reshape_2d(ctx, V, hd * nkv, T), k_idxs));

        ggml_tensor * kv = ggml_view_3d(ctx, lt.k_cache, hd, mp.kv_size, nkv, lt.k_cache->nb[1], ggml_row_size(lt.k_cache->type, hd), 0);
        ggml_tensor * vv = ggml_view_3d(ctx, lt.v_cache, hd, mp.kv_size, nkv, lt.v_cache->nb[1], ggml_row_size(lt.v_cache->type, hd), 0);
        ggml_tensor * q  = ggml_permute(ctx, Q, 0, 2, 1, 3);
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, kv, vv, mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
        ggml_tensor * a2 = ggml_reshape_2d(ctx, fa, hd * nh, T);
        ggml_tensor * at = ggml_mul(ctx, a2, ggml_sigmoid(ctx, gt));
        return ggml_mul_mat(ctx, lt.wo, at);
    }

    // build_layer_ffn of qwen35.cpp with the residual ADD
    ggml_tensor * ffn(int il, ggml_tensor * cur) {
        const layer_params &  lp = w.mp.layers[il];
        const layer_tensors & lt = w.lt[il];
        ggml_tensor *         h  = norm(cur, lt.ffn_norm);
        ggml_tensor *         g  = ggml_mul_mat(ctx, lt.gate, h);
        ggml_tensor *         u  = ggml_mul_mat(ctx, lt.up, h);
        ggml_tensor *         a  = lp.swiglu ? ggml_swiglu_split(ctx, g, u) : ggml_mul(ctx, ggml_silu(ctx, g), u);
        ggml_tensor *         d  = ggml_mul_mat(ctx, lt.down, a);
        if (lp.view_add) {
            ggml_tensor * vd  = ggml_view_2d(ctx, d, d->ne[0], d->ne[1], d->nb[1], 0);
            ggml_tensor * res = ggml_add(ctx, vd, cur);
            if (lp.view_reader) {
                reader(vd);
            }
            return res;
        }
        return ggml_add(ctx, d, cur);
    }
};

// Builds the nodes, the node order, the splits and the allocation of one graph.
bool build_graph(FuzzedDataProvider & fdp, world & w, graph_spec & g, int64_t T, bool head_view_add) {
    const model_params & mp = w.mp;
    ggml_init_params     p  = { ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(2048, false), nullptr, true };
    g.ctx                   = ggml_init(p);
    g.gf                    = ggml_new_graph_custom(g.ctx, 2048, false);

    gbuild b { w, g, fdp, g.ctx, g.gf, T };
    b.rs_head = fdp.ConsumeIntegralInRange<int64_t>(0, mp.mem - 1);

    bool          ok     = true;
    ggml_tensor * logits = nullptr;
    ok = harness::guarded([&] {
        ggml_tensor * x   = b.input(GGML_TYPE_F32, mp.n_embd, T, INPUT_FLOAT, 0);
        ggml_tensor * cur = x;
        for (int il = 0; il < mp.n_layers; il++) {
            const int kind = mp.layers[il].kind;
            if (kind == 0) {
                cur = ggml_add(g.ctx, cur, b.linear(il, b.norm(cur, w.lt[il].attn_norm)));
            } else if (kind == 1) {
                cur = ggml_add(g.ctx, cur, b.attention(il, b.norm(cur, w.lt[il].attn_norm)));
            }
            cur = b.ffn(il, cur);
        }
        if (mp.out_ids) {
            ggml_tensor * ids = b.input(GGML_TYPE_I32, 1 + (T > 1 ? (T - 1) / 2 : 0), 1, INPUT_IDS, (int32_t) T);
            cur               = ggml_get_rows(g.ctx, cur, ids);
        }
        ggml_tensor * h = b.norm(cur, w.out_norm);
        if (mp.mtp) {
            ggml_tensor * e   = b.input(GGML_TYPE_F32, mp.n_embd, h->ne[1], INPUT_FLOAT, 0);
            ggml_tensor * cat = ggml_concat(g.ctx, b.norm(e, w.mtp_norm), h, 0);
            h                 = ggml_mul_mat(g.ctx, w.mtp_proj, cat);
        }
        logits = ggml_mul_mat(g.ctx, w.w_out, h);
        if (head_view_add) {
            // the ADD reads the product through a view at offset 0 with fewer columns
            ggml_tensor * bias = b.input(GGML_TYPE_F32, logits->ne[0] / 2, 1, INPUT_FLOAT, 0);
            logits = ggml_add(g.ctx, ggml_view_2d(g.ctx, logits, logits->ne[0] / 2, logits->ne[1], logits->nb[1], 0), bias);
        }
        b.output(logits);
        ggml_build_forward_expand(g.gf, logits);
        for (ggml_tensor * t : b.late) {
            ggml_build_forward_expand(g.gf, t);
        }
    });
    if (!ok) {
        fakedsp::count("gen fail: ggml abort");
        return false;
    }

    // The node order that ggml built, and the check that the device runs every compute node
    const int n = ggml_graph_n_nodes(g.gf);
    for (int i = 0; i < n; i++) {
        ggml_tensor * t = ggml_graph_node(g.gf, i);
        g.order.push_back(t);
        if (ggml_op_is_empty(t->op) || ggml_is_empty(t)) {
            continue;
        }
        if (!hexhost::supports_op(w.dev, t)) {
            fakedsp::count(std::string("gen fail: unsupported ") + ggml_op_desc(t));
            return false;
        }
    }

    // The splits and their uids, as ggml_backend_sched_split_graph makes them
    g.cuts.push_back(0);
    const int n_splits = harness::rare(fdp, 4) ? fdp.ConsumeIntegralInRange<int>(2, 3) : 1;
    for (int s = 1; s < n_splits && g.cuts.back() + 1 <= n - 1; s++) {
        const int c = fdp.ConsumeIntegralInRange<int>(g.cuts.back() + 1, n - 1);
        if (c > g.cuts.back() && c < n) {
            g.cuts.push_back(c);
        }
    }
    g.cuts.push_back(n);
    for (size_t s = 0; s + 1 < g.cuts.size(); s++) {
        g.uids.push_back(ggml_graph_next_uid());
    }

    // graph_optimize of each split before the allocation, as the scheduler does.
    // A backend with no graph_optimize (the CPU backend) keeps the order.
    if (fdp.ConsumeIntegralInRange<int>(0, 3) != 0 && w.backend->iface.graph_optimize) {
        for (size_t s = 0; s + 1 < g.cuts.size(); s++) {
            ggml_cgraph view = ggml_graph_view(g.gf, g.cuts[s], g.cuts[s + 1]);
            w.backend->iface.graph_optimize(w.backend, &view, nullptr);
        }
        g.optimized = true;
    }

    if (w.no_reuse) {
        g.buf = ggml_backend_alloc_ctx_tensors_from_buft(g.ctx, hexhost::device_buft(w.dev));
        if (!g.buf) {
            fakedsp::count("gen fail: allocation");
            return false;
        }
        ggml_backend_buffer_set_usage(g.buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_buffer_clear(g.buf, 0);
        // A view of a cache has its data at creation but no buffer; ggml-alloc gives it one
        for (ggml_tensor * t = ggml_get_first_tensor(g.ctx); t; t = ggml_get_next_tensor(g.ctx, t)) {
            if (t->view_src && !t->buffer && t->view_src->buffer) {
                ggml_backend_view_init(t);
            }
        }
    } else {
        g.galloc = ggml_gallocr_new(hexhost::device_buft(w.dev));
        if (!ggml_gallocr_alloc_graph(g.galloc, g.gf)) {
            fakedsp::count("gen fail: allocation");
            return false;
        }
    }
    fakedsp::count("gen ok");
    g.desc ="graph T=" + std::to_string(T) + " nodes=" + std::to_string(n) + " splits=" + std::to_string(g.cuts.size() - 1);
    return true;
}

// Frees the allocation and the context of one graph.
void free_graph(graph_spec & g) {
    if (g.galloc) {
        ggml_gallocr_free(g.galloc);
    }
    if (g.buf) {
        ggml_backend_buffer_free(g.buf);
    }
    if (g.ctx) {
        ggml_free(g.ctx);
    }
    g = graph_spec();
}

int64_t pick_tokens(FuzzedDataProvider & fdp) {
    static const int64_t TOK[] = { 1, 1, 1, 1, 1, 2, 3, 4, 5, 8, 33 };
    return pick(fdp, TOK);
}

// Sets the values of every input of a graph and gives each value to the hook. Gives true when
// every slot index is valid. In the numeric mode each slot index stays valid (a real device and
// the CPU backend read outside the table for a slot that is not valid), and the input bytes
// that the generator reads do not change. O(total input elements).
bool set_inputs(FuzzedDataProvider & fdp, graph_spec & g, uint32_t step, std::mt19937 & rng, const step_hooks & h,
                bool numeric) {
    bool all_valid = true;
    auto give      = [&](ggml_tensor * t, const std::vector<int32_t> & v) {
        if (h.on_input) {
            h.on_input(t, v, step);
        }
    };
    for (const auto & in : g.inputs) {
        ggml_tensor * t = in.t;
        const int64_t n = ggml_nelements(t);
        switch (in.kind) {
            case INPUT_SLOT:
            case INPUT_ROWS:
            case INPUT_IDS: {
                std::vector<int32_t> v(n);
                const bool invalid = in.kind == INPUT_SLOT && harness::rare(fdp, 8);
                for (int64_t i = 0; i < n; i++) {
                    if (in.kind == INPUT_ROWS) {
                        v[i] = (int32_t) ((step * n + i) % in.limit);   // distinct rows of the KV cache
                    } else {
                        v[i] = in.limit > 0 ? (int32_t) fdp.ConsumeIntegralInRange<int64_t>(0, in.limit - 1) : 0;
                    }
                }
                if (invalid && n > 0) {
                    const int32_t bad = fdp.ConsumeBool() ? -1 : in.limit;
                    if (!numeric) {
                        v[0]      = bad;
                        all_valid = false;
                    }
                }
                if (t->type == GGML_TYPE_I32) {
                    ggml_backend_tensor_set(t, v.data(), 0, n * sizeof(int32_t));
                    give(t, v);
                } else {
                    std::vector<int64_t> v64(v.begin(), v.end());
                    ggml_backend_tensor_set(t, v64.data(), 0, n * sizeof(int64_t));
                    give(t, {});
                }
                break;
            }
            case INPUT_POS: {
                std::vector<int32_t> v(n);
                for (int64_t i = 0; i < n; i++) {
                    v[i] = (int32_t) (step * n + i);
                }
                ggml_backend_tensor_set(t, v.data(), 0, n * sizeof(int32_t));
                give(t, v);
                break;
            }
            case INPUT_MASK: {
                std::vector<ggml_fp16_t> v(n);
                for (int64_t i = 0; i < n; i++) {
                    v[i] = ggml_fp32_to_fp16((rng() & 3) ? 0.0f : -INFINITY);
                }
                // the row of the newest position stays visible, thus no row of the mask is all -inf
                for (int64_t r = 0; r < t->ne[1]; r++) {
                    v[r * t->ne[0] + (step * t->ne[1] + r) % t->ne[0]] = ggml_fp32_to_fp16(0.0f);
                }
                ggml_backend_tensor_set(t, v.data(), 0, n * sizeof(ggml_fp16_t));
                give(t, {});
                break;
            }
            default: {
                std::vector<float> v(n);
                for (int64_t i = 0; i < n; i++) {
                    v[i] = (float) (rng() % 2001) / 1000.0f - 1.0f;
                }
                ggml_backend_tensor_set(t, v.data(), 0, n * sizeof(float));
                give(t, {});
                break;
            }
        }
    }
    return all_valid;
}

} // namespace

void decode_session(FuzzedDataProvider & fdp, bool async, fakedsp::config & cfg, hexhost::options & o) {
    static const uint32_t threads[] = { 6, 6, 6, 4, 2, 1, 8, 10 };
    static const uint64_t vtcms[]   = { 8u << 20, 8u << 20, 8u << 20, 8u << 20, 2u << 20, 1u << 20 };
    cfg.n_threads = threads[fdp.ConsumeIntegralInRange<size_t>(0, 7)];
    cfg.n_hmx     = harness::rare(fdp, 6) ? 0 : 1;
    cfg.vtcm_size = vtcms[fdp.ConsumeIntegralInRange<size_t>(0, 5)];
    cfg.async     = async;
    cfg.touch     = async || !harness::rare(fdp, 2);
    // 0xA5 bytes read as a negative index, zero bytes as slot 0 (a valid slot of a wrong row)
    cfg.fill      = harness::rare(fdp, 3) ? 0x00 : 0xA5;

    static const int batches[] = { 1280, 1280, 64, 16, 8, 4, 2, 1 };
    static const int queues[]  = { 32, 32, 4, 2, 1 };
    static const int fusions[] = { 1, 1, 1, 1, 0, 4, 8, 16, 64, 8 | 16 | 64, 4 | 8 };
    static const int caches[]  = { 1, 1, 1, 0, 2 };
    static const int gcaches[] = { 8, 8, 1, 2 };
    o.opbatch        = batches[fdp.ConsumeIntegralInRange<size_t>(0, 7)];
    o.opqueue        = queues[fdp.ConsumeIntegralInRange<size_t>(0, 4)];
    o.oppoll         = harness::rare(fdp, 6) ? 1 : 0;
    o.opfusion       = fusions[fdp.ConsumeIntegralInRange<size_t>(0, 10)];
    o.opfusion_state = fdp.ConsumeIntegralInRange<int>(-1, 1);
    o.batchcache     = caches[fdp.ConsumeIntegralInRange<size_t>(0, 4)];
    o.graphcache     = gcaches[fdp.ConsumeIntegralInRange<size_t>(0, 3)];
    o.profile        = harness::rare(fdp, 10) ? (fdp.ConsumeBool() ? 1 : 3) : 0;
    o.hostprof       = harness::rare(fdp, 10) ? 1 : 0;
    o.multirow       = fdp.ConsumeBool() ? 1 : 0;
    o.gdn_chunk      = fdp.ConsumeBool() ? 1 : 0;
    o.nhmx           = cfg.n_hmx ? (harness::rare(fdp, 6) ? 0 : 1) : 1;
    o.mm_select      = harness::rare(fdp, 6) ? fdp.ConsumeIntegralInRange<int>(1, 2) : 3;
    o.vmem           = 64u << 20;
}

void run_steps(FuzzedDataProvider & fdp, world & w, const step_hooks & h) {
    std::mt19937         rng(fdp.ConsumeIntegral<uint32_t>());
    ggml_backend_event_t ev      = nullptr;
    const uint32_t       n_steps = 2 + fdp.ConsumeIntegralInRange<uint32_t>(0, 4);

    for (uint32_t step = 0; step < n_steps; step++) {
        const size_t gi = w.graphs.size() > 1 ? (size_t) fdp.ConsumeIntegralInRange<int>(0, 1) : 0;
        if (harness::rare(fdp, 10)) {
            if (h.on_forget) {
                h.on_forget(w.graphs[gi]);
            }
            if (!rebuild_graph(fdp, w, gi)) {
                break;
            }
        }
        graph_spec & g     = w.graphs[gi];
        const bool   valid = set_inputs(fdp, g, step, rng, h, w.numeric);
        if (h.on_slots) {
            h.on_slots(valid);
        }

        for (size_t s = 0; s + 1 < g.cuts.size(); s++) {
            ggml_cgraph view = ggml_graph_view(g.gf, g.cuts[s], g.cuts[s + 1]);
            view.uid         = g.uids[s];
            const ggml_status st = ggml_backend_graph_compute_async(w.backend, &view);
            if (h.on_split) {
                h.on_split(g, s, st);
            }
            const int ev_mode = fdp.ConsumeIntegralInRange<int>(0, 7);
            if (ev_mode == 0) {
                if (!ev) {
                    ev = ggml_backend_event_new(hexhost::device_dev(w.dev));
                }
                if (ev) {
                    ggml_backend_event_record(ev, w.backend);
                }
            } else if (ev_mode == 1 && ev) {
                ggml_backend_event_synchronize(ev);
            }
        }

        if (harness::rare(fdp, 4) && !g.outputs.empty()) {
            // the read of an output through the async path flushes the session first
            ggml_tensor *        o0 = g.outputs.front();
            std::vector<uint8_t> buf(ggml_nbytes(o0));
            ggml_backend_tensor_get_async(w.backend, o0, buf.data(), 0, buf.size());
            ggml_backend_synchronize(w.backend);
        }
        ggml_backend_synchronize(w.backend);
        if (h.on_slots) {
            h.on_slots(false);
        }
        if (h.after_step && !h.after_step(g, step, valid)) {
            break;
        }
    }
    if (ev) {
        ggml_backend_event_free(ev);
    }
}

bool build_world(FuzzedDataProvider & fdp, hexhost::device * dev, world & w) {
    w.dev     = dev;
    w.backend = ggml_backend_dev_init(hexhost::device_dev(dev), nullptr);
    if (!w.backend) {
        return false;
    }
    pick_params(fdp, w.mp);
    if (!make_tensors(fdp, w)) {
        return false;
    }
    const int n_graphs = harness::rare(fdp, 3) ? 2 : 1;
    for (int gi = 0; gi < n_graphs; gi++) {
        w.graphs.emplace_back();
        const int64_t T = gi == 0 ? pick_tokens(fdp) : 1;
        if (!build_graph(fdp, w, w.graphs.back(), T, harness::rare(fdp, 8))) {
            return false;
        }
    }
    return true;
}

bool rebuild_graph(FuzzedDataProvider & fdp, world & w, size_t gi) {
    free_graph(w.graphs[gi]);
    return build_graph(fdp, w, w.graphs[gi], pick_tokens(fdp), harness::rare(fdp, 8));
}

void free_world(world & w) {
    for (auto & g : w.graphs) {
        free_graph(g);
    }
    w.graphs.clear();
    if (w.buf_w) {
        ggml_backend_buffer_free(w.buf_w);
    }
    if (w.buf_c) {
        ggml_backend_buffer_free(w.buf_c);
    }
    if (w.ctx_w) {
        ggml_free(w.ctx_w);
    }
    if (w.ctx_c) {
        ggml_free(w.ctx_c);
    }
    if (w.backend) {
        ggml_backend_free(w.backend);
    }
    w = world();
}

} // namespace graphgen
