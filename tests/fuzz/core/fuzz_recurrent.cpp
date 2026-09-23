// fuzz_recurrent: the hybrid memory of Qwen3.5 (KV cache plus llama_memory_recurrent), the
// recurrent rollback slots (n_rs_seq) of our patches, and the sequence state blobs.
//
// The harness loads the tiny random qwen35 model (make_tiny_model writes
// data/tiny-qwen35-f32.gguf, and FUZZ_MODEL gives a different file) one time, on
// the CPU (FUZZ_DEVICE names a different device, HTP0 for example). For
// each input it creates a context with parameters from the input (n_seq_max
// 1 to 4, n_rs_seq 0 to 5, n_ubatch, kv_unified, flash attention, threads,
// LLAMA_EMBD_LOOKUP_HOST on or off), then runs a program of operations from
// the input bytes:
//
//   decode      one sequence, 1 to n_rs_seq + 3 tokens (a verify step of
//               speculative decoding has n_rs_seq + 1 tokens)
//   decode-multi two or more sequences in one batch
//   rollback    llama_memory_seq_rm(s, len - r, -1) with r = 1 to 6, as the
//               app does after a rejected draft
//   remove      a random range, or the full sequence
//   copy        llama_memory_seq_cp(src, dst, -1, -1)
//   keep        llama_memory_seq_keep(s)
//   state-seq   llama_state_seq_get_data, then an optional corruption (bytes,
//               a 32-bit value, a truncation), then llama_state_seq_set_data
//               into a sequence, as the app does with its snapshot store
//   state-all   the same for llama_state_get_data and llama_state_set_data
//   clear       llama_memory_clear
//
// The harness keeps a shadow of the tokens that each sequence holds.
// Properties:
//   P1  No crash, no sanitizer report, no exception out of the C API.
//   P2  Oracle: after each decode, the logits of the last token of each
//       sequence equal the logits of a reference context (n_rs_seq 0, one
//       sequence, the same flash attention setting) that decodes the shadow
//       tokens of that sequence from position 0. The limit is FUZZ_RECURRENT_TOL * (1 + the largest
//       absolute reference logit), with FUZZ_RECURRENT_TOL 2e-4 as the preset
//       value. A wrong rollback
//       slot, a stale snapshot group or a wrong state copy gives the logits of
//       a different history, which is far outside the tolerance.
//   P3  llama_memory_seq_pos_max(s) == length - 1 for each sequence.
//   P4  A refused rollback (seq_rm returns false) does not change the memory.
//   P5  An intact state blob loads (returns non-zero) into any sequence.
//   P6  A sequence load that fails (returns 0) leaves no part of the
//       destination sequence, in the KV cache and in the recurrent memory.
//       The loader of patches/fuzz-core/0026 writes the tensor data after it
//       reads the blob, and a NaN stops it after some parts are written: the
//       load then removes the sequence.
//
// A sequence becomes "tainted" after an operation that has a result that the
// shadow cannot predict (a corrupted blob that loads, a middle range remove).
// The oracle ignores a tainted sequence until the next clear.
//
// The known limit (check_known_limit() in LLVMFuzzerInitialize is its test):
// the memory keeps the K and V rows of a cell after a remove of its sequence,
// and the attention without flash attention reads each cell of its window,
// also the cells of the other sequences. A masked cell gets the weight 0
// there, but 0 * NaN is NaN. The flash attention of the CPU skips a masked
// cell. The loader rejects a blob with a NaN or an Inf
// (patches/fuzz-core/0025), thus a blob cannot bring one in. A decode that
// overflows can still write one: then the other sequences of its KV stream,
// and each later sequence in it, get NaN logits without flash attention,
// until a clear with data. Thus a P2 failure with NaN logits, after a decode
// in which a tainted sequence with finite data gave NaN logits (FUZZ_TRACE=1
// writes that line), is this limit and not a new defect.
//
// Switches for a replay:
//   FUZZ_TRACE=1                one line for each operation
//   FUZZ_RECURRENT_CALIBRATE=1  write each difference of P2, and never fail P2
//   FUZZ_NONFINITE=N            for each decode of the context under test, write the first N graph
//                               nodes with a NaN or an Inf, and the count in each of their sources

#include "fuzz_common.h"

#include <cinttypes>
#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr int kMaxSeq     = 4;
constexpr int kMaxHist    = 64;
constexpr int kCtx        = 256;
constexpr int kBatch      = 64;

/**
 * One reference context: n_rs_seq 0, one sequence, F32 KV cache. It keeps
 * the tokens that it holds and the logits of the last one, thus a history
 * that extends the previous one decodes only the new tokens. Any other
 * history decodes again from position 0 in one batch.
 */
struct RefCtx {
    llama_context *          ctx = nullptr;
    std::vector<llama_token> hist;
    std::vector<float>       last;
};

llama_model * g_model = nullptr;
// One set of reference contexts with flash attention off [0] and one with it on [1]. The context
// under test uses the set of its own setting (auto is on for the CPU), because the two attention
// paths differ by more than the tolerance in an x86 build without SIMD (8.8e-3 against 7.5e-5
// with AVX2, the tiny model, three tokens).
RefCtx        g_ref_sets[2][kMaxSeq];
RefCtx *      g_ref = g_ref_sets[1];
int32_t       g_n_vocab = 0;
float         g_tol   = 2e-4f;
bool          g_trace = false;
bool          g_calibrate = false;  // FUZZ_RECURRENT_CALIBRATE=1: print each difference, never fail P2
long          g_nonfinite = 0;      // FUZZ_NONFINITE=N: write the first N nodes with a NaN or an Inf

/** The state of the FUZZ_NONFINITE watch for one decode of the context under test. */
struct NonFiniteWatch {
    int  decode = 0;  // the number of the decode in this input, from 1
    int  node   = 0;  // the nodes that the watch examined in this decode
    long found  = 0;  // the nodes of this decode with a NaN or an Inf, up to g_nonfinite
};
NonFiniteWatch g_watch;

/**
 * The count of the NaN and Inf values of an F32, F16 or BF16 tensor, or -1 for a different type or
 * a tensor with no data. A view can have gaps between its rows, thus the function reads the full
 * span and then each element through the strides. *first gets the flat index of the first
 * non-finite value. O(ggml_nelements(t)).
 */
int64_t count_nonfinite(const ggml_tensor * t, int64_t * first) {
    *first = -1;
    if ((t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) ||
        t->data == nullptr || t->buffer == nullptr || ggml_nelements(t) == 0) {
        return -1;
    }
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    int64_t count = 0, flat = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                const uint8_t * row = bytes.data() + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0, ++flat) {
                    const uint8_t * p = row + i0 * t->nb[0];
                    float v;
                    if (t->type == GGML_TYPE_F32) {
                        memcpy(&v, p, sizeof(v));
                    } else if (t->type == GGML_TYPE_F16) {
                        ggml_fp16_t h;
                        memcpy(&h, p, sizeof(h));
                        v = ggml_fp16_to_fp32(h);
                    } else {
                        ggml_bf16_t h;
                        memcpy(&h, p, sizeof(h));
                        v = ggml_bf16_to_fp32(h);
                    }
                    if (!std::isfinite(v)) {
                        if (count == 0) {
                            *first = flat;
                        }
                        ++count;
                    }
                }
            }
        }
    }
    return count;
}

/**
 * The eval callback of FUZZ_NONFINITE=N. It examines each node of the graph after the backend
 * computes it. For each of the first N nodes with a NaN or an Inf, it writes the node and the count
 * of each source. After N such nodes it examines no more nodes in this decode. It never stops the
 * computation, thus the program of the input does not change. The scheduler computes each examined
 * node alone, thus a fusion of the backend does not occur with the switch on. The count of a
 * source is the count after the node ran: an in-place node writes its source.
 */
bool nonfinite_cb(ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask) {
        return g_watch.found < g_nonfinite;
    }
    ++g_watch.node;
    // A view computes nothing. The reshape of a full cache, and a write into a cache (SET_ROWS,
    // CPY), are views of all the rows of the cache, also the rows that the graph does not read. The
    // first node that computes from a NaN shows it, with its sources.
    if (ggml_is_empty(t) || t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
        t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_SET_ROWS || t->op == GGML_OP_CPY) {
        return true;
    }
    int64_t first = -1;
    const int64_t n = count_nonfinite(t, &first);
    if (n <= 0) {
        return true;
    }
    ++g_watch.found;
    fprintf(stderr, "nonfinite: decode %d, node %d: %s '%s' %s [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64
                    "] has %" PRId64 " NaN or Inf of %" PRId64 " (the first at %" PRId64 ")\n",
            g_watch.decode, g_watch.node, ggml_op_desc(t), t->name, ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2],
            t->ne[3], n, ggml_nelements(t), first);
    for (int k = 0; k < GGML_MAX_SRC && t->src[k] != nullptr; ++k) {
        const ggml_tensor * s = t->src[k];
        const int64_t       m = count_nonfinite(s, &first);
        fprintf(stderr, "nonfinite:   src %d: %s '%s' %s [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]: %" PRId64
                        " NaN or Inf%s\n",
                k, ggml_op_desc(s), s->name, ggml_type_name(s->type), s->ne[0], s->ne[1], s->ne[2], s->ne[3], m,
                m < 0 ? " (not examined)" : "");
    }
    return true;
}

/** The shadow of one sequence: its tokens, and whether the oracle can predict it. */
struct Shadow {
    std::vector<llama_token> hist;
    bool tainted = false;
    bool pending = false;  // a rollback is done and the next decode has not run yet
};

/** True when each of the n values is a NaN or an Inf. O(n). */
bool all_nonfinite(const float * v, int n) {
    for (int i = 0; i < n; ++i) {
        if (std::isfinite(v[i])) {
            return false;
        }
    }
    return true;
}

/** True when each of the n values is finite. O(n). */
bool all_finite(const float * v, int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) {
            return false;
        }
    }
    return true;
}

/** Write one line of the operation trace when FUZZ_TRACE=1. */
void trace(const char * fmt, ...) {
    if (!g_trace) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
}

/** The logits of the last token of hist, from reference context r. */
const std::vector<float> & reference_logits(RefCtx & r, const std::vector<llama_token> & hist) {
    const bool extends = r.hist.size() <= hist.size() && std::equal(r.hist.begin(), r.hist.end(), hist.begin());
    if (extends && r.hist.size() == hist.size() && !r.last.empty()) {
        return r.last;
    }
    size_t from = r.hist.size();
    if (!extends || r.hist.empty()) {
        llama_memory_clear(llama_get_memory(r.ctx), true);
        r.hist.clear();
        from = 0;
    }
    const int32_t n = (int32_t) (hist.size() - from);
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int32_t i = 0; i < n; ++i) {
        b.token[i]     = hist[from + i];
        b.pos[i]       = (llama_pos) (from + i);
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = i + 1 == n;
    }
    b.n_tokens = n;
    const int rc = llama_decode(r.ctx, b);
    llama_batch_free(b);
    if (rc != 0) {
        fuzz::fail("the reference context fails to decode %d tokens from position %zu (code %d)", n, from, rc);
    }
    const float * l = llama_get_logits_ith(r.ctx, -1);
    r.hist = hist;
    r.last.assign(l, l + g_n_vocab);
    return r.last;
}

/** P2: compare the logits of the context with the reference for the tokens of the sequence. */
void check_logits(const float * got, const Shadow & sh, int seq, const char * what) {
    const std::vector<float> & ref = reference_logits(g_ref[seq], sh.hist);
    float max_ref = 0.0f, max_diff = 0.0f;
    int   at = -1;
    for (int i = 0; i < g_n_vocab; ++i) {
        max_ref = std::max(max_ref, std::fabs(ref[i]));
        const float d = std::fabs(got[i] - ref[i]);
        if (!(d <= max_diff)) {  // a NaN also goes here
            max_diff = d;
            at = i;
        }
    }
    const float limit = g_tol * (1.0f + max_ref);
    trace("  check seq %d len %zu: max diff %g (limit %g)", seq, sh.hist.size(), max_diff, limit);
    if (g_calibrate) {
        fprintf(stderr, "calibrate: %s seq %d len %zu max diff %g max ref %g\n", what, seq, sh.hist.size(), max_diff, max_ref);
        return;
    }
    if (!(max_diff <= limit)) {
        fuzz::fail("P2: after %s, seq %d (%zu tokens) has logit %d = %g, the reference gives %g (max diff %g, limit %g)",
                   what, seq, sh.hist.size(), at, at >= 0 ? got[at] : 0.0f, at >= 0 ? ref[at] : 0.0f, max_diff, limit);
    }
}

/** P3 for each sequence that is not tainted. */
void check_pos(llama_memory_t mem, const Shadow * sh, int n_seq, const char * what) {
    for (int s = 0; s < n_seq; ++s) {
        if (sh[s].tainted) {
            continue;
        }
        const llama_pos got  = llama_memory_seq_pos_max(mem, s);
        const llama_pos want = (llama_pos) sh[s].hist.size() - 1;
        if (got != want) {
            fuzz::fail("P3: after %s, seq_pos_max(%d) is %d, the shadow holds %zu tokens", what, s, got, sh[s].hist.size());
        }
    }
}

/** Remove a sequence from the memory and from the shadow. The removal of a full sequence must succeed. */
void drop_seq(llama_memory_t mem, Shadow & sh, int s) {
    if (!llama_memory_seq_rm(mem, s, -1, -1)) {
        fuzz::fail("llama_memory_seq_rm(%d, -1, -1) refuses to remove a full sequence", s);
    }
    sh.hist.clear();
    sh.tainted = false;
    sh.pending = false;
}

/** The context parameters of one input. */
struct CtxParams {
    uint32_t n_seq_max;
    uint32_t n_rs_seq;
    uint32_t n_ubatch;
    bool     unified;
    int      flash;
    int      threads;
    bool     embd_host;
};

/** Create the context under test. */
llama_context * make_ctx(const CtxParams & p) {
    // llama_context reads LLAMA_EMBD_LOOKUP_HOST in its constructor
    setenv("LLAMA_EMBD_LOOKUP_HOST", p.embd_host ? "1" : "0", 1);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = kCtx;
    cp.n_batch         = kBatch;
    cp.n_ubatch        = p.n_ubatch;
    cp.n_seq_max       = p.n_seq_max;
    cp.n_rs_seq        = p.n_rs_seq;
    cp.kv_unified      = p.unified;
    cp.flash_attn_type = (llama_flash_attn_type) p.flash;
    cp.n_threads       = p.threads;
    cp.n_threads_batch = p.threads;
    cp.no_perf         = true;
    // F32 K and V: an F16 cache rounds a difference of one ulp in K to 1e-3 of the logits,
    // which is the size of the effect of a wrong state in the tiny model
    cp.type_k          = GGML_TYPE_F32;
    cp.type_v          = GGML_TYPE_F32;
    if (g_nonfinite > 0) {
        cp.cb_eval           = nonfinite_cb;
        cp.cb_eval_user_data = nullptr;
    }
    llama_context * ctx = llama_init_from_model(g_model, cp);
    setenv("LLAMA_EMBD_LOOKUP_HOST", "0", 1);
    return ctx;
}

/**
 * Corrupt a state blob in place: flip bytes, write a 32-bit value, or change the size. The 32-bit
 * values include counts of 0x7fffffff and more, which a loader must refuse before it sizes an
 * allocation with them.
 */
void corrupt(FuzzedDataProvider & fdp, std::vector<uint8_t> & blob) {
    switch (fdp.ConsumeIntegralInRange<int>(0, 3)) {
        case 0: {
            const int n = fdp.ConsumeIntegralInRange<int>(1, 4);
            for (int i = 0; i < n && !blob.empty(); ++i) {
                blob[fdp.ConsumeIntegralInRange<size_t>(0, blob.size() - 1)] ^= fdp.ConsumeIntegralInRange<uint8_t>(1, 255);
            }
            break;
        }
        case 1: {
            static const uint32_t kValues[] = { 0, 1, 2, 3, 4, 5, 255, 0x7fffffff, 0x80000000, 0xffffffff };
            const uint32_t v = kValues[fdp.ConsumeIntegralInRange<int>(0, 9)];
            if (blob.size() >= 4) {
                // the fields of the state blob are 4-byte aligned
                const size_t off = 4 * fdp.ConsumeIntegralInRange<size_t>(0, blob.size() / 4 - 1);
                memcpy(blob.data() + off, &v, 4);
            }
            break;
        }
        case 2:
            blob.resize(fdp.ConsumeIntegralInRange<size_t>(0, blob.size()));
            break;
        default:
            blob.resize(blob.size() + fdp.ConsumeIntegralInRange<size_t>(1, 64), 0xAB);
            break;
    }
}

/** Run one input. */
void run(FuzzedDataProvider & fdp) {
    static const uint32_t kUbatch[] = { 1, 2, 3, 5, 8, 16, 64 };
    CtxParams p;
    p.n_seq_max = fdp.ConsumeIntegralInRange<uint32_t>(1, kMaxSeq);
    p.n_rs_seq  = fdp.ConsumeIntegralInRange<uint32_t>(0, 5);
    p.n_ubatch  = kUbatch[fdp.ConsumeIntegralInRange<int>(0, 6)];
    p.unified   = fdp.ConsumeBool();
    p.flash     = fdp.ConsumeIntegralInRange<int>(-1, 1);
    // The byte of the thread count is read also when FUZZ_THREADS gives the count, thus an input
    // gives the same program with and without FUZZ_THREADS (the phone commands set it).
    const int threads_in = fdp.ConsumeIntegralInRange<int>(1, 2);
    p.threads   = fuzz::env_long("FUZZ_THREADS", 0) > 0 ? (int) fuzz::env_long("FUZZ_THREADS", 0) : threads_in;
    p.embd_host = fdp.ConsumeBool();
    trace("ctx: n_seq_max %u n_rs_seq %u n_ubatch %u unified %d flash %d threads %d embd_host %d",
          p.n_seq_max, p.n_rs_seq, p.n_ubatch, p.unified, p.flash, p.threads, p.embd_host);

    g_watch = NonFiniteWatch{};
    // the reference contexts with the flash attention setting of this context (auto is on for the CPU)
    g_ref = g_ref_sets[p.flash == 0 ? 0 : 1];
    llama_context * ctx = make_ctx(p);
    // The recurrent memory keeps the last n_rs_seq + 1 tokens of each sequence in one ubatch, thus the
    // context must refuse a smaller n_ubatch. A model with no recurrent rollback gets n_rs_seq 0.
    if (p.n_rs_seq > 0 && p.n_ubatch <= p.n_rs_seq + 1) {
        if (ctx != nullptr && llama_n_rs_seq(ctx) > 0) {
            fuzz::fail("llama_init_from_model accepts n_ubatch %u with n_rs_seq %u", p.n_ubatch, llama_n_rs_seq(ctx));
        }
        if (ctx == nullptr) {
            return;
        }
    }
    if (ctx == nullptr) {
        fuzz::fail("llama_init_from_model fails for n_seq_max %u n_rs_seq %u n_ubatch %u", p.n_seq_max, p.n_rs_seq, p.n_ubatch);
    }
    llama_memory_t mem = llama_get_memory(ctx);
    const int n_seq = (int) p.n_seq_max;
    Shadow sh[kMaxSeq];

    const int n_ops = fdp.ConsumeIntegralInRange<int>(1, 24);
    for (int op_i = 0; op_i < n_ops && fdp.remaining_bytes() > 0; ++op_i) {
        const int op = fdp.ConsumeIntegralInRange<int>(0, 13);
        if (op <= 4) {
            // decode: one sequence (op 0 to 3) or several (op 4)
            std::vector<int> seqs;
            if (op == 4 && n_seq > 1) {
                for (int s = 0; s < n_seq; ++s) {
                    if (fdp.ConsumeBool()) {
                        seqs.push_back(s);
                    }
                }
            }
            if (seqs.empty()) {
                seqs.push_back(fdp.ConsumeIntegralInRange<int>(0, n_seq - 1));
            }
            std::vector<int> n_tok(seqs.size());
            int total = 0;
            for (size_t k = 0; k < seqs.size(); ++k) {
                n_tok[k] = fdp.ConsumeIntegralInRange<int>(1, (int) p.n_rs_seq + 3);
                if (sh[seqs[k]].hist.size() + n_tok[k] > kMaxHist) {
                    drop_seq(mem, sh[seqs[k]], seqs[k]);
                }
                total += n_tok[k];
            }
            llama_batch b = llama_batch_init(total, 0, 1);
            std::vector<int> last_idx(seqs.size());
            int i = 0;
            // the tokens of the sequences interleave when the input says so, as a server batch does
            const bool interleave = fdp.ConsumeBool();
            std::vector<int> done(seqs.size(), 0);
            std::vector<std::vector<llama_token>> added(seqs.size());
            while (i < total) {
                for (size_t k = 0; k < seqs.size() && i < total; ++k) {
                    const int take = interleave ? 1 : n_tok[k];
                    for (int t = 0; t < take && done[k] < n_tok[k]; ++t) {
                        const int s = seqs[k];
                        const llama_token tok = fdp.ConsumeIntegralInRange<llama_token>(0, g_n_vocab - 1);
                        b.token[i]     = tok;
                        b.pos[i]       = (llama_pos) (sh[s].hist.size() + added[k].size());
                        b.n_seq_id[i]  = 1;
                        b.seq_id[i][0] = s;
                        b.logits[i]    = fdp.ConsumeIntegralInRange<int>(0, 7) == 0;
                        added[k].push_back(tok);
                        last_idx[k] = i;
                        done[k]++;
                        i++;
                    }
                }
            }
            for (size_t k = 0; k < seqs.size(); ++k) {
                b.logits[last_idx[k]] = 1;
            }
            b.n_tokens = total;
            g_watch.decode++;
            g_watch.node  = 0;
            g_watch.found = 0;
            const int rc = llama_decode(ctx, b);
            trace("decode %zu seqs, %d tokens: rc %d", seqs.size(), total, rc);
            if (rc != 0) {
                // the memory of a failed batch is not defined for the shadow: remove the sequences
                for (const int s : seqs) {
                    drop_seq(mem, sh[s], s);
                }
            } else {
                for (size_t k = 0; k < seqs.size(); ++k) {
                    Shadow & s = sh[seqs[k]];
                    s.hist.insert(s.hist.end(), added[k].begin(), added[k].end());
                    s.pending = false;
                    const float * l = llama_get_logits_ith(ctx, last_idx[k]);
                    if (l == nullptr) {
                        fuzz::fail("llama_get_logits_ith(%d) gives null after a decode that requested it", last_idx[k]);
                    }
                    if (!s.tainted) {
                        check_logits(l, s, seqs[k], "decode");
                    } else if (!all_finite(l, g_n_vocab)) {
                        // a later P2 failure with NaN logits can be the known limit (the header)
                        trace("  seq %d (tainted) gives NaN or Inf logits", seqs[k]);
                    }
                }
            }
            llama_batch_free(b);
        } else if (op <= 7) {
            // rollback of the last r tokens, as after a rejected draft
            const int s = fdp.ConsumeIntegralInRange<int>(0, n_seq - 1);
            const int r = fdp.ConsumeIntegralInRange<int>(1, 6);
            const int len = (int) sh[s].hist.size();
            if (len - r <= 0) {
                continue;
            }
            const bool ok = llama_memory_seq_rm(mem, s, len - r, -1);
            trace("rollback seq %d by %d (len %d): %s", s, r, len, ok ? "done" : "refused");
            if (ok) {
                sh[s].hist.resize(len - r);
                sh[s].pending = true;
            }
        } else if (op == 8) {
            const int s = fdp.ConsumeIntegralInRange<int>(0, n_seq - 1);
            if (fdp.ConsumeBool()) {
                drop_seq(mem, sh[s], s);
                trace("remove seq %d", s);
            } else {
                const llama_pos p0 = fdp.ConsumeIntegralInRange<llama_pos>(-1, kMaxHist);
                const llama_pos p1 = fdp.ConsumeIntegralInRange<llama_pos>(-1, kMaxHist);
                const bool ok = llama_memory_seq_rm(mem, s, p0, p1);
                trace("remove seq %d [%d, %d): %s", s, p0, p1, ok ? "done" : "refused");
                if (ok) {
                    const llama_pos len = (llama_pos) sh[s].hist.size();
                    const llama_pos a = p0 < 0 ? 0 : p0;
                    const llama_pos e = p1 < 0 ? INT32_MAX : p1;
                    if (a == 0 && e >= len) {
                        sh[s].hist.clear();
                        sh[s].tainted = false;
                    } else if (a < len && e >= len) {
                        sh[s].hist.resize(a);
                        sh[s].pending = true;  // a suffix remove is a rollback
                    } else if (a < len && a < e) {
                        sh[s].tainted = true;  // a middle range: the recurrent state keeps the removed tokens
                    }
                }
            }
        } else if (op == 9) {
            const int src = fdp.ConsumeIntegralInRange<int>(0, n_seq - 1);
            const int dst = fdp.ConsumeIntegralInRange<int>(0, n_seq - 1);
            // A copy replaces the destination, also when the destination holds tokens: the KV cache
            // and the recurrent memory then hold the history of the source for it.
            llama_memory_seq_cp(mem, src, dst, -1, -1);
            trace("copy seq %d -> %d", src, dst);
            if (src != dst) {
                sh[dst] = sh[src];
            }
        } else if (op == 10) {
            const int s = fdp.ConsumeIntegralInRange<int>(0, n_seq - 1);
            llama_memory_seq_keep(mem, s);
            trace("keep seq %d", s);
            for (int o = 0; o < n_seq; ++o) {
                if (o != s) {
                    sh[o] = Shadow{};
                }
            }
        } else if (op == 11) {
            const int src = fdp.ConsumeIntegralInRange<int>(0, n_seq - 1);
            const int dst = fdp.ConsumeIntegralInRange<int>(0, n_seq - 1);
            const size_t size = llama_state_seq_get_size(ctx, src);
            std::vector<uint8_t> blob(size);
            const size_t got = llama_state_seq_get_data(ctx, blob.data(), blob.size(), src);
            if (got == 0 || got > size) {
                fuzz::fail("llama_state_seq_get_data(%d) returns %zu for a buffer of %zu", src, got, size);
            }
            blob.resize(got);
            const bool corrupted = fdp.ConsumeIntegralInRange<int>(0, 2) == 0;
            if (corrupted) {
                corrupt(fdp, blob);
            }
            // the app empties the sequence first, then loads
            llama_memory_seq_rm(mem, dst, -1, -1);
            const size_t rc = llama_state_seq_set_data(ctx, blob.data(), blob.size(), dst);
            trace("state seq %d -> %d, %zu bytes%s: rc %zu", src, dst, blob.size(), corrupted ? " corrupted" : "", rc);
            if (rc == 0) {
                if (!corrupted) {
                    fuzz::fail("P5: an intact state blob of seq %d (%zu bytes) does not load into seq %d", src, blob.size(), dst);
                }
                // P6: the hybrid memory gives the larger of the two minimum positions, thus a part in the
                // KV cache or in the recurrent memory alone shows here
                const llama_pos left_min = llama_memory_seq_pos_min(mem, dst);
                const llama_pos left_max = llama_memory_seq_pos_max(mem, dst);
                if (left_min != -1 || left_max != -1) {
                    fuzz::fail("P6: a failed load into seq %d leaves positions %d to %d of it in the memory", dst, left_min, left_max);
                }
                drop_seq(mem, sh[dst], dst);
            } else {
                sh[dst] = sh[src];
                sh[dst].tainted = sh[dst].tainted || corrupted;
                sh[dst].pending = false;  // a load writes group 0 and sets the rollback index to 0
            }
        } else if (op == 12) {
            const size_t size = llama_state_get_size(ctx);
            std::vector<uint8_t> blob(size);
            const size_t got = llama_state_get_data(ctx, blob.data(), blob.size());
            if (got == 0 || got > size) {
                fuzz::fail("llama_state_get_data returns %zu for a buffer of %zu", got, size);
            }
            blob.resize(got);
            const bool corrupted = fdp.ConsumeIntegralInRange<int>(0, 2) == 0;
            if (corrupted) {
                corrupt(fdp, blob);
            }
            const size_t rc = llama_state_set_data(ctx, blob.data(), blob.size());
            trace("state all, %zu bytes%s: rc %zu", blob.size(), corrupted ? " corrupted" : "", rc);
            if (rc == 0 || corrupted) {
                if (rc == 0 && !corrupted) {
                    fuzz::fail("P5: an intact full state blob (%zu bytes) does not load", blob.size());
                }
                llama_memory_clear(mem, true);
                for (int s = 0; s < n_seq; ++s) {
                    sh[s] = Shadow{};
                }
            } else {
                for (int s = 0; s < n_seq; ++s) {
                    sh[s].pending = false;  // the load writes group 0 of each cell
                }
            }
        } else {
            llama_memory_clear(mem, fdp.ConsumeBool());
            trace("clear");
            for (int s = 0; s < n_seq; ++s) {
                sh[s] = Shadow{};
            }
        }
        check_pos(mem, sh, n_seq, "an operation");
    }

    llama_free(ctx);
}

/** A silent progress callback. */
bool silent(float /*progress*/, void * /*user_data*/) {
    return true;
}

/** The state of the probe of the known limit: while it is armed, each F32 Vcur node gets NaN values. */
struct LimitProbe {
    bool armed = false;
    int  hits  = 0;
};
LimitProbe g_probe;

/** The eval callback of the probe. It writes NaN into the V rows of the decode, as an overflow does. */
bool probe_cb(ggml_tensor * t, bool ask, void * /*user_data*/) {
    const bool vcur = g_probe.armed && strncmp(t->name, "Vcur-", 5) == 0 && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
    if (ask) {
        return vcur;
    }
    if (vcur) {
        const std::vector<float> nan((size_t) ggml_nelements(t), NAN);
        ggml_backend_tensor_set(t, nan.data(), 0, ggml_nbytes(t));
        ++g_probe.hits;
    }
    return true;
}

/** Decode the tokens of one sequence from position 0 in ctx, and give the logits of the last token. */
std::vector<float> probe_decode(llama_context * ctx, int seq, const std::vector<llama_token> & toks) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = seq;
        b.logits[i]    = i + 1 == toks.size();
    }
    b.n_tokens = (int32_t) toks.size();
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) {
        fuzz::fail("the probe of the known limit fails to decode seq %d (code %d)", seq, rc);
    }
    const float * l = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(l, l + g_n_vocab);
}

/**
 * The test of the known limit (the header). A context of two sequences decodes seq 0 with NaN
 * values in its V rows (as a decode that overflows), removes seq 0, and decodes seq 1 in the cells
 * that seq 0 held. One cell of seq 0 stays free with its NaN row in the window. The flash attention
 * of the CPU skips the masked cells: seq 1 must match the reference. The attention without flash
 * attention reads them: each logit of seq 1 must be NaN or Inf. When this second result changes,
 * the limit is gone: correct the header. CPU only.
 */
void check_known_limit() {
    const std::vector<llama_token> h0 = { 11, 48, 85 };
    const std::vector<llama_token> h1 = { 122, 159 };
    for (int f = 0; f < 2; ++f) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = kCtx;
        cp.n_batch         = kBatch;
        cp.n_ubatch        = kBatch;
        cp.n_seq_max       = 2;
        cp.kv_unified      = true;
        cp.n_threads       = 1;
        cp.n_threads_batch = 1;
        cp.no_perf         = true;
        cp.type_k          = GGML_TYPE_F32;
        cp.type_v          = GGML_TYPE_F32;
        cp.flash_attn_type = f == 0 ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.cb_eval         = probe_cb;
        llama_context * ctx = llama_init_from_model(g_model, cp);
        if (ctx == nullptr) {
            fuzz::fail("cannot create the context of the known-limit probe (flash attention %s)", f == 0 ? "off" : "on");
        }
        g_probe = LimitProbe{ true, 0 };
        probe_decode(ctx, 0, h0);
        g_probe.armed = false;
        if (g_probe.hits == 0) {
            fuzz::fail("the known-limit probe finds no contiguous F32 node Vcur-<layer> for its NaN values");
        }
        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        const std::vector<float> got = probe_decode(ctx, 1, h1);
        llama_free(ctx);

        const std::vector<float> & ref = reference_logits(g_ref_sets[f][0], h1);
        float max_ref = 0.0f, max_diff = 0.0f;
        for (int i = 0; i < g_n_vocab; ++i) {
            max_ref  = std::max(max_ref, std::fabs(ref[i]));
            max_diff = std::max(max_diff, std::fabs(got[i] - ref[i]));  // std::max drops a NaN difference
        }
        const float limit = g_tol * (1.0f + max_ref);
        if (f == 1) {
            if (!all_finite(got.data(), g_n_vocab) || (!(max_diff <= limit) && !g_calibrate)) {
                fuzz::fail("the flash attention of the CPU passes the NaN rows of a removed sequence to a new sequence "
                           "(max diff %g, limit %g): the oracle depends on the skip of the masked cells", max_diff, limit);
            }
        } else if (!all_nonfinite(got.data(), g_n_vocab)) {
            fuzz::fail("the known limit does not occur: without flash attention, the NaN rows of a removed sequence do "
                       "not reach a new sequence. Correct the known limit in the header of fuzz_recurrent.cpp.");
        }
    }
    fprintf(stderr, "fuzz_recurrent: known limit: without flash attention, the NaN rows of a removed sequence reach a new "
                    "sequence. The flash attention of the CPU skips them.\n");
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int * /*argc*/, char *** /*argv*/) {
    fuzz::quiet_logs();
    llama_backend_init();
    g_trace     = fuzz::env_long("FUZZ_TRACE", 0) != 0;
    g_calibrate = fuzz::env_long("FUZZ_RECURRENT_CALIBRATE", 0) != 0;
    g_nonfinite = fuzz::env_long("FUZZ_NONFINITE", 0);
    const char * tol = getenv("FUZZ_RECURRENT_TOL");
    if (tol != nullptr) {
        g_tol = strtof(tol, nullptr);
    }
    const char * env = getenv("FUZZ_MODEL");
    const std::string path = env != nullptr ? std::string(env) : fuzz::data_file("tiny-qwen35-f32.gguf");
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent;
    // The CPU only, unless FUZZ_DEVICE names a device (HTP0 for example): a phone build registers the
    // Hexagon backend, and the default parameters give it all the layers.
    static ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    const char * dev_name = getenv("FUZZ_DEVICE");
    devs[0] = dev_name != nullptr ? ggml_backend_dev_by_name(dev_name) : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (devs[0] == nullptr) {
        fuzz::fail("no device %s", dev_name != nullptr ? dev_name : "of the type CPU");
    }
    mp.devices = devs;
    if (dev_name == nullptr) {
        mp.n_gpu_layers = 0;
    }
    g_model = llama_model_load_from_file(path.c_str(), mp);
    if (g_model == nullptr) {
        fuzz::fail("cannot load the model %s", path.c_str());
    }
    g_n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(g_model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = kCtx;
    cp.n_batch    = kCtx;
    cp.n_ubatch   = kCtx;
    cp.n_seq_max  = 1;
    cp.n_threads  = 1;
    cp.n_threads_batch = 1;
    cp.no_perf    = true;
    cp.type_k     = GGML_TYPE_F32;
    cp.type_v     = GGML_TYPE_F32;
    for (int f = 0; f < 2; ++f) {
        cp.flash_attn_type = f == 0 ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
        for (RefCtx & r : g_ref_sets[f]) {
            r.ctx = llama_init_from_model(g_model, cp);
            if (r.ctx == nullptr) {
                fuzz::fail("cannot create a reference context (flash attention %s)", f == 0 ? "off" : "on");
            }
        }
    }

    // The oracle is useful only when the logits depend on the history: a
    // change of one token 4 or 10 positions back must move the last logits far
    // outside the tolerance. A rollback reaches at most 5 tokens back.
    std::vector<llama_token> h1;
    for (int i = 0; i < 12; ++i) {
        h1.push_back((llama_token) ((i * 37 + 11) % g_n_vocab));
    }
    const std::vector<float> l1 = reference_logits(g_ref[0], h1);
    for (const int back : { 4, 10 }) {
        std::vector<llama_token> h2 = h1;
        h2[12 - back] = (h2[12 - back] + 101) % g_n_vocab;
        const std::vector<float> l2 = reference_logits(g_ref[1], h2);
        float max_ref = 0.0f, max_diff = 0.0f;
        for (int i = 0; i < g_n_vocab; ++i) {
            max_ref  = std::max(max_ref, std::fabs(l1[i]));
            max_diff = std::max(max_diff, std::fabs(l1[i] - l2[i]));
        }
        const float limit = g_tol * (1.0f + max_ref);
        fprintf(stderr, "fuzz_recurrent: a change of the token %d back moves the last logits by %g, the tolerance is %g\n",
                back, max_diff, limit);
        if (max_diff < 10.0f * limit && !g_calibrate) {
            fuzz::fail("the model is not sensitive enough to its history for the oracle (%g < 10 x %g). Make the weights larger.",
                       max_diff, limit);
        }
    }

    // the flash attention of a different device can read the masked cells, thus the probe is for the CPU
    if (dev_name == nullptr) {
        check_known_limit();
    }
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz::note_input(data, size);
    FuzzedDataProvider fdp(data, size);
    run(fdp);
    return 0;
}
