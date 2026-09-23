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
//
// A sequence becomes "tainted" after an operation that has a result that the
// shadow cannot predict (a corrupted blob that loads, a middle range remove).
// The oracle ignores a tainted sequence until the next clear.

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

/** The shadow of one sequence: its tokens, and whether the oracle can predict it. */
struct Shadow {
    std::vector<llama_token> hist;
    bool tainted = false;
    bool pending = false;  // a rollback is done and the next decode has not run yet
};

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
    p.threads   = (int) fuzz::env_long("FUZZ_THREADS", 0) > 0 ? (int) fuzz::env_long("FUZZ_THREADS", 0) : fdp.ConsumeIntegralInRange<int>(1, 2);
    p.embd_host = fdp.ConsumeBool();
    trace("ctx: n_seq_max %u n_rs_seq %u n_ubatch %u unified %d flash %d threads %d embd_host %d",
          p.n_seq_max, p.n_rs_seq, p.n_ubatch, p.unified, p.flash, p.threads, p.embd_host);

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
                    if (!s.tainted) {
                        const float * l = llama_get_logits_ith(ctx, last_idx[k]);
                        if (l == nullptr) {
                            fuzz::fail("llama_get_logits_ith(%d) gives null after a decode that requested it", last_idx[k]);
                        }
                        check_logits(l, s, seqs[k], "decode");
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
            // A copy into a sequence that holds tokens has two meanings in the hybrid memory: the
            // recurrent memory replaces the destination, and the unified KV cache adds the cells of
            // the source to the old cells of the destination. llama.h does not say that the
            // destination must be empty, and llama-server clears it first. The harness takes that
            // contract: FUZZ_RECURRENT_CLEAR_BEFORE_COPY=1 clears the destination before each copy.
            static const bool clear_first = fuzz::env_long("FUZZ_RECURRENT_CLEAR_BEFORE_COPY", 0) != 0;
            if (clear_first && src != dst) {
                drop_seq(mem, sh[dst], dst);
            }
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

}  // namespace

extern "C" int LLVMFuzzerInitialize(int * /*argc*/, char *** /*argv*/) {
    fuzz::quiet_logs();
    llama_backend_init();
    g_trace     = fuzz::env_long("FUZZ_TRACE", 0) != 0;
    g_calibrate = fuzz::env_long("FUZZ_RECURRENT_CALIBRATE", 0) != 0;
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
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz::note_input(data, size);
    FuzzedDataProvider fdp(data, size);
    run(fdp);
    return 0;
}
