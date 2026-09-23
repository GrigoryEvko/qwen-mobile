// fuzz_npu_decode: the decode path of the app on a device backend (HTP0 on the phone),
// against the CPU backend, with the same calls in lockstep.
//
// The harness loads one model two times: on the device of FUZZ_DEVICE (for
// example HTP0, or the CPU without FUZZ_DEVICE, to do a check of the harness) with
// all layers offloaded, and on the CPU. It makes one context on each, with the
// parameters of the app (llama_jni.cpp load_impl): n_seq_max 1, kv_unified,
// n_rs_seq 4 (the speculative depth), n_batch = n_ubatch = FUZZ_NPU_UBATCH
// (default 1024). It sets GGML_HEXAGON_OPFUSION=1 and
// GGML_HEXAGON_OPFUSION_STATE=1 as the app does, unless the environment sets
// them. The DSP code has no sanitizer, thus the harness checks the results.
//
// One input is a program of the operations of the app:
//   prompt   decode 1 to 48 tokens, logits of the last one
//   verify   decode 1 to 5 tokens with all logits (a speculative verify
//            step), then roll back 0 to 4 of them with llama_memory_seq_rm
//   restore  take the state of sequence 0 from one context and load it into
//            the same context or into the other one (the app moves the bytes
//            of the prefill context to the decode context)
//   clear    llama_memory_clear on the two contexts
//
// Properties:
//   P1  No crash and no sanitizer report on the host side.
//   P2  Each decode on the device returns what the decode on the CPU returns.
//   P3  The device logits are finite.
//   P4  The device logits are near the CPU logits:
//       max |dev - cpu| <= FUZZ_NPU_TOL * (1 + max |cpu|). The default
//       tolerance 0.05 is for the HTP with F16 products. FUZZ_NPU_CALIBRATE=1
//       prints each difference and does not fail.
//   P5  A rollback gives the same answer (done or refused) on the two backends.
//   P6  The log of llama.cpp and ggml shows no new error line (the health of
//       the device session).
//
// The run prints a summary line at exit: the count of checks and the largest
// difference.

#include "fuzz_common.h"

#include <cinttypes>
#include <cmath>
#include <string>
#include <vector>

namespace {

llama_model *   g_dev_model = nullptr;
llama_model *   g_cpu_model = nullptr;
llama_context * g_dev = nullptr;
llama_context * g_cpu = nullptr;
int32_t         g_n_vocab = 0;
float           g_tol = 0.05f;
bool            g_calibrate = false;
long            g_checks = 0;
float           g_worst  = 0.0f;
constexpr uint32_t kRsSeq = 4;
constexpr int      kCtx   = 512;

/** Print the summary line. atexit() calls it. */
void summary() {
    fprintf(stderr, "fuzz_npu_decode: %ld logit checks, largest relative difference %g, %ld error lines in the log\n",
            g_checks, g_worst, fuzz::log_error_count().load());
}

/** Decode toks at pos on ctx, with logits for all tokens or for the last one. */
int decode(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos, bool all_logits) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = pos + (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = all_logits || i + 1 == toks.size();
    }
    b.n_tokens = (int32_t) toks.size();
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

/** P3 and P4 for the output i of the two contexts. */
void compare(int i, const char * what) {
    const float * d = llama_get_logits_ith(g_dev, i);
    const float * c = llama_get_logits_ith(g_cpu, i);
    if (d == nullptr || c == nullptr) {
        fuzz::fail("no logits for output %d after %s", i, what);
    }
    float max_c = 0.0f, max_diff = 0.0f;
    for (int j = 0; j < g_n_vocab; ++j) {
        if (!std::isfinite(d[j])) {
            fuzz::fail("P3: device logit %d of output %d is %g after %s", j, i, d[j], what);
        }
        max_c = std::max(max_c, std::fabs(c[j]));
        max_diff = std::max(max_diff, std::fabs(d[j] - c[j]));
    }
    const float rel = max_diff / (1.0f + max_c);
    g_checks++;
    g_worst = std::max(g_worst, rel);
    if (g_calibrate) {
        fprintf(stderr, "calibrate: %s output %d relative difference %g\n", what, i, rel);
        return;
    }
    if (!(rel <= g_tol)) {
        fuzz::fail("P4: after %s, output %d differs from the CPU by %g (max |cpu| %g, tolerance %g)", what, i, rel, max_c, g_tol);
    }
}

/** Move the state of sequence 0 from src to dst through a blob, as the app does. Returns false when the load fails. */
bool move_state(llama_context * src, llama_context * dst) {
    const size_t size = llama_state_seq_get_size(src, 0);
    std::vector<uint8_t> blob(size);
    const size_t got = llama_state_seq_get_data(src, blob.data(), size, 0);
    if (got == 0 || got > size) {
        fuzz::fail("llama_state_seq_get_data returns %zu for %zu bytes", got, size);
    }
    llama_memory_seq_rm(llama_get_memory(dst), 0, -1, -1);
    return llama_state_seq_set_data(dst, blob.data(), got, 0) != 0;
}

/** A silent progress callback. */
bool silent(float /*progress*/, void * /*user_data*/) {
    return true;
}

/** Load the model on one device (null: the CPU), with all layers on it. */
llama_model * load(const std::string & path, ggml_backend_dev_t dev) {
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent;
    static ggml_backend_dev_t devs_dev[2];
    static ggml_backend_dev_t devs_cpu[2];
    ggml_backend_dev_t * devs = dev ? devs_dev : devs_cpu;
    devs[0] = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    devs[1] = nullptr;
    mp.devices = devs;
    mp.n_gpu_layers = dev ? 999 : 0;
    return llama_model_load_from_file(path.c_str(), mp);
}

/** A context with the parameters of the app. */
llama_context * make_ctx(llama_model * model, int n_threads) {
    const uint32_t ub = (uint32_t) fuzz::env_long("FUZZ_NPU_UBATCH", 1024);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = kCtx;
    cp.n_batch         = ub;
    cp.n_ubatch        = ub;
    cp.n_seq_max       = 1;
    cp.kv_unified      = true;
    cp.n_rs_seq        = kRsSeq;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.no_perf         = true;
    return llama_init_from_model(model, cp);
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int * /*argc*/, char *** /*argv*/) {
    fuzz::quiet_logs();
    setenv("GGML_HEXAGON_OPFUSION", "1", 0);
    setenv("GGML_HEXAGON_OPFUSION_STATE", "1", 0);
    llama_backend_init();
    g_calibrate = fuzz::env_long("FUZZ_NPU_CALIBRATE", 0) != 0;
    if (const char * t = getenv("FUZZ_NPU_TOL")) {
        g_tol = strtof(t, nullptr);
    }
    const char * env = getenv("FUZZ_MODEL");
    const std::string path = env != nullptr ? std::string(env) : fuzz::data_file("tiny-qwen35-q8_0.gguf");

    ggml_backend_dev_t dev = nullptr;
    if (const char * name = getenv("FUZZ_DEVICE")) {
        dev = ggml_backend_dev_by_name(name);
        if (dev == nullptr) {
            std::string names;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                names += std::string(" ") + ggml_backend_dev_name(ggml_backend_dev_get(i));
            }
            fuzz::fail("the device %s does not exist. The devices are:%s", name, names.c_str());
        }
    }
    g_dev_model = load(path, dev);
    g_cpu_model = load(path, nullptr);
    if (g_dev_model == nullptr || g_cpu_model == nullptr) {
        fuzz::fail("cannot load the model %s", path.c_str());
    }
    g_n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(g_cpu_model));
    const int n_threads = (int) fuzz::env_long("FUZZ_THREADS", 4);
    g_dev = make_ctx(g_dev_model, n_threads);
    g_cpu = make_ctx(g_cpu_model, n_threads);
    if (g_dev == nullptr || g_cpu == nullptr) {
        fuzz::fail("cannot create the contexts");
    }
    fprintf(stderr, "fuzz_npu_decode: model %s, device %s, n_vocab %d, tolerance %g\n",
            path.c_str(), dev ? ggml_backend_dev_name(dev) : "CPU", g_n_vocab, g_tol);
    atexit(summary);
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz::note_input(data, size);
    FuzzedDataProvider fdp(data, size);
    const long errors0 = fuzz::log_error_count().load();
    llama_memory_clear(llama_get_memory(g_dev), true);
    llama_memory_clear(llama_get_memory(g_cpu), true);
    llama_pos pos = 0;

    const int n_ops = fdp.ConsumeIntegralInRange<int>(1, 12);
    for (int op_i = 0; op_i < n_ops && fdp.remaining_bytes() > 0; ++op_i) {
        const int op = fdp.ConsumeIntegralInRange<int>(0, 9);
        if (op <= 2 || pos == 0) {
            const int n = fdp.ConsumeIntegralInRange<int>(1, 48);
            if (pos + n >= kCtx) {
                break;
            }
            std::vector<llama_token> toks;
            for (int i = 0; i < n; ++i) {
                toks.push_back(fdp.ConsumeIntegralInRange<llama_token>(0, g_n_vocab - 1));
            }
            const int rd = decode(g_dev, toks, pos, false);
            const int rc = decode(g_cpu, toks, pos, false);
            if (rd != rc) {
                fuzz::fail("P2: a prompt of %d tokens at %d returns %d on the device and %d on the CPU", n, pos, rd, rc);
            }
            if (rc != 0) {
                break;
            }
            compare(-1, "prompt");
            pos += n;
        } else if (op <= 7) {
            const int n = fdp.ConsumeIntegralInRange<int>(1, (int) kRsSeq + 1);
            if (pos + n >= kCtx) {
                break;
            }
            std::vector<llama_token> toks;
            for (int i = 0; i < n; ++i) {
                toks.push_back(fdp.ConsumeIntegralInRange<llama_token>(0, g_n_vocab - 1));
            }
            const int rd = decode(g_dev, toks, pos, true);
            const int rc = decode(g_cpu, toks, pos, true);
            if (rd != rc) {
                fuzz::fail("P2: a verify of %d tokens at %d returns %d on the device and %d on the CPU", n, pos, rd, rc);
            }
            if (rc != 0) {
                break;
            }
            for (int i = 0; i < n; ++i) {
                compare(i, "verify");
            }
            const int keep = fdp.ConsumeIntegralInRange<int>(1, n);
            pos += keep;
            if (keep < n) {
                const bool od = llama_memory_seq_rm(llama_get_memory(g_dev), 0, pos, -1);
                const bool oc = llama_memory_seq_rm(llama_get_memory(g_cpu), 0, pos, -1);
                if (od != oc) {
                    fuzz::fail("P5: the rollback to %d is %s on the device and %s on the CPU", pos, od ? "done" : "refused", oc ? "done" : "refused");
                }
                if (!od) {
                    llama_memory_clear(llama_get_memory(g_dev), true);
                    llama_memory_clear(llama_get_memory(g_cpu), true);
                    pos = 0;
                }
            }
        } else if (op == 8) {
            // the snapshot store: device to device, device to CPU, or CPU to device
            const int dir = fdp.ConsumeIntegralInRange<int>(0, 2);
            bool ok;
            if (dir == 0) {
                ok = move_state(g_dev, g_dev);
            } else if (dir == 1) {
                // the CPU context takes the state of the device, and the device keeps its own
                ok = move_state(g_dev, g_cpu);
            } else {
                ok = move_state(g_cpu, g_dev);
            }
            if (!ok) {
                fuzz::fail("an intact state blob does not load (direction %d)", dir);
            }
        } else {
            llama_memory_clear(llama_get_memory(g_dev), true);
            llama_memory_clear(llama_get_memory(g_cpu), true);
            pos = 0;
        }
        const llama_pos pd = llama_memory_seq_pos_max(llama_get_memory(g_dev), 0);
        const llama_pos pc = llama_memory_seq_pos_max(llama_get_memory(g_cpu), 0);
        if (pd != pc || pc != pos - 1) {
            fuzz::fail("seq_pos_max is %d on the device and %d on the CPU, the harness expects %d", pd, pc, pos - 1);
        }
    }

    const long errors1 = fuzz::log_error_count().load();
    if (errors1 != errors0) {
        fuzz::fail("P6: the log shows %ld new error lines in this input. Run with FUZZ_VERBOSE=1 to see them.", errors1 - errors0);
    }
    return 0;
}
