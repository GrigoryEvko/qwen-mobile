// tsan_decode: llama_decode with the ggml thread pool under TSan, with concurrent callers.
//
// The harness loads the tiny qwen35 model one time. One input gives the
// thread count of the decode (2 to 4), a program of decodes and rollbacks as
// the app runs them (a prompt, then verify steps of 1 to n_rs_seq + 1 tokens,
// each followed by a rollback of the rejected tokens), and the work of the
// other threads, which run while the decodes run:
//
//   stopper   sets an atomic flag, and the abort callback of the context
//             reads it, as requestStop() of the app does from the UI thread
//   info      reads what modelInfo() of the app reads without the engine
//             mutex: llama_model_desc, llama_model_size, llama_model_n_params,
//             llama_n_ctx, llama_vocab_n_tokens
//   second    a second context on the same model decodes in its own thread,
//             as llama.cpp permits (tests/test-thread-safety.cpp)
//   state     (only with FUZZ_TSAN_STATE_READS=1) reads the state of the
//             context under decode: llama_state_seq_get_size and
//             llama_memory_seq_pos_max. llama.h does not permit a call on a
//             context from two threads, thus a report in this mode shows a
//             use outside the API contract, not a defect.
//
// Properties:
//   P1  TSan: no data race, no lock order inversion, no thread leak.
//   P2  Each decode returns 0, or 2 after an abort that the stopper requested.

#include "fuzz_common.h"

#include <atomic>
#include <thread>
#include <vector>

namespace {

llama_model * g_model   = nullptr;
int32_t       g_n_vocab = 0;
bool          g_state_reads = false;

/** The abort callback: true when the flag is set. The decode thread calls it between graph nodes. */
bool abort_cb(void * data) {
    return static_cast<std::atomic<bool> *>(data)->load(std::memory_order_relaxed);
}

/** Create a context with n_threads threads and n_rs_seq rollback slots. */
llama_context * make_ctx(int n_threads, uint32_t n_rs_seq) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 256;
    cp.n_batch         = 64;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = 1;
    cp.n_rs_seq        = n_rs_seq;
    cp.kv_unified      = true;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.no_perf         = true;
    return llama_init_from_model(g_model, cp);
}

/** Decode n tokens of seq 0 from position pos. Returns the code of llama_decode. */
int decode(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = pos + (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 1;
    }
    b.n_tokens = (int32_t) toks.size();
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

/** One program: a prompt, then verify steps with rollbacks. Stops at the first failed decode. */
void program(llama_context * ctx, const std::vector<uint8_t> & prog, uint32_t n_rs_seq, std::atomic<bool> * stop) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_pos pos = 0;
    size_t i = 0;
    std::vector<llama_token> toks;
    const int n_prompt = prog.empty() ? 4 : 1 + prog[i++] % 12;
    for (int t = 0; t < n_prompt; ++t) {
        toks.push_back((llama_token) (i < prog.size() ? prog[i++] : t) % g_n_vocab);
    }
    int rc = decode(ctx, toks, pos);
    if (rc != 0) {
        if (rc != 2 || stop == nullptr || !stop->load()) {
            fuzz::fail("P2: the prompt decode returns %d", rc);
        }
        return;
    }
    pos += (llama_pos) toks.size();
    while (i + 1 < prog.size() && pos < 200) {
        const int n = 1 + prog[i++] % (n_rs_seq + 1);
        toks.assign(n, (llama_token) (prog[i++] % g_n_vocab));
        rc = decode(ctx, toks, pos);
        if (rc != 0) {
            if (rc != 2 || stop == nullptr || !stop->load()) {
                fuzz::fail("P2: a verify decode of %d tokens returns %d", n, rc);
            }
            return;
        }
        const int keep = 1 + (i < prog.size() ? prog[i++] : 0) % n;
        pos += keep;
        if (keep < n && !llama_memory_seq_rm(llama_get_memory(ctx), 0, pos, -1)) {
            // a refused rollback: replay from the start, as the app restores a snapshot
            llama_memory_clear(llama_get_memory(ctx), true);
            pos = 0;
        }
    }
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int * /*argc*/, char *** /*argv*/) {
    fuzz::quiet_logs();
    llama_backend_init();
    g_state_reads = fuzz::env_long("FUZZ_TSAN_STATE_READS", 0) != 0;
    const char * env = getenv("FUZZ_MODEL");
    const std::string path = env != nullptr ? std::string(env) : fuzz::data_file("tiny-qwen35-f32.gguf");
    g_model = llama_model_load_from_file(path.c_str(), llama_model_default_params());
    if (g_model == nullptr) {
        fuzz::fail("cannot load the model %s", path.c_str());
    }
    g_n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(g_model));
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz::note_input(data, size);
    FuzzedDataProvider fdp(data, size);
    const int      n_threads = fdp.ConsumeIntegralInRange<int>(2, 4);
    const uint32_t n_rs_seq  = fdp.ConsumeIntegralInRange<uint32_t>(0, 4);
    const bool     stopper   = fdp.ConsumeBool();
    const bool     info      = fdp.ConsumeBool();
    const bool     second    = fdp.ConsumeBool();
    const std::vector<uint8_t> prog  = fdp.ConsumeBytes<uint8_t>(fdp.ConsumeIntegralInRange<size_t>(0, 64));
    const std::vector<uint8_t> prog2 = fdp.ConsumeBytes<uint8_t>(fdp.ConsumeIntegralInRange<size_t>(0, 32));

    llama_context * ctx = make_ctx(n_threads, n_rs_seq);
    if (ctx == nullptr) {
        fuzz::fail("cannot create the context");
    }
    std::atomic<bool> stop{false};
    llama_set_abort_callback(ctx, abort_cb, &stop);

    std::atomic<bool> done{false};
    std::vector<std::thread> others;
    if (stopper) {
        const int delay_us = fdp.ConsumeIntegralInRange<int>(0, 3000);
        others.emplace_back([&stop, delay_us] {
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
            stop.store(true);
        });
    }
    if (info) {
        others.emplace_back([&done, ctx] {
            char desc[128];
            while (!done.load()) {
                llama_model_desc(g_model, desc, sizeof(desc));
                (void) llama_model_size(g_model);
                (void) llama_model_n_params(g_model);
                (void) llama_n_ctx(ctx);
                (void) llama_vocab_n_tokens(llama_model_get_vocab(g_model));
                std::this_thread::yield();
            }
        });
    }
    if (g_state_reads) {
        others.emplace_back([&done, ctx] {
            while (!done.load()) {
                (void) llama_state_seq_get_size(ctx, 0);
                (void) llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
                std::this_thread::yield();
            }
        });
    }
    llama_context * ctx2 = nullptr;
    if (second) {
        ctx2 = make_ctx(2, n_rs_seq);
        if (ctx2 == nullptr) {
            fuzz::fail("cannot create the second context");
        }
        others.emplace_back([ctx2, &prog2, n_rs_seq] { program(ctx2, prog2, n_rs_seq, nullptr); });
    }

    program(ctx, prog, n_rs_seq, &stop);
    done.store(true);
    for (auto & t : others) {
        t.join();
    }
    if (ctx2 != nullptr) {
        llama_free(ctx2);
    }
    llama_free(ctx);
    return 0;
}
