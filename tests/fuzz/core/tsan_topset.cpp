// tsan_topset: the top-set sampler (LLAMA_SAMPLER_TOPSET) of our patch from two threads under TSan.
//
// llama.cpp permits two contexts of one model in two threads. Each context has
// its own sampler, thus a sampler that keeps no shared state is safe there.
// The verify mode of the top set (LLAMA_SAMPLER_TOPSET=2) counts the equal
// and the different selections in function-static counters:
//   src/llama-sampler.cpp   llama_sampler_topset_verify   n_same, n_diff
//   common/sampling.cpp     common_sampler_sample          n_same, n_diff
//
// The harness sets LLAMA_SAMPLER_TOPSET before the first sample (the value is
// read one time). FUZZ_TOPSET_MODE gives it, the default is 2.
// FUZZ_TOPSET_KNOWN_COUNTERS=1 sets mode 1 (finding topset-counters), thus the
// campaign can look for other races.
//
// One input gives the sampler of each thread (llama_sampler_sample with the
// chain of the app, or common_sampler_sample with default common parameters),
// a temperature, and the tokens of a prompt. Each thread decodes the prompt on
// its own context and then samples and decodes 1 to 24 tokens.
//
// Properties:
//   P1  TSan: no data race.
//   P2  Each sampled token is in [0, n_vocab).

#include "fuzz_common.h"

#include "common.h"
#include "sampling.h"

#include <thread>
#include <vector>

namespace {

llama_model * g_model   = nullptr;
int32_t       g_n_vocab = 0;

/** The work of one thread: decode a prompt, then sample and decode n_gen tokens. */
void worker(std::vector<llama_token> prompt, int n_gen, bool use_common, float temp, uint32_t seed) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_seq_max = 1;
    cp.n_threads = 1;
    cp.n_threads_batch = 1;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(g_model, cp);
    if (ctx == nullptr) {
        fuzz::fail("cannot create a context");
    }

    llama_sampler * chain = nullptr;
    common_sampler * cs = nullptr;
    if (use_common) {
        common_params_sampling sp;
        sp.seed = seed;
        sp.temp = temp;
        sp.no_perf = true;
        cs = common_sampler_init(g_model, sp);
    } else {
        auto params = llama_sampler_chain_default_params();
        params.no_perf = true;
        chain = llama_sampler_chain_init(params);
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(g_n_vocab, 256, 1.0f, 0.0f, 1.5f));
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(20));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.8f, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    }

    llama_batch b = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
    if (llama_decode(ctx, b) != 0) {
        fuzz::fail("the prompt decode fails");
    }
    for (int i = 0; i < n_gen; ++i) {
        llama_token id = cs ? common_sampler_sample(cs, ctx, -1, false) : llama_sampler_sample(chain, ctx, -1);
        if (id < 0 || id >= g_n_vocab) {
            fuzz::fail("P2: the sampler gives the token %d", id);
        }
        if (cs) {
            common_sampler_accept(cs, id, true);
        }
        llama_batch one = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, one) != 0) {
            fuzz::fail("a decode of one token fails");
        }
    }
    if (cs) {
        common_sampler_free(cs);
    }
    if (chain) {
        llama_sampler_free(chain);
    }
    llama_free(ctx);
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int * /*argc*/, char *** /*argv*/) {
    fuzz::quiet_logs();
    const bool known = fuzz::env_long("FUZZ_TOPSET_KNOWN_COUNTERS", 0) != 0;
    const char * mode = getenv("FUZZ_TOPSET_MODE");
    setenv("LLAMA_SAMPLER_TOPSET", known ? "1" : (mode ? mode : "2"), 1);
    llama_backend_init();
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
    std::vector<std::thread> threads;
    const int n_threads = 2;
    for (int t = 0; t < n_threads; ++t) {
        const bool use_common = fdp.ConsumeBool();
        const float temp = fdp.ConsumeBool() ? 0.8f : fdp.ConsumeFloatingPointInRange<float>(0.05f, 2.0f);
        const uint32_t seed = fdp.ConsumeIntegral<uint32_t>();
        const int n_gen = fdp.ConsumeIntegralInRange<int>(1, 24);
        std::vector<llama_token> prompt;
        const int n_prompt = fdp.ConsumeIntegralInRange<int>(1, 16);
        for (int i = 0; i < n_prompt; ++i) {
            prompt.push_back(fdp.ConsumeIntegralInRange<llama_token>(0, g_n_vocab - 1));
        }
        threads.emplace_back(worker, prompt, n_gen, use_common, temp, seed);
    }
    for (auto & t : threads) {
        t.join();
    }
    return 0;
}
