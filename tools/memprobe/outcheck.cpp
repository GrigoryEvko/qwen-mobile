/**
 * outcheck: the output bytes of the decode paths of llama_jni.cpp for one value of
 * n_outputs_max, as hashes, thus two runs with different values can be compared line by line.
 *
 * The tool makes the target context as load_impl does (n_batch = n_ubatch = 1024, one
 * sequence, a unified KV cache, n_rs_seq 4 and the MTP draft context of setup_speculative
 * with --spec), then:
 *   1. decodes a prompt of wiki text in batches of 1024 with logits for the last token only
 *      (decode_text), and lets the draft context follow each batch (spec_follow),
 *   2. generates with greedy selection: one token for each step without --spec (plain_step),
 *      or the draft, the verify batch of 1 + draft rows and the rollback of spec_step,
 *   3. with --six, decodes one batch of 6 rows that all ask for logits.
 * Each logits vector, and with --spec the nextn embeddings of the target, go into FNV-1a 64
 * hashes that the tool prints.
 *
 *   outcheck -m MODEL -f TEXT [--outputs-max N] [--spec] [--six] [--grow-from N] [-p N_PROMPT] [-n N_STEPS] [-t T]
 *
 * --grow-from N makes the target context with n_ctx N, and after the prompt moves the sequence
 * into a context of 4096 (llama_state_seq_get_data, a new context, llama_state_seq_set_data),
 * as a context that grows with the conversation does. Without --spec only.
 *
 * Build: as kvkl in tools/memprobe/build-phone.sh, with -I TREE/src for llama-ext.h. The link
 * takes -lllama-common -lllama -lggml -lggml-base.
 *
 * It is a measurement tool, not part of the app.
 */

#include "common.h"
#include "llama-ext.h"
#include "llama.h"
#include "speculative.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

/** FNV-1a 64 over n bytes, continued from h. O(n). */
uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto * b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

/** The index of the largest logit. O(n_vocab). */
llama_token argmax(const float * logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, text_path;
    int n_outputs_max = 0, n_prompt = 1500, n_steps = 24, n_threads = 16, grow_from = 0;
    bool spec = false, six = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value of %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m") model_path = next();
        else if (a == "-f") text_path = next();
        else if (a == "--outputs-max") n_outputs_max = std::atoi(next().c_str());
        else if (a == "--spec") spec = true;
        else if (a == "--six") six = true;
        else if (a == "--grow-from") grow_from = std::atoi(next().c_str());
        else if (a == "-p") n_prompt = std::atoi(next().c_str());
        else if (a == "-n") n_steps = std::atoi(next().c_str());
        else if (a == "-t") n_threads = std::atoi(next().c_str());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || text_path.empty()) {
        fprintf(stderr, "usage: outcheck -m MODEL -f TEXT [--outputs-max N] [--spec] [--six] [-p N] [-n N] [-t T]\n");
        return 2;
    }
    constexpr int kBatch = 1024;
    constexpr int kDraftMax = 4;

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp     = spec;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (model == nullptr) { fprintf(stderr, "the model did not load\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = grow_from > 0 ? (uint32_t) grow_from : 4096;
    cp.n_batch         = kBatch;
    cp.n_ubatch        = kBatch;
    cp.n_seq_max       = 1;
    cp.kv_unified      = true;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    if (spec) cp.n_rs_seq = kDraftMax;
    if (n_outputs_max > 0) {
        cp.n_outputs_max         = (uint32_t) n_outputs_max;
        cp.n_outputs_max_per_seq = (uint32_t) n_outputs_max;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) { fprintf(stderr, "the context did not initialize\n"); return 1; }

    common_speculative_init_result_ptr spec_init;
    common_speculative * sp = nullptr;
    llama_context * ctx_dft = nullptr;
    if (spec) {
        common_params params;
        params.model.path                = model_path;
        params.n_ctx                     = (int32_t) llama_n_ctx(ctx);
        params.n_batch                   = kBatch;
        params.n_ubatch                  = kBatch;
        params.n_parallel                = 1;
        params.kv_unified                = true;
        params.no_perf                   = true;
        params.cpuparams.n_threads       = n_threads;
        params.cpuparams_batch.n_threads = n_threads;
        params.speculative.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max   = kDraftMax;
        common_params params_dft = common_base_params_to_speculative(params);
        spec_init = common_speculative_init_from_params(params_dft, model, ctx);
        ctx_dft = spec_init ? spec_init->context() : nullptr;
        if (ctx_dft == nullptr) { fprintf(stderr, "no draft context\n"); return 1; }
        params.speculative.draft.ctx_tgt = ctx;
        params.speculative.draft.ctx_dft = ctx_dft;
        sp = common_speculative_init(params.speculative, 1);
        if (sp == nullptr) { fprintf(stderr, "no speculative driver\n"); return 1; }
    }

    std::ifstream f(text_path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::vector<llama_token> prompt = common_tokenize(vocab, ss.str().substr(0, 200000), true, false);
    prompt.resize(n_prompt);

    // 1. The prompt, as decode_text with logits_last.
    llama_batch b = llama_batch_init(kBatch, 0, 1);
    for (int i = 0; i < n_prompt; i += kBatch) {
        const int count = std::min(kBatch, n_prompt - i);
        for (int j = 0; j < count; ++j) {
            b.token[j] = prompt[i + j]; b.pos[j] = i + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0;
            b.logits[j] = (i + j + 1 == n_prompt);
        }
        b.n_tokens = count;
        const int rc = llama_decode(ctx, b);
        if (rc != 0) { printf("RC prefill %d\n", rc); return 1; }
        if (spec) {
            const float * nx = llama_get_embeddings_nextn(ctx);
            printf("HASH prefill-nextn %d %016llx\n", i, (unsigned long long) fnv(nx, (size_t) count * n_embd * sizeof(float)));
            if (!common_speculative_process(sp, b)) { printf("RC process-prefill failed\n"); return 1; }
        }
    }
    const float * lg = llama_get_logits_ith(ctx, -1);
    printf("HASH prefill-logits %016llx\n", (unsigned long long) fnv(lg, (size_t) n_vocab * sizeof(float)));

    int n_past = n_prompt;
    llama_token id_last = argmax(lg, n_vocab);

    if (grow_from > 0 && !spec) {
        std::vector<uint8_t> st(llama_state_seq_get_size(ctx, 0));
        const size_t got = llama_state_seq_get_data(ctx, st.data(), st.size(), 0);
        llama_free(ctx);
        cp.n_ctx = 4096;
        ctx = llama_init_from_model(model, cp);
        if (ctx == nullptr || llama_state_seq_set_data(ctx, st.data(), got, 0) == 0) {
            fprintf(stderr, "the state did not move into the larger context\n");
            return 1;
        }
        // A state holds no logits: the first token came from the prompt logits of the smaller
        // context, before its release.
        printf("GROW from %d to %u, %zu bytes\n", grow_from, llama_n_ctx(ctx), got);
    }

    llama_tokens spec_prompt(prompt.begin(), prompt.end());
    if (spec) common_speculative_begin(sp, 0, spec_prompt);

    // 2. The answer.
    for (int step = 0; step < n_steps; ++step) {
        if (!spec) {
            b.n_tokens = 1;
            b.token[0] = id_last; b.pos[0] = n_past; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
            const int rc = llama_decode(ctx, b);
            if (rc != 0) { printf("RC decode %d\n", rc); return 1; }
            n_past += 1;
            lg = llama_get_logits_ith(ctx, -1);
            printf("HASH step %d token=%d logits=%016llx\n", step, id_last, (unsigned long long) fnv(lg, (size_t) n_vocab * sizeof(float)));
            id_last = argmax(lg, n_vocab);
            continue;
        }
        const int pos0 = n_past;
        llama_tokens draft;
        common_speculative_get_draft_params(sp, 0) = { true, kDraftMax, pos0, id_last, &spec_prompt, &draft };
        common_speculative_draft(sp);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, pos0, -1);
        b.n_tokens = 1 + (int) draft.size();
        for (int i = 0; i < b.n_tokens; ++i) {
            b.token[i] = i == 0 ? id_last : draft[i - 1];
            b.pos[i] = pos0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        const int rc = llama_decode(ctx, b);
        if (rc != 0) { printf("RC verify %d\n", rc); return 1; }
        uint64_t h = fnv(nullptr, 0);
        for (int i = 0; i < b.n_tokens; ++i) h = fnv(llama_get_logits_ith(ctx, i), (size_t) n_vocab * sizeof(float), h);
        const float * nx = llama_get_embeddings_nextn(ctx);
        const uint64_t hn = fnv(nx, (size_t) b.n_tokens * n_embd * sizeof(float));
        const bool followed = common_speculative_process(sp, b);
        std::vector<llama_token> out;
        size_t i = 0;
        for (; i < draft.size(); ++i) {
            const llama_token id = argmax(llama_get_logits_ith(ctx, (int) i), n_vocab);
            out.push_back(id);
            if (draft[i] != id) break;
        }
        if (i == draft.size()) out.push_back(argmax(llama_get_logits_ith(ctx, (int) i), n_vocab));
        const int accepted = (int) out.size() - 1;
        common_speculative_accept(sp, 0, (uint16_t) accepted);
        std::string dtext;
        for (llama_token t : draft) dtext += std::to_string(t) + ",";
        printf("HASH step %d rows=%d draft=[%s] accepted=%d logits=%016llx nextn=%016llx followed=%d\n", step, b.n_tokens,
               dtext.c_str(), accepted, (unsigned long long) h, (unsigned long long) hn, followed);
        spec_prompt.push_back(id_last);
        for (int k = 0; k < accepted; ++k) spec_prompt.push_back(out[k]);
        n_past = pos0 + 1 + accepted;
        id_last = out.back();
        if (n_past <= pos0 + (int) draft.size()) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1)) { printf("RC rollback failed\n"); return 1; }
        }
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_past, -1);
    }

    // 3. One batch of 6 rows with logits.
    if (six) {
        b.n_tokens = 6;
        for (int i = 0; i < 6; ++i) {
            b.token[i] = id_last; b.pos[i] = n_past + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        fflush(stdout);
        const int rc = llama_decode(ctx, b);
        printf("RC six %d\n", rc);
        if (rc == 0) {
            uint64_t h = fnv(nullptr, 0);
            for (int i = 0; i < 6; ++i) h = fnv(llama_get_logits_ith(ctx, i), (size_t) n_vocab * sizeof(float), h);
            printf("HASH six %016llx\n", (unsigned long long) h);
        }
    }
    fflush(stdout);
    llama_batch_free(b);
    if (sp) common_speculative_free(sp);
    spec_init.reset();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
