// hexhost_logits_hash: a hash of the raw logits of the three real paths of the app on one device,
// for a bit-exact compare of two builds of a backend library (a KL file quantizes the logits).
//
//   hexhost_logits_hash MODEL.gguf DEVICE {decode N | prefill N | mtp N}
//
// decode N   a prompt of 16 tokens, then N greedy steps of one token each
// prefill N  a prompt of N tokens in one llama_decode, with the logits of each token
// mtp N      the MTP draft path of the app: 4 state snapshots, a prompt of 16 tokens, then N steps
//            of one draft of up to 4 tokens and the verify batch of the target
//
// Output: one line for each llama_decode of the target with the FNV-1a hash of its logits and the
// greedy token, and a last line with the hash of all lines. Two runs give the same last line only
// when every logit has the same bits. HEXHOST_LOGITS_DUMP=FILE also writes the raw logits of each
// llama_decode of the target to FILE (float32, one row after the other). tools/logitsdiff.py
// compares two dumps: the first row that differs, the maximum difference and the KL divergence.
// HEXHOST_LOGITS_DUMP_LAST=1 writes only the last row of each llama_decode. HEXHOST_PROMPT_LEN=P
// gives the decode and mtp modes a prompt of P tokens (16 when it is not set), thus the steps run
// at the positions P and up.
//
// The build for the phone, in the Snapdragon container, against a phone build of llama.cpp in
// LLAMA_BUILD (its bin directory holds libllama-common.so, libllama.so, libggml.so and
// libggml-base.so) and the same llama.cpp tree in LLAMA_DIR:
//
//   $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-linux-android34 \
//       -std=c++17 -O2 -I $LLAMA_DIR/include -I $LLAMA_DIR/ggml/include -I $LLAMA_DIR/common \
//       -I $LLAMA_DIR/vendor tests/fuzz/hexhost/tools/logits_hash.cpp -o hexhost_logits_hash \
//       -L $LLAMA_BUILD/bin -lllama-common -lllama -lggml -lggml-base

#include "common.h"
#include "llama.h"
#include "speculative.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// FNV-1a over a byte range. O(n).
uint64_t fnv1a(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ b[i]) * 1099511628211ull;
    }
    return h;
}

// A batch of tokens from position pos0 in sequence 0, each with logits.
llama_batch make_batch(const std::vector<llama_token> & toks, llama_pos pos0) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    b.n_tokens    = (int32_t) toks.size();
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[i]     = toks[i];
        b.pos[i]       = pos0 + (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 1;
    }
    return b;
}

// The argmax of one row of logits. O(n_vocab).
llama_token argmax(const float * row, int32_t n_vocab) {
    llama_token best = 0;
    for (int32_t i = 1; i < n_vocab; i++) {
        if (row[i] > row[best]) {
            best = i;
        }
    }
    return best;
}

uint64_t g_all  = 1469598103934665603ull;
FILE *   g_dump      = nullptr;  // HEXHOST_LOGITS_DUMP: the raw logits of each llama_decode
bool     g_dump_last = false;    // HEXHOST_LOGITS_DUMP_LAST: only the last row of each llama_decode

// Prints the hash of the logits of the last llama_decode (n rows), adds it to the total, and
// gives the greedy token of the last row. With HEXHOST_LOGITS_DUMP the rows also go to that file
// as float32, one row after the other, for a compare of two runs beyond the hash.
llama_token report(llama_context * ctx, const char * what, int step, int32_t n, int32_t n_vocab) {
    const float *  logits = llama_get_logits(ctx);
    const uint64_t h      = fnv1a(logits, (size_t) n * n_vocab * sizeof(float));
    if (g_dump && g_dump_last) {
        fwrite(logits + (size_t) (n - 1) * n_vocab, sizeof(float), (size_t) n_vocab, g_dump);
    } else if (g_dump) {
        fwrite(logits, sizeof(float), (size_t) n * n_vocab, g_dump);
    }
    const llama_token t   = argmax(logits + (size_t) (n - 1) * n_vocab, n_vocab);
    printf("%s %d rows %d hash %016" PRIx64 " token %d\n", what, step, n, h, t);
    g_all = fnv1a(&h, sizeof(h), g_all);
    return t;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s MODEL.gguf DEVICE {decode N | prefill N | mtp N}\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string mode       = argv[3];
    const int         n          = atoi(argv[4]);
    if ((mode != "decode" && mode != "prefill" && mode != "mtp") || n < 1) {
        fprintf(stderr, "hexhost_logits_hash: the mode must be decode, prefill or mtp with a positive count\n");
        return 2;
    }
    if (const char * path = getenv("HEXHOST_LOGITS_DUMP")) {
        g_dump = fopen(path, "wb");
        if (!g_dump) {
            fprintf(stderr, "hexhost_logits_hash: cannot write %s\n", path);
            return 2;
        }
    }
    g_dump_last = getenv("HEXHOST_LOGITS_DUMP_LAST") != nullptr;
    const char * plen    = getenv("HEXHOST_PROMPT_LEN");
    const int    n_first = plen ? atoi(plen) : 16;  // the prompt of the decode and mtp modes
    if (n_first < 1) {
        fprintf(stderr, "hexhost_logits_hash: HEXHOST_PROMPT_LEN must be a positive count\n");
        return 2;
    }
    llama_backend_init();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(argv[2]);
    if (!dev) {
        fprintf(stderr, "hexhost_logits_hash: no device %s\n", argv[2]);
        return 2;
    }
    llama_model_params mp      = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    mp.devices                 = devs;
    mp.n_gpu_layers            = 999;
    mp.load_mtp                = mode == "mtp";
    llama_model * model        = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "hexhost_logits_hash: cannot load %s\n", model_path.c_str());
        return 1;
    }
    // The prompt and the steps (up to 5 tokens each in the mtp mode) fit the context
    const int n_prompt_max  = mode == "prefill" ? n : n_first;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = (uint32_t) std::max(1024, n_prompt_max + 5 * n + 64);
    cp.n_batch              = (uint32_t) std::max(1024, n_prompt_max);
    cp.n_ubatch             = 1024;
    cp.n_seq_max            = 1;
    cp.kv_unified           = true;
    cp.n_rs_seq             = mode == "mtp" ? 4 : 0;
    llama_context * ctx     = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "hexhost_logits_hash: cannot make a context\n");
        llama_model_free(model);
        return 1;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int     n_prompt = n_prompt_max;
    std::vector<llama_token> prompt(n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        prompt[i] = (llama_token) ((1000 + 7 * i) % n_vocab);
    }

    int rc = 0;
    if (mode == "decode" || mode == "prefill") {
        llama_batch b = make_batch(prompt, 0);
        rc            = llama_decode(ctx, b);
        llama_batch_free(b);
        llama_token t = rc == 0 ? report(ctx, "prompt", 0, n_prompt, n_vocab) : 0;
        for (int s = 1; mode == "decode" && s <= n && rc == 0; s++) {
            llama_batch b1 = make_batch({ t }, n_prompt + s - 1);
            rc             = llama_decode(ctx, b1);
            llama_batch_free(b1);
            if (rc == 0) {
                t = report(ctx, "step", s, 1, n_vocab);
            }
        }
    } else {
        common_params params;
        params.model.path              = model_path;
        params.n_ctx                   = (int32_t) llama_n_ctx(ctx);
        params.n_batch                 = (int32_t) llama_n_batch(ctx);
        params.n_ubatch                = (int32_t) llama_n_ubatch(ctx);
        params.n_parallel              = 1;
        params.kv_unified              = true;
        params.speculative.types       = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = 4;
        common_params params_dft       = common_base_params_to_speculative(params);
        auto            init           = common_speculative_init_from_params(params_dft, model, ctx);
        llama_context * ctx_dft        = init ? init->context() : nullptr;
        common_speculative * spec      = nullptr;
        if (ctx_dft) {
            params.speculative.draft.ctx_tgt = ctx;
            params.speculative.draft.ctx_dft = ctx_dft;
            spec                             = common_speculative_init(params.speculative, 1);
        }
        if (!spec) {
            fprintf(stderr, "hexhost_logits_hash: the MTP draft did not initialize (does the model hold the MTP block?)\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        llama_batch b = make_batch(prompt, 0);
        rc            = llama_decode(ctx, b);
        if (rc == 0) {
            common_speculative_process(spec, b);
        }
        llama_batch_free(b);
        llama_token  last = rc == 0 ? report(ctx, "prompt", 0, n_prompt, n_vocab) : 0;
        llama_pos    pos  = n_prompt;
        llama_tokens seq  = prompt;
        common_speculative_begin(spec, 0, seq);
        for (int s = 1; s <= n && rc == 0; s++) {
            llama_tokens draft;
            common_speculative_get_draft_params(spec, 0) = { true, 4, pos, last, &seq, &draft };
            common_speculative_draft(spec);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, pos, -1);
            std::vector<llama_token> verify = { last };
            verify.insert(verify.end(), draft.begin(), draft.end());
            llama_batch bv = make_batch(verify, pos);
            rc             = llama_decode(ctx, bv);
            if (rc != 0) {
                llama_batch_free(bv);
                break;
            }
            common_speculative_process(spec, bv);
            report(ctx, "verify", s, (int32_t) verify.size(), n_vocab);
            // Greedy acceptance: each draft token that equals the argmax of the row before it
            const float * logits   = llama_get_logits(ctx);
            int           accepted = 0;
            llama_token   next     = argmax(logits, n_vocab);
            while (accepted < (int) draft.size() && draft[accepted] == next) {
                accepted++;
                next = argmax(logits + (size_t) accepted * n_vocab, n_vocab);
            }
            llama_batch_free(bv);
            common_speculative_accept(spec, 0, (uint16_t) accepted);
            // The target keeps the token of the step and the accepted drafts. As in the app, one
            // call rolls back the rejected drafts (a recurrent rollback is single use).
            if (accepted < (int) draft.size() && !llama_memory_seq_rm(llama_get_memory(ctx), 0, pos + 1 + accepted, -1)) {
                fprintf(stderr, "hexhost_logits_hash: the rollback to position %d failed\n", (int) (pos + 1 + accepted));
                rc = 1;
            }
            seq.push_back(last);
            seq.insert(seq.end(), draft.begin(), draft.begin() + accepted);
            pos += 1 + accepted;
            last = next;
        }
        common_speculative_free(spec);
    }
    printf("total %016" PRIx64 " rc %d\n", g_all, rc);
    if (g_dump) {
        fclose(g_dump);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc == 0 ? 0 : 1;
}
