// actrange: the largest magnitudes of the activations that a matmul of the Qwen3.5 model reads or writes, on the
// CPU backend of the host with real text. The question: can an F16 copy of the activation (the F16 SwiGLU output
// of the matmul fusions, the F16 activation load and the F16 output tiles of the HMX) hold each value? The largest
// finite F16 value is 65504.
//
// The tensors (the names of src/models/qwen35.cpp and of build_ffn in src/llama-graph.cpp):
//   ffn_swiglu    the SwiGLU output h, the input of the ffn_down MUL_MAT
//   ffn_down      the output of the ffn_down MUL_MAT (an HMX output tile holds it as F16)
//   final_output  the gated norm of a delta net layer, the input of the ssm_out MUL_MAT
//   attn_gated    the gated attention output, the input of the attn_output MUL_MAT
// and each tensor that --also PREFIX names.
//
// Usage: actrange -m MODEL [--ctx N] [--chunks N] [--threads N] [--special] [--also PREFIX] [--above LIST] FILE...
//   Each FILE gives the first N chunks (preset 2) of N tokens (preset 1024). Each chunk is a new sequence, thus
//   its first token is at position 0. --special parses the special tokens of the text (a chat template).
//   --above gives the limits of the counts, comma-separated (preset 32768,65504).
// Output: for each tensor name without the layer, the largest magnitude, the smallest and the largest value, the
// layer, the file and the token of the largest magnitude, and the count of values with a magnitude above each
// limit. Then the largest magnitude of each layer.
// Time: about 1 minute for 3 files of 2 chunks with 32 threads. O(tokens * parameters).

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct stat_t {
    double              max_abs  = 0.0;
    double              max_pos  = 0.0;  // the largest value
    double              min_neg  = 0.0;  // the smallest value
    int                 max_tok  = -1;   // the token in the chunk of the largest magnitude
    std::string         max_file;
    size_t              n        = 0;
    std::vector<size_t> n_above;         // the count of the values with a magnitude above each limit
    size_t              n_nonfinite = 0;
};

struct state_t {
    std::vector<std::string>      prefixes;
    std::vector<double>           limits = { 32768.0, 65504.0 };
    std::string                   file;
    std::map<std::string, stat_t> by_name;  // the full name, with the layer
    std::vector<float>            buf;
};

bool wanted(const state_t & st, const char * name) {
    for (const std::string & p : st.prefixes) {
        if (!strncmp(name, p.c_str(), p.size()) && name[p.size()] == '-') {
            return true;
        }
    }
    return false;
}

// The eval callback: ask gives the tensors to keep, the second call reads the values. O(elements).
bool on_tensor(ggml_tensor * t, bool ask, void * user) {
    auto * st = (state_t *) user;
    if (ask) {
        return wanted(*st, t->name);
    }
    if (!wanted(*st, t->name) || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        return true;
    }
    const size_t n = (size_t) ggml_nelements(t);
    st->buf.resize(n);
    ggml_backend_tensor_get(t, st->buf.data(), 0, n * sizeof(float));
    stat_t &      s   = st->by_name[t->name];
    const int64_t row = t->ne[0];
    s.n_above.resize(st->limits.size(), 0);
    for (size_t i = 0; i < n; i++) {
        const float v = st->buf[i];
        s.n++;
        if (!std::isfinite(v)) {
            s.n_nonfinite++;
            continue;
        }
        const double a = std::fabs((double) v);
        for (size_t l = 0; l < st->limits.size(); l++) {
            s.n_above[l] += a > st->limits[l];
        }
        s.max_pos = std::max(s.max_pos, (double) v);
        s.min_neg = std::min(s.min_neg, (double) v);
        if (a > s.max_abs) {
            s.max_abs  = a;
            s.max_tok  = (int) (i / (size_t) row);
            s.max_file = st->file;
        }
    }
    return true;
}

std::string read_file(const std::string & path) {
    std::ifstream     f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

int main(int argc, char ** argv) {
    std::string              model_path;
    int                      n_ctx   = 1024;
    int                      chunks  = 2;
    int                      threads = 32;
    bool                     special = false;
    state_t                  st;
    std::vector<std::string> files;
    st.prefixes = { "ffn_swiglu", "ffn_down", "final_output", "attn_gated" };
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "--ctx" && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else if (a == "--chunks" && i + 1 < argc) {
            chunks = atoi(argv[++i]);
        } else if (a == "--threads" && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (a == "--special") {
            special = true;
        } else if (a == "--also" && i + 1 < argc) {
            st.prefixes.push_back(argv[++i]);
        } else if (a == "--above" && i + 1 < argc) {
            st.limits.clear();
            std::stringstream ss(argv[++i]);
            for (std::string item; std::getline(ss, item, ',');) {
                st.limits.push_back(atof(item.c_str()));
            }
        } else if (!a.empty() && a[0] != '-') {
            files.push_back(a);
        } else {
            fprintf(stderr, "usage: actrange -m MODEL [--ctx N] [--chunks N] [--threads N] [--special] "
                            "[--also PREFIX] [--above L1,L2,...] FILE...\n");
            return 2;
        }
    }
    if (model_path.empty() || files.empty()) {
        fprintf(stderr, "actrange: give a model (-m) and at least one text file\n");
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers       = 0;
    llama_model * model   = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "actrange: the model %s does not load\n", model_path.c_str());
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = n_ctx;
    cp.n_batch              = n_ctx;
    cp.n_ubatch             = n_ctx;
    cp.n_threads            = threads;
    cp.n_threads_batch      = threads;
    cp.cb_eval              = on_tensor;
    cp.cb_eval_user_data    = &st;
    llama_context * ctx     = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "actrange: the context does not start\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    for (const std::string & path : files) {
        const std::string  text = read_file(path);
        std::vector<llama_token> toks(text.size() + 16);
        const int n_tok = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(),
                                         false, special);
        if (n_tok <= 0) {
            fprintf(stderr, "actrange: %s gives no tokens\n", path.c_str());
            continue;
        }
        toks.resize(n_tok);
        st.file = path.substr(path.find_last_of('/') + 1);
        for (int c = 0; c < chunks && (size_t) (c + 1) * n_ctx <= toks.size(); c++) {
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_batch batch = llama_batch_get_one(toks.data() + (size_t) c * n_ctx, n_ctx);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "actrange: the decode of chunk %d of %s failed\n", c, path.c_str());
                return 1;
            }
            fprintf(stderr, "actrange: %s chunk %d done\n", st.file.c_str(), c);
        }
    }

    // The summary for each name without the layer, then each layer
    std::map<std::string, stat_t> by_kind;
    for (const auto & [name, s] : st.by_name) {
        const std::string kind = name.substr(0, name.find_last_of('-'));
        stat_t &          k    = by_kind[kind];
        k.n_above.resize(st.limits.size(), 0);
        k.n += s.n;
        for (size_t l = 0; l < st.limits.size() && l < s.n_above.size(); l++) {
            k.n_above[l] += s.n_above[l];
        }
        k.n_nonfinite += s.n_nonfinite;
        k.max_pos = std::max(k.max_pos, s.max_pos);
        k.min_neg = std::min(k.min_neg, s.min_neg);
        if (s.max_abs > k.max_abs) {
            k.max_abs  = s.max_abs;
            k.max_tok  = s.max_tok;
            k.max_file = name + " " + s.max_file;
        }
    }
    for (const auto & [kind, k] : by_kind) {
        std::string above;
        for (size_t l = 0; l < st.limits.size(); l++) {
            char t[64];
            snprintf(t, sizeof(t), ", above %g: %zu", st.limits[l], l < k.n_above.size() ? k.n_above[l] : 0);
            above += t;
        }
        printf("actrange %-14s max=%.6g (from %.6g to %.6g) at %s token %d%s, nonfinite: %zu, of %zu\n", kind.c_str(),
               k.max_abs, k.min_neg, k.max_pos, k.max_file.c_str(), k.max_tok, above.c_str(), k.n_nonfinite, k.n);
    }
    for (const auto & [kind, k] : by_kind) {
        std::vector<std::pair<int, double>> layers;
        for (const auto & [name, s] : st.by_name) {
            if (name.compare(0, kind.size() + 1, kind + "-") == 0) {
                layers.emplace_back(atoi(name.c_str() + kind.size() + 1), s.max_abs);
            }
        }
        std::sort(layers.begin(), layers.end());
        printf("actrange %-14s per layer:", kind.c_str());
        for (const auto & [il, m] : layers) {
            printf(" %d:%.4g", il, m);
        }
        printf("\n");
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
