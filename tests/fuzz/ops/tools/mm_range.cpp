// mm_range: the range of the values at each matrix product of a model over a text.
//
// The graph callback of llama.cpp stops at each MUL_MAT and MUL_MAT_ID node of the target model and
// of the MTP draft context. For each node it records the largest |value| of the output (dst) and of
// the activation input (src1), and the counts of |value| > 32768 and > 65504, keyed by the name of
// the weight (src0) and by the context (target or mtp).
//
// Usage: mm_range -m MODEL -f TEXT -c N_CTX -b B -ub B [--spec-type draft-mtp] [-t N]
//                 --mm-out TSV [--mm-tokens N] [--mm-batch N]
//   The text is tokenized and decoded in batches of N tokens (--mm-batch, or B when it is not given).
//   With MTP, the recurrent memory keeps the last draft-n-max + 1 rows of a batch in one ubatch, thus
//   a small batch needs a large -ub, not a small one.
//   All rows of a batch give logits, thus the output head runs on every row. With draft-mtp, each batch then goes to the MTP context, which runs the
//   MTP block on the same rows (context "mtp"). Then the MTP drafts a maximum of 3 tokens from the
//   next text token, one row for each step (context "mtp-draft"); this step also runs the MTP head.
//
// The time is O(tokens * model); the memory is the model plus one context.
//
// Build on the host: a CMake build B of a llama.cpp tree T (Release, BUILD_SHARED_LIBS=ON, the targets
// llama, llama-common, mtmd and ggml-cpu), then
//   g++ -O2 -std=c++17 -IT/include -IT/common -IT/ggml/include -IT/tools/mtmd -IT/vendor mm_range.cpp
//       -o mm_range -LB/bin -lmtmd -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,B/bin
// mm_range_summary.py prints the TSV per weight class.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

// The statistics of the values of one tensor role (the output or the input of one weight).
struct range_stat {
    double  max_abs    = 0.0;
    int64_t n_values   = 0;
    int64_t n_above_32 = 0;   // |value| > 32768
    int64_t n_above_65 = 0;   // |value| > 65504
    int64_t n_nonfinite = 0;
    int64_t n_calls    = 0;
    int64_t max_rows   = 0;
};

// The collector state: the context label of the running decode and the statistics of each key.
struct collector {
    std::string                       phase = "target";
    std::map<std::string, range_stat> stats;   // key: phase \t weight name \t op \t role
    std::vector<float>                buf;
};

// Add the values of a tensor to one statistic. Only F32 tensors are read. O(elements).
void add_values(collector & c, const std::string & key, const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32) {
        return;
    }
    const int64_t n = ggml_nelements(t);
    range_stat &  s = c.stats[key];
    s.n_calls++;
    s.max_rows = std::max(s.max_rows, (int64_t) t->ne[1] * t->ne[2] * t->ne[3]);
    const bool host = t->buffer && ggml_backend_buffer_is_host(t->buffer);
    const float * data = nullptr;
    if (host && ggml_is_contiguous(t)) {
        data = (const float *) t->data;
    } else if (ggml_is_contiguous(t)) {
        c.buf.resize((size_t) n);
        ggml_backend_tensor_get(t, c.buf.data(), 0, (size_t) n * sizeof(float));
        data = c.buf.data();
    } else {
        // a strided view: read it row by row through the strides
        c.buf.resize((size_t) n);
        size_t k = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
                for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                    const size_t off = i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                        float v;
                        if (host) {
                            std::memcpy(&v, (const char *) t->data + off + i0 * t->nb[0], 4);
                        } else {
                            ggml_backend_tensor_get(t, &v, off + i0 * t->nb[0], 4);
                        }
                        c.buf[k++] = v;
                    }
                }
            }
        }
        data = c.buf.data();
    }
    for (int64_t i = 0; i < n; i++) {
        const float v = data[i];
        if (!std::isfinite(v)) {
            s.n_nonfinite++;
            continue;
        }
        const double a = std::fabs((double) v);
        s.max_abs = std::max(s.max_abs, a);
        s.n_above_32 += a > 32768.0 ? 1 : 0;
        s.n_above_65 += a > 65504.0 ? 1 : 0;
    }
    s.n_values += n;
}

// The graph callback: stop at each matrix product, then record its output and its input.
bool on_node(ggml_tensor * t, bool ask, void * user) {
    const bool mm = t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID;
    if (ask) {
        return mm;
    }
    if (!mm) {
        return true;
    }
    collector &        c    = *(collector *) user;
    const std::string  w    = t->src[0] && t->src[0]->name[0] ? t->src[0]->name : "(unnamed)";
    const std::string  op   = t->op == GGML_OP_MUL_MAT ? "MUL_MAT" : "MUL_MAT_ID";
    const std::string  base = c.phase + "\t" + w + "\t" + op + "\t";
    add_values(c, base + "out", t);
    if (t->src[1]) {
        add_values(c, base + "in", t->src[1]);
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string out_path;
    int         max_tokens = 6000;
    int         mm_batch   = 0;   // tokens in each llama_decode call; 0 gives n_ubatch
    // take the options of this tool out of argv before the common parser
    std::vector<char *> rest;
    for (int i = 0; i < argc; i++) {
        if (std::strcmp(argv[i], "--mm-out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--mm-batch") == 0 && i + 1 < argc) {
            mm_batch = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--mm-tokens") == 0 && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else {
            rest.push_back(argv[i]);
        }
    }
    if (out_path.empty()) {
        std::fprintf(stderr, "mm_range: give --mm-out TSV\n");
        return 2;
    }
    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 2;
    }
    collector col;
    params.cb_eval           = on_node;
    params.cb_eval_user_data = &col;
    params.warmup            = false;
    params.n_outputs_max     = params.n_batch;
    params.n_outputs_max_per_seq = params.n_batch;

    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        std::fprintf(stderr, "mm_range: the model or the context did not load\n");
        return 1;
    }

    const bool use_mtp = std::find(params.speculative.types.begin(), params.speculative.types.end(),
                                   COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
    common_speculative_init_result_ptr spec_init;
    common_speculative *               spec = nullptr;
    if (use_mtp) {
        common_params params_dft = common_base_params_to_speculative(params);
        spec_init                = common_speculative_init_from_params(params_dft, model, ctx);
        params.speculative.draft.ctx_tgt = ctx;
        params.speculative.draft.ctx_dft = spec_init->context();
        spec = common_speculative_init(params.speculative, 1);
        if (!spec) {
            std::fprintf(stderr, "mm_range: the MTP speculation did not start\n");
            return 1;
        }
    }

    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, true, true);
    if ((int) toks.size() > max_tokens) {
        toks.resize((size_t) max_tokens);
    }
    if ((int) toks.size() + 8 > (int) llama_n_ctx(ctx)) {
        std::fprintf(stderr, "mm_range: %zu tokens need a context larger than %u\n", toks.size(), llama_n_ctx(ctx));
        return 1;
    }
    const int   B     = mm_batch > 0 ? mm_batch : (int) params.n_ubatch;
    llama_batch batch = llama_batch_init(B, 0, 1);
    int         n_dec = 0;
    int         n_drafts = 0;
    for (size_t p = 0; p < toks.size(); p += (size_t) B) {
        common_batch_clear(batch);
        const size_t e = std::min(toks.size(), p + (size_t) B);
        for (size_t i = p; i < e; i++) {
            common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, true);
        }
        col.phase = "target";
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "mm_range: llama_decode failed at token %zu\n", p);
            return 1;
        }
        if (spec) {
            col.phase = "mtp";
            if (!common_speculative_process(spec, batch)) {
                std::fprintf(stderr, "mm_range: the MTP process failed at token %zu\n", p);
                return 1;
            }
            // one draft from the next text token: the MTP block and the head on one row per step
            if (e < toks.size()) {
                col.phase = "mtp-draft";
                std::vector<llama_token> draft;
                common_speculative_get_draft_params(spec, 0) = {
                    /* .drafting = */ true,
                    /* .n_max    = */ 3,
                    /* .pos0     = */ (llama_pos) e,
                    /* .id_last  = */ toks[e],
                    /* .prompt   = */ &toks,
                    /* .result   = */ &draft,
                };
                common_speculative_draft(spec);
                // the next process call writes these positions again
                llama_memory_seq_rm(llama_get_memory(params.speculative.draft.ctx_dft), 0, (llama_pos) e, -1);
                n_drafts += (int) draft.size();
            }
        }
        n_dec++;
    }
    llama_batch_free(batch);

    FILE * f = std::fopen(out_path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "mm_range: cannot write %s\n", out_path.c_str());
        return 1;
    }
    std::fprintf(f, "context\tweight\top\trole\tmax_abs\tn_values\tn_above_32768\tn_above_65504\tn_nonfinite\tcalls\tmax_rows\n");
    for (const auto & kv : col.stats) {
        const range_stat & s = kv.second;
        std::fprintf(f, "%s\t%.6g\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\n", kv.first.c_str(), s.max_abs, (long long) s.n_values,
                     (long long) s.n_above_32, (long long) s.n_above_65, (long long) s.n_nonfinite, (long long) s.n_calls,
                     (long long) s.max_rows);
    }
    std::fclose(f);
    std::fprintf(stderr, "mm_range: %zu tokens in %d decodes of %d rows, %d draft tokens, %zu keys, written to %s\n",
                 toks.size(), n_dec, B, n_drafts, col.stats.size(), out_path.c_str());
    if (spec) {
        common_speculative_free(spec);
    }
    return 0;
}
