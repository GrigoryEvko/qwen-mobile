// op_audit: the shapes and types of the GET_ROWS, ADD, MUL, SOFT_MAX and FLASH_ATTN_EXT nodes of
// the app graphs, with the conditions of the known backend defects.
//
// The graph callback stops at each node of these ops in the target context, the MTP context (the
// process step and the draft steps) and, with --mmproj IMAGE, the vision encoder of mtmd. For each
// distinct node form (context, op, types, shapes, strides) it counts the nodes. For SOFT_MAX and
// FLASH_ATTN_EXT it also reads the mask and counts the rows that are -inf in each column.
//
// Usage: op_audit -m MODEL -f TEXT -c N_CTX -b 512 -ub 512 [--spec-type draft-mtp] [-t N]
//                 --audit-out TSV [--audit-batch N] [--audit-tokens N] [--audit-image FILE]
//                 [--audit-mmproj FILE]
// The time is O(tokens * model); the memory is the model plus one context.
//
// Build on the host: as mm_range (the header of mm_range.cpp), with op_audit.cpp as the source.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

// The facts of one node form.
struct form_stat {
    int64_t n_nodes      = 0;
    int64_t masked_rows  = 0;  // rows of the mask that are -inf in every column
    int64_t mask_rows    = 0;  // rows of the mask that the node reads
};

struct audit {
    std::string                      phase = "target";
    std::map<std::string, form_stat> forms;
};

// Print the type, the shape and the contiguity of a tensor.
std::string tdesc(const ggml_tensor * t) {
    if (!t) {
        return "-";
    }
    char b[256];
    std::snprintf(b, sizeof(b), "%s[%lld,%lld,%lld,%lld]%s", ggml_type_name(t->type), (long long) t->ne[0],
                  (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                  ggml_is_contiguous(t) ? "" : (ggml_is_contiguous_rows(t) ? " rows" : " strided"));
    return b;
}

// The defect conditions of a node, as text. Empty when no condition holds.
std::string flags_of(const ggml_tensor * t) {
    std::string f;
    const ggml_tensor * a = t->src[0];
    const ggml_tensor * b = t->src[1];
    if (t->op == GGML_OP_GET_ROWS && a && a->type == GGML_TYPE_F16 && a->ne[0] % 32 != 0) {
        f += "getrows-f16-tail ";
    }
    if ((t->op == GGML_OP_ADD || t->op == GGML_OP_MUL) && a && b) {
        if (b->ne[0] == 1 && a->ne[0] > 1) {
            f += "src1-dim0-broadcast ";
        }
        if (b->ne[0] > 1 && b->ne[0] < a->ne[0]) {
            f += "src1-dim0-repeat ";
        }
        for (int d = 1; d < 4; d++) {
            if (b->ne[d] != a->ne[d]) {
                f += "src1-dim" + std::to_string(d) + "-broadcast ";
            }
        }
    }
    return f;
}

// Count the rows of an F16 or F32 mask that are -inf in every column. O(elements of the mask).
void count_masked_rows(const ggml_tensor * m, form_stat & s) {
    if (!m || (m->type != GGML_TYPE_F16 && m->type != GGML_TYPE_F32) || !ggml_is_contiguous(m)) {
        return;
    }
    const int64_t n0 = m->ne[0], rows = ggml_nrows(m);
    std::vector<uint8_t> buf(ggml_nbytes(m));
    ggml_backend_tensor_get(m, buf.data(), 0, buf.size());
    for (int64_t r = 0; r < rows; r++) {
        bool all = true;
        for (int64_t j = 0; j < n0 && all; j++) {
            float v;
            if (m->type == GGML_TYPE_F16) {
                v = ggml_fp16_to_fp32(((const ggml_fp16_t *) buf.data())[r * n0 + j]);
            } else {
                v = ((const float *) buf.data())[r * n0 + j];
            }
            all = v == -INFINITY;
        }
        s.masked_rows += all ? 1 : 0;
    }
    s.mask_rows += rows;
}

bool on_node(ggml_tensor * t, bool ask, void * user) {
    const bool want = t->op == GGML_OP_GET_ROWS || t->op == GGML_OP_ADD || t->op == GGML_OP_MUL ||
                      t->op == GGML_OP_SOFT_MAX || t->op == GGML_OP_FLASH_ATTN_EXT;
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    audit &           A   = *(audit *) user;
    const std::string key = A.phase + "\t" + ggml_op_desc(t) + "\t" + tdesc(t->src[0]) + "\t" + tdesc(t->src[1]) +
                            "\t" + tdesc(t->op == GGML_OP_FLASH_ATTN_EXT ? t->src[3] : t->src[2]) + "\t" + tdesc(t) +
                            "\t" + flags_of(t);
    form_stat & s = A.forms[key];
    s.n_nodes++;
    if (t->op == GGML_OP_SOFT_MAX) {
        count_masked_rows(t->src[1], s);
    } else if (t->op == GGML_OP_FLASH_ATTN_EXT) {
        count_masked_rows(t->src[3], s);
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string out_path, image, mmproj;
    int         max_tokens = 1200, batch = 0;
    std::vector<char *> rest;
    for (int i = 0; i < argc; i++) {
        if (std::strcmp(argv[i], "--audit-out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--audit-batch") == 0 && i + 1 < argc) {
            batch = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--audit-tokens") == 0 && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--audit-image") == 0 && i + 1 < argc) {
            image = argv[++i];
        } else if (std::strcmp(argv[i], "--audit-mmproj") == 0 && i + 1 < argc) {
            mmproj = argv[++i];
        } else {
            rest.push_back(argv[i]);
        }
    }
    if (out_path.empty()) {
        std::fprintf(stderr, "op_audit: give --audit-out TSV\n");
        return 2;
    }
    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 2;
    }
    audit A;
    params.cb_eval               = on_node;
    params.cb_eval_user_data     = &A;
    params.warmup                = false;
    params.n_outputs_max         = params.n_batch;
    params.n_outputs_max_per_seq = params.n_batch;

    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        std::fprintf(stderr, "op_audit: the model or the context did not load\n");
        return 1;
    }

    // the vision encoder: one image through mtmd with the same callback
    if (!image.empty()) {
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu             = false;
        mp.n_threads           = params.cpuparams.n_threads;
        mp.cb_eval             = on_node;
        mp.cb_eval_user_data   = &A;
        mp.warmup              = false;
        mtmd_context * mctx    = mtmd_init_from_file(mmproj.c_str(), model, mp);
        if (!mctx) {
            std::fprintf(stderr, "op_audit: the projector %s did not load\n", mmproj.c_str());
            return 1;
        }
        mtmd_bitmap * bmp =
            mtmd_helper_bitmap_init_from_file(mctx, image.c_str(), false, mtmd_helper_init_opt_default()).bitmap;
        if (!bmp) {
            std::fprintf(stderr, "op_audit: the image %s did not load\n", image.c_str());
            return 1;
        }
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        std::string         text   = std::string("Describe ") + mtmd_default_marker();
        mtmd_input_text     it     = { text.c_str(), text.size(), true, true };
        const mtmd_bitmap * bmps[] = { bmp };
        if (mtmd_tokenize(mctx, chunks, &it, bmps, 1) != 0) {
            std::fprintf(stderr, "op_audit: mtmd_tokenize failed\n");
            return 1;
        }
        A.phase = "vision";
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
            const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                if (mtmd_encode_chunk(mctx, ch) != 0) {
                    std::fprintf(stderr, "op_audit: mtmd_encode_chunk failed\n");
                    return 1;
                }
            }
        }
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        mtmd_free(mctx);
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
            std::fprintf(stderr, "op_audit: the MTP speculation did not start\n");
            return 1;
        }
    }

    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, true, true);
    if ((int) toks.size() > max_tokens) {
        toks.resize((size_t) max_tokens);
    }
    const int   B = batch > 0 ? batch : (int) params.n_ubatch;
    llama_batch bt = llama_batch_init(B, 0, 1);
    for (size_t p = 0; p < toks.size(); p += (size_t) B) {
        common_batch_clear(bt);
        const size_t e = std::min(toks.size(), p + (size_t) B);
        for (size_t i = p; i < e; i++) {
            common_batch_add(bt, toks[i], (llama_pos) i, { 0 }, true);
        }
        A.phase = "target";
        if (llama_decode(ctx, bt) != 0) {
            std::fprintf(stderr, "op_audit: llama_decode failed at token %zu\n", p);
            return 1;
        }
        if (spec) {
            A.phase = "mtp";
            if (!common_speculative_process(spec, bt)) {
                std::fprintf(stderr, "op_audit: the MTP process failed at token %zu\n", p);
                return 1;
            }
            if (e < toks.size()) {
                A.phase = "mtp-draft";
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
                llama_memory_seq_rm(llama_get_memory(params.speculative.draft.ctx_dft), 0, (llama_pos) e, -1);
            }
        }
    }
    llama_batch_free(bt);

    FILE * f = std::fopen(out_path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "op_audit: cannot write %s\n", out_path.c_str());
        return 1;
    }
    std::fprintf(f, "context\top\tsrc0\tsrc1\tsrc2_or_mask\tdst\tflags\tnodes\tmask_rows\tmasked_rows\n");
    for (const auto & kv : A.forms) {
        std::fprintf(f, "%s\t%lld\t%lld\t%lld\n", kv.first.c_str(), (long long) kv.second.n_nodes,
                     (long long) kv.second.mask_rows, (long long) kv.second.masked_rows);
    }
    std::fclose(f);
    std::fprintf(stderr, "op_audit: %zu tokens in batches of %d, %zu forms, written to %s\n", toks.size(), B,
                 A.forms.size(), out_path.c_str());
    if (spec) {
        common_speculative_free(spec);
    }
    return 0;
}
