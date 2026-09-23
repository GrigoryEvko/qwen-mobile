// hexhost_graphs: the real llama.cpp graphs of a model on the fake DSP of the hexhost harness.
//
// The program registers the host part of the Hexagon backend (ggml-hexagon.cpp, built against
// the stub SDK and the fake DSP of tests/fuzz/hexhost/common) as the device HTP0, loads a GGUF
// model onto it, and runs the graphs of one path of the app: a batch of prompt tokens (prefill),
// one token (decode), the MTP draft and its verify batch (mtp), or the vision encoder of an mmproj
// on HTP0 for three photo shapes (vision). It writes:
//   - each supports_op decision of HTP0: the op, its type and shape, its source types and
//     shapes, and the answer (one line each, sorted and without duplicates)
//   - the count of each HTP op that the host sends to the DSP (the fused ops included)
//   - the host warnings about the fused recurrent ops (the fallback to the unfused ops)
//   - each product that the host can fuse with the ADD after it while one more node reads the
//     product through a view (that node reads bytes that no op wrote)
// The fake DSP runs the checks of dsp_model.cpp on each op. It computes no values.
//
//   hexhost_graphs MODEL.gguf {prefill N | decode | mtp N_PROMPT | vision MMPROJ MAX_TOKENS} OUT_PREFIX
//
// Outputs: OUT_PREFIX.supports.txt, OUT_PREFIX.ops.txt, OUT_PREFIX.log.txt, OUT_PREFIX.hazards.txt.
// Environment: GGML_HEXAGON_* as for the app. HEXHOST_IGNORE lists the checks of the fake DSP
// that do not stop the run (refer to run.sh). HEXHOST_TOUCH=1 makes the fake DSP write the outputs.
// The fake DSP has 6 HVX threads, 1 HMX unit and 8 MB of VTCM, as the v79 NPU of the phone.

#include "dsp_model.h"
#include "fake_dsp.h"
#include "matmul-ops.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-hexagon.h"
#include "llama.h"
#include "mtmd.h"
#include "speculative.h"
#include "common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

std::set<std::string>   g_supports;   // the decisions, one line each
std::vector<std::string> g_log;       // the warnings and the errors of ggml
std::mutex               g_mutex;
bool (*g_orig_supports)(ggml_backend_dev_t, const ggml_tensor *) = nullptr;

// Gives the type and the shape of a tensor, "f32[64,1,1,1]".
std::string shape_of(const ggml_tensor * t) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s[%lld,%lld,%lld,%lld]", ggml_type_name(t->type), (long long) t->ne[0],
             (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return buf;
}

// Calls supports_op of HTP0 and records the decision with the op and the shapes.
bool supports_logged(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const bool  ok = g_orig_supports(dev, op);
    std::string line = std::string(ggml_op_desc(op)) + " " + shape_of(op);
    for (int i = 0; i < GGML_MAX_SRC && op->src[i]; i++) {
        line += " " + shape_of(op->src[i]);
    }
    line += ok ? " -> yes" : " -> no";
    std::lock_guard<std::mutex> lock(g_mutex);
    g_supports.insert(line);
    return ok;
}

// ---- The check of the view readers: a product that the host fuses with the ADD after it, and
// that one more node reads through a view. The fused MUL_MAT_ADD writes only the output of the ADD,
// thus that other node reads bytes that no op wrote.

std::vector<std::string> g_hazards;   // one line for each such product
size_t                   g_candidates = 0;   // the products with the fusion conditions (a check of the check)
ggml_status (*g_orig_compute)(ggml_backend_t, ggml_cgraph *) = nullptr;
ggml_backend_t (*g_orig_init)(ggml_backend_dev_t, const char *) = nullptr;

// True for an op that only makes a view (no compute on the device).
bool is_view_op(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
           t->op == GGML_OP_TRANSPOSE;
}

// Finds the products of a split that have the fusion conditions of the host (one direct use,
// and the next compute node is an ADD that reads the whole product) and more than one reader
// through views. O(nodes^2 * GGML_MAX_SRC).
void check_fused_view_readers(const ggml_cgraph * gf) {
    const int n = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(gf));
    for (int i = 0; i < n; i++) {
        ggml_tensor * m = ggml_graph_node(const_cast<ggml_cgraph *>(gf), i);
        if (m->op != GGML_OP_MUL_MAT && m->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        int direct = 0, readers = 0;
        for (int j = 0; j < n; j++) {
            const ggml_tensor * r    = ggml_graph_node(const_cast<ggml_cgraph *>(gf), j);
            bool                reads = false;
            for (int s = 0; s < GGML_MAX_SRC && r->src[s]; s++) {
                direct += r->src[s] == m;
                reads |= r->src[s] == m || r->src[s]->view_src == m;
            }
            readers += reads && !is_view_op(r);
        }
        int k = i + 1;
        while (k < n && (is_view_op(ggml_graph_node(const_cast<ggml_cgraph *>(gf), k)) ||
                         ggml_is_empty(ggml_graph_node(const_cast<ggml_cgraph *>(gf), k)))) {
            k++;
        }
        if (direct != 1 || k >= n) {
            continue;
        }
        const ggml_tensor * add   = ggml_graph_node(const_cast<ggml_cgraph *>(gf), k);
        auto                whole = [m](const ggml_tensor * t) {
            return t && t->data == m->data && ggml_are_same_shape(t, m) && ggml_are_same_stride(t, m);
        };
        if (add->op != GGML_OP_ADD || !(whole(add->src[0]) || whole(add->src[1]))) {
            continue;
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        g_candidates++;
        if (readers < 2) {
            continue;
        }
        g_hazards.push_back(std::string(m->name) + " " + shape_of(m) + " -> " + add->name + ": " + std::to_string(readers) +
                            " readers");
    }
}

// The graph_compute of HTP0 with the check of the view readers first.
ggml_status compute_checked(ggml_backend_t backend, ggml_cgraph * gf) {
    check_fused_view_readers(gf);
    return g_orig_compute(backend, gf);
}

// The init_backend of HTP0: the backend gets compute_checked as its graph_compute.
ggml_backend_t init_checked(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t b = g_orig_init(dev, params);
    if (b) {
        g_orig_compute        = b->iface.graph_compute;
        b->iface.graph_compute = compute_checked;
    }
    return b;
}

// Keeps the warnings and the errors of ggml and llama.cpp.
void log_keep(enum ggml_log_level level, const char * text, void * user) {
    (void) user;
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_log.push_back(text);
    }
}

// Writes the lines of a container to a file. Gives false when the file cannot be written.
template <typename C> bool write_lines(const std::string & path, const C & lines) {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "hexhost_graphs: cannot write %s\n", path.c_str());
        return false;
    }
    for (const auto & l : lines) {
        fputs(l.c_str(), f);
        if (l.empty() || l.back() != '\n') {
            fputc('\n', f);
        }
    }
    fclose(f);
    return true;
}

// A batch of tokens from position pos0 in sequence 0, each with logits, as the app decodes a step.
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

// The MTP draft path of the app: a target context with 4 recurrent state snapshots, a prompt, the
// draft context of the MTP block (common_speculative, type draft-mtp), one draft of up to 4
// tokens, and the verify batch of the target (1 + the draft). Gives the llama_decode code.
int run_mtp(llama_model * model, llama_context * ctx, const std::string & model_path, int n_prompt) {
    const int32_t            n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> prompt(n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        prompt[i] = (llama_token) ((1000 + 7 * i) % n_vocab);
    }
    // The draft driver comes first, as setup_speculative of the app: it makes the target context
    // keep the hidden states that the MTP block reads
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
    auto          init             = common_speculative_init_from_params(params_dft, model, ctx);
    llama_context * ctx_dft        = init ? init->context() : nullptr;
    if (!ctx_dft) {
        fprintf(stderr, "hexhost_graphs: the MTP draft context did not initialize (does the model hold the MTP block?)\n");
        return -1;
    }
    params.speculative.draft.ctx_tgt = ctx;
    params.speculative.draft.ctx_dft = ctx_dft;
    common_speculative * spec        = common_speculative_init(params.speculative, 1);
    if (!spec) {
        fprintf(stderr, "hexhost_graphs: the speculative driver did not initialize\n");
        return -1;
    }

    // The prompt, then the draft context follows it (spec_follow of the app)
    llama_batch bp = make_batch(prompt, 0);
    int         rc = llama_decode(ctx, bp);
    if (rc == 0) {
        common_speculative_process(spec, bp);
    }
    llama_batch_free(bp);
    if (rc != 0) {
        common_speculative_free(spec);
        return rc;
    }
    common_speculative_begin(spec, 0, prompt);

    llama_tokens draft;
    common_speculative_get_draft_params(spec, 0) = { true, 4, (llama_pos) n_prompt, prompt.back(), &prompt, &draft };
    common_speculative_draft(spec);
    llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_prompt, -1);

    std::vector<llama_token> verify = { prompt.back() };
    verify.insert(verify.end(), draft.begin(), draft.end());
    llama_batch bv = make_batch(verify, (llama_pos) n_prompt);
    rc             = llama_decode(ctx, bv);
    if (rc == 0) {
        common_speculative_process(spec, bv);
        common_speculative_accept(spec, 0, 0);
    }
    llama_batch_free(bv);
    printf("hexhost_graphs: the MTP draft gave %zu tokens\n", draft.size());
    common_speculative_free(spec);
    return rc;
}

// The vision encoder of the app on HTP0: the mmproj with the image token limit of the app, and
// three photo shapes (4:3, 16:9 and 1:1) of random pixels. Gives 0 when each image encodes.
int run_vision(llama_model * model, ggml_backend_dev_t htp, const std::string & mmproj, int max_tokens) {
    mtmd_context_params vp = mtmd_context_params_default();
    vp.use_gpu             = true;
    vp.device              = htp;
    vp.n_threads           = 4;
    vp.print_timings       = false;
    vp.warmup              = false;
    vp.image_max_tokens    = max_tokens;
    mtmd_context * mctx    = mtmd_init_from_file(mmproj.c_str(), model, vp);
    if (!mctx) {
        fprintf(stderr, "hexhost_graphs: the vision projector did not load: %s\n", mmproj.c_str());
        return -1;
    }
    int rc = 0;
    for (const auto & wh : std::vector<std::pair<uint32_t, uint32_t>>{ { 1024, 768 }, { 1280, 720 }, { 800, 800 } }) {
        std::vector<unsigned char> rgb((size_t) wh.first * wh.second * 3);
        uint32_t                   x = 12345;
        for (auto & c : rgb) {
            x = x * 1103515245u + 12345u;
            c = (unsigned char) (x >> 24);
        }
        mtmd_bitmap *        bmp    = mtmd_bitmap_init(wh.first, wh.second, rgb.data());
        mtmd_input_chunks *  chunks = mtmd_input_chunks_init();
        const std::string    text   = std::string("describe ") + mtmd_default_marker();
        mtmd_input_text      in     = { text.c_str(), text.size(), true, true };
        const mtmd_bitmap *  bmps[] = { bmp };
        int                  t      = mtmd_tokenize(mctx, chunks, &in, bmps, 1);
        for (size_t i = 0; t == 0 && i < mtmd_input_chunks_size(chunks); i++) {
            const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                t = mtmd_encode_chunk(mctx, ch);
            }
        }
        printf("hexhost_graphs: image %ux%u: %s\n", wh.first, wh.second, t == 0 ? "encoded" : "failed");
        rc |= t;
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
    }
    mtmd_free(mctx);
    return rc;
}

} // namespace

int main(int argc, char ** argv) {
    const char * usage = "Usage: %s MODEL.gguf {prefill N | decode | mtp N_PROMPT | vision MMPROJ MAX_TOKENS} OUT_PREFIX\n";
    if (argc < 4) {
        fprintf(stderr, usage, argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string kind       = argv[2];
    const bool        prefill    = kind == "prefill";
    const bool        mtp        = kind == "mtp";
    const bool        vision     = kind == "vision";
    const int         n_args     = vision ? 6 : (prefill || mtp) ? 5 : 4;
    if ((!prefill && !mtp && !vision && kind != "decode") || argc < n_args) {
        fprintf(stderr, usage, argv[0]);
        return 2;
    }
    const int         n_tokens   = (prefill || mtp) ? atoi(argv[3]) : 1;
    const std::string mmproj     = vision ? argv[3] : "";
    const int         max_tokens = vision ? atoi(argv[4]) : 0;
    const std::string out        = argv[n_args - 1];
    if (n_tokens < 1) {
        fprintf(stderr, "hexhost_graphs: the token count must be positive\n");
        return 2;
    }

    // The arch gate of the backend needs libggml-htp-v79.so in ADSP_LIBRARY_PATH: an empty file
    // in a temporary directory (removed at the end)
    const std::filesystem::path libdir = std::filesystem::temp_directory_path() /
                                         ("hexhost-graphs-" + std::to_string(getpid()));
    std::filesystem::create_directories(libdir);
    fclose(fopen((libdir / "libggml-htp-v79.so").c_str(), "w"));
    setenv("ADSP_LIBRARY_PATH", libdir.c_str(), 1);

    // 6 threads, 1 HMX unit, 8 MB of VTCM, v79. HEXHOST_TOUCH=1: the fake DSP writes 0xA5 into the
    // output bytes of each op when the host sends the batch, thus a host read of bytes that an op
    // of an earlier batch wrote gets a value that is not a slot index.
    fakedsp::config cfg;
    cfg.touch = getenv("HEXHOST_TOUCH") != nullptr;
    fakedsp::configure(cfg);
    ggml_log_set(log_keep, nullptr);
    llama_log_set(log_keep, nullptr);

    ggml_backend_reg_t reg = ggml_backend_hexagon_reg();
    ggml_backend_register(reg);
    ggml_backend_dev_t htp = ggml_backend_dev_by_name("HTP0");
    if (!htp) {
        fprintf(stderr, "hexhost_graphs: no HTP0 device\n");
        std::filesystem::remove_all(libdir);
        return 1;
    }
    g_orig_supports       = htp->iface.supports_op;
    htp->iface.supports_op = supports_logged;
    g_orig_init            = htp->iface.init_backend;
    htp->iface.init_backend = init_checked;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { htp, nullptr };
    mp.devices      = devs;
    mp.n_gpu_layers = 999;
    mp.load_mtp     = mtp;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "hexhost_graphs: cannot load %s\n", model_path.c_str());
        std::filesystem::remove_all(libdir);
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    // The app keeps as many recurrent state snapshots as the longest draft (4)
    cp.n_rs_seq = mtp ? 4 : 0;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "hexhost_graphs: cannot make a context\n");
        llama_model_free(model);
        std::filesystem::remove_all(libdir);
        return 1;
    }

    // The ops of the graphs: the counters of the fake DSP from here
    fakedsp::reset_record();
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int           rc      = 0;
    if (mtp) {
        rc = run_mtp(model, ctx, model_path, n_tokens);
    } else if (vision) {
        rc = run_vision(model, htp, mmproj, max_tokens);
    } else {
        std::vector<llama_token> toks(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            toks[i] = (llama_token) ((1000 + 7 * i) % n_vocab);
        }
        rc = llama_decode(ctx, llama_batch_get_one(toks.data(), n_tokens));
    }
    llama_synchronize(ctx);

    // The count of each HTP op, and for each matmul op also its kernel type, the kernel params of
    // HMX or HVX, and the shape of its first weight (a row count that is not a multiple of 32
    // takes a padded tile)
    std::map<std::string, uint64_t> ops;
    for (const auto & b : fakedsp::take_batches()) {
        for (const auto & op : b.ops) {
            ops[fakedsp::opcode_name(op.opcode)]++;
            const bool mm = op.opcode == HTP_OP_MUL_MAT || op.opcode == HTP_OP_MUL_MAT_ADD || op.opcode == HTP_OP_MUL_MAT_NX ||
                            op.opcode == HTP_OP_MUL_MAT_ID || op.opcode == HTP_OP_MUL_MAT_ID_NX;
            if (mm && op.src[0].present) {
                const auto * k = (const htp_mm_kernel_params *) op.kparams;
                char         line[256];
                snprintf(line, sizeof(line), "  %s kernel_type %d n_hmx %d weight type %u [%u,%u] rows%%32 %u",
                         fakedsp::opcode_name(op.opcode), (int) k->kernel_type, (int) k->n_hmx, op.src[0].type,
                         op.src[0].ne[0], op.src[0].ne[1], op.src[0].ne[1] % 32);
                ops[line]++;
            }
        }
    }
    std::vector<std::string> op_lines;
    for (const auto & kv : ops) {
        op_lines.push_back(kv.first + " " + std::to_string(kv.second));
    }

    // HEXHOST_BENCH=N: N more one-token decodes, timed one by one (the host time of graph_compute
    // and of the fake DSP; GGML_HEXAGON_BATCHCACHE=0 makes each token pack its batch again)
    std::vector<double> times;
    const int n_bench = getenv("HEXHOST_BENCH") ? atoi(getenv("HEXHOST_BENCH")) : 0;
    for (int i = 0; i < n_bench && rc == 0; i++) {
        llama_token t = (llama_token) ((2000 + 13 * i) % n_vocab);
        const int64_t t0 = ggml_time_us();
        rc = llama_decode(ctx, llama_batch_get_one(&t, 1));
        llama_synchronize(ctx);
        times.push_back((double) (ggml_time_us() - t0));
        fakedsp::take_batches();  // the records of the bench tokens are not needed
    }
    if (!times.empty()) {
        std::sort(times.begin(), times.end());
        printf("hexhost_graphs: bench %zu tokens: min %.1f us, median %.1f us, p90 %.1f us\n", times.size(), times.front(),
               times[times.size() / 2], times[times.size() * 9 / 10]);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::filesystem::remove_all(libdir);

    write_lines(out + ".supports.txt", g_supports);
    write_lines(out + ".ops.txt", op_lines);
    write_lines(out + ".log.txt", g_log);
    write_lines(out + ".hazards.txt", g_hazards);
    printf("hexhost_graphs: %s %s %d: llama_decode %d, %zu supports lines, %zu HTP op kinds, %zu log lines, "
           "%zu products with the MUL_MAT_ADD conditions, %zu of them with a view reader\n",
           model_path.c_str(), prefill ? "prefill" : "decode", n_tokens, rc, g_supports.size(), ops.size(), g_log.size(),
           g_candidates, g_hazards.size());
    return rc == 0 ? 0 : 1;
}
