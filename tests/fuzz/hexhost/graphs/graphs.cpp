// hexhost_graphs: the real llama.cpp graphs of a model on the fake DSP of the hexhost harness.
//
// The program registers the host part of the Hexagon backend (ggml-hexagon.cpp, built against
// the stub SDK and the fake DSP of tests/fuzz/hexhost/common) as the device HTP0, loads a GGUF
// model onto it, and runs one llama_decode: a batch of prompt tokens (prefill) or one token
// (decode). It writes:
//   - each supports_op decision of HTP0: the op, its type and shape, its source types and
//     shapes, and the answer (one line each, sorted and without duplicates)
//   - the count of each HTP op that the host sends to the DSP (the fused ops included)
//   - the host warnings about the fused recurrent ops (the fallback to the unfused ops)
// The fake DSP runs the checks of dsp_model.cpp on each op. It computes no values.
//
//   hexhost_graphs MODEL.gguf {prefill N | decode} OUT_PREFIX
//
// Outputs: OUT_PREFIX.supports.txt, OUT_PREFIX.ops.txt, OUT_PREFIX.log.txt.
// Environment: GGML_HEXAGON_* as for the app. HEXHOST_IGNORE lists the known findings that the
// fake DSP lets pass (refer to run.sh). HEXHOST_TOUCH=1 makes the fake DSP write the outputs.
// The fake DSP has 6 HVX threads, 1 HMX unit and 8 MB of VTCM, as the v79 NPU of the phone.

#include "dsp_model.h"
#include "fake_dsp.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-hexagon.h"
#include "llama.h"

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

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4 || (strcmp(argv[2], "prefill") == 0 && argc < 5)) {
        fprintf(stderr, "Usage: %s MODEL.gguf {prefill N | decode} OUT_PREFIX\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const bool        prefill    = strcmp(argv[2], "prefill") == 0;
    const int         n_tokens   = prefill ? atoi(argv[3]) : 1;
    const std::string out        = argv[prefill ? 4 : 3];
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

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { htp, nullptr };
    mp.devices      = devs;
    mp.n_gpu_layers = 999;
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
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "hexhost_graphs: cannot make a context\n");
        llama_model_free(model);
        std::filesystem::remove_all(libdir);
        return 1;
    }

    // The ops of the one decode: the counters of the fake DSP from here
    fakedsp::reset_record();
    std::vector<llama_token> toks(n_tokens);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int i = 0; i < n_tokens; i++) {
        toks[i] = (llama_token) ((1000 + 7 * i) % n_vocab);
    }
    int rc = llama_decode(ctx, llama_batch_get_one(toks.data(), n_tokens));
    llama_synchronize(ctx);

    std::map<std::string, uint64_t> ops;
    for (const auto & b : fakedsp::take_batches()) {
        for (const auto & op : b.ops) {
            ops[fakedsp::opcode_name(op.opcode)]++;
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
    printf("hexhost_graphs: %s %s %d: llama_decode %d, %zu supports lines, %zu HTP op kinds, %zu log lines\n",
           model_path.c_str(), prefill ? "prefill" : "decode", n_tokens, rc, g_supports.size(), ops.size(), g_log.size());
    return rc == 0 ? 0 : 1;
}
