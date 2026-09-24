// ffncheck: a hash of the output of the FFN block of the Qwen3.5 4B on one device (HTP0 is the preset), for the
// bit-exact check of the matmul fusions of the Hexagon backend (htp-mm-fusion.h): GGML_HEXAGON_FUSE_SWIGLU,
// _SWIGLU_DECODE and _F16_ACT on against off.
//
// test-backend-ops draws new inputs for each run, thus two runs cannot give the same bytes. This program gives
// each case the same inputs in each run: a fixed seed for each tensor. It runs each case on the device and writes
// the FNV-1a hash of the output bytes. With --cpu it also writes the NMSE and the largest error against the CPU
// backend, and the largest magnitudes of the SwiGLU output h and of the output y of the CPU backend.
//
// The weights are in a buffer with the usage GGML_BACKEND_BUFFER_USAGE_WEIGHTS, as the model loader of llama.cpp
// gives them (src/llama-model.cpp). The Hexagon backend puts a quantized weight into its tiled layout only in such a
// buffer. The program sets the usage before the first ggml_backend_tensor_set, because the backend reads the usage
// in set_tensor.
//
// The forms of a case, as the graph of the model has them:
//   ffn     y = W_down swiglu(W_gate n, W_up n) + x, with n = rms_norm(x) * w_norm. The fusions give
//           MUL_MAT_NX_SWIGLU and, from 5 tokens, an F16 SwiGLU output that the down MUL_MAT reads.
//   swiglu  h = swiglu(W_gate x, W_up x), with h an output of the graph (no F16 output).
//   bigh    the form ffn with large gate and up weights: some values of h are outside the F16 range.
//   bigy    the form ffn with a large down weight: some values of W_down h are outside the F16 range (the
//           output tiles of the HMX are F16).
// The forms bigh and bigy make sure that the fusions give the bytes of the unfused ops also for the values that
// the F16 conversions cannot hold. Thus a non-finite output value is not a failure for these two forms.
//
// Usage: ffncheck [--dev NAME] [--cpu | --plain] [--threads N] [--only SUBSTRING]
//   --dev NAME  the device under test, HTP0 as the preset. "--dev CPU" gives a test of the program on the host.
//   --plain     the weights stay in a buffer with no WEIGHTS usage, thus the Hexagon backend keeps the plain
//               layout of ggml. The correct result: supports_op refuses a MUL_MAT and graph_compute fails.
//               The line of a case is "ffncheck case=<name> plain supports=<yes|no(op)> compute=<ok|failed>",
//               and a case fails when supports_op accepts each op or when the graph computes.
// Output: one line for each case,
//   ffncheck case=<name> hash=<16 hex digits> nonfinite=<n> [nmse=<x> maxerr=<y> hmax=<a> ymax=<b>] us=<t>
// and at the end "ffncheck done cases=<n> failed=<n>". A case fails when an op is not supported, when the
// allocation or the compute fails, when an output value is not finite (not for bigh and bigy), or with --cpu
// when its NMSE is more than 1e-4 (not for bigh and bigy). The exit code is 1 when a case fails.
//
// Build (the NDK clang++ or the host g++, against the libraries of a llama.cpp tree):
//   clang++ -O2 -std=c++17 -I TREE/ggml/include ffncheck.cpp -o ffncheck -L BUILD/bin
//       -lggml -lggml-cpu -lggml-base -Wl,-rpath,'$ORIGIN/../lib'
// Time: about 20 s on the phone without --cpu, and about 60 s with it.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t N_EMBD = 2560;  // the 4B
constexpr int64_t N_FF   = 9216;

struct form {
    const char *         name;
    bool                 ffn;      // the norm, the down MUL_MAT and the residual, or h only
    float                gate_up;  // the half range of the uniform values of W_gate and W_up
    float                down;     // the half range of the uniform values of W_down
    bool                 big;      // values outside the F16 range occur: a non-finite output is not a failure
    std::vector<int64_t> tokens;
};

// The decode (1 token), the verify step of speculative decoding (2 and 4), the smallest HMX case (5), and the
// prefill with odd token counts that give a partial last token chunk.
const std::vector<form> FORMS = {
    { "ffn", true, 0.05f, 0.03f, false, { 1, 2, 4, 5, 64, 512, 1000, 1024 } },
    { "swiglu", false, 0.05f, 0.03f, false, { 1, 4, 64, 1024 } },
    { "bigh", true, 3.0f, 0.002f, true, { 5, 64, 512 } },
    { "bigy", true, 0.5f, 2.0f, true, { 5, 64, 512 } },
};

uint64_t fnv1a(const void * data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// Uniform values in [lo, hi] from a seed. The same seed gives the same values in each run. O(n).
std::vector<float> uniform(size_t n, uint32_t seed, float lo, float hi) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float>                    v(n);
    for (float & x : v) {
        x = dist(rng);
    }
    return v;
}

// The Q8_0 bytes of a matrix of nrows rows of ncols values. O(nrows * ncols).
std::vector<uint8_t> quantize_q8_0(const std::vector<float> & v, int64_t nrows, int64_t ncols) {
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_0, ncols) * (size_t) nrows);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, v.data(), q.data(), 0, nrows, ncols, nullptr);
    return q;
}

// The weights of one form: w_norm as F32 and the three Q8_0 weights. The program makes them one time for each
// form, because the random values of the three weights take most of the time of a case on the phone CPU.
struct form_weights {
    std::vector<float>   norm;
    std::vector<uint8_t> gate, up, down;
};

// The seeds depend on the tensor only, thus the weights of a form are the same in each case and each run.
// O(N_EMBD * N_FF).
form_weights make_weights(const form & f) {
    form_weights w;
    w.norm = uniform((size_t) N_EMBD, 11, 0.5f, 1.5f);
    w.gate = quantize_q8_0(uniform((size_t) (N_EMBD * N_FF), 12, -f.gate_up, f.gate_up), N_FF, N_EMBD);
    w.up   = quantize_q8_0(uniform((size_t) (N_EMBD * N_FF), 13, -f.gate_up, f.gate_up), N_FF, N_EMBD);
    w.down = quantize_q8_0(uniform((size_t) (N_FF * N_EMBD), 14, -f.down, f.down), N_EMBD, N_FF);
    return w;
}

template <typename T> void set_bytes(ggml_tensor * t, const std::vector<T> & v) {
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(T));
}

std::vector<float> get_values(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

// The largest magnitude of the finite values. O(n).
double max_abs(const std::vector<float> & v) {
    double m = 0.0;
    for (float f : v) {
        if (std::isfinite(f)) {
            m = std::fmax(m, std::fabs((double) f));
        }
    }
    return m;
}

// The tensors and the graph of one case on one backend. The weights have their own context and buffer (the
// buffer of the model), the input and the activations the other.
struct run_ctx {
    ggml_context *        ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_context *        ctx   = nullptr;
    ggml_backend_buffer_t buf   = nullptr;
    ggml_cgraph *         gf    = nullptr;
    ggml_tensor *         out   = nullptr;
    ggml_tensor *         h     = nullptr;

    ~run_ctx() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (buf_w) {
            ggml_backend_buffer_free(buf_w);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        if (ctx_w) {
            ggml_free(ctx_w);
        }
    }
};

// Builds the graph of one case, gives it its inputs, and computes it on the backend. keep_h makes h an output
// (the CPU reference only, because an output h changes the fusions). Returns an empty string on success, and
// the cause of the failure otherwise.
std::string run_case(ggml_backend_t be, const form & f, const form_weights & w, int64_t m, bool keep_h, bool plain,
                     run_ctx & rc, double & us, std::string & refused) {
    const ggml_init_params ipw = { 8 * ggml_tensor_overhead(), nullptr, true };
    const ggml_init_params ip  = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    rc.ctx_w                   = ggml_init(ipw);
    rc.ctx                     = ggml_init(ip);

    ggml_tensor * w_norm = ggml_new_tensor_1d(rc.ctx_w, GGML_TYPE_F32, N_EMBD);
    ggml_tensor * w_gate = ggml_new_tensor_2d(rc.ctx_w, GGML_TYPE_Q8_0, N_EMBD, N_FF);
    ggml_tensor * w_up   = ggml_new_tensor_2d(rc.ctx_w, GGML_TYPE_Q8_0, N_EMBD, N_FF);
    ggml_tensor * w_down = ggml_new_tensor_2d(rc.ctx_w, GGML_TYPE_Q8_0, N_FF, N_EMBD);
    ggml_tensor * x      = ggml_new_tensor_2d(rc.ctx, GGML_TYPE_F32, N_EMBD, m);

    ggml_tensor * cur = f.ffn ? ggml_mul(rc.ctx, ggml_rms_norm(rc.ctx, x, 1e-6f), w_norm) : x;
    // The order of the model: the gate product, then the up product
    ggml_tensor * gate = ggml_mul_mat(rc.ctx, w_gate, cur);
    ggml_tensor * up   = ggml_mul_mat(rc.ctx, w_up, cur);
    rc.h               = ggml_swiglu_split(rc.ctx, gate, up);
    rc.out             = f.ffn ? ggml_add(rc.ctx, ggml_mul_mat(rc.ctx, w_down, rc.h), x) : rc.h;
    ggml_set_output(rc.out);
    if (keep_h) {
        ggml_set_output(rc.h);
    }

    rc.gf = ggml_new_graph(rc.ctx);
    ggml_build_forward_expand(rc.gf, rc.out);
    if (keep_h) {
        ggml_build_forward_expand(rc.gf, rc.h);
    }

    rc.buf_w = ggml_backend_alloc_ctx_tensors(rc.ctx_w, be);
    rc.buf   = ggml_backend_alloc_ctx_tensors(rc.ctx, be);
    if (!rc.buf_w || !rc.buf) {
        return "the allocation failed";
    }
    if (!plain) {
        ggml_backend_buffer_set_usage(rc.buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    set_bytes(w_norm, w.norm);
    set_bytes(w_gate, w.gate);
    set_bytes(w_up, w.up);
    set_bytes(w_down, w.down);
    // The seed of x depends on the token count only
    set_bytes(x, uniform((size_t) (N_EMBD * m), 15 + (uint32_t) m, -2.0f, 2.0f));

    // After the tensor_set calls: the Hexagon backend marks the tiled weights in set_tensor. In the plain mode
    // the program keeps the first refused op and computes the graph also, for the check of graph_compute.
    for (int i = 0; i < ggml_graph_n_nodes(rc.gf); i++) {
        const ggml_tensor * node = ggml_graph_node(rc.gf, i);
        if (!ggml_backend_supports_op(be, node)) {
            const std::string what = std::string(ggml_op_desc(node)) + " of " + node->name;
            if (!plain) {
                return "the backend does not support the op " + what;
            }
            if (refused.empty()) {
                refused = what;
            }
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = ggml_backend_graph_compute(be, rc.gf) == GGML_STATUS_SUCCESS;
    us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    return ok ? "" : "the compute failed";
}

}  // namespace

int main(int argc, char ** argv) {
    std::string dev_name = "HTP0";
    bool        cpu      = false;
    bool        plain    = false;
    int         threads  = 4;
    std::string only;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cpu")) {
            cpu = true;
        } else if (!strcmp(argv[i], "--plain")) {
            plain = true;
        } else if (!strcmp(argv[i], "--dev") && i + 1 < argc) {
            dev_name = argv[++i];
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--only") && i + 1 < argc) {
            only = argv[++i];
        } else {
            fprintf(stderr, "usage: ffncheck [--dev NAME] [--cpu | --plain] [--threads N] [--only SUBSTRING]\n");
            return 2;
        }
    }
    if (cpu && plain) {
        fprintf(stderr, "ffncheck: --plain has no CPU reference. Give --cpu or --plain.\n");
        return 2;
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) {
        fprintf(stderr,
                "ffncheck: no device %s. For HTP0, set ADSP_LIBRARY_PATH to the directory of libggml-htp-v79.so.\n",
                dev_name.c_str());
        return 1;
    }
    ggml_backend_t be     = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu_be = cpu ? ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr)
                                : nullptr;
    if (!be || (cpu && !cpu_be)) {
        fprintf(stderr, "ffncheck: a backend does not start\n");
        return 1;
    }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_cpu_set_n_threads(be, threads);
    }
    if (cpu_be) {
        ggml_backend_cpu_set_n_threads(cpu_be, threads);
    }

    int n_cases = 0, n_failed = 0;
    for (const form & f : FORMS) {
        form_weights w;
        for (int64_t m : f.tokens) {
            char name[64];
            snprintf(name, sizeof(name), "%s_m%" PRId64, f.name, m);
            if (!only.empty() && std::string(name).find(only) == std::string::npos) {
                continue;
            }
            if (w.norm.empty()) {
                w = make_weights(f);
            }
            n_cases++;
            run_ctx           rc;
            double            us = 0.0;
            std::string       refused;
            const std::string err = run_case(be, f, w, m, false, plain, rc, us, refused);
            if (plain) {
                // Correct: supports_op refuses a MUL_MAT and graph_compute fails. A computed graph reads the plain
                // bytes as tiled data, thus its values are not correct.
                const bool computed = err.empty();
                const bool ok       = !refused.empty() && !computed;
                std::string values;
                if (computed) {
                    size_t nonfinite = 0;
                    for (float v : get_values(rc.out)) {
                        nonfinite += !std::isfinite(v);
                    }
                    values = " nonfinite=" + std::to_string(nonfinite);
                }
                printf("ffncheck case=%s plain supports=%s compute=%s%s%s\n", name,
                       refused.empty() ? "yes" : ("no(" + refused + ")").c_str(), computed ? "ok" : "failed",
                       values.c_str(), ok ? "" : " FAILED");
                fflush(stdout);
                n_failed += !ok;
                continue;
            }
            if (!err.empty()) {
                printf("ffncheck case=%s FAILED: %s on %s\n", name, err.c_str(), dev_name.c_str());
                fflush(stdout);
                n_failed++;
                continue;
            }
            const std::vector<float> got       = get_values(rc.out);
            size_t                   nonfinite = 0;
            for (float v : got) {
                nonfinite += !std::isfinite(v);
            }
            bool        bad = nonfinite > 0 && !f.big;
            std::string ref_text;
            if (cpu_be) {
                run_ctx           cr;
                double            cpu_us  = 0.0;
                std::string       cpu_refused;
                const std::string cpu_err = run_case(cpu_be, f, w, m, true, false, cr, cpu_us, cpu_refused);
                if (!cpu_err.empty()) {
                    ref_text = " cpu=FAILED(" + cpu_err + ")";
                    bad      = true;
                } else {
                    const std::vector<float> ref  = get_values(cr.out);
                    double                   err2 = 0.0, ref2 = 0.0, maxerr = 0.0;
                    for (size_t j = 0; j < ref.size(); j++) {
                        const double d = (double) got[j] - (double) ref[j];
                        err2 += d * d;
                        ref2 += (double) ref[j] * ref[j];
                        maxerr = std::fmax(maxerr, std::fabs(d));
                    }
                    const double nmse = ref2 > 0.0 ? err2 / ref2 : err2;
                    char         t[160];
                    snprintf(t, sizeof(t), " nmse=%.3e maxerr=%.4g hmax=%.4g ymax=%.4g", nmse, maxerr,
                             max_abs(get_values(cr.h)), max_abs(ref));
                    ref_text = t;
                    bad |= !f.big && !(nmse <= 1e-4);
                }
            }
            printf("ffncheck case=%s hash=%016" PRIx64 " nonfinite=%zu%s us=%.0f%s\n", name,
                   fnv1a(got.data(), got.size() * sizeof(float)), nonfinite, ref_text.c_str(), us, bad ? " FAILED" : "");
            fflush(stdout);
            n_failed += bad;
        }
    }
    printf("ffncheck done cases=%d failed=%d\n", n_cases, n_failed);
    ggml_backend_free(be);
    if (cpu_be) {
        ggml_backend_free(cpu_be);
    }
    return n_failed ? 1 : 0;
}
