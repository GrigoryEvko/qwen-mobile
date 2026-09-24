// mmcheck: a hash of the output of each MUL_MAT case of the 4B shapes on HTP0, for the bit-exact check of two
// chunk selections of the HMX 2D matmul (GGML_HEXAGON_MM_SOLVER=0 against 1, or GGML_HEXAGON_MM_CHUNKS).
//
// test-backend-ops draws new inputs for each run, thus two runs cannot give the same bytes. This program gives
// each case the same inputs in each run: a fixed seed for each tensor. It runs each case on HTP0, writes the
// FNV-1a hash of the output bytes, and with --cpu also the NMSE and the largest error against the CPU
// backend of the phone (a check that the output is correct, not only equal).
//
// The forms of a case, as the graph of the model has them (GGML_HEXAGON_OPFUSION=1 fuses them):
//   mm   dst = W x
//   nx   g = W x, u = W2 x (two products of one activation: MUL_MAT_NX)
//   add  dst = W x + r (the product and the residual: MUL_MAT_ADD)
//
// Usage: mmcheck [--cpu] [--threads N] [--only SUBSTRING] [--device NAME]
//   --device NAME runs the cases on another device than HTP0, for example CPU for a test of this program on a
//   machine with no HTP0.
// Output: one line for each case,
//   mmcheck case=<name> hash=<16 hex digits>[,<16 hex digits>] nonfinite=<n> [nmse=<x> maxerr=<y>] us=<t>
// and at the end "mmcheck done cases=<n> failed=<n>". A case fails when its compute fails, when an output value
// is not finite, or with --cpu when its NMSE is more than 1e-4. The exit code is 1 when a case fails.
//
// Build (the NDK clang++, against the libraries of a llama.cpp tree):
//   clang++ -O2 -std=c++17 -I TREE/ggml/include mmcheck.cpp -o mmcheck -L BUILD/bin -lggml -lggml-cpu \
//       -lggml-base -Wl,-rpath,'$ORIGIN/../lib'
// Time: O(sum of k x n x m) for each backend.

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
#include <string>
#include <vector>

namespace {

struct shape {
    const char * name;
    ggml_type    type;
    int64_t      k;
    int64_t      n;
    const char * form;  // "mm", "nx" or "add"
    std::vector<int64_t> tokens;
};

// The 4B shapes at the token counts where the old and the new chunk model differ (512 to 1024) and where
// they agree (5, 64), and odd counts that give a partial last token chunk.
const std::vector<shape> SHAPES = {
    { "ffn_down", GGML_TYPE_Q8_0, 9216, 2560, "add", { 5, 64, 256, 512, 1000, 1024 } },
    { "ffn_gate_up", GGML_TYPE_Q8_0, 2560, 9216, "nx", { 5, 64, 512, 896, 1024 } },
    { "attn_qkv", GGML_TYPE_Q8_0, 2560, 8192, "mm", { 5, 64, 512, 1024 } },
    { "attn_gate", GGML_TYPE_Q8_0, 2560, 4096, "mm", { 5, 64, 512, 1024 } },
    { "ssm_out", GGML_TYPE_Q8_0, 4096, 2560, "add", { 5, 64, 512, 768, 896, 1024 } },
    { "attn_k", GGML_TYPE_Q8_0, 2560, 1024, "mm", { 5, 64, 512, 1024 } },
    { "mtp_ffn_down", GGML_TYPE_F16, 9216, 2560, "mm", { 256, 1024 } },
    { "mtp_eh_proj", GGML_TYPE_F16, 5120, 2560, "mm", { 640, 1024 } },
};

uint64_t fnv1a(const void * data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// Uniform values in [lo, hi) from a seed (xorshift64*, 24 bits for each value). The same seed gives the same
// values on each run and on each machine. O(n).
std::vector<float> uniform(size_t n, uint32_t seed, float lo, float hi) {
    uint64_t           s = 0x9e3779b97f4a7c15ull ^ ((uint64_t) seed << 1 | 1);
    std::vector<float> v(n);
    for (float & x : v) {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        const uint32_t r = (uint32_t) ((s * 0x2545f4914f6cdd1dull) >> 40);
        x = lo + (hi - lo) * ((float) r * (1.0f / 16777216.0f));
    }
    return v;
}

// The input values of one case, made one time for the two backends
struct inputs {
    std::vector<float> w, w2, x, r;
};

// Set a tensor from f32 values: a quantization for Q8_0, a conversion for F16.
void set_values(ggml_tensor * t, const std::vector<float> & v) {
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
        return;
    }
    std::vector<uint8_t> buf(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(v.data(), (ggml_fp16_t *) buf.data(), (int64_t) v.size());
    } else {
        ggml_quantize_chunk(t->type, v.data(), buf.data(), 0, t->ne[1], t->ne[0], nullptr);
    }
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

std::vector<float> get_values(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

// The tensors and the graph of one case on one backend.
struct run_ctx {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph *         gf  = nullptr;
    std::vector<ggml_tensor *> outs;

    ~run_ctx() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

// The inputs of one case. The seeds depend on the shape and the tensor, thus a weight is the same for each token
// count, and the activation and the residual depend on the token count too.
inputs make_inputs(const shape & s, int64_t m) {
    const uint32_t base = (uint32_t) fnv1a(s.name, strlen(s.name));
    inputs         in;
    in.w = uniform((size_t) (s.k * s.n), base + 1, -1.0f, 1.0f);
    if (!strcmp(s.form, "nx")) {
        in.w2 = uniform((size_t) (s.k * s.n), base + 2, -1.0f, 1.0f);
    }
    in.x = uniform((size_t) (s.k * m), base + 3 + (uint32_t) m, -1.0f, 1.0f);
    if (!strcmp(s.form, "add")) {
        in.r = uniform((size_t) (s.n * m), base + 4 + (uint32_t) m, -1.0f, 1.0f);
    }
    return in;
}

// Build the graph of one case, give it its inputs, and compute it on the backend. Returns false when the
// allocation or the compute fails.
bool run_case(ggml_backend_t be, const shape & s, int64_t m, const inputs & in, run_ctx & rc, double & us) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    rc.ctx              = ggml_init(ip);
    ggml_tensor * w     = ggml_new_tensor_2d(rc.ctx, s.type, s.k, s.n);
    ggml_tensor * x     = ggml_new_tensor_2d(rc.ctx, GGML_TYPE_F32, s.k, m);
    ggml_tensor * w2    = nullptr;
    ggml_tensor * r     = nullptr;
    rc.gf               = ggml_new_graph(rc.ctx);
    if (!strcmp(s.form, "nx")) {
        w2 = ggml_new_tensor_2d(rc.ctx, s.type, s.k, s.n);
        rc.outs.push_back(ggml_mul_mat(rc.ctx, w, x));
        rc.outs.push_back(ggml_mul_mat(rc.ctx, w2, x));
    } else if (!strcmp(s.form, "add")) {
        r = ggml_new_tensor_2d(rc.ctx, GGML_TYPE_F32, s.n, m);
        rc.outs.push_back(ggml_add(rc.ctx, ggml_mul_mat(rc.ctx, w, x), r));
    } else {
        rc.outs.push_back(ggml_mul_mat(rc.ctx, w, x));
    }
    for (ggml_tensor * o : rc.outs) {
        ggml_build_forward_expand(rc.gf, o);
    }
    rc.buf = ggml_backend_alloc_ctx_tensors(rc.ctx, be);
    if (!rc.buf) {
        return false;
    }
    set_values(w, in.w);
    if (w2) {
        set_values(w2, in.w2);
    }
    set_values(x, in.x);
    if (r) {
        set_values(r, in.r);
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = ggml_backend_graph_compute(be, rc.gf) == GGML_STATUS_SUCCESS;
    us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    bool        cpu     = false;
    int         threads = 4;
    std::string only;
    std::string device  = "HTP0";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cpu")) {
            cpu = true;
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--only") && i + 1 < argc) {
            only = argv[++i];
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = argv[++i];
        } else {
            fprintf(stderr, "usage: mmcheck [--cpu] [--threads N] [--only SUBSTRING] [--device NAME]\n");
            return 2;
        }
    }

    ggml_backend_dev_t htp_dev = ggml_backend_dev_by_name(device.c_str());
    if (!htp_dev) {
        fprintf(stderr, "mmcheck: no device %s. For HTP0, set ADSP_LIBRARY_PATH to the directory of "
                        "libggml-htp-v79.so.\n", device.c_str());
        return 1;
    }
    ggml_backend_t htp = ggml_backend_dev_init(htp_dev, nullptr);
    if (htp && ggml_backend_dev_type(htp_dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_cpu_set_n_threads(htp, threads);
    }
    ggml_backend_t cpu_be = cpu ? ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr)
                                : nullptr;
    if (!htp || (cpu && !cpu_be)) {
        fprintf(stderr, "mmcheck: a backend does not start\n");
        return 1;
    }
    if (cpu_be) {
        ggml_backend_cpu_set_n_threads(cpu_be, threads);
    }

    int n_cases = 0, n_failed = 0;
    for (const shape & s : SHAPES) {
        for (int64_t m : s.tokens) {
            char name[128];
            snprintf(name, sizeof(name), "%s_%s_%" PRId64 "x%" PRId64 "_m%" PRId64 "_%s", s.name, ggml_type_name(s.type),
                     s.k, s.n, m, s.form);
            if (!only.empty() && std::string(name).find(only) == std::string::npos) {
                continue;
            }
            n_cases++;
            const inputs in = make_inputs(s, m);
            run_ctx      rc;
            double       us = 0.0;
            if (!run_case(htp, s, m, in, rc, us)) {
                printf("mmcheck case=%s FAILED: the allocation or the compute on %s failed\n", name, device.c_str());
                n_failed++;
                continue;
            }
            std::string hashes;
            size_t      nonfinite = 0;
            std::vector<std::vector<float>> got;
            for (ggml_tensor * o : rc.outs) {
                got.push_back(get_values(o));
                const std::vector<float> & v = got.back();
                for (float f : v) {
                    nonfinite += !std::isfinite(f);
                }
                char h[20];
                snprintf(h, sizeof(h), "%016" PRIx64, fnv1a(v.data(), v.size() * sizeof(float)));
                hashes += (hashes.empty() ? "" : ",") + std::string(h);
            }
            bool bad = nonfinite > 0;
            std::string ref_text;
            if (cpu_be) {
                run_ctx cr;
                double  cpu_us = 0.0;
                if (!run_case(cpu_be, s, m, in, cr, cpu_us)) {
                    ref_text = " cpu=FAILED";
                    bad      = true;
                } else {
                    double err2 = 0.0, ref2 = 0.0, maxerr = 0.0;
                    for (size_t i = 0; i < cr.outs.size(); i++) {
                        const std::vector<float> ref = get_values(cr.outs[i]);
                        for (size_t j = 0; j < ref.size(); j++) {
                            const double d = (double) got[i][j] - (double) ref[j];
                            err2 += d * d;
                            ref2 += (double) ref[j] * ref[j];
                            maxerr = std::fmax(maxerr, std::fabs(d));
                        }
                    }
                    const double nmse = ref2 > 0.0 ? err2 / ref2 : err2;
                    char         t[96];
                    snprintf(t, sizeof(t), " nmse=%.3e maxerr=%.4g", nmse, maxerr);
                    ref_text = t;
                    bad |= !(nmse <= 1e-4);
                }
            }
            printf("mmcheck case=%s hash=%s nonfinite=%zu%s us=%.0f%s\n", name, hashes.c_str(), nonfinite,
                   ref_text.c_str(), us, bad ? " FAILED" : "");
            fflush(stdout);
            n_failed += bad;
        }
    }
    printf("mmcheck done cases=%d failed=%d\n", n_cases, n_failed);
    ggml_backend_free(htp);
    if (cpu_be) {
        ggml_backend_free(cpu_be);
    }
    return n_failed ? 1 : 0;
}
