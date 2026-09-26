// gemvcheck: the Q8_0 decode matmuls of the 4B on one ggml device, with fixed inputs.
//
// The tool builds each case as a ggml graph: MUL_MAT, MUL_MAT with the residual ADD (the backend
// fuses the two into MUL_MAT_ADD), and two or three MUL_MATs of one activation (the backend fuses
// them into MUL_MAT_NX). The weights are Q8_0 blocks with random quants and scales, and the
// activation and the residual are random values in [-1, 1]. A fixed seed for each case gives the
// same inputs in each run and on each device.
//
//   gemvcheck check [--dev HTP0] [--cases all|NAME,...] [--threads 4]
//       For each case: one compute on the device and one on the CPU backend. The tool prints the
//       FNV-1a hash of the bytes of each output of the device and the NMSE against the CPU:
//         gemvcheck: check CASE out I hash 0x... nmse X
//       Two DSP libraries give the same bits when the lines of the two runs have the same hashes.
//   gemvcheck perf [--dev HTP0] [--cases ...] [--reps 16] [--runs 5]
//       For each case: a graph of REPS copies of the case, each copy with its own activation (thus
//       the copies do not fuse with each other), one compute to warm up and RUNS timed computes.
//       The tool prints the median time of one copy and the rate of its weight bytes:
//         gemvcheck: perf CASE reps R us X gbps Y
//
// The exit code is 0 when each compute returns GGML_STATUS_SUCCESS and each NMSE is at most 5e-4,
// else 1. A usage error gives 2.
//
// Memory: the largest case (the output head, 2560 x 248320) holds 675 MB of weights on the device
// and 675 MB on the CPU, one case at a time. Time: O(bytes of the weights) for each case.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../common/fnv.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int QK = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;  // ggml_half d, then int8_t qs[32]
constexpr double NMSE_LIMIT = 5e-4;

// One case: the rows of k values of each weight (one weight: MUL_MAT; two or three: MUL_MAT_NX),
// the activation rows, and whether a residual ADD follows (one weight only).
struct Case {
    std::string           name;
    int                   k;
    std::vector<int>      m;
    int                   n;
    bool                  add;
};

// The cases of the 4B Q8_0 decode (the op profile of a token) for 1 to 4 activation rows, and two
// shapes with an odd count of 32-value k-tiles.
std::vector<Case> all_cases() {
    std::vector<Case> cases;
    for (int n = 1; n <= 4; n++) {
        const std::string r = "_n" + std::to_string(n);
        cases.push_back({ "qkv_2560x8192" + r, 2560, { 8192 }, n, false });
        cases.push_back({ "gate_2560x4096" + r, 2560, { 4096 }, n, false });
        cases.push_back({ "kv_2560x1024" + r, 2560, { 1024 }, n, false });
        cases.push_back({ "down_add_9216x2560" + r, 9216, { 2560 }, n, true });
        cases.push_back({ "out_add_4096x2560" + r, 4096, { 2560 }, n, true });
        cases.push_back({ "head_2560x248320" + r, 2560, { 248320 }, n, false });
        cases.push_back({ "nx_gate_up_2560x9216" + r, 2560, { 9216, 9216 }, n, false });
        cases.push_back({ "nx_q_v_k_2560x8192" + r, 2560, { 8192, 1024, 1024 }, n, false });
        cases.push_back({ "odd_2080x200" + r, 2080, { 200 }, n, n == 3 });
        cases.push_back({ "nx_odd_2080x200" + r, 2080, { 200, 96 }, n, false });
    }
    return cases;
}

// splitmix64: a fixed sequence for each seed
struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * (float) (next() >> 40) / (float) (1ull << 24); }
};

// The hash of n bytes. This tool uses the basis of the FNV-1a specification, which the tools of
// the memory work and of the vision work do not, thus a value of this tool is comparable with a
// value of this tool only. O(n).
uint64_t fnv1a(const void * p, size_t n) {
    return fnv::hash64(p, n, fnv::kBasisFnv1a);
}

uint64_t seed_of(const std::string & s, int salt) {
    return fnv1a(s.data(), s.size()) ^ (0x51ED27ull * (uint64_t) (salt + 1));
}

// Q8_0 rows of k values: random quants in [-127, 127] and scales in [0.002, 0.02]. O(rows * k).
std::vector<uint8_t> make_q8_0(int k, int rows, uint64_t seed) {
    Rng rng{ seed };
    const size_t blocks = (size_t) rows * (k / QK);
    std::vector<uint8_t> out(blocks * Q8_0_BLOCK_BYTES);
    for (size_t b = 0; b < blocks; b++) {
        uint8_t *         blk = out.data() + b * Q8_0_BLOCK_BYTES;
        const ggml_fp16_t d   = ggml_fp32_to_fp16(rng.uniform(0.002f, 0.02f));
        memcpy(blk, &d, sizeof(d));
        for (int i = 0; i < QK; i++) {
            blk[2 + i] = (uint8_t) (int8_t) ((int) (rng.next() % 255) - 127);
        }
    }
    return out;
}

std::vector<float> make_f32(size_t count, uint64_t seed) {
    Rng rng{ seed };
    std::vector<float> out(count);
    for (auto & v : out) {
        v = rng.uniform(-1.0f, 1.0f);
    }
    return out;
}

// The tensors and the graph of REPS copies of one case. The weights have their own context and
// buffer (wctx, wbuf), as a model loader gives them: the Hexagon backend repacks a Q8_0 weight into
// its tile layout only in a buffer with the usage GGML_BACKEND_BUFFER_USAGE_WEIGHTS. In a buffer with
// no usage it copies the bytes, and the DSP then reads plain Q8_0 blocks as tiles.
struct Built {
    ggml_context *              wctx = nullptr;
    ggml_backend_buffer_t       wbuf = nullptr;
    ggml_context *              ctx  = nullptr;
    ggml_backend_buffer_t       buf  = nullptr;
    ggml_cgraph *               gf   = nullptr;
    std::vector<ggml_tensor *>  w;
    std::vector<ggml_tensor *>  x;    // one activation for each copy
    std::vector<ggml_tensor *>  r;    // one residual for each copy (add only)
    std::vector<ggml_tensor *>  out;  // the outputs of the first copy

    ~Built() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (wbuf) {
            ggml_backend_buffer_free(wbuf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        if (wctx) {
            ggml_free(wctx);
        }
    }
};

// Builds the graph of one case with REPS copies and allocates its tensors on the backend. The weight
// buffer gets the usage GGML_BACKEND_BUFFER_USAGE_WEIGHTS before the first set_tensor of a weight
// (set_inputs). Returns false when a context or a buffer cannot be made.
bool build(const Case & c, int reps, ggml_backend_t backend, Built & b) {
    const size_t n_tensors = (size_t) reps * (2 + 2 * c.m.size());
    const size_t graph_size = std::max<size_t>(GGML_DEFAULT_GRAPH_SIZE, 4 * n_tensors);
    ggml_init_params wp = { ggml_tensor_overhead() * (c.m.size() + 1), nullptr, true };
    ggml_init_params ip = { ggml_tensor_overhead() * (n_tensors + 16) + ggml_graph_overhead_custom(graph_size, false),
                            nullptr, true };
    b.wctx = ggml_init(wp);
    b.ctx  = ggml_init(ip);
    if (!b.wctx || !b.ctx) {
        return false;
    }
    for (size_t i = 0; i < c.m.size(); i++) {
        b.w.push_back(ggml_new_tensor_2d(b.wctx, GGML_TYPE_Q8_0, c.k, c.m[i]));
    }
    b.gf = ggml_new_graph_custom(b.ctx, graph_size, false);
    for (int rep = 0; rep < reps; rep++) {
        ggml_tensor * x = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, c.k, c.n);
        b.x.push_back(x);
        for (size_t i = 0; i < c.m.size(); i++) {
            ggml_tensor * y = ggml_mul_mat(b.ctx, b.w[i], x);
            if (c.add) {
                ggml_tensor * r = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, c.m[i], c.n);
                b.r.push_back(r);
                y = ggml_add(b.ctx, y, r);
            }
            ggml_set_output(y);
            ggml_build_forward_expand(b.gf, y);
            if (rep == 0) {
                b.out.push_back(y);
            }
        }
    }
    b.wbuf = ggml_backend_alloc_ctx_tensors(b.wctx, backend);
    b.buf  = ggml_backend_alloc_ctx_tensors(b.ctx, backend);
    if (!b.wbuf || !b.buf) {
        return false;
    }
    ggml_backend_buffer_set_usage(b.wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return true;
}

// Sets the inputs of the built cases b[0] .. b[nb - 1] (one graph for each backend) to the same
// bytes. Each input is made one time. O(bytes of the inputs).
void set_inputs(const Case & c, Built * b, int nb) {
    for (size_t i = 0; i < b[0].w.size(); i++) {
        const std::vector<uint8_t> w = make_q8_0(c.k, c.m[i], seed_of(c.name, (int) i));
        for (int j = 0; j < nb; j++) {
            ggml_backend_tensor_set(b[j].w[i], w.data(), 0, w.size());
        }
    }
    for (size_t i = 0; i < b[0].x.size(); i++) {
        const std::vector<float> x = make_f32((size_t) c.k * c.n, seed_of(c.name, 100 + (int) i));
        for (int j = 0; j < nb; j++) {
            ggml_backend_tensor_set(b[j].x[i], x.data(), 0, x.size() * sizeof(float));
        }
    }
    for (size_t i = 0; i < b[0].r.size(); i++) {
        const std::vector<float> r = make_f32(ggml_nelements(b[0].r[i]), seed_of(c.name, 1000 + (int) i));
        for (int j = 0; j < nb; j++) {
            ggml_backend_tensor_set(b[j].r[i], r.data(), 0, r.size() * sizeof(float));
        }
    }
}

std::vector<float> get_f32(ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

double nmse(const std::vector<float> & a, const std::vector<float> & ref) {
    double err = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) ref[i];
        err += d * d;
        den += (double) ref[i] * (double) ref[i];
    }
    return den > 0.0 ? err / den : err;
}

// check: one compute on the device and on the CPU, the hash and the NMSE of each output
bool run_check(const Case & c, ggml_backend_t dev, ggml_backend_t cpu) {
    Built b[2];
    Built & bd = b[0];
    Built & bc = b[1];
    if (!build(c, 1, dev, bd) || !build(c, 1, cpu, bc)) {
        printf("gemvcheck: check %s ALLOC-FAIL\n", c.name.c_str());
        return false;
    }
    set_inputs(c, b, 2);
    const ggml_status sd = ggml_backend_graph_compute(dev, bd.gf);
    const ggml_status sc = ggml_backend_graph_compute(cpu, bc.gf);
    if (sd != GGML_STATUS_SUCCESS || sc != GGML_STATUS_SUCCESS) {
        printf("gemvcheck: check %s STATUS device %d cpu %d\n", c.name.c_str(), (int) sd, (int) sc);
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < bd.out.size(); i++) {
        const std::vector<float> got = get_f32(bd.out[i]);
        const std::vector<float> ref = get_f32(bc.out[i]);
        const double e = nmse(got, ref);
        const bool   finite = std::all_of(got.begin(), got.end(), [](float v) { return std::isfinite(v); });
        ok = ok && finite && e <= NMSE_LIMIT;
        printf("gemvcheck: check %s out %zu hash 0x%016" PRIx64 " nmse %.3e%s\n", c.name.c_str(), i,
               fnv1a(got.data(), got.size() * sizeof(float)), e,
               !finite ? " NONFINITE" : (e > NMSE_LIMIT ? " FAIL" : ""));
    }
    fflush(stdout);
    return ok;
}

// perf: the median time of one copy over RUNS computes of a graph of REPS copies
bool run_perf(const Case & c, ggml_backend_t dev, int reps, int runs) {
    Built b;
    if (!build(c, reps, dev, b)) {
        printf("gemvcheck: perf %s ALLOC-FAIL\n", c.name.c_str());
        return false;
    }
    set_inputs(c, &b, 1);
    if (ggml_backend_graph_compute(dev, b.gf) != GGML_STATUS_SUCCESS) {
        printf("gemvcheck: perf %s STATUS\n", c.name.c_str());
        return false;
    }
    std::vector<double> us;
    for (int i = 0; i < runs; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        const ggml_status s = ggml_backend_graph_compute(dev, b.gf);
        ggml_backend_synchronize(dev);
        const auto t1 = std::chrono::steady_clock::now();
        if (s != GGML_STATUS_SUCCESS) {
            printf("gemvcheck: perf %s STATUS\n", c.name.c_str());
            return false;
        }
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / reps);
    }
    std::sort(us.begin(), us.end());
    const double med = us[us.size() / 2];
    double bytes = 0.0;
    for (int m : c.m) {
        bytes += (double) m * (c.k / QK) * Q8_0_BLOCK_BYTES;
    }
    printf("gemvcheck: perf %s reps %d us %.1f min %.1f max %.1f gbps %.2f\n", c.name.c_str(), reps, med, us.front(),
           us.back(), bytes / med / 1e3);
    fflush(stdout);
    return true;
}

[[noreturn]] void usage() {
    fprintf(stderr, "usage: gemvcheck check|perf [--dev HTP0] [--cases all|NAME,...] [--threads 4] [--reps 16] "
                    "[--runs 5]\n");
    exit(2);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage();
    }
    const std::string mode = argv[1];
    if (mode != "check" && mode != "perf") {
        usage();
    }
    std::string dev_name = "HTP0", filter = "all";
    int threads = 4, reps = 16, runs = 5;
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        if (i + 1 >= argc) {
            usage();
        }
        if (a == "--dev") {
            dev_name = argv[++i];
        } else if (a == "--cases") {
            filter = argv[++i];
        } else if (a == "--threads") {
            threads = atoi(argv[++i]);
        } else if (a == "--reps") {
            reps = atoi(argv[++i]);
        } else if (a == "--runs") {
            runs = atoi(argv[++i]);
        } else {
            usage();
        }
    }
    if (threads < 1 || reps < 1 || runs < 1) {
        usage();
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) {
        fprintf(stderr, "gemvcheck: no device %s\n", dev_name.c_str());
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu     = ggml_backend_cpu_init();
    if (!backend || !cpu) {
        fprintf(stderr, "gemvcheck: the backends did not start\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu, threads);

    bool ok = true;
    int  n_run = 0;
    for (const Case & c : all_cases()) {
        if (filter != "all" && ("," + filter + ",").find("," + c.name + ",") == std::string::npos) {
            continue;
        }
        n_run++;
        ok = (mode == "check" ? run_check(c, backend, cpu) : run_perf(c, backend, reps, runs)) && ok;
    }
    printf("gemvcheck: %s %d cases %s\n", mode.c_str(), n_run, ok ? "OK" : "FAIL");
    ggml_backend_free(cpu);
    ggml_backend_free(backend);
    return ok && n_run > 0 ? 0 : 1;
}
