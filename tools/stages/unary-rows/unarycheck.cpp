// unarycheck: the pointwise unary ops of the gated delta net layers of Qwen3.5 on HTP0 against the CPU backend,
// in the shapes of the model. The stage unary-rows runs it with each value of GGML_HEXAGON_UNARY_FLAT, thus it
// compares the old rows of these ops (the value 0, or the HEAD libraries) with the new rows (the value 1).
//
// The cases:
//   sigmoid_beta_t<T>   SIGMOID of [1, 32, T], the beta gate (rows of one element before the change)
//   softplus_gate_t<T>  SOFTPLUS of [32, T], the gate (rows of 32 elements before the change)
//   scale0_s            SCALE by 0, in place, of slot 1 of a state of 4 slots of 524288 elements (one row of
//                       2 MiB before the change), as build_rs of llama-graph.cpp clears a recurrent state
//   scale0_r            the same with slots of 24576 elements (the conv state)
//   chain_beta_t<T>     24 layers of SIGMOID of [1, 32, T] and a MUL by a weight [1, 32, 1] in one graph
//   chain_gate_t<T>     24 layers of an ADD of a bias [32], SOFTPLUS and a MUL by a weight [32] in one graph
// T is 1024 (a prompt ubatch) and 1 (a decode token). The graph allocator of ggml-alloc gives the memory of each
// case, thus the chains reuse the memory of their intermediate tensors, as the graphs of the model do. Each
// tensor has a fixed seed, thus two runs give the same bytes when the device computes the same values.
//
// Usage: unarycheck [--cpu] [--threads N] [--only SUBSTRING] [--dump DIR] [--device NAME]
//   --cpu        also run each case on the CPU backend and compare
//   --dump DIR   write the output of each case of one op (sigmoid_*, softplus_*) to DIR/<case>.f32
//   --device     the device of the cases (preset HTP0). CPU tests this program on a machine with no HTP0.
// Output: one line for each case,
//   unarycheck case=<name> flat=<value> hash=<16 hex digits> nonfinite=<n>
//       [nmse=<x> maxerr=<y> differ=<elements with other bits than the CPU>]
//       [slot_nonzero=<n> other_changed=<n>] us=<t>
// and at the end "unarycheck done cases=<n> failed=<n>". A case fails when the device does not support an op,
// when the compute fails, when an output value is not finite, with --cpu when its NMSE is more than 1e-6, and
// for a SCALE case when an element of the slot is not zero or an element of a different slot changed. The
// exit code is 1 when a case fails. Time: O(elements) for each case and backend.
//
// Build (the NDK clang++, against the libraries of a llama.cpp tree):
//   clang++ -O2 -std=c++17 -I TREE/ggml/include unarycheck.cpp -o unarycheck -L BUILD/bin -lggml -lggml-cpu \
//       -lggml-base -Wl,-rpath,'$ORIGIN/../lib'

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

constexpr int N_LAYERS = 24;
constexpr int N_HEADS  = 32;

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

// An input tensor of a case and its values
struct input {
    ggml_tensor *      t = nullptr;
    std::vector<float> v;
};

// The graph of one case on one backend. The inputs are graph inputs, and the outputs are graph outputs, thus the
// allocator keeps them. For a SCALE case, the output is the state and "slot" gives the cleared elements.
struct graph_case {
    ggml_context *      ctx   = nullptr;
    ggml_gallocr_t      alloc = nullptr;
    ggml_cgraph *       gf    = nullptr;
    std::vector<input>  ins;
    std::vector<ggml_tensor *> outs;
    int64_t             slot_first = -1;  // the first element of the cleared slot, -1 when no SCALE case
    int64_t             slot_n     = 0;

    ~graph_case() {
        if (alloc) {
            ggml_gallocr_free(alloc);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }

    ggml_tensor * new_input(ggml_tensor * t, uint32_t seed, float lo, float hi) {
        ggml_set_input(t);
        ins.push_back({ t, uniform((size_t) ggml_nelements(t), seed, lo, hi) });
        return t;
    }
};

// Build the graph of the case "name" in gc. Returns false for an unknown name.
bool build_case(const std::string & name, graph_case & gc) {
    ggml_init_params ip = { 512 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    gc.ctx              = ggml_init(ip);
    gc.gf               = ggml_new_graph(gc.ctx);
    ggml_context * c    = gc.ctx;
    const uint32_t base = (uint32_t) fnv1a(name.data(), name.size());

    const int64_t t_tokens = name.size() > 6 && name.compare(name.size() - 6, 6, "_t1024") == 0 ? 1024 : 1;
    std::vector<ggml_tensor *> outs;

    if (name.rfind("sigmoid_beta_", 0) == 0) {
        ggml_tensor * x = gc.new_input(ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, N_HEADS, t_tokens), base + 1, -12.0f, 12.0f);
        outs.push_back(ggml_sigmoid(c, x));
    } else if (name.rfind("softplus_gate_", 0) == 0) {
        ggml_tensor * x = gc.new_input(ggml_new_tensor_2d(c, GGML_TYPE_F32, N_HEADS, t_tokens), base + 1, -12.0f, 12.0f);
        outs.push_back(ggml_softplus(c, x));
    } else if (name == "scale0_s" || name == "scale0_r") {
        const int64_t n     = name == "scale0_s" ? 524288 : 24576;
        ggml_tensor * state = gc.new_input(ggml_new_tensor_2d(c, GGML_TYPE_F32, n, 4), base + 1, -1.0f, 1.0f);
        ggml_set_output(state);
        ggml_tensor * slot  = ggml_view_1d(c, state, n, 1 * state->nb[1]);
        ggml_build_forward_expand(gc.gf, ggml_scale_inplace(c, slot, 0.0f));
        gc.outs       = { state };
        gc.slot_first = n;
        gc.slot_n     = n;
        return true;
    } else if (name.rfind("chain_beta_", 0) == 0) {
        for (int l = 0; l < N_LAYERS; l++) {
            ggml_tensor * b = gc.new_input(ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, N_HEADS, t_tokens), base + 10 * l + 1,
                                           -12.0f, 12.0f);
            ggml_tensor * w = gc.new_input(ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, N_HEADS, 1), base + 10 * l + 2, 0.5f, 1.5f);
            outs.push_back(ggml_mul(c, ggml_sigmoid(c, b), w));
        }
    } else if (name.rfind("chain_gate_", 0) == 0) {
        for (int l = 0; l < N_LAYERS; l++) {
            ggml_tensor * a    = gc.new_input(ggml_new_tensor_2d(c, GGML_TYPE_F32, N_HEADS, t_tokens), base + 10 * l + 1,
                                              -6.0f, 6.0f);
            ggml_tensor * bias = gc.new_input(ggml_new_tensor_1d(c, GGML_TYPE_F32, N_HEADS), base + 10 * l + 2, -6.0f, 6.0f);
            ggml_tensor * w    = gc.new_input(ggml_new_tensor_1d(c, GGML_TYPE_F32, N_HEADS), base + 10 * l + 3, -2.0f, -0.1f);
            outs.push_back(ggml_mul(c, ggml_softplus(c, ggml_add(c, a, bias)), w));
        }
    } else {
        return false;
    }
    for (ggml_tensor * o : outs) {
        ggml_set_output(o);
        ggml_build_forward_expand(gc.gf, o);
    }
    gc.outs = outs;
    return true;
}

// The first op of the graph that the backend does not support, or nullptr
const ggml_tensor * unsupported(ggml_backend_t be, ggml_cgraph * gf) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        const ggml_tensor * n = ggml_graph_node(gf, i);
        if (n->op != GGML_OP_VIEW && n->op != GGML_OP_RESHAPE && !ggml_backend_supports_op(be, n)) {
            return n;
        }
    }
    return nullptr;
}

// Allocate the graph of gc on the backend, give it its inputs, compute it, and read the outputs into out (all
// outputs one after the other). Returns an error text, or an empty text.
std::string run_case(ggml_backend_t be, graph_case & gc, std::vector<float> & out, double & us) {
    if (const ggml_tensor * n = unsupported(be, gc.gf)) {
        return std::string("the backend does not support ") + ggml_op_desc(n) + " " + n->name;
    }
    gc.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!gc.alloc || !ggml_gallocr_alloc_graph(gc.alloc, gc.gf)) {
        return "the allocation failed";
    }
    for (const input & in : gc.ins) {
        ggml_backend_tensor_set(in.t, in.v.data(), 0, in.v.size() * sizeof(float));
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(be, gc.gf) != GGML_STATUS_SUCCESS) {
        return "the compute failed";
    }
    us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    out.clear();
    for (const ggml_tensor * o : gc.outs) {
        const size_t n0 = out.size();
        out.resize(n0 + (size_t) ggml_nelements(o));
        ggml_backend_tensor_get(o, out.data() + n0, 0, (size_t) ggml_nelements(o) * sizeof(float));
    }
    return "";
}

const std::vector<std::string> CASES = {
    "sigmoid_beta_t1024", "sigmoid_beta_t1", "softplus_gate_t1024", "softplus_gate_t1",
    "scale0_s",           "scale0_r",        "chain_beta_t1024",    "chain_beta_t1",
    "chain_gate_t1024",   "chain_gate_t1",
};

}  // namespace

int main(int argc, char ** argv) {
    bool        cpu     = false;
    int         threads = 4;
    std::string only, dump;
    std::string device  = "HTP0";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cpu")) {
            cpu = true;
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--only") && i + 1 < argc) {
            only = argv[++i];
        } else if (!strcmp(argv[i], "--dump") && i + 1 < argc) {
            dump = argv[++i];
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = argv[++i];
        } else {
            fprintf(stderr, "usage: unarycheck [--cpu] [--threads N] [--only SUBSTRING] [--dump DIR] [--device NAME]\n");
            return 2;
        }
    }
    const char * flat = getenv("GGML_HEXAGON_UNARY_FLAT");

    ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
    if (!dev) {
        fprintf(stderr, "unarycheck: no device %s. For HTP0, set ADSP_LIBRARY_PATH to the directory of "
                        "libggml-htp-v79.so.\n", device.c_str());
        return 1;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (be && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_cpu_set_n_threads(be, threads);
    }
    ggml_backend_t cpu_be = cpu ? ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr)
                                : nullptr;
    if (!be || (cpu && !cpu_be)) {
        fprintf(stderr, "unarycheck: a backend does not start\n");
        return 1;
    }
    if (cpu_be) {
        ggml_backend_cpu_set_n_threads(cpu_be, threads);
    }

    int n_cases = 0, n_failed = 0;
    for (const std::string & name : CASES) {
        if (!only.empty() && name.find(only) == std::string::npos) {
            continue;
        }
        n_cases++;
        graph_case gc;
        build_case(name, gc);
        std::vector<float> got;
        double             us  = 0.0;
        const std::string  err = run_case(be, gc, got, us);
        if (!err.empty()) {
            printf("unarycheck case=%s flat=%s FAILED: %s on %s\n", name.c_str(), flat ? flat : "unset", err.c_str(),
                   device.c_str());
            n_failed++;
            continue;
        }
        size_t nonfinite = 0;
        for (float f : got) {
            nonfinite += !std::isfinite(f);
        }
        bool        bad = nonfinite > 0;
        std::string extra;
        if (gc.slot_first >= 0) {
            // The input of the state is the first input, and only the slot must change
            const std::vector<float> & before = gc.ins[0].v;
            size_t slot_nonzero = 0, other_changed = 0;
            for (size_t i = 0; i < got.size(); i++) {
                const bool in_slot = (int64_t) i >= gc.slot_first && (int64_t) i < gc.slot_first + gc.slot_n;
                if (in_slot) {
                    slot_nonzero += got[i] != 0.0f;
                } else {
                    other_changed += memcmp(&got[i], &before[i], sizeof(float)) != 0;
                }
            }
            char t[96];
            snprintf(t, sizeof(t), " slot_nonzero=%zu other_changed=%zu", slot_nonzero, other_changed);
            extra += t;
            bad |= slot_nonzero > 0 || other_changed > 0;
        }
        if (cpu_be) {
            graph_case         cc;
            std::vector<float> ref;
            double             cpu_us = 0.0;
            build_case(name, cc);
            const std::string cerr = run_case(cpu_be, cc, ref, cpu_us);
            if (!cerr.empty() || ref.size() != got.size()) {
                extra += " cpu=FAILED";
                bad = true;
            } else {
                double err2 = 0.0, ref2 = 0.0, maxerr = 0.0;
                size_t differ = 0;
                for (size_t j = 0; j < ref.size(); j++) {
                    const double d = (double) got[j] - (double) ref[j];
                    err2 += d * d;
                    ref2 += (double) ref[j] * ref[j];
                    maxerr = std::fmax(maxerr, std::fabs(d));
                    differ += memcmp(&got[j], &ref[j], sizeof(float)) != 0;
                }
                const double nmse = ref2 > 0.0 ? err2 / ref2 : err2;
                char         t[128];
                snprintf(t, sizeof(t), " nmse=%.3e maxerr=%.4g differ=%zu", nmse, maxerr, differ);
                extra += t;
                bad |= !(nmse <= 1e-6);
            }
        }
        if (!dump.empty() && (name.rfind("sigmoid_", 0) == 0 || name.rfind("softplus_", 0) == 0)) {
            const std::string path = dump + "/" + name + ".f32";
            FILE *            f    = fopen(path.c_str(), "wb");
            if (!f || fwrite(got.data(), sizeof(float), got.size(), f) != got.size()) {
                fprintf(stderr, "unarycheck: cannot write %s\n", path.c_str());
                bad = true;
            }
            if (f) {
                fclose(f);
            }
        }
        printf("unarycheck case=%s flat=%s hash=%016" PRIx64 " nonfinite=%zu%s us=%.0f%s\n", name.c_str(),
               flat ? flat : "unset", fnv1a(got.data(), got.size() * sizeof(float)), nonfinite, extra.c_str(), us,
               bad ? " FAILED" : "");
        fflush(stdout);
        n_failed += bad;
    }
    printf("unarycheck done cases=%d failed=%d\n", n_cases, n_failed);
    ggml_backend_free(be);
    if (cpu_be) {
        ggml_backend_free(cpu_be);
    }
    return n_failed ? 1 : 0;
}
