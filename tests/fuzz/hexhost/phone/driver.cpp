// hexhost_phone: the phone driver of the hexhost fuzz targets.
//
// The driver reads graph inputs of the x86 target fuzz_graph (the same bytes give the same
// Qwen3.5-shaped graphs, refer to graphgen.h) and runs each input two times: on the real NPU
// (HTP0, with the fusions, the op batches, the graph cache and the batch cache of the host) and
// on the CPU backend. It compares the outputs and the caches of the two runs after each step.
// The GGML_HEXAGON_* environment variables of the command select the host switches, thus one
// command runs one configuration.
//
//   hexhost_phone [--seconds S] [--nmse X] [--device NAME] [--random N] [--seed S]
//                 [--save DIR [--keep]] [--detail] [--nodes] FILE_OR_DIRECTORY...
//   hexhost_phone [--nmse X] [--device NAME] --gdn-sweep
//   hexhost_phone --selftest-threads
//
// --random N adds N inputs of 64 to 1024 random bytes from the seed (a fixed default), thus a
// run can be done again. --device CPU runs the two sides on the CPU (a check of the driver).
// --save DIR writes each random input that does not give "ok" into DIR (--keep: each random
// input). --detail prints each tensor that differs, with the model params of the input.
// --nodes gives each node its own bytes (no ggml-alloc reuse), compares each node, and prints
// the root nodes of the first step that has a difference. --gdn-sweep runs single GATED_DELTA_NET
// graphs of many shapes (S_v, heads, tokens, snapshot slots) on the device and on the CPU, and
// compares the attention part, each written snapshot slot, and a reader of the first slot.
// --selftest-threads starts one thread and stops: the first step of a phone ASan run.
//
// Output: one line for each input ("ok", "skip", "mismatch" or "error") and a summary line.
// The exit code is 1 when an input gives a mismatch or an error, else 0.

#include "fake_dsp.h"
#include "graphgen.h"
#include "harness.h"
#include "hexhost.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <fuzzer/FuzzedDataProvider.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <map>
#include <random>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

// ---- The functions of the x86 harness that graphgen calls, for the real devices

namespace hexhost {

struct device {
    ggml_backend_dev_t dev = nullptr;
};

ggml_backend_dev_t device_dev(device * d) {
    return d->dev;
}

ggml_backend_buffer_type_t device_buft(device * d) {
    return ggml_backend_dev_buffer_type(d->dev);
}

bool supports_op(device * d, const ggml_tensor * op) {
    return ggml_backend_dev_supports_op(d->dev, op);
}

} // namespace hexhost

namespace harness {
thread_local jmp_buf * g_guard = nullptr;
}

namespace fakedsp {

namespace {
std::map<std::string, uint64_t> g_counts;
}

void count(const std::string & name, uint64_t n) {
    g_counts[name] += n;
}

bool is_ignored(const char * id) {
    const char * list = getenv("HEXHOST_IGNORE");
    if (!list) {
        return false;
    }
    const std::string l = std::string(",") + list + ",";
    return l.find(std::string(",") + id + ",") != std::string::npos;
}

} // namespace fakedsp

namespace {

// Returns to the active guard of graphgen, or prints the message (ggml then aborts).
void on_abort(const char * message) {
    if (harness::g_guard) {
        longjmp(*harness::g_guard, 1);
    }
    fprintf(stderr, "hexhost_phone: GGML ABORT: %s\n", message);
    fflush(stderr);
}

// Writes the ggml log to stderr only when HEXHOST_LOG is set, else only the errors and the warnings.
void log_filter(enum ggml_log_level level, const char * text, void * user) {
    (void) user;
    static const bool on = getenv("HEXHOST_LOG") != nullptr;
    if (on || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

// The values of one tensor after one step
struct snapshot {
    std::string        name;
    std::string        what;          // the op, the type, the shape and the sources (refer to describe)
    uint32_t           step  = 0;
    size_t             graph = 0;
    int                node  = -1;    // the index in graph_spec::order, or -2 - k for the cache k
    std::vector<int>   srcs;          // the same indices for the owners of the sources, -1 for a weight or an input
    int64_t            row  = 0;      // the elements of one row (ne[0])
    std::vector<float> v;
};

// The result of one run of an input on one device
struct run_result {
    bool                  built = false;   // the device runs every node of the graphs
    std::string           desc;
    std::string           params;          // the model params and the mutations of each layer
    std::vector<snapshot> snaps;
};

// The run modes of the driver
struct run_mode {
    bool detail = false;   // print each tensor that differs at each step
    bool nodes  = false;   // no ggml-alloc reuse, and a snapshot of each node (refer to world::no_reuse)
};

// Gives the tensor that owns the bytes of t: t, or the source of the view.
const ggml_tensor * owner(const ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

// Gives a short text for a tensor: the name, the op, the type, the shape, and the name and the
// op of the owner of each source.
std::string describe(const ggml_tensor * t) {
    char buf[160];
    snprintf(buf, sizeof(buf), "%s %s %s [%lld,%lld,%lld,%lld] <-", t->name[0] ? t->name : "(unnamed)", ggml_op_desc(t),
             ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    std::string s = buf;
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        if (!t->src[j]) {
            continue;
        }
        const ggml_tensor * o = owner(t->src[j]);
        s += " ";
        s += o->name[0] ? o->name : "(unnamed)";
        s += ":";
        s += o->op == GGML_OP_NONE ? "leaf" : ggml_op_desc(o);
        if (t->src[j] != o) {
            s += "(view)";
        }
    }
    return s;
}

// Gives the model params and the mutations of a world in one line.
std::string describe_params(const graphgen::world & w) {
    const graphgen::model_params & mp = w.mp;
    std::string                    s  = "embd=" + std::to_string(mp.n_embd) + " S_v=" + std::to_string(mp.S_v) +
                     " H_k=" + std::to_string(mp.H_k) + " H_v=" + std::to_string(mp.H_v) + " mem=" + std::to_string(mp.mem) +
                     " K=" + std::to_string(mp.K) + " hd=" + std::to_string(mp.hd) + " n_head=" + std::to_string(mp.n_head) +
                     " n_kv=" + std::to_string(mp.n_kv) + " mtp=" + std::to_string(mp.mtp) + " out_ids=" + std::to_string(mp.out_ids) +
                     " graphs=" + std::to_string(w.graphs.size());
    for (const auto & g : w.graphs) {
        s += " [" + g.desc + " rs_head=" + std::to_string(g.rs_head) + "]";
    }
    s += " matrix types:";
    for (const ggml_tensor * t : w.weights) {
        if (t->ne[1] > 1) {
            s += std::string(" ") + ggml_type_name(t->type);
        }
    }
    for (int il = 0; il < mp.n_layers; il++) {
        const graphgen::layer_params & lp = mp.layers[il];
        s += "; layer " + std::to_string(il) + ": kind=" + std::to_string(lp.kind);
        if (lp.kind == 0) {
            s += " d_conv=" + std::to_string(lp.d_conv) + " zero_state=" + std::to_string(lp.zero_state) +
                 " extra_rows=" + std::to_string(lp.extra_rows) + " reader=" + std::to_string(lp.reader) +
                 " out_flag=" + std::to_string(lp.out_flag) + " slot_mode=" + std::to_string(lp.slot_mode) +
                 " idx_offset=" + std::to_string(lp.idx_offset) + " keep_qkv=" + std::to_string(lp.keep_qkv);
        }
        s += " swiglu=" + std::to_string(lp.swiglu) + " view_add=" + std::to_string(lp.view_add) +
             " view_reader=" + std::to_string(lp.view_reader);
    }
    return s;
}

// Converts the bytes of a tensor into floats (F32, F16 and I32; other types give no values).
// O(number of elements).
std::vector<float> to_floats(const ggml_tensor * t, const std::vector<uint8_t> & b) {
    const int64_t      n = ggml_nelements(t);
    std::vector<float> f;
    if (!ggml_is_contiguous(t)) {
        return f;
    }
    f.resize(n);
    switch (t->type) {
        case GGML_TYPE_F32: memcpy(f.data(), b.data(), n * sizeof(float)); break;
        case GGML_TYPE_F16: ggml_fp16_to_fp32_row((const ggml_fp16_t *) b.data(), f.data(), n); break;
        case GGML_TYPE_I32:
            for (int64_t i = 0; i < n; i++) {
                int32_t x;
                memcpy(&x, b.data() + i * 4, 4);
                f[i] = (float) x;
            }
            break;
        default: f.clear(); break;
    }
    return f;
}

// Runs one input on one device and records the outputs and the caches after each step (with
// mode.nodes: each node that owns its bytes, and the caches).
run_result run_on(const uint8_t * data, size_t size, ggml_backend_dev_t dev, const run_mode & mode) {
    run_result         r;
    FuzzedDataProvider fdp(data, size);
    fakedsp::config    cfg;
    hexhost::options   o;
    graphgen::decode_session(fdp, false, cfg, o);   // the phone takes its switches from the environment

    hexhost::device d;
    d.dev = dev;
    graphgen::world w;
    w.numeric  = true;
    w.no_reuse = mode.nodes;
    if (!graphgen::build_world(fdp, &d, w)) {
        graphgen::free_world(w);
        return r;
    }
    r.built  = true;
    r.desc   = w.graphs.empty() ? "" : w.graphs[0].desc;
    r.params = describe_params(w);

    // Each cache starts at zero, as a new sequence of llama.cpp does
    for (ggml_tensor * c : w.caches) {
        std::vector<uint8_t> z(ggml_nbytes(c), 0);
        ggml_backend_tensor_set(c, z.data(), 0, z.size());
    }

    graphgen::step_hooks h;
    // The numeric mode keeps each slot index valid, thus each step compares
    h.after_step = [&](graphgen::graph_spec & g, uint32_t step, bool) {
        std::map<const ggml_tensor *, int> index;
        for (size_t i = 0; i < g.order.size(); i++) {
            index[g.order[i]] = (int) i;
        }
        for (size_t k = 0; k < w.caches.size(); k++) {
            index[w.caches[k]] = -2 - (int) k;
        }
        std::vector<ggml_tensor *> ts;
        if (mode.nodes) {
            for (ggml_tensor * t : g.order) {
                const bool no_compute = t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
                                        t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
                if (!t->view_src && !no_compute) {
                    ts.push_back(t);
                }
            }
        } else {
            ts = g.outputs;
        }
        ts.insert(ts.end(), w.caches.begin(), w.caches.end());
        const size_t gi = (size_t) (&g - w.graphs.data());
        for (ggml_tensor * t : ts) {
            std::vector<uint8_t> b(ggml_nbytes(t));
            ggml_backend_tensor_get(t, b.data(), 0, b.size());
            snapshot s;
            s.name  = std::string(t->name[0] ? t->name : "(unnamed)") + " " + ggml_op_desc(t);
            s.what  = describe(t);
            s.step  = step;
            s.graph = gi;
            const auto it = index.find(t);
            s.node  = it == index.end() ? -1 : it->second;
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                if (t->src[j]) {
                    const auto is = index.find(owner(t->src[j]));
                    s.srcs.push_back(is == index.end() ? -1 : is->second);
                }
            }
            s.v   = to_floats(t, b);
            s.row = t->ne[0];
            r.snaps.push_back(std::move(s));
        }
        return true;
    };
    graphgen::run_steps(fdp, w, h);
    graphgen::free_world(w);
    return r;
}

// The error of one tensor: the normalized mean square error and the count of values where one
// run gives a finite value and the other does not. O(n).
struct error {
    double nmse     = 0.0;
    double max_abs  = 0.0;
    size_t nonfinite = 0;
};

error compare(const std::vector<float> & a, const std::vector<float> & b) {
    error  e;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        const bool fa = std::isfinite(a[i]), fb = std::isfinite(b[i]);
        if (fa != fb) {
            e.nonfinite++;
            continue;
        }
        if (!fa) {
            continue;
        }
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) b[i] * b[i];
        e.max_abs = std::max(e.max_abs, std::fabs(d));
    }
    e.nmse = den > 0.0 ? num / den : num;
    return e;
}

// The absolute tolerance near 0: the smallest normal FP16 value (2^-14). The HVX and HMX kernels
// compute in FP16 or in qfloat, and they give 0 for a result below that value, where the CPU keeps a
// very small float. The nmse of such a tensor is large only because its reference is near 0.
constexpr double k_abs_tol = 6.103515625e-05;

// True when the NPU value differs from the CPU value: a value that is finite on one side only, or an
// nmse above the limit with a difference above the absolute tolerance.
bool differs(const error & e, double limit) {
    return e.nonfinite > 0 || (e.nmse > limit && e.max_abs > k_abs_tol);
}

// Prints the tensors that differ. The status of each snapshot: 'd' (the error is more than the
// limit), 'u' (the NPU value is all zero and the CPU value is not: a node that the NPU did not
// write, for example the first node of a fusion), or 'o'. With mode.nodes the function also
// gives the root nodes of the first step that has a difference: a node that differs, and each
// source of it is the same. O(number of snapshots times the values of each).
void print_detail(const run_result & a, const run_result & b, double limit, const run_mode & mode) {
    std::vector<char> st(a.snaps.size(), 'o');
    for (size_t i = 0; i < a.snaps.size(); i++) {
        const error e     = compare(a.snaps[i].v, b.snaps[i].v);
        const bool  zero  = std::all_of(a.snaps[i].v.begin(), a.snaps[i].v.end(), [](float x) { return x == 0.0f; });
        const bool  czero = std::all_of(b.snaps[i].v.begin(), b.snaps[i].v.end(), [](float x) { return x == 0.0f; });
        if (differs(e, limit)) {
            st[i] = (zero && !czero) ? 'u' : 'd';
        }
    }
    printf("  params: %s\n", a.params.c_str());
    size_t first = a.snaps.size();
    for (size_t i = 0; i < a.snaps.size(); i++) {
        if (st[i] == 'd' && first == a.snaps.size()) {
            first = i;
        }
    }
    if (first == a.snaps.size()) {
        printf("  no tensor differs\n");
        return;
    }
    // The snapshots of one step and one graph are in one run of the list
    size_t n_diff = 0, n_unwritten = 0;
    for (size_t i = 0; i < a.snaps.size(); i++) {
        n_diff += st[i] == 'd';
        n_unwritten += st[i] == 'u';
        if (st[i] == 'o' || (!mode.detail && !mode.nodes)) {
            continue;
        }
        if (mode.nodes && a.snaps[i].step != a.snaps[first].step) {
            continue;   // with each node, only the first step that differs
        }
        const error e = compare(a.snaps[i].v, b.snaps[i].v);
        printf("  %c step %u graph %zu #%d nmse %.3g max abs %.3g nonfinite %zu: %s\n", st[i], a.snaps[i].step,
               a.snaps[i].graph, a.snaps[i].node, e.nmse, e.max_abs, e.nonfinite, a.snaps[i].what.c_str());
        // For a cache: the rows (the slots of the state table) that differ
        if (a.snaps[i].node < -1 && a.snaps[i].row > 0) {
            const size_t r = (size_t) a.snaps[i].row;
            std::string  rows;
            for (size_t j = 0; (j + 1) * r <= a.snaps[i].v.size() && j * r < b.snaps[i].v.size(); j++) {
                const std::vector<float> ra(a.snaps[i].v.begin() + j * r, a.snaps[i].v.begin() + (j + 1) * r);
                const std::vector<float> rb(b.snaps[i].v.begin() + j * r, b.snaps[i].v.begin() + (j + 1) * r);
                const error              er = compare(ra, rb);
                if (differs(er, limit)) {
                    char buf[64];
                    snprintf(buf, sizeof(buf), " row %zu nmse %.3g", j, er.nmse);
                    rows += buf;
                }
            }
            printf("    rows that differ:%s\n", rows.empty() ? " none" : rows.c_str());
        }
    }
    printf("  %zu snapshots differ, %zu not written by the NPU, first at step %u\n", n_diff, n_unwritten,
           a.snaps[first].step);
    if (!mode.nodes) {
        return;
    }
    // The roots: the nodes of the first step that differ with no source that differs
    std::map<int, char> of_step;
    for (size_t i = 0; i < a.snaps.size(); i++) {
        if (a.snaps[i].step == a.snaps[first].step && a.snaps[i].node >= 0) {
            of_step[a.snaps[i].node] = st[i];
        }
    }
    for (size_t i = 0; i < a.snaps.size(); i++) {
        if (a.snaps[i].step != a.snaps[first].step || st[i] != 'd' || a.snaps[i].node < 0) {
            continue;
        }
        bool src_differs = false;
        for (int s : a.snaps[i].srcs) {
            const auto it = of_step.find(s);
            src_differs |= it != of_step.end() && it->second == 'd';
        }
        if (!src_differs) {
            const error e = compare(a.snaps[i].v, b.snaps[i].v);
            printf("  root #%d nmse %.3g max abs %.3g: %s\n", a.snaps[i].node, e.nmse, e.max_abs, a.snaps[i].what.c_str());
        }
    }
}

// Adds the files of a path (a file, or the files of a directory in name order) to the list.
void collect(const char * path, std::vector<std::string> & files) {
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "hexhost_phone: cannot read %s\n", path);
        return;
    }
    if (!S_ISDIR(st.st_mode)) {
        files.push_back(path);
        return;
    }
    std::vector<std::string> names;
    if (DIR * d = opendir(path)) {
        while (dirent * e = readdir(d)) {
            if (e->d_name[0] != '.') {
                names.push_back(std::string(path) + "/" + e->d_name);
            }
        }
        closedir(d);
    }
    std::sort(names.begin(), names.end());
    files.insert(files.end(), names.begin(), names.end());
}

// Reads a whole file. Gives false when the file cannot be read.
bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    out.clear();
    uint8_t buf[4096];
    size_t  n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.insert(out.end(), buf, buf + n);
    }
    fclose(f);
    return true;
}

} // namespace

// The results of one GATED_DELTA_NET graph on one device: the attention part of the output, each
// snapshot slot that the op writes (min(T, K) slots), and a reader of slot 0 in the same graph.
struct gdn_result {
    bool                            ok = false;
    std::vector<float>              attn;
    std::vector<std::vector<float>> slots;
    std::vector<float>              reader;
};

// Runs one GATED_DELTA_NET graph on a device: q and k are unit vectors, v is in [-1, 1], the gate
// is in [-2, -0.01], beta is in (0, 1), and the state is in [-0.1, 0.1], as in Qwen3.5. A SCALE of
// slot 0 reads the snapshot part in the same graph. The values come from the seed, thus the two
// devices get the same inputs. O(S_v * S_v * H * T).
gdn_result run_gdn(ggml_backend_dev_t dev, int64_t S_v, int64_t H, int64_t T, int64_t K, uint32_t seed) {
    gdn_result     r;
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) {
        return r;
    }
    ggml_init_params p   = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx = ggml_init(p);
    ggml_tensor *    q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, 1);
    ggml_tensor *    k   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, 1);
    ggml_tensor *    v   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, 1);
    ggml_tensor *    g   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, 1);
    ggml_tensor *    b   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, 1);
    ggml_tensor *    s   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H, 1);
    ggml_tensor *    out = ggml_gated_delta_net(ctx, q, k, v, g, b, s, K);
    const size_t     attn_bytes = (size_t) S_v * H * T * sizeof(float);
    ggml_tensor *    rd  = ggml_scale(ctx, ggml_view_1d(ctx, out, S_v * S_v * H, attn_bytes), 2.0f);
    ggml_set_output(out);
    ggml_set_output(rd);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, rd);

    bool supported = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        supported &= ggml_backend_dev_supports_op(dev, ggml_graph_node(gf, i));
    }
    ggml_backend_buffer_t buf = supported ? ggml_backend_alloc_ctx_tensors(ctx, be) : nullptr;
    if (buf) {
        std::mt19937                          rng(seed);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        auto fill = [&](ggml_tensor * t, float lo, float hi) {
            std::vector<float> x(ggml_nelements(t));
            for (auto & e : x) {
                e = lo + (hi - lo) * 0.5f * (u(rng) + 1.0f);
            }
            ggml_backend_tensor_set(t, x.data(), 0, ggml_nbytes(t));
        };
        auto unit = [&](ggml_tensor * t) {
            std::vector<float> x(ggml_nelements(t));
            for (size_t i = 0; i < x.size(); i += (size_t) S_v) {
                double n2 = 0.0;
                for (int64_t j = 0; j < S_v; j++) {
                    x[i + j] = u(rng);
                    n2 += (double) x[i + j] * x[i + j];
                }
                for (int64_t j = 0; j < S_v; j++) {
                    x[i + j] = (float) (x[i + j] / std::sqrt(n2 + 1e-12));
                }
            }
            ggml_backend_tensor_set(t, x.data(), 0, ggml_nbytes(t));
        };
        unit(q);
        unit(k);
        fill(v, -1.0f, 1.0f);
        fill(g, -2.0f, -0.01f);
        fill(b, 0.05f, 0.95f);
        fill(s, -0.1f, 0.1f);
        if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
            std::vector<float> all(ggml_nelements(out));
            ggml_backend_tensor_get(out, all.data(), 0, ggml_nbytes(out));
            const size_t n_attn = (size_t) S_v * H * T;
            const size_t n_slot = (size_t) S_v * S_v * H;
            r.attn.assign(all.begin(), all.begin() + n_attn);
            for (int64_t sl = 0; sl < std::min(T, K); sl++) {
                r.slots.emplace_back(all.begin() + n_attn + sl * n_slot, all.begin() + n_attn + (sl + 1) * n_slot);
            }
            r.reader.resize(ggml_nelements(rd));
            ggml_backend_tensor_get(rd, r.reader.data(), 0, ggml_nbytes(rd));
            r.ok = true;
        }
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(be);
    return r;
}

// --gdn-sweep: single GATED_DELTA_NET graphs of each shape on the device and on the CPU. Prints
// one line for each shape: the error of the attention part, of each written snapshot slot, and of
// the reader of slot 0. Gives the count of the shapes with an error above the limit.
int gdn_sweep(ggml_backend_dev_t dev, ggml_backend_dev_t cpu, double limit) {
    auto fmt = [](double x) {
        char b[32];
        snprintf(b, sizeof(b), "%.3g", x);
        return std::string(b);
    };
    int n_bad = 0;
    for (int64_t S_v : { 8, 16, 32, 64, 128 }) {
        for (int64_t H : { 1, 2, 4 }) {
            for (int64_t T : { 1, 2, 5, 33 }) {
                for (int64_t K : { 1, 2, 3, 5 }) {
                    const uint32_t   seed = (uint32_t) (S_v * 1000003 + H * 10007 + T * 101 + K);
                    const gdn_result a    = run_gdn(dev, S_v, H, T, K, seed);
                    const gdn_result c    = run_gdn(cpu, S_v, H, T, K, seed);
                    if (!a.ok || !c.ok) {
                        printf("gdn S_v=%lld H=%lld T=%lld K=%lld: skip (%s)\n", (long long) S_v, (long long) H,
                               (long long) T, (long long) K, a.ok ? "the CPU run failed" : "not supported or failed");
                        continue;
                    }
                    std::string line = "attn " + fmt(compare(a.attn, c.attn).nmse);
                    bool        bad  = differs(compare(a.attn, c.attn), limit);
                    for (size_t sl = 0; sl < a.slots.size(); sl++) {
                        const error e = compare(a.slots[sl], c.slots[sl]);
                        line += " slot" + std::to_string(sl) + " " + fmt(e.nmse);
                        bad |= differs(e, limit);
                    }
                    const error er = compare(a.reader, c.reader);
                    line += " reader " + fmt(er.nmse);
                    bad |= differs(er, limit);
                    n_bad += bad;
                    printf("gdn S_v=%lld H=%lld T=%lld K=%lld: %s %s\n", (long long) S_v, (long long) H, (long long) T,
                           (long long) K, bad ? "mismatch" : "ok", line.c_str());
                    fflush(stdout);
                }
            }
        }
    }
    printf("gdn summary: %d shapes above the nmse limit %g (absolute tolerance %g)\n", n_bad, limit, k_abs_tol);
    return n_bad ? 1 : 0;
}

// Starts one thread and waits for it. The first step of each phone ASan run: the ASan runtime of
// the NDK stops each new thread with SIGILL on the SM8750, thus a failure here is a failure of the
// environment, not a defect of the backend. Gives the exit code.
int selftest_threads() {
    std::atomic<bool> ran { false };
    std::thread       t([&] { ran = true; });
    t.join();
    printf("hexhost_phone: thread self-test %s\n", ran ? "ok" : "failed");
    return ran ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc == 2 && !strcmp(argv[1], "--selftest-threads")) {
        return selftest_threads();
    }
    double                   seconds = 80.0;
    double                   limit   = 1e-2;
    const char *             name    = "HTP0";
    size_t                   n_random = 0;
    uint32_t                 seed     = 20260923;
    const char *             save_dir = nullptr;
    bool                     keep     = false;
    run_mode                 mode;
    bool                     gdn      = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seconds") && i + 1 < argc) {
            seconds = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--nmse") && i + 1 < argc) {
            limit = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            name = argv[++i];   // CPU: a check of the driver on a host with no NPU
        } else if (!strcmp(argv[i], "--random") && i + 1 < argc) {
            n_random = (size_t) atol(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = (uint32_t) strtoul(argv[++i], nullptr, 0);
        } else if (!strcmp(argv[i], "--save") && i + 1 < argc) {
            save_dir = argv[++i];   // each random input that gives no "ok" goes into this directory
        } else if (!strcmp(argv[i], "--keep")) {
            keep = true;            // the save directory keeps each random input, also an "ok" one
        } else if (!strcmp(argv[i], "--detail")) {
            mode.detail = true;
        } else if (!strcmp(argv[i], "--nodes")) {
            mode.nodes = true;
        } else if (!strcmp(argv[i], "--gdn-sweep")) {
            gdn = true;
        } else {
            collect(argv[i], files);
        }
    }
    // The random inputs: 64 to 1024 bytes each from the seed, thus a run can be done again
    std::mt19937 gen(seed);
    for (size_t i = 0; i < n_random; i++) {
        files.push_back("random:" + std::to_string(seed) + ":" + std::to_string(i));
    }
    if (files.empty() && !gdn) {
        fprintf(stderr,
                "Usage: %s [--seconds S] [--nmse X] [--device NAME] [--random N] [--seed S] [--save DIR [--keep]]\n"
                "       [--detail] [--nodes] FILE_OR_DIRECTORY...\n"
                "       %s [--nmse X] [--device NAME] --gdn-sweep\n"
                "       %s --selftest-threads\n",
                argv[0], argv[0], argv[0]);
        return 2;
    }

    ggml_log_set(log_filter, nullptr);
    ggml_set_abort_callback(on_abort);
    ggml_backend_dev_t htp = ggml_backend_dev_by_name(name);
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!htp || !cpu) {
        fprintf(stderr, "hexhost_phone: no %s device. For HTP0, set ADSP_LIBRARY_PATH to the directory of libggml-htp-v79.so.\n",
                htp ? "CPU" : name);
        return 2;
    }
    if (gdn) {
        return gdn_sweep(htp, cpu, limit);
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed  = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    size_t n_ok = 0, n_skip = 0, n_bad = 0, n_done = 0;
    std::vector<uint8_t> data;
    for (const auto & path : files) {
        if (elapsed() > seconds) {
            break;
        }
        n_done++;
        std::string  last_saved;
        const char * base = strrchr(path.c_str(), '/');
        base              = base ? base + 1 : path.c_str();
        if (path.rfind("random:", 0) == 0) {
            data.resize(64 + gen() % 961);
            for (auto & x : data) {
                x = (uint8_t) gen();
            }
            // The input goes to the save directory first: a crash of the run leaves it there
            if (save_dir) {
                const std::string out = std::string(save_dir) + "/random-" + std::to_string(seed) + "-" +
                                        path.substr(path.rfind(':') + 1) + ".bin";
                if (FILE * f = fopen(out.c_str(), "wb")) {
                    fwrite(data.data(), 1, data.size(), f);
                    fclose(f);
                }
                last_saved = out;
            }
        } else if (!read_file(path, data)) {
            printf("error %s: cannot read the file\n", base);
            n_bad++;
            continue;
        }
        const run_result a = run_on(data.data(), data.size(), htp, mode);
        if (!a.built) {
            printf("skip %s: %s does not run every node of the graph\n", base, name);
            n_skip++;
            if (!last_saved.empty() && !keep) {
                remove(last_saved.c_str());
            }
            continue;
        }
        const run_result b = run_on(data.data(), data.size(), cpu, mode);
        if (!b.built || a.snaps.size() != b.snaps.size()) {
            printf("error %s: the CPU run gives %zu snapshots, the HTP0 run %zu\n", base, b.snaps.size(), a.snaps.size());
            n_bad++;
            continue;
        }
        // The worst snapshot: the worst one that differs when one differs, else the worst of all
        error       worst;
        std::string where = "-";
        bool        bad   = false;
        size_t      n_val = 0;
        double      mag   = 0.0;
        for (size_t i = 0; i < a.snaps.size(); i++) {
            const error e = compare(a.snaps[i].v, b.snaps[i].v);
            n_val += b.snaps[i].v.size();
            for (float x : b.snaps[i].v) {
                mag += std::isfinite(x) ? std::fabs(x) : 0.0;
            }
            // With --nodes the NPU does not write the first node of a fusion: an all-zero node is not a difference
            const bool unwritten = mode.nodes && a.snaps[i].node >= 0 &&
                                   std::all_of(a.snaps[i].v.begin(), a.snaps[i].v.end(), [](float x) { return x == 0.0f; });
            if (unwritten) {
                continue;
            }
            const bool d = differs(e, limit);
            if (bad && !d) {
                continue;
            }
            if (where == "-" || (d && !bad) || e.nonfinite > worst.nonfinite ||
                (e.nonfinite == worst.nonfinite && e.nmse > worst.nmse)) {
                worst = e;
                where = a.snaps[i].name + " step " + std::to_string(a.snaps[i].step);
            }
            bad |= d;
        }
        printf("%s %s: %s, %zu snapshots, %zu values, mean abs %.3g, worst nmse %.3g max abs %.3g nonfinite %zu at %s\n",
               bad ? "mismatch" : "ok", base, a.desc.c_str(), a.snaps.size(), n_val, n_val ? mag / n_val : 0.0,
               worst.nmse, worst.max_abs, worst.nonfinite, where.c_str());
        if (bad && (mode.detail || mode.nodes)) {
            print_detail(a, b, limit, mode);
        }
        fflush(stdout);
        bad ? n_bad++ : n_ok++;
        if (!bad && !last_saved.empty() && !keep) {
            remove(last_saved.c_str());
        }
    }
    printf("summary: %zu of %zu inputs in %.1f s: %zu ok, %zu skip, %zu mismatch or error (nmse limit %g, absolute "
           "tolerance %g)\n", n_done, files.size(), elapsed(), n_ok, n_skip, n_bad, limit, k_abs_tol);
    return n_bad ? 1 : 0;
}
