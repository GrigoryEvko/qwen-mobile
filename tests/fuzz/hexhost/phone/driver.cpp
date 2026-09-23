// hexhost_phone: the phone driver of the hexhost fuzz campaign.
//
// The driver reads graph inputs of the x86 target fuzz_graph (the same bytes give the same
// Qwen3.5-shaped graphs, refer to graphgen.h) and runs each input two times: on the real NPU
// (HTP0, with the fusions, the op batches, the graph cache and the batch cache of the host) and
// on the CPU backend. It compares the outputs and the caches of the two runs after each step.
// The GGML_HEXAGON_* environment variables of the command select the host switches, thus one
// command runs one configuration.
//
//   hexhost_phone [--seconds S] [--nmse X] [--device NAME] [--random N] [--seed S] FILE_OR_DIRECTORY...
//
// --random N adds N inputs of 64 to 1024 random bytes from the seed (a fixed default), thus a
// run can be done again. --device CPU runs the two sides on the CPU (a check of the driver).
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
    uint32_t           step = 0;
    std::vector<float> v;
};

// The result of one run of an input on one device
struct run_result {
    bool                  built = false;   // the device runs every node of the graphs
    std::string           desc;
    std::vector<snapshot> snaps;
};

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

// Runs one input on one device and records the outputs and the caches after each step.
run_result run_on(const uint8_t * data, size_t size, ggml_backend_dev_t dev) {
    run_result         r;
    FuzzedDataProvider fdp(data, size);
    fakedsp::config    cfg;
    hexhost::options   o;
    graphgen::decode_session(fdp, false, cfg, o);   // the phone takes its switches from the environment

    hexhost::device d;
    d.dev = dev;
    graphgen::world w;
    w.numeric = true;
    if (!graphgen::build_world(fdp, &d, w)) {
        graphgen::free_world(w);
        return r;
    }
    r.built = true;
    r.desc  = w.graphs.empty() ? "" : w.graphs[0].desc;

    // Each cache starts at zero, as a new sequence of llama.cpp does
    for (ggml_tensor * c : w.caches) {
        std::vector<uint8_t> z(ggml_nbytes(c), 0);
        ggml_backend_tensor_set(c, z.data(), 0, z.size());
    }

    graphgen::step_hooks h;
    // The numeric mode keeps each slot index valid, thus each step compares
    h.after_step = [&](graphgen::graph_spec & g, uint32_t step, bool) {
        std::vector<ggml_tensor *> ts = g.outputs;
        ts.insert(ts.end(), w.caches.begin(), w.caches.end());
        for (ggml_tensor * t : ts) {
            std::vector<uint8_t> b(ggml_nbytes(t));
            ggml_backend_tensor_get(t, b.data(), 0, b.size());
            snapshot s;
            s.name = std::string(t->name[0] ? t->name : "(unnamed)") + " " + ggml_op_desc(t);
            s.step = step;
            s.v    = to_floats(t, b);
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

int main(int argc, char ** argv) {
    double                   seconds = 80.0;
    double                   limit   = 1e-2;
    const char *             name    = "HTP0";
    size_t                   n_random = 0;
    uint32_t                 seed     = 20260923;
    const char *             save_dir = nullptr;
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
        } else {
            collect(argv[i], files);
        }
    }
    // The random inputs: 64 to 1024 bytes each from the seed, thus a run can be done again
    std::mt19937 gen(seed);
    for (size_t i = 0; i < n_random; i++) {
        files.push_back("random:" + std::to_string(seed) + ":" + std::to_string(i));
    }
    if (files.empty()) {
        fprintf(stderr, "Usage: %s [--seconds S] [--nmse X] [--device NAME] [--random N] [--seed S] FILE_OR_DIRECTORY...\n",
                argv[0]);
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
        const run_result a = run_on(data.data(), data.size(), htp);
        if (!a.built) {
            printf("skip %s: %s does not run every node of the graph\n", base, name);
            n_skip++;
            if (!last_saved.empty()) {
                remove(last_saved.c_str());
            }
            continue;
        }
        const run_result b = run_on(data.data(), data.size(), cpu);
        if (!b.built || a.snaps.size() != b.snaps.size()) {
            printf("error %s: the CPU run gives %zu snapshots, the HTP0 run %zu\n", base, b.snaps.size(), a.snaps.size());
            n_bad++;
            continue;
        }
        error       worst;
        std::string where = "-";
        size_t      n_val = 0;
        double      mag   = 0.0;
        for (size_t i = 0; i < a.snaps.size(); i++) {
            const error e = compare(a.snaps[i].v, b.snaps[i].v);
            n_val += b.snaps[i].v.size();
            for (float x : b.snaps[i].v) {
                mag += std::isfinite(x) ? std::fabs(x) : 0.0;
            }
            if (i == 0 || e.nonfinite > worst.nonfinite || (e.nonfinite == worst.nonfinite && e.nmse > worst.nmse)) {
                worst = e;
                where = a.snaps[i].name + " step " + std::to_string(a.snaps[i].step);
            }
        }
        const bool bad = worst.nonfinite > 0 || worst.nmse > limit;
        printf("%s %s: %s, %zu snapshots, %zu values, mean abs %.3g, worst nmse %.3g max abs %.3g nonfinite %zu at %s\n",
               bad ? "mismatch" : "ok", base, a.desc.c_str(), a.snaps.size(), n_val, n_val ? mag / n_val : 0.0,
               worst.nmse, worst.max_abs, worst.nonfinite, where.c_str());
        fflush(stdout);
        bad ? n_bad++ : n_ok++;
        if (!bad && !last_saved.empty()) {
            remove(last_saved.c_str());
        }
    }
    printf("summary: %zu of %zu inputs in %.1f s: %zu ok, %zu skip, %zu mismatch or error (nmse limit %g)\n", n_done,
           files.size(), elapsed(), n_ok, n_skip, n_bad, limit);
    return n_bad ? 1 : 0;
}
