// ops_replay: run the cases of a pack on one or more backends and write the raw results. The
// phone runs this program (CPU and HTP0), and ops_oracle compare reads its results on the host.
//
//   ops_replay --pack PACK --out RESULTS --progress FILE --backends CPU,HTP0
//              [--tag-suffix S] [--deadline EPOCH] [--max-bytes B] [--count N]
//              [--repeat K] [--only KIND,KIND...] [--first N]
//
// --count N runs only the first N cases of the pack (a short probe run). --repeat K runs each case K
// times on each backend, each time with new buffers, and flags the run as nondeterministic
// (FLAG_NONDET) when a repeat gives different output bytes. --only runs only the cases of the kinds.
// --first N starts at case N of the pack (with --count, the cases N to N + COUNT - 1).
//
//   ops_replay [--trace] --selftest-threads
//
// --trace (first argument) runs the driver in a child process under ptrace and prints the state
// of each fatal signal of each thread (crashdiag.h). --selftest-threads creates and joins two
// threads and prints each step: a minimal reproducer of a fault in the thread start.
//
// --deadline is an absolute time in seconds since 1970. The driver starts no run after it. A
// relative limit is not sufficient: after a crash the shell loop starts a new process, and the
// outer `timeout -s KILL` of the phone command must not stop a run in the middle, or the next
// process records a crash that did not occur.
//
// The driver can continue after a crash. Before each run it appends "S <case> <tag>" to the
// progress file and syncs it, and after the run it appends "D <case> <tag>". At the start it reads
// the progress file: a run with an S line and no D line stopped the process (an abort of the DSP
// session, a signal, a sanitizer report), thus the driver writes a "crashed" record for it and goes
// on with the next run. The open of a backend has the lines "O" and "P" in the same way: an open
// that stops the process (for example a fault in the setup of the DSP session) gives a "crashed"
// record and an "X" line, and the next process skips that backend. A shell loop restarts the
// driver until it stops with the code 0 (all runs done) or 3 (the deadline).
//
// Exit codes: 0 all runs done, 3 deadline, 1 an error of the arguments or of the files.

#include "case.h"
#include "crashdiag.h"
#include "exec.h"
#include "wire.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <pmmintrin.h>
#include <xmmintrin.h>
#endif

namespace {

using namespace fo;

void usage() {
    std::fprintf(stderr,
                 "usage: ops_replay --pack PACK --out RESULTS --progress FILE --backends CPU,HTP0\n"
                 "                  [--tag-suffix S] [--deadline EPOCH] [--max-bytes B] [--count N]\n"
                 "                  [--repeat K] [--only KIND,KIND...] [--first N]\n");
}

// Split a comma list. O(length).
std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> r;
    size_t                   p = 0;
    while (p <= s.size()) {
        const size_t q    = s.find(',', p);
        const size_t e    = q == std::string::npos ? s.size() : q;
        if (e > p) {
            r.push_back(s.substr(p, e - p));
        }
        p = e + 1;
    }
    return r;
}

// Describe the first output difference between two runs of one case: the output, the count of
// different elements, and the first one with its coordinates (and both values for an f32 output).
// Return an empty text when the outputs are identical. O(output bytes).
std::string describe_diff(const built_case & c, const run_result & a, const run_result & b) {
    if (a.outs.size() != b.outs.size()) {
        return "a different number of outputs";
    }
    for (size_t o = 0; o < a.outs.size(); o++) {
        if (a.outs[o] == b.outs[o]) {
            continue;
        }
        const ggml_tensor * t  = c.outs[o].t;
        const size_t        es = t->type == GGML_TYPE_F32 ? 4 : t->type == GGML_TYPE_F16 ? 2 : 1;
        const size_t        n  = std::min(a.outs[o].size(), b.outs[o].size()) / es;
        size_t              count = 0, first = SIZE_MAX;
        for (size_t i = 0; i < n; i++) {
            if (std::memcmp(a.outs[o].data() + i * es, b.outs[o].data() + i * es, es) != 0) {
                count++;
                first = std::min(first, i);
            }
        }
        char buf[256];
        if (first == SIZE_MAX) {
            std::snprintf(buf, sizeof(buf), "output %s: the sizes differ", c.outs[o].label.c_str());
            return buf;
        }
        const int64_t ne0 = t->ne[0], ne1 = t->ne[1];
        const long long i0 = (long long) (es == 1 ? first : first % (size_t) ne0);
        const long long i1 = (long long) (es == 1 ? 0 : first / (size_t) ne0 % (size_t) ne1);
        const long long i2 = (long long) (es == 1 ? 0 : first / (size_t) (ne0 * ne1));
        if (es == 4) {
            float va, vb;
            std::memcpy(&va, a.outs[o].data() + first * 4, 4);
            std::memcpy(&vb, b.outs[o].data() + first * 4, 4);
            std::snprintf(buf, sizeof(buf), "output %s: %zu of %zu elements differ, the first at [%lld,%lld,%lld]: %.9g and %.9g",
                          c.outs[o].label.c_str(), count, n, i0, i1, i2, va, vb);
        } else {
            std::snprintf(buf, sizeof(buf), "output %s: %zu of %zu elements differ, the first at [%lld,%lld,%lld]",
                          c.outs[o].label.c_str(), count, n, i0, i1, i2);
        }
        return buf;
    }
    return "";
}

// Append one line to the progress file and sync it.
void progress_line(FILE * f, char kind, size_t idx, const std::string & tag) {
    std::fprintf(f, "%c %zu %s\n", kind, idx, tag.c_str());
    std::fflush(f);
    ::fsync(fileno(f));
}

} // namespace

// The replay driver itself. main() can run it under the tracer.
int replay_main(int argc, char ** argv) {
    // Unbuffered: a crash must not lose the lines of the runs before it.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    install_crash_diag();
    if (argc >= 2 && std::strcmp(argv[1], "--selftest-threads") == 0) {
        return selftest_threads();
    }
    std::string pack_path, out_path, progress_path, backends_arg, suffix;
    double      deadline   = 1e30;
    uint64_t    max_bytes  = uint64_t(64) << 20;
    size_t      count      = SIZE_MAX;
    int         repeat     = 1;
    size_t      first      = 0;
    std::string only_arg;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (i + 1 >= argc) {
            usage();
            return 1;
        }
        const std::string v = argv[++i];
        if (a == "--pack") pack_path = v;
        else if (a == "--out") out_path = v;
        else if (a == "--progress") progress_path = v;
        else if (a == "--backends") backends_arg = v;
        else if (a == "--tag-suffix") suffix = v;
        else if (a == "--deadline") deadline = std::atof(v.c_str());
        else if (a == "--max-bytes") max_bytes = std::strtoull(v.c_str(), nullptr, 10);
        else if (a == "--count") count = (size_t) std::strtoull(v.c_str(), nullptr, 10);
        else if (a == "--repeat") repeat = std::max(1, std::atoi(v.c_str()));
        else if (a == "--only") only_arg = v;
        else if (a == "--first") first = (size_t) std::strtoull(v.c_str(), nullptr, 10);
        else {
            usage();
            return 1;
        }
    }
    if (pack_path.empty() || out_path.empty() || progress_path.empty() || backends_arg.empty()) {
        usage();
        return 1;
    }

    std::vector<pack_case> cases;
    std::string            why;
    if (!read_pack(pack_path, cases, why)) {
        std::fprintf(stderr, "ops_replay: %s\n", why.c_str());
        return 1;
    }
    const std::vector<std::string> devs = split(backends_arg);
    const std::vector<std::string> only_list = split(only_arg);
    const std::set<std::string>    only(only_list.begin(), only_list.end());

    // The runs that are done, the run that stopped the last process, and the backends that cannot
    // open. The progress lines: S (run started), D (run done), O (backend open started), P (backend
    // open done), X (the backend cannot open: its runs are skipped).
    std::set<std::pair<size_t, std::string>> done;
    std::set<std::string>                    dead;
    std::pair<size_t, std::string>           started = { SIZE_MAX, "" };
    std::pair<size_t, std::string>           opening = { SIZE_MAX, "" };
    if (FILE * pf = std::fopen(progress_path.c_str(), "r")) {
        char   kind;
        size_t idx;
        char   tag[256];
        while (std::fscanf(pf, " %c %zu %255s", &kind, &idx, tag) == 3) {
            if (kind == 'S') {
                started = { idx, tag };
            } else if (kind == 'D') {
                done.insert({ idx, tag });
                if (started.first == idx && started.second == tag) {
                    started = { SIZE_MAX, "" };
                }
            } else if (kind == 'O') {
                opening = { idx, tag };
            } else if (kind == 'P') {
                opening = { SIZE_MAX, "" };
            } else if (kind == 'X') {
                dead.insert(tag);
                opening = { SIZE_MAX, "" };
            }
        }
        std::fclose(pf);
    }
    FILE * out = std::fopen(out_path.c_str(), "ab");
    FILE * pf  = std::fopen(progress_path.c_str(), "a");
    if (!out || !pf) {
        std::fprintf(stderr, "ops_replay: cannot open %s or %s for append\n", out_path.c_str(), progress_path.c_str());
        return 1;
    }
    if (started.first != SIZE_MAX && !done.count(started)) {
        result_rec rec;
        rec.idx       = (uint32_t) started.first;
        rec.tag       = started.second;
        rec.rr.status = RUN_CRASHED;
        rec.rr.detail = "the process stopped during this run";
        append_result(out, rec);
        progress_line(pf, 'D', started.first, started.second);
        done.insert(started);
        std::printf("ops_replay: case %zu on %s stopped the last process\n", started.first, started.second.c_str());
    }
    if (opening.first != SIZE_MAX) {
        // the open of a backend (for HTP0: the DSP session) stopped the last process
        result_rec rec;
        rec.idx       = (uint32_t) opening.first;
        rec.tag       = opening.second;
        rec.rr.status = RUN_CRASHED;
        rec.rr.detail = "the open of the backend stopped the process; its runs are skipped";
        append_result(out, rec);
        progress_line(pf, 'X', opening.first, opening.second);
        dead.insert(opening.second);
        std::printf("ops_replay: the open of %s stopped the last process, its runs are skipped\n", opening.second.c_str());
    }

    std::vector<backend_ctx> bes(devs.size());
    std::vector<bool>        opened(devs.size(), false);
    size_t                   n_run = 0;
    const size_t last = count == SIZE_MAX ? cases.size() : std::min(cases.size(), first + count);
    for (size_t idx = first; idx < last; idx++) {
        for (size_t d = 0; d < devs.size(); d++) {
            const std::string tag = devs[d] + suffix;
            if (done.count({ idx, tag }) || dead.count(tag)) {
                continue;
            }
            const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            if (now > deadline) {
                std::printf("ops_replay: deadline after %zu runs, case %zu\n", n_run, idx);
                std::fclose(out);
                std::fclose(pf);
                return 3;
            }
            if (!opened[d]) {
                progress_line(pf, 'O', idx, tag);
                if (!open_backend(devs[d], bes[d], why)) {
                    std::fprintf(stderr, "ops_replay: %s\n", why.c_str());
                    result_rec rec;
                    rec.idx       = (uint32_t) idx;
                    rec.tag       = tag;
                    rec.rr.status = RUN_UNSUPPORTED;
                    rec.rr.detail = "no backend: " + why;
                    append_result(out, rec);
                    progress_line(pf, 'X', idx, tag);
                    dead.insert(tag);
                    continue;
                }
                progress_line(pf, 'P', idx, tag);
                opened[d] = true;
            }
            progress_line(pf, 'S', idx, tag);
            result_rec rec;
            rec.idx = (uint32_t) idx;
            rec.tag = tag;
            built_case c;
            if (!build_case(cases[idx].bytes.data(), cases[idx].bytes.size(), cases[idx].forced, max_bytes, c)) {
                rec.rr.status = RUN_INVALID;
            } else if (!only.empty() && !only.count(c.kind->name)) {
                rec.rr.status = RUN_UNSUPPORTED;
                rec.rr.detail = "not in --only";
            } else {
                std::printf("run %zu %s %s: %s\n", idx, tag.c_str(), c.kind->name, c.desc.c_str());
                std::fflush(stdout);
                rec.rr = run_case(c, bes[d]);
                // the repeats: the same case with new buffers; the record keeps the first outputs
                int         n_diff = 0;
                std::string first_diff;
                for (int r = 1; r < repeat && rec.rr.status == RUN_OK; r++) {
                    // a new decode of the same bytes: the tensors of a case get their buffers once
                    built_case c2;
                    if (!build_case(cases[idx].bytes.data(), cases[idx].bytes.size(), cases[idx].forced, max_bytes, c2)) {
                        break;
                    }
                    const run_result again = run_case(c2, bes[d]);
                    if (again.flags & ~rec.rr.flags) {
                        rec.rr.flags |= again.flags;
                        rec.rr.detail += "repeat " + std::to_string(r) + ": " + again.detail;
                    }
                    const std::string diff = again.status == RUN_OK ? describe_diff(c, rec.rr, again)
                                                                    : std::string("repeat status ") + run_status_name(again.status);
                    if (!diff.empty()) {
                        n_diff++;
                        if (first_diff.empty()) {
                            first_diff = "repeat " + std::to_string(r) + ": " + diff;
                        }
                    }
                }
                if (n_diff > 0) {
                    rec.rr.flags |= FLAG_NONDET;
                    rec.rr.detail += "nondeterministic: " + std::to_string(n_diff) + " of " + std::to_string(repeat - 1) +
                                     " repeats differ from the first run; " + first_diff + "; ";
                }
                std::printf("  -> %s%s%s%s %.3f ms %s\n", run_status_name(rec.rr.status),
                            (rec.rr.flags & FLAG_GUARD) ? " GUARD" : "", (rec.rr.flags & FLAG_INPUT) ? " INPUT" : "",
                            (rec.rr.flags & FLAG_NONDET) ? " NONDET" : "", rec.rr.ms, rec.rr.detail.c_str());
                std::fflush(stdout);
            }
            append_result(out, rec);
            progress_line(pf, 'D', idx, tag);
            done.insert({ idx, tag });
            n_run++;
        }
    }
    std::printf("ops_replay: all runs done (%zu in this process)\n", n_run);
    std::fclose(out);
    std::fclose(pf);
    for (size_t d = 0; d < devs.size(); d++) {
        if (opened[d]) {
            close_backend(bes[d]);
        }
    }
    return 0;
}

// "--trace" as the first argument runs the driver under the ptrace tracer of crashdiag.cpp.
int main(int argc, char ** argv) {
#if defined(__x86_64__) || defined(__i386__)
    // A link with -ffp-model=fast sets FTZ and DAZ at the start (crtfastmath.o), thus the decode
    // would turn subnormal values into 0 where the oracle keeps them. Clear the two bits, as
    // fuzz_main.cpp does. On arm64 Android the process keeps subnormals.
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
#endif
    if (argc >= 2 && std::strcmp(argv[1], "--trace") == 0) {
        argv[1] = argv[0];
        return run_traced(argc - 1, argv + 1, replay_main);
    }
    return replay_main(argc, argv);
}
