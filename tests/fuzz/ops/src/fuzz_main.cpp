// The libFuzzer harness of the op fuzzer: the CPU backend of this build against the oracle process.
//
// Each input is a case (refer to case.h). The harness runs the case on the CPU backend of this
// build (native SIMD, with the sanitizers of the build), sends the same bytes to the oracle process
// (ops_oracle serve: the scalar CPU backend with strict IEEE arithmetic), and compares the outputs
// with the bounds of the kind. A write outside a tensor or into an input stops the process, thus
// libFuzzer keeps the input. A numeric result above a bound is a finding: the harness keeps the
// worst case of each kind, path and verdict in the findings directory and continues.
//
// Environment:
//   FUZZ_OPS_ORACLE    the path of ops_oracle (necessary)
//   FUZZ_OPS_GROUP     a group (matmul, gdn, attn, norm, elem, data), a kind name, or "all"
//   FUZZ_OPS_FINDINGS  the findings directory (default ./findings)
//   FUZZ_OPS_STATS     the file of the statistics (default <findings>/stats.tsv)
//   FUZZ_OPS_MAX_BYTES the largest case in bytes of tensor data (default 67108864)
//   FUZZ_OPS_ABORT     stop the process at a case with this verdict or a worse one (above-strict,
//                      above-loose, nonfinite): the test mode, and the minimization of a numeric
//                      finding. The form kind=<name>,verdict=<v>,special=0 also restricts the kind,
//                      and special=0 ignores the inputs with special values (Inf, NaN, subnormal).

#include "case.h"
#include "compare.h"
#include "exec.h"
#include "wire.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>

namespace {

using namespace fo;

struct agg {
    uint64_t    count       = 0;
    double      worst_ratio = -1.0;
    double      worst_ulp   = 0.0;
    std::string file;
    std::string desc;
};

struct kind_stat {
    uint64_t execs = 0, invalid = 0, unsupported = 0, oracle_fail = 0, special = 0;
    uint64_t verdicts[5] = { 0, 0, 0, 0, 0 };
};

struct state {
    backend_ctx                     cpu;
    oracle_proc                     oracle;
    std::vector<int>                group;
    std::string                     findings = "findings";
    std::string                     stats_path;
    uint64_t                        max_bytes = uint64_t(64) << 20;
    int                             abort_v   = 99;
    int                             abort_kind = -1;
    bool                            abort_nospecial = false; // stop only for inputs without special values
    std::map<std::tuple<int, std::string, int, bool>, agg> aggs;
    std::vector<kind_stat>          kstats;
    uint64_t                        n_exec = 0;
};

state * S = nullptr;

// Return a file name from a path class: letters, digits and a few signs.
std::string clean(const std::string & s) {
    std::string r;
    for (char ch : s) {
        const bool okc = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_';
        r += okc ? ch : (ch == '/' ? '.' : '_');
    }
    return r;
}

void write_stats() {
    FILE * f = std::fopen(S->stats_path.c_str(), "w");
    if (!f) {
        return;
    }
    std::fprintf(f, "# kind\texecs\tinvalid\tunsupported\toracle_fail\tspecial\tpass\tsubnormal\tabove_strict\tabove_loose\tnonfinite\n");
    const auto & K = kinds();
    for (size_t k = 0; k < K.size(); k++) {
        const kind_stat & s = S->kstats[k];
        if (s.execs == 0 && s.invalid == 0) {
            continue;
        }
        std::fprintf(f, "K\t%s\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\n", K[k].name,
                     (unsigned long long) s.execs, (unsigned long long) s.invalid, (unsigned long long) s.unsupported,
                     (unsigned long long) s.oracle_fail, (unsigned long long) s.special,
                     (unsigned long long) s.verdicts[0], (unsigned long long) s.verdicts[1],
                     (unsigned long long) s.verdicts[2], (unsigned long long) s.verdicts[3],
                     (unsigned long long) s.verdicts[4]);
    }
    std::fprintf(f, "# kind\tpath\tverdict\tspecial\tcount\tworst_ratio\tworst_ulp\tfile\tdesc\n");
    for (const auto & it : S->aggs) {
        std::fprintf(f, "A\t%s\t%s\t%s\t%d\t%llu\t%.4g\t%.4g\t%s\t%s\n", K[std::get<0>(it.first)].name,
                     std::get<1>(it.first).c_str(), verdict_name(std::get<2>(it.first)), (int) std::get<3>(it.first),
                     (unsigned long long) it.second.count, it.second.worst_ratio, it.second.worst_ulp,
                     it.second.file.c_str(), it.second.desc.c_str());
    }
    std::fclose(f);
}

void at_exit() {
    if (S) {
        write_stats();
        S->oracle.stop();
    }
}

// Return the ratio in the first line "ratio: X" of a finding text, or -1 when the file is missing.
// Another worker process can have written a worse case with the same name.
double stored_ratio(const std::string & txt_path) {
    FILE * f = std::fopen(txt_path.c_str(), "r");
    if (!f) {
        return -1.0;
    }
    double r = -1.0;
    if (std::fscanf(f, "ratio: %lf", &r) != 1) {
        r = -1.0;
    }
    std::fclose(f);
    return r;
}

int parse_verdict(const std::string & v) {
    for (int i = 0; i <= V_NONFINITE; i++) {
        if (v == verdict_name(i)) {
            return i;
        }
    }
    return 99;
}

} // namespace

extern "C" int LLVMFuzzerInitialize(int * argc, char *** argv) {
    (void) argc;
    (void) argv;
    S = new state();
    const char * oracle = std::getenv("FUZZ_OPS_ORACLE");
    if (!oracle) {
        std::fprintf(stderr, "fuzz_ops: set FUZZ_OPS_ORACLE to the path of ops_oracle (build it with run.sh)\n");
        std::exit(2);
    }
    S->oracle.path = oracle;
    const char * g = std::getenv("FUZZ_OPS_GROUP");
    S->group       = group_kinds(g ? g : "all");
    if (S->group.empty()) {
        std::fprintf(stderr, "fuzz_ops: FUZZ_OPS_GROUP=%s names no group and no kind\n", g);
        std::exit(2);
    }
    if (const char * f = std::getenv("FUZZ_OPS_FINDINGS")) {
        S->findings = f;
    }
    ::mkdir(S->findings.c_str(), 0755);
    // In the fork mode of libFuzzer each worker process writes its own statistics file, and
    // ops_oracle stats merges them.
    const char * sp = std::getenv("FUZZ_OPS_STATS");
    S->stats_path   = sp ? sp : S->findings + "/stats-" + std::to_string((long) ::getpid()) + ".tsv";
    if (const char * m = std::getenv("FUZZ_OPS_MAX_BYTES")) {
        S->max_bytes = std::strtoull(m, nullptr, 10);
    }
    if (const char * a = std::getenv("FUZZ_OPS_ABORT")) {
        // a comma list of key=value items, or one verdict name
        const std::string s = a;
        size_t            p = 0;
        while (p <= s.size()) {
            const size_t      q    = s.find(',', p);
            const std::string item = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
            if (item.rfind("kind=", 0) == 0) {
                S->abort_kind = kind_index(item.substr(5));
            } else if (item.rfind("verdict=", 0) == 0) {
                S->abort_v = parse_verdict(item.substr(8));
            } else if (item == "special=0") {
                S->abort_nospecial = true;
            } else if (!item.empty()) {
                S->abort_v = parse_verdict(item);
            }
            if (q == std::string::npos) {
                break;
            }
            p = q + 1;
        }
    }
    S->kstats.resize(kinds().size());
    std::string why;
    if (!open_backend("CPU", S->cpu, why)) {
        std::fprintf(stderr, "fuzz_ops: %s\n", why.c_str());
        std::exit(2);
    }
    if (!S->oracle.start()) {
        std::fprintf(stderr, "fuzz_ops: cannot start the oracle %s\n", oracle);
        std::exit(2);
    }
    std::atexit(at_exit);
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    if (size == 0) {
        return -1;
    }
    const int forced = S->group[data[0] % S->group.size()];
    built_case c;
    if (!build_case(data, size, forced, S->max_bytes, c)) {
        S->kstats[forced].invalid++;
        return -1;
    }
    kind_stat & ks = S->kstats[forced];
    ks.execs++;
    ks.special += c.special ? 1 : 0;
    S->n_exec++;

    run_result rr = run_case(c, S->cpu);
    if (rr.flags != 0) {
        // a write outside a tensor or into an input: a memory defect of the CPU backend
        std::fprintf(stderr, "fuzz_ops: MEMORY DEFECT in %s: %s\n  case: %s\n", c.kind->name, rr.detail.c_str(), c.desc.c_str());
        std::abort();
    }
    if (rr.status == RUN_UNSUPPORTED) {
        ks.unsupported++;
        return 0;
    }
    if (rr.status != RUN_OK) {
        std::fprintf(stderr, "fuzz_ops: %s on the CPU: %s (%s)\n", run_status_name(rr.status), rr.detail.c_str(), c.desc.c_str());
        return 0;
    }

    run_result ref;
    if (!S->oracle.call(forced, S->max_bytes, data, size, ref) || ref.status != RUN_OK) {
        ks.oracle_fail++;
        std::fprintf(stderr, "fuzz_ops: the oracle failed on %s (%s): %s\n", c.kind->name, c.desc.c_str(), ref.detail.c_str());
        return 0;
    }
    if (ref.input_hash != rr.input_hash) {
        // the two processes decoded different inputs from the same bytes: a harness defect
        std::fprintf(stderr, "fuzz_ops: DECODE MISMATCH between this process and the oracle on %s (%s)\n",
                     c.kind->name, c.desc.c_str());
        std::abort();
    }

#ifdef FUZZ_OPS_BACKEND_FAST
    const bool fast = true;  // the CPU backend of this build uses -ffp-model=fast
#else
    const bool fast = false;
#endif
    const case_cmp cc = compare_case(c, rr.outs, ref.outs, fast);
    ks.verdicts[cc.v]++;
    if (cc.v != V_PASS) {
        auto  key = std::make_tuple(forced, c.path, (int) cc.v, c.special);
        agg & a   = S->aggs[key];
        a.count++;
        const std::string dir  = S->findings + "/" + c.kind->name;
        const std::string base = dir + "/" + verdict_name(cc.v) + (c.special ? "-special-" : "-") + clean(c.path);
        if (cc.ratio > a.worst_ratio && cc.ratio > stored_ratio(base + ".txt")) {
            a.worst_ratio = cc.ratio;
            a.worst_ulp   = cc.ulp;
            ::mkdir(dir.c_str(), 0755);
            a.file = base + ".bin";
            a.desc = c.desc;
            write_file(a.file, data, size);
            char rbuf[64];
            std::snprintf(rbuf, sizeof(rbuf), "ratio: %.6g\n", cc.ratio);
            const std::string txt = std::string(rbuf) + "kind: " + std::string(c.kind->name) + "\ncase: " + c.desc + "\npath: " + c.path +
                                    "\nthreads: " + std::to_string(c.n_threads) + (c.cpu_repack ? " repack" : "") +
                                    "\nverdict: " + verdict_name(cc.v) + " severity " + verdict_severity(cc.v, c.special) +
                                    "\ncompare: " + cc.text + "\n";
            write_file(a.file.substr(0, a.file.size() - 4) + ".txt", txt.data(), txt.size());
        }
    }
    if (S->n_exec % 2000 == 0) {
        write_stats();
    }
    if ((int) cc.v >= S->abort_v && (S->abort_kind < 0 || S->abort_kind == forced) &&
        !(S->abort_nospecial && c.special)) {
        std::fprintf(stderr, "fuzz_ops: ABORT on %s %s: %s\n  case: %s\n", c.kind->name, verdict_name(cc.v), cc.text.c_str(),
                     c.desc.c_str());
        std::abort();
    }
    return 0;
}
