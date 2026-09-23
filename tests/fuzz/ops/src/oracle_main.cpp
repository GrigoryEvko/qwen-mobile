// ops_oracle: the reference side of the op fuzzer. This binary links the scalar CPU backend of ggml
// with strict IEEE arithmetic (no SIMD options, -ffp-contract=off, -fno-fast-math), one thread and
// the reference paths of the CPU backend (use_ref), like build/oracle-x86.
//
//   ops_oracle serve
//       Serve the libFuzzer harness on stdin and stdout.
//   ops_oracle gen --out PACK [--group G] [--n N] [--seed S] [--len L] [--max-per-dir M]
//                  [--corpus GROUP:DIR]... [--tame-corpus GROUP:DIR]... [--cases KIND:DIR]...
//                  [--enumerate KIND:COUNT]... [--enumerate-from KIND:FIRST:COUNT]... [--max-bytes B]
//       Write a pack of cases: N random cases of each kind of the group, the inputs of libFuzzer
//       corpora (decoded with the group of the fuzzer that made them; a --tame-corpus with the
//       tame values of the fuzz suite), and case files of one kind (a file with the prefix
//       "tame-" decodes with the tame values).
//   ops_oracle compare --pack PACK --results FILE... [--findings DIR] [--max-bytes B]
//       Compare the results of the replay driver (the phone) with the oracle, and print the
//       per-op error table. Write the worst case of each backend, kind, path and verdict.
//   ops_oracle bounds
//       Print the bound rule of each kind.
//   ops_oracle seeds --out DIR [--group G] [--n N] [--len L] [--seed S]
//       Write N valid seed inputs of each kind of the group for the libFuzzer harness.
//   ops_oracle stats FILE...
//       Merge the statistics files of the libFuzzer workers and print the tables.
//   ops_oracle show FILE [--kind K | --group G | --index N] [--write OUT]
//       Decode one case file (or case N of a pack), print it with the facts of its values, and run
//       the oracle on it.
//       Give FUZZ_OPS_TAME=1 for an input of the fuzz suite. --write (after --index) writes the
//       bytes of the case to OUT.
//   ops_oracle same RESULTS_A RESULTS_B
//       Compare the outputs of two result files run by run (the same pack index and backend).
//       The exit status is 0 only if each paired run is bit-identical.

#include "case.h"
#include "compare.h"
#include "exec.h"
#include "wire.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <map>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <vector>

namespace {

using namespace fo;

backend_ctx g_ref;

void usage() {
    std::fprintf(stderr,
                 "usage: ops_oracle serve\n"
                 "       ops_oracle gen --out PACK [--group G] [--n N] [--seed S] [--len L] [--max-per-dir M]\n"
                 "                      [--corpus GROUP:DIR]... [--tame-corpus GROUP:DIR]...\n"
                 "                      [--cases KIND:DIR]... [--enumerate KIND:COUNT]... [--max-bytes B]\n"
                 "       ops_oracle compare --pack PACK --results FILE... [--findings DIR] [--max-bytes B]\n"
                 "       ops_oracle bounds\n"
                 "       ops_oracle seeds --out DIR [--group G] [--n N] [--len L] [--seed S]\n"
                 "       ops_oracle stats FILE...\n"
                 "       ops_oracle show FILE [--kind K | --group G | --index N] [--write OUT]\n"
                 "       ops_oracle same RESULTS_A RESULTS_B\n");
}

// Open the reference CPU backend. Stop the process when it is missing.
void open_ref() {
    g_ref.use_ref       = true;
    g_ref.force_threads = 1;
    std::string why;
    if (!open_backend("CPU", g_ref, why)) {
        std::fprintf(stderr, "ops_oracle: %s\n", why.c_str());
        std::exit(2);
    }
}

// Compute the oracle result of one case.
run_result oracle_run(int32_t forced, uint64_t max_bytes, const uint8_t * data, size_t size) {
    built_case c;
    if (!build_case(data, size, forced, max_bytes, c)) {
        run_result r;
        r.status = RUN_INVALID;
        return r;
    }
    // The reference backend has use_ref set, thus run_case gives it no repack buffer.
    return run_case(c, g_ref);
}

// List the regular files of a directory, sorted by name.
std::vector<std::string> list_dir(const std::string & dir) {
    std::vector<std::string> r;
    DIR * d = opendir(dir.c_str());
    if (!d) {
        return r;
    }
    while (dirent * e = readdir(d)) {
        const std::string p = dir + "/" + e->d_name;
        struct stat       st;
        if (e->d_name[0] != '.' && ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            r.push_back(p);
        }
    }
    closedir(d);
    std::sort(r.begin(), r.end());
    return r;
}

int cmd_gen(int argc, char ** argv) {
    std::string out, group = "all";
    int         n = 50, len = 96;
    uint64_t    seed = 1, max_bytes = uint64_t(64) << 20;
    size_t      max_per_dir = 0;
    std::vector<std::pair<std::string, std::string>> corpora, casedirs;
    std::vector<bool>                                corpus_tame;  // one flag for each corpora entry
    std::vector<std::pair<std::string, std::pair<int, int>>> enumerate;  // KIND, (FIRST, COUNT)
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "ops_oracle gen: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--out") out = next();
        else if (a == "--group") group = next();
        else if (a == "--n") n = std::atoi(next().c_str());
        else if (a == "--len") len = std::atoi(next().c_str());
        else if (a == "--seed") seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--max-bytes") max_bytes = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--max-per-dir") max_per_dir = (size_t) std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--enumerate" || a == "--enumerate-from") {
            // KIND:COUNT (the entries 0 to COUNT - 1) or KIND:FIRST:COUNT
            const std::string v  = next();
            const size_t      c1 = v.find(':');
            const size_t      c2 = c1 == std::string::npos ? std::string::npos : v.find(':', c1 + 1);
            if (c1 == std::string::npos || (a == "--enumerate-from") != (c2 != std::string::npos)) {
                std::fprintf(stderr, "ops_oracle gen: %s takes %s\n", a.c_str(),
                             a == "--enumerate" ? "KIND:COUNT" : "KIND:FIRST:COUNT");
                return 2;
            }
            if (c2 == std::string::npos) {
                enumerate.push_back({ v.substr(0, c1), { 0, std::atoi(v.substr(c1 + 1).c_str()) } });
            } else {
                enumerate.push_back({ v.substr(0, c1), { std::atoi(v.substr(c1 + 1, c2 - c1 - 1).c_str()),
                                                         std::atoi(v.substr(c2 + 1).c_str()) } });
            }
        } else if (a == "--corpus" || a == "--tame-corpus" || a == "--cases") {
            const std::string v = next();
            const size_t      c = v.find(':');
            if (c == std::string::npos) {
                std::fprintf(stderr, "ops_oracle gen: %s takes NAME:DIR\n", a.c_str());
                return 2;
            }
            if (a == "--cases") {
                casedirs.push_back({ v.substr(0, c), v.substr(c + 1) });
            } else {
                corpora.push_back({ v.substr(0, c), v.substr(c + 1) });
                corpus_tame.push_back(a == "--tame-corpus");
            }
        } else {
            usage();
            return 2;
        }
    }
    if (out.empty()) {
        usage();
        return 2;
    }
    std::vector<pack_case> cases;
    auto add = [&](int32_t forced, const std::string & name, std::vector<uint8_t> bytes) {
        built_case c;
        if (bytes.empty() || !build_case(bytes.data(), bytes.size(), forced, max_bytes, c)) {
            return;
        }
        pack_case p;
        p.forced = forced;
        p.name   = name;
        p.bytes  = std::move(bytes);
        cases.push_back(std::move(p));
    };
    for (size_t ci = 0; ci < corpora.size(); ci++) {
        const auto &           cd = corpora[ci];
        const std::vector<int> g  = group_kinds(cd.first);
        if (g.empty()) {
            std::fprintf(stderr, "ops_oracle gen: %s is not a group\n", cd.first.c_str());
            return 2;
        }
        // A corpus of the fuzz suite came from tame decodes, thus its cases keep that decode.
        const int32_t tame = corpus_tame[ci] ? FORCED_TAME : 0;
        // an even subset of a large corpus: every k-th file in name order
        const std::vector<std::string> files = list_dir(cd.second);
        const size_t step = max_per_dir > 0 && files.size() > max_per_dir ? (files.size() + max_per_dir - 1) / max_per_dir : 1;
        for (size_t fi = 0; fi < files.size(); fi += step) {
            std::vector<uint8_t> b;
            if (read_file(files[fi], b) && !b.empty()) {
                add(g[b[0] % g.size()] | tame, files[fi], b);
            }
        }
    }
    for (const auto & cd : casedirs) {
        const int k = kind_index(cd.first);
        if (k < 0) {
            std::fprintf(stderr, "ops_oracle gen: %s is not a kind\n", cd.first.c_str());
            return 2;
        }
        for (const auto & f : list_dir(cd.second)) {
            std::vector<uint8_t> b;
            // a regression input from the fuzz suite has the prefix "tame-" (refer to run.sh)
            const size_t slash = f.find_last_of('/');
            const bool   tame  = f.compare(slash == std::string::npos ? 0 : slash + 1, 5, "tame-") == 0;
            if (f.size() > 4 && f.substr(f.size() - 4) == ".bin" && read_file(f, b)) {
                add(k | (tame ? FORCED_TAME : 0), f, b);
            }
        }
    }
    // The enumerated cases: 4 threads, the CPU repack buffer, no wild values, a seed of i + 1, and
    // byte 10 = i (the entry of a table kind such as mm_model).
    for (const auto & en : enumerate) {
        const int k = kind_index(en.first);
        if (k < 0) {
            std::fprintf(stderr, "ops_oracle gen: %s is not a kind\n", en.first.c_str());
            return 2;
        }
        for (int i = en.second.first; i < en.second.first + en.second.second; i++) {
            std::vector<uint8_t> b(11, 0);
            b[0] = (uint8_t) k;
            b[1] = 7;
            b[2] = (uint8_t) (i + 1);
            b[10] = (uint8_t) i;
            add(k, std::string("enumerate/") + en.first + "/" + std::to_string(i), b);
        }
    }
    rng r(seed);
    for (int k : group_kinds(group)) {
        for (int i = 0, tries = 0; i < n && tries < n * 20; tries++) {
            std::vector<uint8_t> b((size_t) len);
            for (auto & x : b) {
                x = (uint8_t) r.u32();
            }
            b[0] = (uint8_t) k;
            const size_t before = cases.size();
            add(k, std::string("random/") + kinds()[k].name + "/" + std::to_string(i), b);
            i += cases.size() > before ? 1 : 0;
        }
    }
    std::string why;
    if (!write_pack(out, cases, why)) {
        std::fprintf(stderr, "ops_oracle gen: %s\n", why.c_str());
        return 1;
    }
    std::printf("ops_oracle gen: %zu cases in %s\n", cases.size(), out.c_str());
    return 0;
}

// The error table of one backend, kind and path.
struct row {
    uint64_t    runs = 0, unsupported = 0, crashed = 0, failed = 0, defects = 0, invalid = 0, oracle_fail = 0;
    uint64_t    verdicts[5] = { 0, 0, 0, 0, 0 };
    uint64_t    special = 0;
    double      max_ratio = 0.0, max_ratio_nsp = 0.0, max_ulp = 0.0, max_ulp_nsp = 0.0;
    std::string worst;
    std::string unsupported_example;
};

int cmd_compare(int argc, char ** argv) {
    std::string              pack, findings;
    std::vector<std::string> results;
    uint64_t                 max_bytes = uint64_t(64) << 20;
    const char *             list_env  = std::getenv("FUZZ_OPS_COMPARE_LIST");
    const bool               list_runs = list_env != nullptr && list_env[0] == '1';
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--pack" && i + 1 < argc) pack = argv[++i];
        else if (a == "--results" && i + 1 < argc) results.push_back(argv[++i]);
        else if (a == "--findings" && i + 1 < argc) findings = argv[++i];
        else if (a == "--max-bytes" && i + 1 < argc) max_bytes = std::strtoull(argv[++i], nullptr, 10);
        else {
            usage();
            return 2;
        }
    }
    if (pack.empty() || results.empty()) {
        usage();
        return 2;
    }
    std::vector<pack_case> cases;
    std::string            why;
    if (!read_pack(pack, cases, why)) {
        std::fprintf(stderr, "ops_oracle compare: %s\n", why.c_str());
        return 1;
    }
    std::vector<result_rec> recs;
    for (const auto & rp : results) {
        if (!read_results(rp, recs, why)) {
            std::fprintf(stderr, "ops_oracle compare: %s\n", why.c_str());
            return 1;
        }
    }
    open_ref();
    if (!findings.empty()) {
        ::mkdir(findings.c_str(), 0755);
    }
    std::map<uint32_t, run_result>                               oracle_cache;
    std::map<std::tuple<std::string, std::string, std::string>, row> table;
    std::map<std::tuple<std::string, std::string, std::string, int>, double> worst_written;
    uint64_t n_defects = 0;
    for (const auto & rec : recs) {
        if (rec.idx >= cases.size()) {
            continue;
        }
        const pack_case & pc = cases[rec.idx];
        built_case        c;
        if (!build_case(pc.bytes.data(), pc.bytes.size(), pc.forced, max_bytes, c)) {
            table[{ rec.tag, "?", "?" }].invalid++;
            continue;
        }
        row & rw = table[{ rec.tag, c.kind->name, c.path }];
        rw.runs++;
        if (rec.rr.status != RUN_CRASHED && rec.rr.status != RUN_INVALID && rec.rr.input_hash != case_hash(c)) {
            // the phone and the host decoded different inputs: a harness defect, not an op finding
            rw.invalid++;
            std::printf("DECODE MISMATCH %s case %u %s\n", rec.tag.c_str(), rec.idx, c.desc.c_str());
            continue;
        }
        auto write_finding = [&](int v, const std::string & text) {
            if (findings.empty()) {
                return;
            }
            const std::string dir = findings + "/" + rec.tag;
            ::mkdir(dir.c_str(), 0755);
            const std::string kdir = dir + "/" + c.kind->name;
            ::mkdir(kdir.c_str(), 0755);
            std::string p;
            for (char ch : c.path) {
                p += (std::isalnum((unsigned char) ch) || ch == '-' || ch == '_') ? ch : '.';
            }
            const std::string base = kdir + "/" + (v >= 0 ? verdict_name(v) : "defect") + (c.special ? "-special-" : "-") + p;
            write_file(base + ".bin", pc.bytes.data(), pc.bytes.size());
            const std::string txt = "kind: " + std::string(c.kind->name) + "\ncase: " + c.desc + "\npath: " + c.path +
                                    "\npack index: " + std::to_string(rec.idx) + " (" + pc.name + ")\nbackend: " + rec.tag +
                                    "\nthreads: " + std::to_string(c.n_threads) + "\n" + text + "\n";
            write_file(base + ".txt", txt.data(), txt.size());
        };
        if (rec.rr.flags != 0) {
            rw.defects++;
            n_defects++;
            std::printf("DEFECT %s %s: %s | %s\n", rec.tag.c_str(), c.kind->name, rec.rr.detail.c_str(), c.desc.c_str());
            write_finding(-1, "defect: " + rec.rr.detail);
        }
        if (rec.rr.status == RUN_UNSUPPORTED) {
            rw.unsupported++;
            if (rw.unsupported_example.empty()) {
                rw.unsupported_example = rec.rr.detail;
            }
            continue;
        }
        if (rec.rr.status == RUN_CRASHED) {
            rw.crashed++;
            std::printf("CRASH %s %s: %s | %s\n", rec.tag.c_str(), c.kind->name, rec.rr.detail.c_str(), c.desc.c_str());
            write_finding(-1, "crash: " + rec.rr.detail);
            continue;
        }
        if (rec.rr.status != RUN_OK) {
            rw.failed++;
            continue;
        }
        auto it = oracle_cache.find(rec.idx);
        if (it == oracle_cache.end()) {
            built_case co;
            build_case(pc.bytes.data(), pc.bytes.size(), pc.forced, max_bytes, co);
            it = oracle_cache.emplace(rec.idx, run_case(co, g_ref)).first;
        }
        if (it->second.status != RUN_OK) {
            rw.oracle_fail++;
            continue;
        }
        // The phone CPU backend is built with -ffp-model=fast in both profiles (the app's flags),
        // thus its results get the fast-math slack. The HTP results keep the plain rule.
        const bool     fast = rec.tag.rfind("CPU", 0) == 0;
        const case_cmp cc   = compare_case(c, rec.rr.outs, it->second.outs, fast);
        if (list_runs) {
            // one line for each run: the evidence for a single case (FUZZ_OPS_COMPARE_LIST=1)
            std::printf("RUN %u %s %s %s verdict=%s special=%d ratio=%.3g ulp=%.3g | %s\n", rec.idx, rec.tag.c_str(),
                        c.kind->name, c.path.c_str(), verdict_name(cc.v), (int) c.special, cc.ratio, cc.ulp,
                        c.desc.c_str());
        }
        rw.verdicts[cc.v]++;
        rw.special += c.special ? 1 : 0;
        if (cc.ratio > rw.max_ratio) {
            rw.max_ratio = cc.ratio;
            rw.worst     = std::to_string(rec.idx) + " " + c.desc;
        }
        rw.max_ulp = std::max(rw.max_ulp, cc.ulp);
        if (!c.special) {
            rw.max_ratio_nsp = std::max(rw.max_ratio_nsp, cc.ratio);
            rw.max_ulp_nsp   = std::max(rw.max_ulp_nsp, cc.ulp);
        }
        if (cc.v >= V_STRICT || cc.v == V_SUBNORMAL) {
            double & w = worst_written[{ rec.tag, c.kind->name, c.path, (int) cc.v * 2 + (c.special ? 1 : 0) }];
            if (cc.ratio >= w) {
                w = cc.ratio;
                write_finding(cc.v, std::string("verdict: ") + verdict_name(cc.v) + " severity " +
                                        verdict_severity(cc.v, c.special) + "\ncompare: " + cc.text);
            }
        }
    }
    std::printf("%-12s %-16s %-24s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %10s %10s %10s\n", "backend", "kind", "path",
                "runs", "inval", "unsup", "crash", "defect", "pass", "subn", "strict", "loose", "nonfin", "max_ratio",
                "ratio_nsp", "ulp_nsp");
    for (const auto & it : table) {
        const row & rw = it.second;
        std::printf("%-12s %-16s %-24s %6llu %6llu %6llu %6llu %6llu %6llu %6llu %6llu %6llu %6llu %10.3g %10.3g %10.3g\n",
                    std::get<0>(it.first).c_str(), std::get<1>(it.first).c_str(), std::get<2>(it.first).c_str(),
                    (unsigned long long) rw.runs, (unsigned long long) rw.invalid, (unsigned long long) rw.unsupported,
                    (unsigned long long) rw.crashed,
                    (unsigned long long) rw.defects, (unsigned long long) rw.verdicts[0], (unsigned long long) rw.verdicts[1],
                    (unsigned long long) rw.verdicts[2], (unsigned long long) rw.verdicts[3],
                    (unsigned long long) rw.verdicts[4], rw.max_ratio, rw.max_ratio_nsp, rw.max_ulp_nsp);
    }
    std::printf("\nunsupported examples:\n");
    for (const auto & it : table) {
        if (!it.second.unsupported_example.empty()) {
            std::printf("  %s %s %s: %s\n", std::get<0>(it.first).c_str(), std::get<1>(it.first).c_str(),
                        std::get<2>(it.first).c_str(), it.second.unsupported_example.c_str());
        }
    }
    std::printf("\nrecords %zu, memory defects %llu\n", recs.size(), (unsigned long long) n_defects);
    return 0;
}

// Write seed inputs for the libFuzzer harness of a group: N valid cases of each kind, with byte 0
// set so that the harness of the group selects that kind.
int cmd_seeds(int argc, char ** argv) {
    std::string out, group = "all";
    int         n = 4, len = 96;
    uint64_t    seed = 1;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--out") out = argv[i + 1];
        else if (a == "--group") group = argv[i + 1];
        else if (a == "--n") n = std::atoi(argv[i + 1]);
        else if (a == "--len") len = std::atoi(argv[i + 1]);
        else if (a == "--seed") seed = std::strtoull(argv[i + 1], nullptr, 10);
    }
    const std::vector<int> g = group_kinds(group);
    if (out.empty() || g.empty()) {
        usage();
        return 2;
    }
    ::mkdir(out.c_str(), 0755);
    rng r(seed);
    int written = 0;
    for (size_t j = 0; j < g.size(); j++) {
        for (int i = 0, tries = 0; i < n && tries < 50 * n; tries++) {
            std::vector<uint8_t> b((size_t) len);
            for (auto & x : b) {
                x = (uint8_t) r.u32();
            }
            b[0] = (uint8_t) j;
            built_case c;
            if (!build_case(b.data(), b.size(), g[j], uint64_t(64) << 20, c)) {
                continue;
            }
            const std::string p = out + "/" + kinds()[g[j]].name + "-" + std::to_string(i) + ".bin";
            write_file(p, b.data(), b.size());
            i++;
            written++;
        }
    }
    std::printf("ops_oracle seeds: %d files in %s\n", written, out.c_str());
    return 0;
}

// Split a line at tabs.
std::vector<std::string> split_tabs(const std::string & s) {
    std::vector<std::string> r;
    size_t                   p = 0;
    while (true) {
        const size_t q = s.find('\t', p);
        r.push_back(s.substr(p, q == std::string::npos ? std::string::npos : q - p));
        if (q == std::string::npos) {
            break;
        }
        p = q + 1;
    }
    return r;
}

// Merge the statistics files of the libFuzzer workers (stats-<pid>.tsv) and print two tables:
// the executions and verdicts of each kind, and the worst finding of each kind, path and verdict.
int cmd_stats(int argc, char ** argv) {
    struct kstat {
        uint64_t v[10] = { 0 };
    };
    struct astat {
        uint64_t    count = 0;
        double      ratio = -1.0, ulp = 0.0;
        std::string file, desc;
    };
    std::map<std::string, kstat>                                            ks;
    std::map<std::tuple<std::string, std::string, std::string, std::string>, astat> as;
    for (int i = 2; i < argc; i++) {
        FILE * f = std::fopen(argv[i], "r");
        if (!f) {
            continue;
        }
        char line[4096];
        while (std::fgets(line, sizeof(line), f)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
                s.pop_back();
            }
            const std::vector<std::string> t = split_tabs(s);
            if (t.size() >= 12 && t[0] == "K") {
                kstat & k = ks[t[1]];
                for (int j = 0; j < 10; j++) {
                    k.v[j] += std::strtoull(t[2 + j].c_str(), nullptr, 10);
                }
            } else if (t.size() >= 10 && t[0] == "A") {
                astat &      a = as[{ t[1], t[2], t[3], t[4] }];
                const double r = std::atof(t[6].c_str());
                a.count += std::strtoull(t[5].c_str(), nullptr, 10);
                if (r > a.ratio) {
                    a.ratio = r;
                    a.ulp   = std::atof(t[7].c_str());
                    a.file  = t[8];
                    a.desc  = t[9];
                }
            }
        }
        std::fclose(f);
    }
    std::printf("%-16s %9s %7s %7s %7s %7s %9s %7s %7s %7s %7s\n", "kind", "execs", "invalid", "unsup", "orfail",
                "special", "pass", "subn", "strict", "loose", "nonfin");
    for (const auto & it : ks) {
        const kstat & k = it.second;
        std::printf("%-16s %9llu %7llu %7llu %7llu %7llu %9llu %7llu %7llu %7llu %7llu\n", it.first.c_str(),
                    (unsigned long long) k.v[0], (unsigned long long) k.v[1], (unsigned long long) k.v[2],
                    (unsigned long long) k.v[3], (unsigned long long) k.v[4], (unsigned long long) k.v[5],
                    (unsigned long long) k.v[6], (unsigned long long) k.v[7], (unsigned long long) k.v[8],
                    (unsigned long long) k.v[9]);
    }
    std::printf("\n%-16s %-26s %-13s %-3s %7s %10s %10s  %s\n", "kind", "path", "verdict", "sp", "count", "ratio", "ulp",
                "worst case");
    for (const auto & it : as) {
        std::printf("%-16s %-26s %-13s %-3s %7llu %10.3g %10.3g  %s\n    %s\n", std::get<0>(it.first).c_str(),
                    std::get<1>(it.first).c_str(), std::get<2>(it.first).c_str(), std::get<3>(it.first).c_str(),
                    (unsigned long long) it.second.count, it.second.ratio, it.second.ulp, it.second.desc.c_str(),
                    it.second.file.c_str());
    }
    return 0;
}

int cmd_bounds() {
    for (const auto & k : kinds()) {
        std::printf("%s (%s)\n  %s\n\n", k.name, k.group, k.bound_text);
    }
    return 0;
}

// Compare the outputs of two result files run by index, for the check that a fix leaves the
// output bit-identical. A record pairs with the record of the same pack
// index and the same tag after the suffix of the build is removed (the part from the first "-").
// Print the count of identical, different and unpaired runs and the first differences. The time
// is O(r log r + b) for r records and b output bytes.
int cmd_same(int argc, char ** argv) {
    if (argc < 4) {
        usage();
        return 2;
    }
    std::vector<result_rec> a, b;
    std::string             why;
    if (!read_results(argv[2], a, why) || !read_results(argv[3], b, why)) {
        std::fprintf(stderr, "ops_oracle same: %s\n", why.c_str());
        return 1;
    }
    auto key = [](const result_rec & r) {
        const size_t dash = r.tag.find('-');
        return std::make_pair(r.idx, dash == std::string::npos ? r.tag : r.tag.substr(0, dash));
    };
    std::map<std::pair<uint32_t, std::string>, const result_rec *> ib;
    for (const auto & r : b) {
        ib[key(r)] = &r;
    }
    size_t same = 0, diff = 0, unpaired = 0, shown = 0;
    const char * max_env   = std::getenv("FUZZ_OPS_SAME_MAX");  // the most differences to print
    const size_t max_shown = max_env ? (size_t) std::strtoull(max_env, nullptr, 10) : 10;
    for (const auto & r : a) {
        const auto it = ib.find(key(r));
        if (it == ib.end()) {
            unpaired++;
            continue;
        }
        const result_rec & o = *it->second;
        const bool equal = r.rr.status == o.rr.status && r.rr.outs == o.rr.outs && r.rr.input_hash == o.rr.input_hash;
        if (equal) {
            same++;
            continue;
        }
        diff++;
        if (shown++ < max_shown) {
            std::printf("different: case %u %s: status %u/%u, %zu/%zu outputs\n", r.idx, key(r).second.c_str(),
                        r.rr.status, o.rr.status, r.rr.outs.size(), o.rr.outs.size());
        }
    }
    std::printf("ops_oracle same: %zu identical, %zu different, %zu unpaired runs\n", same, diff, unpaired);
    return diff == 0 ? 0 : 1;
}

// Print the value facts of one input: for a float or quantized input the counts of NaN, Inf,
// subnormal and zero values and the range of the finite magnitudes; for Q4_0 and Q8_0 also the
// block scales (zero, Inf, NaN, 65504, subnormal, negative); for I32 the range. The time is
// O(n) for n elements.
void print_leaf_summary(const leaf & l) {
    const ggml_type type = l.t->type;
    const int64_t   n    = ggml_nelements(l.t);
    if (type == GGML_TYPE_I32) {
        int32_t lo = 0, hi = 0;
        for (int64_t i = 0; i < n; i++) {
            int32_t x;
            std::memcpy(&x, l.bytes.data() + i * 4, 4);
            lo = i == 0 ? x : std::min(lo, x);
            hi = i == 0 ? x : std::max(hi, x);
        }
        std::printf("    i32 range [%d, %d]%s", lo, hi, n <= 16 ? ":" : "\n");
        for (int64_t i = 0; n <= 16 && i < n; i++) {
            int32_t x;
            std::memcpy(&x, l.bytes.data() + i * 4, 4);
            std::printf(" %d%s", x, i + 1 == n ? "\n" : "");
        }
        return;
    }
    const std::vector<float> v = raw_to_f32(type, l.bytes.data(), n);
    int64_t n_nan = 0, n_inf = 0, n_sub = 0, n_zero = 0, i_max = -1;
    float   amin = 0.0f, amax = 0.0f;
    bool    any = false;
    for (int64_t i = 0; i < n; i++) {
        const float f = v[i];
        const float a = std::fabs(f);
        if (std::isfinite(f) && (i_max < 0 || a > amax)) {
            i_max = i;
        }
        if (std::isnan(f)) {
            n_nan++;
        } else if (std::isinf(f)) {
            n_inf++;
        } else if (f == 0.0f) {
            n_zero++;
        } else {
            n_sub += a < 1.17549435e-38f ? 1 : 0;
            amin = any ? std::min(amin, a) : a;
            amax = any ? std::max(amax, a) : a;
            any  = true;
        }
    }
    const int64_t ne0 = l.t->ne[0], ne1 = l.t->ne[1], ne2 = l.t->ne[2];
    std::printf("    values: nan %lld inf %lld subnormal %lld zero %lld |finite nonzero| in [%.3g, %.3g], "
                "the largest at [%lld,%lld,%lld,%lld]\n",
                (long long) n_nan, (long long) n_inf, (long long) n_sub, (long long) n_zero, amin, amax,
                (long long) (i_max % ne0), (long long) (i_max / ne0 % ne1), (long long) (i_max / (ne0 * ne1) % ne2),
                (long long) (i_max / (ne0 * ne1 * ne2)));
    if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q8_0) {
        return;
    }
    const size_t  bs   = type == GGML_TYPE_Q8_0 ? 34 : 18;
    const int64_t nblk = n / 32;
    int64_t       s_zero = 0, s_inf = 0, s_nan = 0, s_max = 0, s_sub = 0, s_neg = 0;
    for (int64_t ib = 0; ib < nblk; ib++) {
        uint16_t d;
        std::memcpy(&d, l.bytes.data() + ib * bs, 2);
        const uint16_t m = d & 0x7fffu;
        s_zero += m == 0 ? 1 : 0;
        s_inf += m == 0x7c00u ? 1 : 0;
        s_nan += m > 0x7c00u ? 1 : 0;
        s_max += m == 0x7bffu ? 1 : 0;
        s_sub += (m != 0 && m < 0x0400u) ? 1 : 0;
        s_neg += (d & 0x8000u) ? 1 : 0;
    }
    std::printf("    block scales (%lld): zero %lld inf %lld nan %lld 65504 %lld subnormal %lld negative %lld\n",
                (long long) nblk, (long long) s_zero, (long long) s_inf, (long long) s_nan, (long long) s_max,
                (long long) s_sub, (long long) s_neg);
}

int cmd_show(int argc, char ** argv) {
    if (argc < 3) {
        usage();
        return 2;
    }
    std::vector<uint8_t> b;
    if (!read_file(argv[2], b) || b.empty()) {
        std::fprintf(stderr, "ops_oracle show: cannot read %s\n", argv[2]);
        return 1;
    }
    int forced = -1;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--kind") {
            forced = kind_index(argv[i + 1]);
        } else if (a == "--group") {
            const std::vector<int> g = group_kinds(argv[i + 1]);
            forced = g.empty() ? -1 : g[b[0] % g.size()];
        } else if (a == "--index") {
            // FILE is a pack: take its case with this index, with its forced kind and flags
            std::vector<pack_case> cases;
            std::string            why;
            const size_t           idx = (size_t) std::strtoull(argv[i + 1], nullptr, 10);
            if (!read_pack(argv[2], cases, why) || idx >= cases.size()) {
                std::fprintf(stderr, "ops_oracle show: no case %zu in the pack %s %s\n", idx, argv[2], why.c_str());
                return 1;
            }
            b      = cases[idx].bytes;
            forced = cases[idx].forced;
            std::printf("pack case %zu: %s\n", idx, cases[idx].name.c_str());
        } else if (a == "--write") {
            // write the bytes of the case (for example case N of a pack) to a regression file
            if (!write_file(argv[i + 1], b.data(), b.size())) {
                std::fprintf(stderr, "ops_oracle show: cannot write %s\n", argv[i + 1]);
                return 1;
            }
            std::printf("wrote %zu bytes to %s\n", b.size(), argv[i + 1]);
        }
    }
    built_case c;
    if (!build_case(b.data(), b.size(), forced, uint64_t(64) << 20, c)) {
        std::printf("invalid case\n");
        return 1;
    }
    std::printf("kind: %s\ncase: %s\npath: %s\nthreads: %d repack: %d special: %d\n", c.kind->name, c.desc.c_str(),
                c.path.c_str(), c.n_threads, (int) c.cpu_repack, (int) c.special);
    for (const auto & l : c.leaves) {
        std::printf("  input %s %s [%lld,%lld,%lld,%lld] role %d\n", l.t->name, ggml_type_name(l.t->type),
                    (long long) l.t->ne[0], (long long) l.t->ne[1], (long long) l.t->ne[2], (long long) l.t->ne[3],
                    (int) l.role);
        print_leaf_summary(l);
    }
    for (int i = 0; i < ggml_graph_n_nodes(c.gf); i++) {
        const ggml_tensor * n = ggml_graph_node(c.gf, i);
        std::printf("  node %d %s %s [%lld,%lld,%lld,%lld]\n", i, ggml_op_desc(n), ggml_type_name(n->type),
                    (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3]);
    }
    open_ref();
    const run_result r = run_case(c, g_ref);
    std::printf("oracle: %s %s %.3f ms\n", run_status_name(r.status), r.detail.c_str(), r.ms);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "serve") {
        open_ref();
        return serve_loop(oracle_run);
    }
    if (cmd == "gen") {
        return cmd_gen(argc, argv);
    }
    if (cmd == "compare") {
        return cmd_compare(argc, argv);
    }
    if (cmd == "bounds") {
        return cmd_bounds();
    }
    if (cmd == "seeds") {
        return cmd_seeds(argc, argv);
    }
    if (cmd == "stats") {
        return cmd_stats(argc, argv);
    }
    if (cmd == "show") {
        return cmd_show(argc, argv);
    }
    if (cmd == "same") {
        return cmd_same(argc, argv);
    }
    usage();
    return 2;
}
