// plan: the chunks that the host gives each HMX 2D matmul of Qwen3.5 4B, with the old cost model
// (htp_mm_hmx_solve_2d_params) and with the cost model of the kernel (htp_mm_hmx_solve_2d_cost), and the
// checks of the new model. The program includes htp/matmul-ops.h of a llama.cpp tree, thus it computes
// what the host of that tree computes. It needs no phone and no DSP.
//
// Build (clang, because the header uses __fp16):
//   clang++ -std=c++17 -O2 -I TREE/ggml/src/ggml-hexagon/htp -I TREE/ggml/src/ggml-hexagon \
//       -I TREE/ggml/include tools/stages/mmsolve/plan.cpp -o plan
//
// Usage:
//   plan table            the chunks of each 4B shape at 5 to 1024 tokens, old and new
//   plan pair K N M TYPE  the chunks of one shape, old and new, and the checks of "check" for it
//   plan sensitivity      for chunk times of 0 to 40 us, the 4B cases whose selection differs from the selection
//                         at HTP_MM_HMX_CHUNK_NS
//   plan candidates K N M TYPE   each n chunk with its largest m chunk, the passes, the chunks and the
//                         cost terms, for one shape (TYPE q8_0, f16 or f32)
//   plan fit K N M TYPE MC NC    the layout of a GGML_HEXAGON_MM_CHUNKS request, as the host gets it
//   plan check            the invariants of the new model over each 4B shape at each token count from 5
//                         to 1024, and over random shapes. The exit code is 1 when one fails.
//
// The invariants of "check":
//   1. The new model gives a layout that is not larger than the budget, and its VTCM value is the value
//      of htp_mm_hmx_get_2d_vtcm_size.
//   2. m_chunk and n_chunk are multiples of 32, m_chunk <= m aligned up to 32, n_chunk <= n.
//   3. The new model never gives more passes than the old model.
//   4. When the old model gives one pass, the new model gives the chunks of the old model.
// Time: O(shapes x tokens x n / 32 x log(m / 32)) layout builds, less than 2 s.

#include "matmul-ops.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// The VTCM budget and the HVX threads of a session on the v79 of the phone. The environment variables
// PLAN_VTCM and PLAN_THREADS set other values, for example the 1 MB of the HMX runs of tools/htp-lab.
size_t VTCM      = 8388608;
int    N_THREADS = 6;

struct shape {
    const char * name;
    int          type;  // HTP_TYPE_*
    uint32_t     k;
    uint32_t     n;
};

// The 2D weights of the 4B Q8_0 file (weights/gguf/Qwen3.5-4B-Q8_0.gguf). The MTP block (blk.32) is F16.
// The head is token_embd (tied): the verify batch of the MTP draft gives logits for each row.
const shape SHAPES[] = {
    { "ffn_gate, ffn_up", HTP_TYPE_Q8_0, 2560, 9216 },
    { "ffn_down", HTP_TYPE_Q8_0, 9216, 2560 },
    { "attn_qkv, attn_q", HTP_TYPE_Q8_0, 2560, 8192 },
    { "attn_gate", HTP_TYPE_Q8_0, 2560, 4096 },
    { "ssm_out, attn_output", HTP_TYPE_Q8_0, 4096, 2560 },
    { "attn_k, attn_v", HTP_TYPE_Q8_0, 2560, 1024 },
    { "head (token_embd)", HTP_TYPE_Q8_0, 2560, 248320 },
    { "draft head shortlist", HTP_TYPE_Q8_0, 2560, 32768 },
    { "ssm_alpha, ssm_beta", HTP_TYPE_F32, 2560, 32 },
    { "mtp ffn_gate, ffn_up", HTP_TYPE_F16, 2560, 9216 },
    { "mtp ffn_down", HTP_TYPE_F16, 9216, 2560 },
    { "mtp attn_q", HTP_TYPE_F16, 2560, 8192 },
    { "mtp attn_output", HTP_TYPE_F16, 4096, 2560 },
    { "mtp attn_k, attn_v", HTP_TYPE_F16, 2560, 1024 },
    { "mtp eh_proj", HTP_TYPE_F16, 5120, 2560 },
};

const uint32_t TOKENS[] = { 5, 6, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024 };

// One selection of the chunks and its counts.
struct plan_t {
    bool   ok       = false;
    bool   pipeline = false;
    size_t mc = 0, nc = 0, vtcm = 0;
    int    threads = 0;

    uint64_t passes(uint32_t m) const { return mc ? ((uint64_t) m + mc - 1) / mc : 0; }

    uint64_t chunks(uint32_t m, uint32_t n) const { return nc ? passes(m) * (((uint64_t) n + nc - 1) / nc) : 0; }
};

uint32_t pad32(uint32_t v) {
    return (v + 31) / 32 * 32;
}

// The solve of ggml_hexagon_precompute_hmx_mm_params for a 2D MUL_MAT: the pipelined layout when the token
// count is more than HTP_MM_HMX_MIN_NROWS, and the serial layout when the pipelined solve fails. chunk_ns is the
// chunk time of the new model.
plan_t solve(const shape & s, uint32_t m, bool new_model, uint64_t chunk_ns = HTP_MM_HMX_CHUNK_NS) {
    plan_t         p;
    const uint32_t ats  = htp_mm_get_weight_aligned_tile_size(s.type);
    const uint32_t npad = pad32(s.n);
    for (int pass = 0; pass < 2; pass++) {
        const bool pipe = pass == 0 && htp_mm_hmx_pipeline(m);
        if (pass == 1 && !htp_mm_hmx_pipeline(m)) {
            break;
        }
        p.pipeline = pipe;
        p.ok       = new_model ? htp_mm_hmx_solve_2d_cost_ns(s.type, s.k, npad, pad32(m), m, N_THREADS, pipe, ats, VTCM,
                                                             chunk_ns, &p.mc, &p.nc, &p.threads, &p.vtcm)
                               : htp_mm_hmx_solve_2d_params(s.type, s.k, 0, npad, pad32(m), m, N_THREADS, pipe, false,
                                                            ats, VTCM, &p.mc, &p.nc, &p.threads, &p.vtcm);
        if (p.ok) {
            return p;
        }
    }
    return p;
}

std::string text(const plan_t & p, uint32_t m, uint32_t n) {
    if (!p.ok) {
        return "no layout";
    }
    char buf[160];
    snprintf(buf, sizeof(buf), "mc %4zu nc %3zu thr %d passes %2" PRIu64 " chunks %4" PRIu64 " vtcm %7zu%s", p.mc,
             p.nc, p.threads, p.passes(m), p.chunks(m, n), p.vtcm, p.pipeline ? "" : " serial");
    return buf;
}

int parse_type(const char * t) {
    if (!strcmp(t, "q8_0")) {
        return HTP_TYPE_Q8_0;
    }
    if (!strcmp(t, "f16")) {
        return HTP_TYPE_F16;
    }
    if (!strcmp(t, "f32")) {
        return HTP_TYPE_F32;
    }
    fprintf(stderr, "plan: the type %s is not q8_0, f16 or f32\n", t);
    exit(2);
}

const char * type_name(int t) {
    return t == HTP_TYPE_Q8_0 ? "q8_0" : t == HTP_TYPE_F16 ? "f16" : "f32";
}

int cmd_table() {
    printf("The chunks of each HMX 2D matmul of the 4B (VTCM %zu, %d threads), old and new cost model\n", VTCM,
           N_THREADS);
    printf("HTP_MM_HMX_CHUNK_NS %d, HTP_MM_HMX_DMA_BYTES_PER_US %d, HTP_MM_HMX_DEQUANT_PER_THREAD_US %d\n",
           HTP_MM_HMX_CHUNK_NS, HTP_MM_HMX_DMA_BYTES_PER_US, HTP_MM_HMX_DEQUANT_PER_THREAD_US);
    for (const shape & s : SHAPES) {
        printf("\n%s %s k %u n %u\n", s.name, type_name(s.type), s.k, s.n);
        for (uint32_t m : TOKENS) {
            const plan_t o = solve(s, m, false);
            const plan_t w = solve(s, m, true);
            const bool   same = o.ok == w.ok && o.mc == w.mc && o.nc == w.nc && o.threads == w.threads;
            printf("  m %4u  old: %-66s new: %s%s\n", m, text(o, m, s.n).c_str(), same ? "same" : text(w, m, s.n).c_str(),
                   (!same && w.passes(m) < o.passes(m)) ? "  (less passes)" : "");
        }
    }
    return 0;
}

// For each chunk time of a list, the count of the 4B cases (each shape at 5 to 1024 tokens) with a selection other
// than the selection at HTP_MM_HMX_CHUNK_NS, and the first such cases. Thus the range of chunk times in which the
// constant decides nothing. O(times x shapes x tokens) solves.
int cmd_sensitivity() {
    const double times_us[] = { 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40 };
    printf("the 4B cases (each shape at 5 to 1024 tokens) whose selection differs from the selection at "
           "HTP_MM_HMX_CHUNK_NS %d ns\n", HTP_MM_HMX_CHUNK_NS);
    for (double us : times_us) {
        const uint64_t ns = (uint64_t) (us * 1000.0 + 0.5);
        size_t         n_diff = 0;
        std::string    first;
        for (const shape & s : SHAPES) {
            for (uint32_t m = 5; m <= 1024; m++) {
                const plan_t a = solve(s, m, true);
                const plan_t b = solve(s, m, true, ns);
                if (a.mc != b.mc || a.nc != b.nc) {
                    if (n_diff < 3) {
                        char buf[200];
                        snprintf(buf, sizeof(buf), "%s%s m %u: %zux%zu -> %zux%zu", first.empty() ? "" : "; ", s.name,
                                 m, a.mc, a.nc, b.mc, b.nc);
                        first += buf;
                    }
                    n_diff++;
                }
            }
        }
        printf("  %5.1f us: %5zu cases differ%s%s\n", us, n_diff, n_diff ? ", the first: " : "", first.c_str());
    }
    return 0;
}

int cmd_candidates(uint32_t k, uint32_t n, uint32_t m, int type) {
    const uint32_t ats    = htp_mm_get_weight_aligned_tile_size(type);
    const bool     pipe   = htp_mm_hmx_pipeline(m);
    const uint64_t row_ns = htp_mm_hmx_row_ns(type, k, N_THREADS);
    const uint32_t npad   = pad32(n);
    printf("k %u n %u m %u %s, pipeline %d, row %" PRIu64 " ns, one pass %.1f us\n", k, n, m, type_name(type), pipe,
           row_ns, row_ns * npad / 1e3);
    printf("  %5s %5s %6s %6s %7s %10s %10s  the chunk time that makes the cost of this nc and of the next "
           "wider nc equal\n",
           "nc", "mc", "passes", "chunks", "vtcm", "pass us", "chunks us");
    uint64_t prev_pass = 0, prev_chunks = 0;
    for (size_t nc = npad / 32 * 32; nc >= 32; nc -= 32) {
        const size_t mc = htp_mm_hmx_max_m_chunk(type, k, nc, pad32(m), pipe, N_THREADS, ats, VTCM);
        if (mc == 0) {
            continue;
        }
        const uint64_t passes = ((uint64_t) m + mc - 1) / mc;
        const uint64_t chunks = passes * ((npad + nc - 1) / nc);
        const uint64_t pass   = passes * npad * row_ns;
        char           even[64] = "";
        if (prev_chunks && chunks > prev_chunks && pass < prev_pass) {
            snprintf(even, sizeof(even), "%.2f us", (prev_pass - pass) / 1e3 / (double) (chunks - prev_chunks));
        }
        printf("  %5zu %5zu %6" PRIu64 " %6" PRIu64 " %7zu %10.1f %10.1f  %s\n", nc, mc, passes, chunks,
               htp_mm_hmx_get_2d_vtcm_size(type, k, mc, nc, pipe, N_THREADS, ats), pass / 1e3,
               chunks * HTP_MM_HMX_CHUNK_NS / 1e3, even);
        prev_pass   = pass;
        prev_chunks = chunks;
        if (passes == 1) {
            break;
        }
    }
    return 0;
}

int cmd_fit(uint32_t k, uint32_t n, uint32_t m, int type, size_t mc_req, size_t nc_req) {
    const uint32_t ats = htp_mm_get_weight_aligned_tile_size(type);
    plan_t         p;
    p.pipeline = htp_mm_hmx_pipeline(m);
    p.ok = htp_mm_hmx_fit_2d_chunks(type, k, pad32(n), pad32(m), N_THREADS, p.pipeline, ats, VTCM, mc_req, nc_req, &p.mc,
                                    &p.nc, &p.threads, &p.vtcm);
    printf("k %u n %u m %u %s request mc %zu nc %zu: %s\n", k, n, m, type_name(type), mc_req, nc_req,
           p.ok ? text(p, m, n).c_str() : "no layout in the budget, the host keeps the chunks of the solver");
    return p.ok ? 0 : 1;
}

// The checks of one shape at one token count. Returns the count of failed checks.
int check_one(const shape & s, uint32_t m, bool verbose) {
    const plan_t o = solve(s, m, false);
    const plan_t w = solve(s, m, true);
    int          bad = 0;
    auto         fail = [&](const char * what) {
        if (bad == 0 || verbose) {
            printf("FAIL %s %s k %u n %u m %u: %s\n  old: %s\n  new: %s\n", s.name, type_name(s.type), s.k, s.n, m, what,
                   text(o, m, s.n).c_str(), text(w, m, s.n).c_str());
        }
        bad++;
    };
    if (o.ok != w.ok) {
        fail("one model has a layout and the other model has none");
        return bad;
    }
    if (!w.ok) {
        return 0;
    }
    const uint32_t ats  = htp_mm_get_weight_aligned_tile_size(s.type);
    const size_t   size = htp_mm_hmx_get_2d_vtcm_size(s.type, s.k, w.mc, w.nc, w.pipeline, w.threads, ats);
    if (size > VTCM || size != w.vtcm) {
        fail("the layout is larger than the budget, or its VTCM value is not the layout value");
    }
    if (w.mc == 0 || w.nc == 0 || w.mc % 32 || w.nc % 32 || w.mc > pad32(m) || w.nc > pad32(s.n)) {
        fail("a chunk is not a multiple of 32 or is out of range");
    }
    if (w.passes(m) > o.passes(m)) {
        fail("the new model gives more passes than the old model");
    }
    if (o.passes(m) == 1 && (o.mc != w.mc || o.nc != w.nc || o.threads != w.threads || o.pipeline != w.pipeline)) {
        fail("the old model gives one pass, and the new model gives other chunks");
    }
    return bad;
}

int cmd_check() {
    int    bad = 0;
    size_t n_cases = 0, n_changed = 0;
    for (const shape & s : SHAPES) {
        for (uint32_t m = 5; m <= 1024; m++) {
            bad += check_one(s, m, false);
            n_cases++;
            const plan_t o = solve(s, m, false);
            const plan_t w = solve(s, m, true);
            n_changed += (o.mc != w.mc || o.nc != w.nc);
        }
    }
    printf("the 4B shapes: %zu cases (each shape at 5 to 1024 tokens), %zu with other chunks, %d failed checks\n",
           n_cases, n_changed, bad);

    // Random shapes: k and n multiples of 32, each weight type of the HMX path, 5 to 4096 tokens
    std::mt19937 rng(0x6d6d736f);
    const int    types[] = { HTP_TYPE_Q8_0, HTP_TYPE_Q4_0, HTP_TYPE_F16, HTP_TYPE_F32 };
    int          bad_rand = 0;
    const int    n_rand   = 20000;
    for (int i = 0; i < n_rand; i++) {
        shape s = { "random", types[rng() % 4], 32 * (1 + (uint32_t) (rng() % 512)), 32 * (1 + (uint32_t) (rng() % 1024)) };
        bad_rand += check_one(s, 5 + (uint32_t) (rng() % 4092), false);
    }
    printf("random shapes: %d cases, %d failed checks\n", n_rand, bad_rand);
    return (bad || bad_rand) ? 1 : 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (const char * v = getenv("PLAN_VTCM")) {
        VTCM = (size_t) strtoull(v, nullptr, 10);
    }
    if (const char * t = getenv("PLAN_THREADS")) {
        N_THREADS = atoi(t) > 0 ? atoi(t) : N_THREADS;
    }
    if (argc >= 2 && !strcmp(argv[1], "table")) {
        return cmd_table();
    }
    if (argc >= 2 && !strcmp(argv[1], "check")) {
        return cmd_check();
    }
    if (argc >= 2 && !strcmp(argv[1], "sensitivity")) {
        return cmd_sensitivity();
    }
    if (argc == 6 && !strcmp(argv[1], "pair")) {
        const shape  s = { "pair", parse_type(argv[5]), (uint32_t) atoi(argv[2]), (uint32_t) atoi(argv[3]) };
        const uint32_t m = (uint32_t) atoi(argv[4]);
        printf("k %u n %u m %u %s  old: %s  new: %s\n", s.k, s.n, m, argv[5], text(solve(s, m, false), m, s.n).c_str(),
               text(solve(s, m, true), m, s.n).c_str());
        return check_one(s, m, true) ? 1 : 0;
    }
    if (argc == 6 && !strcmp(argv[1], "candidates")) {
        return cmd_candidates((uint32_t) atoi(argv[2]), (uint32_t) atoi(argv[3]), (uint32_t) atoi(argv[4]),
                              parse_type(argv[5]));
    }
    if (argc == 8 && !strcmp(argv[1], "fit")) {
        return cmd_fit((uint32_t) atoi(argv[2]), (uint32_t) atoi(argv[3]), (uint32_t) atoi(argv[4]), parse_type(argv[5]),
                       (size_t) atoi(argv[6]), (size_t) atoi(argv[7]));
    }
    fprintf(stderr,
            "usage: plan table | plan check | plan sensitivity | plan pair K N M TYPE | plan candidates K N M TYPE |\n"
            "       plan fit K N M TYPE MC NC\n"
            "  TYPE is q8_0, f16 or f32. Refer to the comment at the start of tools/stages/mmsolve/plan.cpp.\n");
    return 2;
}
