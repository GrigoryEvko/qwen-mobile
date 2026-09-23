// fuzz_sampler: the sampler chain of the app and the top-set patch (LLAMA_SAMPLER_TOPSET).
//
// The patch hexagon-host/0003-llama-sampler-topset.patch gives
// llama_sampler_topset(): for a chain that cuts the candidates with top-k or
// greedy, it selects the m = k + n_moved highest logits plus the tokens that
// the samplers before the cut move by id, and llama_sampler_sample() then runs
// the chain on that small set in the place of all the logits. The patch
// promises that the chain selects the same token on the two paths.
//
// One input gives:
//   - n_vocab: 2 to 300, 1000 to 5000, or 248320 (the vocabulary of Qwen3.5)
//   - the chain: the chain of the app (llama_jni.cpp rebuild_sampler: penalties
//     with last_n 256 and presence 1.5, then top-k 20, top-p, temperature and
//     dist, or greedy when the temperature is 0), or a random chain of the
//     samplers that the top-set bound knows, with random parameters
//   - a history of accepted tokens, which fills the penalty window
//   - 1 to 3 rounds of logits with NaN, +Inf, -Inf, ties, denormals and huge values
//
// For each round, the harness does what llama_sampler_sample() does: the chain
// runs on the top set, and a clone of the chain runs on all the logits (the
// clone has the same random state). Then the chain accepts the token.
//
// Properties:
//   P1  No crash and no sanitizer report.
//   P2  The top set holds at most n_vocab / 2 entries in token order (each id is
//       larger than the id before it), each in [0, n_vocab), each with the logit
//       of the input.
//   P3  Differential: the two paths select the same token. The check is strict
//       when no logit and no sampler parameter is NaN. With a NaN, the order is
//       not defined on either path, thus the harness counts the differences and
//       FUZZ_SAMPLER_STRICT_NAN=1 makes them failures too.
//   P4  The dist sampler alone, on 2 to 8 candidates from the input: when
//       exactly one candidate has the logit +Inf and no logit is NaN, dist
//       selects that candidate, as greedy does.
//
// The switch FUZZ_SAMPLER_KNOWN_DIST_INF=1 keeps the finding dist-inf off. The
// dist sampler computes exp(l - max) with max = +Inf, or with all logits -Inf,
// or with a NaN logit. The sum is then NaN, no candidate gets selected, and
// dist selects the last candidate (release) or fails assert(found) (debug).
// With the switch, the harness turns P4 off and puts a guard in front of each
// dist sampler of a chain. The guard gives dist the logits of the fix
// (fixes/dist-inf.patch): a NaN logit becomes -Inf, a +Inf logit becomes 0 and
// each finite logit becomes -Inf when a +Inf logit is present, and all -Inf
// logits become 0.

#include "fuzz_common.h"

#include <algorithm>
#include <cfloat>
#include <cinttypes>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int32_t kQwenVocab     = 248320;
constexpr int32_t kPenaltyLastN  = 256;  // the value of the app

bool g_strict_nan = false;
long g_nan_diff   = 0;

// The largest |logit bias| and the smallest positive temperature of the chain of the input. With
// FUZZ_SAMPLER_KNOWN_BUCKET_CAST=1, a round with more than 128 candidates runs only when every logit
// that the bucket sort can see stays finite and below 1e8 in magnitude (finding sampler-bucket-cast:
// the conversion of the bucket index overflows int above 3.3e8, and for an Inf or a NaN).
float g_max_bias = 0.0f;
float g_min_temp = 1.0f;
// The largest factor and the largest shift that a penalties sampler of the chain applies to a logit,
// and a top-n-sigma sampler in the chain (it writes -Inf logits). The switch of the finding reads them.
float g_pen_scale = 1.0f;
float g_pen_shift = 0.0f;
bool  g_sigma     = false;

/** The guard of a dist sampler (finding dist-inf). It owns the dist sampler. */
struct DistGuard {
    llama_sampler * dist;
};

/** The name of the guard: the name of the sampler that it guards. */
const char * guard_name(const llama_sampler * /*smpl*/) {
    return "dist";
}

/** Send the accepted token to the dist sampler. */
void guard_accept(llama_sampler * smpl, llama_token token) {
    llama_sampler_accept(static_cast<DistGuard *>(smpl->ctx)->dist, token);
}

/**
 * Give dist the logits of the fix, then let it select. The logits change in place, because dist is the
 * last sampler of each chain. O(n) in the number of candidates.
 */
void guard_apply(llama_sampler * smpl, llama_token_data_array * cur_p) {
    bool any_nan = false, any_pinf = false, all_ninf = cur_p->size > 0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const float l = cur_p->data[i].logit;
        any_nan  = any_nan || std::isnan(l);
        any_pinf = any_pinf || l == INFINITY;
        all_ninf = all_ninf && l == -INFINITY;
    }
    if (any_nan || any_pinf || all_ninf) {
        bool none_left = true;
        for (size_t i = 0; i < cur_p->size; ++i) {
            float & l = cur_p->data[i].logit;
            if (any_pinf) {
                l = l == INFINITY ? 0.0f : -INFINITY;
            } else if (std::isnan(l)) {
                l = -INFINITY;
            }
            none_left = none_left && l == -INFINITY;
        }
        if (none_left) {
            for (size_t i = 0; i < cur_p->size; ++i) {
                cur_p->data[i].logit = 0.0f;
            }
        }
    }
    llama_sampler_apply(static_cast<DistGuard *>(smpl->ctx)->dist, cur_p);
}

/** Reset the dist sampler. */
void guard_reset(llama_sampler * smpl) {
    llama_sampler_reset(static_cast<DistGuard *>(smpl->ctx)->dist);
}

llama_sampler * guard_init(llama_sampler * dist);

/** Clone the guard with a clone of the dist sampler (the same random state). */
llama_sampler * guard_clone(const llama_sampler * smpl) {
    return guard_init(llama_sampler_clone(static_cast<const DistGuard *>(smpl->ctx)->dist));
}

/** Free the guard and the dist sampler. */
void guard_free(llama_sampler * smpl) {
    DistGuard * g = static_cast<DistGuard *>(smpl->ctx);
    llama_sampler_free(g->dist);
    delete g;
}

llama_sampler_i g_guard_iface = {
    /* .name              = */ guard_name,
    /* .accept            = */ guard_accept,
    /* .apply             = */ guard_apply,
    /* .reset             = */ guard_reset,
    /* .clone             = */ guard_clone,
    /* .free              = */ guard_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

/** A guard that owns the dist sampler dist. */
llama_sampler * guard_init(llama_sampler * dist) {
    return llama_sampler_init(&g_guard_iface, new DistGuard{ dist });
}

/** A dist sampler, with a guard when FUZZ_SAMPLER_KNOWN_DIST_INF=1. */
llama_sampler * make_dist(uint32_t seed) {
    static const bool known_dist_inf = fuzz::env_long("FUZZ_SAMPLER_KNOWN_DIST_INF", 0) != 0;
    llama_sampler * dist = llama_sampler_init_dist(seed);
    return known_dist_inf ? guard_init(dist) : dist;
}

/** A sampler parameter: often a usual value from [lo, hi], sometimes a special float. Sets *nan for a NaN. */
float param(FuzzedDataProvider & fdp, float lo, float hi, bool * nan) {
    float v;
    if (fdp.ConsumeIntegralInRange<int>(0, 5) == 0) {
        v = fuzz::special_float(fdp);
    } else {
        v = lo + (hi - lo) * fdp.ConsumeProbability<float>();
    }
    if (std::isnan(v)) {
        *nan = true;
    }
    return v;
}

/** The chain of the app, as llama_jni.cpp rebuild_sampler() makes it. */
llama_sampler * app_chain(FuzzedDataProvider & fdp, int32_t n_vocab, bool * nan, std::string & desc) {
    const float temp  = fdp.ConsumeBool() ? 0.0f : param(fdp, 0.0f, 2.0f, nan);
    const float top_p = param(fdp, 0.0f, 1.0f, nan);
    const uint32_t seed = fdp.ConsumeIntegral<uint32_t>();

    auto params = llama_sampler_chain_default_params();
    params.no_perf = true;
    llama_sampler * chain = llama_sampler_chain_init(params);
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, kPenaltyLastN, 1.0f, 0.0f, 1.5f));
    g_pen_shift = std::max(g_pen_shift, 1.5f);
    if (temp <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(20));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(std::min(std::max(top_p, 0.05f), 1.0f), 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        // The app seeds with LLAMA_DEFAULT_SEED, which reads the clock. A fixed seed keeps the input reproducible.
        llama_sampler_chain_add(chain, make_dist(seed));
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "app(temp=%g, top_p=%g, seed=%u)", temp, top_p, seed);
    desc = buf;
    return chain;
}

/** A random chain: 1 to 6 samplers from the menu, then a sampler that selects the token. */
llama_sampler * random_chain(FuzzedDataProvider & fdp, int32_t n_vocab, bool * nan, std::string & desc) {
    auto params = llama_sampler_chain_default_params();
    params.no_perf = true;
    llama_sampler * chain = llama_sampler_chain_init(params);
    const int n = fdp.ConsumeIntegralInRange<int>(1, 6);
    char buf[160];
    for (int i = 0; i < n; ++i) {
        llama_sampler * s = nullptr;
        switch (fdp.ConsumeIntegralInRange<int>(0, 10)) {
            case 0: {
                std::vector<llama_logit_bias> lb(fdp.ConsumeIntegralInRange<int>(0, 8));
                for (auto & b : lb) {
                    b.token = fdp.ConsumeIntegralInRange<int32_t>(-2, n_vocab + 2);
                    b.bias  = param(fdp, -20.0f, 20.0f, nan);
                    g_max_bias = std::max(g_max_bias, std::fabs(b.bias));
                }
                s = llama_sampler_init_logit_bias(n_vocab, (int32_t) lb.size(), lb.data());
                snprintf(buf, sizeof(buf), "logit_bias(%zu) ", lb.size());
                break;
            }
            case 1: {
                const int32_t last_n = fdp.ConsumeIntegralInRange<int32_t>(-1, 300);
                const float rep  = param(fdp, 0.5f, 2.0f, nan);
                const float freq = param(fdp, -1.0f, 1.0f, nan);
                const float pres = param(fdp, -2.0f, 2.0f, nan);
                // the penalties divide a positive logit by rep, multiply a negative one by rep, and
                // subtract count * freq + pres (count <= 300). A NaN makes the bound infinite.
                const float scale = rep > 0.0f ? std::max(rep, 1.0f / rep) : INFINITY;
                const float shift = std::fabs(freq) * 300.0f + std::fabs(pres);
                g_pen_scale = std::max(g_pen_scale, scale <= FLT_MAX ? scale : INFINITY);
                g_pen_shift = std::max(g_pen_shift, shift <= FLT_MAX ? shift : INFINITY);
                s = llama_sampler_init_penalties(n_vocab, last_n, rep, freq, pres);
                snprintf(buf, sizeof(buf), "penalties(%d, %g, %g, %g) ", last_n, rep, freq, pres);
                break;
            }
            case 2: case 3: {
                const int32_t k = fdp.ConsumeIntegralInRange<int32_t>(-1, 300);
                s = llama_sampler_init_top_k(k);
                snprintf(buf, sizeof(buf), "top_k(%d) ", k);
                break;
            }
            case 4: {
                const float t = param(fdp, 0.0f, 3.0f, nan);
                if (t > 0.0f) {
                    g_min_temp = std::min(g_min_temp, t);
                }
                s = llama_sampler_init_temp(t);
                snprintf(buf, sizeof(buf), "temp(%g) ", t);
                break;
            }
            case 5: {
                const float t = param(fdp, 0.0f, 3.0f, nan), d = param(fdp, -1.0f, 1.0f, nan), e = param(fdp, 0.0f, 2.0f, nan);
                if (t > 0.0f) {
                    // with delta > 0 the dynamic temperature goes down to max(0, t - delta)
                    g_min_temp = std::min(g_min_temp, d > 0.0f ? std::max(0.0f, t - d) : t);
                }
                s = llama_sampler_init_temp_ext(t, d, e);
                snprintf(buf, sizeof(buf), "temp_ext(%g, %g, %g) ", t, d, e);
                break;
            }
            case 6: {
                const float p = param(fdp, 0.0f, 1.0f, nan);
                const size_t mk = fdp.ConsumeIntegralInRange<size_t>(0, 4);
                s = llama_sampler_init_top_p(p, mk);
                snprintf(buf, sizeof(buf), "top_p(%g, %zu) ", p, mk);
                break;
            }
            case 7: {
                const float p = param(fdp, 0.0f, 1.0f, nan);
                const size_t mk = fdp.ConsumeIntegralInRange<size_t>(0, 4);
                s = llama_sampler_init_min_p(p, mk);
                snprintf(buf, sizeof(buf), "min_p(%g, %zu) ", p, mk);
                break;
            }
            case 8: {
                const float p = param(fdp, 0.0f, 1.0f, nan);
                s = llama_sampler_init_typical(p, 1);
                snprintf(buf, sizeof(buf), "typical(%g) ", p);
                break;
            }
            case 9: {
                const float sn = param(fdp, -1.0f, 3.0f, nan);
                g_sigma = g_sigma || !(sn <= 0.0f);
                s = llama_sampler_init_top_n_sigma(sn);
                snprintf(buf, sizeof(buf), "top_n_sigma(%g) ", sn);
                break;
            }
            default: {
                const float p = param(fdp, 0.0f, 1.0f, nan), t = param(fdp, 0.0f, 0.5f, nan);
                const uint32_t seed = fdp.ConsumeIntegral<uint32_t>();
                s = llama_sampler_init_xtc(p, t, 1, seed);
                snprintf(buf, sizeof(buf), "xtc(%g, %g, seed=%u) ", p, t, seed);
                break;
            }
        }
        llama_sampler_chain_add(chain, s);
        desc += buf;
    }
    switch (fdp.ConsumeIntegralInRange<int>(0, 3)) {
        case 0:  llama_sampler_chain_add(chain, llama_sampler_init_greedy()); desc += "greedy"; break;
        case 1: {
            const float tau = param(fdp, 0.0f, 10.0f, nan), eta = param(fdp, 0.0f, 1.0f, nan);
            llama_sampler_chain_add(chain, llama_sampler_init_mirostat_v2(fdp.ConsumeIntegral<uint32_t>(), tau, eta));
            desc += "mirostat_v2";
            break;
        }
        default: {
            const uint32_t seed = fdp.ConsumeIntegral<uint32_t>();
            llama_sampler_chain_add(chain, make_dist(seed));
            snprintf(buf, sizeof(buf), "dist(%u)", seed);
            desc += buf;
            break;
        }
    }
    return chain;
}

/** Fill logits for one round. The mode byte selects the shape. Returns true when a logit is NaN. */
bool make_logits(FuzzedDataProvider & fdp, std::mt19937 & rng, std::vector<float> & logits) {
    const int32_t n = (int32_t) logits.size();
    const int mode = fdp.ConsumeIntegralInRange<int>(0, 3);
    if (mode == 0 && n <= 4096) {
        for (auto & l : logits) {
            l = fdp.remaining_bytes() > 0 ? fuzz::special_float(fdp) : 0.0f;
        }
    } else if (mode <= 1) {
        static const float kSigma[] = { 1e-3f, 1.0f, 8.0f, 1e6f, 1e30f };
        std::normal_distribution<float> nd(0.0f, kSigma[fdp.ConsumeIntegralInRange<int>(0, 4)]);
        for (auto & l : logits) {
            l = nd(rng);
        }
    } else {
        const int levels = fdp.ConsumeIntegralInRange<int>(1, 8);
        for (auto & l : logits) {
            l = (float) ((int) (rng() % levels) - levels / 2);
        }
    }
    const int n_special = fdp.ConsumeIntegralInRange<int>(0, 16);
    for (int i = 0; i < n_special; ++i) {
        logits[fdp.ConsumeIntegralInRange<int32_t>(0, n - 1)] = fuzz::special_float(fdp);
    }
    for (const float l : logits) {
        if (std::isnan(l)) {
            return true;
        }
    }
    return false;
}

/** Check P2 on the top set. */
void check_topset(const std::vector<llama_token_data> & top, size_t n_top, const std::vector<float> & logits) {
    const int32_t n_vocab = (int32_t) logits.size();
    if (n_top > (size_t) n_vocab / 2) {
        fuzz::fail("P2: the top set holds %zu entries for n_vocab %d", n_top, n_vocab);
    }
    for (size_t i = 0; i < n_top; ++i) {
        const llama_token id = top[i].id;
        if (id < 0 || id >= n_vocab) {
            fuzz::fail("P2: top set entry %zu has the id %d", i, id);
        }
        if (i > 0 && id <= top[i - 1].id) {
            fuzz::fail("P2: the top set is not in token order at entry %zu (%d after %d)", i, id, top[i - 1].id);
        }
        if (memcmp(&top[i].logit, &logits[id], sizeof(float)) != 0) {
            fuzz::fail("P2: top set entry %zu (id %d) has the logit %g, the input has %g", i, id, top[i].logit, logits[id]);
        }
    }
}

/** The selected token of an applied array, or LLAMA_TOKEN_NULL when the chain selected nothing. */
llama_token selected(const llama_token_data_array & a) {
    return (a.selected >= 0 && (size_t) a.selected < a.size) ? a.data[a.selected].id : LLAMA_TOKEN_NULL;
}

/** P4: dist on a few candidates. One +Inf logit and no NaN must select the +Inf candidate. */
void check_dist(FuzzedDataProvider & fdp) {
    const int n = fdp.ConsumeIntegralInRange<int>(2, 8);
    std::vector<llama_token_data> cand(n);
    int n_inf = 0, n_nan = 0, id_inf = -1;
    for (int i = 0; i < n; ++i) {
        const float l = fuzz::special_float(fdp);
        cand[i] = llama_token_data{ i, l, 0.0f };
        n_nan += std::isnan(l) ? 1 : 0;
        if (l == INFINITY) {
            n_inf++;
            id_inf = i;
        }
    }
    llama_sampler * dist = llama_sampler_init_dist(fdp.ConsumeIntegral<uint32_t>());
    llama_token_data_array arr = { cand.data(), cand.size(), -1, fdp.ConsumeBool() };
    llama_sampler_apply(dist, &arr);
    llama_sampler_free(dist);
    if (arr.selected < 0 || arr.selected >= n) {
        fuzz::fail("P4: dist selects the index %lld of %d candidates", (long long) arr.selected, n);
    }
    if (n_inf == 1 && n_nan == 0 && arr.data[arr.selected].id != id_inf) {
        fuzz::fail("P4: candidate %d has the logit +Inf, but dist selects candidate %d (logit %g) of %d",
                   id_inf, arr.data[arr.selected].id, arr.data[arr.selected].logit, n);
    }
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int * /*argc*/, char *** /*argv*/) {
    fuzz::quiet_logs();
    g_strict_nan = fuzz::env_long("FUZZ_SAMPLER_STRICT_NAN", 0) != 0;
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz::note_input(data, size);
    FuzzedDataProvider fdp(data, size);

    static const bool known_dist_inf = fuzz::env_long("FUZZ_SAMPLER_KNOWN_DIST_INF", 0) != 0;
    if (!known_dist_inf) {
        check_dist(fdp);
    }

    int32_t n_vocab;
    switch (fdp.ConsumeIntegralInRange<int>(0, 7)) {
        case 0:  n_vocab = kQwenVocab; break;
        case 1:
        case 2:  n_vocab = fdp.ConsumeIntegralInRange<int32_t>(1000, 5000); break;
        default: n_vocab = fdp.ConsumeIntegralInRange<int32_t>(2, 300); break;
    }

    bool nan_param = false;
    std::string desc;
    g_max_bias  = 0.0f;
    g_min_temp  = 1.0f;
    g_pen_scale = 1.0f;
    g_pen_shift = 0.0f;
    g_sigma     = false;
    static const bool known_bucket = fuzz::env_long("FUZZ_SAMPLER_KNOWN_BUCKET_CAST", 0) != 0;
    llama_sampler * chain = fdp.ConsumeBool() ? app_chain(fdp, n_vocab, &nan_param, desc)
                                              : random_chain(fdp, n_vocab, &nan_param, desc);

    std::mt19937 rng(fdp.ConsumeIntegral<uint32_t>());

    // the history: mostly a few repeated ids, thus the penalty window holds repeats
    const int n_hist = fdp.ConsumeIntegralInRange<int>(0, 300);
    const int32_t hist_range = std::min<int32_t>(n_vocab, fdp.ConsumeIntegralInRange<int32_t>(1, 64));
    for (int i = 0; i < n_hist; ++i) {
        llama_sampler_accept(chain, (llama_token) (rng() % hist_range));
    }

    std::vector<float> logits(n_vocab);
    std::vector<llama_token_data> top(n_vocab), part, full(n_vocab);
    const int rounds = fdp.ConsumeIntegralInRange<int>(1, 3);
    for (int r = 0; r < rounds; ++r) {
        const bool nan_logit = make_logits(fdp, rng, logits);
        const bool strict = g_strict_nan || (!nan_logit && !nan_param);
        if (known_bucket && n_vocab > 128) {
            // Only an array of more than 128 candidates goes to the bucket sort. There a logit must
            // stay finite and below 1e8 in magnitude after the samplers before the sort (the
            // conversion overflows above 3.3e8). Else the switch skips the round.
            bool finite = true;
            float max_abs = 0.0f;
            for (const float l : logits) {
                finite = finite && std::isfinite(l);
                max_abs = std::max(max_abs, std::fabs(l));
            }
            const float bound = (max_abs * g_pen_scale + g_pen_shift + g_max_bias + 8.0f) / std::min(1.0f, g_min_temp);
            if (!finite || nan_param || g_sigma || !(bound < 1e8f)) {
                continue;
            }
        }

        const size_t n_top = llama_sampler_topset(chain, logits.data(), n_vocab, top.data());
        check_topset(top, n_top, logits);

        // all the logits, on a clone with the same state
        llama_sampler * clone = llama_sampler_clone(chain);
        for (llama_token id = 0; id < n_vocab; ++id) {
            full[id] = llama_token_data{ id, logits[id], 0.0f };
        }
        llama_token_data_array full_p = { full.data(), full.size(), -1, false };
        llama_sampler_apply(clone, &full_p);
        const llama_token id_full = selected(full_p);
        llama_sampler_free(clone);

        // the path of llama_sampler_sample: the chain itself on the top set, or on all the logits
        llama_token id_chain;
        if (n_top > 0) {
            part.assign(top.begin(), top.begin() + n_top);
            llama_token_data_array part_p = { part.data(), part.size(), -1, false };
            llama_sampler_apply(chain, &part_p);
            id_chain = selected(part_p);
            if (id_chain != id_full) {
                if (strict) {
                    // the first candidates of the two arrays after the chain, for the report
                    for (int side = 0; side < 2; ++side) {
                        const llama_token_data_array & a = side == 0 ? part_p : full_p;
                        fprintf(stderr, "%s: size %zu sorted %d selected %lld:", side == 0 ? "top set" : "all    ",
                                a.size, (int) a.sorted, (long long) a.selected);
                        for (size_t j = 0; j < a.size && j < 8; ++j) {
                            fprintf(stderr, " (%d %.9g %.6g)", a.data[j].id, a.data[j].logit, a.data[j].p);
                        }
                        fputc('\n', stderr);
                    }
                    fuzz::fail("P3: the top set of %zu selects token %d, all %d logits select token %d. chain: %s, round %d",
                               n_top, id_chain, n_vocab, id_full, desc.c_str(), r);
                }
                if (++g_nan_diff == 1) {
                    fprintf(stderr, "note: with a NaN, the top set of %zu selects token %d, all the logits select token %d (%s)\n",
                            n_top, id_chain, id_full, desc.c_str());
                }
            }
        } else {
            for (llama_token id = 0; id < n_vocab; ++id) {
                full[id] = llama_token_data{ id, logits[id], 0.0f };
            }
            llama_token_data_array all_p = { full.data(), full.size(), -1, false };
            llama_sampler_apply(chain, &all_p);
            id_chain = selected(all_p);
        }
        if (id_chain != LLAMA_TOKEN_NULL) {
            llama_sampler_accept(chain, id_chain);
        }
    }

    llama_sampler_free(chain);
    return 0;
}
