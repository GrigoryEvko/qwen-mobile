// The comparison with the oracle. Refer to compare.h.

#include "compare.h"

#include <cmath>
#include <cstdio>
#include <limits>

namespace fo {

namespace {

constexpr float FLT_MIN_NORMAL = 1.17549435e-38f;

// Return the spacing of f32 values at |x|.
double ulp_of(double x) {
    const double a = std::fabs(x);
    if (!(a >= FLT_MIN_NORMAL)) {
        return 1.40129846e-45;
    }
    int e = 0;
    std::frexp(a, &e);
    return std::ldexp(1.0, e - 24);
}

} // namespace

const char * verdict_name(int v) {
    switch (v) {
        case V_PASS:      return "pass";
        case V_SUBNORMAL: return "subnormal";
        case V_STRICT:    return "above-strict";
        case V_LOOSE:     return "above-loose";
        case V_NONFINITE: return "nonfinite";
    }
    return "?";
}

const char * verdict_severity(int v, bool special) {
    switch (v) {
        case V_PASS:      return "none";
        case V_SUBNORMAL: return "low";
        case V_STRICT:    return "medium";
        case V_LOOSE:     return "high";
        case V_NONFINITE: return special ? "medium" : "high";
    }
    return "?";
}

case_cmp compare_case(const built_case & c, const std::vector<std::vector<uint8_t>> & got,
                      const std::vector<std::vector<uint8_t>> & ref, bool fast) {
    case_cmp cc;
    cc.outs.resize(c.outs.size());
    for (size_t o = 0; o < c.outs.size(); o++) {
        const ggml_tensor * t = c.outs[o].t;
        const int64_t       n = ggml_nelements(t);
        out_cmp &           oc = cc.outs[o];
        if (o >= got.size() || o >= ref.size() || got[o].size() != ggml_nbytes(t) || ref[o].size() != ggml_nbytes(t)) {
            oc.n_nonfinite = 1;
            cc.text += c.outs[o].label + ": size mismatch; ";
            continue;
        }
        const std::vector<float> rf = raw_to_f32(t->type, ref[o].data(), n);
        const std::vector<float> gf = raw_to_f32(t->type, got[o].data(), n);
        std::vector<char>        skip((size_t) n, 0);
        for (const auto & s : c.outs[o].skip) {
            for (int64_t i = std::max<int64_t>(0, s.first); i < std::min(n, s.second); i++) {
                skip[(size_t) i] = 1;
            }
        }
        bound_arrays ba;
        c.kind->bound(c, o, rf, ba);
        ba.strict.resize((size_t) n, std::numeric_limits<double>::infinity());
        ba.loose.resize((size_t) n, std::numeric_limits<double>::infinity());
        if (fast) {
            // the slack of a fast-math build (refer to compare.h)
            for (int64_t i = 0; i < n; i++) {
                const double r = std::isfinite(rf[(size_t) i]) ? std::fabs((double) rf[(size_t) i]) : 0.0;
                ba.strict[(size_t) i] = 2.0 * ba.strict[(size_t) i] + 4.0 * 5.9604644775390625e-08 * r;
                ba.loose[(size_t) i]  = std::max(ba.loose[(size_t) i], ba.strict[(size_t) i]);
            }
        }
        for (int64_t i = 0; i < n; i++) {
            if (skip[(size_t) i]) {
                continue;
            }
            oc.n_cmp++;
            const float r = rf[(size_t) i];
            const float g = gf[(size_t) i];
            if (!std::isfinite(r) || !std::isfinite(g)) {
                const bool same = (std::isnan(r) && std::isnan(g)) || r == g;
                if (!same) {
                    if (oc.nf_i < 0) {
                        oc.nf_i   = i;
                        oc.nf_ref = r;
                        oc.nf_got = g;
                    }
                    oc.n_nonfinite++;
                    if (std::isfinite(r) && std::fabs(r) > 65504.0f) {
                        oc.n_nf_f16++;
                    }
                }
                continue;
            }
            oc.max_ref_fin = std::max(oc.max_ref_fin, (double) std::fabs(r));
            const double err = std::fabs((double) g - (double) r);
            if (err == 0.0) {
                continue;
            }
            if (std::fabs(r) < FLT_MIN_NORMAL && std::fabs(g) < FLT_MIN_NORMAL) {
                oc.n_subnormal++;
                continue;
            }
            const double s = ba.strict[(size_t) i];
            const double l = ba.loose[(size_t) i];
            if (std::isnan(s) || std::isinf(s)) {
                continue;
            }
            const double u     = ulp_of(r);
            const double ratio = err / std::max(s, 0.5 * u);
            oc.max_err     = std::max(oc.max_err, err);
            oc.max_ulp     = std::max(oc.max_ulp, err / u);
            oc.max_ratio_l = std::max(oc.max_ratio_l, err / std::max(l, 0.5 * u));
            if (err > s) {
                oc.n_strict++;
            }
            if (err > l) {
                oc.n_loose++;
            }
            if (ratio > oc.max_ratio) {
                oc.max_ratio    = ratio;
                oc.worst_i      = i;
                oc.worst_ref    = r;
                oc.worst_got    = g;
                oc.worst_strict = s;
                oc.worst_loose  = l;
            }
        }
        int v = V_PASS;
        if (oc.n_nonfinite > 0) {
            v = V_NONFINITE;
        } else if (oc.n_loose > 0) {
            v = V_LOOSE;
        } else if (oc.n_strict > 0) {
            v = V_STRICT;
        } else if (oc.n_subnormal > 0) {
            v = V_SUBNORMAL;
        }
        cc.v     = (verdict) std::max((int) cc.v, v);
        cc.ratio = std::max(cc.ratio, oc.max_ratio);
        cc.ulp   = std::max(cc.ulp, oc.max_ulp);
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "%s: n=%lld strict=%lld loose=%lld nonfinite=%lld subnormal=%lld max_err=%.3g ratio=%.3g "
                      "ratio_loose=%.3g ulp=%.3g",
                      c.outs[o].label.c_str(), (long long) oc.n_cmp, (long long) oc.n_strict, (long long) oc.n_loose,
                      (long long) oc.n_nonfinite, (long long) oc.n_subnormal, oc.max_err, oc.max_ratio, oc.max_ratio_l,
                      oc.max_ulp);
        cc.text += buf;
        if (oc.worst_i >= 0) {
            std::snprintf(buf, sizeof(buf), " worst[%lld] ref=%.9g got=%.9g strict=%.3g loose=%.3g",
                          (long long) oc.worst_i, (double) oc.worst_ref, (double) oc.worst_got, oc.worst_strict,
                          oc.worst_loose);
            cc.text += buf;
        }
        if (oc.nf_i >= 0) {
            std::snprintf(buf, sizeof(buf),
                          " nonfinite[%lld] ref=%g got=%g; nonfinite with |ref| > 65504: %lld of %lld; largest |ref| "
                          "with a finite result %.6g",
                          (long long) oc.nf_i, (double) oc.nf_ref, (double) oc.nf_got, (long long) oc.n_nf_f16,
                          (long long) oc.n_nonfinite, oc.max_ref_fin);
            cc.text += buf;
        }
        cc.text += "; ";
    }
    return cc;
}

} // namespace fo
