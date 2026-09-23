// The kinds of cases: one builder and one bound rule for each op or op chain of the Qwen3.5 path.
//
// A builder reads its shapes and parameters from the case bytes. With about 40 % probability it
// uses the shapes of Qwen3.5 2B and 4B (hidden 2048 and 2560, FFN 6144 and 9216, 8 and 16 query
// heads of 256, 2 and 4 KV heads, gated delta net heads of 128, 16 and 32 value heads, conv width
// 4, rotary dims 64 in the sections 11, 11, 10). Otherwise it uses random shapes: odd sizes, a
// row length that is not a multiple of 32 or 128, views, strides and broadcast.
//
// A bound rule gives two per-element bounds of |backend - oracle| (refer to case.h). The oracle is
// the scalar CPU backend of ggml with strict IEEE arithmetic. The rules use the unit roundoff u =
// 2^-24 of f32, u16 = 2^-11 of f16, and gamma(n) = n u / (1 - n u), the worst case of a sum of n
// terms in f32 in any order. Each rule holds its reason in its text.

#include "case.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>

namespace fo {

namespace {

// The multiply-add budget of one case, and the element budget of one tensor. They are constants,
// because the decode must give the same case on every platform.
constexpr int64_t MAX_MACS  = int64_t(1) << 24;
constexpr int64_t MAX_ELEMS = int64_t(1) << 22;

// Return gamma(n), the relative bound of a sum of n terms in f32.
double gam(double n) {
    const double x = n * EPS32;
    return x < 0.5 ? x / (1.0 - x) : std::numeric_limits<double>::infinity();
}

// Return the printf text.
std::string fmt(const char * f, ...) {
    char    buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

// Return the bucket name of a row count: 1 is decode, 2 to 4 is speculative verify, more is prefill.
const char * nbucket(int64_t n) {
    return n == 1 ? "n1" : n <= 4 ? "n2-4" : n <= 32 ? "n5-32" : "n33+";
}

// Return the row index of a contiguous 4D shape.
inline int64_t idx4(const int64_t * ne, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    return i0 + ne[0] * (i1 + ne[1] * (i2 + ne[2] * i3));
}

// Set every bound of an output to the same values.
void fill_bounds(bound_arrays & ba, size_t n, double s, double l) {
    ba.strict.assign(n, s);
    ba.loose.assign(n, l);
}

// Round a value to f16 and back.
double r16(double x) {
    return f16_to_f32(f32_to_f16((float) x));
}

// Return the max of |x| over a vector.
double amax(const std::vector<double> & v) {
    double m = 0.0;
    for (double x : v) {
        m = std::max(m, std::fabs(x));
    }
    return m;
}

// The f16 value of |x| >= 65520 is inf.
constexpr double F16_INF_AT = 65520.0;

// Return the smallest |x| of an f32 input that the op converts to the type `to` with an infinite
// result: F16_INF_AT for f16 (the value itself), 127 F16_INF_AT for Q8_0 (the f16 scale amax/127 of
// its 32-block), and infinity for the other types (no conversion to a narrow type).
double conv_limit(ggml_type to) {
    if (to == GGML_TYPE_F16) {
        return F16_INF_AT;
    }
    if (to == GGML_TYPE_Q8_0) {
        return 127.0 * F16_INF_AT;
    }
    return std::numeric_limits<double>::infinity();
}

// Mark the case special if an f32 input has a value that its conversion to the type `to` makes
// infinite. The op then gives inf or NaN, and which one depends on the order of the operations of
// each backend (for example 0 * inf in one order and a sum of inf in the other), thus the oracle is
// not a reference for this output. Complexity: O(n) in the elements of x.
void mark_conv_range(builder & b, const ggml_tensor * x, ggml_type to) {
    const double lim = conv_limit(to);
    if (std::isfinite(lim) && amax(logical_values(b.c, x)) >= lim) {
        b.c.special = true;
    }
}

// Return the type that the CPU converts the f32 activations to for a weight type (vec_dot_type):
// F16 for an F16 weight, Q8_0 for a Q8_0 or Q4_0 weight, and F32 (no conversion) for an F32 weight.
ggml_type act_type(ggml_type wt) {
    if (wt == GGML_TYPE_F16) {
        return GGML_TYPE_F16;
    }
    return ggml_is_quantized(wt) ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
}

// Pick a quantized or float weight type.
ggml_type pick_wtype(reader & rd) {
    static const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 };
    return rd.pick(types);
}

// ---------------------------------------------------------------------------------------------
// MUL_MAT and its fused forms

// The shared shape choice of the matrix kinds: k (the row length), m (the output rows) and n (the
// activation rows). The product of the three stays inside MAX_MACS.
void mm_shape(builder & b, ggml_type wt, int64_t & k, int64_t & m, int64_t & n, bool & model) {
    reader &   rd = b.rd;
    const bool q  = ggml_is_quantized(wt);
    model         = rd.chance(100);
    if (model) {
        static const int64_t ks[] = { 2048, 2560, 6144, 9216, 4096, 8192, 256, 128 };
        static const int64_t ms[] = { 16, 32, 64, 256, 512, 2048, 48, 96 };
        k = rd.pick(ks);
        m = rd.pick(ms);
    } else {
        k = q ? 32 * rd.range(1, 40) : rd.range(1, 320);
        m = rd.range(1, 80);
    }
    const uint8_t nb = rd.u8();
    n = nb < 100 ? 1 : nb < 170 ? rd.range(2, 4) : nb < 215 ? rd.range(5, 32) : nb < 245 ? rd.range(33, 128)
                                                                                          : rd.range(129, 300);
    while (m > 1 && (k * m * n > MAX_MACS || k * m > MAX_ELEMS)) {
        m = (m + 1) / 2;
    }
    while (n > 1 && k * m * n > MAX_MACS) {
        n = (n + 1) / 2;
    }
}

// Make the weight [k, m, ne2, ne3] of the type, as a leaf or as a view of a wider leaf.
ggml_tensor * mm_weight(builder & b, ggml_type wt, int64_t k, int64_t m, int64_t ne2, int64_t ne3, bool allow_view) {
    const bool q   = ggml_is_quantized(wt);
    const bool pad = allow_view && !q && b.rd.chance(50);
    const vspec v  = b.vs(-0.125f, 0.125f);
    if (!pad) {
        return b.typed(wt, k, m, ne2, ne3, v, leaf_role::WEIGHT);
    }
    const int64_t kp = k + b.rd.range(1, 40);
    ggml_tensor * w  = b.typed(wt, kp, m, ne2, ne3, v, leaf_role::WEIGHT);
    return ggml_view_4d(b.ctx, w, k, m, ne2, ne3, w->nb[1], w->nb[2], w->nb[3], 0);
}

// Make the activations [k, n, ne2, ne3], as a leaf, a view with padded rows, or a permuted leaf.
ggml_tensor * mm_act(builder & b, int64_t k, int64_t n, int64_t ne2, int64_t ne3, bool ties) {
    reader &    rd   = b.rd;
    const vspec v    = b.vs(-4.0f, 4.0f, ties);
    const uint8_t f  = rd.u8();
    if (f < 40) {
        const int64_t kp = k + rd.range(1, 64);
        ggml_tensor * x  = b.f32(kp, n, ne2, ne3, v);
        return ggml_view_4d(b.ctx, x, k, n, ne2, ne3, x->nb[1], x->nb[2], x->nb[3], 0);
    }
    if (f < 60 && ne2 > 1) {
        ggml_tensor * x = b.f32(k, ne2, n, ne3, v);
        return ggml_permute(b.ctx, x, 0, 2, 1, 3);
    }
    return b.f32(k, n, ne2, ne3, v);
}

// Fill S (the sum of |w| |x| of each output) and Q (the sum of |w| times one Q8_0 step of the block
// of x) for dst = W x. The time is O(k m n ne2 ne3).
void mm_mass(const built_case & c, const ggml_tensor * w, const ggml_tensor * x, std::vector<double> & S,
             std::vector<double> & Q) {
    const std::vector<double> W = logical_values(c, w);
    std::vector<double>       X = logical_values(c, x);
    const int64_t k = w->ne[0], m = w->ne[1];
    const int64_t n = x->ne[1], ne2 = x->ne[2], ne3 = x->ne[3];
    const int64_t r2 = ne2 / w->ne[2], r3 = ne3 / w->ne[3];
    S.assign((size_t) (m * n * ne2 * ne3), 0.0);
    Q.assign(S.size(), 0.0);
    if (W.empty() || X.empty()) {
        return;
    }
    const bool q = ggml_is_quantized(w->type);
    if (w->type == GGML_TYPE_F16) {
        // the CPU converts the activations to f16 for an F16 weight, and so does the oracle
        for (double & v : X) {
            v = r16(v);
        }
    }
    // one Q8_0 step (amax / 127) of the 32-block of each activation
    std::vector<double> step(X.size(), 0.0);
    if (q) {
        for (size_t row = 0; row < X.size() / (size_t) k; row++) {
            for (int64_t b0 = 0; b0 < k; b0 += 32) {
                double a = 0.0;
                for (int64_t j = b0; j < std::min(k, b0 + 32); j++) {
                    a = std::max(a, std::fabs(X[row * k + j]));
                }
                for (int64_t j = b0; j < std::min(k, b0 + 32); j++) {
                    step[row * k + j] = a / 127.0;
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t j = 0; j < n; j++) {
                const double * xr = &X[(size_t) (((i3 * ne2 + i2) * n + j) * k)];
                const double * sr = &step[(size_t) (((i3 * ne2 + i2) * n + j) * k)];
                for (int64_t i = 0; i < m; i++) {
                    const double * wr = &W[(size_t) ((((i3 / r3) * w->ne[2] + i2 / r2) * m + i) * k)];
                    double s = 0.0, qs = 0.0;
                    for (int64_t l = 0; l < k; l++) {
                        s  += std::fabs(wr[l] * xr[l]);
                        qs += std::fabs(wr[l]) * sr[l];
                    }
                    S[(size_t) (((i3 * ne2 + i2) * n + j) * m + i)] = s;
                    Q[(size_t) (((i3 * ne2 + i2) * n + j) * m + i)] = qs;
                }
            }
        }
    }
}

// Return the spacing of f32 values at |x|: 2^(e - 23) for a normal x, 2^-149 below the normals.
double ulp32(double x) {
    const double a = std::fabs(x);
    if (!(a >= 1.17549435e-38)) {
        return 1.40129846e-45;
    }
    int e = 0;
    std::frexp(a, &e);
    return std::ldexp(1.0, e - 24);
}

// The bounds of dst = W x from the masses. `extra` is an added mass (a bias). `tight` selects the
// rule of a Q8_0 or Q4_0 weight at 1 to 4 activation rows: the reference quantizer of the
// activations gives the same int8 values and the block sums are exact integers, thus only the f32
// epilogue (the scale product of each block and the sequential sum) can differ, and a backend that
// rounds it RNE in the block order of the oracle gives the same bits. Its strict bound is 4 ulp(y).
void mm_bounds(ggml_type wt, int64_t k, const std::vector<double> & S, const std::vector<double> & Q,
               const std::vector<double> * extra, bool tight, const std::vector<float> & ref, bound_arrays & ba) {
    const size_t n = ref.size();
    ba.strict.resize(n);
    ba.loose.resize(n);
    const bool   q  = ggml_is_quantized(wt);
    const double nt = q ? (double) (k / 32) + 2.0 : (double) k + 2.0;
    for (size_t i = 0; i < n; i++) {
        const double s = i < S.size() ? S[i] : 0.0;
        const double e = extra ? (*extra)[i] : 0.0;
        const double a = std::fabs((double) ref[i]);
        if (q) {
            ba.strict[i] = tight ? 4.0 * ulp32(a) : 2.0 * gam(nt + 1) * (s + Q[i] + e) + 2.0 * EPS32 * a;
            ba.loose[i]  = Q[i] + 2.0 * (gam(k + 3) + 2.0 * EPS16) * (s + Q[i] + e) + 2.0 * EPS32 * a;
        } else {
            ba.strict[i] = 2.0 * gam(nt + 1) * (s + e) + 2.0 * EPS32 * a;
            ba.loose[i]  = 2.0 * (gam(nt + 1) + 2.0 * EPS16) * (s + e) + 2.0 * EPS32 * a;
        }
    }
}

bool build_mul_mat(builder & b) {
    reader &        rd = b.rd;
    const ggml_type wt = pick_wtype(rd);
    const bool      q  = ggml_is_quantized(wt);
    int64_t         k, m, n;
    bool            model;
    mm_shape(b, wt, k, m, n, model);
    int64_t ne02 = 1, ne03 = 1, r2 = 1, r3 = 1;
    if (!q && rd.chance(40)) {
        ne02 = rd.range(1, 3);
        ne03 = rd.range(1, 2);
        r2   = rd.range(1, 2);
        r3   = rd.range(1, 2);
        while (n > 1 && k * m * n * ne02 * ne03 * r2 * r3 > MAX_MACS) {
            n = (n + 1) / 2;
        }
        if (k * m * n * ne02 * ne03 * r2 * r3 > MAX_MACS) {
            return false;
        }
    }
    ggml_tensor * w = mm_weight(b, wt, k, m, ne02, ne03, true);
    ggml_tensor * x = mm_act(b, k, n, ne02 * r2, ne03 * r3, q);
    mark_conv_range(b, x, act_type(wt));
    ggml_tensor * y = ggml_mul_mat(b.ctx, w, x);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.desc = fmt("MUL_MAT w=%s[%lld,%lld,%lld,%lld]%s x=f32[%lld,%lld,%lld,%lld]%s%s", ggml_type_name(wt),
                   (long long) k, (long long) m, (long long) ne02, (long long) ne03, w->view_src ? " view" : "",
                   (long long) k, (long long) n, (long long) (ne02 * r2), (long long) (ne03 * r3),
                   x->view_src ? (x->op == GGML_OP_PERMUTE ? " permuted" : " view") : "", model ? " model" : "");
    b.c.path = fmt("%s/%s", ggml_type_name(wt), nbucket(n));
    return true;
}

void bound_mul_mat(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * y = c.outs[o].t;
    std::vector<double> S, Q;
    mm_mass(c, y->src[0], y->src[1], S, Q);
    const bool tight = ggml_is_quantized(y->src[0]->type) && y->src[1]->ne[1] <= 4;
    mm_bounds(y->src[0]->type, y->src[0]->ne[0], S, Q, nullptr, tight, ref, ba);
}

const char * TXT_MUL_MAT =
    "Q8_0 and Q4_0 weights at n <= 4 activation rows: strict = 4 ulp(y). Otherwise "
    "strict = 2 gamma(t+1) (S + Q) + 2u|y|. Always loose = Q + 2 (gamma(k+3) + 2 u16) (S + Q) + 2u|y|. "
    "S = sum |w||x| (x rounded to f16 for an F16 weight, as the CPU does), Q = sum |w| amax32(x)/127 "
    "(one Q8_0 step, zero for float weights), t = k for float weights and k/32 + 2 for Q8_0 and Q4_0. "
    "Reason: two f32 sums of the same exact terms in any order differ by at most 2 gamma(t) S. The "
    "products of f16 or int8 values are exact in f32, and the Q8_0 path sums whole blocks as integers, "
    "thus only the block terms round. The loose bound adds one Q8_0 step of each activation (another "
    "rounding of the activations) and one f16 rounding of each input (the HMX path). The 4 ulp rule of "
    "the decode rows: the reference quantizer gives the same int8 values, the block sums are exact "
    "integers, and an RNE epilogue in the block order of the oracle gives the same bits.";

// The exact matrix shapes of the Qwen3.5 2B and 4B models at 1 to 8 activation rows (n = 4 is the
// verify step of the speculative decode). Byte 10 of the input selects the entry: the weight type
// (Q8_0, Q4_0), the (k, m) pair and n. The group "shapes" is not a fuzz group: gen --enumerate makes
// its cases for the phone runs (the check that HTP0 gives the same output each time). The shapes are larger than the
// limits of the fuzz kinds, thus the oracle takes approximately 1 s for each case.
bool build_mm_model(builder & b) {
    static const ggml_type types[] = { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 };
    static const int64_t   km[][2] = {
        { 2560, 9216 }, { 9216, 2560 }, { 2560, 2560 }, { 2560, 1024 },   // 4B: ffn up, ffn down, attn, kv
        { 2048, 6144 }, { 6144, 2048 }, { 2048, 2048 }, { 2048, 512 },    // 2B: ffn up, ffn down, attn, kv
    };
    const size_t  n_km  = sizeof(km) / sizeof(km[0]);
    const size_t  n_q   = 2 * n_km * 8;   // the quantized entries
    const uint8_t sel   = b.rd.u8();
    const size_t  entry = sel % (n_q + 16);
    if (entry >= n_q) {
        // The F32 ssm_alpha and ssm_beta weights of the 2B model, (2048, 16), at n = 1 to 8: one
        // MUL_MAT (entries n_q to n_q + 7), or the pair of MUL_MATs with the same input as the GDN
        // layer has it (entries n_q + 8 to n_q + 15), which the HTP0 fusions can join.
        const bool    pair = entry >= n_q + 8;
        const int64_t n    = (int64_t) ((entry - n_q) % 8) + 1;
        ggml_tensor * x    = b.f32(2048, n, 1, 1, b.vs(-4.0f, 4.0f));
        const int     nw   = pair ? 2 : 1;
        for (int i = 0; i < nw; i++) {
            ggml_tensor * w = b.typed(GGML_TYPE_F32, 2048, 16, 1, 1, b.vs(-0.125f, 0.125f), leaf_role::WEIGHT);
            ggml_tensor * y = ggml_mul_mat(b.ctx, w, x);
            ggml_build_forward_expand(b.c.gf, y);
            b.out(y, i == 0 ? "alpha" : "beta");
        }
        b.c.desc = fmt("MUL_MAT model%s w=f32[2048,16] x=f32[2048,%lld] (ssm_alpha, ssm_beta of the 2B)",
                       pair ? " x2 shared x" : "", (long long) n);
        b.c.path = fmt("f32/%s/n%lld", pair ? "pair" : "one", (long long) n);
        return true;
    }
    const ggml_type wt  = types[entry / (n_km * 8)];
    const int64_t k     = km[entry / 8 % n_km][0];
    const int64_t m     = km[entry / 8 % n_km][1];
    const int64_t n     = (int64_t) (entry % 8) + 1;
    const vspec   v     = b.vs(-0.125f, 0.125f);
    ggml_tensor * w     = b.typed(wt, k, m, 1, 1, v, leaf_role::WEIGHT);
    ggml_tensor * x     = b.f32(k, n, 1, 1, b.vs(-4.0f, 4.0f, true));
    mark_conv_range(b, x, act_type(wt));
    ggml_tensor * y     = ggml_mul_mat(b.ctx, w, x);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.desc = fmt("MUL_MAT model w=%s[%lld,%lld] x=f32[%lld,%lld]", ggml_type_name(wt), (long long) k, (long long) m,
                   (long long) k, (long long) n);
    b.c.path = fmt("%s/n%lld", ggml_type_name(wt), (long long) n);
    return true;
}

bool build_mul_mat_add(builder & b) {
    reader &        rd = b.rd;
    const ggml_type wt = pick_wtype(rd);
    int64_t         k, m, n;
    bool            model;
    mm_shape(b, wt, k, m, n, model);
    ggml_tensor * w     = mm_weight(b, wt, k, m, 1, 1, false);
    ggml_tensor * x     = mm_act(b, k, n, 1, 1, ggml_is_quantized(wt));
    mark_conv_range(b, x, act_type(wt));
    ggml_tensor * y     = ggml_mul_mat(b.ctx, w, x);
    const bool    bias  = rd.chance(128);
    ggml_tensor * add   = bias ? b.f32(m, 1, 1, 1, b.vs(-1.0f, 1.0f), leaf_role::WEIGHT)
                               : b.f32(m, n, 1, 1, b.vs(-4.0f, 4.0f));
    const bool    swap  = !bias && rd.chance(64);
    ggml_tensor * z     = swap ? ggml_add(b.ctx, add, y) : ggml_add(b.ctx, y, add);
    ggml_build_forward_expand(b.c.gf, z);
    b.out(z, "z");
    b.c.desc = fmt("MUL_MAT+ADD w=%s[%lld,%lld] x=f32[%lld,%lld]%s %s%s", ggml_type_name(wt), (long long) k,
                   (long long) m, (long long) k, (long long) n, x->view_src ? " view" : "",
                   bias ? "bias[m]" : (swap ? "residual first" : "residual"), model ? " model" : "");
    b.c.path = fmt("%s/%s", ggml_type_name(wt), nbucket(n));
    return true;
}

void bound_mul_mat_add(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * z  = c.outs[o].t;
    const bool          yf = z->src[0]->op == GGML_OP_MUL_MAT;
    const ggml_tensor * y  = yf ? z->src[0] : z->src[1];
    const ggml_tensor * a  = yf ? z->src[1] : z->src[0];
    std::vector<double> S, Q;
    mm_mass(c, y->src[0], y->src[1], S, Q);
    const std::vector<double> A = logical_values(c, a);
    std::vector<double>       E(ref.size(), 0.0);
    const int64_t             m = z->ne[0];
    for (size_t i = 0; i < E.size() && !A.empty(); i++) {
        E[i] = std::fabs(A[a->ne[1] == 1 ? i % (size_t) m : i]);
    }
    mm_bounds(y->src[0]->type, y->src[0]->ne[0], S, Q, &E, false, ref, ba);
}

const char * TXT_MUL_MAT_ADD =
    "The MUL_MAT rule with the bias or residual |b| added to the mass S. Reason: the add is one more "
    "term of the same sum, and the fused matvec adds it in another place of the order.";

bool build_mul_mat_multi(builder & b) {
    reader &        rd = b.rd;
    const ggml_type wt = pick_wtype(rd);
    int64_t         k, m, n;
    bool            model;
    mm_shape(b, wt, k, m, n, model);
    const int     nw = (int) rd.range(2, 3);
    ggml_tensor * x  = mm_act(b, k, n, 1, 1, ggml_is_quantized(wt));
    mark_conv_range(b, x, act_type(wt));
    for (int i = 0; i < nw; i++) {
        const int64_t mi = std::max<int64_t>(1, m >> rd.range(0, 2));
        ggml_tensor * w  = mm_weight(b, wt, k, mi, 1, 1, false);
        ggml_tensor * y  = ggml_mul_mat(b.ctx, w, x);
        ggml_build_forward_expand(b.c.gf, y);
        b.out(y, "y");
    }
    b.c.desc = fmt("MUL_MAT x%d shared x: w=%s k=%lld m<=%lld n=%lld%s", nw, ggml_type_name(wt), (long long) k,
                   (long long) m, (long long) n, model ? " model" : "");
    b.c.path = fmt("%s/%s", ggml_type_name(wt), nbucket(n));
    return true;
}

bool build_mul_mat_id(builder & b) {
    reader &        rd       = b.rd;
    const ggml_type wt       = pick_wtype(rd);
    const bool      q        = ggml_is_quantized(wt);
    const int64_t   n_expert = rd.range(2, 8);
    const int64_t   n_used   = rd.range(1, std::min<int64_t>(4, n_expert));
    const int64_t   k        = q ? 32 * rd.range(1, 24) : rd.range(1, 200);
    int64_t         m        = rd.range(1, 64);
    int64_t         nt       = rd.range(1, 16);
    const int64_t   b1       = rd.chance(128) ? 1 : n_used;
    while (m > 1 && k * m * n_used * nt > MAX_MACS / 4) {
        m = (m + 1) / 2;
    }
    ggml_tensor * as  = b.typed(wt, k, m, n_expert, 1, b.vs(-0.125f, 0.125f), leaf_role::WEIGHT);
    ggml_tensor * x   = b.f32(k, b1, nt, 1, b.vs(-4.0f, 4.0f, q));
    mark_conv_range(b, x, act_type(wt));
    ggml_tensor * ids = b.idx_distinct(GGML_TYPE_I32, n_used, nt, n_expert);
    ggml_tensor * y   = ggml_mul_mat_id(b.ctx, as, x, ids);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.desc = fmt("MUL_MAT_ID w=%s[%lld,%lld,%lld] x=f32[%lld,%lld,%lld] used=%lld", ggml_type_name(wt), (long long) k,
                   (long long) m, (long long) n_expert, (long long) k, (long long) b1, (long long) nt,
                   (long long) n_used);
    b.c.path = fmt("%s/%s", ggml_type_name(wt), nbucket(nt));
    return true;
}

void bound_mul_mat_id(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * y   = c.outs[o].t;
    const ggml_tensor * as  = y->src[0];
    const ggml_tensor * x   = y->src[1];
    const ggml_tensor * ids = y->src[2];
    const std::vector<double> W = logical_values(c, as);
    const std::vector<double> X = logical_values(c, x);
    const std::vector<double> I = logical_values(c, ids);
    const int64_t k = as->ne[0], m = as->ne[1], n_used = y->ne[1], nt = y->ne[2], b1 = x->ne[1];
    std::vector<double> S(ref.size(), 0.0), Q(ref.size(), 0.0);
    for (int64_t t = 0; t < nt && !W.empty(); t++) {
        for (int64_t e = 0; e < n_used; e++) {
            const int64_t  ex = (int64_t) I[(size_t) (t * n_used + e)];
            const double * xr = &X[(size_t) ((t * b1 + e % b1) * k)];
            for (int64_t i = 0; i < m; i++) {
                const double * wr = &W[(size_t) ((ex * m + i) * k)];
                double s = 0.0, qs = 0.0;
                for (int64_t b0 = 0; b0 < k; b0 += 32) {
                    double a = 0.0;
                    for (int64_t l = b0; l < std::min(k, b0 + 32); l++) {
                        a = std::max(a, std::fabs(xr[l]));
                    }
                    for (int64_t l = b0; l < std::min(k, b0 + 32); l++) {
                        const double xv = as->type == GGML_TYPE_F16 ? r16(xr[l]) : xr[l];
                        s  += std::fabs(wr[l] * xv);
                        qs += std::fabs(wr[l]) * a / 127.0;
                    }
                }
                S[(size_t) ((t * n_used + e) * m + i)] = s;
                Q[(size_t) ((t * n_used + e) * m + i)] = ggml_is_quantized(as->type) ? qs : 0.0;
            }
        }
    }
    mm_bounds(as->type, k, S, Q, nullptr, ggml_is_quantized(as->type) && nt <= 4, ref, ba);
}

// ---------------------------------------------------------------------------------------------
// GATED_DELTA_NET and its chains

struct gdn_shape {
    int64_t S_v, H_k, rep, T, n_seqs, K;
    bool    kda, permuted;
};

void gdn_pick(builder & b, gdn_shape & s, bool one_token) {
    reader & rd = b.rd;
    const bool model = rd.chance(100);
    if (model) {
        s.S_v = 128;
        s.H_k = rd.chance(128) ? 16 : 8;
        s.rep = rd.chance(128) ? 1 : 2;
    } else {
        static const int64_t sv[] = { 16, 32, 64, 128, 8, 24, 48, 96 };
        s.S_v = rd.chance(200) ? rd.pick(sv) : rd.range(1, 128);
        s.H_k = rd.range(1, 8);
        s.rep = rd.chance(170) ? 1 : 2;
    }
    const uint8_t tb = rd.u8();
    s.T      = one_token ? 1 : (tb < 100 ? 1 : tb < 150 ? rd.range(2, 8) : tb < 175 ? rd.range(9, 31) : rd.range(32, 96));
    s.n_seqs = one_token ? 1 : (rd.chance(200) ? 1 : rd.range(2, 3));
    s.K      = rd.chance(150) ? 1 : rd.range(2, 4);
    s.kda    = !one_token && rd.chance(24);
    s.permuted = !one_token && rd.chance(36);
    // keep the state and the token work inside the budgets
    while (s.H_k > 1 && s.S_v * s.S_v * s.H_k * s.rep * s.n_seqs * (s.K + 1) > MAX_ELEMS / 2) {
        s.H_k = (s.H_k + 1) / 2;
    }
    while (s.T > 1 && s.S_v * s.S_v * s.H_k * s.rep * s.n_seqs * s.T > MAX_MACS) {
        s.T = (s.T + 1) / 2;
    }
}

// Make q or k [S_v, H, T, n_seqs], plain or permuted, and give its l2 norm like qwen35.cpp does.
ggml_tensor * gdn_qk(builder & b, const gdn_shape & s) {
    ggml_tensor * t;
    if (s.permuted) {
        t = ggml_permute(b.ctx, b.f32(s.S_v, s.T, s.H_k, s.n_seqs, b.vs(-1.0f, 1.0f)), 0, 2, 1, 3);
    } else {
        t = b.f32(s.S_v, s.H_k, s.T, s.n_seqs, b.vs(-1.0f, 1.0f));
    }
    return ggml_l2_norm(b.ctx, t, 1e-6f);
}

bool build_gdn(builder & b) {
    gdn_shape s;
    gdn_pick(b, s, false);
    const int64_t H  = s.H_k * s.rep;
    ggml_tensor * q  = gdn_qk(b, s);
    ggml_tensor * k  = gdn_qk(b, s);
    ggml_tensor * v;
    if (s.permuted) {
        v = ggml_permute(b.ctx, b.f32(s.S_v, s.T, H, s.n_seqs, b.vs(-0.3f, 5.0f)), 0, 2, 1, 3);
    } else {
        v = b.f32(s.S_v, H, s.T, s.n_seqs, b.vs(-0.3f, 5.0f));
    }
    // A gate near zero keeps the state alive over a batch, thus the chunks carry their errors.
    const float   gmin = s.T >= 32 && b.rd.chance(128) ? -0.5f : -20.0f;
    ggml_tensor * g    = b.f32(s.kda ? s.S_v : 1, H, s.T, s.n_seqs, b.vs(gmin, -1e-4f));
    ggml_tensor * beta = b.f32(1, H, s.T, s.n_seqs, b.vs(0.0f, 1.0f));
    ggml_tensor * st   = b.f32(s.S_v, s.S_v, H, s.n_seqs, b.vs(-1.0f, 1.0f));
    ggml_tensor * y    = ggml_gated_delta_net(b.ctx, q, k, v, g, beta, st, s.K);
    ggml_build_forward_expand(b.c.gf, y);
    output &      o    = b.out(y, "y");
    // The op writes min(T, K) state slots, and the later slots belong to the caller.
    const int64_t attn = s.S_v * H * s.T * s.n_seqs;
    const int64_t D    = s.S_v * s.S_v * H * s.n_seqs;
    const int64_t nw   = std::min(s.T, s.K);
    if (nw < s.K) {
        o.skip.push_back({ attn + nw * D, attn + s.K * D });
    }
    b.c.prm  = { (double) s.S_v, (double) H, (double) s.T, (double) s.n_seqs, (double) s.K };
    b.c.desc = fmt("GATED_DELTA_NET S_v=%lld H_k=%lld H_v=%lld T=%lld seqs=%lld K=%lld%s%s gmin=%g", (long long) s.S_v,
                   (long long) s.H_k, (long long) H, (long long) s.T, (long long) s.n_seqs, (long long) s.K,
                   s.kda ? " kda" : "", s.permuted ? " permuted" : "", (double) gmin);
    const bool chunk = s.T - (s.K > 1 ? s.K : 0) >= 32 && !s.kda && s.n_seqs == 1 && s.S_v % 32 == 0 && s.S_v <= 128;
    b.c.path = fmt("%s%s", s.T == 1 ? "T1" : (chunk ? "chunkable" : "seq"), s.K > 1 ? "/K>1" : "");
    return true;
}

// The GDN bound from the state magnitude M, the token count T and the head size S_v.
void gdn_bounds(double S_v, double T, double M, const std::vector<float> & ref, int64_t attn,
                bound_arrays & ba) {
    // The attention part is scale q^T S with scale = 1/sqrt(S_v) and |q| = 1: its bound is the bound
    // of a state entry times scale * sqrt(S_v) = 1, thus the two parts share one formula. The
    // argument `attn` is kept for a rule that separates them.
    (void) attn;
    const size_t n = ref.size();
    ba.strict.resize(n);
    ba.loose.resize(n);
    for (size_t i = 0; i < n; i++) {
        const double a = std::fabs((double) ref[i]);
        ba.strict[i]   = 8.0 * EPS32 * (T + 2.0) * (S_v + 4.0) * M + 4.0 * EPS32 * a;
        ba.loose[i]    = ba.strict[i] + 32.0 * EPS16 * (T + 2.0) * M;
    }
}

void bound_gdn(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * y = c.outs[o].t;
    const double S_v = c.prm[0], H = c.prm[1], T = c.prm[2], n_seqs = c.prm[3];
    double M = std::max(amax(logical_values(c, y->src[5])), amax(logical_values(c, y->src[2])));
    const int64_t attn = (int64_t) (S_v * H * T * n_seqs);
    for (size_t i = (size_t) attn; i < ref.size(); i++) {
        if (std::isfinite(ref[i])) {
            M = std::max(M, (double) std::fabs(ref[i]));
        }
    }
    gdn_bounds(S_v, T, M, ref, attn, ba);
}

const char * TXT_GDN =
    "strict = 8u (T+2)(S_v+4) M + 4u|y|, loose = strict + 32 u16 (T+2) M, with M = max(|s0|, |v|, |s_out|). "
    "Reason: q and k have unit l2 norm and beta is in [0, 1], thus the update (I - beta k k^T) is a "
    "contraction and exp(g) <= 1. Each token adds at most (S_v+4)u M of rounding in f32 dot products "
    "of length S_v, and the errors do not grow. The attention output is scale q^T S with |q| = 1, "
    "thus it has the bound of the state. The loose bound adds one f16 rounding of each HMX product "
    "for each token (the chunked kernel).";

bool build_gdn_state_chain(builder & b) {
    reader &  rd = b.rd;
    gdn_shape s;
    gdn_pick(b, s, true);
    const int64_t H        = s.H_k * s.rep;
    const int64_t D        = s.S_v * s.S_v * H;
    const int64_t mem_size = rd.range(1, 3);
    const int64_t n_rs     = s.K - 1;
    const int64_t n_rows   = mem_size * (1 + n_rs);
    const int64_t kv_head  = rd.range(0, mem_size - 1);
    if (D * n_rows > MAX_ELEMS) {
        return false;
    }
    ggml_tensor * cache  = b.f32(D, n_rows, 1, 1, b.vs(-1.0f, 1.0f), leaf_role::STATE);
    ggml_tensor * s_copy = b.i32(1, 1, 1, 1, 0, (int32_t) (n_rows - 1));
    ggml_tensor * states = ggml_reshape_2d(b.ctx, cache, D, n_rows);
    if (rd.chance(128)) {
        // build_rs clears one state with a SCALE by 0. rs_z = -1 gives an empty view.
        ggml_build_forward_expand(b.c.gf, ggml_scale_inplace(b.ctx, ggml_view_1d(b.ctx, states, 0, 0), 0.0f));
    }
    ggml_tensor * R     = ggml_get_rows(b.ctx, states, s_copy);
    ggml_tensor * state = ggml_reshape_4d(b.ctx, R, s.S_v, s.S_v, H, 1);
    ggml_tensor * q     = gdn_qk(b, s);
    ggml_tensor * k     = gdn_qk(b, s);
    ggml_tensor * v     = b.f32(s.S_v, H, 1, 1, b.vs(-0.3f, 5.0f));
    ggml_tensor * g     = b.f32(1, H, 1, 1, b.vs(-5.0f, -1e-4f));
    ggml_tensor * beta  = b.f32(1, H, 1, 1, b.vs(0.0f, 1.0f));
    ggml_tensor * G     = ggml_gated_delta_net(b.ctx, q, k, v, g, beta, state, s.K);
    ggml_build_forward_expand(b.c.gf, G);
    const int64_t attn  = s.S_v * H;
    ggml_tensor * src   = ggml_view_3d(b.ctx, G, D, 1, 1, D * 4, D * 4, attn * 4);
    const size_t  row   = (size_t) D * 4;
    ggml_tensor * dst   = ggml_view_3d(b.ctx, cache, D, 1, 1, cache->nb[1], (size_t) mem_size * row, (size_t) kv_head * row);
    ggml_build_forward_expand(b.c.gf, ggml_cpy(b.ctx, src, dst));
    output & og = b.out(G, "G");
    if (s.K > 1) {
        og.skip.push_back({ attn + D, attn + s.K * D });
    }
    b.out(cache, "cache", 1);
    b.c.prm  = { (double) s.S_v, (double) H, 1.0, 1.0, (double) s.K };
    b.c.desc = fmt("GDN state chain S_v=%lld H=%lld K=%lld rows=%lld head=%lld", (long long) s.S_v, (long long) H,
                   (long long) s.K, (long long) n_rows, (long long) kv_head);
    b.c.path = fmt("S_v%lld", (long long) s.S_v);
    return true;
}

void bound_gdn_chain(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    // The initial state comes from a row of the cache table (a GET_ROWS output, not a leaf), thus M
    // takes the whole cache input, v, and the finite values of this output.
    const ggml_tensor * G     = c.outs[0].t;
    const ggml_tensor * cache = c.outs[1].t;
    double M = std::max(amax(logical_values(c, cache)), amax(logical_values(c, G->src[2])));
    for (float r : ref) {
        if (std::isfinite(r)) {
            M = std::max(M, (double) std::fabs(r));
        }
    }
    // The cache rows that the chain does not write are copies, and the slot row holds the state
    // tail of the GDN output: the rule of the state applies to every element.
    gdn_bounds(c.prm[0], 1.0, M, ref, 0, ba);
}

const char * TXT_GDN_CHAIN =
    "The GATED_DELTA_NET rule for the output and for the cache table. Reason: the chain is GET_ROWS, "
    "the delta net step and a CPY of the state tail, and the two copies are exact.";

bool build_gdn_conv_chain(builder & b) {
    reader &      rd       = b.rd;
    const bool    model    = rd.chance(100);
    const int64_t d_conv   = model || rd.chance(200) ? 4 : rd.range(2, 6);
    int64_t       n_ch;
    if (model) {
        n_ch = rd.chance(128) ? 6144 : 8192;
    } else {
        n_ch = rd.chance(128) ? 32 * rd.range(1, 16) : rd.range(1, 300);
    }
    const uint8_t tb       = rd.u8();
    const int64_t T        = tb < 128 ? 1 : tb < 150 ? rd.range(2, 3) : rd.range(4, 64);
    const int64_t n_rs     = rd.chance(150) ? 0 : rd.range(1, 3);
    const int64_t mem_size = rd.range(1, 3);
    const int64_t kv_head  = rd.range(0, mem_size - 1);
    const int64_t n_rows   = mem_size * (1 + n_rs);
    const int64_t row      = (d_conv - 1) * n_ch;
    if (row * n_rows > MAX_ELEMS || n_ch * T * d_conv > MAX_MACS) {
        return false;
    }
    ggml_tensor * cache  = b.f32(row, n_rows, 1, 1, b.vs(-2.0f, 2.0f), leaf_role::STATE);
    ggml_tensor * s_copy = b.i32(1, 1, 1, 1, 0, (int32_t) (n_rows - 1));
    ggml_tensor * states = ggml_reshape_2d(b.ctx, cache, row, n_rows);
    ggml_tensor * R      = ggml_get_rows(b.ctx, states, s_copy);
    ggml_tensor * cs     = ggml_reshape_3d(b.ctx, R, d_conv - 1, n_ch, 1);
    ggml_tensor * x      = b.f32(n_ch, T, 1, 1, b.vs(-2.0f, 2.0f));
    ggml_tensor * xt     = ggml_transpose(b.ctx, x);
    ggml_tensor * CI     = ggml_concat(b.ctx, cs, xt, 0);
    const size_t  rowb   = (size_t) row * 4;
    const int64_t K      = n_rs + 1;
    for (int64_t t = 1; t <= K; ++t) {
        const int64_t s_idx  = n_rs == 0 ? CI->ne[0] - cs->ne[0] : std::max<int64_t>(0, CI->ne[0] - cs->ne[0] - K + t);
        const int64_t s_slot = n_rs == 0 ? 0 : K - t;
        ggml_tensor * last   = ggml_view_3d(b.ctx, CI, d_conv - 1, n_ch, 1, CI->nb[1], CI->nb[2], (size_t) s_idx * 4);
        ggml_tensor * upd    = ggml_view_2d(b.ctx, cache, row, 1, cache->nb[1], (size_t) (s_slot * mem_size + kv_head) * rowb);
        ggml_build_forward_expand(b.c.gf, ggml_cpy(b.ctx, last, upd));
    }
    ggml_tensor * W = b.f32(d_conv, n_ch, 1, 1, b.vs(-0.5f, 0.5f), leaf_role::WEIGHT);
    ggml_tensor * C = ggml_ssm_conv(b.ctx, CI, W);
    ggml_tensor * S = ggml_silu(b.ctx, C);
    ggml_build_forward_expand(b.c.gf, S);
    b.out(S, "silu");
    b.out(cache, "cache", 1);
    b.c.prm  = { (double) d_conv };
    b.c.desc = fmt("GDN conv chain d_conv=%lld n_ch=%lld T=%lld rs=%lld rows=%lld head=%lld%s", (long long) d_conv,
                   (long long) n_ch, (long long) T, (long long) n_rs, (long long) n_rows, (long long) kv_head,
                   model ? " model" : "");
    b.c.path = fmt("%s%s", T == 1 ? "T1" : "batch", d_conv == 4 ? "" : "/dconv!=4");
    return true;
}

// The mass sum |x| |w| of each conv output from the CONCAT input values and the weights.
void conv_mass(const std::vector<double> & X, int64_t len, const std::vector<double> & W, int64_t d_conv,
               int64_t n_ch, int64_t T, int64_t n_s, std::vector<double> & S) {
    S.assign((size_t) (n_ch * T * n_s), 0.0);
    for (int64_t s = 0; s < n_s; s++) {
        for (int64_t t = 0; t < T; t++) {
            for (int64_t ch = 0; ch < n_ch; ch++) {
                double a = 0.0;
                for (int64_t j = 0; j < d_conv; j++) {
                    a += std::fabs(X[(size_t) ((s * n_ch + ch) * len + t + j)] * W[(size_t) (ch * d_conv + j)]);
                }
                S[(size_t) ((s * T + t) * n_ch + ch)] = a;
            }
        }
    }
}

void bound_gdn_conv(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    if (c.outs[o].bound == 1) {
        fill_bounds(ba, ref.size(), 0.0, 0.0);
        return;
    }
    const ggml_tensor * S  = c.outs[o].t;
    const ggml_tensor * C  = S->src[0];
    const ggml_tensor * CI = C->src[0];
    const ggml_tensor * W  = C->src[1];
    const ggml_tensor * R  = CI->src[0]->view_src; // the GET_ROWS
    const ggml_tensor * xt = CI->src[1];
    const int64_t d_conv = W->ne[0], n_ch = W->ne[1], T = S->ne[1], len = CI->ne[0];
    // rebuild the CONCAT: the state row that s_copy selects, then the tokens
    const std::vector<double> cache = logical_values(c, R->src[0]);
    const std::vector<double> idx   = logical_values(c, R->src[1]);
    const std::vector<double> xv    = logical_values(c, xt);
    const std::vector<double> wv    = logical_values(c, W);
    if (cache.empty() || idx.empty() || xv.empty() || wv.empty()) {
        fill_bounds(ba, ref.size(), 0.0, 0.0);
        return;
    }
    const int64_t        row = (d_conv - 1) * n_ch;
    const int64_t        sel = (int64_t) idx[0];
    std::vector<double>  X((size_t) (len * n_ch));
    for (int64_t ch = 0; ch < n_ch; ch++) {
        for (int64_t j = 0; j < d_conv - 1; j++) {
            X[(size_t) (ch * len + j)] = cache[(size_t) (sel * row + ch * (d_conv - 1) + j)];
        }
        for (int64_t t = 0; t < T; t++) {
            X[(size_t) (ch * len + d_conv - 1 + t)] = xv[(size_t) (ch * T + t)];
        }
    }
    std::vector<double> M;
    conv_mass(X, len, wv, d_conv, n_ch, T, 1, M);
    ba.strict.resize(ref.size());
    ba.loose.resize(ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        const double a = std::fabs((double) ref[i]);
        ba.strict[i]   = 1.1 * 2.0 * gam((double) d_conv + 2) * M[i] + 16.0 * EPS32 * (2.0 + M[i]) * a;
        ba.loose[i]    = ba.strict[i] + 1.1 * 4.0 * EPS16 * M[i] + 4.0 * EPS16 * a;
    }
}

const char * TXT_GDN_CONV =
    "SILU output: strict = 1.1 * 2 gamma(d+2) M + 16u (2 + M)|y|, loose = strict + 4.4 u16 M + 4 u16 |y|, "
    "with M = sum |x||w| of the conv. Cache table: exact (0). Reason: the conv is a sum of d = 4 "
    "products, silu' <= 1.1 carries its error, and the silu itself has an exp error that grows with "
    "|x| <= M. The cache rows are copies of the CONCAT columns.";

bool build_ssm_conv(builder & b) {
    reader &      rd     = b.rd;
    const bool    model  = rd.chance(80);
    const int64_t d_conv = model ? 4 : rd.range(1, 8);
    const int64_t d_in   = model ? (rd.chance(128) ? 6144 : 8192) : rd.range(1, 300);
    const int64_t T      = rd.chance(128) ? 1 : rd.range(2, 64);
    const int64_t n_s    = rd.chance(200) ? 1 : rd.range(2, 3);
    if (d_in * (d_conv - 1 + T) * n_s > MAX_ELEMS) {
        return false;
    }
    ggml_tensor * sx = b.f32(d_conv - 1 + T, d_in, n_s, 1, b.vs(-2.0f, 2.0f));
    ggml_tensor * cw = b.f32(d_conv, d_in, 1, 1, b.vs(-0.5f, 0.5f), leaf_role::WEIGHT);
    ggml_tensor * y  = ggml_ssm_conv(b.ctx, sx, cw);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.desc = fmt("SSM_CONV d_conv=%lld d_inner=%lld T=%lld seqs=%lld", (long long) d_conv, (long long) d_in,
                   (long long) T, (long long) n_s);
    b.c.path = fmt("%s", T == 1 ? "T1" : "batch");
    return true;
}

void bound_ssm_conv(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * y  = c.outs[o].t;
    const ggml_tensor * sx = y->src[0];
    const ggml_tensor * cw = y->src[1];
    std::vector<double> M;
    conv_mass(logical_values(c, sx), sx->ne[0], logical_values(c, cw), cw->ne[0], cw->ne[1], y->ne[1], y->ne[2], M);
    ba.strict.resize(ref.size());
    ba.loose.resize(ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ba.strict[i] = 2.0 * gam((double) cw->ne[0] + 1) * M[i] + 2.0 * EPS32 * std::fabs((double) ref[i]);
        ba.loose[i]  = ba.strict[i] + 4.0 * EPS16 * M[i];
    }
}

const char * TXT_SSM_CONV =
    "strict = 2 gamma(d+1) M + 2u|y|, loose = strict + 4 u16 M, M = sum over the d taps of |x||w|. "
    "Reason: a sum of d f32 products in any order.";

// ---------------------------------------------------------------------------------------------
// ROPE

bool build_rope(builder & b) {
    reader &      rd    = b.rd;
    const bool    model = rd.chance(100);
    static const int modes[] = { GGML_ROPE_TYPE_IMROPE, GGML_ROPE_TYPE_IMROPE, GGML_ROPE_TYPE_IMROPE,
                                 GGML_ROPE_TYPE_NEOX, GGML_ROPE_TYPE_NEOX, GGML_ROPE_TYPE_NORMAL,
                                 GGML_ROPE_TYPE_MROPE };
    const int     mode  = model ? GGML_ROPE_TYPE_IMROPE : rd.pick(modes);
    const int64_t hd    = model ? 256 : 2 * rd.range(1, 150);
    const int     n_dims = model ? 64 : (int) (2 * rd.range(1, hd / 2));
    const int64_t nh    = model ? (rd.chance(128) ? 8 : 2) : rd.range(1, 16);
    const int64_t T     = rd.chance(100) ? 1 : rd.range(2, 64);
    int           sections[4] = { 11, 11, 10, 0 };
    const bool    mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;
    if (mrope && !model) {
        const int half = n_dims / 2;
        sections[0]    = (int) rd.range(0, half);
        sections[1]    = (int) rd.range(0, half - sections[0]);
        sections[2]    = (int) rd.range(0, half - sections[0] - sections[1]);
        sections[3]    = 0;
        if (sections[0] + sections[1] + sections[2] == 0) {
            sections[0] = 1;
        }
    }
    static const float bases[] = { 1e7f, 1e4f, 1e6f, 5e5f };
    const float base   = model ? 1e7f : rd.pick(bases);
    static const float fscales[] = { 1.0f, 1.0f, 0.5f, 0.25f };
    const float fscale = model ? 1.0f : rd.pick(fscales);
    const float ext    = model || rd.chance(200) ? 0.0f : 1.0f;
    const int   n_ctx_orig = ext != 0.0f ? 4096 : 0;
    static const int32_t pmax[] = { 32, 4096, 131072, 1 << 20 };
    const int32_t pm   = model ? 4096 : rd.pick(pmax);
    // the input: plain, or the Q view of Qcur_full (a row stride of two heads) like qwen35.cpp
    ggml_tensor * a;
    const uint8_t view = rd.u8();
    if (view < 90) {
        ggml_tensor * full = b.f32(hd * 2, nh, T, 1, b.vs(-2.0f, 2.0f));
        const size_t  off  = view < 45 ? 0 : (size_t) hd * 4;
        a = ggml_view_3d(b.ctx, full, hd, nh, T, full->nb[1], full->nb[2], off);
    } else {
        a = b.f32(hd, nh, T, 1, b.vs(-2.0f, 2.0f));
    }
    ggml_tensor * pos = b.i32(mrope ? 4 * T : T, 1, 1, 1, b.wild && b.rd.chance(32) ? -pm : 0, pm);
    ggml_tensor * ff  = !model && rd.chance(40) ? b.f32(n_dims / 2, 1, 1, 1, b.vs(0.9f, 1.1f)) : nullptr;
    if (ff != nullptr) {
        // A wide distribution can give a factor near 0, and the angle pos / ff can then go past
        // 2^64. At such an angle one ulp is more than 2 pi, thus sin and cos (or inf and NaN when
        // the angle overflows) depend on the order of the operations, and the case is special.
        double ffmin = std::numeric_limits<double>::infinity();
        for (double f : logical_values(b.c, ff)) {
            ffmin = std::min(ffmin, std::fabs(f));
        }
        if (!((double) pm * std::max(1.0, (double) fscale) < 0x1p64 * ffmin)) {
            b.c.special = true;
        }
    }
    ggml_tensor * y;
    if (mrope) {
        y = ggml_rope_multi(b.ctx, a, pos, ff, n_dims, sections, mode, n_ctx_orig, base, fscale, ext, 1.0f, 32.0f, 1.0f);
    } else {
        y = ggml_rope_ext(b.ctx, a, pos, ff, n_dims, mode, n_ctx_orig, base, fscale, ext, 1.0f, 32.0f, 1.0f);
    }
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.prm  = { (double) mode, (double) n_dims, (double) base, (double) fscale, (double) ext };
    const char * mname = mode == GGML_ROPE_TYPE_IMROPE ? "imrope" : mode == GGML_ROPE_TYPE_MROPE ? "mrope"
                       : mode == GGML_ROPE_TYPE_NEOX ? "neox" : "normal";
    b.c.desc = fmt("ROPE %s hd=%lld n_dims=%d heads=%lld T=%lld sec=[%d,%d,%d] base=%g fs=%g ext=%g pos<=%d%s%s",
                   mname, (long long) hd, n_dims, (long long) nh, (long long) T, sections[0], sections[1], sections[2],
                   (double) base, (double) fscale, (double) ext, pm, ff ? " ff" : "", a->view_src ? " view" : "");
    b.c.path = fmt("%s/%s", mname, pm <= 4096 ? "pos<=4096" : "pos>4096");
    return true;
}

void bound_rope(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * y    = c.outs[o].t;
    const ggml_tensor * a    = y->src[0];
    const int     mode   = (int) c.prm[0];
    const int64_t n_dims = (int64_t) c.prm[1];
    const double  base   = c.prm[2];
    const double  fscale = c.prm[3];
    const std::vector<double> X  = logical_values(c, a);
    const std::vector<double> P  = logical_values(c, y->src[1]);
    const std::vector<double> FF = y->src[2] ? logical_values(c, y->src[2]) : std::vector<double>();
    const int64_t hd = a->ne[0], nh = a->ne[1], T = a->ne[2];
    const bool    neox_like = mode != GGML_ROPE_TYPE_NORMAL;
    const bool    mrope     = (mode & GGML_ROPE_TYPE_MROPE) != 0;
    ba.strict.assign(ref.size(), 0.0);
    ba.loose.assign(ref.size(), 0.0);
    for (int64_t t = 0; t < T; t++) {
        double pmax = std::fabs(P[(size_t) t]);
        if (mrope) {
            for (int s = 1; s < 4; s++) {
                pmax = std::max(pmax, std::fabs(P[(size_t) (t + s * T)]));
            }
        }
        for (int64_t h = 0; h < nh; h++) {
            const int64_t base_i = (t * nh + h) * hd;
            for (int64_t i = 0; i < n_dims / 2; i++) {
                const int64_t i0 = neox_like ? i : 2 * i;
                const int64_t i1 = neox_like ? i + n_dims / 2 : 2 * i + 1;
                // a factor of 0 gives an infinite angle, thus an infinite bound
                const double  ff = FF.empty() ? 1.0 : std::fabs(FF[(size_t) i]);
                const double  th = pmax * std::pow(base, -2.0 * (double) i / (double) n_dims) * std::max(1.0, fscale) / ff;
                const double  xm = 1.5 * (std::fabs(X[(size_t) (base_i + i0)]) + std::fabs(X[(size_t) (base_i + i1)]));
                const double  s  = xm * ((double) (n_dims / 2 + 4) * 4.0 * EPS32 * (th + 1.0));
                const double  l  = s + xm * 4.0 * EPS16;
                for (int64_t e : { i0, i1 }) {
                    const double r = std::fabs((double) ref[(size_t) (base_i + e)]);
                    ba.strict[(size_t) (base_i + e)] = s + 2.0 * EPS32 * r;
                    ba.loose[(size_t) (base_i + e)]  = l + 2.0 * EPS32 * r;
                }
            }
        }
    }
}

const char * TXT_ROPE =
    "rotated pair: strict = 1.5 (|x0|+|x1|) (n_dims/2 + 4) 4u (|theta| + 1) + 2u|y|, loose = strict + "
    "1.5 (|x0|+|x1|) 4 u16; the other dims are copies (0). |theta| <= max|pos| base^(-2i/n_dims) "
    "max(1, freq_scale) / |ff|. Reason: the CPU makes theta by n_dims/2 repeated f32 products, thus "
    "the angle carries a relative error of about (i+2)u, and a large position turns it into an "
    "absolute angle error. The factor 1.5 covers the YaRN magnitude scale. The loose bound adds one f16 "
    "rounding of sin and cos.";

// ---------------------------------------------------------------------------------------------
// FLASH_ATTN_EXT

bool build_flash_attn(builder & b) {
    reader &      rd    = b.rd;
    const bool    model = rd.chance(100);
    static const int64_t dks[] = { 64, 128, 256, 40, 80, 96, 32, 16 };
    const int64_t D     = model ? 256 : rd.pick(dks);
    const int64_t nhkv  = model ? (rd.chance(128) ? 2 : 4) : rd.range(1, 4);
    static const int64_t gqas[] = { 1, 2, 4 };
    const int64_t gqa   = model ? 4 : rd.pick(gqas);
    const int64_t nh    = nhkv * gqa;
    const uint8_t qb    = rd.u8();
    int64_t       n_q   = qb < 100 ? 1 : qb < 160 ? rd.range(2, 4) : rd.range(5, 64);
    int64_t       n_kv  = rd.chance(128) ? rd.range(1, 128) : rd.range(129, 1024);
    const bool    kq8   = D % 32 == 0 && rd.chance(50);
    const ggml_type kt  = kq8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    const ggml_type vt  = kq8 && rd.chance(200) ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    while (n_kv > 1 && n_q * nh * n_kv * D > MAX_MACS / 2) {
        n_kv = (n_kv + 1) / 2;
    }
    if (n_q > n_kv && rd.chance(128)) {
        n_q = n_kv;
    }
    // K and V are views of a cache with room for more positions, like the KV cache of llama
    const int64_t kv_max = n_kv + (rd.chance(128) ? rd.range(0, 64) : 0);
    ggml_tensor * kc = kt == GGML_TYPE_F16 ? b.f16(D, kv_max, nhkv, 1, b.vs(-1.0f, 1.0f))
                                           : b.quant(kt, D, kv_max, nhkv, 1, -9, -5);
    ggml_tensor * vc = vt == GGML_TYPE_F16 ? b.f16(D, kv_max, nhkv, 1, b.vs(-1.0f, 1.0f))
                                           : b.quant(vt, D, kv_max, nhkv, 1, -9, -5);
    ggml_tensor * k  = ggml_view_4d(b.ctx, kc, D, n_kv, nhkv, 1, kc->nb[1], kc->nb[2], kc->nb[3], 0);
    ggml_tensor * v  = ggml_view_4d(b.ctx, vc, D, n_kv, nhkv, 1, vc->nb[1], vc->nb[2], vc->nb[3], 0);
    // Q: llama permutes Qcur [D, heads, n_q] to [D, n_q, heads]
    ggml_tensor * q;
    if (rd.chance(170)) {
        q = ggml_permute(b.ctx, b.f32(D, nh, n_q, 1, b.vs(-2.0f, 2.0f)), 0, 2, 1, 3);
    } else {
        q = b.f32(D, n_q, nh, 1, b.vs(-2.0f, 2.0f));
    }
    // The CPU converts Q to the type of K for the dot products. With an F16 V, its one-row path
    // adds the weighted V rows in an f16 accumulator, and the sum of n_kv rows with weights of 1 or
    // less stays below n_kv max|V|. Half of the f16 range keeps the rounding of the f16 sum away
    // from the limit.
    mark_conv_range(b, q, kt);
    if (vt == GGML_TYPE_F16 && (double) n_kv * amax(logical_values(b.c, v)) >= 0.5 * F16_INF_AT) {
        b.c.special = true;
    }
    const uint8_t mb     = rd.u8();
    ggml_tensor * mask   = mb < 25 && !model ? nullptr : b.mask_f16(n_kv, n_q, 1, 1, model || mb < 160);
    static const float scales[] = { 0.0625f, 0.125f, 0.08838835f, 1.0f };
    const float   scale  = model ? 1.0f / std::sqrt((float) D) : rd.pick(scales);
    const float   mbias  = mask && !model && rd.chance(20) ? 8.0f : 0.0f;
    const float   softcap = !model && rd.chance(12) ? 30.0f : 0.0f;
    ggml_tensor * y      = ggml_flash_attn_ext(b.ctx, q, k, v, mask, scale, mbias, softcap);
    ggml_tensor * sinks  = nullptr;
    if (!model && rd.chance(24)) {
        sinks = b.f32(nh, 1, 1, 1, b.vs(-10.0f, 10.0f));
        ggml_flash_attn_ext_add_sinks(y, sinks);
    }
    ggml_prec_set_acc(y, rd.chance(180) ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.prm  = { (double) scale, (double) mbias, (double) softcap };
    b.c.desc = fmt("FLASH_ATTN_EXT D=%lld heads=%lld kv_heads=%lld n_q=%lld n_kv=%lld k=%s v=%s mask=%s scale=%g "
                   "bias=%g softcap=%g%s%s", (long long) D, (long long) nh, (long long) nhkv, (long long) n_q,
                   (long long) n_kv, ggml_type_name(kt), ggml_type_name(vt),
                   mask ? (mb < 160 || model ? "causal" : "random") : "none", (double) scale, (double) mbias,
                   (double) softcap, sinks ? " sinks" : "", q->op == GGML_OP_PERMUTE ? " q-permuted" : "");
    b.c.path = fmt("%s/%s", ggml_type_name(kt), nbucket(n_q));
    return true;
}

void bound_flash_attn(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * y     = c.outs[o].t;
    const ggml_tensor * q     = y->src[0];
    const ggml_tensor * k     = y->src[1];
    const ggml_tensor * v     = y->src[2];
    const ggml_tensor * mask  = y->src[3];
    const ggml_tensor * sinks = y->src[4];
    const std::vector<double> Qv = logical_values(c, q);
    const std::vector<double> Kv = logical_values(c, k);
    const std::vector<double> Vv = logical_values(c, v);
    const std::vector<double> Mv = mask ? logical_values(c, mask) : std::vector<double>();
    const std::vector<double> Sv = sinks ? logical_values(c, sinks) : std::vector<double>();
    const double scale = c.prm[0], mbias = c.prm[1], softcap = c.prm[2];
    const int64_t D = q->ne[0], n_q = q->ne[1], nh = q->ne[2], n_kv = k->ne[1], nhkv = k->ne[2], DV = v->ne[0];
    const bool    q8k = k->type == GGML_TYPE_Q8_0;
    const bool    v16 = v->type == GGML_TYPE_F16;
    const uint32_t n_head_log2 = 1u << (uint32_t) std::floor(std::log2((double) nh));
    const double  m0 = std::pow(2.0, -mbias / n_head_log2);
    const double  m1 = std::pow(2.0, -(mbias / 2.0) / n_head_log2);
    ba.strict.assign(ref.size(), 0.0);
    ba.loose.assign(ref.size(), 0.0);
    std::vector<double> s((size_t) n_kv), ds((size_t) n_kv), p((size_t) n_kv), qq((size_t) D);
    for (int64_t iq = 0; iq < n_q; iq++) {
        for (int64_t h = 0; h < nh; h++) {
            const int64_t hk    = h / (nh / nhkv);
            const double  slope = mbias > 0 ? (h < (int64_t) n_head_log2 ? std::pow(m0, h + 1) : std::pow(m1, 2 * (h - n_head_log2) + 1)) : 1.0;
            double qstep = 0.0;
            for (int64_t d = 0; d < D; d++) {
                const double x = Qv[(size_t) ((h * n_q + iq) * D + d)];
                qq[(size_t) d] = q8k ? x : r16(x);
                qstep = std::max(qstep, std::fabs(x) / 127.0);
            }
            double smax = -INFINITY, dsmax = 0.0;
            for (int64_t j = 0; j < n_kv; j++) {
                double dot = 0.0, mass = 0.0, kabs = 0.0;
                for (int64_t d = 0; d < D; d++) {
                    const double kv = Kv[(size_t) ((hk * n_kv + j) * D + d)];
                    dot  += qq[(size_t) d] * kv;
                    mass += std::fabs(qq[(size_t) d] * kv);
                    kabs += std::fabs(kv);
                }
                double sv = dot * scale;
                if (softcap != 0.0) {
                    sv = softcap * std::tanh(sv / softcap);
                }
                const double mv = mask ? slope * Mv[(size_t) (iq * n_kv + j)] : 0.0;
                sv += mv;
                s[(size_t) j]  = sv;
                ds[(size_t) j] = scale * (2.0 * (double) (D + 4) * EPS32 * mass + (q8k ? kabs * qstep : 0.0)) +
                                 4.0 * EPS32 * std::fabs(sv);
                if (sv > smax) {
                    smax = sv;
                }
            }
            if (sinks && std::isfinite(Sv[(size_t) h])) {
                smax = std::max(smax, Sv[(size_t) h]);
            }
            double sum = 0.0;
            for (int64_t j = 0; j < n_kv; j++) {
                p[(size_t) j] = std::isfinite(s[(size_t) j]) ? std::exp(s[(size_t) j] - smax) : 0.0;
                sum += p[(size_t) j];
                if (p[(size_t) j] > 0.0) {
                    dsmax = std::max(dsmax, ds[(size_t) j]);
                }
            }
            if (sinks) {
                sum += std::exp(Sv[(size_t) h] - smax);
            }
            for (int64_t d = 0; d < DV; d++) {
                double E = 0.0;
                for (int64_t j = 0; j < n_kv; j++) {
                    if (p[(size_t) j] > 0.0) {
                        E += p[(size_t) j] * std::fabs(Vv[(size_t) ((hk * n_kv + j) * DV + d)]);
                    }
                }
                E = sum > 0.0 ? E / sum : 0.0;
                const size_t oi = (size_t) ((iq * nh + h) * DV + d);
                const double r  = std::fabs((double) ref[oi]);
                const double st = (2.0 * dsmax + (double) (n_kv + 4) * 4.0 * EPS32) * E +
                                  (v16 ? (double) (n_kv + 2) * EPS16 * E : 0.0) + 4.0 * EPS32 * r;
                ba.strict[oi] = st;
                ba.loose[oi]  = st + 12.0 * EPS16 * E;
            }
        }
    }
}

const char * TXT_FLASH_ATTN =
    "strict = (2 max ds + (n_kv+4) 4u) E + [F16 V: (n_kv+2) u16 E] + 4u|y|, loose = strict + 12 u16 E, "
    "with E = sum_j p_j |v_j| and ds_j = scale (2 (D+4) u sum|q||k| + [Q8_0 K: sum|k| amax(q)/127]) + 4u|s_j|. "
    "Reason: an error ds in each logit moves the softmax weights by a relative 2 ds, and the output "
    "is a convex sum of the V rows. The oracle converts Q to f16 (Q8_0 for a Q8_0 K) and, for an F16 V, "
    "keeps its running V sum in f16 (VKQ16 in ops.cpp), which rounds at each of the n_kv steps: that "
    "error of the oracle itself is part of the strict bound.";

// ---------------------------------------------------------------------------------------------
// SOFT_MAX

bool build_soft_max(builder & b) {
    reader &      rd    = b.rd;
    int64_t       ne0   = rd.chance(128) ? rd.range(1, 64) : (rd.chance(128) ? 32 * rd.range(1, 64) : rd.range(1, 2048));
    const int64_t rows  = rd.range(1, 32);
    const int64_t ne2   = rd.range(1, 4);
    const int64_t ne3   = rd.chance(200) ? 1 : 2;
    while (ne0 > 1 && ne0 * rows * ne2 * ne3 > MAX_ELEMS / 4) {
        ne0 = (ne0 + 1) / 2;
    }
    ggml_tensor * a     = b.f32(ne0, rows, ne2, ne3, b.vs(-10.0f, 10.0f));
    const uint8_t mb    = rd.u8();
    ggml_tensor * mask  = nullptr;
    if (mb >= 100) {
        const int64_t m2 = rd.chance(128) ? 1 : ne2;
        const int64_t mr = rows + (rd.chance(64) ? rd.range(0, 4) : 0);
        if (mb < 180) {
            mask = b.mask_f16(ne0, mr, m2, 1, false);
        } else {
            mask = b.f32(ne0, mr, m2, 1, b.vs(-4.0f, 0.0f));
        }
    }
    static const float scales[] = { 1.0f, 0.0625f, 0.125f, 0.08838835f, 3.0f };
    const float   scale = rd.pick(scales);
    const float   mbias = mask && rd.chance(30) ? 8.0f : 0.0f;
    ggml_tensor * y     = ggml_soft_max_ext(b.ctx, a, mask, scale, mbias);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.prm  = { (double) scale, (double) mbias };
    b.c.desc = fmt("SOFT_MAX [%lld,%lld,%lld,%lld] mask=%s scale=%g bias=%g", (long long) ne0, (long long) rows,
                   (long long) ne2, (long long) ne3, mask ? ggml_type_name(mask->type) : "none", (double) scale,
                   (double) mbias);
    b.c.path = fmt("%s", ne0 <= 32 ? "ne0<=32" : (ne0 % 32 == 0 ? "ne0%32==0" : "ne0-odd"));
    return true;
}

void bound_soft_max(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * y    = c.outs[o].t;
    const ggml_tensor * a    = y->src[0];
    const ggml_tensor * mask = y->src[1];
    const std::vector<double> X = logical_values(c, a);
    const std::vector<double> M = mask ? logical_values(c, mask) : std::vector<double>();
    const double  scale = c.prm[0], mbias = c.prm[1];
    const int64_t ne0 = a->ne[0], ne1 = a->ne[1], ne2 = a->ne[2], ne3 = a->ne[3];
    const uint32_t n_head_log2 = 1u << (uint32_t) std::floor(std::log2((double) ne2));
    const double  m0 = std::pow(2.0, -mbias / n_head_log2);
    const double  m1 = std::pow(2.0, -(mbias / 2.0) / n_head_log2);
    ba.strict.assign(ref.size(), 0.0);
    ba.loose.assign(ref.size(), 0.0);
    std::vector<double> av((size_t) ne0), dv((size_t) ne0), p((size_t) ne0);
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const double slope = mbias > 0 ? (i2 < (int64_t) n_head_log2 ? std::pow(m0, i2 + 1) : std::pow(m1, 2 * (i2 - n_head_log2) + 1)) : 1.0;
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                const int64_t base = ((i3 * ne2 + i2) * ne1 + i1) * ne0;
                double amx = -INFINITY;
                for (int64_t j = 0; j < ne0; j++) {
                    double m = 0.0;
                    if (mask) {
                        const int64_t mi = (((i3 % mask->ne[3]) * mask->ne[2] + i2 % mask->ne[2]) * mask->ne[1] + i1) * ne0 + j;
                        m = slope * M[(size_t) mi];
                    }
                    const double xs = X[(size_t) (base + j)] * scale;
                    av[(size_t) j]  = xs + m;
                    dv[(size_t) j]  = 2.0 * EPS32 * (std::fabs(xs) + std::fabs(m));
                    amx = std::max(amx, av[(size_t) j]);
                }
                double sum = 0.0;
                for (int64_t j = 0; j < ne0; j++) {
                    p[(size_t) j] = std::isfinite(av[(size_t) j]) ? std::exp(av[(size_t) j] - amx) : 0.0;
                    sum += p[(size_t) j];
                }
                double A = 0.0, Dl = 0.0;
                for (int64_t j = 0; j < ne0 && sum > 0.0; j++) {
                    const double pj = p[(size_t) j] / sum;
                    if (pj > 0.0) {
                        A  += pj * std::fabs(av[(size_t) j] - amx);
                        Dl += pj * dv[(size_t) j];
                    }
                }
                for (int64_t j = 0; j < ne0; j++) {
                    const double yj = sum > 0.0 ? p[(size_t) j] / sum : 0.0;
                    const double d  = std::isfinite(av[(size_t) j]) ? std::fabs(av[(size_t) j] - amx) : 0.0;
                    const double st = yj * (2.0 * (dv[(size_t) j] + Dl) + 8.0 * EPS32 * (d + A + 2.0) + (double) (ne0 + 4) * EPS32);
                    ba.strict[(size_t) (base + j)] = st;
                    ba.loose[(size_t) (base + j)]  = st + 4.0 * EPS16 * yj * (1.0 + d + A);
                }
            }
        }
    }
}

const char * TXT_SOFT_MAX =
    "strict = y (2 (da + sum p da) + 8u (|a - max| + A + 2) + (n+4)u), loose = strict + 4 u16 y (1 + |a - max| + A), "
    "a = x scale + slope m, da = 2u (|x scale| + |slope m|), A = sum p |a - max|. Reason: the rounding "
    "of the argument moves exp by a relative da, the exp itself has an error of a few u relative to "
    "its argument, and the normalizing sum has n terms.";

// ---------------------------------------------------------------------------------------------
// RMS_NORM, L2_NORM

bool build_norm(builder & b, bool l2) {
    reader &      rd    = b.rd;
    const bool    model = rd.chance(100);
    static const int64_t mn[] = { 2048, 2560, 256, 128 };
    const int64_t n     = model ? rd.pick(mn) : rd.range(1, 600);
    const int64_t rows  = rd.range(1, 32);
    const int64_t ne2   = rd.chance(180) ? 1 : rd.range(2, 4);
    static const float epss[] = { 1e-6f, 1e-6f, 1e-5f, 0.0f, 1e-12f };
    const float   eps   = model ? 1e-6f : rd.pick(epss);
    if (n * rows * ne2 > MAX_ELEMS / 2) {
        return false;
    }
    ggml_tensor * x;
    if (rd.chance(80)) {
        // a row stride of two rows, like the Q view of Qcur_full that qwen35.cpp normalizes
        ggml_tensor * full = b.f32(2 * n, rows, ne2, 1, b.vs(-3.0f, 3.0f));
        x = ggml_view_3d(b.ctx, full, n, rows, ne2, full->nb[1], full->nb[2], rd.chance(128) ? 0 : (size_t) n * 4);
    } else {
        x = b.f32(n, rows, ne2, 1, b.vs(-3.0f, 3.0f));
    }
    ggml_tensor * y = l2 ? ggml_l2_norm(b.ctx, x, eps) : ggml_rms_norm(b.ctx, x, eps);
    const bool    fuse = !l2 && rd.chance(128);
    if (fuse) {
        y = ggml_mul(b.ctx, y, b.f32(n, 1, 1, 1, b.vs(0.5f, 1.5f), leaf_role::WEIGHT));
    }
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.desc = fmt("%s n=%lld rows=%lld ne2=%lld eps=%g%s%s", l2 ? "L2_NORM" : "RMS_NORM", (long long) n, (long long) rows,
                   (long long) ne2, (double) eps, x->view_src ? " view" : "", fuse ? " +MUL" : "");
    b.c.path = fmt("%s%s", n % 32 == 0 ? "n%32==0" : "n-odd", fuse ? "/fused-mul" : "");
    return true;
}

bool build_rms_norm(builder & b) { return build_norm(b, false); }
bool build_l2_norm(builder & b) { return build_norm(b, true); }

void bound_norm(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * y = c.outs[o].t;
    const double        n = (double) y->ne[0];
    ba.strict.resize(ref.size());
    ba.loose.resize(ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        const double a = std::fabs((double) ref[i]);
        ba.strict[i]   = (n + 10.0) * EPS32 * a;
        ba.loose[i]    = ba.strict[i] + 4.0 * EPS16 * a;
    }
}

const char * TXT_NORM =
    "strict = (n+10) u |y|, loose = strict + 4 u16 |y|. Reason: the sum of n squares in f32 in any "
    "order has a relative error of gamma(n), the square root halves it, and the scale, the product "
    "and the fused weight add a few roundings.";

// ---------------------------------------------------------------------------------------------
// Elementwise ops

// A random shape [ne0, ne1, ne2, ne3], or a shape of the model.
void ew_shape(builder & b, int64_t ne[4], bool & model) {
    reader & rd = b.rd;
    model       = rd.chance(80);
    if (model) {
        static const int64_t n0[] = { 2048, 2560, 6144, 9216, 16, 32, 1, 128 };
        ne[0] = rd.pick(n0);
        ne[1] = rd.range(1, 16);
        ne[2] = 1;
        ne[3] = 1;
    } else {
        ne[0] = rd.chance(64) ? rd.range(1, 4) : rd.range(1, 300);
        ne[1] = rd.range(1, 12);
        ne[2] = rd.chance(170) ? 1 : rd.range(2, 3);
        ne[3] = rd.chance(220) ? 1 : 2;
    }
    while (ne[1] > 1 && ne[0] * ne[1] * ne[2] * ne[3] > MAX_ELEMS / 4) {
        ne[1] = (ne[1] + 1) / 2;
    }
}

bool build_binary(builder & b) {
    reader &   rd = b.rd;
    int64_t    ne[4];
    bool       model;
    ew_shape(b, ne, model);
    const bool mul = rd.chance(128);
    const bool f16 = rd.chance(40);
    const ggml_type t = f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * a;
    if (!f16 && rd.chance(60)) {
        // Two byte reads in the arguments of one call have no fixed order in C++ (gcc and clang
        // differ), thus each read has its own statement.
        const int64_t pad  = rd.range(1, 16);
        const vspec   va   = b.vs(-4.0f, 4.0f);
        ggml_tensor * full = b.f32(ne[0] + pad, ne[1], ne[2], ne[3], va);
        a = ggml_view_4d(b.ctx, full, ne[0], ne[1], ne[2], ne[3], full->nb[1], full->nb[2], full->nb[3], 0);
    } else {
        a = b.typed(t, ne[0], ne[1], ne[2], ne[3], b.vs(-4.0f, 4.0f));
    }
    int64_t nb_[4];
    for (int d = 0; d < 4; d++) {
        const uint8_t f = rd.u8();
        nb_[d] = f < 90 ? 1 : ne[d];
        if (f >= 240 && ne[d] % 2 == 0) {
            nb_[d] = ne[d] / 2;
        }
    }
    ggml_tensor * bt = b.typed(t, nb_[0], nb_[1], nb_[2], nb_[3], b.vs(-4.0f, 4.0f));
    ggml_tensor * y  = mul ? ggml_mul(b.ctx, a, bt) : ggml_add(b.ctx, a, bt);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.desc = fmt("%s %s a=[%lld,%lld,%lld,%lld]%s b=[%lld,%lld,%lld,%lld]", mul ? "MUL" : "ADD", ggml_type_name(t),
                   (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3], a->view_src ? " view" : "",
                   (long long) nb_[0], (long long) nb_[1], (long long) nb_[2], (long long) nb_[3]);
    b.c.path = fmt("%s/%s", mul ? "mul" : "add", ggml_type_name(t));
    return true;
}

void bound_exact(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    (void) c;
    (void) o;
    ba.strict.assign(ref.size(), 0.0);
    ba.loose.resize(ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ba.loose[i] = 2.0 * EPS16 * std::fabs((double) ref[i]);
    }
}

const char * TXT_EXACT =
    "strict = 0, loose = 2 u16 |y|. Reason: one IEEE operation (or a copy, or a round to nearest even "
    "conversion) has one correctly rounded result, thus a correct backend gives the same bits. The "
    "loose bound is one f16 rounding.";

bool build_scale(builder & b) {
    reader & rd = b.rd;
    int64_t  ne[4];
    bool     model;
    ew_shape(b, ne, model);
    static const float ss[] = { 0.0f, 1.0f, 0.0625f, -2.5f, 1e-3f, 3.0f };
    const float   s    = rd.pick(ss);
    const bool    bias = rd.chance(100);
    const float   bv   = bias ? rd.pick(ss) : 0.0f;
    ggml_tensor * a    = b.f32(ne[0], ne[1], ne[2], ne[3], b.vs(-4.0f, 4.0f));
    ggml_tensor * y    = bias ? ggml_scale_bias(b.ctx, a, s, bv) : ggml_scale(b.ctx, a, s);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.prm  = { (double) s, (double) bv };
    b.c.desc = fmt("SCALE [%lld,%lld,%lld,%lld] s=%g b=%g", (long long) ne[0], (long long) ne[1], (long long) ne[2],
                   (long long) ne[3], (double) s, (double) bv);
    b.c.path = bias ? "scale+bias" : "scale";
    return true;
}

void bound_scale(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const std::vector<double> X = logical_values(c, c.outs[o].t->src[0]);
    const double s = c.prm[0], bv = c.prm[1];
    ba.strict.resize(ref.size());
    ba.loose.resize(ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        const double m = std::fabs(X[i] * s) + std::fabs(bv);
        ba.strict[i]   = bv != 0.0 ? 2.0 * EPS32 * m : 0.0;
        ba.loose[i]    = ba.strict[i] + 2.0 * EPS16 * m;
    }
}

const char * TXT_SCALE =
    "strict = 2u (|x s| + |b|) with a bias, 0 without, loose = strict + 2 u16 (|x s| + |b|). Reason: x s + b "
    "is one product and one sum, and a fused multiply-add rounds once where the scalar code rounds "
    "twice: the two results differ by up to one rounding of the product and one of the sum. Without a "
    "bias it is one correctly rounded product.";

enum unary_id { U_SILU, U_SIGMOID, U_GELU, U_EXP, U_SOFTPLUS };

bool build_unary(builder & b) {
    reader & rd = b.rd;
    int64_t  ne[4];
    bool     model;
    ew_shape(b, ne, model);
    static const int ops[] = { U_SILU, U_SIGMOID, U_GELU, U_EXP, U_SOFTPLUS };
    const int op = rd.pick(ops);
    ggml_tensor * a;
    const vspec   v = b.vs(-8.0f, 8.0f);
    if (rd.chance(60)) {
        ggml_tensor * full = b.f32(ne[0] + rd.range(1, 16), ne[1], ne[2], ne[3], v);
        a = ggml_view_4d(b.ctx, full, ne[0], ne[1], ne[2], ne[3], full->nb[1], full->nb[1] * ne[1], full->nb[1] * ne[1] * ne[2], 0);
    } else {
        a = b.f32(ne[0], ne[1], ne[2], ne[3], v);
    }
    ggml_tensor * y = nullptr;
    const char *  name = "";
    switch (op) {
        case U_SILU:     y = ggml_silu(b.ctx, a);     name = "SILU"; break;
        case U_SIGMOID:  y = ggml_sigmoid(b.ctx, a);  name = "SIGMOID"; break;
        case U_GELU:     y = ggml_gelu(b.ctx, a);     name = "GELU"; break;
        case U_EXP:      y = ggml_exp(b.ctx, a);      name = "EXP"; break;
        case U_SOFTPLUS: y = ggml_softplus(b.ctx, a); name = "SOFTPLUS"; break;
    }
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.prm  = { (double) op };
    b.c.desc = fmt("%s [%lld,%lld,%lld,%lld]%s", name, (long long) ne[0], (long long) ne[1], (long long) ne[2],
                   (long long) ne[3], a->view_src ? " view" : "");
    b.c.path = fmt("%s/%s", name, ne[0] < 32 ? "ne0<32" : (ne[0] % 32 == 0 ? "ne0%32==0" : "ne0-odd"));
    return true;
}

void bound_unary(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const std::vector<double> X = logical_values(c, c.outs[o].t->src[0]);
    const int op = (int) c.prm[0];
    ba.strict.resize(ref.size());
    ba.loose.resize(ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        const double x = std::fabs(X[i]);
        const double y = std::fabs((double) ref[i]);
        double       s;
        if (op == U_GELU) {
            // the oracle takes GELU from an f16 table of f16(x) for |x| < 10
            s = X[i] > -10.0 && X[i] < 10.0 ? 2.0 * EPS16 * (y + 1.2 * x) + 1e-30 : 4.0 * EPS32 * y;
        } else if (op == U_EXP) {
            s = 16.0 * EPS32 * (1.0 + x) * y;
        } else {
            s = 16.0 * EPS32 * (2.0 + x) * y;
        }
        ba.strict[i] = s;
        ba.loose[i]  = s + 4.0 * EPS16 * (1.0 + x) * y + (op == U_GELU ? 2.0 * EPS16 * x : 0.0);
    }
}

const char * TXT_UNARY =
    "SILU, SIGMOID, SOFTPLUS: strict = 16u (2 + |x|)|y|; EXP: strict = 16u (1 + |x|)|y|; GELU: strict = "
    "2 u16 (|y| + 1.2|x|) for |x| < 10. loose = strict + 4 u16 (1 + |x|)|y|. Reason: exp(x) has a "
    "condition number |x|, and an f32 exp has an error of a few u (the oracle takes ggml_v_expf for "
    "groups of 4 on x86-64 SSE2). The oracle GELU reads an f16 table at f16(x), thus it carries two "
    "f16 roundings, and gelu' <= 1.13.";

bool build_swiglu(builder & b) {
    reader & rd = b.rd;
    int64_t  ne[4];
    bool     model;
    ew_shape(b, ne, model);
    const bool    split = rd.chance(128);
    const vspec   v     = b.vs(-6.0f, 6.0f);
    ggml_tensor * y;
    if (split) {
        ggml_tensor * g = b.f32(ne[0], ne[1], ne[2], ne[3], v);
        ggml_tensor * u = b.f32(ne[0], ne[1], ne[2], ne[3], b.vs(-6.0f, 6.0f));
        y = ggml_swiglu_split(b.ctx, g, u);
    } else {
        ggml_tensor * a = b.f32(2 * ne[0], ne[1], ne[2], ne[3], v);
        y = ggml_swiglu(b.ctx, a);
    }
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.desc = fmt("SWIGLU %s [%lld,%lld,%lld,%lld]", split ? "split" : "fused", (long long) ne[0], (long long) ne[1],
                   (long long) ne[2], (long long) ne[3]);
    b.c.path = split ? "split" : "fused";
    return true;
}

void bound_swiglu(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * y = c.outs[o].t;
    std::vector<double> G;
    if (y->src[1]) {
        G = logical_values(c, y->src[0]);
    } else {
        const std::vector<double> A = logical_values(c, y->src[0]);
        const int64_t n = y->ne[0];
        G.resize(ref.size());
        for (size_t i = 0; i < ref.size(); i++) {
            G[i] = A[(i / (size_t) n) * 2 * (size_t) n + i % (size_t) n];
        }
    }
    ba.strict.resize(ref.size());
    ba.loose.resize(ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        const double yv = std::fabs((double) ref[i]);
        ba.strict[i] = 16.0 * EPS32 * (3.0 + std::fabs(G[i])) * yv;
        ba.loose[i]  = ba.strict[i] + 4.0 * EPS16 * (1.0 + std::fabs(G[i])) * yv;
    }
}

const char * TXT_SWIGLU =
    "strict = 16u (3 + |g|)|y|, loose = strict + 4 u16 (1 + |g|)|y|, g = the gate input. Reason: the "
    "SILU rule of the gate plus one rounding of the product with the up input.";

bool build_gate_chain(builder & b) {
    reader &      rd    = b.rd;
    const bool    model = rd.chance(128);
    const int64_t H     = model ? (rd.chance(128) ? 16 : 32) : rd.range(1, 64);
    const int64_t T     = rd.chance(100) ? 1 : rd.range(2, 64);
    ggml_tensor * alpha = b.f32(H, T, 1, 1, b.vs(-6.0f, 6.0f));
    ggml_tensor * dt    = b.f32(H, 1, 1, 1, b.vs(-4.0f, 2.0f), leaf_role::WEIGHT);
    ggml_tensor * A     = b.f32(H, 1, 1, 1, b.vs(-16.0f, -0.01f), leaf_role::WEIGHT);
    ggml_tensor * sp    = ggml_softplus(b.ctx, ggml_add(b.ctx, alpha, dt));
    ggml_tensor * y     = ggml_mul(b.ctx, sp, A);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "gate");
    b.c.desc = fmt("gate chain ADD+SOFTPLUS+MUL H=%lld T=%lld%s", (long long) H, (long long) T, model ? " model" : "");
    b.c.path = fmt("H%s", H % 32 == 0 ? "%32==0" : "-odd");
    return true;
}

void bound_gate_chain(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * y   = c.outs[o].t;
    const ggml_tensor * add = y->src[0]->src[0];
    const std::vector<double> AL = logical_values(c, add->src[0]);
    const std::vector<double> DT = logical_values(c, add->src[1]);
    const int64_t H = y->ne[0];
    ba.strict.resize(ref.size());
    ba.loose.resize(ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        const double t = std::fabs(AL[i] + DT[i % (size_t) H]);
        const double a = std::fabs((double) ref[i]);
        ba.strict[i]   = (16.0 * EPS32 * (3.0 + t) + 2.0 * EPS32) * a;
        ba.loose[i]    = ba.strict[i] + 4.0 * EPS16 * (1.0 + t) * a;
    }
}

const char * TXT_GATE =
    "strict = (16u (3 + |t|) + 2u)|y|, loose = strict + 4 u16 (1 + |t|)|y|, t = alpha + dt. Reason: the "
    "softplus rule plus the roundings of the add and of the product with A.";

bool build_cumsum(builder & b) {
    reader &      rd  = b.rd;
    const int64_t ne0 = rd.chance(128) ? rd.range(1, 64) : rd.range(65, 1024);
    const int64_t ne1 = rd.range(1, 16);
    const int64_t ne2 = rd.chance(200) ? 1 : rd.range(2, 3);
    ggml_tensor * a   = b.f32(ne0, ne1, ne2, 1, b.vs(-1.0f, 1.0f));
    ggml_tensor * y   = ggml_cumsum(b.ctx, a);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.desc = fmt("CUMSUM [%lld,%lld,%lld]", (long long) ne0, (long long) ne1, (long long) ne2);
    b.c.path = ne0 <= 64 ? "ne0<=64" : "ne0>64";
    return true;
}

void bound_cumsum(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * y = c.outs[o].t;
    const std::vector<double> X = logical_values(c, y->src[0]);
    const int64_t ne0 = y->ne[0];
    ba.strict.resize(ref.size());
    ba.loose.resize(ref.size());
    for (size_t r = 0; r < ref.size() / (size_t) ne0; r++) {
        double m = 0.0;
        for (int64_t i = 0; i < ne0; i++) {
            const size_t k = r * (size_t) ne0 + (size_t) i;
            m += std::fabs(X[k]);
            ba.strict[k] = 2.0 * gam((double) i + 2) * m;
            ba.loose[k]  = ba.strict[k] + 2.0 * EPS16 * m;
        }
    }
}

const char * TXT_CUMSUM =
    "strict = 2 gamma(i+2) sum_{j<=i}|x_j|, loose = strict + 2 u16 sum|x|. Reason: a prefix sum of i+1 "
    "terms in f32, sequential in the oracle and possibly a scan on a backend.";

// ---------------------------------------------------------------------------------------------
// Data movement: CPY, CONT, SET_ROWS, GET_ROWS, CONCAT

bool build_cpy(builder & b) {
    reader &      rd  = b.rd;
    static const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16 };
    const ggml_type st  = rd.pick(types);
    static const ggml_type dtypes[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_Q8_0 };
    ggml_type     dt  = rd.pick(dtypes);
    int64_t       ne[4];
    ne[0] = rd.chance(64) ? 32 * rd.range(1, 64) : rd.range(1, 300);
    ne[1] = rd.range(1, 64);
    ne[2] = rd.chance(128) ? 1 : rd.range(2, 4);
    ne[3] = 1;
    while (ne[1] > 1 && ne[0] * ne[1] * ne[2] > MAX_ELEMS / 4) {
        ne[1] = (ne[1] + 1) / 2;
    }
    ggml_tensor * src  = b.typed(st, ne[0], ne[1], ne[2], ne[3], b.vs(-3.0f, 3.0f));
    const uint8_t form = rd.u8();
    const char *  fname = "plain";
    if (form < 70) {
        src   = ggml_transpose(b.ctx, src);
        fname = "transposed";
    } else if (form < 120 && ne[2] > 1) {
        src   = ggml_permute(b.ctx, src, 0, 2, 1, 3);
        fname = "permuted";
    }
    // a quantized destination needs whole blocks in each row and contiguous source rows
    if (dt == GGML_TYPE_Q8_0 && (src->ne[0] % 32 != 0 || src->op != GGML_OP_NONE || st != GGML_TYPE_F32)) {
        dt = GGML_TYPE_F16;
    }
    ggml_tensor * y;
    const bool    cont = rd.chance(60) && dt == st;
    if (cont) {
        y = ggml_cont(b.ctx, src);
        ggml_build_forward_expand(b.c.gf, y);
        b.out(y, "y");
    } else {
        ggml_tensor * dst = ggml_new_tensor_4d(b.ctx, dt, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
        leaf l;
        l.t    = dst;
        l.role = leaf_role::STATE;
        l.bytes.assign(ggml_nbytes(dst), 0);
        b.c.leaves.push_back(std::move(l));
        y = ggml_cpy(b.ctx, src, dst);
        ggml_build_forward_expand(b.c.gf, y);
        b.out(dst, "dst");
    }
    b.c.desc = fmt("%s %s->%s [%lld,%lld,%lld] %s", cont ? "CONT" : "CPY", ggml_type_name(st), ggml_type_name(dt),
                   (long long) ne[0], (long long) ne[1], (long long) ne[2], fname);
    b.c.path = fmt("%s->%s/%s", ggml_type_name(st), ggml_type_name(dt), fname);
    return true;
}

void bound_cpy(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba) {
    const ggml_tensor * t = c.outs[o].t;
    if (t->type == GGML_TYPE_Q8_0) {
        // one quantization step of the block: a tie that rounds the other way
        ba.strict.assign(ref.size(), 0.0);
        ba.loose.resize(ref.size());
        for (size_t i = 0; i < ref.size(); i += 32) {
            double a = 0.0;
            for (size_t j = i; j < std::min(ref.size(), i + 32); j++) {
                a = std::max(a, (double) std::fabs(ref[j]));
            }
            for (size_t j = i; j < std::min(ref.size(), i + 32); j++) {
                ba.loose[j] = a / 127.0 * 1.01;
            }
        }
        return;
    }
    bound_exact(c, o, ref, ba);
}

const char * TXT_CPY =
    "strict = 0; loose = 2 u16 |y|, or one Q8_0 step amax32/127 for a Q8_0 destination. Reason: a copy "
    "and a round to nearest even conversion are exact. A Q8_0 destination takes quantize_row_q8_0 "
    "(roundf, ties away from zero in the oracle), and a tie that rounds the other way moves one step.";

bool build_set_rows(builder & b) {
    reader &      rd    = b.rd;
    const bool    model = rd.chance(100);
    static const ggml_type dts[] = { GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0 };
    ggml_type     dt    = rd.pick(dts);
    int64_t       n     = model ? (rd.chance(128) ? 512 : 1024) : (rd.chance(128) ? 32 * rd.range(1, 16) : rd.range(1, 300));
    if (dt == GGML_TYPE_Q8_0 && n % 32 != 0) {
        dt = GGML_TYPE_F16;
    }
    const int64_t n_rows = rd.range(1, 256);
    const int64_t n_set  = rd.range(1, std::min<int64_t>(n_rows, 64));
    const ggml_type it   = rd.chance(128) ? GGML_TYPE_I64 : GGML_TYPE_I32;
    if (n * n_rows > MAX_ELEMS / 2) {
        return false;
    }
    ggml_tensor * dst = b.typed(dt, n, n_rows, 1, 1, b.vs(-1.0f, 1.0f), leaf_role::STATE);
    ggml_tensor * src = b.f32(n, n_set, 1, 1, b.vs(-3.0f, 3.0f, dt == GGML_TYPE_Q8_0));
    ggml_tensor * idx = b.idx_distinct(it, n_set, 1, n_rows);
    ggml_tensor * y   = ggml_set_rows(b.ctx, dst, src, idx);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(dst, "dst");
    b.c.desc = fmt("SET_ROWS dst=%s[%lld,%lld] src=f32[%lld,%lld] idx=%s%s", ggml_type_name(dt), (long long) n,
                   (long long) n_rows, (long long) n, (long long) n_set, ggml_type_name(it), model ? " model" : "");
    b.c.path = fmt("%s/%s", ggml_type_name(dt), ggml_type_name(it));
    return true;
}

bool build_get_rows(builder & b) {
    reader &      rd  = b.rd;
    static const ggml_type sts[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0 };
    ggml_type     st  = rd.pick(sts);
    int64_t       n   = rd.chance(128) ? 32 * rd.range(1, 64) : rd.range(1, 300);
    if (st == GGML_TYPE_Q8_0 && n % 32 != 0) {
        st = GGML_TYPE_F16;
    }
    const int64_t n_rows = rd.range(1, 512);
    const int64_t ne2    = rd.chance(200) ? 1 : rd.range(2, 3);
    const int64_t n_get  = rd.range(1, 64);
    if (n * n_rows * ne2 > MAX_ELEMS / 2) {
        return false;
    }
    ggml_tensor * src = b.typed(st, n, n_rows, ne2, 1, b.vs(-3.0f, 3.0f));
    ggml_tensor * idx = b.i32(n_get, ne2, 1, 1, 0, (int32_t) (n_rows - 1));
    ggml_tensor * y   = ggml_get_rows(b.ctx, src, idx);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.desc = fmt("GET_ROWS src=%s[%lld,%lld,%lld] n_get=%lld", ggml_type_name(st), (long long) n, (long long) n_rows,
                   (long long) ne2, (long long) n_get);
    b.c.path = fmt("%s", ggml_type_name(st));
    return true;
}

bool build_concat(builder & b) {
    reader &      rd  = b.rd;
    const int     dim = (int) rd.range(0, 3);
    int64_t       ne[4];
    ne[0] = rd.range(1, 200);
    ne[1] = rd.range(1, 16);
    ne[2] = rd.range(1, 3);
    ne[3] = rd.range(1, 2);
    int64_t nb_[4] = { ne[0], ne[1], ne[2], ne[3] };
    nb_[dim]       = rd.range(1, dim == 0 ? 64 : 8);
    const bool f16 = rd.chance(48);
    const ggml_type t = f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * a = b.typed(t, ne[0], ne[1], ne[2], ne[3], b.vs(-3.0f, 3.0f));
    ggml_tensor * c;
    if (!f16 && dim == 0 && rd.chance(100)) {
        // the transposed second input of the conv state path
        ggml_tensor * x = b.f32(nb_[1], nb_[0], nb_[2], nb_[3], b.vs(-3.0f, 3.0f));
        c = ggml_transpose(b.ctx, x);
    } else {
        c = b.typed(t, nb_[0], nb_[1], nb_[2], nb_[3], b.vs(-3.0f, 3.0f));
    }
    ggml_tensor * y = ggml_concat(b.ctx, a, c, dim);
    ggml_build_forward_expand(b.c.gf, y);
    b.out(y, "y");
    b.c.desc = fmt("CONCAT %s dim=%d a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld]%s", ggml_type_name(t), dim,
                   (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3], (long long) nb_[0],
                   (long long) nb_[1], (long long) nb_[2], (long long) nb_[3], c->view_src ? " b-transposed" : "");
    b.c.path = fmt("dim%d/%s", dim, ggml_type_name(t));
    return true;
}

} // namespace

const std::vector<kind_def> & kinds() {
    static const std::vector<kind_def> list = {
        { "mul_mat",          "matmul",  build_mul_mat,          bound_mul_mat,     TXT_MUL_MAT },
        { "mul_mat_add",      "matmul",  build_mul_mat_add,      bound_mul_mat_add, TXT_MUL_MAT_ADD },
        { "mul_mat_multi",    "matmul",  build_mul_mat_multi,    bound_mul_mat,     TXT_MUL_MAT },
        { "mul_mat_id",       "matmul",  build_mul_mat_id,       bound_mul_mat_id,  TXT_MUL_MAT },
        { "gated_delta_net",  "gdn",     build_gdn,              bound_gdn,         TXT_GDN },
        { "gdn_state_chain",  "gdn",     build_gdn_state_chain,  bound_gdn_chain,   TXT_GDN_CHAIN },
        { "gdn_conv_chain",   "gdn",     build_gdn_conv_chain,   bound_gdn_conv,    TXT_GDN_CONV },
        { "ssm_conv",         "gdn",     build_ssm_conv,         bound_ssm_conv,    TXT_SSM_CONV },
        { "rope",             "attn",    build_rope,             bound_rope,        TXT_ROPE },
        { "flash_attn_ext",   "attn",    build_flash_attn,       bound_flash_attn,  TXT_FLASH_ATTN },
        { "soft_max",         "attn",    build_soft_max,         bound_soft_max,    TXT_SOFT_MAX },
        { "rms_norm",         "norm",    build_rms_norm,         bound_norm,        TXT_NORM },
        { "l2_norm",          "norm",    build_l2_norm,          bound_norm,        TXT_NORM },
        { "binary",           "elem",    build_binary,           bound_exact,       TXT_EXACT },
        { "scale",            "elem",    build_scale,            bound_scale,       TXT_SCALE },
        { "unary",            "elem",    build_unary,            bound_unary,       TXT_UNARY },
        { "swiglu",           "elem",    build_swiglu,           bound_swiglu,      TXT_SWIGLU },
        { "gate_chain",       "elem",    build_gate_chain,       bound_gate_chain,  TXT_GATE },
        { "cumsum",           "elem",    build_cumsum,           bound_cumsum,      TXT_CUMSUM },
        { "cpy",              "data",    build_cpy,              bound_cpy,         TXT_CPY },
        { "set_rows",         "data",    build_set_rows,         bound_cpy,         TXT_CPY },
        { "get_rows",         "data",    build_get_rows,         bound_exact,       TXT_EXACT },
        { "concat",           "data",    build_concat,           bound_exact,       TXT_EXACT },
        // not a fuzz group: the fixed model shapes of the phone runs (gen --enumerate mm_model:N)
        { "mm_model",         "shapes",  build_mm_model,         bound_mul_mat,     TXT_MUL_MAT },
    };
    return list;
}

int kind_index(const std::string & name) {
    const auto & K = kinds();
    for (size_t i = 0; i < K.size(); i++) {
        if (name == K[i].name) {
            return (int) i;
        }
    }
    return -1;
}

std::vector<int> group_kinds(const std::string & group) {
    std::vector<int> r;
    const auto &     K = kinds();
    for (size_t i = 0; i < K.size(); i++) {
        // "all" means the fuzz groups: the group "shapes" joins only by its own name
        const bool in_all = group == "all" && std::strcmp(K[i].group, "shapes") != 0;
        if (in_all || group == K[i].group || group == K[i].name) {
            r.push_back((int) i);
        }
    }
    return r;
}

std::vector<std::string> group_names() {
    return { "matmul", "gdn", "attn", "norm", "elem", "data" };
}

} // namespace fo
