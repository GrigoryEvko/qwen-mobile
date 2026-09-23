// The case decoder and the input generators. Refer to case.h.

#include "case.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace fo {

built_case::~built_case() {
    if (ctx) {
        ggml_free(ctx);
    }
}

// Return the seed of leaf number `i` of a case.
static uint64_t leaf_seed(uint64_t seed, int i) {
    return seed ^ (0x9e3779b97f4a7c15ull * (uint64_t) (i + 1)) ^ 0x5851f42d4c957f2dull;
}

// FUZZ_OPS_TAME=1 keeps the special values (Inf, NaN, subnormal, huge), the point injections of
// raw bits, the very wide magnitudes and the fully masked rows out of the inputs, and it reads
// fewer bytes for the form of a case. Thus the same bytes can give a different case. The suites
// do not set it: the value asserts of the ggml CPU ops run only with GGML_CPU_VALUE_ASSERTS, and
// a debug build accepts the special values. The switch remains for the regression inputs with the
// prefix "tame-", which exist only with this decode. The oracle process inherits the variable, thus
// the two sides decode the same case. A case of a pack or of a regression input can also carry the
// switch itself (FORCED_TAME in its forced kind, refer to case.h): build_case sets g_case_tame
// for the time of the decode.
static int g_case_tame = -1;

static bool tame_values() {
    static const bool env_tame = [] {
        const char * s = std::getenv("FUZZ_OPS_TAME");
        return s != nullptr && s[0] == '1';
    }();
    return g_case_tame >= 0 ? g_case_tame == 1 : env_tame;
}

vspec builder::vs(float lo, float hi, bool allow_ties) {
    vspec v;
    v.prof = VP_UNIFORM;
    v.lo   = lo;
    v.hi   = hi;
    const uint8_t m = rd.u8();
    if (allow_ties && m < 24) {
        // Exact ties of the Q8_0 rounding of the activations: in the model limits too.
        v.prof = VP_TIES_Q8;
        return v;
    }
    if (!wild || m < 120) {
        return v;
    }
    if (tame_values() && (m < 160 || m >= 186)) {
        // the wide and the special distributions: keep the default range
        return v;
    }
    if (m < 160) {
        v.prof = VP_LOGUNI;
        v.emin = (int8_t) -rd.range(0, 40);
        v.emax = (int8_t) rd.range(-20, 40);
        if (v.emax < v.emin) {
            std::swap(v.emin, v.emax);
        }
        return v;
    }
    if (m < 176) {
        v.prof = VP_SPARSE;
        return v;
    }
    if (m < 186) {
        v.prof = VP_CONST;
        return v;
    }
    v.special = (uint16_t) rd.range(1, 4096);
    v.sp_set  = (uint8_t) (rd.u8() & SP_ALL);
    if (v.sp_set == 0) {
        v.sp_set = SP_ALL;
    }
    c.special = true;
    return v;
}

// Add a leaf to the case and give the tensor a name.
static leaf & add_leaf(builder & b, ggml_tensor * t, leaf_role role) {
    leaf l;
    l.t    = t;
    l.role = role;
    l.bytes.assign(ggml_nbytes(t), 0);
    b.c.leaves.push_back(std::move(l));
    ggml_format_name(t, "in%d", (int) b.c.leaves.size() - 1);
    return b.c.leaves.back();
}

void builder::inject(leaf & l) {
    if (!wild || tame_values() || !rd.chance(48)) {
        return;
    }
    const int64_t n     = ggml_nelements(l.t);
    const int     count = (int) rd.range(1, 3);
    for (int k = 0; k < count && n > 0; k++) {
        const int64_t  pos  = (int64_t) (rd.u32() % (uint64_t) n);
        const uint32_t bits = rd.u32();
        if (l.t->type == GGML_TYPE_F32) {
            std::memcpy(l.bytes.data() + pos * 4, &bits, 4);
            const float f = bits_f32(bits);
            if (!std::isfinite(f) || (f != 0.0f && std::fabs(f) < 1.17549435e-38f)) {
                c.special = true;
            }
        } else if (l.t->type == GGML_TYPE_F16) {
            const uint16_t h = (uint16_t) bits;
            std::memcpy(l.bytes.data() + pos * 2, &h, 2);
            if ((h & 0x7c00u) == 0x7c00u || ((h & 0x7c00u) == 0 && (h & 0x3ffu) != 0)) {
                c.special = true;
            }
        }
    }
}

ggml_tensor * builder::f32(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, const vspec & v, leaf_role role) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    leaf &        l = add_leaf(*this, t, role);
    gen_f32((float *) l.bytes.data(), ggml_nelements(t), ne0, v, leaf_seed(seed, n_leaf++));
    inject(l);
    return t;
}

ggml_tensor * builder::f16(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, const vspec & v, leaf_role role) {
    ggml_tensor *      t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, ne0, ne1, ne2, ne3);
    leaf &             l = add_leaf(*this, t, role);
    const int64_t      n = ggml_nelements(t);
    std::vector<float> tmp(n);
    gen_f32(tmp.data(), n, ne0, v, leaf_seed(seed, n_leaf++));
    for (int64_t i = 0; i < n; i++) {
        const uint16_t h = f32_to_f16(tmp[i]);
        std::memcpy(l.bytes.data() + i * 2, &h, 2);
    }
    inject(l);
    return t;
}

ggml_tensor * builder::quant(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int emin, int emax,
                             leaf_role role) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
    leaf &        l = add_leaf(*this, t, role);
    rng           r(leaf_seed(seed, n_leaf++));
    // In a wild case the bytes select rare block forms: special scales, -128 codes, zero blocks.
    const uint8_t form = wild && !tame_values() ? rd.u8() : 0;
    const bool    q8   = type == GGML_TYPE_Q8_0;
    const size_t  bs   = q8 ? 34 : 18;
    const int64_t nblk = ggml_nelements(t) / 32;
    for (int64_t ib = 0; ib < nblk; ib++) {
        uint8_t * blk = l.bytes.data() + ib * bs;
        // the scale: a normal or subnormal f16 with the exponent in [emin, emax]
        const int e    = (int) r.range(emin, emax);
        uint16_t  d    = 0;
        const int eb   = e + 15;
        const uint16_t man = (uint16_t) (r.u32() & 0x3ffu);
        if (eb >= 1) {
            d = (uint16_t) ((eb > 30 ? 30 : eb) << 10) | man;
        } else {
            d = (uint16_t) (man == 0 ? 1 : man);
        }
        if (!q8 && (r.u32() & 1u)) {
            d |= 0x8000u; // Q4_0 has a negative scale for a positive block maximum
        }
        if (form >= 200) {
            const uint32_t pick = r.u32() & 63u;
            if (pick == 0) {
                d = 0;
            } else if (pick == 1) {
                d = 0x7c00u;
            } else if (pick == 2) {
                d = 0x7e00u;
            } else if (pick == 3) {
                d = 0x7bffu;
            } else if (pick == 4 && q8) {
                d |= 0x8000u;
            }
            if (pick <= 3) {
                c.special = true;
            }
        }
        std::memcpy(blk, &d, 2);
        if (q8) {
            const uint32_t bform = form >= 200 ? (r.u32() & 31u) : 99u;
            for (int j = 0; j < 32; j++) {
                int v = (int) r.range(-127, 127);
                if (bform == 0) {
                    v = 0;
                } else if (bform == 1) {
                    v = (j & 1) ? 127 : -127;
                } else if (bform == 2 && (r.u32() & 3u) == 0) {
                    v = -128;
                }
                blk[2 + j] = (uint8_t) (int8_t) v;
            }
        } else {
            for (int j = 0; j < 16; j++) {
                blk[2 + j] = (uint8_t) (r.u32() & 0xffu);
            }
        }
    }
    return t;
}

ggml_tensor * builder::typed(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, const vspec & v,
                             leaf_role role) {
    switch (type) {
        case GGML_TYPE_F32: return f32(ne0, ne1, ne2, ne3, v, role);
        case GGML_TYPE_F16: return f16(ne0, ne1, ne2, ne3, v, role);
        default: {
            int emin = -12;
            int emax = -5;
            if (wild && rd.chance(64)) {
                emin = (int) rd.range(-24, 0);
                emax = (int) rd.range(emin, 15);
            }
            return quant(type, ne0, ne1, ne2, ne3, emin, emax, role);
        }
    }
}

ggml_tensor * builder::i32(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int32_t lo, int32_t hi) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne0, ne1, ne2, ne3);
    leaf &        l = add_leaf(*this, t, leaf_role::DATA);
    rng           r(leaf_seed(seed, n_leaf++));
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; i++) {
        const int32_t v = (int32_t) r.range(lo, hi);
        std::memcpy(l.bytes.data() + i * 4, &v, 4);
    }
    return t;
}

ggml_tensor * builder::idx_distinct(ggml_type type, int64_t ne0, int64_t ne1, int64_t n_max) {
    ggml_tensor *        t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    leaf &               l = add_leaf(*this, t, leaf_role::DATA);
    rng                  r(leaf_seed(seed, n_leaf++));
    std::vector<int64_t> perm((size_t) n_max);
    const size_t         es = ggml_type_size(type);
    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t i = 0; i < n_max; i++) {
            perm[i] = i;
        }
        // a partial Fisher-Yates shuffle of the first ne0 items, O(ne0)
        for (int64_t i = 0; i < ne0; i++) {
            const int64_t j = r.range(i, n_max - 1);
            std::swap(perm[i], perm[j]);
            uint8_t * dst = l.bytes.data() + (row * ne0 + i) * es;
            if (type == GGML_TYPE_I64) {
                std::memcpy(dst, &perm[i], 8);
            } else {
                const int32_t v = (int32_t) perm[i];
                std::memcpy(dst, &v, 4);
            }
        }
    }
    return t;
}

ggml_tensor * builder::mask_f16(int64_t n_kv, int64_t n_q, int64_t ne2, int64_t ne3, bool causal) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, ne2, ne3);
    leaf &        l = add_leaf(*this, t, leaf_role::DATA);
    rng           r(leaf_seed(seed, n_leaf++));
    const uint16_t ninf = 0xfc00u;
    // In a wild case a row can be fully masked: the output is then 0 on the CPU.
    const bool allow_empty = wild && !tame_values() && rd.chance(32);
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t iq = 0; iq < n_q; iq++) {
                uint16_t * row = (uint16_t *) (l.bytes.data() + ((i3 * ne2 + i2) * n_q + iq) * n_kv * 2);
                int64_t    visible = 0;
                for (int64_t j = 0; j < n_kv; j++) {
                    uint16_t v;
                    if (causal) {
                        // query iq sees the kv positions up to n_kv - n_q + iq
                        v = (j <= n_kv - n_q + iq) ? 0 : ninf;
                    } else {
                        const uint32_t pick = r.u32() & 15u;
                        v = pick < 3 ? ninf : f32_to_f16(-1.0f + 2.0f * r.unit());
                    }
                    if (v != ninf) {
                        visible++;
                    }
                    row[j] = v;
                }
                if (visible == 0 && !allow_empty) {
                    row[r.range(0, n_kv - 1)] = 0;
                }
            }
        }
    }
    return t;
}

output & builder::out(ggml_tensor * t, const char * label, int bound) {
    output o;
    o.t     = t;
    o.label = label;
    o.bound = bound;
    c.outs.push_back(o);
    return c.outs.back();
}

leaf * builder::find(ggml_tensor * t) {
    for (auto & l : c.leaves) {
        if (l.t == t) {
            return &l;
        }
    }
    return nullptr;
}

std::vector<float> raw_to_f32(ggml_type type, const uint8_t * data, int64_t n) {
    std::vector<float> v((size_t) n);
    switch (type) {
        case GGML_TYPE_F32:
            std::memcpy(v.data(), data, (size_t) n * 4);
            break;
        case GGML_TYPE_F16:
            for (int64_t i = 0; i < n; i++) {
                uint16_t h;
                std::memcpy(&h, data + i * 2, 2);
                v[i] = f16_to_f32(h);
            }
            break;
        case GGML_TYPE_I32:
            for (int64_t i = 0; i < n; i++) {
                int32_t x;
                std::memcpy(&x, data + i * 4, 4);
                v[i] = (float) x;
            }
            break;
        case GGML_TYPE_Q8_0:
            for (int64_t i = 0; i < n; i++) {
                const uint8_t * blk = data + (i / 32) * 34;
                uint16_t        d;
                std::memcpy(&d, blk, 2);
                v[i] = f16_to_f32(d) * (float) (int8_t) blk[2 + i % 32];
            }
            break;
        case GGML_TYPE_Q4_0:
            for (int64_t i = 0; i < n; i++) {
                const uint8_t * blk = data + (i / 32) * 18;
                uint16_t        d;
                std::memcpy(&d, blk, 2);
                const int j = (int) (i % 32);
                const int q = j < 16 ? (blk[2 + j] & 0x0f) : (blk[2 + j - 16] >> 4);
                v[i] = f16_to_f32(d) * (float) (q - 8);
            }
            break;
        default:
            break;
    }
    return v;
}

// Return the value of element i0 of a row of the type. The time is O(1).
static double elem(ggml_type type, const uint8_t * row, int64_t i0, size_t nb0) {
    switch (type) {
        case GGML_TYPE_F32: {
            float f;
            std::memcpy(&f, row + i0 * nb0, 4);
            return f;
        }
        case GGML_TYPE_F16: {
            uint16_t h;
            std::memcpy(&h, row + i0 * nb0, 2);
            return f16_to_f32(h);
        }
        case GGML_TYPE_I32: {
            int32_t x;
            std::memcpy(&x, row + i0 * nb0, 4);
            return x;
        }
        case GGML_TYPE_I64: {
            int64_t x;
            std::memcpy(&x, row + i0 * nb0, 8);
            return (double) x;
        }
        case GGML_TYPE_Q8_0: {
            const uint8_t * blk = row + (i0 / 32) * 34;
            uint16_t        d;
            std::memcpy(&d, blk, 2);
            return (double) f16_to_f32(d) * (double) (int8_t) blk[2 + i0 % 32];
        }
        case GGML_TYPE_Q4_0: {
            const uint8_t * blk = row + (i0 / 32) * 18;
            uint16_t        d;
            std::memcpy(&d, blk, 2);
            const int j = (int) (i0 % 32);
            const int q = j < 16 ? (blk[2 + j] & 0x0f) : (blk[2 + j - 16] >> 4);
            return (double) f16_to_f32(d) * (double) (q - 8);
        }
        default:
            return 0.0;
    }
}

std::vector<double> logical_values(const built_case & c, const ggml_tensor * t) {
    const ggml_tensor * root = t->view_src ? t->view_src : t;
    const size_t        offs = t->view_src ? t->view_offs : 0;
    const leaf *        lf   = nullptr;
    for (const auto & l : c.leaves) {
        if (l.t == root) {
            lf = &l;
        }
    }
    if (lf == nullptr) {
        return {};
    }
    const int64_t       n = ggml_nelements(t);
    std::vector<double> v((size_t) n);
    int64_t             k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                const size_t o = offs + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                if (o >= lf->bytes.size()) {
                    return {};
                }
                const uint8_t * row = lf->bytes.data() + o;
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    v[k++] = elem(t->type, row, i0, t->nb[0]);
                }
            }
        }
    }
    return v;
}

bool build_case(const uint8_t * data, size_t size, int forced, size_t max_bytes, built_case & c) {
    // The tame flag of the case holds for this decode only. The decode is single-threaded.
    struct tame_scope {
        explicit tame_scope(int v) { g_case_tame = v; }
        ~tame_scope() { g_case_tame = -1; }
    } scope(forced >= 0 && (forced & FORCED_TAME) ? 1 : -1);
    if (forced >= 0) {
        forced &= ~FORCED_TAME;
    }
    reader        rd(data, size);
    const auto &  K    = kinds();
    const uint8_t k0   = rd.u8();
    if (forced >= (int) K.size()) {
        return false;
    }
    const int     kid  = forced >= 0 ? forced : (int) (k0 % K.size());
    const uint8_t opt  = rd.u8();
    const uint64_t seed = rd.u64();

    c.kind       = &K[kid];
    c.kind_id    = kid;
    c.n_threads  = 1 + (opt & 3);
    c.cpu_repack = ((opt >> 2) & 1) != 0;
    const bool wild = ((opt >> 3) & 7) >= 5;

    const size_t ctx_size = 640 * ggml_tensor_overhead() + ggml_graph_overhead_custom(1024, false);
    ggml_init_params ip   = { ctx_size, nullptr, true };
    c.ctx                 = ggml_init(ip);
    c.gf                  = ggml_new_graph_custom(c.ctx, 1024, false);

    builder b(rd, c, c.ctx, seed, wild);
    if (!K[kid].build(b)) {
        return false;
    }
    const int n_nodes = ggml_graph_n_nodes(c.gf);
    if (c.outs.empty() || n_nodes == 0) {
        return false;
    }
    size_t total = 0;
    for (const auto & l : c.leaves) {
        total += l.bytes.size();
    }
    for (int i = 0; i < n_nodes; i++) {
        const ggml_tensor * n = ggml_graph_node(c.gf, i);
        if (n->view_src == nullptr) {
            total += ggml_nbytes(n);
        }
    }
    return total <= max_bytes;
}

} // namespace fo
