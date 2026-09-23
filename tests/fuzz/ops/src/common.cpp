// Shared items of the op fuzzer. Refer to common.h.

#include "common.h"

#include <cstdio>

namespace fo {

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t       man  = h & 0x3ffu;
    uint32_t       bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            // A subnormal half is a normal float: shift the mantissa until the implicit bit shows.
            int e = -14;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                e--;
            }
            man &= 0x3ffu;
            bits = sign | ((uint32_t) (e + 127) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    return bits_f32(bits);
}

uint16_t f32_to_f16(float f) {
    const uint32_t x    = f32_bits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp  = (x >> 23) & 0xffu;
    uint32_t       man  = x & 0x7fffffu;
    if (exp == 0xff) {
        return (uint16_t) (man ? (sign | 0x7e00u | (man >> 13)) : (sign | 0x7c00u));
    }
    const int e = (int) exp - 127 + 15;
    if (e >= 31) {
        return (uint16_t) (sign | 0x7c00u);
    }
    if (e <= 0) {
        // The result is a half subnormal or zero. A value below 2^-25 rounds to zero.
        if (e < -10) {
            return (uint16_t) sign;
        }
        man |= 0x800000u;
        const int      shift = 14 - e;
        uint32_t       half  = man >> shift;
        const uint32_t rem   = man & ((1u << shift) - 1u);
        const uint32_t mid   = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) {
            half++;
        }
        return (uint16_t) (sign | half);
    }
    uint32_t       half = ((uint32_t) e << 10) | (man >> 13);
    const uint32_t rem  = man & 0x1fffu;
    // A carry out of the mantissa moves into the exponent, and 0x7bff + 1 gives the infinity.
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) {
        half++;
    }
    return (uint16_t) (sign | half);
}

// Return 2^e as a float for e in [-126, 127]. The value is exact.
static float pow2f(int e) {
    if (e < -126) {
        e = -126;
    }
    if (e > 127) {
        e = 127;
    }
    return bits_f32((uint32_t) (e + 127) << 23);
}

// Return one special value of the set.
static float special_value(rng & r, uint8_t set) {
    static const uint32_t zero[] = { 0x00000000u, 0x80000000u };
    static const uint32_t inf[]  = { 0x7f800000u, 0xff800000u };
    static const uint32_t nan[]  = { 0x7fc00000u, 0xffc00001u, 0x7fc12345u };
    static const uint32_t subn[] = { 0x00000001u, 0x80000001u, 0x007fffffu, 0x00800000u, 0x80800000u, 0x00400000u };
    static const uint32_t huge[] = { 0x7f7fffffu, 0xff7fffffu, 0x7f61b1e6u, 0x7149f2cau, 0xf149f2cau };
    // 65504, -65504, 65520, 65519.996, 2^-24, 2^-25, 2^-14, 1e5
    static const uint32_t f16[]  = { 0x477fe000u, 0xc77fe000u, 0x477ff000u, 0x477fefffu,
                                     0x33800000u, 0x33000000u, 0x38800000u, 0x47c35000u };
    const uint32_t * lists[6] = { zero, inf, nan, subn, huge, f16 };
    const int        sizes[6] = { 2, 2, 3, 6, 5, 8 };
    int              avail[6];
    int              n_avail = 0;
    for (int k = 0; k < 6; k++) {
        if (set & (1u << k)) {
            avail[n_avail++] = k;
        }
    }
    if (n_avail == 0) {
        return 0.0f;
    }
    const int k = avail[r.range(0, n_avail - 1)];
    return bits_f32(lists[k][r.range(0, sizes[k] - 1)]);
}

void gen_f32(float * dst, int64_t n, int64_t ne0, const vspec & vs, uint64_t seed) {
    rng         r(seed);
    const float width = vs.hi - vs.lo;
    float       row_const = 0.0f;
    int         tie_e     = 0;
    int64_t     tie_pos   = 0;
    if (ne0 <= 0) {
        ne0 = 1;
    }
    for (int64_t i = 0; i < n; i++) {
        const int64_t col = i % ne0;
        float         v   = 0.0f;
        switch (vs.prof) {
            case VP_UNIFORM:
                v = vs.lo + width * r.unit();
                break;
            case VP_LOGUNI: {
                const int      e    = (int) r.range(vs.emin, vs.emax);
                const uint32_t man  = r.u32() & 0x7fffffu;
                const uint32_t sign = (r.u32() & 1u) << 31;
                int            eb   = e + 127;
                eb                  = eb < 1 ? 1 : (eb > 254 ? 254 : eb);
                v                   = bits_f32(sign | ((uint32_t) eb << 23) | man);
                break;
            }
            case VP_TIES_Q8: {
                if (col % 32 == 0) {
                    tie_e   = (int) r.range(-12, 8);
                    tie_pos = r.range(0, 31);
                }
                if (col % 32 == tie_pos) {
                    v = (r.u32() & 1u) ? -127.0f : 127.0f;
                    v = v * pow2f(tie_e);
                } else {
                    const int q = (int) r.range(-127, 126);
                    v           = (float) (2 * q + 1) * pow2f(tie_e - 1);
                }
                break;
            }
            case VP_SPARSE:
                if ((r.u32() & 7u) != 0) {
                    v = (r.u32() & 1u) ? -0.0f : 0.0f;
                } else {
                    v = vs.lo + width * r.unit();
                }
                break;
            case VP_CONST:
                if (col == 0) {
                    row_const = vs.lo + width * r.unit();
                }
                v = row_const;
                break;
            case VP_INT:
                v = (float) r.range((int64_t) vs.lo, (int64_t) vs.hi);
                break;
        }
        if (vs.special > 0 && (r.u32() & 0xffffu) < vs.special) {
            v = special_value(r, vs.sp_set);
        }
        dst[i] = v;
    }
}

const char * vprof_name(vprof p) {
    switch (p) {
        case VP_UNIFORM: return "uniform";
        case VP_LOGUNI:  return "loguni";
        case VP_TIES_Q8: return "q8ties";
        case VP_SPARSE:  return "sparse";
        case VP_CONST:   return "const";
        case VP_INT:     return "int";
    }
    return "?";
}

uint64_t fnv1a(const uint8_t * data, size_t size) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < size; i++) {
        h ^= data[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

std::string hex64(uint64_t v) {
    char buf[20];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) v);
    return buf;
}

} // namespace fo
