// Shared items of the op fuzzer: the byte reader, the value generator, the f16 conversions.
//
// Every function here gives the same bits on x86_64 and on arm64. The value generator uses integer
// operations and IEEE float operations only, and the build compiles this code with
// -ffp-contract=off and without fast math. Thus the phone and the host make the same input bytes
// from the same case bytes.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace fo {

// The unit roundoff of f32 and of f16.
constexpr double EPS32 = 5.9604644775390625e-08; // 2^-24
constexpr double EPS16 = 4.8828125e-04;          // 2^-11

// Return the bits of a float.
inline uint32_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// Return the float of the bits.
inline float bits_f32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Convert an IEEE half to a float. The conversion is exact for every input, NaN included.
float f16_to_f32(uint16_t h);

// Convert a float to an IEEE half with round to nearest even. NaN gives a quiet NaN with the sign kept,
// and an overflow gives an infinity.
uint16_t f32_to_f16(float f);

// A splitmix64 generator. It is small, it has no state other than one word, and it gives the same
// sequence on every platform.
struct rng {
    uint64_t s;

    explicit rng(uint64_t seed) : s(seed) {}

    // Return the next 64 random bits.
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    // Return 32 random bits.
    uint32_t u32() { return (uint32_t) (next() >> 32); }

    // Return an integer in [lo, hi]. The caller makes sure that lo <= hi.
    int64_t range(int64_t lo, int64_t hi) {
        const uint64_t span = (uint64_t) (hi - lo) + 1;
        return lo + (int64_t) (span == 0 ? next() : next() % span);
    }

    // Return a float in [0, 1) with 24 random bits. The value is exact.
    float unit() { return (float) (u32() >> 8) * (1.0f / 16777216.0f); }
};

// A reader of the case bytes, in the manner of the FuzzedDataProvider of libFuzzer. When the bytes
// run out, each read gives zero. Thus every byte string is a valid case.
struct reader {
    const uint8_t * p;
    size_t          n;
    size_t          i = 0;

    reader(const uint8_t * data, size_t size) : p(data), n(size) {}

    // Return the next byte, or 0 after the end.
    uint8_t u8() { return i < n ? p[i++] : 0; }

    // Return the next 16 bits, little endian.
    uint16_t u16() {
        const uint16_t lo = u8();
        return (uint16_t) (lo | ((uint16_t) u8() << 8));
    }

    // Return the next 32 bits, little endian.
    uint32_t u32() {
        const uint32_t lo = u16();
        return lo | ((uint32_t) u16() << 16);
    }

    // Return the next 64 bits, little endian.
    uint64_t u64() {
        const uint64_t lo = u32();
        return lo | ((uint64_t) u32() << 32);
    }

    // Return an integer in [lo, hi] from one or two bytes. The caller makes sure that lo <= hi.
    int64_t range(int64_t lo, int64_t hi) {
        const uint64_t span = (uint64_t) (hi - lo) + 1;
        const uint64_t v    = span <= 256 ? u8() : span <= 65536 ? u16() : u32();
        return lo + (int64_t) (v % span);
    }

    // Return true with the probability num/256.
    bool chance(int num) { return (int) u8() < num; }

    // Return one item of a list.
    template <typename T, size_t N> T pick(const T (&items)[N]) { return items[range(0, (int64_t) N - 1)]; }
};

// The distribution of the values of one float input tensor.
enum vprof : uint8_t {
    VP_UNIFORM = 0, // uniform in [lo, hi]
    VP_LOGUNI  = 1, // random sign, |x| = 2^e * (1 + m) with e in [emin, emax]
    VP_TIES_Q8 = 2, // blocks of 32: amax = 127 * 2^e, the other values are exact ties (q + 0.5) * 2^e
    VP_SPARSE  = 3, // 7 of 8 values are +0 or -0, the others uniform in [lo, hi]
    VP_CONST   = 4, // one value for each row of ne0 elements
    VP_INT     = 5, // integers in [lo, hi], exact
};

// The special values that the generator can put into a float tensor.
enum vspecial : uint8_t {
    SP_ZERO  = 1,   // +0 and -0
    SP_INF   = 2,   // +Inf and -Inf
    SP_NAN   = 4,   // quiet NaN
    SP_SUBN  = 8,   // f32 subnormals and the smallest normals
    SP_HUGE  = 16,  // values near FLT_MAX
    SP_F16   = 32,  // values at the f16 limits: 65504, 65520, 2^-24, 2^-14
    SP_ALL   = 63,
};

// The value spec of one float tensor.
struct vspec {
    vprof    prof    = VP_UNIFORM;
    float    lo      = -1.0f;
    float    hi      = 1.0f;
    int8_t   emin    = -8;
    int8_t   emax    = 4;
    uint16_t special = 0;       // expected number of special values in each 65536 values
    uint8_t  sp_set  = SP_ALL;  // the kinds of special values
};

// Make the float values of n elements with rows of ne0 elements. The same seed gives the same values
// on every platform. The time is O(n).
void gen_f32(float * dst, int64_t n, int64_t ne0, const vspec & vs, uint64_t seed);

// Return a readable name of a value profile.
const char * vprof_name(vprof p);

// Return the FNV-1a hash of a byte string. The finding files use it as a name.
uint64_t fnv1a(const uint8_t * data, size_t size);

// Return the hexadecimal text of a 64-bit value.
std::string hex64(uint64_t v);

} // namespace fo
