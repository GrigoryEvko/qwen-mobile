// The check of the f32-to-f16 conversions of ggml with the float flags of the profile. It checks
// three functions: ggml_compute_fp32_to_fp16 (ggml-impl.h, inline, thus compiled here),
// ggml_fp32_to_fp16_row (ggml-base) and ggml_cpu_fp32_to_fp16 (ggml-cpu). This file gets the flags
// of the profile (-ffp-model=fast), and not the strict flags of the harness, because a defect of a
// fast-math build shows only with these flags.
//
// The reference is a table of the correct f16 bits of each listed value (round to nearest even).
// On an x86 host with F16C, the check also compares the listed values and a sweep of the f32 bit
// patterns with the hardware conversion. A NaN input must give a NaN.
//
//   ops_f16_check     The exit code is 0 if each result agrees, and 1 if one result does not agree.

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#    include <immintrin.h>
#    define F16_CHECK_X86 1
#else
#    define F16_CHECK_X86 0
#endif

namespace {

// One value and its correct f16 bits.
struct f16_case {
    float    value;
    uint16_t bits;
};

// The f16 range stops at 65504 (0x7bff). A value with |f| >= 65520 rounds to inf, and a value with
// |f| < 65520 rounds to 65504 or less. 65519.99f is the f32 value 65519.9921875.
const f16_case CASES[] = {
    { 0.0f, 0x0000 },          { -0.0f, 0x8000 },          { 1.0f, 0x3c00 },
    { -1.0f, 0xbc00 },         { 65504.0f, 0x7bff },       { -65504.0f, 0xfbff },
    { 65519.99f, 0x7bff },     { -65519.99f, 0xfbff },     { 65520.0f, 0x7c00 },
    { -65520.0f, 0xfc00 },     { 70000.0f, 0x7c00 },       { -70000.0f, 0xfc00 },
    { 1e6f, 0x7c00 },          { -1e6f, 0xfc00 },          { 2.7e11f, 0x7c00 },
    { -2.7e11f, 0xfc00 },      { 3.0e38f, 0x7c00 },        { -3.0e38f, 0xfc00 },
    { INFINITY, 0x7c00 },      { -INFINITY, 0xfc00 },      { 0x1p-14f, 0x0400 },
    { 0x1p-24f, 0x0001 },      { -0x1p-24f, 0x8001 },      { 0x1p-25f, 0x0000 },
};

// The number of differences that the check prints. It counts all of them.
constexpr int PRINT_MAX = 20;

int g_diffs = 0;

/// Return true if the f16 bits are a NaN.
bool f16_is_nan(uint16_t h) {
    return (h & 0x7c00u) == 0x7c00u && (h & 0x03ffu) != 0;
}

/// Return true if the f32 value is a NaN, from its bits (the fast-math flags can remove a float test).
bool f32_is_nan(float f) {
    uint32_t w;
    std::memcpy(&w, &f, sizeof(w));
    return (w & 0x7fffffffu) > 0x7f800000u;
}

/// Record one result: a NaN input must give a NaN, each other input must give the expected bits.
void expect(const char * func, float value, uint16_t got, uint16_t want) {
    const bool ok = f32_is_nan(value) ? f16_is_nan(got) : got == want;
    if (ok) {
        return;
    }
    if (g_diffs < PRINT_MAX) {
        uint32_t w;
        std::memcpy(&w, &value, sizeof(w));
        std::printf("f16_check: %s(%.9g [0x%08x]) = 0x%04x, expected 0x%04x\n", func, (double) value, w, got, want);
    }
    g_diffs++;
}

/// Convert one value with ggml_compute_fp32_to_fp16. The volatile copy keeps the compiler from a
/// conversion at compile time, thus the check runs the code of the profile.
uint16_t inline_conv(float value) {
    volatile float v = value;
    return ggml_compute_fp32_to_fp16(v);
}

#if F16_CHECK_X86
/// Return true if the host has F16C.
bool have_f16c() {
    return __builtin_cpu_supports("f16c");
}

/// The hardware conversion (round to nearest even), the reference on an x86 host with F16C.
__attribute__((target("f16c"))) uint16_t hw_conv(float value) {
    return (uint16_t) _cvtss_sh(value, 0);
}
#endif

/// Check the three functions with the values of one row. The row functions get the full row, thus
/// their SIMD blocks and their scalar tail both see the values. Complexity: O(n).
void check_row(const std::vector<float> & row, const std::vector<uint16_t> & want) {
    const int64_t            n = (int64_t) row.size();
    std::vector<ggml_fp16_t> out_base(row.size()), out_cpu(row.size());
    ggml_fp32_to_fp16_row(row.data(), out_base.data(), n);
    ggml_cpu_fp32_to_fp16(row.data(), out_cpu.data(), n);
    for (size_t i = 0; i < row.size(); i++) {
        expect("ggml_compute_fp32_to_fp16", row[i], inline_conv(row[i]), want[i]);
        expect("ggml_fp32_to_fp16_row", row[i], out_base[i], want[i]);
        expect("ggml_cpu_fp32_to_fp16", row[i], out_cpu[i], want[i]);
    }
}

/// The listed values and a NaN, in rows of different lengths (37 values is not a multiple of 4, 8
/// or 16), with the table as the reference, and with the hardware as a second reference.
int check_cases(bool hw) {
    std::vector<float>    row;
    std::vector<uint16_t> want;
    for (int rep = 0; rep < 2; rep++) {
        for (const f16_case & c : CASES) {
            row.push_back(c.value);
            want.push_back(c.bits);
        }
    }
    row.push_back(std::numeric_limits<float>::quiet_NaN());
    want.push_back(0x7e00);
    row.push_back(-std::numeric_limits<float>::quiet_NaN());
    want.push_back(0xfe00);
    for (size_t len : { (size_t) 1, (size_t) 7, (size_t) 37, row.size() }) {
        const size_t n = len < row.size() ? len : row.size();
        check_row(std::vector<float>(row.begin(), row.begin() + (long) n),
                  std::vector<uint16_t>(want.begin(), want.begin() + (long) n));
    }
#if F16_CHECK_X86
    if (hw) {
        for (size_t i = 0; i < row.size(); i++) {
            expect("hardware (table check)", row[i], hw_conv(row[i]), want[i]);
        }
    }
#else
    (void) hw;
#endif
    return (int) row.size();
}

#if F16_CHECK_X86
/// A sweep of the f32 bit patterns (a prime stride over the full range, and each pattern near
/// 65520 and near the f16 subnormal limit, with the two signs) against the hardware conversion.
/// Complexity: O(2^32 / stride + the two near ranges).
int check_sweep() {
    std::vector<float> row;
    for (uint64_t w = 0; w <= 0xffffffffull; w += 65521) {
        row.push_back(fp32_from_bits((uint32_t) w));
    }
    for (uint32_t center : { 0x477ff000u, 0x33800000u }) {
        for (uint32_t w = center - 8192; w <= center + 8192; w++) {
            row.push_back(fp32_from_bits(w));
            row.push_back(fp32_from_bits(w | 0x80000000u));
        }
    }
    std::vector<uint16_t> want(row.size());
    for (size_t i = 0; i < row.size(); i++) {
        want[i] = hw_conv(row[i]);
    }
    check_row(row, want);
    return (int) row.size();
}
#endif

}  // namespace

int main() {
    bool hw = false;
#if F16_CHECK_X86
    hw = have_f16c();
#endif
    int n = check_cases(hw);
#if F16_CHECK_X86
    if (hw) {
        n += check_sweep();
    }
#endif
    std::printf("f16_check: %d values, 3 functions, hardware reference %s, %d differences\n", n, hw ? "yes" : "no",
                g_diffs);
    return g_diffs == 0 ? 0 : 1;
}
