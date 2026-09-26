/**
 * FNV-1a over 64 bits, for the tools that show that two runs moved the same bytes.
 *
 * A tool prints the hash of a logits vector, of an image embedding or of a matmul output, and a
 * stage report compares the hash of one run against the hash of another run OF THE SAME TOOL.
 *
 * THE OFFSET BASIS IS A PARAMETER, AND THE TOOLS DO NOT SHARE ONE VALUE.
 *
 *   kBasisFnv1a  The basis of the FNV-1a specification. tools/gemv/gemvcheck.cpp uses it.
 *   kBasisShort  That value without its last decimal digit. tools/memprobe/memprobe.cpp,
 *                tools/memprobe/kvkl.cpp, tools/memprobe/outcheck.cpp and tools/vit/vitprobe.cpp
 *                use it.
 *
 * Thus two tools give two different values for the same bytes, and a comparison of a hash of one
 * tool against a hash of another tool is not valid. Each caller keeps the basis that it has,
 * because the recorded hashes of the phone stages and the acceptance criteria of landed patches
 * hold those values. A caller that changes its basis makes every recorded value of that tool
 * invalid.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace fnv {

/** The offset basis of the FNV-1a specification. */
inline constexpr uint64_t kBasisFnv1a = 0xCBF29CE484222325ull;

/** The basis of the specification without its last decimal digit. */
inline constexpr uint64_t kBasisShort = 1469598103934665603ull;

/** The prime of FNV-1a over 64 bits. */
inline constexpr uint64_t kPrime = 0x100000001B3ull;

/**
 * One FNV-1a step over a whole 64-bit value, and not over its bytes.
 *
 * The step takes the value in one exclusive-or, thus it gives a different result from four steps
 * over the four bytes of a 32-bit value. A caller that hashes a list of numbers uses this step,
 * and a caller that hashes a buffer uses hash64.
 *
 * @param h     The value of the hash before the step
 * @param value The value of the step
 * @return      The value of the hash after the step
 */
inline uint64_t step64(uint64_t h, uint64_t value) {
    return (h ^ value) * kPrime;
}

/**
 * The FNV-1a hash of n bytes, continued from h.
 *
 * @param p The first byte
 * @param n The number of bytes
 * @param h The basis, or the hash of the bytes before these bytes
 * @return  The hash
 *
 * Complexity: O(n).
 */
inline uint64_t hash64(const void * p, size_t n, uint64_t h) {
    const auto * b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) {
        h = step64(h, b[i]);
    }
    return h;
}

} // namespace fnv
