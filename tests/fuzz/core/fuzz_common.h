// Shared helpers of the core fuzz harnesses.
//
// Each harness includes this header. It gives:
//   - the log filter: llama.cpp and ggml write nothing, unless FUZZ_VERBOSE=1.
//     The filter counts the error lines, and a harness can read the count
//     (the NPU harness uses it as a check of the session health).
//   - the data directory: FUZZ_DATA_DIR, or <directory of the executable>/../core/data.
//     run.sh writes the vocab-only GGUF, the tiny model and the chat template there.
//   - fuzz_fail(): write a message and call abort(). libFuzzer records the input.
//   - small byte helpers on top of FuzzedDataProvider.
//   - gguf_scan(): the checks of the switches FUZZ_GGUF_KNOWN_ENUM_LOAD and
//     FUZZ_GGUF_KNOWN_NELEMENTS, which fuzz_gguf and fuzz_model_load use (the two
//     read GGUF files).
//   - note_input(): the first call of each target. It records the input for the
//     shared TSan death callback of tests/sanitizers/fuzz_death.h.

#pragma once

#include <fuzzer/FuzzedDataProvider.h>

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include "fuzz_death.h"  // tests/sanitizers: the TSan death callback without a deadlock

#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace fuzz {

/** The count of error lines that llama.cpp and ggml wrote since the start. */
inline std::atomic<long> & log_error_count() {
    static std::atomic<long> count{0};
    return count;
}

/** The log callback: count the errors, and write the text only when FUZZ_VERBOSE=1. */
inline void log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    static const bool verbose = [] {
        const char * s = getenv("FUZZ_VERBOSE");
        return s != nullptr && atoi(s) != 0;
    }();
    if (level == GGML_LOG_LEVEL_ERROR) {
        log_error_count().fetch_add(1, std::memory_order_relaxed);
    }
    if (verbose) {
        fputs(text, stderr);
    }
}

/** Install the log callback in llama.cpp and in ggml. Call it one time, from LLVMFuzzerInitialize. */
inline void quiet_logs() {
    llama_log_set(log_callback, nullptr);
    ggml_log_set(log_callback, nullptr);
}

/**
 * Record the input of this call for the shared death callback (tests/sanitizers/fuzz_death.h). In a
 * TSan build the first call installs the callback that writes the input to
 * FUZZ_ARTIFACT_DIR/crash-tsan-<pid> without a deadlock. Each target calls this function first in
 * LLVMFuzzerTestOneInput.
 */
inline void note_input(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
}

/** Write a message to stderr and stop the process with abort(). libFuzzer keeps the input as a crash. */
[[noreturn]] inline void fail(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fputs("FUZZ FAILURE: ", stderr);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    va_end(args);
    fflush(stderr);
    abort();
}

/** The directory of the running executable, from /proc/self/exe. Empty if the link cannot be read. */
inline std::string exe_dir() {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return "";
    }
    buf[n] = '\0';
    std::string path(buf);
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

/** The directory of the generated inputs: FUZZ_DATA_DIR, or <exe dir>/../core/data. */
inline std::string data_dir() {
    const char * s = getenv("FUZZ_DATA_DIR");
    if (s != nullptr && s[0] != '\0') {
        return s;
    }
    return exe_dir() + "/../core/data";
}

/** The full path of a file in the data directory. Stop with a clear message if the file is missing. */
inline std::string data_file(const char * name) {
    const std::string path = data_dir() + "/" + name;
    if (access(path.c_str(), R_OK) != 0) {
        fail("the data file %s is missing. Run 'tests/fuzz/core/run.sh data' first, or set FUZZ_DATA_DIR.", path.c_str());
    }
    return path;
}

/** An integer environment variable, or the fallback when it is not set. */
inline long env_long(const char * name, long fallback) {
    const char * s = getenv(name);
    return (s != nullptr && s[0] != '\0') ? strtol(s, nullptr, 0) : fallback;
}

/**
 * A float from the input that is often a special value: 1 byte selects the
 * class, then 4 bytes give the value when the class is "any bits".
 * The classes are NaN, +Inf, -Inf, 0, -0, the smallest denormal, FLT_MAX,
 * -FLT_MAX, a small integer (which makes ties), and any bit pattern.
 */
inline float special_float(FuzzedDataProvider & fdp) {
    switch (fdp.ConsumeIntegralInRange<int>(0, 15)) {
        case 0:  return NAN;
        case 1:  return INFINITY;
        case 2:  return -INFINITY;
        case 3:  return 0.0f;
        case 4:  return -0.0f;
        case 5:  return 1.4e-45f;
        case 6:  return 3.4028235e38f;
        case 7:  return -3.4028235e38f;
        case 8:
        case 9:
        case 10: return (float) fdp.ConsumeIntegralInRange<int>(-4, 4);
        default: {
            uint32_t bits = fdp.ConsumeIntegral<uint32_t>();
            float f;
            memcpy(&f, &bits, sizeof(f));
            return f;
        }
    }
}

/** The value range of a C enum without a fixed type: [0, 2^k) for the smallest k with 2^k > max. */
constexpr int64_t enum_range(int64_t max) {
    int64_t r = 1;
    while (r <= max) {
        r *= 2;
    }
    return r;
}

/** The conditions of a GGUF file that gguf_scan finds. */
struct GgufScan {
    bool bad_enum           = false;  // finding gguf-enum-load
    bool nelements_overflow = false;  // finding gguf-nelements-overflow
};

/**
 * Walk the header of a GGUF file as the reader of ggml/src/gguf.cpp does, and find two conditions:
 *   bad_enum            The reader loads an enum value outside the range of its type (finding
 *                       gguf-enum-load, a UBSan "invalid-enum-load" report): the file ends at the type
 *                       field of a KV pair (the reader then loads the initial value -1) or of a tensor
 *                       (the reader then loads an uninitialized value), or a type is outside the range of
 *                       gguf_type or ggml_type.
 *   nelements_overflow  The product of the dimensions of a tensor overflows int64_t in the order of
 *                       ggml_nelements (finding gguf-nelements-overflow: a UBSan signed-integer-overflow
 *                       report, and a product that wraps to 0 gives a tensor of 0 bytes).
 * The walk stops where the reader stops. It does not do all the checks of the reader, thus it can find
 * a condition in a file that the reader refuses before that point. O(size).
 */
inline GgufScan gguf_scan(const uint8_t * data, size_t size) {
    GgufScan r;
    constexpr int64_t kGgufRange = enum_range(GGUF_TYPE_COUNT);
    constexpr int64_t kGgmlRange = enum_range(GGML_TYPE_COUNT);
    // the size of one value of each scalar gguf_type, 0 for STRING and ARRAY
    static const uint64_t kSize[GGUF_TYPE_COUNT] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
    const uint8_t * p = data;
    size_t n = size;
    auto get = [&](void * dst, size_t k) {
        if (n < k) {
            return false;
        }
        memcpy(dst, p, k);
        p += k;
        n -= k;
        return true;
    };
    auto skip = [&](uint64_t k) {
        if (n < k) {
            return false;
        }
        p += k;
        n -= (size_t) k;
        return true;
    };
    uint32_t version;
    int64_t  n_tensors, n_kv;
    if (n < 4 || memcmp(p, "GGUF", 4) != 0 || !skip(4) || !get(&version, 4) || !get(&n_tensors, 8) || !get(&n_kv, 8) ||
        n_tensors < 0 || n_kv < 0) {
        return r;
    }
    for (int64_t i = 0; i < n_kv; ++i) {
        uint64_t key_len;
        if (!get(&key_len, 8) || key_len == 0 || !skip(key_len)) {
            return r;
        }
        int32_t type;
        if (!get(&type, 4) || type < 0 || type >= kGgufRange) {
            r.bad_enum = true;
            return r;
        }
        uint64_t count = 1;
        if (type == GGUF_TYPE_ARRAY) {
            if (!get(&type, 4)) {
                return r;
            }
            if (type < 0 || type >= kGgufRange) {
                r.bad_enum = true;
                return r;
            }
            if (!get(&count, 8)) {
                return r;
            }
        }
        if (type == GGUF_TYPE_STRING) {
            for (uint64_t j = 0; j < count; ++j) {
                uint64_t len;
                if (!get(&len, 8) || !skip(len)) {
                    return r;
                }
            }
        } else if (type < GGUF_TYPE_COUNT && kSize[type] != 0) {
            if (count > n / kSize[type]) {
                return r;
            }
            skip(count * kSize[type]);
        } else {
            return r;  // the reader refuses a type in the range of the enum: no report
        }
    }
    for (int64_t i = 0; i < n_tensors; ++i) {
        uint64_t name_len;
        uint32_t n_dims;
        if (!get(&name_len, 8) || name_len >= GGML_MAX_NAME || !skip(name_len) || !get(&n_dims, 4) || n_dims > GGML_MAX_DIMS) {
            return r;
        }
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (uint32_t d = 0; d < n_dims; ++d) {
            if (!get(&ne[d], 8) || ne[d] < 0) {
                return r;
            }
        }
        // ggml_nelements multiplies ne[0] * ne[1] * ne[2] * ne[3] from the left in int64_t
        int64_t prod = 1;
        for (int d = 0; d < GGML_MAX_DIMS && !r.nelements_overflow; ++d) {
            r.nelements_overflow = __builtin_mul_overflow(prod, ne[d], &prod);
        }
        int32_t type;
        if (!get(&type, 4) || type < 0 || type >= kGgmlRange) {
            r.bad_enum = true;
            return r;
        }
        uint64_t offset;
        if (!get(&offset, 8)) {
            return r;
        }
    }
    return r;
}

/** True when the GGUF reader loads an enum value outside the range of its type on this file (see gguf_scan). */
inline bool gguf_bad_enum(const uint8_t * data, size_t size) {
    return gguf_scan(data, size).bad_enum;
}

/** True when the byte string is valid UTF-8 without overlong forms, surrogates, or code points above U+10FFFF. */
inline bool is_valid_utf8(const std::string & s) {
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const unsigned char c = (unsigned char) s[i];
        size_t len;
        uint32_t cp;
        if (c < 0x80) {
            i += 1;
            continue;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2; cp = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3; cp = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4; cp = c & 0x07;
        } else {
            return false;
        }
        if (i + len > n) {
            return false;
        }
        for (size_t k = 1; k < len; ++k) {
            const unsigned char cc = (unsigned char) s[i + k];
            if ((cc & 0xC0) != 0x80) {
                return false;
            }
            cp = (cp << 6) | (cc & 0x3F);
        }
        const bool overlong = (len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000);
        if (overlong || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            return false;
        }
        i += len;
    }
    return true;
}

}  // namespace fuzz
