// Shared helpers of the core fuzz harnesses.
//
// Each harness includes this header. It gives:
//   - the log filter: llama.cpp and ggml write nothing, unless FUZZ_VERBOSE=1.
//     The filter counts the error lines, and a harness can read the count
//     (the NPU harness uses it as a check of the session health).
//   - the data directory: FUZZ_DATA_DIR, or <directory of the executable>/../core/data.
//     run.sh writes the vocab-only GGUF, the tiny model and the chat template there.
//   - fuzz_fail(): write a message and call abort(). libFuzzer records the input
//     (on Android the SIGABRT handler of note_input() does).
//   - small byte helpers on top of FuzzedDataProvider.
//   - note_input(): the first call of each target. It records the input for the
//     shared TSan death callback of tests/sanitizers/fuzz_death.h. On Android it
//     also installs the SIGABRT handler that keeps the input of an abort.

#pragma once

#include <fuzzer/FuzzedDataProvider.h>

#include "ggml.h"
#include "llama.h"

#include "fuzz_death.h"  // tests/sanitizers: the TSan death callback without a deadlock

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace fuzz {

#if defined(__ANDROID__)
// On Android the libc installs the handlers of debuggerd (with SA_SIGINFO) for SIGABRT and the other
// fatal signals in each process, and libFuzzer keeps an existing SA_SIGINFO handler of each signal
// except SIGSEGV. Thus an abort (fuzz::fail, GGML_ABORT, assert, std::terminate) gives a tombstone
// but no crash file of libFuzzer, and the input is lost. The handler below keeps it. A sanitizer
// report does not need it: the death callback of libFuzzer writes that input.

/** The path $FUZZ_ARTIFACT_DIR/crash-abort-<pid> (or ./crash-abort-<pid>), made at the first input. */
inline char * abort_unit_path() {
    static char path[1024];
    return path;
}

/** The SIGABRT action before the handler of the harness: the handler of debuggerd. */
inline struct sigaction & old_abort_action() {
    static struct sigaction action;
    return action;
}

/**
 * The SIGABRT handler: write the input of the running call to the crash file with raw system calls
 * only (the abort can come from any state of the allocator or of a lock), then run the handler of
 * debuggerd, which writes the tombstone and ends the process.
 */
inline void abort_unit_handler(int sig, siginfo_t * info, void * uctx) {
    const long fd = syscall(SYS_openat, AT_FDCWD, abort_unit_path(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
        const long w = syscall(SYS_write, fd, fuzz_death_data, fuzz_death_size);
        (void) w;
        syscall(SYS_close, fd);
    }
    static const char msg[] = "fuzz: SIGABRT: the input of this call is in the crash-abort file of FUZZ_ARTIFACT_DIR\n";
    const long w = syscall(SYS_write, 2, msg, sizeof(msg) - 1);
    (void) w;
    const struct sigaction & old = old_abort_action();
    if ((old.sa_flags & SA_SIGINFO) && old.sa_sigaction != nullptr) {
        old.sa_sigaction(sig, info, uctx);
        return;
    }
    // no handler before: the default action. abort() raises SIGABRT again when this handler returns.
    sigaction(sig, &old, nullptr);
}

/** Make the path of the crash file and install the SIGABRT handler, one time. */
inline void install_abort_unit_handler() {
    static bool installed = false;
    if (installed) {
        return;
    }
    installed = true;
    const char * dir = getenv("FUZZ_ARTIFACT_DIR");
    snprintf(abort_unit_path(), 1024, "%s/crash-abort-%d", (dir != nullptr && dir[0] != '\0') ? dir : ".", (int) getpid());
    struct sigaction action = {};
    action.sa_sigaction = abort_unit_handler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    sigaction(SIGABRT, &action, &old_abort_action());
}
#endif

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
 * FUZZ_ARTIFACT_DIR/crash-tsan-<pid> without a deadlock. On Android the first call installs the
 * SIGABRT handler that writes the input to FUZZ_ARTIFACT_DIR/crash-abort-<pid>. libFuzzer sets its
 * handlers before the first input, thus the first input is the correct time. Each target calls this
 * function first in LLVMFuzzerTestOneInput.
 */
inline void note_input(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
#if defined(__ANDROID__)
    install_abort_unit_handler();
#endif
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
