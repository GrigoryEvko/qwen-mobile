/**
 * The input of the running fuzz iteration, for the checks of the harnesses.
 *
 * A failed check of a harness (not a sanitizer report) stops the process
 * with stop(). An abort() would raise SIGABRT, and the crash handler of
 * libFuzzer writes the input from inside that signal: ThreadSanitizer then
 * reports "signal-unsafe call inside of a signal" and stops before the file
 * is written. Thus stop() writes the input itself, to the directory of the
 * environment variable FUZZ_ARTIFACT_DIR (the one variable of all fuzz areas
 * for crash files, refer to tests/sanitizers/fuzz_death.h), and stops with _exit(1).
 *
 * No other dependency, thus each fuzzer includes this header.
 */
#pragma once

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace crash_input {

inline const uint8_t * g_data = nullptr;
inline size_t          g_size = 0;

/** Keep the input of the iteration that starts. The caller keeps the bytes alive until the next call. */
inline void remember(const uint8_t * data, size_t size) {
    g_data = data;
    g_size = size;
}

/** Write the input of the iteration (when there is one) and stop the process with the exit code 1. */
[[noreturn]] inline void stop() {
    const char * dir = std::getenv("FUZZ_ARTIFACT_DIR");
    if (g_data != nullptr && dir != nullptr) {
        uint64_t h = 14695981039346656037ull;
        for (size_t i = 0; i < g_size; ++i) {
            h = (h ^ g_data[i]) * 1099511628211ull;
        }
        char name[32];
        std::snprintf(name, sizeof(name), "crash-%016llx", (unsigned long long) h);
        const std::string path = std::string(dir) + "/" + name;
        if (FILE * f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(g_data, 1, g_size, f);
            std::fclose(f);
            std::fprintf(stderr, "==HARNESS== the input of the failed check is in %s\n", path.c_str());
        }
    }
    std::fflush(stderr);
    _exit(1);
}

}  // namespace crash_input
