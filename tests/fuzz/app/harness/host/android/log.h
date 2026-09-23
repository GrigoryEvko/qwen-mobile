/**
 * A host stub of <android/log.h> for the fuzz harnesses. The functions write
 * to stderr when the environment variable FUZZ_APP_LOG is set, and else they
 * do nothing, thus the log does not slow the fuzzer.
 */
#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

enum android_LogPriority {
    ANDROID_LOG_UNKNOWN = 0,
    ANDROID_LOG_DEFAULT,
    ANDROID_LOG_VERBOSE,
    ANDROID_LOG_DEBUG,
    ANDROID_LOG_INFO,
    ANDROID_LOG_WARN,
    ANDROID_LOG_ERROR,
    ANDROID_LOG_FATAL,
    ANDROID_LOG_SILENT,
};

/** True when the log goes to stderr. The value is read one time. */
inline bool fuzz_app_log_enabled() {
    static const bool on = std::getenv("FUZZ_APP_LOG") != nullptr;
    return on;
}

/** Write one formatted line with its tag. */
inline int __android_log_print(int prio, const char * tag, const char * fmt, ...) {
    if (!fuzz_app_log_enabled()) {
        return 0;
    }
    std::fprintf(stderr, "[%d %s] ", prio, tag);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    return 1;
}

/** Write one line with its tag. */
inline int __android_log_write(int prio, const char * tag, const char * text) {
    if (!fuzz_app_log_enabled()) {
        return 0;
    }
    return std::fprintf(stderr, "[%d %s] %s", prio, tag, text);
}
