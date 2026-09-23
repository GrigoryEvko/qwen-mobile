/*
 * The shared death callback of the libFuzzer targets (C and C++).
 *
 * The problem (found by fuzz-core, evidence in build/fuzz/core/triage/tsan-hang/):
 * a TSan report calls Die() while it holds a TSan lock (the slot lock of
 * ReportRace). A death callback that enters an interceptor of TSan then
 * waits for that lock forever:
 *   libFuzzer:     Die -> StaticDeathCallback -> DumpCurrentUnit -> Sha1ToString
 *                  -> free -> OnUserFree -> SlotLock -> FutexWait
 *   open/write/close (the first version of this file, 06:34):
 *                  Die -> fuzz_death_tsan_callback -> close -> FdClose -> SlotLock
 * One tsan run hung for 31 minutes.
 *
 * The solution: in a TSan build, fuzz_death_note_input() puts
 * fuzz_death_tsan_callback() in the place of the callback of libFuzzer. That
 * callback calls only raw system calls through syscall(2): SYS_openat,
 * SYS_write and SYS_close. TSan does not intercept syscall(2). The callback
 * allocates nothing, formats nothing and takes no lock. The path
 * $FUZZ_ARTIFACT_DIR/crash-tsan-<pid> is built one time, in a static buffer,
 * at the first input (getenv and getpid run there, not in the callback).
 * bionic has the same system call numbers through syscall(2).
 * The other builds keep the callback of libFuzzer.
 *
 * FUZZ_ARTIFACT_DIR is the one variable that each area uses for crash files
 * (rule L9: no crash file in the working directory). Without it, the file
 * goes to the working directory.
 *
 * Usage: call fuzz_death_note_input(data, size) as the first statement of
 * LLVMFuzzerTestOneInput. libFuzzer sets its callback after
 * LLVMFuzzerInitialize, thus the first input is the first correct time.
 * tests/sanitizers/check-rules.sh requires this header and this call in each
 * libFuzzer target. tests/sanitizers/tsan-death-selftest.sh proves the
 * callback: a deliberate data race must give its crash file and stop in
 * 30 s, in the two profiles.
 *
 * The state has internal linkage (static), thus call fuzz_death_note_input
 * from the one translation unit that has LLVMFuzzerTestOneInput.
 */

#ifndef QWEN_MOBILE_FUZZ_DEATH_H
#define QWEN_MOBILE_FUZZ_DEATH_H

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define FUZZ_DEATH_TSAN 1
#include <sanitizer/common_interface_defs.h> /* __sanitizer_set_death_callback */
#endif
#endif

/* The input of the running LLVMFuzzerTestOneInput call. */
static const uint8_t * fuzz_death_data = NULL;
/* The size of the input of the running LLVMFuzzerTestOneInput call. */
static size_t fuzz_death_size = 0;
/* The path of the crash file, made at the first input. */
static char fuzz_death_path[1024];

/*
 * Make the path $FUZZ_ARTIFACT_DIR/crash-tsan-<pid> (or ./crash-tsan-<pid>)
 * in fuzz_death_path, with no allocation and no formatted output.
 */
static inline void fuzz_death_make_path(void) {
    const char * dir = getenv("FUZZ_ARTIFACT_DIR");
    const char * part;
    char digits[24];
    size_t n = 0;
    int d = 0;
    long pid;

    part = (dir != NULL && dir[0] != '\0') ? dir : ".";
    while (*part != '\0' && n + 1 < sizeof(fuzz_death_path)) {
        fuzz_death_path[n++] = *part++;
    }
    part = "/crash-tsan-";
    while (*part != '\0' && n + 1 < sizeof(fuzz_death_path)) {
        fuzz_death_path[n++] = *part++;
    }
    for (pid = (long) getpid(); pid > 0 && d < 23; pid /= 10) {
        digits[d++] = (char) ('0' + pid % 10);
    }
    while (d > 0 && n + 1 < sizeof(fuzz_death_path)) {
        fuzz_death_path[n++] = digits[--d];
    }
    fuzz_death_path[n] = '\0';
}

/*
 * Write the input of the running call to the crash file. Raw system calls
 * only: TSan intercepts open, write and close, but not syscall(2).
 */
static inline void fuzz_death_tsan_callback(void) {
    const long fd = syscall(SYS_openat, AT_FDCWD, fuzz_death_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
        const long w = syscall(SYS_write, fd, fuzz_death_data, fuzz_death_size);
        (void) w;
        syscall(SYS_close, fd);
    }
}

/*
 * Record the input of this call. In a TSan build, the first call also
 * makes the path of the crash file and installs fuzz_death_tsan_callback as
 * the death callback.
 */
static inline void fuzz_death_note_input(const uint8_t * data, size_t size) {
#if defined(FUZZ_DEATH_TSAN)
    static int installed = 0;
    if (!installed) {
        fuzz_death_make_path();
        __sanitizer_set_death_callback(fuzz_death_tsan_callback);
        installed = 1;
    }
#endif
    fuzz_death_data = data;
    fuzz_death_size = size;
}

#endif /* QWEN_MOBILE_FUZZ_DEATH_H */
