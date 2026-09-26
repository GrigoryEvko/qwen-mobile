/*
 * The shared death callback of the libFuzzer targets (C and C++).
 *
 * The problem: a TSan report calls Die() while it holds a TSan lock (the slot
 * lock of ReportRace). A death callback that enters an interceptor of TSan
 * then waits for that lock forever, and the fuzz job hangs:
 *   the callback of libFuzzer:  Die -> StaticDeathCallback -> DumpCurrentUnit
 *                               -> Sha1ToString -> free -> OnUserFree -> SlotLock
 *   open, write or close:       Die -> callback -> close -> FdClose -> SlotLock
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
 * The same problem on Android: the libc installs the handlers of debuggerd
 * (with SA_SIGINFO) for SIGABRT and the other fatal signals in each process,
 * and libFuzzer keeps an existing SA_SIGINFO handler of each signal except
 * SIGSEGV. Thus an abort (a failed check of a harness, GGML_ABORT, assert,
 * std::terminate, a sanitizer report with abort_on_error=1) gives a tombstone
 * but no crash file of libFuzzer, and the input is lost. On Android the first
 * call of fuzz_death_note_input() installs fuzz_death_abort_handler() for
 * SIGABRT. It writes the input to $FUZZ_ARTIFACT_DIR/crash-abort-<pid> with
 * the same raw system calls (the abort can come from any state of the
 * allocator or of a lock), then it runs the handler of debuggerd, which writes
 * the tombstone and ends the process.
 *
 * FUZZ_ARTIFACT_DIR is the one variable that each area uses for crash files
 * (rule L9: no crash file in the working directory). Without it, the file
 * goes to the working directory.
 *
 * Usage: call fuzz_death_note_input(data, size) as the first statement of
 * LLVMFuzzerTestOneInput. libFuzzer sets its callback and its signal handlers
 * after LLVMFuzzerInitialize, thus the first input is the first correct time.
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

#if defined(__ANDROID__)
#include <signal.h>
#include <string.h>
#endif

/* The input of the running LLVMFuzzerTestOneInput call. */
static const uint8_t * fuzz_death_data = NULL;
/* The size of the input of the running LLVMFuzzerTestOneInput call. */
static size_t fuzz_death_size = 0;
/* The path of the crash file of the TSan death callback, made at the first input. */
static char fuzz_death_path[1024];

/*
 * Make the path $FUZZ_ARTIFACT_DIR/<name><pid> (or ./<name><pid>) in the
 * buffer, with no allocation and no formatted output. The path is shorter
 * than the buffer: a longer path is cut.
 */
static inline void fuzz_death_make_path(char * path, size_t cap, const char * name) {
    const char * dir = getenv("FUZZ_ARTIFACT_DIR");
    const char * part;
    char digits[24];
    size_t n = 0;
    int d = 0;
    long pid;

    part = (dir != NULL && dir[0] != '\0') ? dir : ".";
    while (*part != '\0' && n + 1 < cap) {
        path[n++] = *part++;
    }
    if (n + 1 < cap) {
        path[n++] = '/';
    }
    part = name;
    while (*part != '\0' && n + 1 < cap) {
        path[n++] = *part++;
    }
    for (pid = (long) getpid(); pid > 0 && d < 23; pid /= 10) {
        digits[d++] = (char) ('0' + pid % 10);
    }
    while (d > 0 && n + 1 < cap) {
        path[n++] = digits[--d];
    }
    path[n] = '\0';
}

/*
 * Write the input of the running call to a crash file. Raw system calls
 * only: TSan intercepts open, write and close, but not syscall(2), and a
 * signal handler must not take a lock.
 */
static inline void fuzz_death_write_input(const char * path) {
    const long fd = syscall(SYS_openat, AT_FDCWD, path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
        const long w = syscall(SYS_write, fd, fuzz_death_data, fuzz_death_size);
        (void) w;
        syscall(SYS_close, fd);
    }
}

/* The TSan death callback: write the input of the running call to the crash file. */
static inline void fuzz_death_tsan_callback(void) {
    fuzz_death_write_input(fuzz_death_path);
}

#if defined(__ANDROID__)
/* The path of the crash file of an abort, made at the first input. */
static char fuzz_death_abort_path[1024];
/* The SIGABRT action before the handler of this file: the handler of debuggerd. */
static struct sigaction fuzz_death_old_abort;

/*
 * The SIGABRT handler: write the input of the running call to the crash file,
 * then run the handler before it (debuggerd writes the tombstone and ends the
 * process). With no handler before it, restore the default action: abort()
 * raises SIGABRT again when this handler returns.
 */
static inline void fuzz_death_abort_handler(int sig, siginfo_t * info, void * uctx) {
    static const char msg[] = "fuzz: SIGABRT: the input of this call is in the crash-abort file of FUZZ_ARTIFACT_DIR\n";
    long w;
    fuzz_death_write_input(fuzz_death_abort_path);
    w = syscall(SYS_write, 2, msg, sizeof(msg) - 1);
    (void) w;
    if ((fuzz_death_old_abort.sa_flags & SA_SIGINFO) && fuzz_death_old_abort.sa_sigaction != NULL) {
        fuzz_death_old_abort.sa_sigaction(sig, info, uctx);
        return;
    }
    sigaction(sig, &fuzz_death_old_abort, NULL);
}
#endif

/*
 * Record the input of this call. The first call also makes the path of the
 * crash file and installs the handler: in a TSan build the death callback
 * fuzz_death_tsan_callback, and on Android the SIGABRT handler
 * fuzz_death_abort_handler.
 */
static inline void fuzz_death_note_input(const uint8_t * data, size_t size) {
#if defined(FUZZ_DEATH_TSAN) || defined(__ANDROID__)
    static int installed = 0;
    if (!installed) {
#if defined(FUZZ_DEATH_TSAN)
        fuzz_death_make_path(fuzz_death_path, sizeof(fuzz_death_path), "crash-tsan-");
        __sanitizer_set_death_callback(fuzz_death_tsan_callback);
#endif
#if defined(__ANDROID__)
        struct sigaction action;
        memset(&action, 0, sizeof(action));
        fuzz_death_make_path(fuzz_death_abort_path, sizeof(fuzz_death_abort_path), "crash-abort-");
        action.sa_sigaction = fuzz_death_abort_handler;
        action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&action.sa_mask);
        sigaction(SIGABRT, &action, &fuzz_death_old_abort);
#endif
        installed = 1;
    }
#endif
    fuzz_death_data = data;
    fuzz_death_size = size;
}

#endif /* QWEN_MOBILE_FUZZ_DEATH_H */
