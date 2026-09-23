/*
 * The shared death callback of the libFuzzer targets (C and C++).
 *
 * The problem (found by fuzz-core, evidence in build/fuzz/core/triage/tsan-hang/):
 * a TSan report calls Die() while it holds a TSan lock. The death callback of
 * libFuzzer then writes the input (DumpCurrentUnit), and Sha1ToString calls
 * free, which waits for that lock forever:
 *   ReportRace -> OutputReport -> Die -> StaticDeathCallback -> DumpCurrentUnit
 *   -> Sha1ToString -> free -> OnUserFree -> SlotLock -> FutexWait.
 * One tsan run hung for 31 minutes.
 *
 * The solution: in a TSan build, fuzz_death_note_input() puts
 * fuzz_death_tsan_callback() in the place of the callback of libFuzzer. That
 * callback writes the input to $FUZZ_ARTIFACT_DIR/crash-tsan-<pid> with the
 * system calls open, write and close only. It allocates nothing and takes no
 * lock. The other builds keep the callback of libFuzzer.
 *
 * FUZZ_ARTIFACT_DIR is the one variable that each area uses for crash files
 * (rule L9: no crash file in the working directory). Without it, the file
 * goes to the working directory.
 *
 * Usage: call fuzz_death_note_input(data, size) as the first statement of
 * LLVMFuzzerTestOneInput. libFuzzer sets its callback after
 * LLVMFuzzerInitialize, thus the first input is the first correct time.
 * tests/sanitizers/check-rules.sh requires this header and this call in each
 * libFuzzer target of a tsan build.
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

/*
 * Write the input of the running call to $FUZZ_ARTIFACT_DIR/crash-tsan-<pid>.
 * It uses system calls only: no allocation, no lock, no stdio.
 */
static void fuzz_death_tsan_callback(void) {
    const char * dir = getenv("FUZZ_ARTIFACT_DIR");
    const char * part;
    char path[1024];
    char digits[24];
    size_t n = 0;
    int d = 0;
    long pid;
    int fd;

    part = (dir != NULL && dir[0] != '\0') ? dir : ".";
    while (*part != '\0' && n + 1 < sizeof(path)) {
        path[n++] = *part++;
    }
    part = "/crash-tsan-";
    while (*part != '\0' && n + 1 < sizeof(path)) {
        path[n++] = *part++;
    }
    for (pid = (long) getpid(); pid > 0 && d < 23; pid /= 10) {
        digits[d++] = (char) ('0' + pid % 10);
    }
    while (d > 0 && n + 1 < sizeof(path)) {
        path[n++] = digits[--d];
    }
    path[n] = '\0';
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
        const ssize_t w = write(fd, fuzz_death_data, fuzz_death_size);
        (void) w;
        close(fd);
    }
}

/*
 * Record the input of this call. In a TSan build, the first call also
 * installs fuzz_death_tsan_callback as the death callback.
 */
static inline void fuzz_death_note_input(const uint8_t * data, size_t size) {
#if defined(FUZZ_DEATH_TSAN)
    static int installed = 0;
    if (!installed) {
        __sanitizer_set_death_callback(fuzz_death_tsan_callback);
        installed = 1;
    }
#endif
    fuzz_death_data = data;
    fuzz_death_size = size;
}

#endif /* QWEN_MOBILE_FUZZ_DEATH_H */
