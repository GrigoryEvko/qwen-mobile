/* The TSan probe for the phone: mode "race" has a data race on a global
 * counter, mode "clean" has the same threads with a mutex.
 *
 * The question: does an arm64 Android executable with -fsanitize=thread run
 * on the phone, and does it report a data race? Build it with the clang of
 * the NDK:
 *   clang --target=aarch64-linux-android34 -fsanitize=thread -O1 -g
 *         -fno-omit-frame-pointer tsan_android_probe.c -o race_tsan
 * Push race_tsan and libclang_rt.tsan-aarch64-android.so of the same NDK to
 * one phone directory R, then run each mode there:
 *   env LD_LIBRARY_PATH=R TSAN_OPTIONS=halt_on_error=1 ./race_tsan race
 *   env LD_LIBRARY_PATH=R TSAN_OPTIONS=halt_on_error=1 ./race_tsan clean
 * The expected results: the report "WARNING: ThreadSanitizer: data race" and
 * the exit code 66 for "race", no report and the exit code 0 for "clean".
 *
 * With the NDK r29 (clang 21) on the OnePlus 13 (2026-09-23), the two modes
 * stop with "ThreadSanitizer: CHECK failed: tsan_rtl.cpp:1043" and the exit
 * code 139. The phone configurations of the fuzz areas are none, asan,
 * hwasan and ubsan, without tsan. */
#include <pthread.h>
#include <stdio.h>
#include <string.h>

static int g_counter;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_locked;

static void *worker(void *arg) {
    (void) arg;
    for (int i = 0; i < 100000; ++i) {
        if (g_locked) pthread_mutex_lock(&g_mu);
        g_counter++;
        if (g_locked) pthread_mutex_unlock(&g_mu);
    }
    return NULL;
}

int main(int argc, char **argv) {
    g_locked = argc > 1 && strcmp(argv[1], "clean") == 0;
    pthread_t t[2];
    for (int i = 0; i < 2; ++i) pthread_create(&t[i], NULL, worker, NULL);
    for (int i = 0; i < 2; ++i) pthread_join(t[i], NULL);
    printf("tsan-probe: %s counter %d\n", g_locked ? "clean" : "race", g_counter);
    return 0;
}
