/*
 * The self-test of tests/sanitizers/fuzz_death.h (C and C++).
 *
 * A libFuzzer target with a deliberate data race: an input that starts with
 * 'R' starts two threads that increment one counter with no lock. Under
 * TSan with halt_on_error=1 the race report calls Die(), and the death
 * callback of fuzz_death.h must write the input to
 * $FUZZ_ARTIFACT_DIR/crash-tsan-<pid> and let the process stop.
 * tests/sanitizers/tsan-death-selftest.sh builds it and gives each run 30 s.
 */

#include "fuzz_death.h"

#include <pthread.h>

/* The counter that the two threads increment with no lock. */
static int g_counter;

/* Increment the shared counter with no lock: the deliberate race. */
static void * racer(void * arg) {
    for (int i = 0; i < 100000; ++i) {
        g_counter++;
    }
    return arg;
}

#ifdef __cplusplus
extern "C"
#endif
int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
    if (size > 0 && data[0] == 'R') {
        pthread_t t[2];
        for (int i = 0; i < 2; ++i) {
            pthread_create(&t[i], NULL, racer, NULL);
        }
        for (int i = 0; i < 2; ++i) {
            pthread_join(t[i], NULL);
        }
    }
    return 0;
}
