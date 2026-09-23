/**
 * The libFuzzer front end of the thread fuzzer: the program of
 * fuzz_jni_api runs on the engine thread, and a second thread calls
 * requestStop and hasMtp (and free, with FUZZ_APP_CROSS_FREE=1) at the
 * times of a schedule from the input. Build it with ThreadSanitizer.
 */
#include "app_harness.h"
#include "fuzz_death.h"

namespace {
harness::Options g_opt;
}  // namespace

/** Read the options, turn the second thread on, and initialize the backends one time. */
extern "C" int LLVMFuzzerInitialize(int *, char ***) {
    g_opt = harness::options_from_env();
    g_opt.threads = true;
    harness::init_once(g_opt);
    return 0;
}

/** Run one program with the second thread. */
extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
    return harness::run_program(g_opt, data, size);
}
