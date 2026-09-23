/**
 * The libFuzzer front end of the JNI API fuzzer: one input is one program
 * of the interpreter in harness/app_harness.cpp, on one thread.
 */
#include "app_harness.h"
#include "fuzz_death.h"

namespace {
harness::Options g_opt;
}  // namespace

/** Read the options and initialize the backends one time. */
extern "C" int LLVMFuzzerInitialize(int *, char ***) {
    g_opt = harness::options_from_env();
    harness::init_once(g_opt);
    return 0;
}

/** Run one program. */
extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
    return harness::run_program(g_opt, data, size);
}
