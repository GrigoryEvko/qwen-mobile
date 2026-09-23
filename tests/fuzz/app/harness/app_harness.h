/**
 * The interpreter of the app fuzzers. One program is a byte string: each
 * operation takes one byte for its code and some bytes for its arguments,
 * and calls the JNI entry points of llama_jni.cpp in the order that the
 * Kotlin side can call them. After each operation the interpreter checks the
 * invariants of the engine (refer to app_harness.cpp).
 *
 * The libFuzzer front ends and the standalone driver share this interpreter.
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace harness {

/** The configuration of the interpreter. The environment gives the values (refer to options_from_env). */
struct Options {
    /** The directory of the tiny models (make_tiny_model.py). */
    std::string model_dir;
    /** A real model and its projector, for short programs on the phone. Empty for none. */
    std::string real_model;
    std::string real_mmproj;
    /** The ggml device of the accelerator, for example HTP0. Empty on a host without one. */
    std::string device;
    /** A directory for the cache directories and the scratch files of one program. */
    std::string work_dir;
    /** Compare the model memory of an engine with a clean decode of its tokens after a turn. */
    bool oracle = true;
    /** Run a second thread that calls requestStop (and free, with cross_free) while the program runs. */
    bool threads = false;
    bool cross_free = false;
    /** The upper limit of the operations of one program. */
    int max_ops = 48;
    /** The upper limit of generateNext calls in one operation. */
    int max_gen = 96;
    /** Print each operation to stderr. */
    bool trace = false;
};

/** The options from the environment: FUZZ_APP_MODEL_DIR, FUZZ_APP_REAL_MODEL, FUZZ_APP_REAL_MMPROJ,
 *  FUZZ_APP_DEVICE, FUZZ_APP_WORK, FUZZ_APP_ORACLE, FUZZ_APP_THREADS, FUZZ_APP_CROSS_FREE,
 *  FUZZ_APP_MAX_OPS, FUZZ_APP_MAX_GEN, FUZZ_APP_TRACE. */
Options options_from_env();

/** Initialize the backends one time, as LlamaNative.initialize does, and register the Java methods. */
void init_once(const Options & opt);

/** Run one program. Returns 0. A failed invariant aborts the process with a message. */
int run_program(const Options & opt, const uint8_t * data, size_t size);

/**
 * The check of the thread priorities: a thread of the app that starts during a load,
 * after the first listing of /proc/self/task, gets the compute priority and
 * a place in the ADPF session. The check starts such a thread, loads the
 * tiny model, and reports the priority calls. Returns 0 when the load did not
 * change the priority of that thread, else 1.
 */
int check_priority(const Options & opt);

/**
 * Run one deterministic scenario of a defect class. The names:
 * image-shape, image-twice, jni-pending, spec-disable, spec-parity, sampler-nan, priority. Returns 0
 * when the defect does not occur, 1 when the priority check reports it,
 * and 2 for an unknown name. The other defects stop the process with the
 * message of the check or the report of the sanitizer.
 */
int run_scenario(const Options & opt, const std::string & name);

/**
 * The speed of the answer path on the host, to compare two builds of the decode and
 * sample path: the tiny model (F32, one thread) answers one message with up
 * to 64 tokens, reps times without and reps times with speculation. Prints
 * the median, the minimum and the maximum of the time of one generateNext
 * for each mode. Returns 0. O(reps x 64) generateNext calls.
 */
int speed_check(const Options & opt, int reps);

/** The counters of all programs of this process, for the summary line of the driver. Two threads write them. */
struct Counters {
    std::atomic<uint64_t> programs{0};
    std::atomic<uint64_t> ops{0};
    std::atomic<uint64_t> loads_ok{0};
    std::atomic<uint64_t> loads_failed{0};
    std::atomic<uint64_t> turns{0};
    std::atomic<uint64_t> tokens{0};
    std::atomic<uint64_t> java_exceptions{0};
    std::atomic<uint64_t> stops{0};
    std::atomic<uint64_t> oracle_checks{0};
    std::atomic<uint64_t> images{0};
    std::atomic<uint64_t> faults{0};
    std::atomic<uint64_t> drafted{0};
    std::atomic<uint64_t> accepted{0};
};
const Counters & counters();

/** Write the summary of the counters and of the distinct Java exceptions to stderr. */
void print_summary();

}  // namespace harness
