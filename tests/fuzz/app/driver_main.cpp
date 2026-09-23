/**
 * The standalone driver of the app fuzzers, for the phone and for long runs
 * without libFuzzer. It gives the interpreter of harness/app_harness.cpp
 * programs of random bytes from a seeded generator, or the programs of files.
 *
 *   app_fuzz_driver [--seconds S] [--iters N] [--seed X] [--size B] [--threads] [--cross-free]
 *                   [--replay FILE...] [--check-priority] [--scenario NAME] [--speed REPS]
 *                   [--selftest-threads]
 *
 * --seconds   Stop after S seconds (the default is 60).
 * --iters     Stop after N programs (the default has no limit).
 * --seed      The seed of the first program. Program i uses seed + i, thus a
 *             failed program runs again with --seed <its seed> --iters 1.
 * --size      The bytes of one program (the default is 1024).
 * --threads   Run the second thread of fuzz_jni_threads.
 * --replay    Run the files as programs, then stop.
 * --check-priority  Run the check of the thread priorities and stop with its result.
 * --scenario  Run one scenario of a defect class (image-shape, image-twice,
 *             jni-pending, spec-disable, spec-parity, spec-image, sampler-nan, priority)
 *             and stop.
 * --speed     Measure the time of one generateNext of the tiny model, REPS
 *             answers without and REPS with speculation, and stop.
 * --selftest-threads  Start one thread, join it, and stop: the first step of
 *             a phone ASan run. A runtime that traps in a new
 *             thread stops the process here.
 *
 * The environment variables of app_harness.h select the models and the device.
 * With FUZZ_APP_LAST_PROGRAM=<file>, the driver writes each program to the
 * file before it runs, thus a crash leaves its input there.
 */
#include "app_harness.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

/** Print the usage text and return the exit code 2. */
int usage() {
    fprintf(stderr,
            "usage: app_fuzz_driver [--seconds S] [--iters N] [--seed X] [--size B] [--threads] [--cross-free]\n"
            "                       [--replay FILE...] [--check-priority] [--scenario NAME] [--speed REPS]\n"
            "                       [--selftest-threads]\n");
    return 2;
}

/** The bytes of a file, or an empty vector when the file is not readable. */
std::vector<uint8_t> read_all(const char * path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

}  // namespace

/** Parse the arguments and run the programs. */
int main(int argc, char ** argv) {
    harness::Options opt = harness::options_from_env();
    double seconds = 60.0;
    uint64_t iters = UINT64_MAX;
    uint64_t seed = 1;
    size_t size = 1024;
    std::vector<const char *> replay;
    bool check_priority = false;
    std::string scenario;
    int speed_reps = 0;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "--seconds") {
            const char * v = next();
            if (v == nullptr) return usage();
            seconds = atof(v);
        } else if (a == "--iters") {
            const char * v = next();
            if (v == nullptr) return usage();
            iters = strtoull(v, nullptr, 10);
        } else if (a == "--seed") {
            const char * v = next();
            if (v == nullptr) return usage();
            seed = strtoull(v, nullptr, 10);
        } else if (a == "--size") {
            const char * v = next();
            if (v == nullptr) return usage();
            size = strtoull(v, nullptr, 10);
        } else if (a == "--threads") {
            opt.threads = true;
        } else if (a == "--cross-free") {
            opt.cross_free = true;
        } else if (a == "--check-priority") {
            check_priority = true;
        } else if (a == "--scenario") {
            const char * v = next();
            if (v == nullptr) return usage();
            scenario = v;
        } else if (a == "--selftest-threads") {
            selftest = true;
        } else if (a == "--speed") {
            const char * v = next();
            if (v == nullptr || atoi(v) <= 0) return usage();
            speed_reps = atoi(v);
        } else if (a == "--replay") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                replay.push_back(argv[++i]);
            }
        } else {
            return usage();
        }
    }
    if (selftest) {
        // No backend loads: the test is only the start of one thread under the sanitizer runtime.
        std::atomic<bool> ran{false};
        std::thread t([&ran] { ran = true; });
        t.join();
        fprintf(stderr, "selftest: one thread %s\n", ran ? "started and joined" : "did not run");
        return ran ? 0 : 1;
    }
    harness::init_once(opt);
    if (check_priority) {
        return harness::check_priority(opt);
    }
    if (!scenario.empty()) {
        return harness::run_scenario(opt, scenario);
    }
    if (speed_reps > 0) {
        return harness::speed_check(opt, speed_reps);
    }
    if (!replay.empty()) {
        for (const char * path : replay) {
            const std::vector<uint8_t> bytes = read_all(path);
            fprintf(stderr, "replay %s (%zu bytes)\n", path, bytes.size());
            harness::run_program(opt, bytes.data(), bytes.size());
        }
        harness::print_summary();
        return 0;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const char * last_program = getenv("FUZZ_APP_LAST_PROGRAM");
    std::vector<uint8_t> bytes(size);
    for (uint64_t i = 0; i < iters; ++i) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed >= seconds) {
            break;
        }
        std::mt19937_64 rng(seed + i);
        for (uint8_t & b : bytes) {
            b = (uint8_t) rng();
        }
        fprintf(stderr, "program seed %llu\n", (unsigned long long) (seed + i));
        if (last_program != nullptr) {
            // The bytes of the program that runs, thus a crash leaves its input for libFuzzer.
            std::ofstream out(last_program, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char *>(bytes.data()), (std::streamsize) bytes.size());
        }
        harness::run_program(opt, bytes.data(), bytes.size());
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "driver: %.1f s, seeds %llu to %llu\n", elapsed, (unsigned long long) seed,
            (unsigned long long) (seed + harness::counters().programs - 1));
    harness::print_summary();
    return 0;
}
