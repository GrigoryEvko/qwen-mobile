// fuzz_envparse: the backend init (ggml_hexagon_init) with fuzzed environment
// variables: the device list GGML_HEXAGON_DEVICES (and the old GGML_HEXAGON_NDEV),
// the arch gate (GGML_HEXAGON_ARCH, the queried arch, and ADSP_LIBRARY_PATH that
// ggml_hexagon_htp_lib_found of patches/hexagon-arch/0001 reads), the profile
// list GGML_HEXAGON_PROFILE, the op filter GGML_HEXAGON_OPFILTER (a regex), and
// the numeric switches. The invariants: no crash, no exception out of the init,
// no hang (the libFuzzer timeout), at most 16 devices, and a device only when the
// arch is in [73, 81] and the library of that arch is in ADSP_LIBRARY_PATH.
//
// ADSP_LIBRARY_PATH can hold a directory of the harness that has an empty
// libggml-htp-v79.so: envparse-lib next to the fuzz binary. The parallel jobs of a
// run share it, and a run that ends with a kill leaves no temporary directory.

#include "fake_dsp.h"
#include "fuzz_death.h"
#include "harness.h"
#include "hexhost.h"

#include <fuzzer/FuzzedDataProvider.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

namespace {

std::string g_libdir;   // a directory with libggml-htp-v79.so

// Makes the directory envparse-lib next to the fuzz binary with an empty libggml-htp-v79.so, and
// gives its path, or an empty string when the directory cannot be made.
std::string make_libdir() {
    std::error_code             ec;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) {
        return "";
    }
    const std::filesystem::path dir = exe.parent_path() / "envparse-lib";
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        return "";
    }
    // "a" keeps the file of a parallel job that made it first
    FILE * f = fopen((dir / "libggml-htp-v79.so").c_str(), "a");
    if (!f) {
        return "";
    }
    fclose(f);
    return dir.string();
}

// Gives a short printable string from the input, with the characters that the parsers split on.
std::string fuzz_string(FuzzedDataProvider & fdp, size_t max_len) {
    static const char alphabet[] = "HTP0123456789[]:,-; vV/x";
    const size_t n = fdp.ConsumeIntegralInRange<size_t>(0, max_len);
    std::string  s;
    for (size_t i = 0; i < n; i++) {
        if (fdp.ConsumeIntegralInRange<int>(0, 7) == 0) {
            s += (char) fdp.ConsumeIntegralInRange<int>(33, 126);
        } else {
            s += alphabet[fdp.ConsumeIntegralInRange<size_t>(0, sizeof(alphabet) - 2)];
        }
    }
    return s;
}

// Sets or clears an environment variable from the input.
void fuzz_env(FuzzedDataProvider & fdp, const char * name, const std::string & value) {
    if (fdp.ConsumeBool()) {
        setenv(name, value.c_str(), 1);
    } else {
        unsetenv(name);
    }
}

} // namespace

extern "C" int LLVMFuzzerInitialize(int * argc, char *** argv) {
    (void) argc;
    (void) argv;
    harness::init();
    g_libdir = make_libdir();
    if (g_libdir.empty()) {
        fprintf(stderr, "fuzz_envparse: cannot make the directory envparse-lib next to the fuzz binary: "
                        "the inputs with the library directory run without it\n");
    }
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
    FuzzedDataProvider fdp(data, size);

    fakedsp::config cfg;
    static const int arches[] = { 79, 79, 73, 75, 81, 68, 69, 85, 0 };
    cfg.arch      = arches[fdp.ConsumeIntegralInRange<size_t>(0, 8)];
    cfg.discovery = fdp.ConsumeIntegralInRange<int>(0, 1);
    fakedsp::configure(cfg);

    // The checks after the init look at the behavior of the init, thus a backend with a correct
    // parser passes the regression inputs.
    std::string dev_value;
    {
        // A physical range "A-B" of GGML_HEXAGON_DEVICES: the parser loops over the range, thus a
        // range must be short (the watchdog below)
        dev_value = fuzz_string(fdp, 48);
        if (fdp.ConsumeBool()) {
            setenv("GGML_HEXAGON_DEVICES", dev_value.c_str(), 1);
        } else {
            unsetenv("GGML_HEXAGON_DEVICES");
        }
    }
    fuzz_env(fdp, "GGML_HEXAGON_NDEV", fuzz_string(fdp, 8));
    fuzz_env(fdp, "GGML_HEXAGON_ARCH", fuzz_string(fdp, 6));
    bool profile_empty = false;
    {
        // The list of PMU events is empty in mode 1, and for a list that has not 1 or 8 items. The
        // init then prints the empty list with vec_to_str.
        const std::string prof  = fuzz_string(fdp, 24);
        uint32_t          first = 0;
        const int         n     = hexhost::profile_items(prof.c_str(), &first);
        const bool        empty = n >= 0 && n != 8 && !(n == 1 && first != 1);
        if (fdp.ConsumeBool()) {
            setenv("GGML_HEXAGON_PROFILE", prof.c_str(), 1);
            profile_empty = empty;
        } else {
            unsetenv("GGML_HEXAGON_PROFILE");
        }
    }
    {
        // The default trace size is the op batch times 256: a large OPBATCH must not overflow it
        const std::string ob = fuzz_string(fdp, 8);
        if (fdp.ConsumeBool()) {
            setenv("GGML_HEXAGON_OPBATCH", ob.c_str(), 1);
        } else {
            unsetenv("GGML_HEXAGON_OPBATCH");
        }
    }
    fuzz_env(fdp, "GGML_HEXAGON_OPQUEUE", fuzz_string(fdp, 8));
    fuzz_env(fdp, "GGML_HEXAGON_VMEM", fuzz_string(fdp, 12));
    fuzz_env(fdp, "GGML_HEXAGON_MBUF", fuzz_string(fdp, 12));
    fuzz_env(fdp, "GGML_HEXAGON_NHVX", fuzz_string(fdp, 6));

    // the op filter: a regex from the input (the init must survive a regex that is not valid)
    if (fdp.ConsumeIntegralInRange<int>(0, 7) == 0) {
        setenv("GGML_HEXAGON_OPFILTER", fuzz_string(fdp, 12).c_str(), 1);
    } else {
        unsetenv("GGML_HEXAGON_OPFILTER");
    }

    // ADSP_LIBRARY_PATH: the directory of the harness, other directories, empty parts
    std::string path;
    const int   n_parts = fdp.ConsumeIntegralInRange<int>(0, 4);
    for (int i = 0; i < n_parts; i++) {
        if (i) {
            path += ';';
        }
        const int kind = fdp.ConsumeIntegralInRange<int>(0, 3);
        path += kind == 0 ? g_libdir : kind == 1 ? std::string("/nonexistent/dsp") : kind == 2 ? std::string("") : fuzz_string(fdp, 16);
    }
    const bool path_set = fdp.ConsumeBool();
    if (path_set) {
        setenv("ADSP_LIBRARY_PATH", path.c_str(), 1);
    } else {
        unsetenv("ADSP_LIBRARY_PATH");
    }

    // The init runs under a watchdog: a range of devices can make it loop for a very long time
    size_t n_dev = 0;
    {
        std::mutex              m;
        std::condition_variable cv;
        bool                    done = false;
        std::thread             watchdog([&] {
            std::unique_lock<std::mutex> lock(m);
            if (!cv.wait_for(lock, std::chrono::seconds(20), [&] { return done; })) {
                fakedsp::violation("env-devices-range", "ggml_hexagon_init runs for more than 20 s with "
                                   "GGML_HEXAGON_DEVICES='%s' (the loop over a physical range, ggml-hexagon.cpp:8108)",
                                   dev_value.c_str());
            }
        });
        auto stop = [&] {
            {
                std::lock_guard<std::mutex> lock(m);
                done = true;
            }
            cv.notify_one();
            watchdog.join();
        };
        try {
            n_dev = hexhost::init_from_env();
        } catch (const std::exception & e) {
            stop();
            fakedsp::violation("env-exception", "ggml_hexagon_init throws %s: %s (the process terminates at the backend init)",
                               typeid(e).name(), e.what());
            return 0;
        }
        stop();
    }

    // The behavior after the init
    if (profile_empty && hexhost::profile_empty_size() != 0) {
        fakedsp::violation("env-profile-empty", "an empty PMU event list gives a string of %zu characters: vec_to_str "
                           "(ggml-hexagon.cpp:7789) calls pop_back on an empty string", hexhost::profile_empty_size());
    }
    if (hexhost::max_device_group() > 16) {
        fakedsp::violation("env-devices-range", "GGML_HEXAGON_DEVICES='%s' gives a device group of %zu physical devices, "
                           "the maximum is 16", dev_value.c_str(), hexhost::max_device_group());
    }
    if (!getenv("GGML_HEXAGON_OPTRACE")) {
        const long long want = std::min<long long>(std::max<long long>((long long) hexhost::get_options().opbatch * 256, 0),
                                                   INT32_MAX);
        if (hexhost::optrace() != want) {
            fakedsp::violation("env-int-overflow", "opbatch %d gives the trace size %d, and the 64-bit product clamped to "
                               "int is %lld: opbatch * 256 (ggml-hexagon.cpp:7983) overflows int",
                               hexhost::get_options().opbatch, hexhost::optrace(), want);
        }
    }

    if (n_dev > 16) {
        fakedsp::violation("env-ndev", "ggml_hexagon_init registers %zu devices, the maximum is 16", n_dev);
    }
    // The arch gate: with no GGML_HEXAGON_ARCH, the queried arch must be in [73, 81] and its library in the path
    const char * arch_env = getenv("GGML_HEXAGON_ARCH");
    if (!arch_env && n_dev > 0) {
        const int a = cfg.arch;
        if (a < 73 || a <= 0) {
            fakedsp::violation("env-arch-gate", "a device registers for the queried arch v%d, below v73", a);
        }
        if (path_set && !path.empty()) {
            const int  eff = a > 81 ? 81 : a;
            const bool has = eff == 79 && !g_libdir.empty() && path.find(g_libdir) != std::string::npos;
            if (!has && eff == 79) {
                fakedsp::violation("env-lib-gate", "a device registers for v79 while ADSP_LIBRARY_PATH '%s' has no libggml-htp-v79.so",
                                   path.c_str());
            }
        }
    }
    return 0;
}
