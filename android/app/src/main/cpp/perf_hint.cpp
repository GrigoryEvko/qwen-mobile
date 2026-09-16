#include "perf_hint.h"

#include <android/log.h>
#include <dlfcn.h>

namespace {

// The function types of the ADPF C API in libandroid.so (API 33).
using FnGetManager   = void * (*)();
using FnCreate       = void * (*)(void *, const int32_t *, size_t, int64_t);
using FnUpdateTarget = int (*)(void *, int64_t);
using FnReport       = int (*)(void *, int64_t);
using FnClose        = void (*)(void *);

struct Api {
    FnGetManager   get_manager   = nullptr;
    FnCreate       create        = nullptr;
    FnUpdateTarget update_target = nullptr;
    FnReport       report        = nullptr;
    FnClose        close         = nullptr;
    bool           loaded        = false;
};

/** Load the ADPF entry points one time. Missing symbols leave the API unusable. */
const Api & api() {
    static Api a = [] {
        Api r;
        void * lib = dlopen("libandroid.so", RTLD_NOW);
        if (lib == nullptr) {
            return r;
        }
        r.get_manager   = (FnGetManager)   dlsym(lib, "APerformanceHint_getManager");
        r.create        = (FnCreate)       dlsym(lib, "APerformanceHint_createSession");
        r.update_target = (FnUpdateTarget) dlsym(lib, "APerformanceHint_updateTargetWorkDuration");
        r.report        = (FnReport)       dlsym(lib, "APerformanceHint_reportActualWorkDuration");
        r.close         = (FnClose)        dlsym(lib, "APerformanceHint_closeSession");
        r.loaded = r.get_manager && r.create && r.update_target && r.report && r.close;
        return r;
    }();
    return a;
}

} // namespace

PerfHintSession::PerfHintSession(const std::vector<int32_t> & tids, int64_t target_ns) {
    const Api & a = api();
    if (!a.loaded || tids.empty()) {
        __android_log_print(ANDROID_LOG_WARN, "QwenMobile", "ADPF is not available on this phone");
        return;
    }
    void * manager = a.get_manager();
    if (manager == nullptr) {
        return;
    }
    session_ = a.create(manager, tids.data(), tids.size(), target_ns);
    __android_log_print(ANDROID_LOG_INFO, "QwenMobile", "ADPF session %s for %zu threads",
                        session_ ? "open" : "refused", tids.size());
}

PerfHintSession::~PerfHintSession() {
    if (session_ != nullptr) {
        api().close(session_);
    }
}

void PerfHintSession::set_target(int64_t target_ns) {
    if (session_ != nullptr) {
        api().update_target(session_, target_ns);
    }
}

void PerfHintSession::report(int64_t actual_ns) {
    if (session_ != nullptr) {
        api().report(session_, actual_ns);
    }
}
