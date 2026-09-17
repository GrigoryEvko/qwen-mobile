#include "trace.h"

#include <dlfcn.h>

namespace {

// The function types of the ATrace C API in libandroid.so (API 23).
using FnBegin     = void (*)(const char *);
using FnEnd       = void (*)();
using FnIsEnabled = bool (*)();

struct Api {
    FnBegin     begin      = nullptr;
    FnEnd       end        = nullptr;
    FnIsEnabled is_enabled = nullptr;
    bool        loaded     = false;
};

/** Load the ATrace entry points one time. Missing symbols leave the API unusable. */
const Api & api() {
    static Api a = [] {
        Api r;
        void * lib = dlopen("libandroid.so", RTLD_NOW);
        if (lib == nullptr) {
            return r;
        }
        r.begin      = (FnBegin)     dlsym(lib, "ATrace_beginSection");
        r.end        = (FnEnd)       dlsym(lib, "ATrace_endSection");
        r.is_enabled = (FnIsEnabled) dlsym(lib, "ATrace_isEnabled");
        r.loaded     = r.begin && r.end && r.is_enabled;
        return r;
    }();
    return a;
}

} // namespace

TraceSection::TraceSection(const char * name) {
    const Api & a = api();
    if (a.loaded && a.is_enabled()) {
        a.begin(name);
        open_ = true;
    }
}

TraceSection::~TraceSection() {
    if (open_) {
        api().end();
    }
}

bool TraceSection::available() {
    return api().loaded;
}
