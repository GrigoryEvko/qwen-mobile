// The run-time loader of libcdsprpc.so. Refer to cdsprpc.h.

#include "cdsprpc.h"

#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>

#include "AEEStdErr.h"
#include "remote.h"

typedef int (*handle64_open_fn)(const char * name, remote_handle64 * ph);
typedef int (*handle64_invoke_fn)(remote_handle64 h, uint32_t scalars, remote_arg * pra);
typedef int (*handle64_close_fn)(remote_handle64 h);
typedef int (*handle_control_fn)(uint32_t req, void * data, uint32_t datalen);
typedef int (*session_control_fn)(uint32_t req, void * data, uint32_t datalen);

static struct {
    void *             lib;
    handle64_open_fn   handle64_open;
    handle64_invoke_fn handle64_invoke;
    handle64_close_fn  handle64_close;
    handle_control_fn  handle_control;
    session_control_fn session_control;
} g_rpc;

// Find one function of the library. Returns NULL after an error message.
static void * find(const char * name) {
    void * fn = dlsym(g_rpc.lib, name);
    if (fn == NULL) {
        const char * why = dlerror();
        fprintf(stderr, "isaprobe: error: libcdsprpc.so has no function %s (%s). The FastRPC library of this phone "
                        "is too old for the probe.\n", name, why != NULL ? why : "no reason given");
    }
    return fn;
}

int cdsprpc_load(void) {
    if (g_rpc.lib != NULL) {
        return 0;
    }
    g_rpc.lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (g_rpc.lib == NULL) {
        const char * why = dlerror();
        fprintf(stderr, "isaprobe: error: dlopen(libcdsprpc.so) failed: %s. The phone must have a compute DSP with "
                        "FastRPC (/vendor/lib64/libcdsprpc.so). Run the program from adb shell, not from an app.\n",
                why != NULL ? why : "no reason given");
        return -1;
    }
    g_rpc.handle64_open   = (handle64_open_fn) find("remote_handle64_open");
    g_rpc.handle64_invoke = (handle64_invoke_fn) find("remote_handle64_invoke");
    g_rpc.handle64_close  = (handle64_close_fn) find("remote_handle64_close");
    g_rpc.handle_control  = (handle_control_fn) find("remote_handle_control");
    g_rpc.session_control = (session_control_fn) find("remote_session_control");
    if (g_rpc.handle64_open == NULL || g_rpc.handle64_invoke == NULL || g_rpc.handle64_close == NULL ||
        g_rpc.handle_control == NULL || g_rpc.session_control == NULL) {
        dlclose(g_rpc.lib);
        g_rpc.lib = NULL;
        return -1;
    }
    return 0;
}

// The FastRPC functions that the qaic stub and isaprobe.c call. Before cdsprpc_load, each one
// gives AEE_EUNABLETOLOAD.

int remote_handle64_open(const char * name, remote_handle64 * ph) {
    return g_rpc.lib != NULL ? g_rpc.handle64_open(name, ph) : AEE_EUNABLETOLOAD;
}

int remote_handle64_invoke(remote_handle64 h, uint32_t scalars, remote_arg * pra) {
    return g_rpc.lib != NULL ? g_rpc.handle64_invoke(h, scalars, pra) : AEE_EUNABLETOLOAD;
}

int remote_handle64_close(remote_handle64 h) {
    return g_rpc.lib != NULL ? g_rpc.handle64_close(h) : AEE_EUNABLETOLOAD;
}

int remote_handle_control(uint32_t req, void * data, uint32_t datalen) {
    return g_rpc.lib != NULL ? g_rpc.handle_control(req, data, datalen) : AEE_EUNABLETOLOAD;
}

int remote_session_control(uint32_t req, void * data, uint32_t datalen) {
    return g_rpc.lib != NULL ? g_rpc.session_control(req, data, datalen) : AEE_EUNABLETOLOAD;
}
