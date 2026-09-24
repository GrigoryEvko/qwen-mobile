// The run-time loader of libcdsprpc.so. Refer to rpc.h.

#include "rpc.h"

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "AEEStdErr.h"
#include "dspqueue.h"
#include "remote.h"
#include "rpcmem.h"

typedef int (*handle64_open_fn)(const char * name, remote_handle64 * ph);
typedef int (*handle64_invoke_fn)(remote_handle64 h, uint32_t scalars, remote_arg * pra);
typedef int (*handle64_close_fn)(remote_handle64 h);
typedef int (*handle_control_fn)(uint32_t req, void * data, uint32_t datalen);
typedef int (*handle64_control_fn)(remote_handle64 h, uint32_t req, void * data, uint32_t datalen);
typedef int (*session_control_fn)(uint32_t req, void * data, uint32_t datalen);
typedef void * (*rpcmem_alloc_fn)(int heapid, uint32 flags, int size);
typedef void * (*rpcmem_alloc2_fn)(int heapid, uint32 flags, size_t size);
typedef void (*rpcmem_free_fn)(void * po);
typedef int (*rpcmem_to_fd_fn)(void * po);
typedef int (*fastrpc_mmap_fn)(int domain, int fd, void * addr, int offset, size_t length, enum fastrpc_map_flags flags);
typedef int (*fastrpc_munmap_fn)(int domain, int fd, void * addr, size_t length);
typedef AEEResult (*dq_create_fn)(int domain, uint32_t flags, uint32_t req_size, uint32_t resp_size,
                                  dspqueue_callback_t packet_cb, dspqueue_callback_t error_cb, void * ctx,
                                  dspqueue_t * queue);
typedef AEEResult (*dq_close_fn)(dspqueue_t queue);
typedef AEEResult (*dq_export_fn)(dspqueue_t queue, uint64_t * queue_id);
typedef AEEResult (*dq_write_fn)(dspqueue_t queue, uint32_t flags, uint32_t num_buffers,
                                 struct dspqueue_buffer * buffers, uint32_t message_length,
                                 const uint8_t * message, uint32_t timeout_us);
typedef AEEResult (*dq_read_fn)(dspqueue_t queue, uint32_t * flags, uint32_t max_buffers, uint32_t * num_buffers,
                                struct dspqueue_buffer * buffers, uint32_t max_message_length,
                                uint32_t * message_length, uint8_t * message, uint32_t timeout_us);
typedef AEEResult (*dq_read_noblock_fn)(dspqueue_t queue, uint32_t * flags, uint32_t max_buffers,
                                        uint32_t * num_buffers, struct dspqueue_buffer * buffers,
                                        uint32_t max_message_length, uint32_t * message_length, uint8_t * message);

static struct {
    void *              lib;
    handle64_open_fn    handle64_open;
    handle64_invoke_fn  handle64_invoke;
    handle64_close_fn   handle64_close;
    handle_control_fn   handle_control;
    handle64_control_fn handle64_control;
    session_control_fn  session_control;
    rpcmem_alloc_fn     alloc;
    rpcmem_alloc2_fn    alloc2;
    rpcmem_free_fn      free;
    rpcmem_to_fd_fn     to_fd;
    fastrpc_mmap_fn     mmap;
    fastrpc_munmap_fn   munmap;
    dq_create_fn        dq_create;
    dq_close_fn         dq_close;
    dq_export_fn        dq_export;
    dq_write_fn         dq_write;
    dq_read_fn          dq_read;
    dq_read_noblock_fn  dq_read_noblock;
} g_rpc;

// Find one function of the library. Returns NULL, with a message when "required" is 1.
static void * find(const char * name, int required) {
    void * fn = dlsym(g_rpc.lib, name);
    if (fn == NULL && required) {
        const char * why = dlerror();
        fprintf(stderr, "ddrbw: error: libcdsprpc.so has no function %s (%s). The FastRPC library of this phone "
                        "is too old for the probe.\n", name, why != NULL ? why : "no reason given");
    }
    return fn;
}

int rpc_load(void) {
    if (g_rpc.lib != NULL) {
        return 0;
    }
    g_rpc.lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (g_rpc.lib == NULL) {
        const char * why = dlerror();
        fprintf(stderr, "ddrbw: error: dlopen(libcdsprpc.so) failed: %s. The phone must have a compute DSP with "
                        "FastRPC (/vendor/lib64/libcdsprpc.so). Run the program from adb shell.\n",
                why != NULL ? why : "no reason given");
        return -1;
    }
    g_rpc.handle64_open    = (handle64_open_fn) find("remote_handle64_open", 1);
    g_rpc.handle64_invoke  = (handle64_invoke_fn) find("remote_handle64_invoke", 1);
    g_rpc.handle64_close   = (handle64_close_fn) find("remote_handle64_close", 1);
    g_rpc.handle_control   = (handle_control_fn) find("remote_handle_control", 1);
    g_rpc.handle64_control = (handle64_control_fn) find("remote_handle64_control", 0);
    g_rpc.session_control  = (session_control_fn) find("remote_session_control", 1);
    g_rpc.alloc            = (rpcmem_alloc_fn) find("rpcmem_alloc", 1);
    g_rpc.alloc2           = (rpcmem_alloc2_fn) find("rpcmem_alloc2", 0);
    g_rpc.free             = (rpcmem_free_fn) find("rpcmem_free", 1);
    g_rpc.to_fd            = (rpcmem_to_fd_fn) find("rpcmem_to_fd", 1);
    g_rpc.mmap             = (fastrpc_mmap_fn) find("fastrpc_mmap", 1);
    g_rpc.munmap           = (fastrpc_munmap_fn) find("fastrpc_munmap", 1);
    g_rpc.dq_create        = (dq_create_fn) find("dspqueue_create", 0);
    g_rpc.dq_close         = (dq_close_fn) find("dspqueue_close", 0);
    g_rpc.dq_export        = (dq_export_fn) find("dspqueue_export", 0);
    g_rpc.dq_write         = (dq_write_fn) find("dspqueue_write", 0);
    g_rpc.dq_read          = (dq_read_fn) find("dspqueue_read", 0);
    g_rpc.dq_read_noblock  = (dq_read_noblock_fn) find("dspqueue_read_noblock", 0);
    if (g_rpc.handle64_open == NULL || g_rpc.handle64_invoke == NULL || g_rpc.handle64_close == NULL ||
        g_rpc.handle_control == NULL || g_rpc.session_control == NULL || g_rpc.alloc == NULL ||
        g_rpc.free == NULL || g_rpc.to_fd == NULL || g_rpc.mmap == NULL || g_rpc.munmap == NULL) {
        dlclose(g_rpc.lib);
        g_rpc.lib = NULL;
        return -1;
    }
    return 0;
}

// The functions that the qaic stub and ddrbw.c call. Before rpc_load, or when the library has no
// such function, each one gives AEE_EUNABLETOLOAD or AEE_EUNSUPPORTED.

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

int remote_handle64_control(remote_handle64 h, uint32_t req, void * data, uint32_t datalen) {
    return g_rpc.handle64_control != NULL ? g_rpc.handle64_control(h, req, data, datalen) : AEE_EUNSUPPORTED;
}

int remote_session_control(uint32_t req, void * data, uint32_t datalen) {
    return g_rpc.lib != NULL ? g_rpc.session_control(req, data, datalen) : AEE_EUNABLETOLOAD;
}

void * rpcmem_alloc(int heapid, uint32 flags, int size) {
    return g_rpc.alloc != NULL ? g_rpc.alloc(heapid, flags, size) : NULL;
}

void * rpcmem_alloc2(int heapid, uint32 flags, size_t size) {
    if (g_rpc.alloc2 != NULL) {
        return g_rpc.alloc2(heapid, flags, size);
    }
    return size <= 0x7fffffffu ? rpcmem_alloc(heapid, flags, (int) size) : NULL;
}

void rpcmem_free(void * po) {
    if (g_rpc.free != NULL) {
        g_rpc.free(po);
    }
}

int rpcmem_to_fd(void * po) {
    return g_rpc.to_fd != NULL ? g_rpc.to_fd(po) : -1;
}

int fastrpc_mmap(int domain, int fd, void * addr, int offset, size_t length, enum fastrpc_map_flags flags) {
    return g_rpc.mmap != NULL ? g_rpc.mmap(domain, fd, addr, offset, length, flags) : AEE_EUNABLETOLOAD;
}

int fastrpc_munmap(int domain, int fd, void * addr, size_t length) {
    return g_rpc.munmap != NULL ? g_rpc.munmap(domain, fd, addr, length) : AEE_EUNABLETOLOAD;
}

AEEResult dspqueue_create(int domain, uint32_t flags, uint32_t req_queue_size, uint32_t resp_queue_size,
                          dspqueue_callback_t packet_callback, dspqueue_callback_t error_callback,
                          void * callback_context, dspqueue_t * queue) {
    return g_rpc.dq_create != NULL ? g_rpc.dq_create(domain, flags, req_queue_size, resp_queue_size, packet_callback,
                                                     error_callback, callback_context, queue)
                                   : AEE_EUNSUPPORTED;
}

AEEResult dspqueue_close(dspqueue_t queue) {
    return g_rpc.dq_close != NULL ? g_rpc.dq_close(queue) : AEE_EUNSUPPORTED;
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    return g_rpc.dq_export != NULL ? g_rpc.dq_export(queue, queue_id) : AEE_EUNSUPPORTED;
}

AEEResult dspqueue_write(dspqueue_t queue, uint32_t flags, uint32_t num_buffers, struct dspqueue_buffer * buffers,
                         uint32_t message_length, const uint8_t * message, uint32_t timeout_us) {
    return g_rpc.dq_write != NULL ? g_rpc.dq_write(queue, flags, num_buffers, buffers, message_length, message, timeout_us)
                                  : AEE_EUNSUPPORTED;
}

AEEResult dspqueue_read(dspqueue_t queue, uint32_t * flags, uint32_t max_buffers, uint32_t * num_buffers,
                        struct dspqueue_buffer * buffers, uint32_t max_message_length, uint32_t * message_length,
                        uint8_t * message, uint32_t timeout_us) {
    return g_rpc.dq_read != NULL ? g_rpc.dq_read(queue, flags, max_buffers, num_buffers, buffers, max_message_length,
                                                 message_length, message, timeout_us)
                                 : AEE_EUNSUPPORTED;
}

AEEResult dspqueue_read_noblock(dspqueue_t queue, uint32_t * flags, uint32_t max_buffers, uint32_t * num_buffers,
                                struct dspqueue_buffer * buffers, uint32_t max_message_length,
                                uint32_t * message_length, uint8_t * message) {
    return g_rpc.dq_read_noblock != NULL ? g_rpc.dq_read_noblock(queue, flags, max_buffers, num_buffers, buffers,
                                                                 max_message_length, message_length, message)
                                         : AEE_EUNSUPPORTED;
}
