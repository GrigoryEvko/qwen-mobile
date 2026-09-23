// The DSP side of the ISA silicon probe: the implementation of isaprobe.idl.
//
// The library runs in the unsigned protection domain (PD) of the compute DSP. For each handle,
// the first call of info() or run() does the setup:
//   1. The power votes for the HVX and the HMX, as htp/main.c of the llama.cpp backend does. The
//      HMX request type is HAP_power_set_HMX before v75 and HAP_power_set_HMX_v2 from v75.
//   2. The VTCM and the HMX from the compute resource manager (HAP_compute_res_acquire). If the
//      HMX is not available, the library holds the VTCM only. If the VTCM is not available, the
//      library holds nothing and the HVX-only ops still run.
//
// For each call of run() (a census op) and of kernel_run() (a candidate kernel), the library:
//   1. Copies each input into a buffer with 128-byte alignment and a tail of zero bytes. Thus
//      the kernel always gets aligned pointers, and a read after the end of an input gives zeros.
//   2. Starts a QuRT thread with a 256 KB stack, because the stack of the FastRPC thread is small.
//      The thread locks a 128-byte HVX context (qurt_hvx_lock), locks the HMX if the setup got it
//      (HAP_compute_res_hmx_lock), calls isa_run or the kernel, and unlocks the two units.
//   3. Examines the guard bytes after the output, and copies the output into the FastRPC buffer.
// kernel_run() also measures each iteration of the kernel with the processor cycle counter
// (UPCYCLE) and the QTimer, and gives the minimum.
//
// Cache state: FastRPC copies the IDL buffers and does the cache maintenance for them. The
// library also flushes the output buffer from the DSP cache before the return.
//
// The library also gives the lab functions for the VTCM (lab_vtcm_base, lab_vtcm_size,
// lab_vtcm_alloc) and lab_hmx_enable as weak symbols, thus a census kernel that uses them links
// without a change. lab_hmx_enable does nothing here, because the SSR write of the lab is not
// permitted in a user PD and HAP_compute_res_hmx_lock enables the HMX for the thread.

#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_power.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <qurt_memory.h>
#include <qurt_thread.h>

#include "candidates.h"
#include "isaprobe.h"
#include "isaprobe_status.h"

#ifdef ISAPROBE_HAVE_ISA_KERNELS_H
// The real census kernels: the op table, the stream names, and isa_run. A different declaration
// of isa_run below is then a compile error.
#    include "isa_kernels.h"
#endif

// A DSP image without one of these functions gives a NULL address instead of a load failure.
// qurt_hvx_lock and qurt_hvx_unlock are deprecated in QuRT: without them the HVX scheduler of QuRT
// gives the thread an HVX context at its first HVX instruction.
#pragma weak qurt_sysenv_get_arch_version
#pragma weak qurt_hvx_lock
#pragma weak qurt_hvx_unlock

#define ISAPROBE_ALIGN          128u
#define ISAPROBE_INPUT_TAIL     16384u    // zero bytes after each input
#define ISAPROBE_OUTPUT_GUARD   16384u    // guard bytes after the output
#define ISAPROBE_GUARD_BYTE     0xA5u
#define ISAPROBE_STACK_BYTES    (256u * 1024u)
#define ISAPROBE_ACQUIRE_US     2000000u  // wait for the VTCM and the HMX, in microseconds

int isa_run(int op_id, const void * in0, const void * in1, const void * in2, void * out, int n_vectors);

struct probe_ctx {
    int          setup_done;
    int          setup_status;  // ISAPROBE_SETUP_*
    int          setup_error;   // the error code of that step
    int          hmx_powered;
    int          hmx_held;      // the resource manager gave the HMX
    unsigned int rctx;          // the compute resource context, 0 if none
    uint8_t *    vtcm_base;
    unsigned int vtcm_bytes;
};

// The work of one call on the probe thread.
struct probe_job {
    struct probe_ctx * ctx;
    int (*fn)(void * arg);  // the work: isa_call_fn or kernel_call_fn
    void *             arg;
    int                status;  // the value of fn, or an ISAPROBE_STATUS_* code of a lock
    int                detail;  // the error code of a failed lock
};

// The arguments of one census op.
struct isa_call {
    int          op_id;
    const void * in[3];
    void *       out;
    int          n_vectors;
};

// The arguments and the timing of one candidate kernel.
struct kernel_call {
    const cand_kernel * kernel;
    cand_problem        problem;
    uint32_t            iters;
    uint64_t            pcycles;  // the minimum over the iterations
    uint64_t            usecs;    // the minimum over the iterations
};

// The VTCM of the lab functions. One handle at a time owns it: the last setup sets it.
static uint8_t *    g_vtcm_base;
static size_t       g_vtcm_size;
static size_t       g_vtcm_used;
static int          g_vtcm_overflow;

// Record the first failed setup step. The later steps do not replace it.
static void setup_fail(struct probe_ctx * ctx, int step, int error) {
    FARF(ERROR, "isaprobe: setup step %d failed with 0x%08x", step, (unsigned) error);
    if (ctx->setup_status == ISAPROBE_SETUP_OK) {
        ctx->setup_status = step;
        ctx->setup_error  = error;
    }
}

// Vote for the power of the HVX and the HMX. Returns 0, or the fatal error of the apptype or HVX
// vote. A failure of the HMX vote clears hmx_powered and is not fatal.
static int setup_power(struct probe_ctx * ctx) {
    HAP_power_request_t request;
    int                 err;

    memset(&request, 0, sizeof(request));
    request.type    = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if ((err = HAP_power_set((void *) ctx, &request)) != 0) {
        setup_fail(ctx, ISAPROBE_SETUP_APPTYPE, err);
        return err;
    }

    memset(&request, 0, sizeof(request));
    request.type         = HAP_power_set_HVX;
    request.hvx.power_up = TRUE;
    if ((err = HAP_power_set((void *) ctx, &request)) != 0) {
        setup_fail(ctx, ISAPROBE_SETUP_HVX_POWER, err);
        return err;
    }

    memset(&request, 0, sizeof(request));
#if __HVX_ARCH__ >= 75
    // From v75 the HMX has its own clock. The request is the one of htp/main.c.
    request.type                 = HAP_power_set_HMX_v2;
    request.hmx_v2.set_power     = TRUE;
    request.hmx_v2.power_up      = TRUE;
    request.hmx_v2.set_clock     = TRUE;
    request.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
    request.hmx_v2.min_corner    = HAP_DCVS_EXP_VCORNER_MAX;
    request.hmx_v2.max_corner    = HAP_DCVS_EXP_VCORNER_MAX;
    request.hmx_v2.perf_mode     = HAP_CLK_PERF_HIGH;
#else
    // Before v75 the HMX_v2 request with a clock gives AEE_EBADPARM.
    request.type         = HAP_power_set_HMX;
    request.hmx.power_up = TRUE;
#endif
    if ((err = HAP_power_set((void *) ctx, &request)) != 0) {
        setup_fail(ctx, ISAPROBE_SETUP_HMX_POWER, err);
        ctx->hmx_powered = 0;
    } else {
        ctx->hmx_powered = 1;
    }
    return 0;
}

// Acquire the full VTCM, together with the HMX when with_hmx is 1. Returns the context ID, or 0.
static unsigned int acquire_vtcm(unsigned int vtcm_bytes, int with_hmx, compute_res_attr_t * attr) {
    HAP_compute_res_attr_init(attr);
    HAP_compute_res_attr_set_serialize(attr, 0);
    HAP_compute_res_attr_set_vtcm_param_v2(attr, vtcm_bytes, vtcm_bytes, vtcm_bytes);
    HAP_compute_res_attr_set_hmx_param(attr, with_hmx ? 1 : 0);
    return HAP_compute_res_acquire(attr, ISAPROBE_ACQUIRE_US);
}

// Get the VTCM and the HMX. A failure removes the HMX or the VTCM, and it is never fatal.
static void setup_resources(struct probe_ctx * ctx) {
    unsigned int       vtcm_bytes = 0;
    compute_res_attr_t attr;
    int                err;

    err = HAP_compute_res_query_VTCM(0, &vtcm_bytes, NULL, NULL, NULL);
    if (err != 0 || vtcm_bytes == 0) {
        setup_fail(ctx, ISAPROBE_SETUP_VTCM_QUERY, err != 0 ? err : AEE_ENOMEMORY);
        return;
    }

    unsigned int rctx = 0;
    if (ctx->hmx_powered) {
        rctx = acquire_vtcm(vtcm_bytes, 1, &attr);
        if (rctx == 0) {
            setup_fail(ctx, ISAPROBE_SETUP_ACQUIRE_HMX, AEE_ERESOURCENOTFOUND);
        } else {
            ctx->hmx_held = 1;
        }
    }
    if (rctx == 0) {
        rctx = acquire_vtcm(vtcm_bytes, 0, &attr);
        if (rctx == 0) {
            setup_fail(ctx, ISAPROBE_SETUP_ACQUIRE_VTCM, AEE_ERESOURCENOTFOUND);
            return;
        }
    }

    void *       ptr  = NULL;
    unsigned int size = 0;
    err               = HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &ptr, &size);
    if (err != 0 || ptr == NULL || size == 0) {
        setup_fail(ctx, ISAPROBE_SETUP_VTCM_PTR, err != 0 ? err : AEE_ENOMEMORY);
        HAP_compute_res_release(rctx);
        ctx->hmx_held = 0;
        return;
    }

    ctx->rctx       = rctx;
    ctx->vtcm_base  = (uint8_t *) ptr;
    ctx->vtcm_bytes = size;
    g_vtcm_base     = ctx->vtcm_base;
    g_vtcm_size     = size;
    g_vtcm_used     = 0;
}

// Do the setup one time for each handle. Returns 0, or the error of a fatal setup step.
static int setup(struct probe_ctx * ctx) {
    if (ctx->setup_done) {
        return ISAPROBE_SETUP_IS_FATAL(ctx->setup_status) ? ctx->setup_error : 0;
    }
    ctx->setup_done = 1;

    // Send all FARF levels to the log (logcat tag adsprpc), as htp/main.c does.
    HAP_setFARFRuntimeLoggingParams(0xffff, NULL, 0);

    const int err = setup_power(ctx);
    if (err != 0) {
        return err;
    }
    setup_resources(ctx);
    FARF(ALWAYS, "isaprobe: setup status %d error 0x%08x hmx %d vtcm %u bytes at %p", ctx->setup_status,
         (unsigned) ctx->setup_error, ctx->hmx_held, ctx->vtcm_bytes, (void *) ctx->vtcm_base);
    return 0;
}

// Return n rounded up to a multiple of the power of two a.
static size_t round_up(size_t n, size_t a) {
    return (n + a - 1) & ~(a - 1);
}

// Return an aligned copy of an input with a zero tail, or NULL for an empty input. *fail is 1
// when the allocation fails.
static void * copy_input(const uint8 * src, int len, int * fail) {
    if (len <= 0) {
        return NULL;
    }
    const size_t n   = (size_t) len;
    uint8_t *    buf = (uint8_t *) memalign(ISAPROBE_ALIGN, round_up(n, ISAPROBE_ALIGN) + ISAPROBE_INPUT_TAIL);
    if (buf == NULL) {
        *fail = 1;
        return NULL;
    }
    memcpy(buf, src, n);
    memset(buf + n, 0, round_up(n, ISAPROBE_ALIGN) - n + ISAPROBE_INPUT_TAIL);
    return buf;
}

// Return 1 if all guard bytes after the output are intact. O(ISAPROBE_OUTPUT_GUARD).
static int guard_intact(const uint8_t * out, size_t len) {
    for (size_t i = 0; i < ISAPROBE_OUTPUT_GUARD; i++) {
        if (out[len + i] != ISAPROBE_GUARD_BYTE) {
            return 0;
        }
    }
    return 1;
}

// The thread of one op: lock the units, run the kernel, unlock the units.
static void probe_thread(void * arg) {
    struct probe_job * job = (struct probe_job *) arg;

    const int hvx_explicit = 0 != qurt_hvx_lock && 0 != qurt_hvx_unlock;
    if (hvx_explicit) {
        const int hvx_err = qurt_hvx_lock(QURT_HVX_MODE_128B);
        if (hvx_err != QURT_EOK) {
            job->status = ISAPROBE_STATUS_HVX_LOCK;
            job->detail = hvx_err;
            return;
        }
    }

    int hmx_locked = 0;
    if (job->ctx->hmx_held) {
        const int hmx_err = HAP_compute_res_hmx_lock(job->ctx->rctx);
        if (hmx_err != 0) {
            if (hvx_explicit) {
                qurt_hvx_unlock();
            }
            job->status = ISAPROBE_STATUS_HMX_LOCK;
            job->detail = hmx_err;
            return;
        }
        hmx_locked = 1;
    }

    job->status = job->fn(job->arg);

    if (hmx_locked) {
        const int err = HAP_compute_res_hmx_unlock(job->ctx->rctx);
        if (err != 0) {
            FARF(ERROR, "isaprobe: HAP_compute_res_hmx_unlock failed with 0x%08x", (unsigned) err);
        }
    }
    if (hvx_explicit) {
        const int err = qurt_hvx_unlock();
        if (err != QURT_EOK) {
            FARF(ERROR, "isaprobe: qurt_hvx_unlock failed with %d", err);
        }
    }
}

// Run the job on a new QuRT thread with a large stack and wait for it. Returns 0 or an AEE error.
static int run_on_thread(struct probe_job * job) {
    void * stack = memalign(4096, ISAPROBE_STACK_BYTES);
    if (stack == NULL) {
        FARF(ERROR, "isaprobe: no memory for a thread stack of %u bytes", ISAPROBE_STACK_BYTES);
        return AEE_ENOMEMORY;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, stack);
    qurt_thread_attr_set_stack_size(&attr, ISAPROBE_STACK_BYTES);
    qurt_thread_attr_set_priority(&attr, (unsigned short) qurt_thread_get_priority(qurt_thread_get_id()));
    qurt_thread_attr_set_name(&attr, "isaprobe");

    qurt_thread_t tid = 0;
    int           err = qurt_thread_create(&tid, &attr, probe_thread, job);
    if (err != QURT_EOK) {
        FARF(ERROR, "isaprobe: qurt_thread_create failed with %d", err);
        free(stack);
        return AEE_EFAILED;
    }

    // QURT_ENOTHREAD tells that the thread stopped before the join. The job holds the result.
    int exit_status = 0;
    err             = qurt_thread_join(tid, &exit_status);
    free(stack);
    if (err != QURT_EOK && err != QURT_ENOTHREAD) {
        FARF(ERROR, "isaprobe: qurt_thread_join failed with %d", err);
        return AEE_EFAILED;
    }
    return 0;
}

AEEResult isaprobe_open(const char * uri, remote_handle64 * handle) {
    (void) uri;
    struct probe_ctx * ctx = (struct probe_ctx *) calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return AEE_ENOMEMORY;
    }
    *handle = (remote_handle64) ctx;
    return AEE_SUCCESS;
}

AEEResult isaprobe_close(remote_handle64 handle) {
    struct probe_ctx * ctx = (struct probe_ctx *) handle;
    if (ctx == NULL) {
        return AEE_EBADPARM;
    }
    if (ctx->rctx != 0) {
        const int err = HAP_compute_res_release(ctx->rctx);
        if (err != 0) {
            FARF(ERROR, "isaprobe: HAP_compute_res_release failed with 0x%08x", (unsigned) err);
        }
        if (g_vtcm_base == ctx->vtcm_base) {
            g_vtcm_base = NULL;
            g_vtcm_size = 0;
            g_vtcm_used = 0;
        }
    }
    if (ctx->setup_done) {
        // Remove the power votes of this client. An image without the function keeps them until
        // the PD stops.
        const int err = HAP_power_destroy((void *) ctx);
        if (err != 0 && err != AEE_EUNSUPPORTEDAPI) {
            FARF(ERROR, "isaprobe: HAP_power_destroy failed with 0x%08x", (unsigned) err);
        }
    }
    free(ctx);
    return AEE_SUCCESS;
}

AEEResult isaprobe_info(remote_handle64 handle, uint32 * arch, uint32 * skel_arch, uint32 * hvx_threads,
                        uint32 * hmx_count, uint32 * vtcm_bytes, int32 * setup_status, int32 * setup_error,
                        uint32 * n_ops, uint32 * n_streams) {
    struct probe_ctx * ctx = (struct probe_ctx *) handle;
    if (ctx == NULL || arch == NULL || skel_arch == NULL || hvx_threads == NULL || hmx_count == NULL ||
        vtcm_bytes == NULL || setup_status == NULL || setup_error == NULL || n_ops == NULL || n_streams == NULL) {
        return AEE_EBADPARM;
    }
#ifdef ISAPROBE_HAVE_ISA_KERNELS_H
    *n_ops     = ISA_N_OPS;
    *n_streams = ISA_S_COUNT;
#else
    *n_ops     = 0;
    *n_streams = 0;
#endif

    // A fatal setup failure still gives the facts below. The host reads setup_status.
    (void) setup(ctx);

    *arch = 0;
    if (0 != qurt_sysenv_get_arch_version) {
        qurt_arch_version_t v;
        if (qurt_sysenv_get_arch_version(&v) == QURT_EOK) {
            *arch = v.arch_version;
        }
    }
    *skel_arch    = __HVX_ARCH__;
    *hvx_threads  = ((unsigned) qurt_hvx_get_units() >> 8) & 0xFFu;
    *hmx_count    = ctx->hmx_held ? 1u : 0u;
    *vtcm_bytes   = ctx->vtcm_bytes;
    *setup_status = ctx->setup_status;
    *setup_error  = ctx->setup_error;
    return AEE_SUCCESS;
}

// Copy a string into a FastRPC string buffer. Returns 0, or AEE_EBUFFERTOOSMALL.
static AEEResult copy_name(const char * src, char * dst, int dst_len) {
    const size_t n = strlen(src);
    if (dst == NULL || dst_len <= 0 || n >= (size_t) dst_len) {
        return AEE_EBUFFERTOOSMALL;
    }
    memcpy(dst, src, n + 1);
    return AEE_SUCCESS;
}

AEEResult isaprobe_op_info(remote_handle64 handle, uint32 op_id, isaprobe_op * desc, char * name, int nameLen) {
    if (handle == 0 || desc == NULL) {
        return AEE_EBADPARM;
    }
#ifdef ISAPROBE_HAVE_ISA_KERNELS_H
    if (op_id >= ISA_N_OPS) {
        return AEE_EBADPARM;
    }
    const isa_op_desc * op = &isa_ops[op_id];
    memset(desc, 0, sizeof(*desc));
    desc->kind      = op->kind;
    desc->min_arch  = op->min_arch;
    desc->out_type  = op->out_type;
    for (int j = 0; j < 3; j++) {
        desc->in_type[j]   = op->in_type[j];
        desc->in_stream[j] = op->in_stream[j];
        desc->in_bytes[j]  = op->in_bytes[j];
    }
    desc->out_bytes = op->out_bytes;
    desc->n_vectors = op->n_vectors;
    desc->available = isa_available((int) op_id) ? 1u : 0u;
    return copy_name(op->name, name, nameLen);
#else
    (void) op_id;
    (void) name;
    (void) nameLen;
    return AEE_EUNSUPPORTED;
#endif
}

AEEResult isaprobe_stream_name(remote_handle64 handle, uint32 stream_id, char * name, int nameLen) {
    if (handle == 0) {
        return AEE_EBADPARM;
    }
#ifdef ISAPROBE_HAVE_ISA_KERNELS_H
    if (stream_id >= ISA_S_COUNT) {
        return AEE_EBADPARM;
    }
    return copy_name(isa_stream_names[stream_id], name, nameLen);
#else
    (void) stream_id;
    (void) name;
    (void) nameLen;
    return AEE_EUNSUPPORTED;
#endif
}

// Call isa_run with the arguments of one census op.
static int isa_call_fn(void * arg) {
    const struct isa_call * c = (const struct isa_call *) arg;
    return isa_run(c->op_id, c->in[0], c->in[1], c->in[2], c->out, c->n_vectors);
}

// Read the processor cycle counter UPCYCLE, as hex_get_cycles of the backend does.
static inline uint64_t upcycles(void) {
    uint64_t cycles;
    __asm__ volatile("%0 = c15:14" : "=r"(cycles));
    return cycles;
}

// Prepare the kernel, run it iters times with a measurement of each run, and release it. Returns
// the first nonzero value of a kernel function, else 0.
static int kernel_call_fn(void * arg) {
    struct kernel_call * c = (struct kernel_call *) arg;
    void *               state = NULL;
    int                  err   = c->kernel->prepare(&c->problem, &state);
    if (err != CAND_OK) {
        return err;
    }
    c->pcycles = UINT64_MAX;
    c->usecs   = UINT64_MAX;
    for (uint32_t i = 0; i < c->iters && err == CAND_OK; i++) {
        const uint64_t t0 = HAP_perf_get_qtimer_count();
        const uint64_t c0 = upcycles();
        err               = c->kernel->run(&c->problem, state);
        const uint64_t c1 = upcycles();
        const uint64_t t1 = HAP_perf_get_qtimer_count();
        if (c1 - c0 < c->pcycles) {
            c->pcycles = c1 - c0;
        }
        const uint64_t us = HAP_perf_qtimer_count_to_us(t1 - t0);
        if (us < c->usecs) {
            c->usecs = us;
        }
    }
    c->kernel->release(state);
    return err;
}

// Run fn(arg) on the probe thread with b_out as its output buffer, then check the guard after
// out_len bytes and copy the output into dst. Returns 0 or an AEE error, and writes the status of
// the work into *status.
static AEEResult run_guarded(struct probe_ctx * ctx, int (*fn)(void *), void * arg, uint8_t * b_out, size_t out_len,
                             uint8 * dst, int32 * status) {
    // The lab allocator gives zeroed DDR, thus the bytes that the op does not write stay zero here too.
    memset(b_out, 0, out_len);
    memset(b_out + out_len, ISAPROBE_GUARD_BYTE, ISAPROBE_OUTPUT_GUARD);
    g_vtcm_used     = 0;
    g_vtcm_overflow = 0;

    struct probe_job job = { .ctx = ctx, .fn = fn, .arg = arg, .status = 0, .detail = 0 };
    const AEEResult  result = run_on_thread(&job);
    if (result != AEE_SUCCESS) {
        return result;
    }

    if (job.status == ISAPROBE_STATUS_HVX_LOCK || job.status == ISAPROBE_STATUS_HMX_LOCK) {
        FARF(ERROR, "isaprobe: the work did not run: lock status 0x%08x error %d", (unsigned) job.status, job.detail);
    } else if (!guard_intact(b_out, out_len)) {
        FARF(ERROR, "isaprobe: the work wrote after the end of its %u output bytes", (unsigned) out_len);
        job.status = ISAPROBE_STATUS_OVERRUN;
    } else if (g_vtcm_overflow) {
        job.status = ISAPROBE_STATUS_VTCM_FULL;
    }
    *status = job.status;

    if (out_len > 0) {
        memcpy(dst, b_out, out_len);
        const int err =
            qurt_mem_cache_clean((qurt_addr_t) dst, (qurt_size_t) out_len, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        if (err != QURT_EOK) {
            FARF(ERROR, "isaprobe: the cache flush of the output failed with %d", err);
        }
    }
    return AEE_SUCCESS;
}

// Allocate the output buffer with its guard. Returns NULL when there is no memory.
static uint8_t * alloc_output(size_t out_len) {
    return (uint8_t *) memalign(ISAPROBE_ALIGN, round_up(out_len, ISAPROBE_ALIGN) + ISAPROBE_OUTPUT_GUARD);
}

AEEResult isaprobe_run(remote_handle64 handle, int32 op_id, const uint8 * in0, int in0Len, const uint8 * in1,
                       int in1Len, const uint8 * in2, int in2Len, uint8 * dst, int dstLen, int32 n_vectors,
                       int32 * status) {
    struct probe_ctx * ctx = (struct probe_ctx *) handle;
    if (ctx == NULL || status == NULL || n_vectors < 0 || in0Len < 0 || in1Len < 0 || in2Len < 0 || dstLen < 0 ||
        (dstLen > 0 && dst == NULL)) {
        return AEE_EBADPARM;
    }

    const int setup_err = setup(ctx);
    if (setup_err != 0) {
        return setup_err;
    }

    int          alloc_fail = 0;
    void * const in[3]      = {
        copy_input(in0, in0Len, &alloc_fail),
        copy_input(in1, in1Len, &alloc_fail),
        copy_input(in2, in2Len, &alloc_fail),
    };
    const size_t out_len = (size_t) dstLen;
    uint8_t *    b_out   = alloc_output(out_len);

    AEEResult result;
    if (alloc_fail || b_out == NULL) {
        FARF(ERROR, "isaprobe: no memory for the buffers of op %d (inputs %d %d %d, output %d bytes)", (int) op_id,
             in0Len, in1Len, in2Len, dstLen);
        result = AEE_ENOMEMORY;
    } else {
        struct isa_call call = {
            .op_id = (int) op_id, .in = { in[0], in[1], in[2] }, .out = b_out, .n_vectors = (int) n_vectors
        };
        result = run_guarded(ctx, isa_call_fn, &call, b_out, out_len, dst, status);
    }

    free(in[0]);
    free(in[1]);
    free(in[2]);
    free(b_out);
    return result;
}

AEEResult isaprobe_kernel_info(remote_handle64 handle, uint32 kernel_id, uint32 * n_kernels, uint32 * op, char * name,
                               int nameLen) {
    if (handle == 0 || n_kernels == NULL || op == NULL) {
        return AEE_EBADPARM;
    }
    *n_kernels = cand_n_kernels;
    if (kernel_id >= cand_n_kernels) {
        return AEE_EBADPARM;
    }
    *op = cand_kernels[kernel_id].op;
    return copy_name(cand_kernels[kernel_id].name, name, nameLen);
}

AEEResult isaprobe_kernel_run(remote_handle64 handle, uint32 kernel_id, const isaprobe_kparams * params,
                              const uint8 * in0, int in0Len, const uint8 * in1, int in1Len, uint8 * dst, int dstLen,
                              int32 * status, uint64 * pcycles, uint64 * usecs) {
    struct probe_ctx * ctx = (struct probe_ctx *) handle;
    if (ctx == NULL || params == NULL || status == NULL || pcycles == NULL || usecs == NULL || in0Len < 0 ||
        in1Len < 0 || dstLen < 0 || (dstLen > 0 && dst == NULL) || kernel_id >= cand_n_kernels || params->iters == 0) {
        return AEE_EBADPARM;
    }
    const cand_kernel * k = &cand_kernels[kernel_id];

    // The sizes of the op. A mismatch is a bad parameter, not a kernel failure.
    if (params->op != k->op || params->op != CAND_OP_MATVEC_Q8_0 || params->cols % 32 != 0 ||
        (uint64_t) in0Len != (uint64_t) params->rows * (params->cols / 32) * 34u ||
        (uint64_t) in1Len != (uint64_t) params->cols * sizeof(float) ||
        (uint64_t) dstLen != (uint64_t) params->rows * sizeof(float)) {
        FARF(ERROR, "isaprobe: kernel %u: the op or the sizes do not agree (op %u rows %u cols %u, %d %d %d bytes)",
             (unsigned) kernel_id, (unsigned) params->op, (unsigned) params->rows, (unsigned) params->cols, in0Len,
             in1Len, dstLen);
        return AEE_EBADPARM;
    }

    const int setup_err = setup(ctx);
    if (setup_err != 0) {
        return setup_err;
    }

    int          alloc_fail = 0;
    void * const w          = copy_input(in0, in0Len, &alloc_fail);
    void * const x          = copy_input(in1, in1Len, &alloc_fail);
    const size_t out_len    = (size_t) dstLen;
    uint8_t *    b_out      = alloc_output(out_len);

    AEEResult result;
    if (alloc_fail || b_out == NULL) {
        FARF(ERROR, "isaprobe: no memory for the buffers of kernel %u", (unsigned) kernel_id);
        result = AEE_ENOMEMORY;
    } else {
        struct kernel_call call = {
            .kernel  = k,
            .problem = {
                .op = params->op, .rows = params->rows, .cols = params->cols, .w = w, .x = (const float *) x,
                .y = (float *) b_out, .vtcm = ctx->vtcm_base, .vtcm_bytes = ctx->vtcm_bytes,
            },
            .iters   = params->iters,
            .pcycles = 0,
            .usecs   = 0,
        };
        result   = run_guarded(ctx, kernel_call_fn, &call, b_out, out_len, dst, status);
        *pcycles = call.pcycles;
        *usecs   = call.usecs;
    }

    free(w);
    free(x);
    free(b_out);
    return result;
}

// The lab functions for a census kernel that uses the VTCM. Refer to tools/htp-lab/lab/lab.h.

__attribute__((weak)) uint8_t * lab_vtcm_base(void) {
    return g_vtcm_base;
}

__attribute__((weak)) size_t lab_vtcm_size(void) {
    return g_vtcm_size;
}

// A bump allocator in the VTCM. run() sets it to empty before each op. Returns NULL and sets the
// op status to ISAPROBE_STATUS_VTCM_FULL when the VTCM has no space. The lab version stops the
// program instead.
__attribute__((weak)) void * lab_vtcm_alloc(size_t bytes, size_t align) {
    if (g_vtcm_base == NULL || align == 0 || (align & (align - 1)) != 0) {
        g_vtcm_overflow = 1;
        return NULL;
    }
    const size_t off = round_up(g_vtcm_used, align);
    if (off > g_vtcm_size || bytes > g_vtcm_size - off) {
        g_vtcm_overflow = 1;
        return NULL;
    }
    g_vtcm_used = off + bytes;
    return g_vtcm_base + off;
}

__attribute__((weak)) void lab_hmx_enable(void) {
}
