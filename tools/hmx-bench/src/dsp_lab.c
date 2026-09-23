// The lab runtime of tools/htp-lab/lab/lab.h for the device.
//
// A lab target of tools/htp-lab (for example target_i8probe.c) runs in the simulator with the
// standalone runtime. This file gives the same functions in an unsigned protection domain on the
// phone, where run_main_on_hexagon calls main(). Thus one source runs in the simulator and on the
// silicon, and the host script can compare the two outputs byte for byte.
//
// lab_init() acquires the whole VTCM and the HMX with HAP_compute_res, sets the power corners to
// the maximum (the same requests as htp/main.c of the Hexagon backend), and locks the HMX for the
// calling thread. lab_fini() releases them. Only the functions that the device targets use are
// here: lab_run_threads and lab_compare_f32 are not.

#include "lab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_power.h"

// A message to the log of the phone, through printf and through FARF. Each of the two reaches
// logcat only with a .farf mask file next to the program (refer to tools/README.md).
#define LAB_LOG(...)                    \
    do {                                \
        printf(__VA_ARGS__);            \
        printf("\n");                   \
        FARF(ALWAYS, __VA_ARGS__);      \
    } while (0)

static uint8_t *     g_vtcm_base;
static size_t        g_vtcm_size;
static size_t        g_vtcm_used;
static unsigned int  g_rctx;
static uint32_t      g_rand_state = 0x9E3779B9u;
static int           g_power_ctx;  // the address is the power client identity

// Sets the power corners of the core, the bus, the HVX and the HMX to the maximum.
// Returns 0 or the error of the first failed request.
static int lab_power_up(void) {
    HAP_power_request_t req;
    int                 err;

    memset(&req, 0, sizeof(req));
    req.type    = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if ((err = HAP_power_set(&g_power_ctx, &req)) != 0) {
        return err;
    }

    memset(&req, 0, sizeof(req));
    req.type                              = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable           = TRUE;
    req.dcvs_v3.dcvs_enable               = FALSE;
    req.dcvs_v3.set_bus_params            = TRUE;
    req.dcvs_v3.bus_params.min_corner     = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.max_corner     = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.target_corner  = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_core_params           = TRUE;
    req.dcvs_v3.core_params.min_corner    = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.max_corner    = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_sleep_disable         = TRUE;
    req.dcvs_v3.sleep_disable             = TRUE;
    if ((err = HAP_power_set(&g_power_ctx, &req)) != 0) {
        return err;
    }

    memset(&req, 0, sizeof(req));
    req.type         = HAP_power_set_HVX;
    req.hvx.power_up = TRUE;
    if ((err = HAP_power_set(&g_power_ctx, &req)) != 0) {
        return err;
    }

    memset(&req, 0, sizeof(req));
    req.type                 = HAP_power_set_HMX_v2;
    req.hmx_v2.set_power     = TRUE;
    req.hmx_v2.power_up      = TRUE;
    req.hmx_v2.set_clock     = TRUE;
    req.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.min_corner    = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.max_corner    = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.perf_mode     = HAP_CLK_PERF_HIGH;
    return HAP_power_set(&g_power_ctx, &req);
}

// The HVX needs no lock: QuRT gives the thread an HVX context at its first vector instruction.
// The resource is acquired in the cached mode, as htp/main.c does, thus it needs
// HAP_compute_res_acquire_cached before the VTCM and the HMX are usable. Each step logs its
// return code before any exit, thus a failure on the phone names its call.
void lab_init(void) {
    HAP_setFARFRuntimeLoggingParams(0xffff, NULL, 0);
    unsigned int vtcm_size = 8u * 1024u * 1024u;
    const int qerr = HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);
    LAB_LOG("lab: query_VTCM ret %d size %u", qerr, vtcm_size);

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);
    g_rctx = HAP_compute_res_acquire(&attr, 1000000);
    LAB_LOG("lab: acquire rctx 0x%x", g_rctx);
    if (!g_rctx) {
        LAB_LOG("lab: error: the acquire of %u bytes of VTCM and the HMX failed", vtcm_size);
        exit(2);
    }
    void *    ptr  = NULL;
    const int verr = HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &ptr, &vtcm_size);
    LAB_LOG("lab: get_vtcm_ptr ret %d ptr %p size %u", verr, ptr, vtcm_size);
    if (verr != 0 || ptr == NULL) {
        exit(2);
    }
    const int cerr = HAP_compute_res_acquire_cached(g_rctx, 10000000u);
    LAB_LOG("lab: acquire_cached ret %d", cerr);
    if (cerr != 0) {
        exit(2);
    }
    g_vtcm_base = (uint8_t *) ptr;
    g_vtcm_size = vtcm_size;
    g_vtcm_used = 0;

    const int perr = lab_power_up();
    LAB_LOG("lab: power ret %d", perr);
    if (perr != 0) {
        exit(2);
    }
    const int herr = HAP_compute_res_hmx_lock(g_rctx);
    LAB_LOG("lab: hmx_lock ret %d", herr);
    if (herr != 0) {
        exit(2);
    }
    g_rand_state = 0x9E3779B9u;
    LAB_LOG("lab: core vtcm_base = %p vtcm_size = %u KB", (void *) g_vtcm_base, (unsigned) (g_vtcm_size / 1024));
}

void lab_fini(void) {
    if (g_rctx) {
        HAP_compute_res_hmx_unlock(g_rctx);
        HAP_compute_res_release_cached(g_rctx);
        HAP_compute_res_release(g_rctx);
        g_rctx = 0;
    }
}

uint8_t * lab_vtcm_base(void) {
    return g_vtcm_base;
}

size_t lab_vtcm_size(void) {
    return g_vtcm_size;
}

void * lab_vtcm_alloc(size_t bytes, size_t align) {
    const size_t off = (g_vtcm_used + align - 1) & ~(align - 1);
    if (off + bytes > g_vtcm_size) {
        LAB_LOG("lab: error: the VTCM is full (%u of %u bytes used, %u requested)", (unsigned) g_vtcm_used,
             (unsigned) g_vtcm_size, (unsigned) bytes);
        exit(2);
    }
    g_vtcm_used = off + bytes;
    return g_vtcm_base + off;
}

void * lab_ddr_alloc(size_t bytes, size_t align) {
    void * p = memalign(align < sizeof(void *) ? sizeof(void *) : align, bytes);
    if (p == NULL) {
        LAB_LOG("lab: error: the DDR allocation of %u bytes failed", (unsigned) bytes);
        exit(2);
    }
    memset(p, 0, bytes);
    return p;
}

uint32_t lab_rand_u32(void) {
    uint32_t x = g_rand_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rand_state = x;
    return x;
}

void lab_hmx_enable(void) {}
