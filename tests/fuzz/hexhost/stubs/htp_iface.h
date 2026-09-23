// Stub of the header that the qaic tool makes from htp/htp_iface.idl, for the
// x86 fuzz harness. The fake DSP of the harness (common/fake_dsp.cpp) gives the
// implementation.
#pragma once

#include <stdint.h>

#include "AEEStdDef.h"
#include "remote.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct htp_iface_pmu_conf {
    uint32_t events[8];
} htp_iface_pmu_conf;

AEEResult htp_iface_open(const char * uri, remote_handle64 * handle);
AEEResult htp_iface_close(remote_handle64 handle);
AEEResult htp_iface_start(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_queue_id, uint32_t n_hvx,
                          uint32_t n_hmx, uint64_t max_vmem);
AEEResult htp_iface_stop(remote_handle64 handle);
AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint32_t size);
AEEResult htp_iface_munmap(remote_handle64 handle, uint32_t fd);
AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, const htp_iface_pmu_conf * pmu);
AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable);
AEEResult htp_iface_hwinfo(remote_handle64 handle, unsigned int * n_threads, unsigned int * n_hvx,
                           unsigned int * n_hmx, unsigned long long * vtcm_size);

#ifdef __cplusplus
}
#endif
