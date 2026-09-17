// Stub of the compute resource API (VTCM and HMX locks). The lab maps VTCM and enables HMX itself.
#ifndef LAB_STUB_HAP_COMPUTE_RES_H
#define LAB_STUB_HAP_COMPUTE_RES_H

static inline int HAP_compute_res_hmx_lock(unsigned int context_id) { (void) context_id; return 0; }
static inline int HAP_compute_res_hmx_unlock(unsigned int context_id) { (void) context_id; return 0; }

#endif
