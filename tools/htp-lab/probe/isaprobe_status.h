// The codes that the DSP library of the ISA probe gives to the host. The host program and the DSP
// library include this file. It has no dependency, thus the two compilers can read it.
#ifndef ISAPROBE_STATUS_H
#define ISAPROBE_STATUS_H

// The "status" of run() is the return value of isa_run, or one of these codes when the probe
// finds a problem with the op. The codes are far from the small values that a kernel returns.
#define ISAPROBE_STATUS_HVX_LOCK  0x15A00001  // qurt_hvx_lock failed. The op did not run.
#define ISAPROBE_STATUS_HMX_LOCK  0x15A00002  // HAP_compute_res_hmx_lock failed. The op did not run.
#define ISAPROBE_STATUS_OVERRUN   0x15A00003  // The op wrote after the end of the output buffer.
#define ISAPROBE_STATUS_VTCM_FULL 0x15A00004  // lab_vtcm_alloc had no VTCM for the op.

// The "setup_status" of info() is the first setup step that failed, 0 if all steps passed.
// "setup_error" is the error code of that step.
#define ISAPROBE_SETUP_OK           0
#define ISAPROBE_SETUP_APPTYPE      1  // HAP_power_set(apptype). Fatal: run() stops.
#define ISAPROBE_SETUP_HVX_POWER    2  // HAP_power_set(HVX). Fatal: run() stops.
#define ISAPROBE_SETUP_HMX_POWER    3  // HAP_power_set(HMX or HMX_v2). The probe runs without HMX.
#define ISAPROBE_SETUP_VTCM_QUERY   4  // HAP_compute_res_query_VTCM. The probe runs without VTCM.
#define ISAPROBE_SETUP_ACQUIRE_HMX  5  // HAP_compute_res_acquire of VTCM and HMX. The probe tries VTCM only.
#define ISAPROBE_SETUP_ACQUIRE_VTCM 6  // HAP_compute_res_acquire of VTCM only. The probe runs without VTCM.
#define ISAPROBE_SETUP_VTCM_PTR     7  // HAP_compute_res_attr_get_vtcm_ptr_v2. The probe runs without VTCM.

// The first two setup steps stop the probe. The other steps only remove the HMX or the VTCM.
#define ISAPROBE_SETUP_IS_FATAL(step) ((step) == ISAPROBE_SETUP_APPTYPE || (step) == ISAPROBE_SETUP_HVX_POWER)

#endif
