// The constants that the host program and the DSP library of the DDR read probe share.
//
// The host program (host/ddrbw.c) and the DSP library (dsp/ddrbw_dsp.c) exchange these values
// through the FastRPC interface ddrbw.idl. Keep the two sides on one copy of this header.
#ifndef DDRBW_COMMON_H
#define DDRBW_COMMON_H

// The power votes of ddrbw_setup. The library sends each vote of the mask one time, from its own
// context, and keeps it until the process stops.
#define DDRBW_VOTE_BACKEND 0x01u  // the votes of htp/main.c of the llama.cpp backend
#define DDRBW_VOTE_DDRPERF 0x02u  // the DDR performance mode (HAP_set_ddr_perf_mode)
#define DDRBW_VOTE_BUSPERF 0x04u  // the highest core and bus clock in the voltage corner
#define DDRBW_VOTE_EXPV    0x08u  // the expanded voltage corners of core, bus and CENG at the maximum
#define DDRBW_VOTE_CENG    0x10u  // the CENG bus at the maximum voltage corner
#define DDRBW_VOTE_BW      0x20u  // a MIPS and bus bandwidth vote at the maximum values of the DSP
#define DDRBW_VOTE_ALL     0x3Fu

// The flags of one read run.
#define DDRBW_FLAG_SUM    0x01u  // a DMA thread adds all bytes of each chunk, not only the first 128
#define DDRBW_FLAG_CACHED 0x02u  // a DMA descriptor reads the DDR through the L2 cache (no bypass)

// The kind of one thread of a read run.
#define DDRBW_KIND_HVX 0u
#define DDRBW_KIND_DMA 1u

// The status of ddrbw_read. It is 0 when the run is valid.
#define DDRBW_OK          0
#define DDRBW_E_PARAM     1  // a parameter is not valid (the host program gives the text)
#define DDRBW_E_NOMAP     2  // the fd has no mapping on the DSP
#define DDRBW_E_VTCM      3  // the DMA slots of one thread do not fit in its part of the VTCM
#define DDRBW_E_THREAD    4  // a QuRT thread did not start
#define DDRBW_E_NOMEM     5  // an allocation on the DSP failed
#define DDRBW_E_TIMEOUT   6  // the DMA engine did not complete a descriptor

// The indices of the return codes of ddrbw_setup. A value of 0x7fffffff means "not sent".
enum ddrbw_rc {
    DDRBW_RC_APPTYPE,    // HAP_power_set_apptype
    DDRBW_RC_PROTECTED,  // HAP_set_dcvs_v3_protected_bus_corners
    DDRBW_RC_DDRPERF,    // HAP_set_ddr_perf_mode
    DDRBW_RC_COREPERF,   // HAP_set_dcvs_v3_core_perf_mode
    DDRBW_RC_BUSPERF,    // HAP_set_dcvs_v3_bus_perf_mode
    DDRBW_RC_EXPV,       // HAP_dcvs_config(HAP_DCVS_SET_EXP_VCORNERS)
    DDRBW_RC_DCVS,       // HAP_power_set_DCVS_v3
    DDRBW_RC_HVX,        // HAP_power_set_HVX
    DDRBW_RC_HMX,        // HAP_power_set_HMX_v2
    DDRBW_RC_CENG,       // HAP_power_set_CENG_bus
    DDRBW_RC_MIPSBW,     // HAP_power_set_mips_bw
    DDRBW_RC_CAPS,       // HAP_dcvs_config(HAP_DCVS_CAPABILITY_QUERY)
    DDRBW_RC_VTCM,       // HAP_compute_res_acquire of the full VTCM
    DDRBW_RC_COUNT
};
#define DDRBW_RC_NOT_SENT 0x7fffffff

// The indices of the facts of ddrbw_setup and ddrbw_query.
enum ddrbw_fact {
    DDRBW_FACT_ARCH,        // __HVX_ARCH__ of the library build
    DDRBW_FACT_HVX,         // the HVX contexts of the DSP
    DDRBW_FACT_VTCM,        // the bytes of VTCM that the library holds
    DDRBW_FACT_MAX_MIPS,    // HAP_power_get_max_mips
    DDRBW_FACT_MAX_BUS_BW,  // HAP_power_get_max_bus_bw, in bytes per second
    DDRBW_FACT_CORE_HZ,     // HAP_power_get_clk_Freq
    DDRBW_FACT_HMX_HZ,      // HAP_power_get_hmx_core_clk_Freq
    DDRBW_FACT_DCVS,        // HAP_power_get_dcvsEnabled
    DDRBW_FACT_CAPS,        // the capability bits of HAP_DCVS_CAPABILITY_QUERY
    DDRBW_FACT_PRIO,        // the QuRT priority of the FastRPC thread
    DDRBW_FACT_COUNT
};

// The modes of the DSP thread of the queue test (ddrbw_queue_start).
#define DDRBW_QMODE_PEEK    0u  // wait in dspqueue_peek for each packet
#define DDRBW_QMODE_BACKEND 1u  // the loop of htp/main.c: dspqueue_peek, then 100 polls with qurt_sleep(100)
#define DDRBW_QMODE_SPIN    2u  // dspqueue_read_noblock in a loop with no sleep
#define DDRBW_QMODE_MASK    0x0Fu

// The command of one queue packet.
#define DDRBW_QCMD_ECHO 1u
#define DDRBW_QCMD_STOP 2u

// One queue packet, in the two directions.
struct ddrbw_qmsg {
    unsigned int       seq;
    unsigned int       cmd;
    unsigned long long t_dsp;  // the qtimer count when the DSP read the request
};

// The fence test: the host writes the sequence number at offset 0 of the fence buffer, the DSP
// writes it at DDRBW_FENCE_DSP. The two slots are on different cache lines.
#define DDRBW_FENCE_DSP   128u
#define DDRBW_FENCE_BYTES 65536u

#endif
