// The fake DSP of the hexhost fuzz harness.
//
// The fake DSP gives the FastRPC, rpcmem, dspqueue and htp_iface functions
// that the host part of the Hexagon backend calls. It reads each op batch that
// the host sends, does a check of the descriptors, applies a model of the DSP
// side checks of each op (htp/*.c), records the ops, and sends a response.
//
// Two modes:
//   - sync:  dspqueue_write processes the batch immediately on the host thread.
//   - async: each queue has its own DSP thread, as the real DSP does. The DSP
//            thread reads the input bytes and writes the output bytes of each
//            op, thus ThreadSanitizer sees the same memory traffic as on the
//            phone.
//
// A violation is a condition that the real DSP refuses or that gives an
// incorrect result on the phone. The fake DSP writes a report and aborts,
// unless the id of the violation is in the environment variable
// HEXHOST_IGNORE (a comma list). An ignored refusal gives HTP_STATUS_OK, thus
// the fuzzer continues past a known finding.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fakedsp {

// The hardware and the behavior of the fake DSP
struct config {
    uint32_t n_threads = 6;           // hwinfo: the HVX thread count
    uint32_t n_hmx     = 1;           // hwinfo: the HMX unit count
    uint64_t vtcm_size = 8u << 20;    // hwinfo: the VTCM size in bytes
    int      arch      = 79;          // htpdrv_get_arch
    bool     async     = false;       // a DSP thread for each queue
    bool     touch     = false;       // read the inputs and write the outputs of each op
    // The byte that the touch mode writes into each output. 0xA5 gives a negative int32, thus a
    // slot index that the host reads from such bytes is not valid and the unfused ops run. The
    // environment variable HEXHOST_TOUCH_FILL (0 to 255) overrides it.
    uint8_t  fill      = 0xA5;
    int      discovery = 0;          // 0 = FASTRPC_GET_DOMAINS fails, 1 = it gives one NSP domain
};

// Sets the configuration. Call it before the host opens a session.
void configure(const config & cfg);

// Gives the configuration that the fake DSP uses at this time.
const config & get_config();

// A tensor descriptor as the DSP sees it, with the absolute host address
struct tensor_ref {
    bool     present = false;
    uint64_t addr    = 0;   // host address of the first byte
    uint32_t size    = 0;   // htp_tensor.size
    uint32_t type    = 0;
    uint32_t flags   = 0;
    uint32_t ne[4]   = {0, 0, 0, 0};
    uint32_t nb[4]   = {0, 0, 0, 0};
    uint16_t index   = 0xffff;  // index of the descriptor in the batch
};

// One op as the DSP executed it
struct op_record {
    uint32_t   opcode = 0;
    uint32_t   flags  = 0;
    int32_t    params[16]  = {0};
    int32_t    kparams[32] = {0};
    tensor_ref src[10];
    tensor_ref dst[4];
    uint32_t   status = 1;      // HTP_STATUS_*
    std::string note;           // the reason of a refusal
};

// One batch as the DSP executed it
struct batch_record {
    uint64_t               seq     = 0;
    uint64_t               queue   = 0;
    uint32_t               status  = 1;
    std::vector<op_record> ops;
};

// Gives the recorded batches in execution order and clears the record.
std::vector<batch_record> take_batches();

// Clears the record and the counters. Call it at the start of a fuzz input.
void reset_record();

// Reports a violation. It writes the message to stderr and aborts, unless the
// id is in HEXHOST_IGNORE. Gives true when the violation is ignored.
bool violation(const char * id, const char * fmt, ...) __attribute__((format(printf, 2, 3)));

// Gives true when the id is in HEXHOST_IGNORE.
bool is_ignored(const char * id);

// Looks up the rpcmem allocation that holds addr. Gives false when no allocation holds it.
bool lookup_alloc(uint64_t addr, uint64_t * base, uint64_t * size, int * fd);

// Gives the number of live rpcmem allocations (a leak check of the session release).
size_t live_allocs();

// Gives the number of live queues and live htp_iface handles.
size_t live_queues();
size_t live_handles();

// Adds one to a named counter. With HEXHOST_STATS=1 the counters print at exit.
void count(const std::string & name, uint64_t n = 1);

} // namespace fakedsp
