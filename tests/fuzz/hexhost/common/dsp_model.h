// The model of the DSP side of the Hexagon backend (htp/main.c and htp/*-ops.c)
// for the hexhost fuzz harness.
//
// process_batch() does what process_opbatch() of htp/main.c does before and
// around each op: the block size check, the buffer and tensor preparation, and
// the op loop. model_op() gives the status that the op function of the DSP
// returns for the descriptors and the kernel params that the host sent, without
// the arithmetic. A comment names the source lines of each check in the DSP
// sources that the model follows.
#pragma once

#include "fake_dsp.h"

#include <dspqueue.h>

#include <cstdint>
#include <string>

#include "htp-ops.h"

namespace fakedsp {

// The state of the DSP context that the checks read
struct dsp_ctx {
    uint32_t n_threads = 6;          // ctx->n_threads
    uint32_t n_hmx     = 1;          // ctx->hmx_enabled
    uint64_t vtcm_size = 8u << 20;   // ctx->vtcm_size
    uint64_t max_vmem  = 0;          // ctx->max_vmem, 0 = no limit
    uint32_t profiler  = 0;          // ctx->profiler
    bool     touch     = false;      // read the inputs and write the outputs of each op
    uint8_t  fill      = 0xA5;       // the byte that the touch mode writes into each output
};

// The result of the model of one op
struct op_verdict {
    uint32_t    status = HTP_STATUS_OK;  // the status that the DSP returns
    bool        silent = false;          // the DSP returns OK and computes nothing
    bool        vtcm_overflow = false;   // the DSP does not check VTCM and the layout is larger
    std::string why;                     // the check that failed
    // The class of the defect when the harness knows it (for example "vtcm-binary"). The id of the
    // violation is then "dsp-<tag>", else "dsp-refuse-<op>", "dsp-silent-<op>" or "dsp-vtcm-overflow-<op>".
    std::string tag;
};

// Gives the violation id of a verdict that is not OK: "dsp-<tag>", or the id from the kind and the op name.
std::string verdict_id(const op_verdict & v, const char * op_name);

// Processes one batch as htp/main.c process_opbatch does. It fills the record
// and the response. A problem that the real DSP does not report, for example a
// tensor outside its buffer, is a violation.
void process_batch(const dsp_ctx & ctx, const htp_opbatch_req & req, const struct dspqueue_buffer & dbuf,
                   batch_record & rec, htp_opbatch_rsp & rsp);

// Gives the verdict of the DSP op function for one op.
op_verdict model_op(const dsp_ctx & ctx, const op_record & op);

// Gives the byte extent of a tensor descriptor from its shape and strides (the
// size that the DSP can touch). A repacked tensor gives its size field.
uint64_t tensor_extent(const tensor_ref & t);

// Gives the name of an opcode.
const char * opcode_name(uint32_t opcode);

} // namespace fakedsp
