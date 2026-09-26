// The replay of the dirty range tracker of the DSP (htp/htp-tensor.c) on the batches that
// hexhost_graphs sends to the fake DSP.
//
// The fake DSP computes no values, thus it does not run the tracker. The replay runs htp-tensor.c of
// the tree under test (with the stubs of stubs/dsp) on the ops of each batch, as proc_op_req of
// htp/main.c calls it: htp_tensor_flush_all on the inputs and htp_tensor_dirty_all on the outputs of
// each op, and htp_flush_dirty_ranges after a FENCE of mode 1 or a CPY_FENCE (the replay flushes as the
// copy without DMA does). Each batch starts with no range, as process_opbatch does after its flush of
// the whole cache. The dirty model (common/dirty_model.h) gives the path of each htp_tensor_dirty_all
// call and the bytes that the tracker loses.
//
// The DSP maps the buffers of a batch into its 32-bit address space, and the replay needs such
// addresses. It puts the buffers in the order of the buffer table of the batch, from 16 MiB, with a
// gap between two buffers: HEXHOST_TRACKER_GAP bytes (1 MiB when the variable is not set), rounded up
// to 4 KiB. With a gap the tracker cannot join the ranges of two buffers. With a gap of 0 two buffers
// can touch, as when the DSP maps them next to each other. The real layout of the DSP is not known.
//
// HEXHOST_TRACKER_RESERVE=N or N:BYTES (the positive control of the counts): each batch starts with N
// ranges in use (N is 1 to HTP_MAX_DIRTY_RANGES) of BYTES dirty bytes each (256 by default), after the
// buffers of the batch in the DSP addresses. Thus a graph that needs more than HTP_MAX_DIRTY_RANGES - N
// ranges makes the tracker evict, and with BYTES above HEX_L2_FLUSH_ALL_THRESHOLD the first eviction of
// a batch flushes the whole cache.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tracker_replay {

// The counts of the replay
struct counts {
    uint64_t batches     = 0;  // the batches that the replay ran
    uint64_t skipped     = 0;  // the batches whose buffers do not fit in the 32-bit address space
    uint64_t unmapped    = 0;  // the tensors outside each buffer of their batch (the replay ignores them)
    uint64_t ops         = 0;  // the ops, thus the calls of htp_tensor_dirty_all
    uint64_t fences      = 0;  // the calls of htp_flush_dirty_ranges
    uint64_t evict       = 0;  // the calls of htp_tensor_dirty_all that evict ranges (each range in use)
    uint64_t flush_all   = 0;  // of these, the calls that flush the whole cache
    uint64_t evict_bytes = 0;  // the bytes of the lines that the other evictions flushed
    uint64_t lost        = 0;  // the ops after which a byte that an op wrote is in no range
    uint64_t stale       = 0;  // the ops with an input that holds such a byte after htp_tensor_flush_all
    uint64_t fence_left  = 0;  // the fences after which such a byte stays
    uint32_t max_ranges  = 0;  // the most ranges in use after one op
};

// Makes the fake DSP call the replay after each batch. Call it before the host sends a batch. Stops the
// process with a message when HEXHOST_TRACKER_GAP or HEXHOST_TRACKER_RESERVE is not valid.
void install();

// Clears the counts and the lines.
void reset();

// Gives the sum of the counts of the batches since the last reset.
counts total();

// Gives one line for each batch with an eviction, a lost byte or a stale input since the last reset.
std::vector<std::string> lines();

// Gives the counts as one line of text.
std::string summary(const counts & c);

} // namespace tracker_replay
