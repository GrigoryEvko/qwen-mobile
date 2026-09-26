// The dirty model of the hexhost harness: the true set of the dirty bytes beside the dirty range
// tracker of the DSP (htp/htp-tensor.c), for the x86 build of htp-tensor.c with the stubs of
// stubs/dsp. fuzz_dirty and the tracker replay of hexhost_graphs (graphs/tracker_replay.cpp) use it.
//
// A dirty byte is a byte that an op wrote and that no flush covered. The model gives the two cache
// functions that htp-tensor.c calls: hex_l2flush, with the line rounding of the real function, and
// qurt_mem_cache_clean, the flush of the whole cache. Each one removes the bytes that it flushes
// from the dirty set.
//
// run_op does the tracker calls of proc_op_req (htp/main.c) before an op: htp_tensor_flush_all on
// the inputs and htp_tensor_dirty_all on the outputs. It gives the path of htp_tensor_dirty_all
// from the flushes of the call. The call flushes only when it evicts ranges (each range is in use
// and an output joins no range), and it flushes the whole cache only when the evicted bytes are
// more than HEX_L2_FLUSH_ALL_THRESHOLD. Thus the model reads the path from the stubs, and it has
// no copy of the logic of the tracker.
//
// The model has one dirty set for the process. Use it from one thread.
#pragma once

#include <cstdint>

struct htp_context;
struct htp_tensor;

namespace dirty_model {

// The path of one call of htp_tensor_dirty_all
enum class dirty_path {
    keep,       // each output joins a range or takes a free range, and the call flushes nothing
    evict,      // each range is in use: the call flushes ranges and gives them to the outputs
    flush_all,  // the evicted bytes are more than HEX_L2_FLUSH_ALL_THRESHOLD: the call flushes the whole cache
};

// Gives the name of a path: "keep", "evict" or "flush-all".
const char * path_name(dirty_path p);

// The result of the tracker calls before one op
struct op_result {
    dirty_path path        = dirty_path::keep;
    uint64_t   evict_bytes = 0;   // the bytes of the lines that the eviction flushed with hex_l2flush
    int        stale_input = -1;  // an input that holds a dirty byte after htp_tensor_flush_all, or -1
    uint64_t   stale_byte  = 0;   // that dirty byte
    uint64_t   lost_byte   = 0;   // a dirty byte that no range of the tracker holds after the op, or 0
    uint32_t   n_ranges    = 0;   // the ranges in use after the op
};

// Flushes the whole cache and clears the ranges of the tracker, as process_opbatch of htp/main.c
// does at the start and at the end of a batch.
void batch_edge(htp_context & ctx);

// Does the tracker calls of proc_op_req for one op, then adds the outputs to the dirty set. srcs has
// HTP_OP_MAX_INPUTS entries and dsts has HTP_OP_MAX_OUTPUTS entries, nullptr for no tensor. The
// addresses are DSP addresses above zero. O(k * HTP_MAX_DIRTY_RANGES) for k intervals in the dirty set.
op_result run_op(htp_context & ctx, const htp_tensor * const * srcs, const htp_tensor * const * dsts);

// Does htp_flush_dirty_ranges, as a fence does. Gives a byte that stays dirty after it, or 0.
uint64_t fence(htp_context & ctx);

// Puts [start, end) into the range `index` (below HTP_MAX_DIRTY_RANGES) of the tracker and into the
// dirty set, as when an earlier op of the batch wrote these bytes. The caller keeps the range apart
// from the other ranges.
void occupy(htp_context & ctx, uint32_t index, uint32_t start, uint32_t end);

} // namespace dirty_model
