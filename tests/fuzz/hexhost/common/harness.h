// Helpers that each hexhost fuzz target uses: the log filter, the guard of the
// ggml asserts during graph construction, and the conversion of a packed op of
// the host into a record of the DSP model.
#pragma once

#include "fake_dsp.h"
#include "hexhost.h"

#include "ggml.h"

#include <csetjmp>
#include <cstdint>

namespace harness {

// Gives true with the probability 1/n for random input bytes. An exhausted
// input gives false, thus a short input takes the common path.
template <typename P> inline bool rare(P & fdp, int n) {
    return fdp.template ConsumeIntegralInRange<int>(0, n - 1) == n - 1;
}

// Sets the log filter (HEXHOST_LOG=1 shows the logs of ggml) and the abort callback of ggml.
void init();

// Tells the log filter that each slot index of the graph that runs is valid. Then the log line
// of the host "the unfused ops run" is a violation of the check gdn-slot-stale: the host read an
// index that the harness did not write.
void expect_valid_slots(bool on);

// The jump buffer of the active guard, or nullptr. A ggml abort while a guard
// is active returns to the guard: the fuzz input is not valid ggml.
extern thread_local jmp_buf * g_guard;

// Runs one construction step of a graph. Gives false when ggml aborted in it.
// The step must only call ggml functions and write plain values: an abort
// skips the destructors of its frames.
template <typename F> bool guarded(F && step) {
    jmp_buf   env;
    jmp_buf * prev = g_guard;
    g_guard        = &env;
    if (setjmp(env) != 0) {
        g_guard = prev;
        return false;
    }
    step();
    g_guard = prev;
    return true;
}

// Converts a packed op of the host into a record of the DSP model.
fakedsp::op_record to_record(const hexhost::packed_op & p);

// Gives the DSP context of a device, as htp_iface_start makes it.
struct dsp_limits {
    uint32_t n_threads;
    uint32_t n_hmx;
    uint64_t vtcm;
};
dsp_limits limits_of(hexhost::device * d);

// Runs the DSP model on a packed op and reports a violation for a refusal, a
// silent wrong result, or a VTCM overflow.
void check_packed(hexhost::device * d, const hexhost::packed_op & p, const char * context);

} // namespace harness
