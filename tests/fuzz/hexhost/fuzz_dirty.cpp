// fuzz_dirty: the dirty range tracker of the DSP (htp/htp-tensor.c), which
// decides which cache lines the DSP flushes to DDR before a DMA reads them.
//
// The harness compiles htp-tensor.c as it is, with the stubs of stubs/dsp. A
// fuzz input is a sequence of ops as proc_op_req of htp/main.c runs them:
// htp_tensor_flush_all on the inputs, htp_tensor_dirty_all on the outputs, the
// writes of the op, and at times htp_flush_dirty_ranges (a fence) or the end of
// the batch. The dirty model (common/dirty_model.h) keeps the true set of the
// bytes that the ops wrote and that no flush covered. The invariants:
//   - no input of an op holds a byte that a previous op wrote and no flush covered
//     (else the DMA of the op reads stale DDR),
//   - each such byte is inside a range of the tracker (else no later flush covers it),
//   - after a fence no such byte remains.
// With HEXHOST_STATS=1 the harness counts the path of each htp_tensor_dirty_all
// call ("tracker keep", "tracker evict", "tracker flush-all") and prints the
// counts at exit.

#include <fuzzer/FuzzedDataProvider.h>

#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "qurt.h"
#include "hex-utils.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
}

#include "dirty_model.h"
#include "fake_dsp.h"
#include "fuzz_death.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
    FuzzedDataProvider fdp(data, size);

    struct htp_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.n_threads     = fdp.ConsumeIntegralInRange<uint32_t>(1, HTP_MAX_NTHREADS);
    ctx.n_threads_div = init_fastdiv_values(ctx.n_threads);
    dirty_model::batch_edge(ctx);

    // The arena of the tensors: a DSP address range above zero
    const uint32_t ARENA = 0x40000000u;
    const uint32_t SPAN  = 24u << 20;

    // More tensors than the tracker has ranges: a new output with each range in use makes the tracker
    // evict ranges (htp_tensor_dirty_all). With as many tensors as ranges that path cannot occur.
    // tools/make_dirty_seeds.py writes the seeds of this input format.
    std::vector<htp_tensor> pool(HTP_MAX_DIRTY_RANGES + 8);
    for (auto & t : pool) {
        memset(&t, 0, sizeof(t));
        const bool big = fdp.ConsumeIntegralInRange<int>(0, 7) == 0;
        t.size         = big ? fdp.ConsumeIntegralInRange<uint32_t>(1, 9u << 20) : fdp.ConsumeIntegralInRange<uint32_t>(1, 64u << 10);
        t.data         = ARENA + (fdp.ConsumeIntegralInRange<uint32_t>(0, SPAN - t.size) & ~3u);
        t.flags        = fdp.ConsumeIntegralInRange<int>(0, 15) == 0 ? HTP_TENSOR_WEIGHT : 0;
    }

    const int n_ops = fdp.ConsumeIntegralInRange<int>(1, 64);
    for (int i = 0; i < n_ops && fdp.remaining_bytes() > 0; i++) {
        const struct htp_tensor * srcs[HTP_OP_MAX_INPUTS] = { nullptr };
        const struct htp_tensor * dsts[HTP_OP_MAX_OUTPUTS] = { nullptr };
        const int n_src = fdp.ConsumeIntegralInRange<int>(0, 4);
        const int n_dst = fdp.ConsumeIntegralInRange<int>(1, 2);
        for (int s = 0; s < n_src; s++) {
            srcs[s] = &pool[fdp.ConsumeIntegralInRange<size_t>(0, pool.size() - 1)];
        }
        for (int d = 0; d < n_dst; d++) {
            dsts[d] = &pool[fdp.ConsumeIntegralInRange<size_t>(0, pool.size() - 1)];
        }

        const dirty_model::op_result r = dirty_model::run_op(ctx, srcs, dsts);
        static const std::string counter[] = { std::string("tracker ") + dirty_model::path_name(dirty_model::dirty_path::keep),
                                               std::string("tracker ") + dirty_model::path_name(dirty_model::dirty_path::evict),
                                               std::string("tracker ") + dirty_model::path_name(dirty_model::dirty_path::flush_all) };
        fakedsp::count(counter[(int) r.path]);
        if (r.stale_input >= 0) {
            const uint64_t a = srcs[r.stale_input]->data, b = a + srcs[r.stale_input]->size;
            fakedsp::violation("dirty-stale-input", "op %d: input %d [0x%" PRIx64 ", 0x%" PRIx64 ") holds the dirty byte 0x%" PRIx64
                               " after htp_tensor_flush_all (the DMA reads stale DDR)", i, r.stale_input, a, b, r.stale_byte);
        }
        if (r.lost_byte) {
            fakedsp::violation("dirty-lost", "op %d: the dirty byte 0x%" PRIx64 " is in no range of the tracker", i, r.lost_byte);
        }

        const int ev = fdp.ConsumeIntegralInRange<int>(0, 9);
        if (ev == 0) {
            // op_fence of htp/main.c: each dirty byte goes to DDR
            if (const uint64_t x = dirty_model::fence(ctx)) {
                fakedsp::violation("dirty-fence-leftover", "op %d: after htp_flush_dirty_ranges the byte 0x%" PRIx64 " is dirty", i, x);
            }
        } else if (ev == 1) {
            // the end of a batch and the start of the next (process_opbatch of htp/main.c)
            dirty_model::batch_edge(ctx);
        }
    }
    return 0;
}
