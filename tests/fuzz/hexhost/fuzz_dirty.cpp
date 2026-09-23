// fuzz_dirty: the dirty range tracker of the DSP (htp/htp-tensor.c), which
// decides which cache lines the DSP flushes to DDR before a DMA reads them.
//
// The harness compiles htp-tensor.c as it is, with the stubs of stubs/dsp. A
// fuzz input is a sequence of ops as proc_op_req of htp/main.c runs them:
// htp_tensor_flush_all on the inputs, htp_tensor_dirty_all on the outputs, the
// writes of the op, and at times htp_flush_dirty_ranges (a fence) or the end of
// the batch. A second model keeps the true set of the bytes that the ops wrote
// and that no flush covered. The invariants:
//   - no input of an op holds a byte that a previous op wrote and no flush covered
//     (else the DMA of the op reads stale DDR),
//   - each such byte is inside a range of the tracker (else no later flush covers it),
//   - after a fence no such byte remains.
// The harness also counts the bytes that the tracker flushes: the fused state op
// declares its whole output dirty, and each such byte costs a flush.

#include <fuzzer/FuzzedDataProvider.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <vector>

extern "C" {
#include "qurt.h"
#include "hex-utils.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
}

#include "fake_dsp.h"
#include "fuzz_death.h"

namespace {

// The true set of dirty bytes as disjoint intervals [start, end)
std::map<uint64_t, uint64_t> g_dirty;
uint64_t                     g_flushed_bytes = 0;
bool                         g_flushed_all   = false;

// Removes [s, e) from the true dirty set.
void clean(uint64_t s, uint64_t e) {
    auto it = g_dirty.lower_bound(s);
    if (it != g_dirty.begin()) {
        --it;
    }
    while (it != g_dirty.end() && it->first < e) {
        const uint64_t a = it->first, b = it->second;
        if (b <= s) {
            ++it;
            continue;
        }
        it = g_dirty.erase(it);
        if (a < s) {
            g_dirty[a] = s;
        }
        if (b > e) {
            g_dirty[e] = b;
        }
    }
}

// Adds [s, e) to the true dirty set.
void mark(uint64_t s, uint64_t e) {
    clean(s, e);
    g_dirty[s] = e;
}

// Gives the first dirty byte in [s, e), or 0.
uint64_t first_dirty(uint64_t s, uint64_t e) {
    auto it = g_dirty.lower_bound(s);
    if (it != g_dirty.begin()) {
        auto p = std::prev(it);
        if (p->second > s) {
            return s;
        }
    }
    if (it != g_dirty.end() && it->first < e) {
        return it->first;
    }
    return 0;
}

} // namespace

extern "C" {

// The flush of a range of lines, with the rounding of the real hex_l2flush
void hex_l2flush(void * addr, size_t size) {
    const uint64_t a = (uint64_t) (uintptr_t) addr;
    const uint64_t s = a & ~(uint64_t) (HEX_L2_LINE_SIZE - 1);
    const uint64_t e = (a + size + HEX_L2_LINE_SIZE - 1) & ~(uint64_t) (HEX_L2_LINE_SIZE - 1);
    clean(s, e);
    g_flushed_bytes += e - s;
}

// The flush of the whole data cache
int qurt_mem_cache_clean(qurt_addr_t addr, qurt_size_t size, int op, int type) {
    (void) addr;
    (void) size;
    (void) type;
    if (op == QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL) {
        g_dirty.clear();
        g_flushed_all = true;
    }
    return 0;
}

} // extern "C"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
    FuzzedDataProvider fdp(data, size);
    g_dirty.clear();
    g_flushed_bytes = 0;

    struct htp_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.n_threads     = fdp.ConsumeIntegralInRange<uint32_t>(1, HTP_MAX_NTHREADS);
    ctx.n_threads_div = init_fastdiv_values(ctx.n_threads);

    // The arena of the tensors: a DSP address range above zero
    const uint32_t ARENA = 0x40000000u;
    const uint32_t SPAN  = 24u << 20;

    std::vector<htp_tensor> pool(32);
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

        // proc_op_req (main.c:1069): the inputs must be clean in DDR after this call
        htp_tensor_flush_all(&ctx, srcs, HTP_OP_MAX_INPUTS);
        for (int s = 0; s < n_src; s++) {
            const uint64_t a = srcs[s]->data, b = a + srcs[s]->size;
            if (const uint64_t x = first_dirty(a, b)) {
                fakedsp::violation("dirty-stale-input", "op %d: input %d [0x%" PRIx64 ", 0x%" PRIx64 ") holds the dirty byte 0x%" PRIx64
                                   " after htp_tensor_flush_all (the DMA reads stale DDR)", i, s, a, b, x);
            }
        }

        // proc_op_req (main.c:1087): the outputs go into the tracker, then the op writes them
        htp_tensor_dirty_all(&ctx, dsts, HTP_OP_MAX_OUTPUTS);
        for (int d = 0; d < n_dst; d++) {
            if (!(dsts[d]->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
                mark(dsts[d]->data, (uint64_t) dsts[d]->data + dsts[d]->size);
            }
        }

        // Each dirty byte must be inside a range of the tracker
        for (const auto & kv : g_dirty) {
            for (uint64_t x = kv.first; x < kv.second;) {
                bool     inside = false;
                uint64_t next   = kv.second;
                for (int r = 0; r < HTP_MAX_DIRTY_RANGES; r++) {
                    const auto & dr = ctx.dirty_ranges[r];
                    if (dr.start && x >= dr.start && x < dr.end) {
                        inside = true;
                        next   = dr.end < kv.second ? dr.end : kv.second;
                        break;
                    }
                }
                if (!inside) {
                    fakedsp::violation("dirty-lost", "op %d: the dirty byte 0x%" PRIx64 " is in no range of the tracker", i, x);
                }
                x = next;
            }
        }

        const int ev = fdp.ConsumeIntegralInRange<int>(0, 9);
        if (ev == 0) {
            // op_fence (main.c:723): every dirty byte goes to DDR
            htp_flush_dirty_ranges(&ctx);
            if (!g_dirty.empty()) {
                fakedsp::violation("dirty-fence-leftover", "op %d: after htp_flush_dirty_ranges the byte 0x%" PRIx64 " is dirty", i,
                                   g_dirty.begin()->first);
            }
        } else if (ev == 1) {
            // the end of a batch (main.c:1216) and the start of the next (main.c:1147-1149)
            qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
            memset(ctx.dirty_ranges, 0, sizeof(ctx.dirty_ranges));
        }
    }
    return 0;
}
