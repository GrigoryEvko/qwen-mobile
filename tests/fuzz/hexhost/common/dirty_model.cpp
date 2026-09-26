// The dirty model of the hexhost harness (refer to dirty_model.h). Build it with the stubs of
// stubs/dsp before the real headers of htp/.

#include "dirty_model.h"

#include <cstring>
#include <iterator>
#include <map>

extern "C" {
#include "qurt.h"
#include "hex-utils.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
}

namespace {

// The dirty set as disjoint intervals [start, end)
std::map<uint64_t, uint64_t> g_dirty;
uint64_t                     g_range_calls = 0;  // the calls of hex_l2flush
uint64_t                     g_line_bytes  = 0;  // the bytes of the lines that hex_l2flush flushed
uint64_t                     g_full        = 0;  // the flushes of the whole cache

// Removes [s, e) from the dirty set. O(log n + k) for k intervals that [s, e) touches.
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

// Adds [s, e) to the dirty set.
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

// Gives the first dirty byte that no range of the tracker holds, or 0.
uint64_t first_lost(const htp_context & ctx) {
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
                return x;
            }
            x = next;
        }
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
    g_range_calls++;
    g_line_bytes += e - s;
}

// The flush of the whole data cache
int qurt_mem_cache_clean(qurt_addr_t addr, qurt_size_t size, int op, int type) {
    (void) addr;
    (void) size;
    (void) type;
    if (op == QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL) {
        g_dirty.clear();
        g_full++;
    }
    return 0;
}

} // extern "C"

namespace dirty_model {

const char * path_name(dirty_path p) {
    switch (p) {
        case dirty_path::keep:
            return "keep";
        case dirty_path::evict:
            return "evict";
        case dirty_path::flush_all:
            return "flush-all";
    }
    return "unknown";
}

void batch_edge(htp_context & ctx) {
    qurt_mem_cache_clean(0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    memset(ctx.dirty_ranges, 0, sizeof(ctx.dirty_ranges));
}

op_result run_op(htp_context & ctx, const htp_tensor * const * srcs, const htp_tensor * const * dsts) {
    op_result res;

    // The inputs must be clean in DDR after this call, else the DMA of the op reads stale DDR
    htp_tensor_flush_all(&ctx, srcs, HTP_OP_MAX_INPUTS);
    for (int s = 0; s < HTP_OP_MAX_INPUTS && res.stale_input < 0; s++) {
        if (srcs[s]) {
            if (const uint64_t x = first_dirty(srcs[s]->data, (uint64_t) srcs[s]->data + srcs[s]->size)) {
                res.stale_input = s;
                res.stale_byte  = x;
            }
        }
    }

    // The outputs go into the tracker, then the op writes them
    const uint64_t calls = g_range_calls, bytes = g_line_bytes, full = g_full;
    htp_tensor_dirty_all(&ctx, dsts, HTP_OP_MAX_OUTPUTS);
    if (g_full != full) {
        res.path = dirty_path::flush_all;
    } else if (g_range_calls != calls) {
        res.path = dirty_path::evict;
    }
    res.evict_bytes = g_line_bytes - bytes;
    for (int d = 0; d < HTP_OP_MAX_OUTPUTS; d++) {
        if (dsts[d] && !(dsts[d]->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            mark(dsts[d]->data, (uint64_t) dsts[d]->data + dsts[d]->size);
        }
    }

    res.lost_byte = first_lost(ctx);
    for (int r = 0; r < HTP_MAX_DIRTY_RANGES; r++) {
        res.n_ranges += ctx.dirty_ranges[r].start != 0;
    }
    return res;
}

uint64_t fence(htp_context & ctx) {
    htp_flush_dirty_ranges(&ctx);
    return g_dirty.empty() ? 0 : g_dirty.begin()->first;
}

void occupy(htp_context & ctx, uint32_t index, uint32_t start, uint32_t end) {
    ctx.dirty_ranges[index].start = start;
    ctx.dirty_ranges[index].end   = end;
    mark(start, end);
}

} // namespace dirty_model
