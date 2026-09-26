// The replay of the dirty range tracker of the DSP on the batches of hexhost_graphs (refer to
// tracker_replay.h). Build it with the stubs of stubs/dsp before the real headers of htp/.

#include "tracker_replay.h"

#include "dirty_model.h"
#include "dsp_model.h"
#include "fake_dsp.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

extern "C" {
#include "hex-fastdiv.h"
#include "htp-ctx.h"
}

namespace tracker_replay {

namespace {

static_assert(HTP_OP_MAX_INPUTS <= 10 && HTP_OP_MAX_OUTPUTS <= 4, "the op record of fake_dsp.h holds 10 inputs and 4 outputs");

constexpr uint64_t k_first = 16ull << 20;  // the DSP address of the first buffer of a batch
constexpr uint64_t k_page  = 4096;         // the alignment of each buffer

std::mutex               g_mutex;
counts                   g_total;
std::vector<std::string> g_lines;

// The settings of the replay (install reads them from the environment)
uint64_t g_gap           = 1ull << 20;  // HEXHOST_TRACKER_GAP: the gap between two buffers
uint32_t g_reserve       = 0;           // HEXHOST_TRACKER_RESERVE: the ranges in use at the start of a batch
uint64_t g_reserve_bytes = 256;         // the dirty bytes of each such range

// Reads an unsigned number of the environment variable `name` (decimal, or hex with 0x) from s to the
// character `stop`. Gives the character after the number. Stops the process when s has no such number.
const char * parse_number(const char * name, const char * s, char stop, uint64_t & out) {
    char *   end = nullptr;
    uint64_t v   = strtoull(s, &end, 0);
    if (end == s || (*end != stop && *end != '\0') || *s == '-') {
        fprintf(stderr, "hexhost_graphs: %s=%s is not valid (refer to graphs/tracker_replay.h)\n", name, getenv(name));
        exit(2);
    }
    out = v;
    return *end ? end + 1 : end;
}

// Gives the descriptor of a tensor with its DSP address. Gives false when no buffer of the batch holds the tensor.
bool to_dsp(const fakedsp::tensor_ref & t, const htp_buf_desc * bufs, const std::vector<uint64_t> & base, htp_tensor & h) {
    for (size_t i = 0; i < base.size(); i++) {
        if (t.addr >= bufs[i].base && t.addr + t.size <= bufs[i].base + bufs[i].size) {
            memset(&h, 0, sizeof(h));
            h.data  = (uint32_t) (base[i] + (t.addr - bufs[i].base));
            h.size  = t.size;
            h.flags = t.flags;
            h.type  = t.type;
            h.bi    = (uint16_t) i;
            memcpy(h.ne, t.ne, sizeof(h.ne));
            memcpy(h.nb, t.nb, sizeof(h.nb));
            return true;
        }
    }
    return false;
}

// Runs the tracker on the ops of one batch and adds the counts. O(ops * (k + n_bufs) * HTP_MAX_DIRTY_RANGES)
// for k intervals in the dirty set.
void observe(const htp_buf_desc * bufs, uint32_t n_bufs, const fakedsp::batch_record & rec) {
    std::lock_guard<std::mutex> lock(g_mutex);

    // The DSP address of each buffer: the order of the table, from k_first, a gap between two buffers.
    // The reserved ranges come after the buffers, one page or more apart.
    std::vector<uint64_t> base(n_bufs);
    uint64_t              next = k_first;
    for (uint32_t i = 0; i < n_bufs; i++) {
        base[i] = next;
        next    = (next + bufs[i].size + g_gap + k_page - 1) / k_page * k_page;
    }
    const uint64_t reserve_step = (g_reserve_bytes + k_page + k_page - 1) / k_page * k_page;
    if (next + g_reserve * reserve_step > UINT32_MAX) {
        g_total.skipped++;
        return;
    }

    htp_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.n_threads     = fakedsp::get_config().n_threads;
    ctx.n_threads_div = init_fastdiv_values(ctx.n_threads);
    dirty_model::batch_edge(ctx);
    for (uint32_t i = 0; i < g_reserve; i++) {
        const uint64_t start = next + i * reserve_step;
        dirty_model::occupy(ctx, i, (uint32_t) start, (uint32_t) (start + g_reserve_bytes));
    }

    counts                          c;
    std::map<std::string, uint64_t> evict_ops;  // the ops that evict, by opcode name
    for (const auto & op : rec.ops) {
        htp_tensor                src_t[HTP_OP_MAX_INPUTS];
        htp_tensor                dst_t[HTP_OP_MAX_OUTPUTS];
        const struct htp_tensor * srcs[HTP_OP_MAX_INPUTS]  = { nullptr };
        const struct htp_tensor * dsts[HTP_OP_MAX_OUTPUTS] = { nullptr };
        for (int s = 0; s < HTP_OP_MAX_INPUTS; s++) {
            if (op.src[s].present) {
                if (to_dsp(op.src[s], bufs, base, src_t[s])) {
                    srcs[s] = &src_t[s];
                } else {
                    c.unmapped++;
                }
            }
        }
        for (int d = 0; d < HTP_OP_MAX_OUTPUTS; d++) {
            if (op.dst[d].present) {
                if (to_dsp(op.dst[d], bufs, base, dst_t[d])) {
                    dsts[d] = &dst_t[d];
                } else {
                    c.unmapped++;
                }
            }
        }

        const dirty_model::op_result r = dirty_model::run_op(ctx, srcs, dsts);
        c.ops++;
        if (r.path != dirty_model::dirty_path::keep) {
            c.evict++;
            c.evict_bytes += r.evict_bytes;
            evict_ops[std::string(fakedsp::opcode_name(op.opcode)) + (r.path == dirty_model::dirty_path::flush_all ? " flush-all" : "")]++;
        }
        c.flush_all += r.path == dirty_model::dirty_path::flush_all;
        c.lost += r.lost_byte != 0;
        c.stale += r.stale_input >= 0;
        c.max_ranges = std::max(c.max_ranges, r.n_ranges);

        const bool fence = (op.opcode == HTP_OP_FENCE && op.params[1] == 1) || op.opcode == HTP_OP_CPY_FENCE;
        if (fence) {
            c.fences++;
            c.fence_left += dirty_model::fence(ctx) != 0;
        }
    }
    dirty_model::batch_edge(ctx);

    if (c.evict || c.lost || c.stale || c.fence_left || c.unmapped) {
        std::string line = "batch " + std::to_string(rec.seq) + " queue " + std::to_string(rec.queue) + ": " + summary(c);
        for (const auto & kv : evict_ops) {
            line += ", " + kv.first + " " + std::to_string(kv.second);
        }
        g_lines.push_back(line);
    }
    g_total.batches++;
    g_total.unmapped += c.unmapped;
    g_total.ops += c.ops;
    g_total.fences += c.fences;
    g_total.evict += c.evict;
    g_total.flush_all += c.flush_all;
    g_total.evict_bytes += c.evict_bytes;
    g_total.lost += c.lost;
    g_total.stale += c.stale;
    g_total.fence_left += c.fence_left;
    g_total.max_ranges = std::max(g_total.max_ranges, c.max_ranges);
}

} // namespace

void install() {
    if (const char * s = getenv("HEXHOST_TRACKER_GAP")) {
        parse_number("HEXHOST_TRACKER_GAP", s, '\0', g_gap);
    }
    if (const char * s = getenv("HEXHOST_TRACKER_RESERVE")) {
        uint64_t n = 0;
        s          = parse_number("HEXHOST_TRACKER_RESERVE", s, ':', n);
        if (*s) {
            parse_number("HEXHOST_TRACKER_RESERVE", s, '\0', g_reserve_bytes);
        }
        if (n < 1 || n > HTP_MAX_DIRTY_RANGES || g_reserve_bytes < 1 || g_reserve_bytes > (1ull << 30)) {
            fprintf(stderr, "hexhost_graphs: HEXHOST_TRACKER_RESERVE=%s: N must be 1 to %d and BYTES 1 to 1 GiB\n",
                    getenv("HEXHOST_TRACKER_RESERVE"), HTP_MAX_DIRTY_RANGES);
            exit(2);
        }
        g_reserve = (uint32_t) n;
    }
    fakedsp::set_batch_observer(observe);
}

void reset() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_total = counts();
    g_lines.clear();
}

counts total() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_total;
}

std::vector<std::string> lines() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_lines;
}

std::string summary(const counts & c) {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "%" PRIu64 " ops, %" PRIu64 " evictions, %" PRIu64 " flushes of the whole cache, %" PRIu64 " bytes of evicted lines, "
             "%" PRIu64 " ops with a lost byte, %" PRIu64 " ops with a stale input, %" PRIu64 " fences, %" PRIu64
             " fences with a dirty byte left, %u ranges at most, %" PRIu64 " unmapped tensors",
             c.ops, c.evict, c.flush_all, c.evict_bytes, c.lost, c.stale, c.fences, c.fence_left, c.max_ranges, c.unmapped);
    std::string s = buf;
    if (c.batches || c.skipped) {
        s = std::to_string(c.batches) + " batches, " + std::to_string(c.skipped) + " batches over 4 GiB, " + s;
        if (g_reserve) {
            s += ", " + std::to_string(g_reserve) + " reserved ranges of " + std::to_string(g_reserve_bytes) + " bytes";
        }
    }
    return s;
}

} // namespace tracker_replay
