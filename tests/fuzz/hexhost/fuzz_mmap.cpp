// fuzz_mmap: the mapping table of the DSP (prep_op_bufs of htp/main.c), which maps the buffers of each op
// batch into the VA of the DSP and removes mappings when the new ones do not fit in max_vmem.
//
// CMakeLists.txt copies the functions of the table from main.c into one C file (dsp/htp-mmap.c), and this
// harness gives htp_mmap and htp_munmap: a fake VA, a first-fit allocator with a capacity, thus a map can
// fail and the second pass of prep_op_bufs (remove each mapping, then map the batch again) runs. A fuzz
// input gives a pool of buffers, max_vmem, the capacity of the VA (at least max_vmem, as on the phone), the
// flag HTP_OPBATCH_MMAP_LRU, and a sequence of batches. Each batch is a set of pool buffers with a sum of
// at most max_vmem and at most HTP_OP_MAX_BUFS buffers, as the host packs them (fit_op of
// ggml-hexagon.cpp). A batch can repeat an earlier batch, thus cycles occur, as the batches of the decode
// tokens.
//
// The checks after each batch:
//   - mmap-events: the map and unmap calls equal the calls of a model of the policy of the flag
//     (without the flag: each unused mapping goes when the new ones do not fit; with it: one mapping at a
//     time, the lowest batch number of last use first, then the smallest mapping that frees sufficient
//     bytes, else the largest one)
//   - mmap-table: each slot of the table equals the slot of the model
//   - mmap-batch-base: each buffer of the batch has a mapping at its base
//   - mmap-vmem: when the first pass maps the batch, the table holds at most max_vmem bytes
//     (the second pass runs when the VA has no free range or the table no free slot)
//   - mmap-unmap: each unmap names a live range of the VA with its size
//   - mmap-batch-removed: before a second pass, no removal takes a mapping of the batch

#include "fake_dsp.h"
#include "fuzz_death.h"
#include "harness.h"
#include "htp-mmap-api.h"
#include "htp-ops.h"

#include <fuzzer/FuzzedDataProvider.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr uint64_t MiB = 1024ull * 1024;

// A first-fit VA allocator in [base, base + capacity). Each call adds an event.
struct fake_va {
    uint64_t                     base     = 0x10000000;
    uint64_t                     capacity = 0;
    std::map<uint64_t, uint64_t> used;   // start -> size
    std::map<uint64_t, uint32_t> fd_at;  // start -> fd
    std::vector<std::string>     events;
    bool                         bad_unmap = false;

    uint64_t map(uint32_t fd, uint32_t size) {
        uint64_t at = base;
        for (const auto & [s, n] : used) {
            if (s - at >= size) {
                break;
            }
            at = s + n;
        }
        if (at + size > base + capacity) {
            events.push_back("fail " + std::to_string(fd));
            return 0;
        }
        used[at]  = size;
        fd_at[at] = fd;
        events.push_back("map " + std::to_string(fd));
        return at;
    }

    void unmap(uint64_t at, uint32_t size) {
        auto it = used.find(at);
        if (it == used.end() || it->second != size) {
            bad_unmap = true;
            events.push_back("bad-unmap");
            return;
        }
        events.push_back("unmap " + std::to_string(fd_at[at]));
        used.erase(it);
        fd_at.erase(at);
    }
};

fake_va g_dsp;    // the VA of the copy of main.c
fake_va g_model;  // the VA of the model

struct buffer {
    uint32_t fd;
    uint64_t size;
};

// The model of prep_op_bufs with the two policies. It keeps the slots in the order of the table.
struct model {
    struct slot {
        uint32_t fd   = 0;
        uint64_t size = 0;
        uint64_t base = 0;
        uint32_t last = 0;
    };
    std::vector<slot> table;
    uint64_t          max_vmem = 0;
    bool              second   = false;  // the last batch took the second pass

    void drop(size_t i) {
        g_model.unmap(table[i].base, (uint32_t) table[i].size);
        table[i] = slot{ (uint32_t) -1, 0, 0, table[i].last };
    }

    // Maps a buffer into the first free slot, as mmap_buf does. Gives false when no slot is free or the
    // VA has no free range.
    bool map_one(const buffer & b, uint32_t seq) {
        for (auto & s : table) {
            if (!s.size) {
                const uint64_t at = g_model.map(b.fd, (uint32_t) b.size);
                if (!at) {
                    return false;
                }
                s = slot{ b.fd, b.size, at, seq };
                return true;
            }
        }
        return false;
    }

    int find(uint32_t fd) const {
        for (size_t i = 0; i < table.size(); i++) {
            if (table[i].size && table[i].fd == fd) {
                return (int) i;
            }
        }
        return -1;
    }

    void prep(const std::vector<buffer> & bufs, uint32_t seq, bool lru) {
        second = false;
        std::vector<bool> used(table.size(), false);
        uint64_t          e_vmem  = 0;
        uint32_t          n_new   = 0;
        for (const auto & b : bufs) {
            const int i = find(b.fd);
            if (i >= 0) {
                used[i] = true;
            } else {
                e_vmem += b.size;
                n_new++;
            }
        }
        for (size_t i = 0; i < table.size(); i++) {
            if (used[i]) {
                table[i].last = seq;
            }
        }
        if (n_new == 0) {
            return;
        }
        uint64_t m_vmem = 0;
        uint32_t n_maps = 0;
        for (const auto & s : table) {
            if (s.size) {
                m_vmem += s.size;
                n_maps++;
            }
        }
        if (!lru) {
            if (m_vmem + e_vmem > max_vmem) {
                for (size_t i = 0; i < table.size(); i++) {
                    if (table[i].size && !used[i]) {
                        drop(i);
                    }
                }
            }
        } else {
            while (m_vmem + e_vmem > max_vmem || n_maps + n_new > table.size()) {
                const uint64_t need = m_vmem + e_vmem > max_vmem ? m_vmem + e_vmem - max_vmem : 0;
                uint32_t       age  = 0;
                for (size_t i = 0; i < table.size(); i++) {
                    if (table[i].size && !used[i] && seq - table[i].last > age) {
                        age = seq - table[i].last;
                    }
                }
                int fit = -1, big = -1;
                for (size_t i = 0; i < table.size(); i++) {
                    if (!table[i].size || used[i] || seq - table[i].last != age) {
                        continue;
                    }
                    if (table[i].size >= need && (fit < 0 || table[i].size < table[fit].size)) {
                        fit = (int) i;
                    }
                    if (big < 0 || table[i].size > table[big].size) {
                        big = (int) i;
                    }
                }
                const int v = fit >= 0 ? fit : big;
                if (v < 0) {
                    break;
                }
                m_vmem -= table[v].size;
                n_maps--;
                drop((size_t) v);
            }
        }
        bool ok = true;
        for (const auto & b : bufs) {
            if (find(b.fd) < 0 && !map_one(b, seq)) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            second = true;
            for (size_t i = 0; i < table.size(); i++) {
                if (table[i].size) {
                    drop(i);
                }
            }
            for (const auto & b : bufs) {
                if (!map_one(b, seq)) {
                    return;  // the copy aborts here, and the harness reports the abort
                }
            }
        }
    }
};

// The buffers of one batch: pool buffers in the order that the input gives, with a sum of at most
// max_vmem and at most HTP_OP_MAX_BUFS buffers. O(pool size).
std::vector<buffer> make_batch(FuzzedDataProvider & fdp, const std::vector<buffer> & pool, uint64_t max_vmem) {
    std::vector<buffer> out;
    uint64_t            sum   = 0;
    const size_t        start = fdp.ConsumeIntegralInRange<size_t>(0, pool.size() - 1);
    const size_t        want  = fdp.ConsumeIntegralInRange<size_t>(1, HTP_OP_MAX_BUFS);
    const size_t        step  = fdp.ConsumeIntegralInRange<size_t>(1, pool.size());
    for (size_t k = 0; k < pool.size() && out.size() < want; k++) {
        const buffer & b = pool[(start + k * step) % pool.size()];
        if (std::any_of(out.begin(), out.end(), [&](const buffer & o) { return o.fd == b.fd; })) {
            continue;
        }
        if (sum + b.size <= max_vmem) {
            out.push_back(b);
            sum += b.size;
        }
    }
    return out;
}

} // namespace

extern "C" void * htp_mmap(uint32_t fd, uint32_t size) {
    return (void *) (uintptr_t) g_dsp.map(fd, size);
}

extern "C" void htp_munmap(void * va, uint32_t size) {
    g_dsp.unmap((uint64_t) (uintptr_t) va, size);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
    FuzzedDataProvider fdp(data, size);

    // The limits: max_vmem of the phone is 3158 MiB for model chunks of 1024 MiB. A small unit gives the
    // same shapes with smaller numbers, and exercises the slot limit of the table.
    static const uint64_t units[] = { MiB, MiB, 64 * 1024, 4096 };
    const uint64_t        unit     = units[fdp.ConsumeIntegralInRange<size_t>(0, 3)];
    const uint64_t        max_vmem = unit * fdp.ConsumeIntegralInRange<uint64_t>(64, 3300);
    const uint64_t        slack    = harness::rare(fdp, 3) ? 0 : unit * fdp.ConsumeIntegralInRange<uint64_t>(0, 1024);
    const bool            lru      = fdp.ConsumeBool();

    g_dsp   = fake_va();
    g_model = fake_va();
    g_dsp.capacity = g_model.capacity = max_vmem + slack;

    const size_t        n_pool = fdp.ConsumeIntegralInRange<size_t>(1, 48);
    std::vector<buffer> pool;
    for (size_t i = 0; i < n_pool; i++) {
        uint64_t s = unit * fdp.ConsumeIntegralInRange<uint64_t>(1, 1100);
        s          = std::min(s, max_vmem);
        pool.push_back({ (uint32_t) (100 + i), s });
    }

    std::vector<uint8_t> ctx(hexmmap_ctx_size());
    hexmmap_init(ctx.data(), max_vmem);
    model m;
    m.table.resize(hexmmap_slots());
    for (auto & s : m.table) {
        s.fd = 0;
    }
    m.max_vmem = max_vmem;

    std::vector<std::vector<buffer>> seen;
    const uint32_t                   n_batches = fdp.ConsumeIntegralInRange<uint32_t>(1, 96);
    for (uint32_t seq = 1; seq <= n_batches && fdp.remaining_bytes() > 0; seq++) {
        std::vector<buffer> batch;
        if (!seen.empty() && !harness::rare(fdp, 3)) {
            batch = seen[fdp.ConsumeIntegralInRange<size_t>(0, seen.size() - 1)];
        } else {
            batch = make_batch(fdp, pool, max_vmem);
            seen.push_back(batch);
        }
        std::vector<htp_buf_desc> descs(batch.size());
        for (size_t i = 0; i < batch.size(); i++) {
            memset(&descs[i], 0, sizeof(descs[i]));
            descs[i].fd   = batch[i].fd;
            descs[i].size = batch[i].size;
        }

        const size_t ev0 = g_dsp.events.size();
        hexmmap_prep(ctx.data(), descs.data(), (uint32_t) descs.size(), seq, lru);
        m.prep(batch, seq, lru);

        if (g_dsp.bad_unmap) {
            fakedsp::violation("mmap-unmap", "batch %u: an unmap names no live range of the VA with its size", seq);
        }
        if (g_dsp.events != g_model.events) {
            size_t at = 0;
            while (at < g_dsp.events.size() && at < g_model.events.size() && g_dsp.events[at] == g_model.events[at]) {
                at++;
            }
            fakedsp::violation("mmap-events", "batch %u (lru %d): event %zu is \"%s\", the model gives \"%s\"", seq, (int) lru, at,
                               at < g_dsp.events.size() ? g_dsp.events[at].c_str() : "none",
                               at < g_model.events.size() ? g_model.events[at].c_str() : "none");
        }
        uint64_t total = 0;
        for (uint32_t i = 0; i < hexmmap_slots(); i++) {
            uint32_t fd   = 0;
            uint64_t sz   = 0;
            uint64_t base = 0;
            hexmmap_slot(ctx.data(), i, &fd, &sz, &base);
            const auto & s = m.table[i];
            if (sz != s.size || (sz && (fd != s.fd || base != s.base))) {
                fakedsp::violation("mmap-table", "batch %u: slot %u holds fd %u size %" PRIu64 ", the model fd %u size %" PRIu64,
                                   seq, i, fd, sz, s.fd, s.size);
            }
            total += sz;
        }
        // The events agree (mmap-events), thus the model tells whether the copy took the second pass
        const bool second_pass = m.second;
        for (size_t b = 0; b < descs.size(); b++) {
            auto it = g_dsp.fd_at.find(descs[b].base);
            if (!descs[b].base || it == g_dsp.fd_at.end() || it->second != descs[b].fd) {
                fakedsp::violation("mmap-batch-base", "batch %u: buffer fd %u has base 0x%" PRIx64 ", which is not its mapping", seq,
                                   descs[b].fd, (uint64_t) descs[b].base);
            }
        }
        if (!second_pass) {
            if (total > max_vmem) {
                fakedsp::violation("mmap-vmem", "batch %u: the table holds %" PRIu64 " bytes, max_vmem is %" PRIu64, seq, total,
                                   max_vmem);
            }
            for (size_t e = ev0; e < g_dsp.events.size(); e++) {
                if (g_dsp.events[e].rfind("unmap ", 0) != 0) {
                    continue;
                }
                const uint32_t fd = (uint32_t) strtoul(g_dsp.events[e].c_str() + 6, nullptr, 10);
                if (std::any_of(batch.begin(), batch.end(), [&](const buffer & b) { return b.fd == fd; })) {
                    fakedsp::violation("mmap-batch-removed", "batch %u: the table removed fd %u, a buffer of the batch", seq, fd);
                }
            }
        }
        fakedsp::count(second_pass ? "batches with a second pass" : "batches with one pass");
        fakedsp::count(lru ? "batches with the flag" : "batches without the flag");
    }
    return 0;
}
