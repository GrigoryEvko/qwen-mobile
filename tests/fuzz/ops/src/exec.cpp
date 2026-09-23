// The runner. Refer to exec.h.
//
// The layout of each buffer: a guard region, then for each tensor its bytes and a guard region
// after them. The buffer is filled with the byte 0xA5 before the tensors get their data, thus a
// guard byte that is not 0xA5 after the run is a write outside the tensor. The runner does not use
// the graph allocator: each tensor has its own bytes, and no tensor shares bytes with another.

#include "exec.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace fo {

namespace {

constexpr uint8_t GUARD_BYTE = 0xA5;
constexpr size_t  GUARD_SIZE = 256;

// One tensor of a buffer plan.
struct region {
    ggml_tensor * t    = nullptr;
    size_t        off  = 0;
    size_t        size = 0;  // the allocation size of the backend
};

// One buffer and the tensors in it.
struct buf_plan {
    ggml_backend_buffer_type_t buft    = nullptr;
    bool                       weights = false;
    std::vector<region>        tensors;
    std::vector<std::pair<size_t, size_t>> guards;   // [offset, size]
    std::vector<std::string>   guard_owner;          // the tensor before each guard
    size_t                     total = 0;
    ggml_backend_buffer_t      buf   = nullptr;
};

size_t align_up(size_t x, size_t a) {
    return (x + a - 1) / a * a;
}

// Give the tensors of the plan their offsets and the guard regions between them. O(tensors).
void plan_layout(buf_plan & p) {
    const size_t al  = ggml_backend_buft_get_alignment(p.buft);
    size_t       off = align_up(GUARD_SIZE, al);
    p.guards.push_back({ 0, off });
    p.guard_owner.push_back("the buffer start");
    for (auto & r : p.tensors) {
        r.off  = off;
        r.size = ggml_backend_buft_get_alloc_size(p.buft, r.t);
        const size_t end  = off + r.size;
        const size_t next = align_up(end + GUARD_SIZE, al);
        p.guards.push_back({ end, next - end });
        p.guard_owner.push_back(r.t->name);
        off = next;
    }
    p.total = off;
}

// Read `size` bytes at `off` of the buffer. A buffer without get_tensor (the CPU repack buffer) is
// host memory, and the base pointer reads it.
bool read_buffer(ggml_backend_buffer_t buf, ggml_context * ctx, size_t off, size_t size, std::vector<uint8_t> & out) {
    out.resize(size);
    if (buf->iface.get_tensor == nullptr) {
        std::memcpy(out.data(), (const uint8_t *) ggml_backend_buffer_get_base(buf) + off, size);
        return true;
    }
    // a temporary I8 tensor over the bytes, and the get_tensor of the buffer
    ggml_tensor * g = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) size);
    if (g == nullptr) {
        return false;
    }
    if (ggml_backend_tensor_alloc(buf, g, (uint8_t *) ggml_backend_buffer_get_base(buf) + off) != GGML_STATUS_SUCCESS) {
        return false;
    }
    ggml_backend_tensor_get(g, out.data(), 0, size);
    return true;
}

// Return true when the tensor is a leaf of the case.
const leaf * find_leaf(const built_case & c, const ggml_tensor * t) {
    for (const auto & l : c.leaves) {
        if (l.t == t) {
            return &l;
        }
    }
    return nullptr;
}

void noop_dep(void *, ggml_tensor *, ggml_tensor *) {}

} // namespace

uint64_t case_hash(const built_case & c) {
    // FUZZ_OPS_TRACE=1 prints the hash of each input, to find the first input that differs.
    static const bool trace = std::getenv("FUZZ_OPS_TRACE") != nullptr;
    uint64_t h = fnv1a((const uint8_t *) c.desc.data(), c.desc.size());
    for (const auto & l : c.leaves) {
        const uint64_t lh = fnv1a(l.bytes.data(), l.bytes.size());
        if (trace) {
            std::fprintf(stderr, "trace: %s %s [%lld,%lld,%lld,%lld] %s\n", l.t->name, ggml_type_name(l.t->type),
                         (long long) l.t->ne[0], (long long) l.t->ne[1], (long long) l.t->ne[2], (long long) l.t->ne[3],
                         hex64(lh).c_str());
        }
        h ^= lh + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    }
    if (trace) {
        std::fprintf(stderr, "trace: desc '%s' threads %d special %d repack %d\n", c.desc.c_str(), c.n_threads,
                     (int) c.special, (int) c.cpu_repack);
    }
    return h ^ (uint64_t) c.n_threads ^ ((uint64_t) c.special << 8) ^ ((uint64_t) c.cpu_repack << 9);
}

const char * run_status_name(uint32_t s) {
    switch (s) {
        case RUN_OK:          return "ok";
        case RUN_UNSUPPORTED: return "unsupported";
        case RUN_ALLOC:       return "alloc-failed";
        case RUN_COMPUTE:     return "compute-failed";
        case RUN_CRASHED:     return "crashed";
        case RUN_INVALID:     return "invalid";
    }
    return "?";
}

bool open_backend(const std::string & dev_name, backend_ctx & be, std::string & why) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (dev == nullptr) {
        why = "no device " + dev_name + " (the devices:";
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            why += std::string(" ") + ggml_backend_dev_name(ggml_backend_dev_get(i));
        }
        why += ")";
        return false;
    }
    be.dev     = dev;
    be.backend = ggml_backend_dev_init(dev, nullptr);
    if (be.backend == nullptr) {
        why = "the device " + dev_name + " gave no backend";
        return false;
    }
    be.buft   = ggml_backend_dev_buffer_type(dev);
    be.is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    be.name   = dev_name;
    if (be.is_cpu) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        using extra_fn = ggml_backend_buffer_type_t * (*) (ggml_backend_dev_t);
        auto fn = (extra_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        if (fn) {
            for (ggml_backend_buffer_type_t * p = fn(dev); p && *p; p++) {
                if (std::string(ggml_backend_buft_name(*p)) == "CPU_REPACK") {
                    be.repack_buft = *p;
                }
            }
        }
        if (be.use_ref) {
            using ref_fn = void (*)(ggml_backend_t, bool);
            auto set_ref = (ref_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref");
            if (set_ref == nullptr) {
                why = "the CPU backend has no ggml_backend_cpu_set_use_ref";
                return false;
            }
            set_ref(be.backend, true);
        }
    }
    return true;
}

void close_backend(backend_ctx & be) {
    if (be.backend) {
        ggml_backend_free(be.backend);
    }
    be.backend = nullptr;
}

run_result run_case(built_case & c, backend_ctx & be) {
    run_result r;
    r.input_hash = case_hash(c);

    // 1. The scheduler lets the backend reorder the graph before the allocation. Do the same.
    if (be.backend->iface.graph_optimize) {
        ggml_backend_graph_optimize_params op = { noop_dep, nullptr };
        be.backend->iface.graph_optimize(be.backend, c.gf, &op);
    }

    // 2. The support check. It comes before the allocation, like in the scheduler: the Hexagon
    // backend marks the quantized weights of MUL_MAT for repacking here.
    const int n_nodes = ggml_graph_n_nodes(c.gf);
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * n = ggml_graph_node(c.gf, i);
        if (!ggml_backend_supports_op(be.backend, n)) {
            r.status = RUN_UNSUPPORTED;
            r.detail = std::string(ggml_op_desc(n)) + " " + ggml_type_name(n->type);
            for (int s = 0; s < GGML_MAX_SRC && n->src[s]; s++) {
                char buf[96];
                std::snprintf(buf, sizeof(buf), " src%d=%s[%lld,%lld,%lld,%lld]", s, ggml_type_name(n->src[s]->type),
                              (long long) n->src[s]->ne[0], (long long) n->src[s]->ne[1],
                              (long long) n->src[s]->ne[2], (long long) n->src[s]->ne[3]);
                r.detail += buf;
            }
            return r;
        }
    }

    // 3. The buffer plans: data, weights, and CPU repacked weights.
    buf_plan data, weights, repack;
    data.buft       = be.buft;
    weights.buft    = be.buft;
    weights.weights = true;
    repack.buft     = be.repack_buft;
    repack.weights  = true;

    ggml_backend_buffer_t dummy = nullptr;
    if (be.is_cpu && !be.use_ref && c.cpu_repack && be.repack_buft) {
        dummy = ggml_backend_buft_alloc_buffer(be.repack_buft, 0);
    }
    for (auto & l : c.leaves) {
        if (l.role != leaf_role::WEIGHT) {
            data.tensors.push_back({ l.t, 0, 0 });
            continue;
        }
        bool use_repack = false;
        if (dummy) {
            // like llama: the weight goes into the repack buffer when the extra buffer type takes its op
            for (int i = 0; i < n_nodes; i++) {
                ggml_tensor * n = ggml_graph_node(c.gf, i);
                if ((n->op == GGML_OP_MUL_MAT || n->op == GGML_OP_MUL_MAT_ID) && n->src[0] == l.t) {
                    l.t->buffer = dummy;
                    use_repack  = ggml_backend_dev_supports_op(be.dev, n);
                    l.t->buffer = nullptr;
                    break;
                }
            }
        }
        (use_repack ? repack : weights).tensors.push_back({ l.t, 0, 0 });
    }
    if (dummy) {
        ggml_backend_buffer_free(dummy);
    }
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * n = ggml_graph_node(c.gf, i);
        if (n->view_src == nullptr && find_leaf(c, n) == nullptr) {
            data.tensors.push_back({ n, 0, 0 });
        }
    }

    buf_plan * plans[3] = { &data, &weights, &repack };
    for (buf_plan * p : plans) {
        if (p->tensors.empty()) {
            continue;
        }
        plan_layout(*p);
        p->buf = ggml_backend_buft_alloc_buffer(p->buft, p->total);
        if (p->buf == nullptr) {
            r.status = RUN_ALLOC;
            r.detail = std::string("buffer of ") + std::to_string(p->total) + " bytes in " + ggml_backend_buft_name(p->buft);
            for (buf_plan * q : plans) {
                if (q->buf) {
                    ggml_backend_buffer_free(q->buf);
                }
            }
            return r;
        }
        if (p->weights) {
            ggml_backend_buffer_set_usage(p->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        ggml_backend_buffer_clear(p->buf, GUARD_BYTE);
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(p->buf);
        for (auto & t : p->tensors) {
            if (ggml_backend_tensor_alloc(p->buf, t.t, base + t.off) != GGML_STATUS_SUCCESS) {
                r.status = RUN_ALLOC;
                r.detail = std::string("tensor ") + t.t->name;
            }
        }
    }
    auto free_all = [&]() {
        for (buf_plan * p : plans) {
            if (p->buf) {
                ggml_backend_buffer_free(p->buf);
                p->buf = nullptr;
            }
        }
    };
    if (r.status != RUN_OK) {
        free_all();
        return r;
    }
    // the views: a view takes the bytes of its source
    for (ggml_tensor * t = ggml_get_first_tensor(c.ctx); t != nullptr; t = ggml_get_next_tensor(c.ctx, t)) {
        if (t->view_src && t->data == nullptr && t->view_src->buffer) {
            ggml_backend_view_init(t);
        }
    }

    // 4. The inputs.
    for (const auto & l : c.leaves) {
        ggml_backend_tensor_set(l.t, l.bytes.data(), 0, l.bytes.size());
    }

    // 5. The compute.
    if (be.is_cpu) {
        ggml_backend_cpu_set_n_threads(be.backend, be.force_threads > 0 ? be.force_threads : c.n_threads);
    }
    const auto          t0 = std::chrono::steady_clock::now();
    const ggml_status   st = ggml_backend_graph_compute(be.backend, c.gf);
    ggml_backend_synchronize(be.backend);
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (st != GGML_STATUS_SUCCESS) {
        r.status = RUN_COMPUTE;
        r.detail = std::string("graph_compute status ") + ggml_status_to_string(st);
        free_all();
        return r;
    }

    // 6. The outputs. The async get checks the DSP status like llama does.
    r.outs.resize(c.outs.size());
    for (size_t i = 0; i < c.outs.size(); i++) {
        r.outs[i].resize(ggml_nbytes(c.outs[i].t));
        ggml_backend_tensor_get_async(be.backend, c.outs[i].t, r.outs[i].data(), 0, r.outs[i].size());
    }
    ggml_backend_synchronize(be.backend);

    // 7. The guard regions.
    std::vector<uint8_t> tmp;
    for (buf_plan * p : plans) {
        if (!p->buf) {
            continue;
        }
        for (size_t gi = 0; gi < p->guards.size(); gi++) {
            if (!read_buffer(p->buf, c.ctx, p->guards[gi].first, p->guards[gi].second, tmp)) {
                continue;
            }
            // The extent of the write: the first and the last changed byte and the count. The
            // extent tells the store width (for example a full 128-byte HVX vector). O(guard size).
            size_t first = tmp.size(), last = 0, count = 0;
            for (size_t j = 0; j < tmp.size(); j++) {
                if (tmp[j] != GUARD_BYTE) {
                    first = std::min(first, j);
                    last  = j;
                    count++;
                }
            }
            if (count > 0) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "guard write after %s at +%zu (byte 0x%02x), %zu bytes changed in +%zu..+%zu of %zu, in %s; ",
                              p->guard_owner[gi].c_str(), first, tmp[first], count, first, last, tmp.size(),
                              ggml_backend_buft_name(p->buft));
                r.flags |= FLAG_GUARD;
                r.detail += buf;
            }
        }
    }

    // 8. The inputs must not change. A STATE input is written by the graph, and a repacked CPU
    // weight has no get_tensor.
    for (const auto & l : c.leaves) {
        if (l.role == leaf_role::STATE || l.t->buffer == repack.buf) {
            continue;
        }
        tmp.resize(l.bytes.size());
        ggml_backend_tensor_get(l.t, tmp.data(), 0, tmp.size());
        if (std::memcmp(tmp.data(), l.bytes.data(), tmp.size()) != 0) {
            size_t j = 0;
            while (j < tmp.size() && tmp[j] == l.bytes[j]) {
                j++;
            }
            char buf[256];
            std::snprintf(buf, sizeof(buf), "input %s (%s) changed at byte %zu of %zu; ", l.t->name,
                          ggml_type_name(l.t->type), j, tmp.size());
            r.flags |= FLAG_INPUT;
            r.detail += buf;
        }
    }

    free_all();
    return r;
}

} // namespace fo
