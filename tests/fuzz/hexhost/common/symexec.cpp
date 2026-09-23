// The symbolic check of the hexhost graph fuzzer. Refer to symexec.h.

#include "symexec.h"

#include "fake_dsp.h"
#include "dsp_model.h"
#include "hexhost.h"

#include "ggml.h"
#include "ggml-backend.h"

#include "htp-ops.h"
#include "matmul-ops.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace symexec {

namespace {

constexpr val CONC = 1ull << 63;  // the tag of a concrete int32

enum tag : uint64_t {
    T_INIT = 1, T_UNINIT, T_INPUT, T_WEIGHT, T_OP, T_ROW, T_ALL, T_CONTENT, T_CONST, T_SETROW,
};

// Mixes a value into a hash. The result never has the concrete tag.
inline val mix(val h, val x) {
    h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= h >> 31;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebull;
    h ^= h >> 31;
    return h & ~CONC;
}

// The value of a byte that no write touched
inline val init_of(uint64_t addr) {
    return mix(T_INIT, addr);
}

// The content of a tensor whose n elements all hold v
val const_content(val v, uint64_t n) {
    val h = mix(T_CONTENT, n);
    for (uint64_t i = 0; i < n; i++) {
        h = mix(h, v);
    }
    return h;
}

// Calls f(byte offset, element size) for each element of a tensor in logical order.
// Complexity: O(number of elements).
template <typename F> void for_each_elem(uint32_t type, const int64_t ne[4], const uint64_t nb[4], F && f) {
    const int64_t blck = ggml_blck_size((ggml_type) type);
    const size_t  ts   = ggml_type_size((ggml_type) type);
    const int64_t n0   = ne[0] / blck;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < n0; i0++) {
                    f(i0 * nb[0] + i1 * nb[1] + i2 * nb[2] + i3 * nb[3], ts, i1, i2, i3);
                }
            }
        }
    }
}

// The shape of a ggml tensor as int64 arrays
struct shape {
    int64_t  ne[4];
    uint64_t nb[4];
    uint32_t type;
};

shape shape_of(const ggml_tensor * t) {
    shape s;
    for (int i = 0; i < 4; i++) {
        s.ne[i] = t->ne[i];
        s.nb[i] = t->nb[i];
    }
    s.type = t->type;
    return s;
}

shape shape_of(const fakedsp::tensor_ref & t) {
    shape s;
    for (int i = 0; i < 4; i++) {
        s.ne[i] = t.ne[i];
        s.nb[i] = t.nb[i];
    }
    s.type = t.type;
    return s;
}

uint64_t nelements(const int64_t ne[4]) {
    return (uint64_t) ne[0] * ne[1] * ne[2] * ne[3];
}

// The value of an op: a hash of the key (the HTP opcode), the params, the
// output shape and type, and the contents of the inputs.
val op_value(uint32_t key, const int32_t * params, const int64_t ne[4], uint32_t type, std::vector<val> contents,
             bool commutative) {
    if (commutative && contents.size() == 2 && contents[0] > contents[1]) {
        std::swap(contents[0], contents[1]);
    }
    val h = mix(T_OP, key);
    for (int i = 0; i < 16; i++) {
        h = mix(h, (uint32_t) (params ? params[i] : 0));
    }
    for (int i = 0; i < 4; i++) {
        h = mix(h, (uint64_t) ne[i]);
    }
    h = mix(h, type);
    for (val c : contents) {
        h = mix(h, c);
    }
    return h;
}

bool is_commutative(uint32_t key) {
    return key == HTP_OP_MUL || key == HTP_OP_ADD;
}

bool is_view_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}

const int32_t ZERO_PARAMS[16] = {0};

} // namespace

// ---- memory

val memory::get(uint64_t addr) {
    if (!(last_vec && addr >= last_base && addr < last_base + last_size)) {
        uint64_t base = 0, size = 0;
        int      fd   = -1;
        if (!fakedsp::lookup_alloc(addr, &base, &size, &fd)) {
            fakedsp::violation("sym-address", "the symbolic check reads address 0x%" PRIx64 " outside every rpcmem allocation", addr);
            return init_of(addr);
        }
        auto it = blocks.find(base);
        if (it == blocks.end()) {
            std::vector<val> v(size);
            for (uint64_t i = 0; i < size; i++) {
                v[i] = init_of(base + i);
            }
            it = blocks.emplace(base, std::move(v)).first;
        }
        last_base = base;
        last_size = size;
        last_vec  = &it->second;
    }
    return (*last_vec)[addr - last_base];
}

void memory::set(uint64_t addr, size_t n, val v) {
    for (size_t i = 0; i < n; i++) {
        get(addr + i);
        (*last_vec)[addr + i - last_base] = v;
    }
}

// ---- checker

checker::checker(graphgen::world & w) : w_(w) {
    for (ggml_tensor * t : w.weights) {
        weights_[(uint64_t) (uintptr_t) t->data] = t;
    }
}

bool checker::is_persistent(const ggml_tensor * root) const {
    return root->buffer && (root->buffer == w_.buf_c || root->buffer == w_.buf_w);
}

val checker::ref_get(const ggml_tensor * root, size_t off) {
    if (is_persistent(root)) {
        return ref_mem_.get((uint64_t) (uintptr_t) root->data + off);
    }
    std::vector<val> & s = ref_store_[root];
    if (s.empty()) {
        const size_t n = ggml_nbytes(root);
        s.resize(n);
        for (size_t i = 0; i < n; i++) {
            s[i] = mix(T_UNINIT, ((uint64_t) (uintptr_t) root) ^ (i << 20));
        }
    }
    if (off >= s.size()) {
        fakedsp::violation("sym-harness", "the reference reads byte %zu of %s, which has %zu bytes", off, root->name, s.size());
        return 0;
    }
    return s[off];
}

void checker::ref_set(const ggml_tensor * root, size_t off, size_t n, val v) {
    if (is_persistent(root)) {
        ref_mem_.set((uint64_t) (uintptr_t) root->data + off, n, v);
        return;
    }
    ref_get(root, off);
    std::vector<val> & s = ref_store_[root];
    for (size_t i = 0; i < n && off + i < s.size(); i++) {
        s[off + i] = v;
    }
}

// The root of a view chain and the byte offset of the view in it
static const ggml_tensor * root_of(const ggml_tensor * t, size_t * off) {
    const ggml_tensor * r = t->view_src ? t->view_src : t;
    *off = (size_t) ((const char *) t->data - (const char *) r->data);
    return r;
}

val checker::ref_content(const ggml_tensor * t) {
    size_t              base = 0;
    const ggml_tensor * r    = root_of(t, &base);
    if (r->buffer == w_.buf_w) {
        return mix(T_WEIGHT, (uint64_t) (uintptr_t) t->data);
    }
    const shape s = shape_of(t);
    val         h = mix(T_CONTENT, nelements(s.ne) / ggml_blck_size(t->type));
    for_each_elem(s.type, s.ne, s.nb, [&](uint64_t off, size_t, int64_t, int64_t, int64_t) { h = mix(h, ref_get(r, base + off)); });
    return h;
}

val checker::ref_row_content(const ggml_tensor * t, int64_t row) {
    size_t              base = 0;
    const ggml_tensor * r    = root_of(t, &base);
    if (r->buffer == w_.buf_w) {
        return mix(mix(T_WEIGHT, (uint64_t) (uintptr_t) t->data), (uint64_t) row);
    }
    shape s = shape_of(t);
    s.ne[1] = 1;
    s.ne[2] = 1;
    s.ne[3] = 1;
    base += (size_t) row * t->nb[1];
    val h = mix(T_CONTENT, s.ne[0] / ggml_blck_size(t->type));
    for_each_elem(s.type, s.ne, s.nb, [&](uint64_t off, size_t, int64_t, int64_t, int64_t) { h = mix(h, ref_get(r, base + off)); });
    return h;
}

void checker::ref_write_all(const ggml_tensor * t, val v) {
    size_t              base = 0;
    const ggml_tensor * r    = root_of(t, &base);
    const shape         s    = shape_of(t);
    for_each_elem(s.type, s.ne, s.nb, [&](uint64_t off, size_t ts, int64_t, int64_t, int64_t) { ref_set(r, base + off, ts, v); });
}

void checker::ref_write_row(const ggml_tensor * t, int64_t row, val v) {
    size_t              base = 0;
    const ggml_tensor * r    = root_of(t, &base);
    shape               s    = shape_of(t);
    s.ne[1] = s.ne[2] = s.ne[3] = 1;
    base += (size_t) row * t->nb[1];
    for_each_elem(s.type, s.ne, s.nb, [&](uint64_t off, size_t ts, int64_t, int64_t, int64_t) { ref_set(r, base + off, ts, v); });
}

bool checker::ref_ints(const ggml_tensor * t, std::vector<int64_t> & out) {
    out.clear();
    if (t->type != GGML_TYPE_I32 || t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    size_t              base = 0;
    const ggml_tensor * r    = root_of(t, &base);
    for (int64_t i = 0; i < t->ne[0]; i++) {
        const val v = ref_get(r, base + i * t->nb[0]);
        if (!(v & CONC)) {
            return false;
        }
        out.push_back((int32_t) (uint32_t) v);
    }
    return true;
}

void checker::set_input(ggml_tensor * t, const std::vector<int32_t> & ints, uint32_t step) {
    size_t              base = 0;
    const ggml_tensor * r    = root_of(t, &base);
    const shape         s    = shape_of(t);
    uint64_t            e    = 0;
    for_each_elem(s.type, s.ne, s.nb, [&](uint64_t off, size_t ts, int64_t, int64_t, int64_t) {
        const val v = ints.empty() ? mix(mix(mix(T_INPUT, (uint64_t) (uintptr_t) t), step), e)
                                   : (CONC | (uint32_t) ints[e < ints.size() ? e : 0]);
        ref_set(r, base + off, ts, v);
        dev_mem_.set((uint64_t) (uintptr_t) t->data + off, ts, v);
        if (!ints.empty()) {
            input_ints_[(uint64_t) (uintptr_t) t->data + off] = (int32_t) (uint32_t) v;
        } else {
            input_ints_.erase((uint64_t) (uintptr_t) t->data + off);
        }
        e++;
    });
}

void checker::run_reference(const graphgen::graph_spec & g) {
    std::vector<int64_t> rows;
    for (ggml_tensor * n : g.order) {
        if (is_view_op(n->op) || ggml_is_empty(n) || !(n->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            continue;
        }
        const uint32_t key = (uint32_t) hexhost::htp_opcode(n);
        switch (n->op) {
            case GGML_OP_GET_ROWS: {
                const ggml_tensor * src = n->src[0];
                if (ref_ints(n->src[1], rows) && src->ne[2] == 1 && src->ne[3] == 1 &&
                    std::all_of(rows.begin(), rows.end(), [&](int64_t r) { return r >= 0 && r < src->ne[1]; })) {
                    for (size_t j = 0; j < rows.size(); j++) {
                        ref_write_row(n, (int64_t) j, mix(T_ROW, ref_row_content(src, rows[j])));
                    }
                } else {
                    ref_write_all(n, mix(mix(T_ALL, ref_content(src)), ref_content(n->src[1])));
                }
                break;
            }
            case GGML_OP_SET_ROWS: {
                const ggml_tensor * b = n->src[0];
                if (ref_ints(n->src[1], rows) && b->ne[2] == 1 && b->ne[3] == 1 &&
                    std::all_of(rows.begin(), rows.end(), [&](int64_t r) { return r >= 0 && r < n->ne[1]; })) {
                    for (size_t j = 0; j < rows.size(); j++) {
                        ref_write_row(n, rows[j], mix(T_SETROW, ref_row_content(b, (int64_t) j)));
                    }
                } else {
                    ref_write_all(n, mix(mix(T_ALL + 100, ref_content(b)), ref_content(n->src[1])));
                }
                break;
            }
            case GGML_OP_CPY:
            case GGML_OP_CONT:
            case GGML_OP_DUP: {
                ref_write_all(n, op_value(HTP_OP_CPY, ZERO_PARAMS, n->ne, n->type, { ref_content(n->src[0]) }, false));
                break;
            }
            default: {
                std::vector<val> c;
                for (int i = 0; i < GGML_MAX_SRC; i++) {
                    if (n->src[i]) {
                        c.push_back(ref_content(n->src[i]));
                    }
                }
                ref_write_all(n, op_value(key, n->op_params, n->ne, n->type, c, is_commutative(key)));
                break;
            }
        }
    }
}

// ---- the device

val checker::dev_content(const fakedsp::tensor_ref & t) {
    if (t.flags & HTP_TENSOR_WEIGHT) {
        return mix(T_WEIGHT, t.addr);
    }
    const shape s = shape_of(t);
    val         h = mix(T_CONTENT, nelements(s.ne) / ggml_blck_size((ggml_type) t.type));
    for_each_elem(s.type, s.ne, s.nb, [&](uint64_t off, size_t, int64_t, int64_t, int64_t) { h = mix(h, dev_mem_.get(t.addr + off)); });
    return h;
}

val checker::dev_row_content(const fakedsp::tensor_ref & t, int64_t row) {
    if (t.flags & HTP_TENSOR_WEIGHT) {
        return mix(mix(T_WEIGHT, t.addr), (uint64_t) row);
    }
    shape s = shape_of(t);
    s.ne[1] = s.ne[2] = s.ne[3] = 1;
    const uint64_t base = t.addr + (uint64_t) row * t.nb[1];
    val            h    = mix(T_CONTENT, s.ne[0] / ggml_blck_size((ggml_type) t.type));
    for_each_elem(s.type, s.ne, s.nb, [&](uint64_t off, size_t, int64_t, int64_t, int64_t) { h = mix(h, dev_mem_.get(base + off)); });
    return h;
}

void checker::dev_write_all(const fakedsp::tensor_ref & t, val v, int64_t only_row) {
    const shape s = shape_of(t);
    for_each_elem(s.type, s.ne, s.nb, [&](uint64_t off, size_t ts, int64_t i1, int64_t i2, int64_t i3) {
        if (only_row < 0 || (i1 == only_row && i2 == 0 && i3 == 0)) {
            dev_mem_.set(t.addr + off, ts, v);
        }
    });
}

bool checker::dev_ints(const fakedsp::tensor_ref & t, std::vector<int64_t> & out) {
    out.clear();
    if (t.type != GGML_TYPE_I32 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        return false;
    }
    for (uint32_t i = 0; i < t.ne[0]; i++) {
        const val v = dev_mem_.get(t.addr + (uint64_t) i * t.nb[0]);
        if (!(v & CONC)) {
            return false;
        }
        out.push_back((int32_t) (uint32_t) v);
    }
    return true;
}

// The host reads the slot index of a fused recurrent op from the s_copy bytes at enqueue time
// (ggml-hexagon.cpp:3813) and puts it into kernel_params[0]. At the position of the op in the op
// stream, the s_copy bytes must still hold that index. When an earlier op of the stream wrote
// the bytes (the allocator gave them to a later tensor, or an op computes the index), the host
// read a value that the graph does not give.
// The check compares the slot of the host with the index that the harness wrote. A reuse of the
// bytes with the correct value in real memory (no batch went to the DSP in between) is not a
// wrong result of the fused op, thus the check does not report it.
void checker::check_slot(const fakedsp::op_record & op, const fakedsp::tensor_ref & idx) {
    if (!idx.present) {
        return;
    }
    const int32_t slot = op.kparams[0];
    const auto    it   = input_ints_.find(idx.addr);
    if (it == input_ints_.end()) {
        const val v = dev_mem_.get(idx.addr);
        fakedsp::violation("gdn-slot-from-input", "%s: the host gives slot %d, but the index at 0x%" PRIx64
                           " is %s, not an input of the graph: the host reads it at enqueue time, before the DSP writes it",
                           fakedsp::opcode_name(op.opcode), slot, idx.addr,
                           (v & CONC) ? "a concrete value" : "the output of an op of the graph");
        return;
    }
    if (it->second != slot) {
        fakedsp::violation("gdn-slot-from-input", "%s: the host gives slot %d, the harness wrote the index %d at 0x%" PRIx64
                           ": the host read bytes that the DSP wrote for a later tensor", fakedsp::opcode_name(op.opcode),
                           slot, it->second, idx.addr);
    }
}

void checker::dev_op(const fakedsp::op_record & op) {
    if (op.status != HTP_STATUS_OK) {
        return;
    }
    const fakedsp::tensor_ref & d0 = op.dst[0];
    std::vector<int64_t>        rows;
    switch (op.opcode) {
        case HTP_OP_FENCE:
        case HTP_OP_MDEV_GROUP:
            return;
        case HTP_OP_CPY:
        case HTP_OP_CPY_FENCE: {
            int64_t ne[4] = { d0.ne[0], d0.ne[1], d0.ne[2], d0.ne[3] };
            dev_write_all(d0, op_value(HTP_OP_CPY, ZERO_PARAMS, ne, d0.type, { dev_content(op.src[0]) }, false));
            return;
        }
        case HTP_OP_GET_ROWS: {
            const auto & src = op.src[0];
            if (dev_ints(op.src[1], rows) && src.ne[2] == 1 && src.ne[3] == 1 &&
                std::all_of(rows.begin(), rows.end(), [&](int64_t r) { return r >= 0 && r < (int64_t) src.ne[1]; })) {
                for (size_t j = 0; j < rows.size(); j++) {
                    fakedsp::tensor_ref row = d0;
                    row.addr += (uint64_t) j * d0.nb[1];
                    row.ne[1] = row.ne[2] = row.ne[3] = 1;
                    dev_write_all(row, mix(T_ROW, dev_row_content(src, rows[j])));
                }
            } else {
                dev_write_all(d0, mix(mix(T_ALL, dev_content(src)), dev_content(op.src[1])));
            }
            return;
        }
        case HTP_OP_SET_ROWS: {
            const auto & b = op.src[0];
            if (dev_ints(op.src[1], rows) && b.ne[2] == 1 && b.ne[3] == 1 &&
                std::all_of(rows.begin(), rows.end(), [&](int64_t r) { return r >= 0 && r < (int64_t) d0.ne[1]; })) {
                for (size_t j = 0; j < rows.size(); j++) {
                    fakedsp::tensor_ref row = d0;
                    row.addr += (uint64_t) rows[j] * d0.nb[1];
                    row.ne[1] = row.ne[2] = row.ne[3] = 1;
                    dev_write_all(row, mix(T_SETROW, dev_row_content(b, (int64_t) j)));
                }
            } else {
                dev_write_all(d0, mix(mix(T_ALL + 100, dev_content(b)), dev_content(op.src[1])));
            }
            return;
        }
        case HTP_OP_RMS_NORM_MUL: {
            const auto & x = op.src[0];
            int64_t      xne[4] = { x.ne[0], x.ne[1], x.ne[2], x.ne[3] };
            int64_t      dne[4] = { d0.ne[0], d0.ne[1], d0.ne[2], d0.ne[3] };
            const val    v_rms  = op_value(HTP_OP_RMS_NORM, op.params, xne, x.type, { dev_content(x) }, false);
            dev_write_all(d0, op_value(HTP_OP_MUL, ZERO_PARAMS, dne, d0.type,
                                       { const_content(v_rms, nelements(xne)), dev_content(op.src[1]) }, true));
            return;
        }
        case HTP_OP_MUL_MAT_ADD: {
            const auto & wt  = op.src[0];
            const auto & x   = op.src[1];
            auto         it  = weights_.find(wt.addr);
            const int64_t n1 = it != weights_.end() ? it->second->ne[1] : wt.ne[1];
            int64_t      mne[4] = { n1, x.ne[1], x.ne[2], x.ne[3] };
            int64_t      dne[4] = { d0.ne[0], d0.ne[1], d0.ne[2], d0.ne[3] };
            const val    v_mm   = op_value(HTP_OP_MUL_MAT, op.params, mne, GGML_TYPE_F32, { dev_content(wt), dev_content(x) }, false);
            dev_write_all(d0, op_value(HTP_OP_ADD, ZERO_PARAMS, dne, d0.type,
                                       { const_content(v_mm, nelements(mne)), dev_content(op.src[2]) }, true));
            return;
        }
        case HTP_OP_MUL_MAT_NX:
        case HTP_OP_MUL_MAT_ID_NX: {
            const bool     id = op.opcode == HTP_OP_MUL_MAT_ID_NX;
            const uint32_t nw = (uint32_t) ((const htp_mm_kernel_params *) op.kparams)->n_weights;
            for (uint32_t p = 0; p < nw && p < 4; p++) {
                const auto & d = op.dst[p];
                int64_t      dne[4] = { d.ne[0], d.ne[1], d.ne[2], d.ne[3] };
                std::vector<val> c = { dev_content(op.src[p]), dev_content(op.src[nw]) };
                if (id) {
                    c.push_back(dev_content(op.src[nw + 1]));
                }
                dev_write_all(d, op_value(id ? HTP_OP_MUL_MAT_ID : HTP_OP_MUL_MAT, op.params, dne, d.type, c, false));
            }
            return;
        }
        case HTP_OP_GDN_CONV_STEP:
        case HTP_OP_GDN_CONV_CHUNK: {
            check_slot(op, op.src[1]);
            const auto &  states = op.src[0];
            const auto &  xt     = op.src[2];
            const auto &  wc     = op.src[3];
            const auto &  slots  = op.src[4];
            const int64_t n_ch   = wc.ne[1];
            const int64_t dconv  = wc.ne[0];
            const int64_t T      = xt.ne[0];
            const int64_t row    = (dconv - 1) * n_ch;
            const val     v_r    = mix(T_ROW, dev_row_content(states, op.kparams[0]));
            int64_t       cine[4] = { dconv - 1 + T, n_ch, 1, 1 };
            const val     v_ci   = op_value(HTP_OP_CONCAT, ZERO_PARAMS, cine, GGML_TYPE_F32,
                                            { const_content(v_r, (uint64_t) row), dev_content(xt) }, false);
            int64_t       cne[4] = { n_ch, T, 1, 1 };
            const val     v_c    = op_value(HTP_OP_SSM_CONV, ZERO_PARAMS, cne, GGML_TYPE_F32,
                                            { const_content(v_ci, nelements(cine)), dev_content(wc) }, false);
            const val     v_s    = op_value(HTP_OP_UNARY_SILU, op.params, cne, GGML_TYPE_F32,
                                            { const_content(v_c, nelements(cne)) }, false);
            dev_write_all(d0, v_s);
            int64_t   sne[4] = { row, 1, 1, 1 };
            const val v_p    = op_value(HTP_OP_CPY, ZERO_PARAMS, sne, GGML_TYPE_F32, { const_content(v_ci, (uint64_t) row) }, false);
            const uint32_t n_slots = slots.ne[1] > 0 ? slots.ne[1] : 1;
            for (uint32_t g = 0; g < n_slots; g++) {
                fakedsp::tensor_ref r = slots;
                r.addr += (uint64_t) g * slots.nb[1];
                r.ne[1] = r.ne[2] = r.ne[3] = 1;
                dev_write_all(r, v_p);
            }
            return;
        }
        case HTP_OP_GDN_STATE_STEP: {
            check_slot(op, op.src[6]);
            const auto &  v      = op.src[2];
            const auto &  states = op.src[5];
            const auto &  slot   = op.src[7];
            const int64_t S_v    = v.ne[0];
            const int64_t H      = v.ne[1];
            const int64_t D      = S_v * S_v * H;
            const val     v_r    = mix(T_ROW, dev_row_content(states, op.kparams[0]));
            int64_t       gne[4] = { d0.ne[0], d0.ne[1], d0.ne[2], d0.ne[3] };
            const val     v_g    = op_value(HTP_OP_GATED_DELTA_NET, op.params, gne, GGML_TYPE_F32,
                                            { dev_content(op.src[0]), dev_content(op.src[1]), dev_content(op.src[2]),
                                              dev_content(op.src[3]), dev_content(op.src[4]), const_content(v_r, (uint64_t) D) },
                                            false);
            dev_write_all(d0, v_g, 0);
            int64_t   sne[4] = { slot.ne[0], slot.ne[1], slot.ne[2], slot.ne[3] };
            dev_write_all(slot, op_value(HTP_OP_CPY, ZERO_PARAMS, sne, GGML_TYPE_F32, { const_content(v_g, (uint64_t) D) }, false));
            return;
        }
        default: {
            if (!d0.present) {
                return;
            }
            std::vector<val> c;
            for (int i = 0; i < 10; i++) {
                if (op.src[i].present) {
                    c.push_back(dev_content(op.src[i]));
                }
            }
            int64_t dne[4] = { d0.ne[0], d0.ne[1], d0.ne[2], d0.ne[3] };
            dev_write_all(d0, op_value(op.opcode, op.params, dne, d0.type, c, is_commutative(op.opcode)));
            return;
        }
    }
}

void checker::run_device(const std::vector<fakedsp::batch_record> & batches) {
    last_ops_.clear();
    for (const auto & b : batches) {
        for (const auto & op : b.ops) {
            dev_op(op);
            last_ops_.push_back(op);
        }
    }
}

void checker::forget(const graphgen::graph_spec & g) {
    input_ints_.clear();  // the next set_input of each graph writes its indices again
    for (ggml_tensor * n : g.order) {
        ref_store_.erase(n);
    }
    if (g.ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(g.ctx); t; t = ggml_get_next_tensor(g.ctx, t)) {
            ref_store_.erase(t);
        }
    }
}

void checker::compare(const std::vector<const graphgen::graph_spec *> & ran, uint32_t step, const std::string & context) {
    auto report = [&](const char * what, const ggml_tensor * t, uint64_t addr, val rv, val dv) {
        std::string ops;
        for (size_t i = last_ops_.size(); i-- > 0;) {
            const auto & op = last_ops_[i];
            for (int d = 0; d < 4; d++) {
                const auto & dt = op.dst[d];
                if (dt.present && addr >= dt.addr && addr < dt.addr + fakedsp::tensor_extent(dt)) {
                    char buf[160];
                    snprintf(buf, sizeof(buf), " op #%zu %s (dst %d)", i, fakedsp::opcode_name(op.opcode), d);
                    ops += buf;
                    i = 0;
                    break;
                }
            }
        }
        fakedsp::violation("sym-mismatch", "%s step %u: the device result differs from the graph at %s %s (address 0x%" PRIx64
                           "): graph 0x%016" PRIx64 " device 0x%016" PRIx64 "; the last device writer:%s",
                           context.c_str(), step, what, t->name, addr, rv, dv, ops.empty() ? " none" : ops.c_str());
    };

    // the caches
    for (ggml_tensor * c : w_.caches) {
        const uint64_t base = (uint64_t) (uintptr_t) c->data;
        const size_t   n    = ggml_nbytes(c);
        for (size_t i = 0; i < n; i++) {
            const val rv = ref_mem_.get(base + i);
            const val dv = dev_mem_.get(base + i);
            if (rv != dv) {
                report("the cache", c, base + i, rv, dv);
                return;
            }
        }
    }
    // the outputs
    for (const auto * g : ran) {
        for (ggml_tensor * o : g->outputs) {
            size_t              base = 0;
            const ggml_tensor * r    = root_of(o, &base);
            const shape         s    = shape_of(o);
            bool                bad  = false;
            for_each_elem(s.type, s.ne, s.nb, [&](uint64_t off, size_t, int64_t, int64_t, int64_t) {
                if (bad) {
                    return;
                }
                const uint64_t addr = (uint64_t) (uintptr_t) o->data + off;
                const val      rv   = ref_get(r, base + off);
                const val      dv   = dev_mem_.get(addr);
                if (rv != dv) {
                    bad = true;
                    report("the output", o, addr, rv, dv);
                }
            });
            if (bad) {
                return;
            }
        }
    }
}

} // namespace symexec
