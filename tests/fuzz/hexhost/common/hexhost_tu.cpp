// The translation unit of the host part of the Hexagon backend for the hexhost
// fuzz harness. It includes ggml-hexagon.cpp as it is (the stub SDK headers of
// tests/fuzz/hexhost/stubs replace the Hexagon SDK), and gives the functions of
// hexhost.h, which reach the static options and functions of the backend.

#include "ggml-hexagon.cpp"

#include "hexhost.h"

namespace hexhost {

// A device of the backend: the ggml device and its device context
struct device {
    ggml_backend_device                   dev {};
    ggml_backend_hexagon_device_context * ctx = nullptr;
};

void set_options(const options & o) {
    opt_opbatch        = o.opbatch;
    opt_opqueue        = o.opqueue;
    opt_oppoll         = o.oppoll;
    opt_opfusion       = o.opfusion;
    opt_opfusion_state = o.opfusion_state;
    opt_multirow       = o.multirow;
    opt_gdn_chunk      = o.gdn_chunk;
    opt_profile        = o.profile;
    opt_nhmx           = o.nhmx;
    opt_mm_select      = o.mm_select;
    opt_fa_select      = o.fa_select;
    opt_ar_select      = o.ar_select;
    opt_batchcache     = o.batchcache;
    opt_graphcache     = o.graphcache;
    opt_hostprof       = o.hostprof;
    opt_verbose        = o.verbose;
    opt_nhvx           = o.nhvx;
    opt_vmem           = o.vmem;
    opt_mbuf           = o.mbuf;
    opt_arch           = o.arch;
    opt_optrace        = o.opbatch * 4;
}

options get_options() {
    options o;
    o.opbatch        = opt_opbatch;
    o.opqueue        = opt_opqueue;
    o.oppoll         = opt_oppoll;
    o.opfusion       = opt_opfusion;
    o.opfusion_state = opt_opfusion_state;
    o.multirow       = opt_multirow;
    o.gdn_chunk      = opt_gdn_chunk;
    o.profile        = opt_profile;
    o.nhmx           = opt_nhmx;
    o.mm_select      = opt_mm_select;
    o.fa_select      = opt_fa_select;
    o.ar_select      = opt_ar_select;
    o.batchcache     = opt_batchcache;
    o.graphcache     = opt_graphcache;
    o.hostprof       = opt_hostprof;
    o.verbose        = opt_verbose;
    o.nhvx           = opt_nhvx;
    o.vmem           = opt_vmem;
    o.mbuf           = opt_mbuf;
    o.arch           = opt_arch;
    return o;
}

device * device_new() {
    auto * d = new device();
    ggml_hexagon_device_config cfg;
    cfg.physical_idx = 0;
    cfg.virtual_idx  = 0;
    cfg.domain_id    = CDSP_DOMAIN_ID;
    cfg.domain_name  = CDSP_DOMAIN_NAME;
    cfg.name         = "HTP0";
    d->dev.iface     = ggml_backend_hexagon_device_i;
    d->dev.reg       = nullptr;
    d->ctx           = new ggml_backend_hexagon_device_context(0, cfg, &d->dev);
    d->dev.context   = d->ctx;
    return d;
}

void device_free(device * d) {
    if (!d) {
        return;
    }
    delete d->ctx;
    delete d;
}

ggml_backend_dev_t device_dev(device * d) {
    return &d->dev;
}

ggml_backend_buffer_type_t device_buft(device * d) {
    return &d->ctx->buffer_type;
}

bool device_open(device * d) {
    try {
        return d->ctx->session() != nullptr;
    } catch (const std::exception & e) {
        fprintf(stderr, "hexhost: the session did not open: %s\n", e.what());
        return false;
    }
}

uint32_t session_n_threads(device * d) {
    return d->ctx->session()->n_threads;
}

uint32_t session_n_hmx(device * d) {
    return d->ctx->session()->n_hmx;
}

uint64_t session_vtcm(device * d) {
    return d->ctx->session()->vtcm_size;
}

bool supports_op(device * d, const ggml_tensor * op) {
    return ggml_backend_hexagon_device_supports_op(&d->dev, op);
}

// Builds the op node of one graph node with the kernel params, as the make_node
// lambda of ggml_backend_hexagon_graph_compute does.
static htp_opnode make_node(ggml_hexagon_session * sess, ggml_tensor * n) {
    htp_opnode node(HTP_OP_INVALID, n);
    node.opcode = op_remap_to_htp(n);
    if (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID) {
        ggml_hexagon_precompute_matmul_params(sess, node.node->src[0], node.node->src[1], node.node,
                                              (struct htp_mm_kernel_params *) node.kernel_params);
    } else if (node.opcode == HTP_OP_FLASH_ATTN_EXT) {
        ggml_hexagon_precompute_flash_attn_params(sess, node.node, (struct htp_fa_kernel_params *) node.kernel_params);
    } else if (htp_op_is_unary(node.opcode)) {
        auto                       inputs = node.get_inputs();
        const struct ggml_tensor * src0   = inputs[0];
        const struct ggml_tensor * src1   = inputs.size() > 1 ? inputs[1] : nullptr;
        ggml_hexagon_precompute_unary_params(sess, node.opcode, src0, src1, node.dst(),
                                             (struct htp_unary_kernel_params *) node.kernel_params);
    } else if (node.opcode == HTP_OP_GET_ROWS) {
        ggml_hexagon_precompute_get_rows_params(sess, node.node->src[0], node.node->src[1], node.dst(),
                                                (struct htp_get_rows_kernel_params *) node.kernel_params);
    } else if (node.opcode == HTP_OP_SET_ROWS) {
        ggml_hexagon_precompute_set_rows_params(sess, node.node->src[0], node.node->src[1], node.dst(),
                                                (struct htp_set_rows_kernel_params *) node.kernel_params);
    } else if (node.opcode == HTP_OP_ROPE) {
        ggml_hexagon_precompute_rope_params(sess, node.node, (struct htp_rope_kernel_params *) node.kernel_params);
    } else if (node.opcode == HTP_OP_GATED_DELTA_NET) {
        node.kernel_params[0] = opt_gdn_chunk ? 0 : 1;
    }
    return node;
}

std::vector<packed_op> pack_nodes(device * d, const std::vector<ggml_tensor *> & nodes) {
    ggml_hexagon_session * sess = d->ctx->session();
    ggml_hexagon_opbatch   b(sess, 64, SIZE_MAX / 4);

    for (ggml_tensor * n : nodes) {
        htp_opnode node = make_node(sess, n);
        if (opt_opfusion && b.try_fuse(node)) {
            continue;
        }
        if (!b.fit_op(node)) {
            break;
        }
        b.add_op(node);
    }

    std::vector<packed_op> out(b.n_ops);
    for (uint32_t i = 0; i < b.n_ops; i++) {
        const htp_op_desc & o = b.h_ops[i];
        packed_op &         p = out[i];
        p.opcode              = o.opcode;
        memcpy(p.params, o.params, sizeof(p.params));
        memcpy(p.kparams, o.kernel_params, sizeof(p.kparams));
        auto fill = [&](packed_tensor & t, uint16_t idx) {
            if (idx == 0xffff) {
                return;
            }
            const htp_tensor & h = b.h_tens[idx];
            t.present            = true;
            t.addr               = b.h_bufs[h.bi].base + h.data;
            t.size               = h.size;
            t.type               = h.type;
            t.flags              = h.flags;
            memcpy(t.ne, h.ne, sizeof(t.ne));
            memcpy(t.nb, h.nb, sizeof(t.nb));
        };
        for (int s = 0; s < HTP_OP_MAX_INPUTS; s++) {
            fill(p.src[s], o.src[s]);
        }
        for (int k = 0; k < HTP_OP_MAX_OUTPUTS; k++) {
            fill(p.dst[k], o.dst[k]);
        }
    }
    return out;
}

std::vector<packed_op> pack_graph(device * d, ggml_cgraph * graph) {
    // The tags of the fusable tensors, as ggml_backend_hexagon_graph_compute sets them
    for (int i = 0; i < graph->n_nodes; i++) {
        auto * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
        if (!extra) {
            continue;
        }
        extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;
        if (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
        } else if (graph->nodes[i]->op == GGML_OP_MUL_MAT || graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
            if ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD &&
                 ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                ggml_node_has_n_uses(graph, i, 1)) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            }
        }
    }
    std::vector<ggml_tensor *> nodes;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (op_is_compute(graph->nodes[i])) {
            nodes.push_back(graph->nodes[i]);
        }
    }
    return pack_nodes(d, nodes);
}

ggml_backend_buffer_t fake_buffer_new(device * d, size_t size) {
    return ggml_backend_buft_alloc_buffer(&d->ctx->buffer_type, size);
}

void init_tensor(ggml_backend_buffer_t buf, ggml_tensor * t, bool weight) {
    t->buffer = buf;
    ggml_backend_hexagon_buffer_init_tensor(buf, t);
    auto * extra = (ggml_hexagon_tensor_extra *) t->extra;
    if (weight) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        if (ggml_hexagon_is_repack_type(t->type)) {
            extra->flags |= GGML_HEXAGON_TENSOR_REPACK;
        }
    }
}

size_t init_from_env() {
    delete opt_opfilter;
    opt_opfilter = nullptr;
    ggml_backend_reg reg = { GGML_BACKEND_API_VERSION, ggml_backend_hexagon_reg_i, nullptr };
    ggml_hexagon_init(&reg);
    const size_t n = opt_ndev;
    delete static_cast<ggml_hexagon_registry *>(reg.context);
    delete opt_opfilter;
    opt_opfilter = nullptr;
    return n;
}

int profile_items(const char * value, uint32_t * first) {
    try {
        const auto v = str_to_vec<uint32_t>(value);
        *first       = v.empty() ? 0 : v[0];
        return (int) v.size();
    } catch (...) {
        return -1;
    }
}

size_t graph_cache_size(device * d) {
    return d->ctx->session()->graph_cache.size();
}

uint32_t pending_ops(device * d) {
    return d->ctx->session()->op_batch->n_ops;
}

int htp_opcode(const ggml_tensor * node) {
    return (int) op_remap_to_htp(node);
}

void session_counters(device * d, uint32_t * hits, uint32_t * replays, uint32_t * verified) {
    const auto & hp = d->ctx->session()->hp;
    *hits           = hp.n_hits;
    *replays        = hp.n_replays;
    *verified       = hp.n_verified;
}

} // namespace hexhost
