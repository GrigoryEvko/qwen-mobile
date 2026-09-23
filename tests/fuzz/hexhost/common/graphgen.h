// The graph generator of the hexhost graph fuzzer.
//
// It builds small graphs with the structure of Qwen3.5: the linear attention
// layer (the conv chain of build_conv_state, the state read of build_rs, the
// gated delta net of build_recurrent_attn with and without rollback slots), the
// full attention layer (the KV cache writes with SET_ROWS and FLASH_ATTN_EXT),
// the FFN, the MTP head, and random mutations of these chains. The weights and
// the caches go into buffers of the Hexagon device, and ggml-alloc places the
// nodes of each graph in a compute buffer of the device, as the scheduler of
// llama.cpp does.
#pragma once

#include "fake_dsp.h"
#include "hexhost.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <fuzzer/FuzzedDataProvider.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace graphgen {

// The kind of an input tensor that the harness sets before each step
enum input_kind {
    INPUT_FLOAT,      // random values
    INPUT_SLOT,       // the s_copy slot indices of the recurrent cache
    INPUT_POS,        // the rope positions
    INPUT_ROWS,       // the rows of the KV cache that SET_ROWS writes
    INPUT_MASK,       // the attention mask (F16)
    INPUT_IDS,        // row ids of GET_ROWS
};

struct input_spec {
    ggml_tensor * t      = nullptr;
    input_kind    kind   = INPUT_FLOAT;
    int32_t       limit  = 0;   // the valid range [0, limit) of an index input
};

// One graph with its node order and its compute buffer
struct graph_spec {
    ggml_context *              ctx    = nullptr;   // the nodes and the inputs of the graph
    ggml_cgraph *               gf     = nullptr;
    ggml_gallocr_t              galloc = nullptr;
    ggml_backend_buffer_t       buf    = nullptr;   // the buffer of the nodes when world::no_reuse is true
    std::vector<ggml_tensor *>  order;               // the node order that ggml built (before graph_optimize)
    std::vector<input_spec>     inputs;
    std::vector<ggml_tensor *>  outputs;
    std::vector<int>            cuts;                // the split points: split k is [cuts[k], cuts[k + 1])
    std::vector<uint64_t>       uids;                // the uid of each split
    bool                        optimized = false;
    int64_t                     rs_head   = 0;       // the first row of the recurrent cache that the graph writes
    std::string                 desc;                // a short description for the reports
};

// The mutations and the kind of one layer
struct layer_params {
    int  kind          = 0;      // 0 = linear attention, 1 = full attention, 2 = FFN only
    int  d_conv        = 4;      // the conv width (the DSP kernel takes 4)
    bool zero_state    = false;  // the state_zero SCALE of build_rs writes a row (rs_zero >= 0)
    bool extra_rows    = false;  // n_rs > n_seqs: the GET_ROWS and the CPY of the extra states
    int  reader        = -1;     // an extra reader of an intermediate: 0 R, 1 CONCAT, 2 SSM_CONV, 3 G tail, 4 G head, 5 R of the state
    int  out_flag      = -1;     // FLAG_OUTPUT on an intermediate, the same codes
    bool computed_idx  = false;  // s_copy comes from a GET_ROWS of an I32 table, not from an input
    int  slot_mode     = 0;      // 0 the llama.cpp slots, 1 two CPYs to one slot, 2 overlapped slots
    int  idx_offset    = 0;      // the element of s_copy that the chains read
    bool swiglu        = true;   // the FFN uses SWIGLU, else SILU and MUL
    bool view_add      = false;  // the residual ADD reads the down projection through a view
    bool view_reader   = false;  // the view of view_add has one more reader
    bool keep_qkv      = true;   // the projection has one more reader, thus its bytes stay allocated
    int  wtype[8]      = {0};    // the weight types of the projections
};

// The dims of the model and the layers
struct model_params {
    int64_t n_embd  = 64;
    int64_t n_ff    = 128;
    int64_t S_v     = 16;
    int64_t H_k     = 1;
    int64_t H_v     = 2;
    int64_t mem     = 1;      // the cells of the recurrent cache
    int64_t K       = 1;      // the rollback slots (n_rs_seq + 1)
    int64_t hd      = 32;     // the head size of the full attention
    int64_t n_head  = 2;
    int64_t n_kv    = 1;
    int64_t kv_size = 32;
    int64_t vocab   = 256;
    int     n_layers = 1;
    layer_params layers[4];
    bool    mtp     = false;  // an MTP head: CONCAT and a projection
    bool    out_ids = false;  // a GET_ROWS of the output rows before the head
    int     head_wtype = 0;
};

// The weights and the caches of one layer
struct layer_tensors {
    ggml_tensor * attn_norm = nullptr;
    // linear attention
    ggml_tensor * wqkv = nullptr, * wz = nullptr, * wbeta = nullptr, * walpha = nullptr;
    ggml_tensor * dt = nullptr, * ssm_a = nullptr, * conv = nullptr, * ssm_norm = nullptr, * ssm_out = nullptr;
    ggml_tensor * conv_cache = nullptr, * ssm_cache = nullptr, * idx_table = nullptr;
    // full attention
    ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr, * q_norm = nullptr, * k_norm = nullptr;
    ggml_tensor * k_cache = nullptr, * v_cache = nullptr;
    // FFN
    ggml_tensor * ffn_norm = nullptr, * up = nullptr, * gate = nullptr, * down = nullptr;
};

// The weights, the caches and the graphs of one fuzz input
struct world {
    model_params                       mp;
    std::vector<layer_tensors>         lt;
    ggml_tensor *                      out_norm = nullptr;
    ggml_tensor *                      w_out    = nullptr;
    ggml_tensor *                      mtp_proj = nullptr;
    ggml_tensor *                      mtp_norm = nullptr;
    hexhost::device *                  dev     = nullptr;
    ggml_backend_t                     backend = nullptr;
    ggml_context *                     ctx_w   = nullptr;   // the weights
    ggml_context *                     ctx_c   = nullptr;   // the caches
    ggml_backend_buffer_t              buf_w   = nullptr;
    ggml_backend_buffer_t              buf_c   = nullptr;
    std::vector<ggml_tensor *>         weights;
    std::vector<ggml_tensor *>         caches;
    std::vector<graph_spec>            graphs;
    // true: the weights hold values for a numeric compare (the phone driver). The input bytes
    // that the generator reads do not change, thus one input gives the same graph in the two modes.
    bool                               numeric = false;
    // true: each node of a graph gets its own bytes (no ggml-alloc reuse), and the bytes start
    // at zero. The phone driver then reads each node after a step. This changes the addresses
    // but not the nodes, the node order or the fusions.
    bool                               no_reuse = false;
};

// Reads the session of one graph input: the hardware of the fake DSP and the switches of the
// host. async is the build of the fake DSP (a DSP thread for each queue). The phone driver reads
// the same bytes with async = false and ignores the values, thus one input gives the same graph
// on x86 and on the phone.
void decode_session(FuzzedDataProvider & fdp, bool async, fakedsp::config & cfg, hexhost::options & o);

// Builds the weights, the caches and one or two graphs from the fuzz input.
// Gives false when the input does not give a graph that the device runs fully.
bool build_world(FuzzedDataProvider & fdp, hexhost::device * dev, world & w);

// The hooks of run_steps. Each hook can be empty.
struct step_hooks {
    // An input value: the index values (I32 index inputs and positions), or empty for other inputs
    std::function<void(ggml_tensor *, const std::vector<int32_t> &, uint32_t)> on_input;
    // Before the rebuild of a graph
    std::function<void(const graph_spec &)> on_forget;
    // Before the compute of a step: true when each slot index of the step is valid
    std::function<void(bool)> on_slots;
    // After the compute of each split: the split and the status of graph_compute
    std::function<void(const graph_spec &, size_t, ggml_status)> on_split;
    // After the synchronize of a step, with true when each slot index of the step was valid. A
    // false result stops the steps of the input.
    std::function<bool(graph_spec &, uint32_t, bool)> after_step;
};

// Runs the steps of one input on a world (2 to 6 steps): for each step it picks a graph,
// sometimes builds it again, sets its inputs, computes its splits with events between them,
// sometimes reads an output through the async path, and synchronizes.
void run_steps(FuzzedDataProvider & fdp, world & w, const step_hooks & h);

// Releases the graphs, the buffers and the contexts of a world (not the device).
void free_world(world & w);

// Builds the graph of index gi again: a new context, new nodes and a new uid, as
// llama.cpp does when the batch shape changes. Gives false on failure.
bool rebuild_graph(FuzzedDataProvider & fdp, world & w, size_t gi);

} // namespace graphgen
