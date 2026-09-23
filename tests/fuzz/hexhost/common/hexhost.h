// The interface of the hexhost fuzz harness to the host part of the Hexagon
// backend. hexhost_tu.cpp includes ggml-hexagon.cpp into one translation unit,
// thus these functions reach the static options and functions of the backend.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hexhost {

// The switches of ggml_hexagon_init (the GGML_HEXAGON_* environment variables)
struct options {
    int    opbatch        = 1280;
    int    opqueue        = 32;
    int    oppoll         = 0;
    int    opfusion       = 1;
    int    opfusion_state = -1;
    int    multirow       = 1;
    int    gdn_chunk      = 1;
    int    profile        = 0;
    int    nhmx           = 1;
    int    mm_select      = 3;
    int    fa_select      = 2;
    int    ar_select      = 2;
    int    batchcache     = 1;
    int    graphcache     = 8;
    int    hostprof       = 0;
    int    verbose        = 0;
    size_t nhvx           = 0;
    size_t vmem           = 3355443200u;
    size_t mbuf           = 1ul << 30;
    int    arch           = 79;
};

// Writes the options into the static option variables of the backend.
void set_options(const options & o);

// Gives the options that the backend uses at this time.
options get_options();

// A device of the backend (a ggml_backend_device with its device context)
struct device;

// Creates a device. The session opens at the first use.
device * device_new();

// Releases a device and its session. Release every buffer and backend of the device first.
void device_free(device * d);

ggml_backend_dev_t          device_dev(device * d);
ggml_backend_buffer_type_t  device_buft(device * d);

// Opens the session of the device and gives true when it opened.
bool device_open(device * d);

// The hardware values that the session read from the fake DSP
uint32_t session_n_threads(device * d);
uint32_t session_n_hmx(device * d);
uint64_t session_vtcm(device * d);

// Gives the answer of supports_op of the device.
bool supports_op(device * d, const ggml_tensor * op);

// A tensor descriptor and an op as add_op of the host packs them
struct packed_tensor {
    bool     present = false;
    uint64_t addr    = 0;
    uint32_t size    = 0;
    uint32_t type    = 0;
    uint32_t flags   = 0;
    uint32_t ne[4]   = {0, 0, 0, 0};
    uint32_t nb[4]   = {0, 0, 0, 0};
};

struct packed_op {
    uint32_t      opcode = 0;
    int32_t       params[16]  = {0};
    int32_t       kparams[32] = {0};
    packed_tensor src[10];
    packed_tensor dst[4];
};

// Packs a sequence of graph nodes with the op batch of the session, as
// enqueue_op does, with the fusion of the host. The nodes must be in a buffer
// of the device or in a fake buffer (refer to fake_buffer_new). The batch is
// not sent. Gives the packed ops.
std::vector<packed_op> pack_nodes(device * d, const std::vector<ggml_tensor *> & nodes);

// Tags the fusable tensors of a graph as graph_compute does, and packs the
// compute nodes of the graph with pack_nodes.
std::vector<packed_op> pack_graph(device * d, ggml_cgraph * graph);

// Creates a buffer of the device that holds no memory for its tensors: the
// data pointers are fake addresses with the alignment that the caller gives.
// Such a buffer is only for pack_nodes; never send its ops.
ggml_backend_buffer_t fake_buffer_new(device * d, size_t size);

// Gives each tensor of the list an extra of the backend (init_tensor), with
// the weight and the repack flags when weight is true.
void init_tensor(ggml_backend_buffer_t buf, ggml_tensor * t, bool weight);

// Runs ggml_hexagon_init with the environment of this process, and releases
// the registry again. Gives the device count.
size_t init_from_env();

// Parses a GGML_HEXAGON_PROFILE value with str_to_vec of the backend. Gives the
// item count and the first item, or -1 when the parser throws.
int profile_items(const char * value, uint32_t * first);

// Gives the size of the string that vec_to_str of the backend makes from an empty list: 0 when the
// backend handles an empty list, else the size after a pop_back of an empty string.
size_t profile_empty_size();

// The state that the last ggml_hexagon_init left: the largest device group (the physical devices of
// one device) and the trace size opt_optrace.
size_t max_device_group();
int    optrace();

// Gives the graph cache size and the batch cache state of the session, for the
// checks of the graph fuzzer.
size_t graph_cache_size(device * d);

// Gives the number of ops in the op batch that the session did not send yet.
uint32_t pending_ops(device * d);

// Gives the HTP opcode of a graph node (op_remap_to_htp). The node must be an op that the backend supports.
int htp_opcode(const ggml_tensor * node);

// Gives the host-side counters of the session (LLAMA_HOSTPROF must be on): the
// graph cache hits, the batch replays and the verified replays since the last print.
void session_counters(device * d, uint32_t * hits, uint32_t * replays, uint32_t * verified);

} // namespace hexhost
