// The runner: it allocates the tensors of a case on one backend with guard regions around each
// tensor, uploads the inputs, computes the graph, reads the outputs, and checks the guard regions
// and the inputs after the run.

#pragma once

#include "case.h"

#include "ggml-backend.h"

#include <string>
#include <vector>

namespace fo {

// The status of one run.
enum run_status : uint32_t {
    RUN_OK          = 0,
    RUN_UNSUPPORTED = 1, // the backend does not support a node of the graph
    RUN_ALLOC       = 2, // a buffer or a tensor allocation failed
    RUN_COMPUTE     = 3, // graph_compute gave an error status
    RUN_CRASHED     = 4, // the process stopped during the case (the replay driver sets it)
    RUN_INVALID     = 5, // the bytes give no valid case
};

// Defects that the run found in addition to the status.
enum run_flag : uint32_t {
    FLAG_GUARD  = 1, // a byte of a guard region changed: a write outside a tensor
    FLAG_INPUT  = 2, // an input tensor changed: a write into a source
    FLAG_NONDET = 4, // a repeat of the same case on the same backend gave different output bytes
};

// The result of one run.
struct run_result {
    uint32_t                          status = RUN_OK;
    uint32_t                          flags  = 0;
    std::string                       detail;
    std::vector<std::vector<uint8_t>> outs;   // the raw bytes of each output of the case
    double                            ms = 0.0;
    uint64_t                          input_hash = 0; // case_hash() of the decoded case
};

// Return the FNV-1a hash of the input bytes and of the text of a decoded case. Two processes that
// decode the same case bytes must give the same hash: a different hash is a defect of the decoder
// (for example a C++ evaluation order that differs between compilers), not a finding of an op.
uint64_t case_hash(const built_case & c);

// One backend and its buffer types.
struct backend_ctx {
    ggml_backend_t             backend     = nullptr;
    ggml_backend_dev_t         dev         = nullptr;
    ggml_backend_buffer_type_t buft        = nullptr;
    ggml_backend_buffer_type_t repack_buft = nullptr; // the extra buffer type of the CPU, or nullptr
    bool                       is_cpu      = false;
    bool                       use_ref     = false;   // the oracle: the reference paths of the CPU backend
    int                        force_threads = 0;     // > 0 replaces the thread count of the case
    std::string                name;
};

// Open the backend of a device ("CPU", "HTP0"). Return false and a reason when the device is missing.
bool open_backend(const std::string & dev_name, backend_ctx & be, std::string & why);

// Free the backend.
void close_backend(backend_ctx & be);

// Run the case on the backend. The case must be freshly decoded: the run allocates its tensors.
run_result run_case(built_case & c, backend_ctx & be);

// Return the name of a status.
const char * run_status_name(uint32_t s);

} // namespace fo
