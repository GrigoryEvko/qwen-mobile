// The symbolic check of the hexhost graph fuzzer.
//
// Each element of memory holds a 64-bit symbolic value. An op writes to each
// element of its output one value: a hash of the op, its params, its output
// shape, and the contents of its inputs (the values of their elements in
// logical order). Two executions that give the same symbolic values at the
// observable elements (the caches and the graph outputs) compute the same
// function. The check runs two executions of each step:
//
//   - the reference: the nodes of the graph in the order that ggml built, each
//     compute tensor in its own storage (no reuse of memory), the caches in
//     memory. This is the meaning of the graph.
//   - the device: the ops that the fake DSP recorded, in the order that it ran
//     them, on the real addresses of the descriptors. A fused op has the value
//     of the chain that it replaces.
//
// A difference at an observable element is a violation: the host changed the
// result by a fusion, a reorder, a batch replay, a wrong slot index, or a
// write to a tensor that is still alive.
//
// A concrete int32 (an index that the harness set) has the top bit set, thus
// GET_ROWS and SET_ROWS of the two executions select the same rows.
#pragma once

#include "fake_dsp.h"
#include "graphgen.h"

#include "ggml.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace symexec {

using val = uint64_t;

// The memory of one execution: a symbolic value for each byte of each rpcmem
// allocation. A byte that no write touched has the value init(address). The key
// of a block is the fd of its allocation, because a new allocation can get the
// base address of a released one with a different size.
struct memory {
    std::unordered_map<int, std::vector<val>> blocks;         // allocation fd -> byte values
    uint64_t                                  last_base = 0;
    uint64_t                                  last_size = 0;
    uint64_t                                  last_gen  = 0;  // fakedsp::alloc_generation of the last lookup
    std::vector<val> *                        last_vec  = nullptr;

    val  get(uint64_t addr);
    void set(uint64_t addr, size_t n, val v);
};

// The checker of one fuzz input
class checker {
  public:
    explicit checker(graphgen::world & w);

    // Writes the values of an input tensor into the two executions. The ints
    // are concrete when ints is not empty; else each element gets a new value.
    void set_input(ggml_tensor * t, const std::vector<int32_t> & ints, uint32_t step);

    // Runs the reference over the nodes of a graph (the order that ggml built).
    void run_reference(const graphgen::graph_spec & g);

    // Runs the device over the recorded batches.
    void run_device(const std::vector<fakedsp::batch_record> & batches);

    // Compares the caches and the outputs of the graphs that ran in this step.
    // Reports a violation for the first difference.
    void compare(const std::vector<const graphgen::graph_spec *> & ran, uint32_t step, const std::string & context);

    // Forgets the reference storage of the tensors of a graph that the harness frees.
    void forget(const graphgen::graph_spec & g);

  private:
    graphgen::world & w_;
    memory            ref_mem_;     // the caches of the reference
    memory            dev_mem_;     // all memory of the device
    std::unordered_map<const ggml_tensor *, std::vector<val>> ref_store_;  // the compute tensors of the reference
    std::unordered_map<uint64_t, const ggml_tensor *>         weights_;    // weight address -> tensor
    std::vector<fakedsp::op_record>                           last_ops_;   // the device ops of this step
    std::unordered_map<uint64_t, int32_t>                     input_ints_; // element address -> index that the harness wrote

    bool is_persistent(const ggml_tensor * root) const;
    val  ref_get(const ggml_tensor * root, size_t off);
    void ref_set(const ggml_tensor * root, size_t off, size_t n, val v);
    val  ref_content(const ggml_tensor * t);
    val  ref_row_content(const ggml_tensor * t, int64_t row);
    void ref_write_all(const ggml_tensor * t, val v);
    void ref_write_row(const ggml_tensor * t, int64_t row, val v);
    bool ref_ints(const ggml_tensor * t, std::vector<int64_t> & out);

    void check_slot(const fakedsp::op_record & op, const fakedsp::tensor_ref & idx);
    val  dev_content(const fakedsp::tensor_ref & t);
    val  dev_row_content(const fakedsp::tensor_ref & t, int64_t row);
    void dev_write_all(const fakedsp::tensor_ref & t, val v, int64_t only_row = -1);
    bool dev_ints(const fakedsp::tensor_ref & t, std::vector<int64_t> & out);
    void dev_op(const fakedsp::op_record & op);
};

} // namespace symexec
