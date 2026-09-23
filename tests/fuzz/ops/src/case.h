// The fuzz case: the decode of the case bytes into a ggml graph with its input data.
//
// A case is a byte string. Byte 0 selects the kind (a one-op graph or a small multi-op chain of the
// Qwen3.5 path), bytes 1 to 8 give the seed of the values, and the kind builder reads its shapes
// and parameters from the bytes that follow. The decode is deterministic and platform independent,
// thus the libFuzzer process, the oracle process and the phone make the same graph and the same
// input bytes from the same case.

#pragma once

#include "common.h"

#include "ggml.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fo {

// The role of an input tensor. A WEIGHT goes into a buffer with the weights usage, thus the Hexagon
// backend repacks a quantized weight. A STATE is written in place by the graph (a cache table or a
// SET_ROWS destination), thus the input check does not compare it after the run.
enum class leaf_role : uint8_t { DATA, WEIGHT, STATE };

// One input tensor and its contents.
struct leaf {
    ggml_tensor *        t = nullptr;
    std::vector<uint8_t> bytes;  // ggml_nbytes(t) bytes in the layout of the tensor type
    leaf_role            role = leaf_role::DATA;
};

// One output tensor that the harness reads and compares.
struct output {
    ggml_tensor * t = nullptr;    // a whole tensor, not a view
    std::string   label;
    int           bound = 0;      // the index of the bound rule of the kind for this output
    // Element ranges [first, last) that the comparison ignores: regions that the op does not write.
    std::vector<std::pair<int64_t, int64_t>> skip;
};

struct kind_def;

// A decoded case. Each process decodes the case again for each backend, because the tensors of a
// context hold the addresses of one allocation.
struct built_case {
    const kind_def *     kind = nullptr;
    int                  kind_id = -1;
    ggml_context *       ctx  = nullptr;
    ggml_cgraph *        gf   = nullptr;
    std::vector<leaf>    leaves;
    std::vector<output>  outs;
    int                  n_threads  = 1;
    bool                 cpu_repack = false; // weights go into the extra buffer type of the CPU backend
    // The inputs hold Inf, NaN, subnormal or huge values, or a value that a conversion of the op
    // (to f16 or Q8_0) makes infinite, or a rope factor that makes an angle past 2^64, or a norm row
    // with eps = 0 whose divisor is below FLT_MIN (an infinite scale).
    bool                 special    = false;
    std::string          desc;               // the shapes and types, readable
    std::string          path;               // the path class of the backends, for the error table
    std::vector<double>  prm;                // numbers that the bound rules of the kind read

    built_case() = default;
    built_case(const built_case &) = delete;
    built_case & operator=(const built_case &) = delete;
    ~built_case();
};

// The per-element bounds of one output against the oracle. `strict` is the bound of an exact
// f32 computation in another order. `loose` is the bound of a computation that rounds its inputs
// to f16 once and accumulates in f32. The comparison reports an error above `strict` as a finding,
// and an error above `loose` as a finding of high severity.
struct bound_arrays {
    std::vector<double> strict;
    std::vector<double> loose;
    // The f32 overflow factor F of the kind. A backend result of +Inf or -Inf, with the sign of a
    // finite oracle value r and F |r| >= FLT_MAX, is the limit of the f32 range and not a finding:
    // the backend forms a value up to F |r| in f32 on the way to r. 0 turns the rule off.
    double overflow_factor = 0.0;
};

// One kind of case.
struct kind_def {
    const char * name;
    const char * group;
    // Build the graph. Return false when the bytes give no valid case.
    bool (*build)(struct builder & b);
    // Fill the bounds of output `o` from the inputs and the oracle output `ref`.
    void (*bound)(const built_case & c, size_t o, const std::vector<float> & ref, bound_arrays & ba);
    // The bound rule and the reason for it.
    const char * bound_text;
};

// The helper that a kind builder uses: it reads the bytes and makes the input tensors.
struct builder {
    reader &       rd;
    built_case &   c;
    ggml_context * ctx;
    uint64_t       seed;
    bool           wild;      // the case can hold shapes and values outside the limits of the model
    int            n_leaf = 0;

    builder(reader & r, built_case & bc, ggml_context * cx, uint64_t s, bool w)
        : rd(r), c(bc), ctx(cx), seed(s), wild(w) {}

    // Return the value spec with the range [lo, hi]. In a wild case, the bytes can change it to a
    // wide, sparse, constant or tie distribution, or add special values.
    vspec vs(float lo, float hi, bool allow_ties = false);

    // Make an F32 input with the values of the spec.
    ggml_tensor * f32(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, const vspec & v,
                      leaf_role role = leaf_role::DATA);
    // Make an F16 input with the values of the spec, rounded to f16 with round to nearest even.
    ggml_tensor * f16(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, const vspec & v,
                      leaf_role role = leaf_role::DATA);
    // Make a Q8_0 or Q4_0 input with random blocks. The scale exponent is in [emin, emax].
    ggml_tensor * quant(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int emin, int emax,
                        leaf_role role = leaf_role::DATA);
    // Make an input of the type: F32, F16, Q8_0 or Q4_0.
    ggml_tensor * typed(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, const vspec & v,
                        leaf_role role = leaf_role::DATA);
    // Make an I32 input with values in [lo, hi].
    ggml_tensor * i32(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int32_t lo, int32_t hi);
    // Make an I32 or I64 input with distinct values in [0, n_max) for each row. The caller makes
    // sure that ne0 <= n_max.
    ggml_tensor * idx_distinct(ggml_type type, int64_t ne0, int64_t ne1, int64_t n_max);
    // Make an F16 attention mask [n_kv, n_q, ne2, ne3]: causal -Inf, -Inf blocks, and small values.
    ggml_tensor * mask_f16(int64_t n_kv, int64_t n_q, int64_t ne2, int64_t ne3, bool causal);

    // Add an output that the harness reads.
    output & out(ggml_tensor * t, const char * label, int bound = 0);
    // Return the leaf of the tensor, or nullptr.
    leaf * find(ggml_tensor * t);
    // Put point values from the bytes into an F32 or F16 leaf. Wild cases only.
    void inject(leaf & l);
};

// Return the list of the kinds.
const std::vector<kind_def> & kinds();

// Return the index of the kind with the name, or -1.
int kind_index(const std::string & name);

// Return the indexes of the kinds of a group. The name "all" gives every kind.
std::vector<int> group_kinds(const std::string & group);

// Return the names of the groups.
std::vector<std::string> group_names();

// The flag in a forced kind that decodes the case with the tame values (FUZZ_OPS_TAME=1), for
// the inputs that a tame fuzz run made. The flag is part of the pack format.
constexpr int FORCED_TAME = 1 << 30;

// Decode a case. `forced` >= 0 selects the kind and ignores byte 0; FORCED_TAME in `forced`
// decodes with the tame values. Return false when the bytes give no valid case, or when the case
// is larger than `max_bytes` of tensor data.
bool build_case(const uint8_t * data, size_t size, int forced, size_t max_bytes, built_case & c);

// Return the values of the tensor as doubles, in the logical order of its elements. The tensor is a
// leaf or a view of a leaf. The time is O(number of elements). An op output gives an empty vector.
std::vector<double> logical_values(const built_case & c, const ggml_tensor * t);

// Convert the raw bytes of a whole contiguous tensor of the type into floats.
std::vector<float> raw_to_f32(ggml_type type, const uint8_t * data, int64_t n);

} // namespace fo
