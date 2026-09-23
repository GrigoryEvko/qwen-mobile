// The CPU oracle of the HVX census: the naive IEEE result of each census op that has an IEEE
// counterpart, computed by scalar C on the ARM CPU of the phone (the host program).
//
// The rules of the oracle (oracle.c has the table of the ops):
//   - Plain scalar C, compiled with -ffp-contract=off and without fast math. No SIMD intrinsics.
//   - Each arithmetic operation rounds to nearest even to the format of its result. An f16 result
//     comes from the f32 operation on the exact f32 values of the inputs, then one conversion to
//     f16. For add, sub and mul that is the correctly rounded f16 result (24 >= 2 x 11 + 2).
//   - A chain (x += a * b, a * b + c, a0 * b0 + a1 * b1 + c) rounds after each operation, from
//     left to right, as the C expression does without contraction.
//   - Every float to integer conversion rounds to nearest even (rintf), saturates, and gives 0 for
//     a NaN. Every integer to float conversion rounds to nearest even.
//   - max and min prefer the number to a NaN, and +0 to -0 (max) or -0 to +0 (min).
//   - For a qf32 or qf16 result, the oracle is the f32 or f16 result of the same operation.
//   - An op with a qf, f8 or raw input has no oracle, because no IEEE value holds such an input.
#ifndef ISAPROBE_ORACLE_H
#define ISAPROBE_ORACLE_H

#include <stddef.h>
#include <stdint.h>

// One census op, from the op table of the DSP library (isaprobe_op_info) or of isa_ops.csv.
struct oracle_op {
    const char * name;         // the census name, for example "cc.Q6_Vsf_vadd_VsfVsf"
    uint32_t     in_type[3];   // enum isa_type
    uint32_t     in_bytes[3];  // bytes per iteration from each input
    uint32_t     out_type;     // enum isa_type of the DSP output
    uint32_t     out_bytes;    // bytes per iteration of the output
    uint32_t     n_vectors;    // iterations
};

// The oracle of one op, found by oracle_find.
struct oracle_spec {
    int kind;    // the operation (enum in oracle.c)
    int layout;  // the lane layout (enum in oracle.c)
    int pred;    // the predicate update of a compare (enum in oracle.c)
};

// Return 1 and fill *spec when the op has an oracle, else 0. O(size of the table).
int oracle_find(const struct oracle_op * op, struct oracle_spec * spec);

// Return the enum isa_type of the oracle output: sf for a qf32 result, hf for a qf16 result,
// else the type of the DSP output.
uint32_t oracle_out_type(uint32_t out_type);

// Compute the oracle output of n_vectors x out_bytes bytes into out. in[j] holds at least
// n_vectors x in_bytes[j] bytes, or is NULL for an input that the op does not use.
// O(n_vectors x lanes).
void oracle_run(const struct oracle_op * op, const struct oracle_spec * spec, const uint8_t * const in[3], uint8_t * out);

// Return the name of the operation of a spec, for the record.
const char * oracle_kind_name(const struct oracle_spec * spec);

#endif
