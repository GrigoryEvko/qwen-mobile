// The scalar ggml reference of the Q8_0 matrix-vector product: the activation quantizer
// quantize_row_q8_0_ref and the generic dot ggml_vec_dot_q8_0_q8_0_generic, copied from the
// llama.cpp submodule (third_party/llama.cpp, commit c6824a9, files not changed by the patch
// series). q8_0_ref.c has the copies. The host program uses them as the CPU oracle, and the DSP
// library runs the same code as the candidate kernel 0.
//
// Compile q8_0_ref.c with -ffp-contract=off and without fast math. The result then is the naive
// IEEE execution of the ggml code: each float operation rounds to nearest even, and no multiply
// and add fuse into one instruction.
#ifndef ISAPROBE_Q8_0_REF_H
#define ISAPROBE_Q8_0_REF_H

#include <stddef.h>
#include <stdint.h>

#define GGML_RESTRICT restrict

// ggml/src/ggml-common.h:6
typedef uint16_t ggml_half;
typedef uint16_t ggml_fp16_t;

// ggml/src/ggml-common.h:251-255
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

// Quantize k f32 values (k a multiple of QK8_0) into k / QK8_0 blocks.
void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);

// *s = the dot of n values (n a multiple of QK8_0) of the Q8_0 rows vx and vy. nrc must be 1.
void ggml_vec_dot_q8_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// The f32 value of an fp16 value, and the fp16 value of an f32 value (round to nearest even).
float       ggml_ref_fp16_to_fp32(ggml_fp16_t h);
ggml_fp16_t ggml_ref_fp32_to_fp16(float f);

// The matrix-vector product of the ggml CPU path: quantize x (cols values) with
// quantize_row_q8_0_ref into q (cols / QK8_0 blocks of scratch), then y[r] =
// ggml_vec_dot_q8_0_q8_0_generic of row r of w (rows x cols / QK8_0 blocks) and q. O(rows x cols).
void q8_0_ref_matvec(const block_q8_0 * w, const float * x, float * y, block_q8_0 * q, uint32_t rows, uint32_t cols);

#endif
