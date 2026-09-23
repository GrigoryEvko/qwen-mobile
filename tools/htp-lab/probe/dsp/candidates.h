// The candidate kernels of the probe: kernels that run on silicon against the ggml CPU oracle.
//
// A candidate kernel solves one problem (one op with its inputs). The host program reads the
// problem from a file, the DSP library runs the kernel of the given ID, and the host writes the
// DSP result and the result of the ggml scalar reference (q8_0_ref.c) next to each other.
//
// Add a kernel:
//   1. Write its three functions in candidates.c (or in a file that candidates.c includes).
//      prepare converts the problem into the state of the kernel (for example the weight tiles
//      of the llama.cpp backend). run is the measured part. release frees the state.
//   2. Add one line to cand_kernels[]. The index in that table is the kernel ID.
// The DSP library calls prepare one time, run for each iteration, and release one time, on a
// thread that holds an HVX context and the HMX (when the library holds it).
#ifndef ISAPROBE_CANDIDATES_H
#define ISAPROBE_CANDIDATES_H

#include <stddef.h>
#include <stdint.h>

// The ops of the problems.
#define CAND_OP_MATVEC_Q8_0 1  // y[rows] = W[rows x cols] (ggml Q8_0 blocks) . x[cols] (f32)

// The status values of the functions. Any other nonzero value is a failure of the kernel.
#define CAND_OK          0
#define CAND_ERR_SHAPE   (-1)  // the kernel does not support these dimensions
#define CAND_ERR_MEMORY  (-2)  // no memory for the state
#define CAND_ERR_RUNTIME (-3)  // the DSP image lacks a function that the kernel needs

typedef struct cand_problem {
    uint32_t      op;          // CAND_OP_*
    uint32_t      rows;        // the rows of W, and the length of y
    uint32_t      cols;        // the length of x, a multiple of 32
    const void *  w;           // rows x cols / 32 block_q8_0 (34 bytes each), the ggml layout
    const float * x;           // cols f32 values
    float *       y;           // rows f32 values: the output
    uint8_t *     vtcm;        // the VTCM that the library holds, NULL when it holds none
    size_t        vtcm_bytes;
} cand_problem;

typedef struct cand_kernel {
    const char * name;  // "<method>.<op>", for example "ref.q8_0_generic"
    uint32_t     op;    // the op that the kernel solves
    int (*prepare)(const cand_problem * p, void ** state);
    int (*run)(const cand_problem * p, void * state);
    void (*release)(void * state);
} cand_kernel;

extern const cand_kernel cand_kernels[];
extern const uint32_t    cand_n_kernels;

#endif
