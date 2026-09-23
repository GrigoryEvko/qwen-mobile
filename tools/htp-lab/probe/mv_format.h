// The files of the candidate-kernel harness (isaprobe kernel, mv_case.py, oracle_compare.py).
//
// A problem file: one header with the magic "IPMV", then the inputs of the op.
//   CAND_OP_MATVEC_Q8_0: rows x cols / 32 block_q8_0 (34 bytes: fp16 d, 32 int8), then cols f32.
// A result file: one header with the magic "IPMR", then the output of the op.
//   CAND_OP_MATVEC_Q8_0: rows f32.
// All fields are little endian. oracle_compare.py and mv_case.py read the same layout with the
// Python format "<4s7IQQiI64s8s".
#ifndef ISAPROBE_MV_FORMAT_H
#define ISAPROBE_MV_FORMAT_H

#include <stdint.h>

#define MV_MAGIC_PROBLEM "IPMV"
#define MV_MAGIC_RESULT  "IPMR"
#define MV_VERSION       1
#define MV_KERNEL_ORACLE 0xFFFFFFFFu  // the kernel_id of the result of the ggml CPU reference
#define MV_SOURCE_CHIP   1
#define MV_SOURCE_ORACLE 2

typedef struct mv_header {
    char     magic[4];   // MV_MAGIC_PROBLEM or MV_MAGIC_RESULT
    uint32_t version;    // MV_VERSION
    uint32_t op;         // CAND_OP_* of dsp/candidates.h
    uint32_t rows;
    uint32_t cols;
    uint32_t kernel_id;  // result: the candidate kernel, or MV_KERNEL_ORACLE
    uint32_t source;     // result: MV_SOURCE_CHIP or MV_SOURCE_ORACLE
    uint32_t iters;      // result: the timed iterations
    uint64_t pcycles;    // result: the minimum DSP cycles of one iteration (0 for the oracle)
    uint64_t usecs;      // result: the minimum time of one iteration in microseconds
    int32_t  status;     // result: the kernel status (0 is success)
    uint32_t seed;       // problem: the seed of mv_case.py (information only)
    char     name[64];   // problem: a label; result: the kernel name, zero padded
    uint8_t  reserved[8];
} mv_header;

_Static_assert(sizeof(mv_header) == 128, "the harness file header must have 128 bytes");

#endif
