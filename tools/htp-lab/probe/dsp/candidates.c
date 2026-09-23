// The table of the candidate kernels. Refer to candidates.h.
//
// Kernel 0, ref.q8_0_generic, runs the ggml scalar reference (q8_0_ref.c) on the DSP scalar unit.
// It is the test of the harness: the host computes the same code on the ARM CPU, thus the two
// results must agree bit for bit when the two scalar float units are IEEE.
//
// The next kernels are variants of the tiled Q8_0 matvec. Each needs a prepare function that
// repacks the ggml blocks into the 1152-byte weight tiles of the backend, and a run function that
// quantizes x (htp/hvx-q8-ref.h gives the reference quants) and calls the dot of each 32-row tile.

#include "candidates.h"

#include <math.h>
#include <stdlib.h>

#include "q8_0_ref.h"

// Refer to q8_0_ref.c: a DSP image without roundf gives a NULL address.
#pragma weak roundf

// The state of kernel 0: the quantized activation.
static int ref_prepare(const cand_problem * p, void ** state) {
    if (p->op != CAND_OP_MATVEC_Q8_0 || p->cols == 0 || p->cols % QK8_0 != 0) {
        return CAND_ERR_SHAPE;
    }
    if (0 == roundf) {
        return CAND_ERR_RUNTIME;
    }
    *state = malloc((size_t) (p->cols / QK8_0) * sizeof(block_q8_0));
    return *state != NULL ? CAND_OK : CAND_ERR_MEMORY;
}

static int ref_run(const cand_problem * p, void * state) {
    q8_0_ref_matvec((const block_q8_0 *) p->w, p->x, p->y, (block_q8_0 *) state, p->rows, p->cols);
    return CAND_OK;
}

static void ref_release(void * state) {
    free(state);
}

const cand_kernel cand_kernels[] = {
    { "ref.q8_0_generic", CAND_OP_MATVEC_Q8_0, ref_prepare, ref_run, ref_release },
};

const uint32_t cand_n_kernels = sizeof(cand_kernels) / sizeof(cand_kernels[0]);
