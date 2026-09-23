// A stub of tools/htp-lab/isa/isa_kernels.c for the silicon probe.
//
// The build of the DSP library uses this file only when tools/htp-lab/isa/isa_kernels.c does not
// exist. It has the entry point of the real file, isa_run, and two ops, but no op table. Thus the
// library builds and "isaprobe info" runs without the census, and "isaprobe census" stops.
//
// The ops (n_vectors vectors of 128 bytes in each input and in the output):
//   ISA_STUB_OP_QF32_MUL (0)  out = sf(qf32(vmpy(in0.sf, in1.sf))). in0 and in1 are IEEE singles.
//                             The pair is "vN.qf32 = vmpy(vA.sf, vB.sf)" then "vM.sf = vN.qf32".
//   ISA_STUB_OP_I16_ADD  (1)  out = vadd(in0.h, in1.h), modulo 2^16 in each lane.
// The op IDs are provisional. The real op table of isa_kernels.h replaces them.
//
// The return value of isa_run is 0 on success, ISA_STUB_ERR_OP for an unknown op, and
// ISA_STUB_ERR_ARG for a NULL pointer or a negative vector count.

#include <stddef.h>
#include <stdint.h>

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

#define ISA_STUB_OP_QF32_MUL 0
#define ISA_STUB_OP_I16_ADD  1

#define ISA_STUB_ERR_OP  (-1)
#define ISA_STUB_ERR_ARG (-2)

int isa_run(int op_id, const void * in0, const void * in1, const void * in2, void * out, int n_vectors);

// out[i] = sf(qf32(a[i] * b[i])) for n vectors. O(n).
static void op_qf32_mul(const HVX_Vector * a, const HVX_Vector * b, HVX_Vector * out, int n) {
    for (int i = 0; i < n; i++) {
        const HVX_Vector q = Q6_Vqf32_vmpy_VsfVsf(a[i], b[i]);
        out[i] = Q6_Vsf_equals_Vqf32(q);
    }
}

// out[i] = a[i] + b[i] in each int16 lane, modulo 2^16, for n vectors. O(n).
static void op_i16_add(const HVX_Vector * a, const HVX_Vector * b, HVX_Vector * out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = Q6_Vh_vadd_VhVh(a[i], b[i]);
    }
}

// Run one op. The pointers must have 128-byte alignment. The DSP library of the probe gives
// aligned copies of the input and output buffers. The stub ops do not use in2.
int isa_run(int op_id, const void * in0, const void * in1, const void * in2, void * out, int n_vectors) {
    (void) in2;
    if (op_id != ISA_STUB_OP_QF32_MUL && op_id != ISA_STUB_OP_I16_ADD) {
        return ISA_STUB_ERR_OP;
    }
    if (n_vectors < 0) {
        return ISA_STUB_ERR_ARG;
    }
    if (n_vectors == 0) {
        return 0;
    }
    if (in0 == NULL || in1 == NULL || out == NULL) {
        return ISA_STUB_ERR_ARG;
    }

    switch (op_id) {
        case ISA_STUB_OP_QF32_MUL:
            op_qf32_mul((const HVX_Vector *) in0, (const HVX_Vector *) in1, (HVX_Vector *) out, n_vectors);
            return 0;
        case ISA_STUB_OP_I16_ADD:
            op_i16_add((const HVX_Vector *) in0, (const HVX_Vector *) in1, (HVX_Vector *) out, n_vectors);
            return 0;
        default:
            return ISA_STUB_ERR_OP;
    }
}
