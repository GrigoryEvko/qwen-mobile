// Target 8: the activation family of the backend, SIGMOID, SILU and SWIGLU.
//
// On the phone these three are the largest part of the elementwise family of a 512-token prefill
// of the 4B (the family is 114 ms of 536 ms of DSP op time, measured 2026-09-20). The op reads
// and writes a VTCM scratch pad that the DMA fills from the DDR, thus this target puts its
// buffers in the VTCM and measures the kernel alone, without the DMA and the work queue.
//
// The program calls the same helpers as the op:
//   SIGMOID  hvx_sigmoid_f32_aa(dst, src, n)
//   SILU     hvx_sigmoid_f32_aa(dst, src, n) then a multiply pass over the row
//   SWIGLU   hvx_sigmoid_f32_aa(dst, src0, n) then hvx_mul_mul_f32_aa(dst, src0, dst, src1, n)
// Thus each kernel walks the row twice, which the proposal changes to one pass.
//
// Under LAB_PROPOSED the target also measures the fused paths of the proposal and reports the
// error of each against the same reference.
//
// One invocation measures one kernel, because the register allocation of the compiler depends on
// the whole program: with six kernels in one program the baseline sigmoid measured 155 cycles for
// each vector and alone it measures 51, thus the compiler had dropped its unrolling. --only
// selects the kernel and the default runs each of them in a separate pass of the same program,
// which is not a fair comparison. Use --only for every number that goes into a report.
//
// Arguments: --nc 9216 --iters 3 --range 8 --only <name>
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "hvx-utils.h"
#include "hvx-sigmoid.h"
#include "hvx-arith.h"

#ifdef LAB_PROPOSED
#include "hvx-act-i16.h"
#endif

#define TARGET "act"
#define VEC_F32 32

// The three references in double precision. The value of a reference is the value that the CPU
// backend of ggml computes, thus the comparison measures the error of the HVX path alone.
static void ref_sigmoid(const float * x, double * y, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        y[i] = 1.0 / (1.0 + exp(-(double) x[i]));
    }
}

static void ref_silu(const float * x, double * y, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        y[i] = (double) x[i] / (1.0 + exp(-(double) x[i]));
    }
}

static void ref_swiglu(const float * x0, const float * x1, double * y, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        y[i] = (double) x1[i] * (double) x0[i] / (1.0 + exp(-(double) x0[i]));
    }
}

// The normalized mean squared error against the double reference
static double nmse(const float * got, const double * ref, uint32_t n) {
    double se = 0.0;
    double sr = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        const double d = (double) got[i] - ref[i];
        se += d * d;
        sr += ref[i] * ref[i];
    }
    return sr > 0.0 ? se / sr : 0.0;
}

// The largest relative error, and where it is
static double worst_rel(const float * got, const double * ref, uint32_t n, const float * x, uint32_t * at) {
    double w = 0.0;
    *at = 0;
    for (uint32_t i = 0; i < n; i++) {
        const double r = fabs(ref[i]);
        const double e = fabs((double) got[i] - ref[i]) / (r + 1e-6);
        if (e > w) {
            w = e;
            *at = i;
        }
    }
    (void) x;
    return w;
}

// The kernels of the checkout, as the op calls them. Each is a function of its own and none of
// them is inline: the register allocation of one kernel then does not depend on the others, which
// is how unary-ops.c and act-ops.c compile them, and one program can hold every kernel without a
// kernel becoming slower because a second one exists.
#define LAB_KERNEL static __attribute__((noinline))

LAB_KERNEL void base_sigmoid(uint8_t * dst, const uint8_t * src, uint32_t n) {
    hvx_sigmoid_f32_aa(dst, src, n);
}

LAB_KERNEL void base_silu(uint8_t * dst, const uint8_t * src, uint32_t n) {
    hvx_sigmoid_f32_aa(dst, src, n);
    hvx_mul_f32_aaa(dst, src, dst, n);
}

LAB_KERNEL void base_swiglu(uint8_t * dst, const uint8_t * src0, const uint8_t * src1, uint32_t n) {
    hvx_sigmoid_f32_aa(dst, src0, n);
    hvx_mul_mul_f32_aa(dst, src0, dst, src1, n);
}

LAB_KERNEL void base_copy(uint8_t * dst, const uint8_t * src, uint32_t n) {
    hvx_copy_f32_aa(dst, src, n);
}

#ifdef LAB_PROPOSED
LAB_KERNEL void i16_sigmoid(uint8_t * dst, const uint8_t * src, uint32_t n) {
    hvx_sigmoid_i16_f32_aa(dst, src, n);
}

LAB_KERNEL void i16_silu(uint8_t * dst, const uint8_t * src, uint32_t n) {
    hvx_silu_i16_f32_aa(dst, src, n);
}

LAB_KERNEL void i16_swiglu(uint8_t * dst, const uint8_t * src0, const uint8_t * src1, uint32_t n) {
    hvx_swiglu_i16_f32_aa(dst, src0, src1, n);
}
#endif

int main(int argc, char ** argv) {
    const uint32_t nc      = (uint32_t) lab_arg_long(argc, argv, "--nc", 9216);
    const uint32_t iters   = (uint32_t) lab_arg_long(argc, argv, "--iters", 3);
    const float    x_range = (float) lab_arg_long(argc, argv, "--range", 8);

    // --only <name> keeps one kernel in the program. The argument parser of the lab reads
    // numbers, thus the name comes from argv directly.
    const char * only = NULL;
    for (int i = 1; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--only")) {
            only = argv[i + 1];
        }
    }

    lab_init();

    const uint32_t nvec  = nc / VEC_F32;
    const size_t   bytes = (size_t) nc * sizeof(float);

    // The op works on a VTCM scratch pad, thus so does this target.
    float * src0 = lab_vtcm_alloc(bytes, 128);
    float * src1 = lab_vtcm_alloc(bytes, 128);
    float * dst  = lab_vtcm_alloc(bytes, 128);

    float *  x0  = lab_ddr_alloc(bytes, 128);
    float *  x1  = lab_ddr_alloc(bytes, 128);
    float *  got = lab_ddr_alloc(bytes, 128);
    double * ref = lab_ddr_alloc((size_t) nc * sizeof(double), 128);

    lab_fill_f32(x0, nc, -x_range, x_range);
    lab_fill_f32(x1, nc, -2.0f, 2.0f);

    printf("lab: %s nc = %u vectors = %u range = +-%g\n", TARGET, nc, nvec, (double) x_range);

    // One case: fill the inputs, run the kernel iters times, keep the fastest, then check.
#define RUN_CASE(name, ref_call, kernel_call)                                                     \
    do {                                                                                          \
        if (only && strcmp(only, name)) {                                                         \
            break;                                                                                \
        }                                                                                         \
        memcpy(src0, x0, bytes);                                                                  \
        memcpy(src1, x1, bytes);                                                                  \
        kernel_call;                          /* the warm-up fills the caches */                  \
        uint64_t best = UINT64_MAX;                                                               \
        for (uint32_t it = 0; it < iters; it++) {                                                 \
            LAB_BARRIER();                                                                        \
            const uint64_t t0 = lab_cycles();                                                     \
            kernel_call;                                                                          \
            const uint64_t t1 = lab_cycles();                                                     \
            LAB_BARRIER();                                                                        \
            if (t1 - t0 < best) {                                                                 \
                best = t1 - t0;                                                                   \
            }                                                                                     \
        }                                                                                         \
        memcpy(got, dst, bytes);                                                                  \
        ref_call;                                                                                 \
        uint32_t at = 0;                                                                          \
        const double w = worst_rel(got, ref, nc, x0, &at);                                        \
        printf("lab: %s %-16s %8.2f cycles/vector  %7.2f packets/vector  nmse %9.3g"              \
               "  worst rel %8.3g at x = % .4f\n",                                                \
               TARGET, name, (double) best / nvec, (double) best / nvec / 2.0,                    \
               nmse(got, ref, nc), w, (double) x0[at]);                                           \
    } while (0)

    RUN_CASE("sigmoid.base", ref_sigmoid(x0, ref, nc),
             base_sigmoid((uint8_t *) dst, (const uint8_t *) src0, nc));
    RUN_CASE("silu.base", ref_silu(x0, ref, nc),
             base_silu((uint8_t *) dst, (const uint8_t *) src0, nc));
    RUN_CASE("swiglu.base", ref_swiglu(x0, x1, ref, nc),
             base_swiglu((uint8_t *) dst, (const uint8_t *) src0, (const uint8_t *) src1, nc));

#ifdef LAB_PROPOSED
    RUN_CASE("sigmoid.i16", ref_sigmoid(x0, ref, nc),
             i16_sigmoid((uint8_t *) dst, (const uint8_t *) src0, nc));
    RUN_CASE("silu.i16", ref_silu(x0, ref, nc),
             i16_silu((uint8_t *) dst, (const uint8_t *) src0, nc));
    RUN_CASE("swiglu.i16", ref_swiglu(x0, x1, ref, nc),
             i16_swiglu((uint8_t *) dst, (const uint8_t *) src0, (const uint8_t *) src1, nc));
#endif

    // A plain copy of the row gives the memory floor of a one-pass kernel on this pad.
    if (!only || !strcmp(only, "copy.floor")) {
        uint64_t best = UINT64_MAX;
        for (uint32_t it = 0; it < iters; it++) {
            LAB_BARRIER();
            const uint64_t t0 = lab_cycles();
            base_copy((uint8_t *) dst, (const uint8_t *) src0, nc);
            const uint64_t t1 = lab_cycles();
            LAB_BARRIER();
            if (t1 - t0 < best) {
                best = t1 - t0;
            }
        }
        printf("lab: %s %-16s %8.2f cycles/vector  %7.2f packets/vector  (1 load + 1 store)\n",
               TARGET, "copy.floor", (double) best / nvec, (double) best / nvec / 2.0);
    }

    return 0;
}
