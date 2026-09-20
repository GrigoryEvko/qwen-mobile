// Target 8: the softplus op of unary-ops.c, the only scalar element loop of that file.
//
// The op of the checkout evaluates logf(1.0f + expf(x)) one element at a time, at both of its
// sites (softplus_f32 and tile_unary_softplus_f32). On the phone it measures 22.3 ms of the
// 536 ms of DSP op time of a 512-token prefill of the 4B Q8_0, at 2.24 instructions per packet
// with CU_BUSY at 70.9 %, which is the signature of scalar float work.
//
// The program times that scalar loop and, with the proposal applied (LAB_PROPOSED), the two
// vector paths of hvx-softplus.h against it. Each path is checked against a float64 reference.
// The report gives the cycles for each vector of 32 f32 elements, thus the paths compare
// directly whatever the row length.
//
// Arguments: --n 2048 --rows 8 --iters 3 --range 12
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "hvx-utils.h"

#ifdef LAB_PROPOSED
#include "hvx-softplus.h"
#endif

#define TARGET "unary"

// The scalar loop of the checkout, both sites hold it verbatim.
static void softplus_scalar(const float * restrict src, float * restrict dst, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const float x = src[i];
        dst[i] = (x > 20.0f) ? x : logf(1.0f + expf(x));
    }
}

// The reference, in float64 and with no cancellation.
static double softplus_ref(double x) {
    return (x > 0.0 ? x : 0.0) + log1p(exp(-fabs(x)));
}

// Time one row function over the rows, and report its error against the reference. O(rows * n).
static void measure(const char * name, void (*fn)(const float * restrict, float * restrict, uint32_t),
                    const float * src, float * dst, const double * ref,
                    uint32_t n, uint32_t rows, uint32_t iters) {
    fn(src, dst, n);                                   // the warm-up run fills the caches

    uint64_t best = UINT64_MAX;
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        for (uint32_t r = 0; r < rows; r++) {
            fn(src + (size_t) r * n, dst + (size_t) r * n, n);
        }
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best) {
            best = t1 - t0;
        }
    }

    // the error of the first row, which every row repeats
    fn(src, dst, n);
    double se = 0.0, sr = 0.0, worst_abs = 0.0, worst_rel = 0.0, worst_rel_tail = 0.0;
    double at_abs = 0.0, at_rel = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        const double d = (double) dst[i] - ref[i];
        se += d * d;
        sr += ref[i] * ref[i];
        const double a = fabs(d);
        const double rel = a / ref[i];
        if (a > worst_abs)   { worst_abs = a;  at_abs = (double) src[i]; }
        if (rel > worst_rel) { worst_rel = rel; at_rel = (double) src[i]; }
        if (src[i] < -8.0f && rel > worst_rel_tail) { worst_rel_tail = rel; }
    }

    const double vectors = (double) rows * n / VLEN_FP32;
    printf("lab: %s %-14s cyc/vec %9.2f  cyc/elem %7.3f  nmse %9.2e  max abs %9.2e (x %+.3f)"
           "  max rel %9.2e (x %+.3f)  max rel x<-8 %9.2e\n",
           TARGET, name, (double) best / vectors, (double) best / ((double) rows * n),
           sr > 0.0 ? se / sr : 0.0, worst_abs, at_abs, worst_rel, at_rel, worst_rel_tail);
}

#ifdef LAB_PROPOSED
static void softplus_vec_f32(const float * restrict src, float * restrict dst, uint32_t n) {
    hvx_softplus_f32_aa((uint8_t *) dst, (const uint8_t *) src, n);
}

static void softplus_vec_i16(const float * restrict src, float * restrict dst, uint32_t n) {
    hvx_softplus_i16_f32_aa((uint8_t *) dst, (const uint8_t *) src, n);
}
#endif

int main(int argc, char ** argv) {
    const uint32_t n     = (uint32_t) lab_arg_long(argc, argv, "--n", 2048);
    const uint32_t rows  = (uint32_t) lab_arg_long(argc, argv, "--rows", 8);
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 3);
    const float    range = (float) lab_arg_long(argc, argv, "--range", 12);

    lab_init();

    const size_t total = (size_t) rows * n;
    float *  src = lab_ddr_alloc(total * sizeof(float) + 256, 128);
    float *  dst = lab_ddr_alloc(total * sizeof(float) + 256, 128);
    double * ref = lab_ddr_alloc((size_t) n * sizeof(double), 128);

    lab_fill_f32(src, n, -range, range);
    // the first lanes cover the ends of the range, which the random fill reaches rarely
    const float edges[8] = { 0.0f, -1e-4f, 1e-4f, -25.0f, 25.0f, -8.5f, 8.5f, -0.5f };
    for (uint32_t i = 0; i < 8 && i < n; i++) {
        src[i] = edges[i];
    }
    for (uint32_t r = 1; r < rows; r++) {
        memcpy(src + (size_t) r * n, src, (size_t) n * sizeof(float));
    }
    for (uint32_t i = 0; i < n; i++) {
        ref[i] = softplus_ref((double) src[i]);
    }

    printf("lab: %s n = %u rows = %u range = %.1f\n", TARGET, n, rows, (double) range);
    measure("scalar", softplus_scalar, src, dst, ref, n, rows, iters);
#ifdef LAB_PROPOSED
    measure("vector_f32", softplus_vec_f32, src, dst, ref, n, rows, iters);
    measure("vector_i16", softplus_vec_i16, src, dst, ref, n, rows, iters);
#endif
    return 0;
}
