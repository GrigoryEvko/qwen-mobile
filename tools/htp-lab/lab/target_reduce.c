// Target 8: the reduction family. RMS norm (hvx-norm.h) and softmax (softmax-ops.c).
//
// The program includes the kernel sources verbatim and calls the leaf routines, thus the
// measurement covers the kernel loops and not the FastRPC path, the op batch, or the work queue
// wakeup of the phone. The row buffers are in DDR and the softmax pad is in the VTCM, as on the
// phone (the op puts its three scratch rows in the VTCM spad).
//
// On the phone RMS_NORM measures 33.4 ms at 1.27 to 1.63 instructions per packet with
// COPROC_BUSY 40 %, the worst packet fill of the profile after the conv, which is the signature
// of a reduction with one accumulator: each add waits two packets for the one before it.
//
// Arguments: --n 2560 --nsm 512 --iters 20 --range 4
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "hvx-norm.h"
#include "softmax-ops.c"

#define TARGET "reduce"

// The scalar reference of ggml_compute_forward_rms_norm_f32, in float64.
static void ref_rms_norm(const float * x, float * y, uint32_t n, float eps) {
    double s = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        s += (double) x[i] * (double) x[i];
    }
    const double scale = 1.0 / sqrt(s / (double) n + (double) eps);
    for (uint32_t i = 0; i < n; i++) {
        y[i] = (float) ((double) x[i] * scale);
    }
}

// The scalar reference of rms_norm followed by a multiply with a weight row.
static void ref_rms_norm_mul(const float * x, const float * w, float * y, uint32_t n, float eps) {
    double s = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        s += (double) x[i] * (double) x[i];
    }
    const double scale = 1.0 / sqrt(s / (double) n + (double) eps);
    for (uint32_t i = 0; i < n; i++) {
        y[i] = (float) ((double) x[i] * scale * (double) w[i]);
    }
}

// The scalar reference of ggml_compute_forward_soft_max_f32, in float64.
static void ref_softmax(const float * x, float * y, uint32_t n) {
    double m = (double) x[0];
    for (uint32_t i = 1; i < n; i++) {
        if ((double) x[i] > m) {
            m = (double) x[i];
        }
    }
    double s = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        s += exp((double) x[i] - m);
    }
    const double inv = s > 0.0 ? 1.0 / s : 1.0;
    for (uint32_t i = 0; i < n; i++) {
        y[i] = (float) (exp((double) x[i] - m) * inv);
    }
}

// The normalized mean squared error of two float arrays. O(n).
static double nmse(const float * got, const float * ref, size_t n) {
    double se = 0.0;
    double sr = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double) got[i] - (double) ref[i];
        se += d * d;
        sr += (double) ref[i] * (double) ref[i];
    }
    return sr > 0.0 ? se / sr : 0.0;
}

// Runs one kernel iters times and reports the best cycle count per element and per vector.
#define LAB_TIME(label, elems, lanes, call)                                        \
    do {                                                                            \
        uint64_t best = UINT64_MAX;                                                 \
        for (uint32_t it = 0; it < iters; it++) {                                   \
            LAB_BARRIER();                                                          \
            const uint64_t t0 = lab_cycles();                                       \
            call;                                                                   \
            const uint64_t t1 = lab_cycles();                                       \
            LAB_BARRIER();                                                          \
            if (t1 - t0 < best) {                                                   \
                best = t1 - t0;                                                     \
            }                                                                       \
        }                                                                           \
        printf("lab: %s %-18s cycles %8llu  per_elem %7.3f  per_vec %8.2f\n",        \
               TARGET, label, (unsigned long long) best,                            \
               (double) best / (double) (elems), (double) best / ((double) (elems) / (lanes))); \
    } while (0)

int main(int argc, char ** argv) {
    const uint32_t n     = (uint32_t) lab_arg_long(argc, argv, "--n", 2560);
    const uint32_t nsm   = (uint32_t) lab_arg_long(argc, argv, "--nsm", 512);
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 20);
    const float    range = (float) lab_arg_long(argc, argv, "--range", 4);

    lab_init();
#ifdef LAB_PROPOSED
    printf("lab: %s build = proposed\n", TARGET);
#else
    printf("lab: %s build = checkout\n", TARGET);
#endif

    const float eps = 1e-6f;

    float * x   = lab_ddr_alloc((size_t) n * sizeof(float) + 256, 128);
    float * w   = lab_ddr_alloc((size_t) n * sizeof(float) + 256, 128);
    float * y   = lab_ddr_alloc((size_t) n * sizeof(float) + 256, 128);
    float * ref = lab_ddr_alloc((size_t) n * sizeof(float) + 256, 128);

    float * xs  = lab_ddr_alloc((size_t) nsm * sizeof(float) + 256, 128);
    float * ys  = lab_ddr_alloc((size_t) nsm * sizeof(float) + 256, 128);
    float * rs  = lab_ddr_alloc((size_t) nsm * sizeof(float) + 256, 128);
    float * pad = (float *) lab_vtcm_alloc((size_t) nsm * sizeof(float) + 256, 128);

    lab_fill_f32(x, n, -range, range);
    lab_fill_f32(w, n, -1.0f, 1.0f);
    lab_fill_f32(xs, nsm, -range, range);

    printf("lab: %s n = %u  nsm = %u  iters = %u\n", TARGET, n, nsm, iters);

    LAB_TIME("rms_norm_f32", n, 32,
             hvx_fast_rms_norm_f32((const uint8_t *) x, (uint8_t *) y, (int) n, eps));
    ref_rms_norm(x, ref, n, eps);
    printf("lab: %s rms_norm_f32 nmse = %.4g\n", TARGET, nmse(y, ref, n));

    LAB_TIME("rms_norm_mul_f32", n, 32,
             hvx_fast_rms_norm_mul_f32((const uint8_t *) x, (const uint8_t *) w, (uint8_t *) y, (int) n, eps));
    ref_rms_norm_mul(x, w, ref, n, eps);
    printf("lab: %s rms_norm_mul_f32 nmse = %.4g\n", TARGET, nmse(y, ref, n));

    LAB_TIME("softmax_f32", nsm, 32,
             hvx_fast_softmax_f32((const uint8_t *) xs, (uint8_t *) ys, (uint8_t *) pad, (int) nsm));
    ref_softmax(xs, rs, nsm);
    printf("lab: %s softmax_f32 nmse = %.4g\n", TARGET, nmse(ys, rs, nsm));

    lab_report(TARGET, "n", n, "");
    lab_report(TARGET, "nsm", nsm, "");
    return 0;
}
