// Target 8: the binary elementwise family of hvx-arith.h (MUL, ADD, SUB) and the value of fusion.
//
// On the phone these ops are memory bound, not packet bound. One f32 vector of a two-input
// elementwise op moves 384 bytes (two reads and one write) and needs 2 packets of work, thus at
// 24 bytes per cycle of DDR bandwidth the bytes cost about 16 cycles and the packets about 4.
// The op is therefore about four times memory bound, and the lever is passes, not instructions.
//
// The program measures each op twice: with the buffers in the VTCM, which is what the DMA
// double-buffered op loop of binary-ops.c really sees, and with the buffers in DDR, which is what
// a caller that skips the scratchpad sees. It then measures a chain of two ops as two passes
// against the same chain as one fused pass, which is the change that removes bytes.
//
// Arguments: --n 9216 --iters 8
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

// hvx-sqrt.h comes first because hvx-arith.h of the checkout calls HVX_OP_MUL, which only
// hvx-sqrt.h defines. The backend builds only because another header pulls it in first.
// The proposal removes that dependency, and this include keeps the baseline buildable.
#include "hvx-sqrt.h"
#include "hvx-arith.h"

#define TARGET "binary"
#define VEC_F32 32              // f32 elements in one HVX vector
#define MHZ 2112.0              // the DSP clock of the phone
#define CHECK_N 4096            // elements that the scalar reference covers

// One measured case.
struct bin_case {
    const char * name;
    const char * note;
    unsigned     reads;         // vector reads of one output vector
    unsigned     writes;        // vector writes of one output vector
};

// The scalar reference of each chain, in double.
static void ref_mul(const float * a, const float * b, float * y, uint32_t n) {
    if (n > CHECK_N) { n = CHECK_N; }
    for (uint32_t i = 0; i < n; i++) {
        y[i] = (float) ((double) a[i] * (double) b[i]);
    }
}

static void ref_add(const float * a, const float * b, float * y, uint32_t n) {
    if (n > CHECK_N) { n = CHECK_N; }
    for (uint32_t i = 0; i < n; i++) {
        y[i] = (float) ((double) a[i] + (double) b[i]);
    }
}

static void ref_mul_mul(const float * a, const float * b, const float * c, float * y, uint32_t n) {
    if (n > CHECK_N) { n = CHECK_N; }
    for (uint32_t i = 0; i < n; i++) {
        y[i] = (float) (((double) a[i] * (double) b[i]) * (double) c[i]);
    }
}

static void ref_mul_add(const float * a, const float * b, const float * c, float * y, uint32_t n) {
    if (n > CHECK_N) { n = CHECK_N; }
    for (uint32_t i = 0; i < n; i++) {
        y[i] = (float) ((double) a[i] * (double) b[i] + (double) c[i]);
    }
}

// The normalized mean squared error of y against the reference. O(n).
static double nmse(const float * got, const float * ref, uint32_t n) {
    double se = 0.0;
    double sr = 0.0;
    if (n > CHECK_N) { n = CHECK_N; }
    for (uint32_t i = 0; i < n; i++) {
        const double d = (double) got[i] - (double) ref[i];
        se += d * d;
        sr += (double) ref[i] * (double) ref[i];
    }
    return sr > 0.0 ? se / sr : 0.0;
}

// Prints one row of the table: the cycles of one output vector, the packets that the bytes and
// the work each imply, and the bandwidth that the cycles correspond to at the phone clock.
static void row(const char * where, const struct bin_case * c, uint64_t cycles, uint32_t n, double err) {
    const double vectors = (double) n / VEC_F32;
    const double per_vec = (double) cycles / vectors;
    const double bytes   = (double) (c->reads + c->writes) * 128.0;
    printf("lab: %s %-18s %-5s %9.2f %9.2f %8.1f  %9.2e  %s\n", TARGET, c->name, where,
           per_vec, bytes / per_vec, bytes / per_vec * MHZ / 1000.0, err, c->note);
}

int main(int argc, char ** argv) {
    const uint32_t n     = (uint32_t) lab_arg_long(argc, argv, "--n", 9216);
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 8);

    if (n % VEC_F32) {
        printf("lab: %s needs n to be a multiple of %d\n", TARGET, VEC_F32);
        return 2;
    }

    lab_init();

    const size_t bytes = (size_t) n * sizeof(float);

    // Four buffers in each memory, so a two-input op and a three-input chain both fit.
    float * v_a = lab_vtcm_alloc(bytes, 128);
    float * v_b = lab_vtcm_alloc(bytes, 128);
    float * v_c = lab_vtcm_alloc(bytes, 128);
    float * v_y = lab_vtcm_alloc(bytes, 128);
    float * v_t = lab_vtcm_alloc(bytes, 128);

    float * d_a = lab_ddr_alloc(bytes, 128);
    float * d_b = lab_ddr_alloc(bytes, 128);
    float * d_c = lab_ddr_alloc(bytes, 128);
    float * d_y = lab_ddr_alloc(bytes, 128);
    float * d_t = lab_ddr_alloc(bytes, 128);

    float * ref = lab_ddr_alloc(bytes, 128);

    lab_fill_f32(d_a, n, -2.0f, 2.0f);
    lab_fill_f32(d_b, n, -2.0f, 2.0f);
    lab_fill_f32(d_c, n, -2.0f, 2.0f);
    memcpy(v_a, d_a, bytes);
    memcpy(v_b, d_b, bytes);
    memcpy(v_c, d_c, bytes);

    printf("lab: %s n = %u, iters = %u\n", TARGET, n, iters);
    printf("lab: %s %-18s %-5s %9s %9s %8s  %9s  %s\n", TARGET, "case", "mem",
           "cyc/vec", "B/cyc", "GB/s", "nmse", "note");

    // The macro times one expression over iters repetitions and keeps the fastest.
#define TIME(expr)                                     \
    ({                                                 \
        uint64_t best = UINT64_MAX;                    \
        (expr);                                        \
        for (uint32_t it = 0; it < iters; it++) {      \
            LAB_BARRIER();                             \
            const uint64_t t0 = lab_cycles();          \
            (expr);                                    \
            const uint64_t t1 = lab_cycles();          \
            LAB_BARRIER();                             \
            if (t1 - t0 < best) { best = t1 - t0; }    \
        }                                              \
        best;                                          \
    })

    uint64_t cyc;

    // 1. The plain two-input ops, in the VTCM and in DDR.
    static const struct bin_case c_mul = { "mul_f32", "dst = a * b", 2, 1 };
    static const struct bin_case c_add = { "add_f32", "dst = a + b", 2, 1 };

    ref_mul(d_a, d_b, ref, n);
    cyc = TIME(hvx_mul_f32((uint8_t *) v_y, (const uint8_t *) v_a, (const uint8_t *) v_b, n));
    row("vtcm", &c_mul, cyc, n, nmse(v_y, ref, n));
    cyc = TIME(hvx_mul_f32((uint8_t *) d_y, (const uint8_t *) d_a, (const uint8_t *) d_b, n));
    row("ddr", &c_mul, cyc, n, nmse(d_y, ref, n));

    ref_add(d_a, d_b, ref, n);
    cyc = TIME(hvx_add_f32((uint8_t *) v_y, (const uint8_t *) v_a, (const uint8_t *) v_b, n));
    row("vtcm", &c_add, cyc, n, nmse(v_y, ref, n));
    cyc = TIME(hvx_add_f32((uint8_t *) d_y, (const uint8_t *) d_a, (const uint8_t *) d_b, n));
    row("ddr", &c_add, cyc, n, nmse(d_y, ref, n));

    // 2. A chain of two multiplies: two passes through a temporary against one fused pass.
    //    Two passes read a, b, write t, then read t, c, write y: 4 reads and 2 writes.
    //    One pass reads a, b, c and writes y: 3 reads and 1 write.
    static const struct bin_case c_two  = { "mul+mul 2 passes", "t = a * b, then y = t * c", 4, 2 };
    static const struct bin_case c_fuse = { "mul_mul fused",    "y = a * b * c in one pass",  3, 1 };

    ref_mul_mul(d_a, d_b, d_c, ref, n);

    cyc = TIME(({
        hvx_mul_f32((uint8_t *) v_t, (const uint8_t *) v_a, (const uint8_t *) v_b, n);
        hvx_mul_f32((uint8_t *) v_y, (const uint8_t *) v_t, (const uint8_t *) v_c, n);
    }));
    row("vtcm", &c_two, cyc, n, nmse(v_y, ref, n));
    cyc = TIME(({
        hvx_mul_f32((uint8_t *) d_t, (const uint8_t *) d_a, (const uint8_t *) d_b, n);
        hvx_mul_f32((uint8_t *) d_y, (const uint8_t *) d_t, (const uint8_t *) d_c, n);
    }));
    row("ddr", &c_two, cyc, n, nmse(d_y, ref, n));

    cyc = TIME(hvx_mul_mul_f32_aa((uint8_t *) v_y, (const uint8_t *) v_a, (const uint8_t *) v_b,
                                  (const uint8_t *) v_c, n));
    row("vtcm", &c_fuse, cyc, n, nmse(v_y, ref, n));
    cyc = TIME(hvx_mul_mul_f32_aa((uint8_t *) d_y, (const uint8_t *) d_a, (const uint8_t *) d_b,
                                  (const uint8_t *) d_c, n));
    row("ddr", &c_fuse, cyc, n, nmse(d_y, ref, n));

#ifdef LAB_PROPOSED
    // 3. The fused chains that the proposal adds.
    static const struct bin_case c_madd2 = { "mul+add 2 passes", "t = a * b, then y = t + c", 4, 2 };
    static const struct bin_case c_madd  = { "mul_add fused",    "y = a * b + c in one pass",  3, 1 };

    ref_mul_add(d_a, d_b, d_c, ref, n);

    cyc = TIME(({
        hvx_mul_f32((uint8_t *) v_t, (const uint8_t *) v_a, (const uint8_t *) v_b, n);
        hvx_add_f32((uint8_t *) v_y, (const uint8_t *) v_t, (const uint8_t *) v_c, n);
    }));
    row("vtcm", &c_madd2, cyc, n, nmse(v_y, ref, n));

    cyc = TIME(hvx_mul_add_f32((uint8_t *) v_y, (const uint8_t *) v_a, (const uint8_t *) v_b,
                               (const uint8_t *) v_c, n));
    row("vtcm", &c_madd, cyc, n, nmse(v_y, ref, n));
    cyc = TIME(hvx_mul_add_f32((uint8_t *) d_y, (const uint8_t *) d_a, (const uint8_t *) d_b,
                               (const uint8_t *) d_c, n));
    row("ddr", &c_madd, cyc, n, nmse(d_y, ref, n));

    // 4. How the cost of one pass grows with the number of inputs. If a pass with 4 reads costs
    //    about what a pass with 1 read costs, then work inside a pass is free and only passes
    //    are expensive, which is the whole argument for fusion.
    static const struct bin_case c_cp = { "copy 1 read",   "dst = a",             1, 1 };
    static const struct bin_case c_m4 = { "mul4 4 reads",  "y = a * b * c * d",   4, 1 };

    cyc = TIME(hvx_copy_f32_aa((uint8_t *) v_y, (const uint8_t *) v_a, n));
    row("vtcm", &c_cp, cyc, n, 0.0);
    cyc = TIME(hvx_copy_f32_aa((uint8_t *) d_y, (const uint8_t *) d_a, n));
    row("ddr", &c_cp, cyc, n, 0.0);

    for (uint32_t i = 0; i < (n > CHECK_N ? CHECK_N : n); i++) {
        ref[i] = (float) ((double) d_a[i] * (double) d_b[i] * (double) d_c[i] * (double) d_a[i]);
    }
    cyc = TIME(hvx_mul4_f32_aa((uint8_t *) v_y, (const uint8_t *) v_a, (const uint8_t *) v_b,
                               (const uint8_t *) v_c, (const uint8_t *) v_a, n));
    row("vtcm", &c_m4, cyc, n, nmse(v_y, ref, n));
    cyc = TIME(hvx_mul4_f32_aa((uint8_t *) d_y, (const uint8_t *) d_a, (const uint8_t *) d_b,
                               (const uint8_t *) d_c, (const uint8_t *) d_a, n));
    row("ddr", &c_m4, cyc, n, nmse(d_y, ref, n));

    cyc = TIME(({
        hvx_mul_f32((uint8_t *) d_t, (const uint8_t *) d_a, (const uint8_t *) d_b, n);
        hvx_add_f32((uint8_t *) d_y, (const uint8_t *) d_t, (const uint8_t *) d_c, n);
    }));
    ref_mul_add(d_a, d_b, d_c, ref, n);
    row("ddr", &c_madd2, cyc, n, nmse(d_y, ref, n));
#endif

#undef TIME

    printf("lab: %s the packet floor of a two-input op is 2 packets (4 cycles) per vector,\n", TARGET);
    printf("lab: %s because one packet holds one load and the op needs two.\n", TARGET);
    return 0;
}
