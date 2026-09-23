// Target fwht: an HVX fast Walsh-Hadamard transform for the KV cache rotation of llama.cpp.
//
// A quantized KV cache makes llama.cpp rotate Q and K (a 256 x 256 Hadamard matrix for Qwen3.5,
// head dimension 256) and V and the attention output (64 x 64 blocks) with a MUL_MAT of a
// constant matrix (llama_mul_mat_hadamard, src/llama-impl.h), and it marks that MUL_MAT with
// GGML_HINT_SRC0_IS_HADAMARD. The CPU, CUDA, Metal, Vulkan and SYCL backends then run a fast
// transform and never read the matrix. HTP0 runs the MUL_MAT: on the phone one decode token of
// the 2B pays 10 (K), 15 (Q), 6 (V) and 17 (output) us per attention layer, 0.29 ms per token,
// and the 4B 0.48 ms.
//
// hvx_fwht_rows transforms rows of n = 64, 128 or 256 f32 values in place, with the scale
// 1/sqrt(n) that the Hadamard matrix of llama.cpp has (ggml_gen_hadamard). The butterflies run
// in the order of the CPU backend (ggml_compute_forward_fwht_f32: the scale first, then the
// stages of length 1, 2, 4, ...), thus with IEEE float adds each value is the value of the CPU
// transform. A stage inside a vector (length 1 to 16 values) takes the partner of each lane
// with one vdelta and selects the sum or the difference with a lane mask. A stage across
// vectors (length 32 values and more) adds and subtracts whole vectors.
//
// The program measures the cycles of one call for the decode shapes of the 2B and the 4B and
// for a prefill block, and it compares each value with a scalar transform in the order of the
// CPU backend (the count of values that differ) and with a float64 transform (the largest
// error against the largest value of the row).
//
// Arguments: --rows 64 --iters 3
#include "lab.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "hvx-utils.h"

#define TARGET "fwht"

// The lane masks of the stages inside one vector: in mask s, a 32-bit lane is all ones when bit s
// of the lane index is set. And the delta-network controls that swap each lane with its partner
// at the distance 2^s lanes: a permutation i -> i XOR d takes the control d in every byte. Built
// at the start of main.
static HVX_Vector g_lane_mask[5];
static HVX_Vector g_swap_ctl[5];

// One stage of length len = 2^s (1 to 16 values) inside the vector v: lane i with bit s clear
// gets v[i] + v[i + len], lane i with bit s set gets v[i - len] - v[i]. One vdelta brings the
// partner of each lane, thus the stage is one permute, one add, one subtract and one select.
static inline HVX_Vector fwht_stage_in_vector(HVX_Vector v, int s) {
    const HVX_Vector p   = Q6_V_vdelta_VV(v, g_swap_ctl[s]);  // lane i holds v[i XOR len]
    const HVX_Vector sum = hvx_vec_add_f32_f32(v, p);
    const HVX_Vector dif = hvx_vec_sub_f32_f32(p, v);
    return Q6_V_vmux_QVV(Q6_Q_vand_VR(g_lane_mask[s], 0xffffffff), dif, sum);
}

// Transforms G rows of NV vectors (n = 32 NV values each) that start at x, all in registers.
// NV and G are compile-time constants at each call, thus the array t stays in vector registers
// and the stages of the G rows interleave in the packets.
static inline __attribute__((always_inline)) void fwht_group(float * restrict x, const uint32_t NV, const uint32_t G,
                                                             HVX_Vector scale) {
    HVX_Vector   t[16];
    HVX_Vector * v = (HVX_Vector *) x;
#pragma clang loop unroll(full)
    for (uint32_t j = 0; j < NV * G; ++j) {
        t[j] = hvx_vec_mul_f32_f32(v[j], scale);
    }
#pragma clang loop unroll(full)
    for (int s = 0; s < 5; ++s) {
#pragma clang loop unroll(full)
        for (uint32_t j = 0; j < NV * G; ++j) {
            t[j] = fwht_stage_in_vector(t[j], s);
        }
    }
#pragma clang loop unroll(full)
    for (uint32_t k = 1; k < NV; k <<= 1) {
#pragma clang loop unroll(full)
        for (uint32_t r = 0; r < G; ++r) {
#pragma clang loop unroll(full)
            for (uint32_t j = 0; j < NV; j += 2 * k) {
#pragma clang loop unroll(full)
                for (uint32_t i = j; i < j + k; ++i) {
                    const HVX_Vector a = t[r * NV + i];
                    const HVX_Vector b = t[r * NV + i + k];
                    t[r * NV + i]     = hvx_vec_add_f32_f32(a, b);
                    t[r * NV + i + k] = hvx_vec_sub_f32_f32(a, b);
                }
            }
        }
    }
#pragma clang loop unroll(full)
    for (uint32_t j = 0; j < NV * G; ++j) {
        v[j] = t[j];
    }
}

// Transforms rows of n f32 values (n = 64, 128 or 256; rows of 128-byte aligned vectors, row
// stride n values) in place with the scale 1/sqrt(n). Groups of 16 vectors (2 rows of 256, 8
// rows of 64) run together. O(rows * n * log2(n)).
static void __attribute__((noinline)) hvx_fwht_rows(float * restrict x, uint32_t n, uint32_t rows) {
    const HVX_Vector scale = hvx_vec_splat_f32(1.0f / sqrtf((float) n));
    uint32_t         r     = 0;
    if (n == 256) {
        for (; r + 2 <= rows; r += 2) {
            fwht_group(x + (size_t) r * n, 8, 2, scale);
        }
        for (; r < rows; ++r) {
            fwht_group(x + (size_t) r * n, 8, 1, scale);
        }
    } else if (n == 128) {
        for (; r + 4 <= rows; r += 4) {
            fwht_group(x + (size_t) r * n, 4, 4, scale);
        }
        for (; r < rows; ++r) {
            fwht_group(x + (size_t) r * n, 4, 1, scale);
        }
    } else {
        for (; r + 8 <= rows; r += 8) {
            fwht_group(x + (size_t) r * n, 2, 8, scale);
        }
        for (; r < rows; ++r) {
            fwht_group(x + (size_t) r * n, 2, 1, scale);
        }
    }
}

// The scalar transform in the order of ggml_compute_forward_fwht_f32 (the CPU backend).
static void fwht_ref_f32(float * x, uint32_t n) {
    const float scale = 1.0f / sqrtf((float) n);
    for (uint32_t j = 0; j < n; ++j) {
        x[j] *= scale;
    }
    for (uint32_t len = 1; len < n; len <<= 1) {
        for (uint32_t i = 0; i < n; i += 2 * len) {
            for (uint32_t j = 0; j < len; ++j) {
                const float u = x[i + j];
                const float v = x[i + len + j];
                x[i + j]       = u + v;
                x[i + len + j] = u - v;
            }
        }
    }
}

// The float64 transform, for the error bound.
static void fwht_ref_f64(double * x, uint32_t n) {
    for (uint32_t len = 1; len < n; len <<= 1) {
        for (uint32_t i = 0; i < n; i += 2 * len) {
            for (uint32_t j = 0; j < len; ++j) {
                const double u = x[i + j];
                const double v = x[i + len + j];
                x[i + j]       = u + v;
                x[i + len + j] = u - v;
            }
        }
    }
    const double scale = 1.0 / sqrt((double) n);
    for (uint32_t j = 0; j < n; ++j) {
        x[j] *= scale;
    }
}

// Times one shape and checks its values. Returns the count of values that differ from the CPU
// order.
static size_t run_shape(const char * name, uint32_t n, uint32_t rows, uint32_t iters, float * buf, float * src) {
    const size_t count = (size_t) n * rows;
    lab_fill_f32(src, count, -4.0f, 4.0f);
    uint64_t best = UINT64_MAX;
    for (uint32_t it = 0; it < iters; ++it) {
        memcpy(buf, src, count * sizeof(float));
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        hvx_fwht_rows(buf, n, rows);
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        best = t1 - t0 < best ? t1 - t0 : best;
    }
    size_t diff = 0;
    double worst = 0.0;
    float  row[256];
    double row64[256];
    for (uint32_t r = 0; r < rows; ++r) {
        double amax = 0.0;
        for (uint32_t j = 0; j < n; ++j) {
            row[j]   = src[(size_t) r * n + j];
            row64[j] = row[j];
        }
        fwht_ref_f32(row, n);
        fwht_ref_f64(row64, n);
        for (uint32_t j = 0; j < n; ++j) {
            amax = fmax(amax, fabs(row64[j]));
        }
        for (uint32_t j = 0; j < n; ++j) {
            const float got = buf[(size_t) r * n + j];
            uint32_t    a, b;
            memcpy(&a, &got, 4);
            memcpy(&b, &row[j], 4);
            diff += a != b;
            worst = fmax(worst, fabs((double) got - row64[j]) / amax);
        }
    }
    char key[96];
    snprintf(key, sizeof(key), "%s_n%u_rows%u_cycles", name, n, rows);
    lab_report(TARGET, key, (double) best, "cycles");
    snprintf(key, sizeof(key), "%s_n%u_rows%u_cycles_per_row", name, n, rows);
    lab_report(TARGET, key, (double) best / rows, "cycles");
    snprintf(key, sizeof(key), "%s_n%u_rows%u_values_not_cpu_order", name, n, rows);
    lab_report(TARGET, key, (double) diff, "");
    snprintf(key, sizeof(key), "%s_n%u_rows%u_max_err_over_amax", name, n, rows);
    lab_report(TARGET, key, worst, "");
    return diff;
}

int main(int argc, char ** argv) {
    const uint32_t rows_prefill = (uint32_t) lab_arg_long(argc, argv, "--rows", 64);
    const uint32_t iters        = (uint32_t) lab_arg_long(argc, argv, "--iters", 3);
    lab_init();

    for (int s = 0; s < 5; ++s) {
        uint32_t m[32] __attribute__((aligned(128)));
        for (int i = 0; i < 32; ++i) {
            m[i] = (i & (1 << s)) ? 0xffffffffu : 0u;
        }
        g_lane_mask[s] = *(const HVX_Vector *) m;
        uint8_t c[128] __attribute__((aligned(128)));
        memset(c, 4 << s, sizeof(c));
        g_swap_ctl[s] = *(const HVX_Vector *) c;
    }

    const size_t max_values = (size_t) 256 * (rows_prefill > 64 ? rows_prefill : 64);
    float *      buf        = lab_vtcm_alloc(max_values * sizeof(float), 128);
    float *      src        = lab_vtcm_alloc(max_values * sizeof(float), 128);

    size_t bad = 0;
    // The decode shapes of one attention layer: Q and K rows of 256 (one per head), V and the
    // output in blocks of 64 (four per head).
    bad += run_shape("q_2b", 256, 8, iters, buf, src);
    bad += run_shape("k_2b", 256, 2, iters, buf, src);
    bad += run_shape("v_2b", 64, 8, iters, buf, src);
    bad += run_shape("out_2b", 64, 32, iters, buf, src);
    bad += run_shape("q_4b", 256, 16, iters, buf, src);
    bad += run_shape("k_4b", 256, 4, iters, buf, src);
    bad += run_shape("v_4b", 64, 16, iters, buf, src);
    bad += run_shape("out_4b", 64, 64, iters, buf, src);
    // A prefill block: many rows, thus the cycles per row of a long run.
    bad += run_shape("prefill", 256, rows_prefill, iters, buf, src);
    bad += run_shape("prefill", 64, rows_prefill, iters, buf, src);

    lab_report(TARGET, "values_not_cpu_order_total", (double) bad, "");
    return 0;
}
