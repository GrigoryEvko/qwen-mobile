// The scalar ggml reference of the Q8_0 matrix-vector product. Refer to q8_0_ref.h.
//
// Each function between the "verbatim" lines is a copy of the named file and lines of
// third_party/llama.cpp at commit c6824a9. The macros of ggml that the copies use are defined
// here with the same meaning:
//   MAX                    ggml/src/ggml-impl.h:40
//   UNUSED                 ggml/include/ggml.h (a cast to void)
//   GGML_FP32_TO_FP16      ggml/src/ggml-impl.h:449, ggml_compute_fp32_to_fp16 (ggml-base)
//   GGML_CPU_FP16_TO_FP32  ggml/src/ggml-cpu/simd-mappings.h:44 (an exact conversion on every path)

#include "q8_0_ref.h"

#include <assert.h>
#include <math.h>
#include <string.h>

#if defined(__hexagon__)
// The DSP runtime gives roundf, as it gives floorf and expf to the llama.cpp library. A weak
// reference keeps the library loadable on a DSP image without it: candidates.c then refuses
// kernel 0 instead.
#    pragma weak roundf
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define UNUSED(x) (void) (x)
#define GGML_FP32_TO_FP16(x)     ggml_compute_fp32_to_fp16(x)
#define GGML_CPU_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)

// ---- verbatim: ggml/src/ggml-impl.h:378-443 ----
static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}
// ---- end verbatim ----

// ---- verbatim: ggml/src/ggml-quants.c:276-298 ----
void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}
// ---- end verbatim ----

// ---- verbatim: ggml/src/ggml-cpu/quants.c:451-480 ----
void ggml_vec_dot_q8_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}
// ---- end verbatim ----

float ggml_ref_fp16_to_fp32(ggml_fp16_t h) {
    return ggml_compute_fp16_to_fp32(h);
}

ggml_fp16_t ggml_ref_fp32_to_fp16(float f) {
    return ggml_compute_fp32_to_fp16(f);
}

void q8_0_ref_matvec(const block_q8_0 * w, const float * x, float * y, block_q8_0 * q, uint32_t rows, uint32_t cols) {
    const size_t blocks = cols / QK8_0;
    quantize_row_q8_0_ref(x, q, (int64_t) cols);
    for (uint32_t r = 0; r < rows; r++) {
        ggml_vec_dot_q8_0_q8_0_generic((int) cols, &y[r], 0, w + (size_t) r * blocks, 0, q, 0, 1);
    }
}
