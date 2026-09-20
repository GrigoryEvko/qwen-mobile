// Piecewise int16 evaluators for the HVX. tools/htp-lab/gen/poly_i16.py of the qwen-mobile
// repository generates this file, do not edit it. Refer to that file for the scheme, for the
// measured map of vlut16, and for how to add a function.
//
// WHY: on the v79 a packet holds one qfloat multiply and two int16 multiplies, and a result is
// ready two packets after its producer. A transcendental as one chain of qfloat operations costs
// about four cycles for each of its steps. The same function as a piecewise polynomial in int16
// over four independent vectors costs about one cycle for each step.
//
// HOW TO CALL: give the routine 1, 2 or 4 vectors of non-negative f16 magnitudes and a constant
// count. It gives the value times 2^scale as an int16 in the lane order of the input. Four
// vectors is the number that fills the packets.

#ifndef HVX_POLY_I16_H
#define HVX_POLY_I16_H

#include <stdint.h>

#include "hvx-base.h"

// The Horner core. table[d] is the coefficient vector of v^d, e_base is e_lo + 15 (the exponent
// field of the first octave), u_max_bits is the f16 bit pattern of the largest covered value.
// degree and n are constants at the call, thus the loops unroll and the vectors stay in registers.
#define HVX_POLY_I16_EVAL(table, degree, e_base, u_max_bits, a, out, n)                            \
    do {                                                                                           \
        const HVX_Vector _m_abs  = Q6_Vh_vsplat_R(0x7fff);                                         \
        const HVX_Vector _u_max  = Q6_Vh_vsplat_R(u_max_bits);                                     \
        const HVX_Vector _e_base = Q6_Vh_vsplat_R(e_base);                                         \
        const HVX_Vector _m_mant = Q6_Vh_vsplat_R(0x03ff);                                         \
        const HVX_Vector _m_byte = Q6_Vh_vsplat_R(0x00ff);                                         \
        HVX_Vector _u[4], _idx[4], _v[4], _acc[4], _ib[2];                                         \
        const int _np = ((n) + 1) / 2;                                                             \
        for (int _r = 0; _r < (n); _r++) {                                                         \
            _u[_r] = Q6_Vh_vmin_VhVh(Q6_V_vand_VV((a)[_r], _m_abs), _u_max);                       \
        }                                                                                          \
        /* the octave number. Below the table it is negative, its byte does not match the page, */ \
        /* and every coefficient of the lookup is 0, thus the value is 0.                       */ \
        for (int _r = 0; _r < (n); _r++) {                                                         \
            _idx[_r] = Q6_Vh_vsub_VhVh(Q6_Vuh_vlsr_VuhR(_u[_r], 10), _e_base);                     \
        }                                                                                          \
        for (int _r = 0; _r < (n); _r++) {                                                         \
            _v[_r] = Q6_Vh_vasl_VhR(Q6_V_vand_VV(_u[_r], _m_mant), 5);                             \
        }                                                                                          \
        /* the low byte of a lane is an even byte and the high byte an odd byte, thus the first */ \
        /* vector of a pair comes back in lo and the second in hi                              */ \
        for (int _p = 0; _p < _np; _p++) {                                                         \
            const HVX_Vector _b = _idx[(2 * _p + 1 < (n)) ? 2 * _p + 1 : 2 * _p];                  \
            _ib[_p] = Q6_V_vor_VV(Q6_V_vand_VV(_idx[2 * _p], _m_byte), Q6_Vh_vasl_VhR(_b, 8));     \
        }                                                                                          \
        for (int _p = 0; _p < _np; _p++) {                                                         \
            const HVX_VectorPair _c = Q6_Wh_vlut16_VbVhR(_ib[_p], hvx_vmem((table)[degree]), 0);   \
            _acc[2 * _p] = Q6_V_lo_W(_c);                                                          \
            if (2 * _p + 1 < (n)) {                                                                \
                _acc[2 * _p + 1] = Q6_V_hi_W(_c);                                                  \
            }                                                                                      \
        }                                                                                          \
        for (int _d = (degree) - 1; _d >= 0; _d--) {                                               \
            HVX_Vector _co[4];                                                                     \
            for (int _p = 0; _p < _np; _p++) {                                                     \
                const HVX_VectorPair _cp = Q6_Wh_vlut16_VbVhR(_ib[_p], hvx_vmem((table)[_d]), 0);  \
                _co[2 * _p] = Q6_V_lo_W(_cp);                                                      \
                if (2 * _p + 1 < (n)) {                                                            \
                    _co[2 * _p + 1] = Q6_V_hi_W(_cp);                                              \
                }                                                                                  \
            }                                                                                      \
            for (int _r = 0; _r < (n); _r++) {                                                     \
                _acc[_r] = Q6_Vh_vmpy_VhVh_s1_rnd_sat(_acc[_r], _v[_r]);                           \
            }                                                                                      \
            for (int _r = 0; _r < (n); _r++) {                                                     \
                _acc[_r] = Q6_Vh_vadd_VhVh_sat(_acc[_r], _co[_r]);                                 \
            }                                                                                      \
        }                                                                                          \
        for (int _r = 0; _r < (n); _r++) {                                                         \
            (out)[_r] = _acc[_r];                                                                  \
        }                                                                                          \
    } while (0)

// A value of the scale 2^s as a qf32 pair, multiplied by k. The int16 becomes an f16 number, an
// unsigned saturating subtract of (s - 10) from its exponent field divides it by 2^(s-10) and
// keeps a zero a zero, and the widening multiply gives the rest of the scale, the factor k and
// the 32-bit form in one step. k is an IEEE half, thus 0x8400 is -2^-14 and 0x0400 is 2^-14.
static inline __attribute__((always_inline))
HVX_VectorPair hvx_poly_i16_to_qf32(HVX_Vector value, int exp_drop, int k_hf) {
    const HVX_Vector drop = Q6_Vh_vsplat_R(exp_drop << 10);
    const HVX_Vector k    = Q6_Vh_vsplat_R(k_hf);
    return Q6_Wqf32_vmpy_VhfVhf(Q6_Vuh_vsub_VuhVuh_sat(Q6_Vhf_equals_Vh(value), drop), k);
}


// h(u) = u / (1 + e^u), the bump of SiLU: silu(x) = relu(x) - h(|x|). Range [0, 0.2785]
// The table is 16 octaves from 2^-12, degree 4, the value times 2^16.
// Below 2^-12 the table gives 0, thus the reconstruction is exact there as well.
// Measured against float64: max 1.594e-04, rms 3.326e-05.
#define HVX_POLY_SILU_BUMP_DEGREE 4
#define HVX_POLY_SILU_BUMP_SCALE  16
#define HVX_POLY_SILU_BUMP_E_BASE 3
#define HVX_POLY_SILU_BUMP_U_MAX  0x4bff

static const int16_t hvx_poly_silu_bump_table[4 + 1][64] __attribute__((aligned(128))) = {
    // the coefficient of v^0
    { 8, 0, 16, 0, 32, 0, 64, 0, 128, 0, 255, 0, 508, 0, 1008, 0, 1984, 0, 3840, 0, 7173, 0, 12371, 0, 17625, 0, 15628, 0, 4715, 0, 173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^1
    { 8, 0, 16, 0, 32, 0, 64, 0, 128, 0, 254, 0, 504, 0, 992, 0, 1920, 0, 3585, 0, 6165, 0, 8522, 0, 4753, 0, -12093, 0, -13805, 0, -1052, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^2
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -4, 0, -16, 0, -64, 0, -254, 0, -993, 0, -3623, 0, -10021, 0, -4974, 0, 17479, 0, 2468, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^3
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 21, 0, 326, 0, 3720, 0, 9630, 0, -11149, 0, -2546, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^4
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 30, 0, -453, 0, -3479, 0, 2938, 0, 959, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

// silu_bump(|a[r]|) times 2^16 as an int16, for n vectors of 64 f16 lanes. n is 1, 2 or 4 and a
// constant at the call. The lane order of the output is the lane order of the input.
static inline __attribute__((always_inline))
void hvx_poly_silu_bump(const HVX_Vector * a, HVX_Vector * out, const int n) {
    HVX_POLY_I16_EVAL(hvx_poly_silu_bump_table, 4, HVX_POLY_SILU_BUMP_E_BASE,
                      HVX_POLY_SILU_BUMP_U_MAX, a, out, n);
}

// the table holds 0.5 - 1/(1+e^u), thus sigmoid(x) = 0.5 -+ table(|x|), the sign of the term being that of x. Range [0, 0.5]
// The table is 16 octaves from 2^-12, degree 4, the value times 2^16.
// Below 2^-12 the table gives 0, thus the reconstruction is exact there as well.
// Measured against float64: max 1.133e-04, rms 3.082e-05.
#define HVX_POLY_SIGMOID_DROP_DEGREE 4
#define HVX_POLY_SIGMOID_DROP_SCALE  16
#define HVX_POLY_SIGMOID_DROP_E_BASE 3
#define HVX_POLY_SIGMOID_DROP_U_MAX  0x4bff

static const int16_t hvx_poly_sigmoid_drop_table[4 + 1][64] __attribute__((aligned(128))) = {
    // the coefficient of v^0
    { 4, 0, 8, 0, 16, 0, 32, 0, 64, 0, 128, 0, 256, 0, 512, 0, 1024, 0, 2045, 0, 4075, 0, 8025, 0, 15143, 0, 24956, 0, 31593, 0, 32747, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^1
    { 4, 0, 8, 0, 16, 0, 32, 0, 64, 0, 128, 0, 256, 0, 512, 0, 1023, 0, 2040, 0, 4033, 0, 7701, 0, 12891, 0, 13786, 0, 4457, 0, 142, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^2
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -8, 0, -63, 0, -471, 0, -3025, 0, -10719, 0, -7401, 0, -353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^3
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -20, 0, -134, 0, -257, 0, 4295, 0, 6063, 0, 378, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^4
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 21, 0, 204, 0, -728, 0, -1968, 0, -146, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

// sigmoid_drop(|a[r]|) times 2^16 as an int16, for n vectors of 64 f16 lanes. n is 1, 2 or 4 and a
// constant at the call. The lane order of the output is the lane order of the input.
static inline __attribute__((always_inline))
void hvx_poly_sigmoid_drop(const HVX_Vector * a, HVX_Vector * out, const int n) {
    HVX_POLY_I16_EVAL(hvx_poly_sigmoid_drop_table, 4, HVX_POLY_SIGMOID_DROP_E_BASE,
                      HVX_POLY_SIGMOID_DROP_U_MAX, a, out, n);
}

// tanh(u) for u >= 0, odd: tanh(x) = sign(x) * tanh(|x|). Range [0, 1]
// The table is 16 octaves from 2^-12, degree 4, the value times 2^14.
// Below 2^-12 the table gives 0, thus the reconstruction is exact there as well.
// Measured against float64: max 2.429e-04, rms 5.994e-05.
#define HVX_POLY_TANH_HALF_DEGREE 4
#define HVX_POLY_TANH_HALF_SCALE  14
#define HVX_POLY_TANH_HALF_E_BASE 3
#define HVX_POLY_TANH_HALF_U_MAX  0x4bff

static const int16_t hvx_poly_tanh_half_table[4 + 1][64] __attribute__((aligned(128))) = {
    // the coefficient of v^0
    { 4, 0, 8, 0, 16, 0, 32, 0, 64, 0, 128, 0, 256, 0, 512, 0, 1023, 0, 2037, 0, 4013, 0, 7571, 0, 12478, 0, 15796, 0, 16373, 0, 16384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^1
    { 4, 0, 8, 0, 16, 0, 32, 0, 64, 0, 128, 0, 256, 0, 512, 0, 1020, 0, 2016, 0, 3850, 0, 6446, 0, 6893, 0, 2228, 0, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^2
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -31, 0, -235, 0, -1512, 0, -5360, 0, -3701, 0, -176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^3
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -10, 0, -67, 0, -128, 0, 2147, 0, 3032, 0, 189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^4
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 102, 0, -364, 0, -984, 0, -73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

// tanh_half(|a[r]|) times 2^14 as an int16, for n vectors of 64 f16 lanes. n is 1, 2 or 4 and a
// constant at the call. The lane order of the output is the lane order of the input.
static inline __attribute__((always_inline))
void hvx_poly_tanh_half(const HVX_Vector * a, HVX_Vector * out, const int n) {
    HVX_POLY_I16_EVAL(hvx_poly_tanh_half_table, 4, HVX_POLY_TANH_HALF_E_BASE,
                      HVX_POLY_TANH_HALF_U_MAX, a, out, n);
}

// the table holds 1 - e^-u, thus e^-u = 1 - table(u) for u >= 0, the decay of a gate. Range [0, 1]
// The table is 16 octaves from 2^-12, degree 5, the value times 2^15.
// Below 2^-12 the table gives 0, thus the reconstruction is exact there as well.
// Measured against float64: max 2.400e-04, rms 5.006e-05.
#define HVX_POLY_EXP_DROP_DEGREE 5
#define HVX_POLY_EXP_DROP_SCALE  15
#define HVX_POLY_EXP_DROP_E_BASE 3
#define HVX_POLY_EXP_DROP_U_MAX  0x4bff

static const int16_t hvx_poly_exp_drop_table[5 + 1][64] __attribute__((aligned(128))) = {
    // the coefficient of v^0
    { 8, 0, 16, 0, 32, 0, 64, 0, 128, 0, 255, 0, 508, 0, 1008, 0, 1985, 0, 3850, 0, 7248, 0, 12893, 0, 20713, 0, 28333, 0, 32168, 0, 32757, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^1
    { 8, 0, 16, 0, 32, 0, 64, 0, 128, 0, 254, 0, 504, 0, 992, 0, 1924, 0, 3615, 0, 6380, 0, 9937, 0, 12054, 0, 8863, 0, 2378, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^2
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -4, 0, -16, 0, -60, 0, -226, 0, -797, 0, -2484, 0, -6023, 0, -8799, 0, -4511, 0, -258, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^3
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 9, 0, 66, 0, 413, 0, 1989, 0, 5593, 0, 5003, 0, 417, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^4
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -51, 0, -462, 0, -2277, 0, -3098, 0, -333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    // the coefficient of v^5
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 62, 0, 454, 0, 817, 0, 104, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

// exp_drop(|a[r]|) times 2^15 as an int16, for n vectors of 64 f16 lanes. n is 1, 2 or 4 and a
// constant at the call. The lane order of the output is the lane order of the input.
static inline __attribute__((always_inline))
void hvx_poly_exp_drop(const HVX_Vector * a, HVX_Vector * out, const int n) {
    HVX_POLY_I16_EVAL(hvx_poly_exp_drop_table, 5, HVX_POLY_EXP_DROP_E_BASE,
                      HVX_POLY_EXP_DROP_U_MAX, a, out, n);
}

#endif /* HVX_POLY_I16_H */
