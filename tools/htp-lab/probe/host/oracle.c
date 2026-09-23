// The CPU oracle of the HVX census. Refer to oracle.h for the rules.
//
// Compile this file with -ffp-contract=off and without fast math (CMakeLists.txt does).
//
// The lane layouts of the ops with a vector pair are those that tools/htp-lab/isa/compare.py
// finds on the simulator (census.csv, the "layout" column), for example "widen_eo" for
// Vdd.sf = vmpy(Vu.hf, Vv.hf): the low vector holds the products of the even input lanes, the high
// vector those of the odd lanes. test/oracle-sim-test.sh checks each layout against the simulator.

#include "oracle.h"

#include <math.h>
#include <string.h>

#include "isa_kernels.h"

// The operations.
enum {
    K_COPY = 1,  // r = a
    K_ABS,       // r = |a|
    K_NEG,       // r = -a
    K_ADD,       // r = a + b
    K_SUB,       // r = a - b
    K_MUL,       // r = a * b
    K_MAX,       // r = max(a, b)
    K_MIN,       // r = min(a, b)
    K_MUL_ADD,   // t = a * b, r = t + c (also x += a * b with x in c)
    K_ADD_ADD,   // t = a + b, r = t + c
    K_MUL_MUL,   // t = a * b, r = t * c
    K_DMPY,      // r = a0 * b0 + a1 * b1
    K_DMPY_ACC,  // d = a0 * b0 + a1 * b1, r = c + d
    K_CVT,       // r = a in the output format
    K_CMP_GT,    // r = a > b, one byte for each predicate bit
    K_CMP_EQ,    // r = a == b
    K_EXP2,      // r = exp2f(a)
};

// The lane layouts. o is the output lane in one iteration, L_out the output lanes of one iteration.
enum {
    L_ELEM = 1,     // input lane o
    L_SCALAR_B,     // input a lane o, input b lane 0 (the scalar register of the op)
    L_WIDEN_EO,     // input lane 2 (o mod L_out/2) + o div (L_out/2): even lanes, then odd lanes
    L_WIDEN_CONTIG, // input lane o
    L_WIDEN8_H,     // byte lanes of vdeal.h then vunpack: 4 (t div 2) + t mod 2 + 2 (o div (L_out/2))
    L_NARROW_UV,    // input a (o even) or b (o odd), lane o div 2
    L_NARROW_CONTIG,// input a lanes, then input b lanes
    L_DMPY,         // input lanes 2 o and 2 o + 1, accumulator lane o
};

// The predicate update of a compare.
enum { P_NONE = 0, P_AND, P_OR, P_XOR };

struct table_entry {
    const char * name;  // the intrinsic (after "ieee." or "cc.") or the full "seq." name
    int          kind;
    int          layout;
    int          pred;
};

static const struct table_entry k_table[] = {
    // Unary, one lane to one lane
    { "Q6_Vhf_vabs_Vhf", K_ABS, L_ELEM, P_NONE },
    { "Q6_Vsf_vabs_Vsf", K_ABS, L_ELEM, P_NONE },
    { "Q6_Vqf16_vabs_Vhf", K_ABS, L_ELEM, P_NONE },
    { "Q6_Vqf32_vabs_Vsf", K_ABS, L_ELEM, P_NONE },
    { "Q6_Vhf_vfneg_Vhf", K_NEG, L_ELEM, P_NONE },
    { "Q6_Vsf_vfneg_Vsf", K_NEG, L_ELEM, P_NONE },
    { "Q6_Vqf16_vneg_Vhf", K_NEG, L_ELEM, P_NONE },
    { "Q6_Vqf32_vneg_Vsf", K_NEG, L_ELEM, P_NONE },
    { "Q6_Vqf16_equals_Vhf", K_COPY, L_ELEM, P_NONE },
    { "Q6_Vqf32_equals_Vsf", K_COPY, L_ELEM, P_NONE },
    { "seq.qfpair.sf_roundtrip", K_COPY, L_ELEM, P_NONE },
    { "seq.qfpair.hf_roundtrip_qf16", K_COPY, L_ELEM, P_NONE },
    { "seq.hvx_vec_exp2_f16", K_EXP2, L_ELEM, P_NONE },

    // Binary, one lane to one lane
    { "Q6_Vhf_vadd_VhfVhf", K_ADD, L_ELEM, P_NONE },
    { "Q6_Vsf_vadd_VsfVsf", K_ADD, L_ELEM, P_NONE },
    { "Q6_Vqf16_vadd_VhfVhf", K_ADD, L_ELEM, P_NONE },
    { "Q6_Vqf32_vadd_VsfVsf", K_ADD, L_ELEM, P_NONE },
    { "seq.hvx_vec_add_f32_f32", K_ADD, L_ELEM, P_NONE },
    { "seq.hvx_vec_add_f16_f16", K_ADD, L_ELEM, P_NONE },
    { "seq.qfpair.add_f32", K_ADD, L_ELEM, P_NONE },
    { "seq.qfpair.add_f16_qf16", K_ADD, L_ELEM, P_NONE },
    { "seq.qfext.add_f32_via_mem", K_ADD, L_ELEM, P_NONE },
    { "Q6_Vhf_vsub_VhfVhf", K_SUB, L_ELEM, P_NONE },
    { "Q6_Vsf_vsub_VsfVsf", K_SUB, L_ELEM, P_NONE },
    { "Q6_Vqf16_vsub_VhfVhf", K_SUB, L_ELEM, P_NONE },
    { "Q6_Vqf32_vsub_VsfVsf", K_SUB, L_ELEM, P_NONE },
    { "seq.hvx_vec_sub_f32_f32", K_SUB, L_ELEM, P_NONE },
    { "seq.hvx_vec_sub_f16_f16", K_SUB, L_ELEM, P_NONE },
    { "seq.qfpair.sub_f32", K_SUB, L_ELEM, P_NONE },
    { "seq.qfpair.sub_f16_qf16", K_SUB, L_ELEM, P_NONE },
    { "seq.qfext.sub_f16_via_mem", K_SUB, L_ELEM, P_NONE },
    { "Q6_Vhf_vmpy_VhfVhf", K_MUL, L_ELEM, P_NONE },
    { "Q6_Vsf_vmpy_VsfVsf", K_MUL, L_ELEM, P_NONE },
    { "Q6_Vqf16_vmpy_VhfVhf", K_MUL, L_ELEM, P_NONE },
    { "Q6_Vqf32_vmpy_VsfVsf", K_MUL, L_ELEM, P_NONE },
    { "Q6_Vqf32_vmpy_VsfRsf", K_MUL, L_SCALAR_B, P_NONE },
    { "seq.hvx_vec_mul_f32_f32", K_MUL, L_ELEM, P_NONE },
    { "seq.hvx_vec_mul_f16_f16", K_MUL, L_ELEM, P_NONE },
    { "seq.qfpair.mul_f32", K_MUL, L_ELEM, P_NONE },
    { "seq.qfpair.mul_f16_wqf32", K_MUL, L_ELEM, P_NONE },
    { "seq.qfpair.mul_f16_qf16", K_MUL, L_ELEM, P_NONE },
    { "seq.qfext.mul_f32_via_mem", K_MUL, L_ELEM, P_NONE },
    { "seq.qfext.mul_f16_via_mem", K_MUL, L_ELEM, P_NONE },
    { "Q6_Vhf_vfmax_VhfVhf", K_MAX, L_ELEM, P_NONE },
    { "Q6_Vsf_vfmax_VsfVsf", K_MAX, L_ELEM, P_NONE },
    { "Q6_Vhf_vmax_VhfVhf", K_MAX, L_ELEM, P_NONE },
    { "Q6_Vsf_vmax_VsfVsf", K_MAX, L_ELEM, P_NONE },
    { "Q6_Vbf_vmax_VbfVbf", K_MAX, L_ELEM, P_NONE },
    { "Q6_Vhf_vfmin_VhfVhf", K_MIN, L_ELEM, P_NONE },
    { "Q6_Vsf_vfmin_VsfVsf", K_MIN, L_ELEM, P_NONE },
    { "Q6_Vhf_vmin_VhfVhf", K_MIN, L_ELEM, P_NONE },
    { "Q6_Vsf_vmin_VsfVsf", K_MIN, L_ELEM, P_NONE },
    { "Q6_Vbf_vmin_VbfVbf", K_MIN, L_ELEM, P_NONE },

    // Chains of two operations, one lane to one lane
    { "Q6_Vhf_vmpyacc_VhfVhfVhf", K_MUL_ADD, L_ELEM, P_NONE },
    { "seq.chain.mul_add_f32", K_MUL_ADD, L_ELEM, P_NONE },
    { "seq.chain.mul_add_f16", K_MUL_ADD, L_ELEM, P_NONE },
    { "seq.chain.qf_mul_cvt_add_f32", K_MUL_ADD, L_ELEM, P_NONE },
    { "seq.chain.qf_mul_cvt_add_f32_barrier", K_MUL_ADD, L_ELEM, P_NONE },
    { "seq.chain.add_add_f32", K_ADD_ADD, L_ELEM, P_NONE },
    { "seq.chain.mul_mul_f32", K_MUL_MUL, L_ELEM, P_NONE },

    // Conversions, one lane to one lane
    { "Q6_Vh_vcvt_Vhf", K_CVT, L_ELEM, P_NONE },
    { "Q6_Vuh_vcvt_Vhf", K_CVT, L_ELEM, P_NONE },
    { "Q6_Vh_equals_Vhf", K_CVT, L_ELEM, P_NONE },
    { "Q6_Vh_equals_Vhf_rnd", K_CVT, L_ELEM, P_NONE },
    { "seq.hvx_vec_i16_from_hf_rnd_sat", K_CVT, L_ELEM, P_NONE },
    { "Q6_Vhf_vcvt_Vh", K_CVT, L_ELEM, P_NONE },
    { "Q6_Vhf_vcvt_Vuh", K_CVT, L_ELEM, P_NONE },
    { "Q6_Vhf_equals_Vh", K_CVT, L_ELEM, P_NONE },
    { "Q6_Vsf_equals_Vw", K_CVT, L_ELEM, P_NONE },
    { "Q6_Vw_equals_Vsf", K_CVT, L_ELEM, P_NONE },

    // Widening ops: one vector to a vector pair
    { "Q6_Wsf_vadd_VhfVhf", K_ADD, L_WIDEN_EO, P_NONE },
    { "Q6_Wsf_vadd_VbfVbf", K_ADD, L_WIDEN_EO, P_NONE },
    { "Q6_Wsf_vsub_VhfVhf", K_SUB, L_WIDEN_EO, P_NONE },
    { "Q6_Wsf_vsub_VbfVbf", K_SUB, L_WIDEN_EO, P_NONE },
    { "Q6_Wsf_vmpy_VhfVhf", K_MUL, L_WIDEN_EO, P_NONE },
    { "Q6_Wsf_vmpy_VbfVbf", K_MUL, L_WIDEN_EO, P_NONE },
    { "Q6_Wqf32_vmpy_VhfVhf", K_MUL, L_WIDEN_EO, P_NONE },
    { "Q6_Wsf_vmpyacc_WsfVhfVhf", K_MUL_ADD, L_WIDEN_EO, P_NONE },
    { "Q6_Wsf_vmpyacc_WsfVbfVbf", K_MUL_ADD, L_WIDEN_EO, P_NONE },
    { "seq.hvx_vec_mpyacc_f32_f16", K_MUL_ADD, L_WIDEN_EO, P_NONE },
    { "Q6_Wsf_vcvt_Vhf", K_CVT, L_WIDEN_EO, P_NONE },
    { "seq.hvx_vec_f16_to_f32_shuff", K_CVT, L_WIDEN_EO, P_NONE },
    { "seq.qfpair.f16_to_f32", K_CVT, L_WIDEN_EO, P_NONE },
    { "seq.hvx_vec_f16_to_f32", K_CVT, L_WIDEN_CONTIG, P_NONE },
    { "Q6_Whf_vcvt_Vb", K_CVT, L_WIDEN8_H, P_NONE },
    { "Q6_Whf_vcvt_Vub", K_CVT, L_WIDEN8_H, P_NONE },

    // Narrowing ops: two vectors to one vector
    { "Q6_Vhf_vcvt_VsfVsf", K_CVT, L_NARROW_UV, P_NONE },
    { "Q6_Vbf_vcvt_VsfVsf", K_CVT, L_NARROW_UV, P_NONE },
    { "Q6_Vb_vcvt_VhfVhf", K_CVT, L_NARROW_UV, P_NONE },
    { "Q6_Vub_vcvt_VhfVhf", K_CVT, L_NARROW_UV, P_NONE },
    { "seq.hvx_vec_f32_to_f16_shuff", K_CVT, L_NARROW_UV, P_NONE },
    { "seq.qfpair.f32_to_f16_vadd0", K_CVT, L_NARROW_UV, P_NONE },
    { "seq.qfpair.f32_to_f16_equals", K_CVT, L_NARROW_UV, P_NONE },
    { "seq.qfpair.f32_to_f16_mpy1", K_CVT, L_NARROW_UV, P_NONE },
    { "seq.hvx_vec_f32_to_f16", K_CVT, L_NARROW_CONTIG, P_NONE },

    // Two-lane dot products
    { "Q6_Vsf_vdmpy_VhfVhf", K_DMPY, L_DMPY, P_NONE },
    { "Q6_Vsf_vdmpyacc_VsfVhfVhf", K_DMPY_ACC, L_DMPY, P_NONE },

    // Compares: one output byte for each predicate bit
    { "Q6_Q_vcmp_gt_VhfVhf", K_CMP_GT, L_ELEM, P_NONE },
    { "Q6_Q_vcmp_gt_VsfVsf", K_CMP_GT, L_ELEM, P_NONE },
    { "Q6_Q_vcmp_gt_VbfVbf", K_CMP_GT, L_ELEM, P_NONE },
    { "Q6_Q_vcmp_gtand_QVhfVhf", K_CMP_GT, L_ELEM, P_AND },
    { "Q6_Q_vcmp_gtand_QVsfVsf", K_CMP_GT, L_ELEM, P_AND },
    { "Q6_Q_vcmp_gtand_QVbfVbf", K_CMP_GT, L_ELEM, P_AND },
    { "Q6_Q_vcmp_gtor_QVhfVhf", K_CMP_GT, L_ELEM, P_OR },
    { "Q6_Q_vcmp_gtor_QVsfVsf", K_CMP_GT, L_ELEM, P_OR },
    { "Q6_Q_vcmp_gtor_QVbfVbf", K_CMP_GT, L_ELEM, P_OR },
    { "Q6_Q_vcmp_gtxacc_QVhfVhf", K_CMP_GT, L_ELEM, P_XOR },
    { "Q6_Q_vcmp_gtxacc_QVsfVsf", K_CMP_GT, L_ELEM, P_XOR },
    { "Q6_Q_vcmp_gtxacc_QVbfVbf", K_CMP_GT, L_ELEM, P_XOR },
    { "Q6_Q_vcmp_eq_VhfVhf", K_CMP_EQ, L_ELEM, P_NONE },
    { "Q6_Q_vcmp_eq_VsfVsf", K_CMP_EQ, L_ELEM, P_NONE },
    { "Q6_Q_vcmp_eqand_QVhfVhf", K_CMP_EQ, L_ELEM, P_AND },
    { "Q6_Q_vcmp_eqand_QVsfVsf", K_CMP_EQ, L_ELEM, P_AND },
    { "Q6_Q_vcmp_eqor_QVhfVhf", K_CMP_EQ, L_ELEM, P_OR },
    { "Q6_Q_vcmp_eqor_QVsfVsf", K_CMP_EQ, L_ELEM, P_OR },
    { "Q6_Q_vcmp_eqxacc_QVhfVhf", K_CMP_EQ, L_ELEM, P_XOR },
    { "Q6_Q_vcmp_eqxacc_QVsfVsf", K_CMP_EQ, L_ELEM, P_XOR },
};

#define N_TABLE (sizeof(k_table) / sizeof(k_table[0]))

// Return the bytes of one lane of the type, 0 for a type that the oracle does not read or write.
static size_t lane_bytes(uint32_t t) {
    switch (t) {
        case ISA_T_SF: case ISA_T_W: case ISA_T_UW: return 4;
        case ISA_T_HF: case ISA_T_BF: case ISA_T_H: case ISA_T_UH: return 2;
        case ISA_T_B: case ISA_T_UB: case ISA_T_PRED: return 1;
        default: return 0;
    }
}

static int is_float(uint32_t t) {
    return t == ISA_T_SF || t == ISA_T_HF || t == ISA_T_BF;
}

uint32_t oracle_out_type(uint32_t out_type) {
    return out_type == ISA_T_QF32 ? (uint32_t) ISA_T_SF : out_type == ISA_T_QF16 ? (uint32_t) ISA_T_HF : out_type;
}

// The number of inputs that each operation reads, and the input that holds the accumulator or
// the earlier predicate (-1 for none).
static int n_inputs(int kind, int layout, int pred) {
    if (pred != P_NONE) {
        return 3;
    }
    switch (kind) {
        case K_COPY: case K_ABS: case K_NEG: case K_EXP2: return 1;
        case K_CVT: return layout == L_NARROW_UV || layout == L_NARROW_CONTIG ? 2 : 1;
        case K_MUL_ADD: case K_ADD_ADD: case K_MUL_MUL: case K_DMPY_ACC: return 3;
        default: return 2;
    }
}

// The input lane of output lane o for input j. l_out and l_in are the lanes of one iteration.
// Returns the input slot in *slot (for the narrowing layouts, else j).
static size_t in_lane(int layout, int kind, int j, size_t o, size_t l_out, size_t l_in, int * slot) {
    *slot = j;
    const int acc = (kind == K_MUL_ADD || kind == K_ADD_ADD || kind == K_MUL_MUL || kind == K_DMPY_ACC) && j == 2;
    if (acc) {
        return o;  // the accumulator has the layout of the output
    }
    switch (layout) {
        case L_SCALAR_B: return j == 1 ? 0 : o;
        case L_WIDEN_EO: return 2 * (o % (l_out / 2)) + o / (l_out / 2);
        case L_WIDEN8_H: {
            const size_t t = o % (l_out / 2);
            return 4 * (t / 2) + t % 2 + 2 * (o / (l_out / 2));
        }
        case L_NARROW_UV: *slot = (int) (o % 2); return o / 2;
        case L_NARROW_CONTIG: *slot = (int) (o / l_in); return o % l_in;
        case L_DMPY: return 2 * o;  // and 2 o + 1
        default: return o;          // L_ELEM, L_WIDEN_CONTIG
    }
}

// The lanes of one iteration of an input, measured in the lanes of the value (for a compare,
// the output lanes are bytes and each value lane covers lane_bytes bytes).
static size_t out_lanes(const struct oracle_op * op) {
    const size_t b = lane_bytes(oracle_out_type(op->out_type));
    return b ? op->out_bytes / b : 0;
}

int oracle_find(const struct oracle_op * op, struct oracle_spec * spec) {
    const char * base = op->name;
    if (strncmp(base, "ieee.", 5) == 0) {
        base += 5;
    } else if (strncmp(base, "cc.", 3) == 0) {
        base += 3;
    }
    const struct table_entry * e = NULL;
    for (size_t k = 0; k < N_TABLE; k++) {
        if (strcmp(k_table[k].name, base) == 0) {
            e = &k_table[k];
            break;
        }
    }
    if (e == NULL) {
        return 0;
    }

    // The types must be IEEE or integer, and each lane that the layout reads must exist.
    const uint32_t out_t = oracle_out_type(op->out_type);
    const size_t   l_out = out_lanes(op);
    const int      cmp   = e->kind == K_CMP_GT || e->kind == K_CMP_EQ;
    // An integer output comes only from a compare (bytes) or a conversion.
    if (l_out == 0 || (cmp && out_t != ISA_T_PRED) || (!cmp && e->kind != K_CVT && !is_float(out_t)) ||
        (e->kind == K_CVT && out_t == ISA_T_PRED)) {
        return 0;
    }
    const int n_in = n_inputs(e->kind, e->layout, e->pred);
    for (int j = 0; j < n_in; j++) {
        const uint32_t t  = op->in_type[j];
        const size_t   lb = lane_bytes(t);
        if (lb == 0 || op->in_bytes[j] == 0 || (t == ISA_T_PRED) != (cmp && j == 2)) {
            return 0;
        }
        if (!cmp && !is_float(t) && !(e->kind == K_CVT && is_float(out_t))) {
            return 0;  // an integer input is only the source of an integer to float conversion
        }
        const size_t l_in = op->in_bytes[j] / lb;
        for (size_t o = 0; o < l_out; o++) {
            int          slot;
            const size_t v    = cmp && j < 2 ? o / lb : o;  // a compare reads value lane o / lane_bytes
            const size_t lane = in_lane(e->layout, e->kind, j, v, l_out, l_in, &slot);
            const size_t last = lane + (e->layout == L_DMPY && !(e->kind == K_DMPY_ACC && j == 2) ? 1 : 0);
            if (last >= l_in || slot >= n_in) {
                return 0;
            }
        }
    }
    spec->kind   = e->kind;
    spec->layout = e->layout;
    spec->pred   = e->pred;
    return 1;
}

// ---- Lane access -------------------------------------------------------------------------------

static float bf_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float          f;
    memcpy(&f, &u, 4);
    return f;
}

// Round an f32 value to bf16 with round to nearest even. A NaN stays a quiet NaN.
static uint16_t f32_to_bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t) ((u >> 16) | 0x40u);
    }
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// The f32 value of a float lane (exact for sf, hf and bf).
static float get_f(uint32_t t, const uint8_t * p, size_t lane) {
    if (t == ISA_T_SF) {
        float f;
        memcpy(&f, p + 4 * lane, 4);
        return f;
    }
    uint16_t h;
    memcpy(&h, p + 2 * lane, 2);
    if (t == ISA_T_BF) {
        return bf_to_f32(h);
    }
    _Float16 x;
    memcpy(&x, &h, 2);
    return (float) x;
}

// The value of an integer lane.
static int64_t get_i(uint32_t t, const uint8_t * p, size_t lane) {
    switch (t) {
        case ISA_T_B:  return (int8_t) p[lane];
        case ISA_T_UB: return p[lane];
        case ISA_T_PRED: return p[lane] & 1;
        case ISA_T_H:  { int16_t v;  memcpy(&v, p + 2 * lane, 2); return v; }
        case ISA_T_UH: { uint16_t v; memcpy(&v, p + 2 * lane, 2); return v; }
        case ISA_T_W:  { int32_t v;  memcpy(&v, p + 4 * lane, 4); return v; }
        default:       { uint32_t v; memcpy(&v, p + 4 * lane, 4); return v; }
    }
}

// Round an f32 value to the float format t and give it back as f32 (exact for sf).
static float round_to(uint32_t t, float x) {
    if (t == ISA_T_HF) {
        return (float) (_Float16) x;
    }
    if (t == ISA_T_BF) {
        return bf_to_f32(f32_to_bf(x));
    }
    return x;
}

// Store an f32 value in the float format t, with round to nearest even.
static void put_f(uint32_t t, uint8_t * p, size_t lane, float x) {
    if (t == ISA_T_SF) {
        memcpy(p + 4 * lane, &x, 4);
    } else if (t == ISA_T_BF) {
        const uint16_t b = f32_to_bf(x);
        memcpy(p + 2 * lane, &b, 2);
    } else {
        const _Float16 h = (_Float16) x;
        memcpy(p + 2 * lane, &h, 2);
    }
}

// Store an integer value in the integer type t. The value is in the range of t.
static void put_i(uint32_t t, uint8_t * p, size_t lane, int64_t v) {
    switch (t) {
        case ISA_T_B: case ISA_T_UB: case ISA_T_PRED: p[lane] = (uint8_t) v; break;
        case ISA_T_H: case ISA_T_UH: { const uint16_t u = (uint16_t) v; memcpy(p + 2 * lane, &u, 2); break; }
        default: { const uint32_t u = (uint32_t) v; memcpy(p + 4 * lane, &u, 4); break; }
    }
}

// Convert an f32 value to the integer type t: round to nearest even, saturate, NaN to 0.
static int64_t to_int(uint32_t t, float x) {
    if (x != x) {
        return 0;
    }
    const float r = rintf(x);
    float lo, hi;  // the range of t, both exact in f32 except the top of w and uw
    switch (t) {
        case ISA_T_B:  lo = -128.0f;   hi = 127.0f;   break;
        case ISA_T_UB: lo = 0.0f;      hi = 255.0f;   break;
        case ISA_T_H:  lo = -32768.0f; hi = 32767.0f; break;
        case ISA_T_UH: lo = 0.0f;      hi = 65535.0f; break;
        case ISA_T_W:
            if (r >= 2147483648.0f) return INT32_MAX;
            if (r < -2147483648.0f) return INT32_MIN;
            return (int64_t) r;
        default:
            if (r >= 4294967296.0f) return UINT32_MAX;
            if (r < 0.0f) return 0;
            return (int64_t) r;
    }
    return (int64_t) (r < lo ? lo : r > hi ? hi : r);
}

// max and min: the number before a NaN, +0 before -0 (max), -0 before +0 (min).
static float max_num(float a, float b) {
    if (a != a) return b;
    if (b != b) return a;
    if (a == b) return signbit(a) ? b : a;
    return a > b ? a : b;
}

static float min_num(float a, float b) {
    if (a != a) return b;
    if (b != b) return a;
    if (a == b) return signbit(a) ? a : b;
    return a < b ? a : b;
}

// ---- The computation ---------------------------------------------------------------------------

void oracle_run(const struct oracle_op * op, const struct oracle_spec * spec, const uint8_t * const in[3], uint8_t * out) {
    const uint32_t out_t = oracle_out_type(op->out_type);
    const size_t   l_out = out_lanes(op);
    const int      n_in  = n_inputs(spec->kind, spec->layout, spec->pred);
    const int      cmp   = spec->kind == K_CMP_GT || spec->kind == K_CMP_EQ;

    size_t l_in[3] = { 0, 0, 0 };
    for (int j = 0; j < n_in; j++) {
        l_in[j] = op->in_bytes[j] / lane_bytes(op->in_type[j]);
    }

    for (uint32_t i = 0; i < op->n_vectors; i++) {
        const uint8_t * c[3];
        for (int j = 0; j < 3; j++) {
            c[j] = j < n_in ? in[j] + (size_t) i * op->in_bytes[j] : NULL;
        }
        uint8_t * po = out + (size_t) i * op->out_bytes;

        for (size_t o = 0; o < l_out; o++) {
            if (cmp) {
                const size_t v  = o / lane_bytes(op->in_type[0]);
                int          s;
                const float  a  = get_f(op->in_type[0], c[0], in_lane(spec->layout, spec->kind, 0, v, l_out, l_in[0], &s));
                const float  b  = get_f(op->in_type[1], c[1], in_lane(spec->layout, spec->kind, 1, v, l_out, l_in[1], &s));
                int          r  = spec->kind == K_CMP_GT ? a > b : a == b;
                const int    q  = spec->pred != P_NONE ? (int) get_i(ISA_T_PRED, c[2], o) : 0;
                r = spec->pred == P_AND ? (q & r) : spec->pred == P_OR ? (q | r) : spec->pred == P_XOR ? (q ^ r) : r;
                put_i(ISA_T_PRED, po, o, r);
                continue;
            }

            // The source lanes: for a conversion from two vectors the slot comes from the layout.
            int          slot0;
            const size_t lane0 = in_lane(spec->layout, spec->kind, 0, o, l_out, l_in[0], &slot0);
            const uint32_t t0  = op->in_type[slot0];

            if (spec->kind == K_CVT) {
                if (is_float(t0)) {
                    const float a = get_f(t0, c[slot0], lane0);
                    if (is_float(out_t)) {
                        put_f(out_t, po, o, a);
                    } else {
                        put_i(out_t, po, o, to_int(out_t, a));
                    }
                } else {
                    // An integer to float conversion: the C conversion rounds to nearest even.
                    const int64_t v = get_i(t0, c[slot0], lane0);
                    put_f(out_t, po, o, (float) v);
                }
                continue;
            }

            const float a = get_f(t0, c[slot0], lane0);
            float       b = 0.0f, x = 0.0f;
            if (n_in >= 2) {
                int          s1;
                const size_t lane1 = in_lane(spec->layout, spec->kind, 1, o, l_out, l_in[1], &s1);
                b                  = get_f(op->in_type[1], c[1], lane1);
            }
            if (n_in == 3) {
                int          s2;
                const size_t lane2 = in_lane(spec->layout, spec->kind, 2, o, l_out, l_in[2], &s2);
                x                  = get_f(op->in_type[2], c[2], lane2);
            }

            float r;
            switch (spec->kind) {
                case K_COPY: r = a; break;
                case K_ABS:  r = fabsf(a); break;
                case K_NEG:  r = -a; break;
                case K_EXP2: r = exp2f(a); break;
                case K_ADD:  r = a + b; break;
                case K_SUB:  r = a - b; break;
                case K_MUL:  r = a * b; break;
                case K_MAX:  r = max_num(a, b); break;
                case K_MIN:  r = min_num(a, b); break;
                case K_MUL_ADD: {
                    const float t = round_to(out_t, a * b);
                    r = t + x;
                    break;
                }
                case K_ADD_ADD: {
                    const float t = round_to(out_t, a + b);
                    r = t + x;
                    break;
                }
                case K_MUL_MUL: {
                    const float t = round_to(out_t, a * b);
                    r = t * x;
                    break;
                }
                case K_DMPY:
                case K_DMPY_ACC: {
                    // lanes 2 o and 2 o + 1: the products of two f16 values are exact in f32
                    const float a1 = get_f(op->in_type[0], c[0], lane0 + 1);
                    int         s1;
                    const size_t lane1 = in_lane(spec->layout, spec->kind, 1, o, l_out, l_in[1], &s1);
                    const float b1 = get_f(op->in_type[1], c[1], lane1 + 1);
                    const float p0 = a * b;
                    const float p1 = a1 * b1;
                    const float d  = p0 + p1;
                    r = spec->kind == K_DMPY ? d : x + d;
                    break;
                }
                default: r = a; break;
            }
            put_f(out_t, po, o, r);
        }
    }
}

const char * oracle_kind_name(const struct oracle_spec * spec) {
    static const char * const names[] = { "", "copy", "abs", "neg", "add", "sub", "mul", "max", "min", "mul_add",
                                          "add_add", "mul_mul", "dmpy", "dmpy_acc", "cvt", "cmp_gt", "cmp_eq", "exp2" };
    return spec->kind > 0 && spec->kind < (int) (sizeof(names) / sizeof(names[0])) ? names[spec->kind] : "?";
}
