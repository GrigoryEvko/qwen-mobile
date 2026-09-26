// The int8 HMX inline asm macros that tools/htp-lab/lab/target_hmxisa.c uses. This header is part of the
// lab, not of the backend.
//
// Where the macros come from. Each form comes from a probe of hexagon-llvm-mc at -mcpu=hexagonv79
// -mattr=+hvxv79,+hvx-length128b,+hmx, and each one also occurs in the Qualcomm HexKL micro library for the
// same core. Do not change a suffix or the order of two suffixes without a new probe: ":deep:cm" assembles
// and ":cm:deep" does not.
//
// Why the backend does not hold them. The int8 path is a no-go by measurement. The engine does 2.05 times
// the multiply-accumulates of the f16 path for each issue, but the accumulator cannot hold a scale, thus the
// weight needs one scale for each output row and the activation one scale for each token. With that format
// the model gives 3 to 4 times the KL floor, also with the Hadamard rotations, and the gain is 1.28 times
// the prefill at most. The f16 GEMMs stay. The census of this target keeps the int path because it records
// what the silicon does.
//
// Include this header after hmx-utils.h. It removes a definition of the same name, if a kernel tree holds
// one, and gives its own. Thus the target compiles the same text with each kernel tree.
//
// The facts that the macros depend on:
//   - The activation is unsigned and the weight is signed. No other pair of int types assembles for v79.
//   - One int8 tile is 64 activation rows by 32 channels against a 32 by 32 weight tile.
//   - The int32 accumulator has no f16 store and no f32 store. A u8 store writes one byte of each element,
//     thus the full int32 value is four stores of ":after:retain:cm.ub", each with its own conversion word
//     in the bias register. ":retain" keeps the accumulator for the next store, thus the last store has no
//     ":retain". "mxclracc" and a store without ":retain" clear the accumulator.
//   - The conversion word of one column is 32 bits, and "bias = mxmem" loads 32 of them. Bits 0 to 9 are a
//     mantissa m and bits 10 to 14 an exponent e: the scale is (1 + m / 1024) * 2^(e - 24). Bit 15 is the
//     sign, bits 16 to 18 select an activation function (0 is none), and bits 19 to 30 are a u8 bias with 4
//     fraction bits. The word is not an f16 value. With m = 0, no sign, no activation and no bias, a u8
//     store writes floor(acc * 2^(e - 24)) modulo 256.
#ifndef LAB_HMX_I8_H
#define LAB_HMX_I8_H

#undef HMX_I8_ACT_RANGE_ONE_TILE
#undef HMX_I8_WT_RANGE_ONE_TILE
#undef HMX_LOAD_MPY_I8
#undef HMX_LOAD_MPY_DEEP_I8
#undef HMX_LOAD_MPY_DEEP_I4
#undef HMX_CLRACC_I32
#undef HMX_SET_BIAS_I32
#undef HMX_STORE_PLANE_I32
#undef HMX_STORE_PLANE_LAST_I32
#undef HMX_I32_READ_SCALES

// The range operand of the int8 activation load. The vendor library uses 0x1f for one tile of 32 channels.
#define HMX_I8_ACT_RANGE_ONE_TILE 0x1fu

// The range operand of the int8 weight load for one 32 by 32 tile
#define HMX_I8_WT_RANGE_ONE_TILE 0x380u

// Multiplies one u8 activation tile by one s8 weight tile into the int32 accumulator. The activation load is
// channel major, which the int path needs.
#define HMX_LOAD_MPY_I8(act, wt, act_range, wt_range)        \
    "{\n"                                                    \
    "    activation.ub = mxmem(" act ", " act_range "):cm\n" \
    "    weight.b = mxmem(" wt ", " wt_range ")\n"           \
    "}\n"

// The same multiplication with the deep flag, which chains the k tiles of one output tile without a store
// between them
#define HMX_LOAD_MPY_DEEP_I8(act, wt, act_range, wt_range)        \
    "{\n"                                                         \
    "    activation.ub = mxmem(" act ", " act_range "):deep:cm\n" \
    "    weight.b = mxmem(" wt ", " wt_range ")\n"                \
    "}\n"

// The same multiplication with a 4-bit weight tile. Two weights share one byte.
#define HMX_LOAD_MPY_DEEP_I4(act, wt, act_range, wt_range)        \
    "{\n"                                                         \
    "    activation.ub = mxmem(" act ", " act_range "):deep:cm\n" \
    "    weight.n = mxmem(" wt ", " wt_range ")\n"                \
    "}\n"

// Clears the int32 accumulator
#define HMX_CLRACC_I32() "mxclracc\n"

// Loads the output conversion table of the int path. The table is one word for each output channel, not
// two, thus this is mxmem and not mxmem2.
#define HMX_SET_BIAS_I32(scales) "bias = mxmem(" scales ")\n"

// Stores one byte plane of the int32 accumulator and keeps the accumulator
#define HMX_STORE_PLANE_I32(out, range) "mxmem(" out ", " range "):after:retain:cm.ub = acc\n"

// Stores the last byte plane of the int32 accumulator and releases it
#define HMX_STORE_PLANE_LAST_I32(out, range) "mxmem(" out ", " range "):after:cm.ub = acc\n"

// The four conversion words that read byte 0, 1, 2 and 3 of the int32 accumulator. Their exponent fields
// (bits 10 to 14) are 24, 16, 8 and 0, thus their scales are 1, 2^-8, 2^-16 and 2^-24.
#define HMX_I32_READ_SCALES { 0x00006000u, 0x00004000u, 0x00002000u, 0x00000000u }

#endif
