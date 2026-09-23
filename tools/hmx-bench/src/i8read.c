// The cost of the int32 accumulator read of the HMX int path, measured on the device.
//
// The int path multiplies a u8 activation tile (64 rows x 32 channels) by an s8 weight tile
// (32 x 32) into an int32 accumulator, at twice the multiply-accumulate rate of the f16 path. The
// accumulator leaves the HMX only as u8 byte planes: one store of 2048 bytes for each byte, with a
// conversion word in the bias register that selects the byte (tools/htp-lab/lab/target_i8probe.c).
// The full int32 needs four stores, and 24 bits need three. This program measures, in the style of our f16
// kernel (one ":deep" instruction for up to 32 k tiles), these costs on the HMX thread:
//
//   1. The f16 path: pcycles for one 32 x 32 output tile over k f16 tiles, with its one store.
//   2. The int path: pcycles for one 64 x 32 output tile over k int8 tiles, for each read variant:
//        none     no store (the next clear drops the accumulator)
//        p1..p4   1 to 4 plane stores in the row-major layout (":cm"), a bias load before each
//        p4rm     4 plane stores in the 4-row interleaved layout (no ":cm")
//        p4tab    4 plane stores whose conversion table sits at the destination,
//                 each store with :retain, the next tile starts with a clear
//        p4tablast p4tab with the last store without :retain (thus p4 with the tables at the destinations)
//        cvt4     4 conversions "cvt.ub = acc" plus 4 stores of the conversion buffer. The cvt path
//                 saturates, thus it cannot give the int32 value. It is here for its cost only.
//        swap4    two accumulators: the chain of tile c goes to one accumulator, then
//                 "mxswapacc", then the 4 plane stores of tile c - 1 from the other one. If the
//                 HMX stores one accumulator while it multiplies into the other, the read hides.
//                 "mxclracc" and a store without ":retain" clear both accumulators
//                 (tools/htp-lab/lab/target_i8probe.c), thus every store keeps the accumulator,
//                 no clear runs inside the pass, and accumulator A holds tile 0 + tile 2 + ...
//                 The epilogue gets
//                 tile c as read(c) - read(c - 2), exact in int32 arithmetic modulo 2^32.
//        swap3    swap4 with 3 planes (|acc| < 2^23)
//        p4keep, p3keep  4 or 3 plane stores, each with ":retain" (the reads of the swap passes)
//   The f16 path also runs with the swap ("f16swap"). There the stores do not give the tiles
//   (the f16 sums cannot be subtracted exactly), thus that line is a timing of the overlap only.
//   A fit over k gives the marginal cost of one multiply and the fixed cost of one read.
//   3. The HVX cost of the join of four planes into int32 and of the conversion to f32 with a row
//      scale and a column scale, for one 64 x 32 tile, on one HVX context.
//
// Before the timing, the program checks the 4-plane and the 3-plane read against a scalar
// reference on random data (k = 64 and k = 288), thus a wrong read cannot give a plausible time.
//
// Arguments: --out <path> writes every result line to a file as well. The lines also go to the log
// of the phone (printf and FARF, with a .farf mask file next to the library), where the log can drop
// lines. Each result line starts with "i8read:". The simulator build (lab target i8read, MODE=functional) runs the checks;
// its cycle numbers are 0, because the simulator does not time the HMX.
//
// Complexity: O(k_max x n_cols x reps) HMX instructions, about 2 seconds on the device.

#include "lab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HAP_farf.h"
#include "HAP_perf.h"
#include "hexagon_protos.h"
#include "hexagon_types.h"

#ifdef LAB_DEVICE
void lab_fini(void);
#else
static inline void lab_fini(void) {}
#endif

// The result file (--out), which holds every line: the log of the phone drops lines when many come
// at once. NULL writes to the log only.
static FILE * g_out;

#define LOG(...)                                  \
    do {                                          \
        printf("i8read: " __VA_ARGS__);           \
        printf("\n");                             \
        FARF(ALWAYS, "i8read: " __VA_ARGS__);     \
        if (g_out) {                              \
            fprintf(g_out, "i8read: " __VA_ARGS__); \
            fprintf(g_out, "\n");                 \
            fflush(g_out);                        \
        }                                         \
    } while (0)

#define N_COLS      8u     // output tiles of one timed pass
#define N_REPS      12u    // timed passes, the best one counts
#define K_MAX_TILES 288u   // the 9216 reduction of the down projection of the 4B, in tiles of 32
#define ACT_TILE    2048u  // bytes of one activation tile (int8: 64 x 32, f16: 32 x 32)
#define WGT_TILE_I8 1024u  // bytes of one int8 weight tile
#define WGT_TILE_HF 2048u  // bytes of one f16 weight tile
#define PLANE       2048u  // bytes of one u8 plane of a 64 x 32 accumulator

// ---- the HMX forms (target_i8probe.c of tools/htp-lab gave each one in the simulator and on the phone)

static inline void clr_i32(void) { asm volatile("mxclracc\n" ::: "memory"); }
static inline void clr_f16(void) { asm volatile("mxclracc.hf\n" ::: "memory"); }
static inline void swap_i32(void) { asm volatile("mxswapacc\n" ::: "memory"); }
static inline void swap_f16(void) { asm volatile("mxswapacc.hf\n" ::: "memory"); }

static inline void mac_i8_deep(const uint8_t * a, const uint8_t * w, uint32_t ar, uint32_t wr) {
    asm volatile("{ activation.ub = mxmem(%0, %2):deep:cm\n weight.b = mxmem(%1, %3) }\n"
                 : : "r"(a), "r"(w), "r"(ar), "r"(wr) : "memory");
}

static inline void mac_f16_deep(const uint8_t * a, const uint8_t * w, uint32_t r) {
    asm volatile("{ activation.hf = mxmem(%0, %2):deep\n weight.hf = mxmem(%1, %2) }\n"
                 : : "r"(a), "r"(w), "r"(r) : "memory");
}

static inline void bias_i32(const uint8_t * t) { asm volatile("bias = mxmem(%0)\n" : : "r"(t) : "memory"); }
static inline void bias_f16(const uint8_t * t) { asm volatile("bias = mxmem2(%0)\n" : : "r"(t) : "memory"); }

static inline void st_plane_keep(uint8_t * o) {
    asm volatile("mxmem(%0, %1):after:retain:cm.ub = acc\n" : : "r"(o), "r"(0) : "memory");
}
static inline void st_plane_last(uint8_t * o) {
    asm volatile("mxmem(%0, %1):after:cm.ub = acc\n" : : "r"(o), "r"(0) : "memory");
}
static inline void st_plane_keep_rm(uint8_t * o) {
    asm volatile("mxmem(%0, %1):after:retain.ub = acc\n" : : "r"(o), "r"(0) : "memory");
}
static inline void st_plane_last_rm(uint8_t * o) {
    asm volatile("mxmem(%0, %1):after.ub = acc\n" : : "r"(o), "r"(0) : "memory");
}
static inline void st_f16(uint8_t * o) {
    asm volatile("mxmem(%0, %1):after.hf = acc\n" : : "r"(o), "r"(0) : "memory");
}
static inline void cvt_ub_keep(void) { asm volatile("cvt.ub = acc(%0)\n" : : "r"(1) : "memory"); }
static inline void cvt_ub_last(void) { asm volatile("cvt.ub = acc(%0)\n" : : "r"(0) : "memory"); }
static inline void st_cvt(uint8_t * o) {
    asm volatile("mxmem(%0, %1):cm = cvt\n" : : "r"(o), "r"(0) : "memory");
}

// Waits for the memory transactions of this thread, the HMX stores included
static inline void sync_mem(void) { asm volatile("syncht\n" ::: "memory"); }

// ---- the areas

static struct {
    uint8_t * act;     // K_MAX_TILES activation tiles (int8 or f16, 2048 bytes each)
    uint8_t * wgt;     // N_COLS x K_MAX_TILES weight tiles (2048 bytes each, the int8 path uses 1024).
                       // Output tile c has its own K_MAX_TILES tiles, as the column blocks of a matmul.
    uint8_t * ref;     // DDR, the outputs of the plain pass that the swap pass must reproduce
    uint8_t * out;     // N_COLS x 4 planes
    uint8_t * tab;     // the 4 conversion tables of 128 bytes, then the f16 bias area of 256 bytes
    uint8_t * tabs;    // N_COLS copies of the 4 conversion tables (p4tabcopy)
    int32_t * i32;     // one joined 64 x 32 int32 tile (the HVX test)
    float *   f32;     // one 64 x 32 f32 tile (the HVX test)
} g;

// The four conversion words that select byte 0, 1, 2 and 3 of the int32 accumulator: the
// exponent field (bits 10 to 14) scales by 2^(e - 24), thus e = 24, 16, 8, 0.
static const uint32_t k_byte_words[4] = { 24u << 10, 16u << 10, 8u << 10, 0u };

// Runs the k-tile chain of one int8 output tile. Each instruction covers at most 32 tiles, because
// the range operand holds 16 bits. O(k / 32) instructions.
static inline void chain_i8(const uint8_t * a, const uint8_t * w, uint32_t k) {
    while (k >= 32) {
        mac_i8_deep(a, w, 32 * ACT_TILE - 1, 32 * WGT_TILE_I8 - 1);
        a += 32 * ACT_TILE;
        w += 32 * WGT_TILE_I8;
        k -= 32;
    }
    if (k) {
        mac_i8_deep(a, w, k * ACT_TILE - 1, k * WGT_TILE_I8 - 1);
    }
}

static inline void chain_f16(const uint8_t * a, const uint8_t * w, uint32_t k) {
    while (k >= 32) {
        mac_f16_deep(a, w, 32 * ACT_TILE - 1);
        a += 32 * ACT_TILE;
        w += 32 * WGT_TILE_HF;
        k -= 32;
    }
    if (k) {
        mac_f16_deep(a, w, k * ACT_TILE - 1);
    }
}

enum variant { V_NONE, V_P1, V_P2, V_P3, V_P4, V_P4RM, V_P4TAB, V_CVT4, V_SWAP4, V_P4KEEP, V_SWAP3, V_P3KEEP, V_P4TABLAST, V_P4TABCOPY, V_COUNT };
static const char * const k_variant_name[V_COUNT] = { "none", "p1", "p2", "p3", "p4", "p4rm", "p4tab", "cvt4",
                                                      "swap4", "p4keep", "swap3", "p3keep", "p4tablast", "p4tabcopy" };

// Reads the accumulator of one output tile with the given variant into o (4 x PLANE bytes).
static inline void read_tile(enum variant v, uint8_t * o) {
    const uint8_t * t = g.tab;
    switch (v) {
        case V_NONE:
            break;
        case V_P1:
            bias_i32(t);
            st_plane_last(o);
            break;
        case V_P2:
            bias_i32(t);
            st_plane_keep(o);
            bias_i32(t + 128);
            st_plane_last(o + PLANE);
            break;
        case V_P3:
            bias_i32(t);
            st_plane_keep(o);
            bias_i32(t + 128);
            st_plane_keep(o + PLANE);
            bias_i32(t + 256);
            st_plane_last(o + 2 * PLANE);
            break;
        case V_P4:
            bias_i32(t);
            st_plane_keep(o);
            bias_i32(t + 128);
            st_plane_keep(o + PLANE);
            bias_i32(t + 256);
            st_plane_keep(o + 2 * PLANE);
            bias_i32(t + 384);
            st_plane_last(o + 3 * PLANE);
            break;
        case V_P3KEEP:
            bias_i32(t);
            st_plane_keep(o);
            bias_i32(t + 128);
            st_plane_keep(o + PLANE);
            bias_i32(t + 256);
            st_plane_keep(o + 2 * PLANE);
            break;
        case V_P4KEEP:
            // p4 with :retain on every store (the reads of the swap passes)
            bias_i32(t);
            st_plane_keep(o);
            bias_i32(t + 128);
            st_plane_keep(o + PLANE);
            bias_i32(t + 256);
            st_plane_keep(o + 2 * PLANE);
            bias_i32(t + 384);
            st_plane_keep(o + 3 * PLANE);
            break;
        case V_P4RM:
            bias_i32(t);
            st_plane_keep_rm(o);
            bias_i32(t + 128);
            st_plane_keep_rm(o + PLANE);
            bias_i32(t + 256);
            st_plane_keep_rm(o + 2 * PLANE);
            bias_i32(t + 384);
            st_plane_last_rm(o + 3 * PLANE);
            break;
        case V_P4TAB:
            // the table of each plane is at its destination: the load reads it before the store
            // replaces it. The caller writes the tables there.
            for (uint32_t p = 0; p < 4; p++) {
                bias_i32(o + p * PLANE);
                st_plane_keep(o + p * PLANE);
            }
            break;
        case V_P4TABCOPY:
            // the tables of this output tile in their own 512 bytes, not at the destination
            for (uint32_t p = 0; p < 4; p++) {
                const uint8_t * tp = g.tabs + (size_t) (o - g.out) / (4 * PLANE) * 512 + p * 128;
                bias_i32(tp);
                if (p < 3) {
                    st_plane_keep(o + p * PLANE);
                } else {
                    st_plane_last(o + p * PLANE);
                }
            }
            break;
        case V_P4TABLAST:
            for (uint32_t p = 0; p < 4; p++) {
                bias_i32(o + p * PLANE);
                if (p < 3) {
                    st_plane_keep(o + p * PLANE);
                } else {
                    st_plane_last(o + p * PLANE);
                }
            }
            break;
        case V_CVT4:
            bias_i32(t);
            cvt_ub_keep();
            st_cvt(o);
            bias_i32(t + 128);
            cvt_ub_keep();
            st_cvt(o + PLANE);
            bias_i32(t + 256);
            cvt_ub_keep();
            st_cvt(o + 2 * PLANE);
            bias_i32(t + 384);
            cvt_ub_last();
            st_cvt(o + 3 * PLANE);
            break;
        default:
            break;
    }
}

// Writes the four conversion tables of the byte planes into the destination of each plane of each
// output tile, for V_P4TAB. O(N_COLS x 4 x 32).
static void write_tables_at_destinations(void) {
    for (uint32_t c = 0; c < N_COLS; c++) {
        for (uint32_t p = 0; p < 4; p++) {
            uint32_t * w = (uint32_t *) (g.out + (c * 4 + p) * PLANE);
            for (uint32_t i = 0; i < 32; i++) {
                w[i] = k_byte_words[p];
            }
        }
    }
    sync_mem();
}

// pcycles of one pass over N_COLS int8 output tiles with k tiles each. O(N_COLS x k).
static uint64_t pass_i8(enum variant v, uint32_t k) {
    if (v == V_P4TAB || v == V_P4TABLAST) {
        write_tables_at_destinations();
    }
    const uint64_t t0 = HAP_perf_get_pcycles();
    if (v == V_SWAP4 || v == V_SWAP3) {
        const enum variant rv = v == V_SWAP4 ? V_P4KEEP : V_P3KEEP;
        // tile c goes to the active accumulator, the swap makes the other one active, and its
        // stores read tile c - 1 plus the earlier tiles of the same accumulator (refer to swap4)
        clr_i32();
        for (uint32_t c = 0; c < N_COLS; c++) {
            chain_i8(g.act, g.wgt + (size_t) c * K_MAX_TILES * WGT_TILE_I8, k);
            swap_i32();
            if (c > 0) {
                read_tile(rv, g.out + (c - 1) * 4 * PLANE);
            }
        }
        swap_i32();
        read_tile(rv, g.out + (N_COLS - 1) * 4 * PLANE);
    } else {
        for (uint32_t c = 0; c < N_COLS; c++) {
            clr_i32();
            chain_i8(g.act, g.wgt + (size_t) c * K_MAX_TILES * WGT_TILE_I8, k);
            read_tile(v, g.out + c * 4 * PLANE);
        }
    }
    clr_i32();
    sync_mem();
    return HAP_perf_get_pcycles() - t0;
}

// pcycles of one pass over N_COLS f16 output tiles (32 x 32) with k tiles each, with or without
// the accumulator swap (refer to pass_i8). O(N_COLS x k).
static uint64_t pass_f16(uint32_t k, int swap) {
    bias_f16(g.tab + 512);
    const uint64_t t0 = HAP_perf_get_pcycles();
    if (swap) {
        clr_f16();
        for (uint32_t c = 0; c < N_COLS; c++) {
            chain_f16(g.act, g.wgt + (size_t) c * K_MAX_TILES * WGT_TILE_HF, k);
            swap_f16();
            if (c > 0) {
                st_f16(g.out + (c - 1) * ACT_TILE);
            }
        }
        swap_f16();
        st_f16(g.out + (N_COLS - 1) * ACT_TILE);
    } else {
        for (uint32_t c = 0; c < N_COLS; c++) {
            clr_f16();
            chain_f16(g.act, g.wgt + (size_t) c * K_MAX_TILES * WGT_TILE_HF, k);
            st_f16(g.out + c * ACT_TILE);
        }
    }
    sync_mem();
    return HAP_perf_get_pcycles() - t0;
}

// The best of N_REPS passes, after one warm pass
static uint64_t best_i8(enum variant v, uint32_t k) {
    uint64_t best = (uint64_t) -1;
    (void) pass_i8(v, k);
    for (uint32_t r = 0; r < N_REPS; r++) {
        const uint64_t c = pass_i8(v, k);
        best = c < best ? c : best;
    }
    return best;
}

static uint64_t best_f16(uint32_t k, int swap) {
    uint64_t best = (uint64_t) -1;
    (void) pass_f16(k, swap);
    for (uint32_t r = 0; r < N_REPS; r++) {
        const uint64_t c = pass_f16(k, swap);
        best = c < best ? c : best;
    }
    return best;
}

// ---- the correctness check against a scalar reference

// The activation tile layout of ":cm": byte r * 32 + kk. The weight tile layout: byte
// (kk / 4) * 128 + c * 4 + kk % 4.
static inline uint32_t act_off(uint32_t r, uint32_t kk) { return r * 32 + kk; }
static inline uint32_t wgt_off(uint32_t kk, uint32_t c) { return (kk / 4) * 128 + c * 4 + kk % 4; }

// Fills k tiles with random u8 activations and s8 weights, runs the chain, reads with p4 (or p3),
// and compares with the scalar int64 product. Returns the count of wrong elements. O(64 x 32 x 32 k).
static uint32_t check_read_v(uint32_t k, int planes, uint32_t amax, uint32_t wmax, enum variant v);

static uint32_t check_read(uint32_t k, int planes, uint32_t amax, uint32_t wmax) {
    return check_read_v(k, planes, amax, wmax, planes == 4 ? V_P4 : V_P3);
}

static uint32_t check_read_v(uint32_t k, int planes, uint32_t amax, uint32_t wmax, enum variant v) {
    for (uint32_t i = 0; i < k * ACT_TILE; i++) {
        g.act[i] = (uint8_t) (lab_rand_u32() % (amax + 1));
    }
    for (uint32_t i = 0; i < k * WGT_TILE_I8; i++) {
        g.wgt[i] = (uint8_t) (int8_t) ((int32_t) (lab_rand_u32() % (2 * wmax + 1)) - (int32_t) wmax);
    }
    if (v == V_P4TAB || v == V_P4TABLAST) {
        write_tables_at_destinations();
    }
    clr_i32();
    chain_i8(g.act, g.wgt, k);
    read_tile(v, g.out);
    sync_mem();

    uint32_t bad = 0;
    for (uint32_t r = 0; r < 64; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            int64_t ref = 0;
            for (uint32_t t = 0; t < k; t++) {
                const uint8_t * at = g.act + t * ACT_TILE;
                const int8_t *  wt = (const int8_t *) (g.wgt + t * WGT_TILE_I8);
                for (uint32_t kk = 0; kk < 32; kk++) {
                    ref += (int64_t) at[act_off(r, kk)] * (int64_t) wt[wgt_off(kk, c)];
                }
            }
            const uint32_t e  = r * 32 + c;
            uint32_t       u  = (uint32_t) g.out[e] | ((uint32_t) g.out[PLANE + e] << 8) | ((uint32_t) g.out[2 * PLANE + e] << 16);
            int32_t        v;
            if (planes == 4) {
                u |= (uint32_t) g.out[3 * PLANE + e] << 24;
                v = (int32_t) u;
            } else {
                v = ((int32_t) (u << 8)) >> 8;  // the sign of bit 23
            }
            if ((int64_t) v != ref) {
                if (bad < 3) {
                    LOG("check k=%u planes=%d: element (%u, %u) is %ld, the reference is %ld", (unsigned) k, planes,
                        (unsigned) r, (unsigned) c, (long) v, (long) ref);
                }
                bad++;
            }
        }
    }
    return bad;
}

// The int32 element e of the 4 planes at o
static inline uint32_t plane_word(const uint8_t * o, uint32_t e) {
    return (uint32_t) o[e] | ((uint32_t) o[PLANE + e] << 8) | ((uint32_t) o[2 * PLANE + e] << 16) |
           ((uint32_t) o[3 * PLANE + e] << 24);
}

// Runs a plain p4 pass and a swap4 pass on the same data. Tile c of the swap pass minus its tile
// c - 2 (modulo 2^32) must equal tile c of the plain pass. Returns the count of wrong elements.
// O(N_COLS x k).
static uint32_t check_swap(uint32_t k) {
    const uint32_t n = N_COLS * 4 * PLANE;
    memset(g.out, 0x55, n);
    (void) pass_i8(V_P4, k);
    memcpy(g.ref, g.out, n);
    memset(g.out, 0xAA, n);
    (void) pass_i8(V_SWAP4, k);
    uint32_t bad = 0;
    for (uint32_t c = 0; c < N_COLS; c++) {
        for (uint32_t e = 0; e < 2048; e++) {
            uint32_t v = plane_word(g.out + c * 4 * PLANE, e);
            if (c >= 2) {
                v -= plane_word(g.out + (c - 2) * 4 * PLANE, e);
            }
            bad += v != plane_word(g.ref + c * 4 * PLANE, e);
        }
    }
    return bad;
}

// ---- the HVX epilogue: 4 planes -> int32 -> f32 with scales

// Joins the four u8 planes of one 64 x 32 tile into int32 in row-major order. 16 vectors of each
// plane, 3 byte and halfword shuffles for each group of 128 elements. O(2048).
static void hvx_join4(int32_t * restrict dst, const uint8_t * restrict planes) {
    const HVX_Vector * p0 = (const HVX_Vector *) planes;
    const HVX_Vector * p1 = (const HVX_Vector *) (planes + PLANE);
    const HVX_Vector * p2 = (const HVX_Vector *) (planes + 2 * PLANE);
    const HVX_Vector * p3 = (const HVX_Vector *) (planes + 3 * PLANE);
    HVX_Vector *       d  = (HVX_Vector *) dst;
    for (uint32_t i = 0; i < PLANE / 128; i++) {
        const HVX_VectorPair lo = Q6_W_vshuff_VVR(p1[i], p0[i], -1);  // halfwords b0 | b1 << 8
        const HVX_VectorPair hi = Q6_W_vshuff_VVR(p3[i], p2[i], -1);  // halfwords b2 | b3 << 8
        const HVX_VectorPair w0 = Q6_W_vshuff_VVR(Q6_V_lo_W(hi), Q6_V_lo_W(lo), -2);
        const HVX_VectorPair w1 = Q6_W_vshuff_VVR(Q6_V_hi_W(hi), Q6_V_hi_W(lo), -2);
        d[4 * i + 0] = Q6_V_lo_W(w0);
        d[4 * i + 1] = Q6_V_hi_W(w0);
        d[4 * i + 2] = Q6_V_lo_W(w1);
        d[4 * i + 3] = Q6_V_hi_W(w1);
    }
}

int main(int argc, char ** argv) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--out") == 0) {
            g_out = fopen(argv[i + 1], "w");
        }
    }
    lab_init();

    g.act = lab_vtcm_alloc(K_MAX_TILES * ACT_TILE, 4096);
    g.wgt = lab_vtcm_alloc((size_t) N_COLS * K_MAX_TILES * WGT_TILE_HF, 4096);
    g.ref = lab_ddr_alloc(N_COLS * 4 * PLANE, 128);
    g.out = lab_vtcm_alloc(N_COLS * 4 * PLANE, 4096);
    g.tab = lab_vtcm_alloc(4096, 4096);
    g.tabs = lab_vtcm_alloc(N_COLS * 512, 4096);
    g.i32 = lab_vtcm_alloc(64 * 32 * 4, 4096);
    g.f32 = lab_vtcm_alloc(64 * 32 * 4, 4096);

    for (uint32_t p = 0; p < 4; p++) {
        uint32_t * w = (uint32_t *) (g.tab + p * 128);
        for (uint32_t i = 0; i < 32; i++) {
            w[i] = k_byte_words[p];
        }
        for (uint32_t c = 0; c < N_COLS; c++) {
            uint32_t * wc = (uint32_t *) (g.tabs + c * 512 + p * 128);
            for (uint32_t i = 0; i < 32; i++) {
                wc[i] = k_byte_words[p];
            }
        }
    }
    // the f16 bias area: scale 1.0 (0x3c00) and bias 0 for each column, then the second 128 bytes zero
    uint32_t * fb = (uint32_t *) (g.tab + 512);
    for (uint32_t i = 0; i < 32; i++) {
        fb[i]      = 0x3c00u;
        fb[32 + i] = 0;
    }
    sync_mem();

    // 1. correctness on the silicon
#ifdef LAB_DEVICE
    const uint32_t kbig = K_MAX_TILES;
#else
    const uint32_t kbig = 40;  // the scalar reference is slow in the simulator
#endif
    const uint32_t bad64  = check_read(64, 4, 255, 128);
    const uint32_t badbig = check_read(kbig, 4, 255, 128);
    const uint32_t bad3   = check_read(kbig, 3, 40, 40);  // |acc| <= 288 * 32 * 40 * 40 < 2^23
    LOG("check p4 k=64 bad %u of 2048, p4 k=%u bad %u of 2048, p3 k=%u (|acc| < 2^23) bad %u of 2048",
        (unsigned) bad64, (unsigned) kbig, (unsigned) badbig, (unsigned) kbig, (unsigned) bad3);
    LOG("check p4tab k=%u bad %u of 2048, p4tablast bad %u, p4tabcopy bad %u", (unsigned) kbig,
        (unsigned) check_read_v(kbig, 4, 255, 128, V_P4TAB), (unsigned) check_read_v(kbig, 4, 255, 128, V_P4TABLAST),
        (unsigned) check_read_v(kbig, 4, 255, 128, V_P4TABCOPY));

    // the timed data: any bytes, the rate does not depend on the values
    // the byte mask keeps each f16 value finite and normal or zero
    for (uint32_t i = 0; i < K_MAX_TILES * ACT_TILE; i++) {
        g.act[i] = (uint8_t) ((i * 7u + 1u) & 0x3F);
    }
    for (uint32_t i = 0; i < N_COLS * K_MAX_TILES * WGT_TILE_HF; i++) {
        g.wgt[i] = (uint8_t) ((i * 13u + 3u) & 0x3F);  // small f16 values and int8 weights
    }
    sync_mem();

    // the swap passes must give the bytes of the plain passes
    LOG("check swap4 (tile c = read c - read c-2) k=80 bad %u of %u", (unsigned) check_swap(80),
        (unsigned) (N_COLS * 2048));

    // the HVX join must give the int32 of the planes
    {
        (void) pass_i8(V_P4, 64);
        hvx_join4(g.i32, g.out);
        uint32_t bad = 0;
        for (uint32_t e = 0; e < 2048; e++) {
            const uint32_t u = (uint32_t) g.out[e] | ((uint32_t) g.out[PLANE + e] << 8) |
                               ((uint32_t) g.out[2 * PLANE + e] << 16) | ((uint32_t) g.out[3 * PLANE + e] << 24);
            bad += (uint32_t) g.i32[e] != u;
        }
        LOG("check hvx join4 bad %u of 2048", (unsigned) bad);
    }

    // 2. the rates. The simulator does not time the HMX, thus its build stops here.
#ifndef LAB_DEVICE
    LOG("done (the simulator build runs the checks only)");
    return 0;
#endif
    static const uint32_t ks[] = { 1, 2, 4, 8, 16, 32, 48, 64, 80, 160, 288 };
    const uint32_t        nk   = sizeof(ks) / sizeof(ks[0]);
    for (int swap = 0; swap < 2; swap++) {
        for (uint32_t i = 0; i < nk; i++) {
            const uint64_t t = best_f16(ks[i], swap);
            LOG("%s k=%u pass=%llu per_tile_x100=%llu", swap ? "f16swap" : "f16", (unsigned) ks[i],
                (unsigned long long) t, (unsigned long long) (t * 100 / N_COLS));
        }
    }
    for (int v = 0; v < V_COUNT; v++) {
        for (uint32_t i = 0; i < nk; i++) {
            const uint64_t t = best_i8((enum variant) v, ks[i]);
            LOG("i8 %s k=%u pass=%llu per_tile_x100=%llu", k_variant_name[v], (unsigned) ks[i],
                (unsigned long long) t, (unsigned long long) (t * 100 / N_COLS));
        }
    }

    // 3. the HVX join of one tile, one context
    {
        (void) pass_i8(V_P4, 64);
        uint64_t best = (uint64_t) -1;
        for (uint32_t r = 0; r < N_REPS; r++) {
            const uint64_t t0 = HAP_perf_get_pcycles();
            for (uint32_t c = 0; c < N_COLS; c++) {
                hvx_join4(g.i32, g.out + c * 4 * PLANE);
            }
            const uint64_t t = HAP_perf_get_pcycles() - t0;
            best = t < best ? t : best;
        }
        LOG("hvx join4 per_tile_x100=%llu", (unsigned long long) (best * 100 / N_COLS));
    }

    lab_fini();
    LOG("done");
    if (g_out) {
        fclose(g_out);
    }
    return 0;
}
