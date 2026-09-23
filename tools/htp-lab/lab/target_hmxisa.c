// Target hmxisa: the census of the HMX arithmetic on Hexagon v73, v75, v79 and v81.
//
// The program runs the HMX instruction forms of the kernels (the macros of hmx-utils.h) on input
// tiles that integer code makes. Thus the input bytes are the same on each version. Each case
// writes one file "<case>.bin" in the run directory and one line "lab: hmxisa <case> <crc32>" on
// stdout. The CRC-32 covers the output bytes only. tools/htp-lab/hmxisa/compare.py reads the files
// of the four versions, compares them, and compares each result with the exact rational value.
//
// The file format (little endian). A header of 64 bytes (struct case_hdr), then the bias area,
// then the A section, then the W section, then the output section.
//   kind 1 (f16): A is the logical activation, u16 [m][k]. W is the logical weight, u16 [k][n].
//                 The output is u16 [n_snap][m][n], the stored f16 bits in logical order.
//   kind 2 (u8 activation, s8 weight) and kind 3 (u8 activation, s4 weight): A and W are the raw
//                 tile bytes as the HMX reads them. The output is the raw bytes that the u8 stores
//                 wrote: [reads][4][2048]. The layouts are in the comments of i8_act_off and
//                 i8_wgt_off, and compare.py decodes them.
//   kind 4 (the source data of a Q8_0 matmul): refer to q8_src_make.
//
// The f16 tile layout is the layout of hmx-mm-kernels-tiled.h: element (i, j) of a 32x32 tile is at
// byte (i / 2) * 128 + j * 4 + (i % 2) * 2. The activation tile holds rows x k, the weight tile
// holds k x columns, and the output tile holds rows x columns.
//
// The simulator runs the HMX in functional mode only (refer to run.sh). Run with MODE=functional.
// The default core of "--mv81" (v81dgb_1) and some v73 cores have an HMX without the f16 path:
// on those cores each f16 case gives zeros and no exception. Refer to tools/htp-lab/hmxisa/run.sh.
//
// Arguments:
//   --only <prefix>   Run only the cases whose names start with the prefix
//   --list            Print the case names and stop
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"

#include "lab.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hmx-utils.h"

#define TILE_BYTES    2048u
#define I8_ACT_BYTES  2048u  // one u8 activation tile: 64 rows x 32 channels
#define I8_WGT_BYTES  1024u  // one s8 weight tile: 32 channels x 32 columns
#define I4_WGT_BYTES  512u   // one s4 weight tile: 32 channels x 32 columns
#define I8_ROWS       64u
#define I8_PLANE      2048u  // one u8 store of the 64 x 32 int32 accumulator

// The VTCM arena. v81 cores have 2 MB of VTCM or more, thus the arena stays below 1.9 MB.
#define ARENA_ACT     (640u * 1024u)
#define ARENA_WGT     (576u * 1024u)
#define ARENA_OUT     (512u * 1024u)
#define ARENA_CFG     (32u * 1024u)

enum case_kind {
    KIND_F16    = 1,
    KIND_I8     = 2,
    KIND_I4     = 3,
    KIND_Q8_SRC = 4,  // the source data of a Q8_0 matmul (refer to q8_src_write)
};

// The accumulation methods. The comment names the kernel that uses each one.
enum method {
    M_DEEP      = 1,  // core_dot_chunk_fp16: one ":deep" load for each 32 k tiles (matmul)
    M_DEEP_TILE = 2,  // one ":deep" load for each k tile
    M_FLAT      = 3,  // one load without ":deep" for each k tile (hmx_fa_qk_dot_tile)
    M_GDN2      = 4,  // two ":deep" loads, first half and second half of k (gdn_ch_hmx_run)
    M_RETAIN    = 5,  // M_DEEP_TILE with a ":retain" store after each k tile (census only)
    M_NOCLEAR   = 6,  // M_DEEP after a store of other data, with no clear between (census only)
    M_BLOCK     = 7,  // for each k tile: clear, one load, store (one sum for each 32-k block)
    M_I8_SINGLE = 10, // one u8 x s8 load for each k tile, no ":deep" (HMX_LOAD_MPY_I8)
    M_I8_DEEP   = 11, // one ":deep:cm" load for each 32 k tiles (HMX_LOAD_MPY_DEEP_I8)
    M_I8_REPEAT = 12, // tile pair 0 param0 times, then tile pair 1 param1 times, no ":deep"
    M_I4_DEEP   = 13, // one ":deep:cm" load with s4 weights for each 32 k tiles (HMX_LOAD_MPY_DEEP_I4)
    M_I8_BLOCK  = 14, // for each k tile: clear, one u8 x s8 load, the four-plane read (one int32 sum
                      // for each 32-k block, the read that a Q8_0 matmul needs)
};

// The file header. 64 bytes.
struct case_hdr {
    char     magic[8];     // "HMXISA1"
    uint32_t kind;         // enum case_kind
    uint32_t method;       // enum method
    uint32_t m;            // rows
    uint32_t k;            // inner dimension
    uint32_t n;            // columns
    uint32_t bias_bytes;   // bytes of the bias area
    uint32_t a_bytes;      // bytes of the A section
    uint32_t w_bytes;      // bytes of the W section
    uint32_t out_bytes;    // bytes of the output section
    uint32_t n_snap;       // number of output snapshots
    uint32_t param[3];     // method parameters (refer to the method)
    uint32_t usr;          // the USR value during the HMX instructions
};

// ---- the common state of all cases

static struct {
    uint8_t *    act;       // VTCM, activation tiles
    uint8_t *    wgt;       // VTCM, weight tiles
    uint8_t *    out;       // VTCM, output tiles
    uint8_t *    cfg;       // VTCM, bias area and conversion tables
    uint16_t *   a_log;     // DDR, logical activation
    uint16_t *   w_log;     // DDR, logical weight
    uint8_t *    res;       // DDR, the output section of the current case
    uint32_t     bias[64];  // the f16 bias area of the current case (256 bytes)
    uint32_t     usr_set;   // the bits that run_f16 sets in USR (0: USR stays as it is)
    uint32_t     usr_mask;  // the USR field that usr_set replaces
    const char * only;
    bool         list;
    uint32_t     n_cases;
} g;

static uint32_t g_crc_table[256];

// ---- small helpers

// Makes the CRC-32 table (reflected polynomial 0xEDB88320).
static void crc_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int b = 0; b < 8; b++) {
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        g_crc_table[i] = c;
    }
}

// Returns the CRC-32 of n bytes. O(n).
static uint32_t crc32_bytes(const uint8_t * p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        c = g_crc_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

// The random generator of the cases. Each case seeds it from a name, thus the data of a case does
// not change when other cases are added or removed.
static uint32_t g_rng;

static void rng_seed(const char * name) {
    uint32_t h = 2166136261u;
    for (const char * p = name; *p; p++) {
        h = (h ^ (uint8_t) *p) * 16777619u;
    }
    g_rng = h ? h : 0x9E3779B9u;
}

static uint32_t rng_u32(void) {
    uint32_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return x;
}

// Returns a value in [0, n). The bias of the modulo is not important here.
static uint32_t rng_below(uint32_t n) {
    return rng_u32() % n;
}

// The f16 bits of (-1)^neg * 2^e for e in [-24, 15]. Values below 2^-14 are subnormal.
static uint16_t f16_pow2(int e, int neg) {
    if (e < -24 || e > 15) {
        printf("lab: error: f16_pow2(%d) is out of range\n", e);
        exit(2);
    }
    const uint16_t bits = (e >= -14) ? (uint16_t) ((e + 15) << 10) : (uint16_t) (1u << (e + 24));
    return (uint16_t) (bits | (neg ? 0x8000u : 0u));
}

// The f16 bits of (-1)^neg * (1 + mant / 1024) * 2^e for a normal e in [-14, 15]
static uint16_t f16_make(int neg, int e, uint32_t mant) {
    return (uint16_t) ((neg ? 0x8000u : 0u) | ((uint32_t) (e + 15) << 10) | (mant & 0x3FFu));
}

// A random finite f16 value with a random sign and an exponent in [elo, ehi] (normal range)
static uint16_t f16_rand(int elo, int ehi) {
    const int e = elo + (int) rng_below((uint32_t) (ehi - elo + 1));
    return f16_make((int) (rng_u32() & 1), e, rng_u32() & 0x3FF);
}

// The byte offset of element (i, j) in a 32x32 f16 tile
static inline size_t tile_off(uint32_t i, uint32_t j) {
    return (size_t) (i / 2) * 128 + (size_t) j * 4 + (i % 2) * 2;
}

static uint32_t usr_read(void) {
    uint32_t v;
    __asm__ volatile("%0 = usr" : "=r"(v));
    return v;
}

static void usr_write(uint32_t v) {
    __asm__ volatile("usr = %0" : : "r"(v) : "memory");
}

// Returns true if the case runs. Prints the name for --list.
static bool case_selected(const char * name) {
    if (g.list) {
        printf("lab: hmxisa case %s\n", name);
        return false;
    }
    if (g.only && strncmp(name, g.only, strlen(g.only)) != 0) {
        return false;
    }
    g.n_cases++;
    return true;
}

// Writes the file of one case and prints the CRC line. Stops the program if the write fails,
// because a case without its file makes the comparison incomplete.
static void case_write(const char * name, const struct case_hdr * h, const void * bias, const void * a,
                       const void * w, const void * out) {
    char path[160];
    snprintf(path, sizeof(path), "%s.bin", name);
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        printf("lab: error: cannot open %s for writing\n", path);
        exit(3);
    }
    bool ok = fwrite(h, sizeof(*h), 1, f) == 1;
    ok = ok && (h->bias_bytes == 0 || fwrite(bias, h->bias_bytes, 1, f) == 1);
    ok = ok && (h->a_bytes == 0 || fwrite(a, h->a_bytes, 1, f) == 1);
    ok = ok && (h->w_bytes == 0 || fwrite(w, h->w_bytes, 1, f) == 1);
    ok = ok && (h->out_bytes == 0 || fwrite(out, h->out_bytes, 1, f) == 1);
    if (fclose(f) != 0 || !ok) {
        printf("lab: error: the write of %s failed\n", path);
        exit(3);
    }
    printf("lab: hmxisa %s %08lx\n", name, (unsigned long) crc32_bytes((const uint8_t *) out, h->out_bytes));
}

static void hdr_init(struct case_hdr * h, uint32_t kind, uint32_t method, uint32_t m, uint32_t k, uint32_t n) {
    memset(h, 0, sizeof(*h));
    memcpy(h->magic, "HMXISA1", 8);
    h->kind   = kind;
    h->method = method;
    h->m      = m;
    h->k      = k;
    h->n      = n;
    h->n_snap = 1;
}

// ---- the f16 bias area

// Sets the 256-byte bias area: word c of the first 128 bytes is lo | hi << 16, and word c of the
// second 128 bytes is v1, for all 32 columns. The kernels use lo = 0x3c00 (1.0), hi = 0 and v1 = 0
// (hmx_init_column_scales). flash-attn-ops.c uses lo = hi = the f16 of the qk scale.
static void bias_set(uint16_t lo, uint16_t hi, uint32_t v1) {
    for (int c = 0; c < 32; c++) {
        g.bias[c]      = (uint32_t) lo | ((uint32_t) hi << 16);
        g.bias[32 + c] = v1;
    }
}

// ---- the f16 path

static inline uint16_t * A_at(uint32_t k, uint32_t r, uint32_t kk) {
    return &g.a_log[(size_t) r * k + kk];
}

static inline uint16_t * W_at(uint32_t n, uint32_t kk, uint32_t c) {
    return &g.w_log[(size_t) kk * n + c];
}

// Sets the logical A [m][k] and W [k][n] to +0
static void f16_zero(uint32_t m, uint32_t k, uint32_t n) {
    memset(g.a_log, 0, (size_t) m * k * 2);
    memset(g.w_log, 0, (size_t) k * n * 2);
}

// Packs the logical A [m][k] and W [k][n] into the tile layout of core_dot_chunk_fp16: the k tiles
// of one row block follow each other, and the k tiles of one column block follow each other.
// O(m k + k n).
static void pack_f16(uint32_t m, uint32_t k, uint32_t n) {
    const uint32_t n_kt = k / 32;
    for (uint32_t r = 0; r < m; r++) {
        for (uint32_t kk = 0; kk < k; kk++) {
            const size_t tile = (size_t) (r / 32) * n_kt + kk / 32;
            memcpy(g.act + tile * TILE_BYTES + tile_off(r % 32, kk % 32), A_at(k, r, kk), 2);
        }
    }
    for (uint32_t kk = 0; kk < k; kk++) {
        for (uint32_t c = 0; c < n; c++) {
            const size_t tile = (size_t) (c / 32) * n_kt + kk / 32;
            memcpy(g.wgt + tile * TILE_BYTES + tile_off(kk % 32, c % 32), W_at(n, kk, c), 2);
        }
    }
}

// Runs the k tiles of one output tile with the given method. The accumulator is not cleared here.
static void f16_chain(const uint8_t * a, const uint8_t * w, uint32_t n_kt, uint32_t method) {
    switch (method) {
        case M_DEEP:
        case M_NOCLEAR:
            // the loop of core_dot_chunk_fp16 and core_dot_chunk_fp16_short
            while (n_kt >= 32) {
                asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(a), "r"(w) : "memory");
                a += 32 * TILE_BYTES;
                w += 32 * TILE_BYTES;
                n_kt -= 32;
            }
            if (n_kt > 0) {
                const uint32_t range = TILE_BYTES * n_kt - 1;
                asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(a), "r"(w) : "memory");
            }
            break;
        case M_DEEP_TILE:
            for (uint32_t t = 0; t < n_kt; t++) {
                asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(2047), "r"(a), "r"(w) : "memory");
                a += TILE_BYTES;
                w += TILE_BYTES;
            }
            break;
        case M_FLAT:
            for (uint32_t t = 0; t < n_kt; t++) {
                asm volatile(HMX_LOAD_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(a), "r"(w) : "memory");
                a += TILE_BYTES;
                w += TILE_BYTES;
            }
            break;
        case M_GDN2: {
            // gdn_ch_hmx_run: two ":deep" loads into one accumulator. Each part has at most 32 tiles.
            const uint32_t n1 = n_kt / 2;
            const uint32_t n2 = n_kt - n1;
            if (n1 > 0) {
                asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(TILE_BYTES * n1 - 1), "r"(a), "r"(w) : "memory");
            }
            asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0")
                         : : "r"(TILE_BYTES * n2 - 1), "r"(a + (size_t) n1 * TILE_BYTES), "r"(w + (size_t) n1 * TILE_BYTES)
                         : "memory");
            break;
        }
        default:
            printf("lab: error: f16_chain does not know method %u\n", (unsigned) method);
            exit(2);
    }
}

// Stores the f16 accumulator to one output tile and releases the accumulator
static inline void f16_store(uint8_t * out) {
    asm volatile(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(out), "r"(0) : "memory");
}

// Stores the f16 accumulator to one output tile and keeps the accumulator
static inline void f16_store_retain(uint8_t * out) {
    asm volatile("mxmem(%0, %1):after:retain.hf = acc\n" : : "r"(out), "r"(0) : "memory");
}

static inline void f16_clear(void) {
    asm volatile(HMX_CLRACC_F16() : : : "memory");
}

static inline void f16_set_bias(const uint8_t * cfg) {
    asm volatile(HMX_SET_BIAS("%0") : : "r"(cfg) : "memory");
}

// Copies one output tile into the logical output [m][n] (u16) at row block rb and column block cb
static void unpack_f16_tile(uint16_t * dst, uint32_t n, uint32_t rb, uint32_t cb, const uint8_t * tile) {
    for (uint32_t i = 0; i < 32; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            memcpy(&dst[(size_t) (rb * 32 + i) * n + cb * 32 + j], tile + tile_off(i, j), 2);
        }
    }
}

// Runs one f16 case: A [m][k] and W [k][n] are in g.a_log and g.w_log, the bias area is in g.bias.
// The output tiles come in the order of core_dot_chunk_fp16: for each row block, for each column
// block, clear, chain, store. O(m n k).
static void run_f16(const char * name, uint32_t m, uint32_t k, uint32_t n, uint32_t method) {
    const uint32_t n_kt   = k / 32;
    const uint32_t n_rb   = m / 32;
    const uint32_t n_cb   = n / 32;
    const uint32_t n_snap = (method == M_RETAIN) ? n_kt + 1 : (method == M_BLOCK) ? n_kt : 1;
    const bool bad_shape  = m % 32 || k % 32 || n % 32 || m == 0 || k == 0 || n == 0 ||
                           (size_t) m * k * 2 > ARENA_ACT || (size_t) k * n * 2 > ARENA_WGT ||
                           (size_t) m * n * 2 * n_snap > ARENA_OUT || (method == M_GDN2 && n_kt > 64);
    if (bad_shape) {
        printf("lab: error: case %s has a bad shape %u x %u x %u\n", name, (unsigned) m, (unsigned) k, (unsigned) n);
        exit(2);
    }
    pack_f16(m, k, n);
    memcpy(g.cfg, g.bias, sizeof(g.bias));

    struct case_hdr h;
    hdr_init(&h, KIND_F16, method, m, k, n);
    h.n_snap = n_snap;
    const size_t snap_elems = (size_t) m * n;
    uint16_t *   res        = (uint16_t *) g.res;
    memset(g.out, 0x55, (size_t) m * n * 2 * n_snap);

    const uint32_t usr_old = usr_read();
    const uint32_t usr_run = g.usr_mask ? ((usr_old & ~g.usr_mask) | g.usr_set) : usr_old;
    h.usr = usr_run;

    LAB_BARRIER();
    usr_write(usr_run);
    f16_set_bias(g.cfg);
    for (uint32_t rb = 0; rb < n_rb; rb++) {
        for (uint32_t cb = 0; cb < n_cb; cb++) {
            const uint8_t * a = g.act + (size_t) rb * n_kt * TILE_BYTES;
            const uint8_t * w = g.wgt + (size_t) cb * n_kt * TILE_BYTES;
            uint8_t *       o = g.out + (size_t) (rb * n_cb + cb) * TILE_BYTES;
            if (method == M_RETAIN) {
                // Snapshot t holds the accumulator after k tile t. The last one is the final store.
                f16_clear();
                for (uint32_t t = 0; t < n_kt; t++) {
                    asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0")
                                 : : "r"(2047), "r"(a + (size_t) t * TILE_BYTES), "r"(w + (size_t) t * TILE_BYTES)
                                 : "memory");
                    f16_store_retain(o + (size_t) t * n_rb * n_cb * TILE_BYTES);
                }
                f16_store(o + (size_t) n_kt * n_rb * n_cb * TILE_BYTES);
            } else if (method == M_BLOCK) {
                // Snapshot t holds the sum of k tile t alone
                for (uint32_t t = 0; t < n_kt; t++) {
                    f16_clear();
                    asm volatile(HMX_LOAD_MPY_F16("%1", "%2", "%0")
                                 : : "r"(2047), "r"(a + (size_t) t * TILE_BYTES), "r"(w + (size_t) t * TILE_BYTES)
                                 : "memory");
                    f16_store(o + (size_t) t * n_rb * n_cb * TILE_BYTES);
                }
            } else if (method == M_NOCLEAR) {
                // The first chain uses the weight of an other column block, thus it leaves a nonzero
                // accumulator. The store without ":retain" must release it before the second chain.
                const uint8_t * w_other = g.wgt + (size_t) ((cb + 1) % n_cb) * n_kt * TILE_BYTES;
                const uint8_t * a_other = g.act + (size_t) ((rb + 1) % n_rb) * n_kt * TILE_BYTES;
                f16_clear();
                f16_chain(a_other, w_other, n_kt, M_DEEP);
                f16_store(o);
                f16_chain(a, w, n_kt, M_DEEP);
                f16_store(o);
            } else {
                f16_clear();
                f16_chain(a, w, n_kt, method);
                f16_store(o);
            }
        }
    }
    usr_write(usr_old);
    LAB_BARRIER();

    for (uint32_t s = 0; s < n_snap; s++) {
        for (uint32_t rb = 0; rb < n_rb; rb++) {
            for (uint32_t cb = 0; cb < n_cb; cb++) {
                const uint8_t * o = g.out + ((size_t) s * n_rb * n_cb + rb * n_cb + cb) * TILE_BYTES;
                unpack_f16_tile(res + s * snap_elems, n, rb, cb, o);
            }
        }
    }
    h.bias_bytes = sizeof(g.bias);
    h.a_bytes    = m * k * 2;
    h.w_bytes    = k * n * 2;
    h.out_bytes  = (uint32_t) (snap_elems * 2 * n_snap);
    case_write(name, &h, g.bias, g.a_log, g.w_log, res);
}

// ---- the f16 cases

// The 32 values of the special tables: zeros, subnormals, the normal limits, simple values, the
// maximum, Inf and NaN. Each value is an f16 bit pattern.
static const uint16_t k_special[32] = {
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0200, 0x0400, 0x3c00, 0xbc00, 0x3c01, 0x3fff,
    0x3555, 0x3800, 0x4200, 0x5c00, 0x2000, 0x0c00, 0x7800, 0x7bff, 0xfbff, 0x7bfe, 0x7c00,
    0xfc00, 0x7e00, 0x7c01, 0x7fff, 0xfe00, 0x6c00, 0x1400, 0x5bff, 0x4000, 0xc000,
};

// Every finite f16 value (63488 values) through an identity weight scaled by 2^s. The scale 2^12
// makes large values overflow, and the scale 2^-12 makes small values subnormal or zero, thus the
// output conversion must round.
static void cases_pass(void) {
    static const struct { const char * name; int s; } v[] = {
        { "pass_s0", 0 }, { "pass_sp12", 12 }, { "pass_sm12", -12 },
    };
    for (size_t i = 0; i < sizeof(v) / sizeof(v[0]); i++) {
        if (!case_selected(v[i].name)) {
            continue;
        }
        const uint32_t m = 1984, k = 32, n = 32;
        f16_zero(m, k, n);
        for (uint32_t idx = 0; idx < m * 32; idx++) {
            *A_at(k, idx / 32, idx % 32) = (uint16_t) (idx < 31744 ? idx : 0x8000u + (idx - 31744));
        }
        for (uint32_t c = 0; c < n; c++) {
            *W_at(n, c, c) = f16_pow2(v[i].s, 0);
        }
        bias_set(0x3c00, 0, 0);
        run_f16(v[i].name, m, k, n, M_DEEP);
    }
}

// The product table S[r] x S[c] and the sum table S[r] + S[c] of the special values
static void cases_special(void) {
    const uint32_t m = 32, k = 32, n = 32;
    if (case_selected("special_mul")) {
        f16_zero(m, k, n);
        for (uint32_t i = 0; i < 32; i++) {
            *A_at(k, i, 0) = k_special[i];
            *W_at(n, 0, i) = k_special[i];
        }
        bias_set(0x3c00, 0, 0);
        run_f16("special_mul", m, k, n, M_DEEP);
    }
    if (case_selected("special_add")) {
        f16_zero(m, k, n);
        for (uint32_t i = 0; i < 32; i++) {
            *A_at(k, i, 0) = k_special[i];
            *A_at(k, i, 1) = 0x3c00;
            *W_at(n, 0, i) = 0x3c00;
            *W_at(n, 1, i) = k_special[i];
        }
        bias_set(0x3c00, 0, 0);
        run_f16("special_add", m, k, n, M_DEEP);
    }
}

// The sign of zero results. Rows 0 to 15 and columns 0 to 15 hold +0 and -0 in 16 sign patterns,
// thus each product is a signed zero. Rows 16 to 31 hold x and -x in the first two channels, and
// columns 16 to 31 hold 1.0 there, thus the sum is an exact cancellation.
static void cases_zero(void) {
    if (!case_selected("zero_sign")) {
        return;
    }
    static const uint32_t pat[16] = {
        0x00000000u, 0xFFFFFFFFu, 0x55555555u, 0xAAAAAAAAu, 0x00000001u, 0x80000000u, 0xFFFFFFFEu, 0x7FFFFFFFu,
        0x0000FFFFu, 0xFFFF0000u, 0x000000FFu, 0xFF000000u, 0x0F0F0F0Fu, 0x00000100u, 0xFFFFFEFFu, 0x10000000u,
    };
    static const uint16_t val[16] = {
        0x3c00, 0x0001, 0x7bff, 0x3555, 0x0400, 0x03ff, 0x5800, 0x2e66,
        0xbc00, 0x8001, 0xfbff, 0xb555, 0x8400, 0x83ff, 0xd800, 0xae66,
    };
    const uint32_t m = 32, k = 32, n = 32;
    f16_zero(m, k, n);
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t kk = 0; kk < k; kk++) {
            const uint32_t p = pat[r % 16];
            *A_at(k, r, kk) = ((p >> kk) & 1) ? 0x8000 : 0x0000;
            *W_at(n, kk, r) = ((pat[(r + 5) % 16] >> kk) & 1) ? 0x8000 : 0x0000;
        }
        if (r >= 16) {
            *A_at(k, r, 0) = val[r - 16];
            *A_at(k, r, 1) = (uint16_t) (val[r - 16] ^ 0x8000);
            *W_at(n, 0, r) = 0x3c00;
            *W_at(n, 1, r) = 0x3c00;
        }
    }
    bias_set(0x3c00, 0, 0);
    run_f16("zero_sign", m, k, n, M_DEEP);
}

// The rounding at the store and the width of the sticky information. Row r holds x (a random
// significand, both parities, both signs) in channel p0, the half ulp of x in channel p1 and a tiny
// term 2^(e - 23) in channel p2. The columns select x + half + tiny with tiny = +2^(e-11-d)
// (column tile 0), x + half - 2^(e-11-d) (tile 1), x + 2^(e-11-d) (tile 2) and x - 2^(e-11-d)
// (tile 3), with d = column % 32. Column 0 of each tile has no tiny term, and in tile 1 the half
// ulp is negative.
static void round_fill(uint32_t k, uint32_t p0, uint32_t p1, uint32_t p2) {
    static const int e_grp[4] = { 0, 3, 10, -1 };
    const uint32_t m = 32, n = 128;
    rng_seed("round_rows");
    f16_zero(m, k, n);
    for (uint32_t r = 0; r < m; r++) {
        const int      e    = e_grp[r / 8];
        const int      neg  = (int) ((r >> 1) & 1);
        const uint32_t par  = r & 1;
        const uint32_t mant = (rng_u32() & 0x3FEu) | par;
        *A_at(k, r, p0) = f16_make(neg, e, mant);
        *A_at(k, r, p1) = f16_pow2(e - 11, neg);
        *A_at(k, r, p2) = f16_pow2(e - 23, neg);
    }
    for (uint32_t c = 0; c < n; c++) {
        const uint32_t t = c / 32;
        const uint32_t d = c % 32;
        *W_at(n, p0, c) = 0x3c00;
        *W_at(n, p1, c) = (t == 0 || t == 1) ? ((t == 1 && d == 0) ? 0xbc00 : 0x3c00) : 0x0000;
        *W_at(n, p2, c) = d == 0 ? 0x0000 : f16_pow2(12 - (int) d, t == 1 || t == 3);
    }
    bias_set(0x3c00, 0, 0);
}

static void round_case(const char * name, uint32_t k, uint32_t p0, uint32_t p1, uint32_t p2) {
    if (!case_selected(name)) {
        return;
    }
    round_fill(k, p0, p1, p2);
    run_f16(name, 32, k, 128, M_DEEP);
}

static void cases_round(void) {
    round_case("round_near", 32, 0, 1, 2);
    round_case("round_far", 32, 0, 1, 31);
    round_case("round_k64", 64, 0, 16, 40);
}

// The alignment depth. Row r holds +big, tiny and -big in three channels (8 position patterns x 4
// magnitudes). The exact result is the tiny product alone, 2^(ta + 15 - c) in column tile 0 and
// 2^(ta + 7 - c') in column tile 1, at a depth of d = pa - ta + c (tile 0) or pa - ta + 8 + c'
// (tile 1) bits below the big product 2^(pa + 15). The tiny channels hold the column dependent
// weights, all other channels hold the weight 2^15.
static void cancel_case(const char * name, uint32_t k, const uint8_t pos[8][3], const uint8_t * tiny_slots,
                        uint32_t n_tiny, uint32_t method) {
    if (!case_selected(name)) {
        return;
    }
    static const int pa_v[4] = { 15, 15, 15, 0 };
    static const int ta_v[4] = { 15, 7, -1, -8 };
    const uint32_t m = 32, n = 64;
    f16_zero(m, k, n);
    for (uint32_t kk = 0; kk < k; kk++) {
        bool tiny = false;
        for (uint32_t i = 0; i < n_tiny; i++) {
            tiny = tiny || tiny_slots[i] == kk;
        }
        for (uint32_t c = 0; c < n; c++) {
            *W_at(n, kk, c) = tiny ? f16_pow2(c < 32 ? 15 - (int) c : 7 - (int) (c - 32), 0) : f16_pow2(15, 0);
        }
    }
    for (uint32_t r = 0; r < m; r++) {
        const uint8_t * p = pos[r / 4];
        *A_at(k, r, p[0]) = f16_pow2(pa_v[r % 4], 0);
        *A_at(k, r, p[1]) = f16_pow2(ta_v[r % 4], 0);
        *A_at(k, r, p[2]) = f16_pow2(pa_v[r % 4], 1);
    }
    bias_set(0x3c00, 0, 0);
    run_f16(name, m, k, n, method);
}

static void cases_cancel(void) {
    // {+big, tiny, -big} channel positions
    static const uint8_t pos32[8][3] = {
        { 0, 3, 1 }, { 0, 9, 1 }, { 0, 3, 8 }, { 8, 3, 16 }, { 0, 3, 4 }, { 0, 9, 12 }, { 16, 31, 24 }, { 8, 31, 0 },
    };
    static const uint8_t tiny32[3] = { 3, 9, 31 };
    static const uint8_t pos128[8][3] = {
        { 0, 3, 32 }, { 0, 35, 64 }, { 40, 3, 100 }, { 0, 127, 1 }, { 64, 73, 120 }, { 0, 9, 126 }, { 32, 31, 33 }, { 96, 127, 0 },
    };
    static const uint8_t tiny128[6] = { 3, 9, 31, 35, 73, 127 };
    cancel_case("cancel_k32", 32, pos32, tiny32, 3, M_DEEP);
    cancel_case("cancel_k32_flat", 32, pos32, tiny32, 3, M_FLAT);
    cancel_case("cancel_k128_deep", 128, pos128, tiny128, 6, M_DEEP);
    cancel_case("cancel_k128_tile", 128, pos128, tiny128, 6, M_DEEP_TILE);
    cancel_case("cancel_k128_flat", 128, pos128, tiny128, 6, M_FLAT);
    cancel_case("cancel_k128_gdn2", 128, pos128, tiny128, 6, M_GDN2);
    cancel_case("cancel_k128_retain", 128, pos128, tiny128, 6, M_RETAIN);
}

// The growth of the partial sum. Each row adds K/2 - 2 products +2^(pa + 15), then the same number
// of products -2^(pa + 15), thus the partial sum peaks at (K/2 - 2) 2^(pa + 15) and the exact result
// is the tiny product 2^(ta + 15 - c). The tiny product is in channel 0, K/2 - 1 or K - 1 (row % 3).
static void growth_case(const char * name, uint32_t k) {
    if (!case_selected(name)) {
        return;
    }
    static const int pa_v[4] = { 15, 10, 5, 0 };
    const int        ta     = -9;
    const uint32_t   m = 32, n = 32;
    const uint32_t   slots[3] = { 0, k / 2 - 1, k - 1 };
    f16_zero(m, k, n);
    for (uint32_t kk = 0; kk < k; kk++) {
        const bool tiny = kk == slots[0] || kk == slots[1] || kk == slots[2];
        for (uint32_t c = 0; c < n; c++) {
            *W_at(n, kk, c) = tiny ? f16_pow2(15 - (int) c, 0) : f16_pow2(15, 0);
        }
    }
    for (uint32_t r = 0; r < m; r++) {
        const int pa = pa_v[(r / 3) % 4];
        for (uint32_t kk = 1; kk + 1 < k / 2; kk++) {
            *A_at(k, r, kk) = f16_pow2(pa, 0);
        }
        for (uint32_t kk = k / 2 + 1; kk + 1 < k; kk++) {
            *A_at(k, r, kk) = f16_pow2(pa, 1);
        }
        *A_at(k, r, slots[r % 3]) = f16_pow2(ta, 0);
    }
    bias_set(0x3c00, 0, 0);
    run_f16(name, m, k, n, M_DEEP);
}

static void cases_growth(void) {
    growth_case("growth_k256", 256);
    growth_case("growth_k1024", 1024);
    growth_case("growth_k4096", 4096);
}

enum dist {
    DIST_UNIT   = 0,  // exponents -8 to -1, random signs: the range of activations and weights
    DIST_WIDE   = 1,  // exponents -14 to 8, random signs
    DIST_CANCEL = 2,  // the second half of k repeats the first half with a negated weight that
                      // differs in its last mantissa bits, thus the sum cancels to a small value
};

// Fills A [m][k] and W [k][n] with random values of the distribution, from the seed name
static void rand_fill(const char * seed, uint32_t m, uint32_t k, uint32_t n, enum dist dist) {
    rng_seed(seed);
    const int elo = dist == DIST_WIDE ? -14 : -8;
    const int ehi = dist == DIST_WIDE ? 8 : -1;
    for (uint32_t r = 0; r < m; r++) {
        for (uint32_t kk = 0; kk < k; kk++) {
            *A_at(k, r, kk) = f16_rand(elo, ehi);
        }
    }
    for (uint32_t kk = 0; kk < k; kk++) {
        for (uint32_t c = 0; c < n; c++) {
            *W_at(n, kk, c) = f16_rand(elo, ehi);
        }
    }
    if (dist == DIST_CANCEL) {
        for (uint32_t kk = k / 2; kk < k; kk++) {
            for (uint32_t r = 0; r < m; r++) {
                *A_at(k, r, kk) = *A_at(k, r, kk - k / 2);
            }
            for (uint32_t c = 0; c < n; c++) {
                *W_at(n, kk, c) = (uint16_t) ((*W_at(n, kk - k / 2, c) ^ 0x8000u) ^ (rng_u32() & 0x7u));
            }
        }
    }
}

// Random dot products for K = 32 to 4096 in the three distributions, with the matmul method
static void cases_rand(void) {
    static const char * dname[3] = { "unit", "wide", "cancel" };
    for (int d = 0; d < 3; d++) {
        for (uint32_t k = 32; k <= 4096; k *= 2) {
            char name[64];
            snprintf(name, sizeof(name), "rand_%s_k%u", dname[d], (unsigned) k);
            if (!case_selected(name)) {
                continue;
            }
            rand_fill(name, 32, k, 64, (enum dist) d);
            bias_set(0x3c00, 0, 0);
            run_f16(name, 32, k, 64, M_DEEP);
        }
    }
}

// The drift of the accumulator exponent. In the first K - 32 channels each group of 8 holds x and
// -x (with the same weight) and six zeros, thus each group has nonzero products and a zero sum. The
// last 32 channels hold random values. Each zero sum lowers the accumulator exponent by the
// normalization step, and the exponent field has 7 bits on v73 to v79 and 9 bits on v81.
static void drift_case(const char * name, uint32_t k) {
    if (!case_selected(name)) {
        return;
    }
    const uint32_t m = 32, n = 64;
    rand_fill(name, m, k, n, DIST_UNIT);
    for (uint32_t kk = 0; kk + 32 < k; kk++) {
        for (uint32_t r = 0; r < m; r++) {
            uint16_t * a = A_at(k, r, kk);
            *a = (kk % 8 == 1) ? (uint16_t) (*A_at(k, r, kk - 1) ^ 0x8000u) : (kk % 8 == 0) ? *a : 0;
        }
        if (kk % 8 == 1) {
            for (uint32_t c = 0; c < n; c++) {
                *W_at(n, kk, c) = *W_at(n, kk - 1, c);
            }
        }
    }
    bias_set(0x3c00, 0, 0);
    run_f16(name, m, k, n, M_DEEP);
}

static void cases_drift(void) {
    drift_case("drift_k256", 256);
    drift_case("drift_k512", 512);
    drift_case("drift_k1024", 1024);
    drift_case("drift_k4096", 4096);
}

// The same random data with each accumulation method. meth_wide_deep has the data of
// rand_wide_k256 but its own name, thus the two files must be equal.
static void cases_method(void) {
    static const struct { const char * name; uint32_t method; } v[] = {
        { "meth_wide_deep", M_DEEP }, { "meth_wide_tile", M_DEEP_TILE }, { "meth_wide_flat", M_FLAT },
        { "meth_wide_gdn2", M_GDN2 }, { "meth_wide_retain", M_RETAIN }, { "meth_wide_noclear", M_NOCLEAR },
    };
    for (size_t i = 0; i < sizeof(v) / sizeof(v[0]); i++) {
        if (!case_selected(v[i].name)) {
            continue;
        }
        rand_fill("rand_wide_k256", 64, 256, 64, DIST_WIDE);
        bias_set(0x3c00, 0, 0);
        run_f16(v[i].name, 64, 256, 64, v[i].method);
    }
}

// The output conversion: out = acc * lo + hi, with the words of the bias area. Each case uses the
// random data of rand_wide_k64.
static void bias_case(const char * name, uint16_t lo, uint16_t hi, uint32_t v1) {
    if (!case_selected(name)) {
        return;
    }
    rand_fill("rand_wide_k64", 32, 64, 32, DIST_WIDE);
    bias_set(lo, hi, v1);
    run_f16(name, 32, 64, 32, M_DEEP);
}

static void cases_bias(void) {
    bias_case("bias_id", 0x3c00, 0x0000, 0);        // the kernels (hmx_init_column_scales)
    bias_case("bias_qk128", 0x2da8, 0x2da8, 0);     // flash-attn-ops.c for DK = 128
    bias_case("bias_qk256", 0x2c00, 0x2c00, 0);     // flash-attn-ops.c for DK = 256
    bias_case("bias_lo_qk128", 0x2da8, 0x0000, 0);  // the qk scale without the high half
    bias_case("bias_hi_one", 0x3c00, 0x3c00, 0);
    bias_case("bias_hi_neg", 0x3c00, 0xbc00, 0);
    bias_case("bias_hi_sub", 0x3c00, 0x0001, 0);
    bias_case("bias_hi_max", 0x3c00, 0x7bff, 0);
    bias_case("bias_hi_third", 0x3c00, 0x3555, 0);
    bias_case("bias_lo_two", 0x4000, 0x0000, 0);
    bias_case("bias_lo_third", 0x3555, 0x0000, 0);
    bias_case("bias_lo_neg", 0xbc00, 0x0000, 0);
    bias_case("bias_lo_zero", 0x0000, 0x0000, 0);
    bias_case("bias_lo_sub", 0x0001, 0x0000, 0);
    bias_case("bias_lo_max", 0x7bff, 0x0000, 0);
    bias_case("bias_lo_inf", 0x7c00, 0x0000, 0);
    bias_case("bias_lo_nan", 0x7e00, 0x0000, 0);
    bias_case("bias_lo_third_hi_third", 0x3555, 0x3555, 0);
    // one bit of the second 128 bytes of the bias area at a time
    for (int b = 0; b < 32; b++) {
        char name[32];
        snprintf(name, sizeof(name), "bias_v1_bit%02d", b);
        bias_case(name, 0x3c00, 0x0000, 1u << b);
    }
    if (case_selected("bias_percol")) {
        rand_fill("rand_wide_k64", 32, 64, 32, DIST_WIDE);
        for (uint32_t c = 0; c < 32; c++) {
            const uint16_t lo = f16_pow2((int) c - 16, (int) (c & 1));
            const uint16_t hi = f16_make((int) ((c >> 1) & 1), -4 + (int) (c % 8), (c * 37u) & 0x3FF);
            g.bias[c]      = (uint32_t) lo | ((uint32_t) hi << 16);
            g.bias[32 + c] = 0;
        }
        run_f16("bias_percol", 32, 64, 32, M_DEEP);
    }
}

// The rounding mode field of USR (bits 23:22, FPRND) and other USR bits during the HMX instructions.
// The kernels do not write USR, thus these cases show if the HMX reads it.
static void usr_case(const char * name, uint32_t mask, uint32_t set, const char * input) {
    if (!case_selected(name)) {
        return;
    }
    g.usr_mask = mask;
    g.usr_set  = set;
    if (strcmp(input, "special") == 0) {
        f16_zero(32, 32, 32);
        for (uint32_t i = 0; i < 32; i++) {
            *A_at(32, i, 0) = k_special[i];
            *W_at(32, 0, i) = k_special[i];
        }
        bias_set(0x3c00, 0, 0);
        run_f16(name, 32, 32, 32, M_DEEP);
    } else {
        round_fill(32, 0, 1, 2);  // the data of round_near
        run_f16(name, 32, 32, 128, M_DEEP);
    }
    g.usr_mask = 0;
    g.usr_set  = 0;
}

static void cases_usr(void) {
    usr_case("usr_rnd1_round", 3u << 22, 1u << 22, "round");
    usr_case("usr_rnd2_round", 3u << 22, 2u << 22, "round");
    usr_case("usr_rnd3_round", 3u << 22, 3u << 22, "round");
    for (int b = 6; b <= 21; b++) {
        if (b >= 8 && b <= 18) {
            continue;  // LPCFG, the packet counters and the prefetch controls
        }
        char name[32];
        snprintf(name, sizeof(name), "usr_bit%02d_special", b);
        usr_case(name, 1u << b, 1u << b, "special");
    }
    usr_case("usr_bits1920_special", 3u << 19, 3u << 19, "special");
    usr_case("usr_bits2021_special", 3u << 20, 3u << 20, "special");
}

// ---- the int path (u8 activation, s8 or s4 weight, int32 accumulator)

// The activation tile of the int path: 64 rows x 32 channels, row major, byte r * 32 + k. This is
// the "flat row-major" layout of the HexKL documentation for hexkl_micro_hmx_mm_u8i8.
static inline size_t i8_act_off(uint32_t r, uint32_t kk) {
    return (size_t) r * 32 + kk;
}

// The s8 weight tile: 8 rows of 128 bytes. Row q holds channels 4q to 4q + 3 of the 32 columns:
// byte (k / 4) * 128 + c * 4 + k % 4. The weight range 0x380 of HexKL is the offset of row 7.
static inline size_t i8_wgt_off(uint32_t kk, uint32_t c) {
    return (size_t) (kk / 4) * 128 + (size_t) c * 4 + kk % 4;
}

// The conversion tables of the int32 read of HexKL (hexkl_micro_hmx_setup_acc_read_int32): four
// tables at 1 KiB intervals. Table t holds 64 words of value vals[t] at offset 0 and zero after.
static void i8_tables_set(uint8_t * cfg, const uint32_t vals[4]) {
    memset(cfg, 0, 4096);
    for (int t = 0; t < 4; t++) {
        uint32_t * w = (uint32_t *) (cfg + t * 1024);
        for (int i = 0; i < 64; i++) {
            w[i] = vals[t];
        }
    }
}

// Reads the int32 accumulator as four u8 planes, the sequence of hexkl_micro_hmx_acc_read_int32:
// for each table, "bias = mxmem(table)" and one ":after:retain:cm.ub" store. The last store
// releases the accumulator, as HMX_STORE_PLANE_LAST_I32 does.
static void i8_read_planes(uint8_t * out, const uint8_t * cfg) {
    asm volatile(HMX_SET_BIAS_I32("%0") : : "r"(cfg + 0 * 1024) : "memory");
    asm volatile(HMX_STORE_PLANE_I32("%0", "%1") : : "r"(out + 0 * I8_PLANE), "r"(0) : "memory");
    asm volatile(HMX_SET_BIAS_I32("%0") : : "r"(cfg + 1 * 1024) : "memory");
    asm volatile(HMX_STORE_PLANE_I32("%0", "%1") : : "r"(out + 1 * I8_PLANE), "r"(0) : "memory");
    asm volatile(HMX_SET_BIAS_I32("%0") : : "r"(cfg + 2 * 1024) : "memory");
    asm volatile(HMX_STORE_PLANE_I32("%0", "%1") : : "r"(out + 2 * I8_PLANE), "r"(0) : "memory");
    asm volatile(HMX_SET_BIAS_I32("%0") : : "r"(cfg + 3 * 1024) : "memory");
    asm volatile(HMX_STORE_PLANE_LAST_I32("%0", "%1") : : "r"(out + 3 * I8_PLANE), "r"(0) : "memory");
}

static const uint32_t k_i32_read_scales[4] = HMX_I32_READ_SCALES;

// The ":deep" int loads of up to 32 k tiles each, as core_dot_chunk_fp16 does for f16: the range
// operand holds 16 bits, thus one load covers at most 65536 bytes of activation. param_range
// selects the range operand: 0 is the byte count minus 1 (correct), 1 is 32 n - 1 (the single tile
// value 0x1f of HexKL scaled by n, a usage error), 2 is one load for all tiles (an overflow of the
// range for more than 32 tiles).
static void int_deep_chain(uint32_t kind, uint32_t n_kt, uint32_t wt_bytes, uint32_t param_range) {
    const uint32_t step = param_range == 2 ? n_kt : 32;
    for (uint32_t t0 = 0; t0 < n_kt; t0 += step) {
        const uint32_t  n       = n_kt - t0 < step ? n_kt - t0 : step;
        const uint32_t  a_range = param_range == 1 ? 32 * n - 1 : I8_ACT_BYTES * n - 1;
        const uint32_t  w_range = wt_bytes * n - 1;
        const uint8_t * a       = g.act + (size_t) t0 * I8_ACT_BYTES;
        const uint8_t * w       = g.wgt + (size_t) t0 * wt_bytes;
        if (kind == KIND_I4) {
            asm volatile(HMX_LOAD_MPY_DEEP_I4("%1", "%2", "%0", "%3") : : "r"(a_range), "r"(a), "r"(w), "r"(w_range) : "memory");
        } else {
            asm volatile(HMX_LOAD_MPY_DEEP_I8("%1", "%2", "%0", "%3") : : "r"(a_range), "r"(a), "r"(w), "r"(w_range) : "memory");
        }
    }
}

// Runs one int case. The raw activation tiles (n_kt x 2048 bytes) and weight tiles (n_kt x 1024 or
// 512 bytes) are in g.act and g.wgt. param[0] and param[1] are the repeat counts of M_I8_REPEAT,
// param[2] selects the range operand of the ":deep" loads (int_deep_chain). The output is four
// planes of 2048 bytes. O(k).
static void run_int(const char * name, uint32_t kind, uint32_t k, uint32_t method, const uint32_t param[3]) {
    const uint32_t n_kt     = k / 32;
    const uint32_t wt_bytes = kind == KIND_I4 ? I4_WGT_BYTES : I8_WGT_BYTES;
    i8_tables_set(g.cfg, k_i32_read_scales);

    struct case_hdr h;
    hdr_init(&h, kind, method, I8_ROWS, k, 32);
    memcpy(h.param, param, sizeof(h.param));
    h.usr = usr_read();
    const uint32_t n_reads = method == M_I8_BLOCK ? n_kt : 1;
    if ((size_t) n_reads * 4 * I8_PLANE > ARENA_OUT) {
        printf("lab: error: case %s needs %u int32 reads, more than the output arena\n", name, (unsigned) n_reads);
        exit(2);
    }
    memset(g.out, 0x55, (size_t) n_reads * 4 * I8_PLANE);

    LAB_BARRIER();
    asm volatile(HMX_CLRACC_I32() : : : "memory");
    switch (method) {
        case M_I8_BLOCK:
            for (uint32_t t = 0; t < n_kt; t++) {
                asm volatile(HMX_CLRACC_I32() : : : "memory");
                asm volatile(HMX_LOAD_MPY_I8("%1", "%2", "%0", "%3")
                             : : "r"(HMX_I8_ACT_RANGE_ONE_TILE), "r"(g.act + (size_t) t * I8_ACT_BYTES),
                               "r"(g.wgt + (size_t) t * wt_bytes), "r"(HMX_I8_WT_RANGE_ONE_TILE) : "memory");
                i8_read_planes(g.out + (size_t) t * 4 * I8_PLANE, g.cfg);
            }
            break;
        case M_I8_SINGLE:
            for (uint32_t t = 0; t < n_kt; t++) {
                asm volatile(HMX_LOAD_MPY_I8("%1", "%2", "%0", "%3")
                             : : "r"(HMX_I8_ACT_RANGE_ONE_TILE), "r"(g.act + (size_t) t * I8_ACT_BYTES),
                               "r"(g.wgt + (size_t) t * wt_bytes), "r"(HMX_I8_WT_RANGE_ONE_TILE) : "memory");
            }
            break;
        case M_I8_REPEAT:
            for (uint32_t p = 0; p < 2; p++) {
                for (uint32_t i = 0; i < param[p]; i++) {
                    asm volatile(HMX_LOAD_MPY_I8("%1", "%2", "%0", "%3")
                                 : : "r"(HMX_I8_ACT_RANGE_ONE_TILE), "r"(g.act + (size_t) p * I8_ACT_BYTES),
                                   "r"(g.wgt + (size_t) p * I8_WGT_BYTES), "r"(HMX_I8_WT_RANGE_ONE_TILE) : "memory");
                }
            }
            break;
        case M_I8_DEEP:
        case M_I4_DEEP:
            int_deep_chain(kind, n_kt, wt_bytes, param[2]);
            break;
        default:
            printf("lab: error: run_int does not know method %u\n", (unsigned) method);
            exit(2);
    }
    if (method != M_I8_BLOCK) {
        i8_read_planes(g.out, g.cfg);
    }
    LAB_BARRIER();

    const uint32_t n_tiles = method == M_I8_REPEAT ? 2 : n_kt;
    h.bias_bytes = 4096;
    h.a_bytes    = n_tiles * I8_ACT_BYTES;
    h.w_bytes    = n_tiles * wt_bytes;
    h.out_bytes  = n_reads * 4 * I8_PLANE;
    h.n_snap     = 4 * n_reads;
    memcpy(g.res, g.out, h.out_bytes);
    case_write(name, &h, g.cfg, g.act, g.wgt, g.res);
}

// Fills n_kt int tiles with random bytes: full range u8 activations and s8 (or packed s4) weights
static void int_fill_rand(const char * seed, uint32_t n_kt, uint32_t wt_bytes) {
    rng_seed(seed);
    for (uint32_t i = 0; i < n_kt * I8_ACT_BYTES; i++) {
        g.act[i] = (uint8_t) rng_u32();
    }
    for (uint32_t i = 0; i < n_kt * wt_bytes; i++) {
        g.wgt[i] = (uint8_t) rng_u32();
    }
}

static void cases_int(void) {
    const uint32_t p0[3] = { 0, 0, 0 };
    if (case_selected("i8_rand_k32")) {
        int_fill_rand("i8_rand_k32", 1, I8_WGT_BYTES);
        run_int("i8_rand_k32", KIND_I8, 32, M_I8_SINGLE, p0);
    }
    if (case_selected("i8_extreme_k32")) {
        // activation rows: all 255, all 0, 255 on even channels, and random 0 or 255; weight columns:
        // all -128, all 127, alternating -128 and 127, and random -128, -1, 0 or 127
        static const uint8_t wv[4] = { 0x80, 0xFF, 0x00, 0x7F };
        rng_seed("i8_extreme_k32");
        for (uint32_t r = 0; r < I8_ROWS; r++) {
            for (uint32_t kk = 0; kk < 32; kk++) {
                const uint32_t mode = r % 4;
                const uint8_t  a    = mode == 0 ? 255 : mode == 1 ? 0 : mode == 2 ? ((kk & 1) ? 0 : 255) : ((rng_u32() & 1) ? 255 : 0);
                g.act[i8_act_off(r, kk)] = a;
            }
        }
        for (uint32_t kk = 0; kk < 32; kk++) {
            for (uint32_t c = 0; c < 32; c++) {
                const uint32_t mode = c % 4;
                const uint8_t  w    = mode == 0 ? 0x80 : mode == 1 ? 0x7F : mode == 2 ? ((kk & 1) ? 0x7F : 0x80) : wv[rng_u32() & 3];
                g.wgt[i8_wgt_off(kk, c)] = w;
            }
        }
        run_int("i8_extreme_k32", KIND_I8, 32, M_I8_SINGLE, p0);
    }
    for (uint32_t k = 64; k <= 4096; k *= 4) {
        char name[48];
        char seed[48];
        snprintf(seed, sizeof(seed), "i8_rand_k%u", (unsigned) k);
        snprintf(name, sizeof(name), "i8_single_k%u", (unsigned) k);
        if (case_selected(name)) {
            int_fill_rand(seed, k / 32, I8_WGT_BYTES);
            run_int(name, KIND_I8, k, M_I8_SINGLE, p0);
        }
        snprintf(name, sizeof(name), "i8_deep_k%u", (unsigned) k);
        if (case_selected(name)) {
            int_fill_rand(seed, k / 32, I8_WGT_BYTES);
            run_int(name, KIND_I8, k, M_I8_DEEP, p0);
        }
    }
    // two usage errors of the range operand, for the record: 32 n - 1, and one load of 128 tiles
    if (case_selected("i8_badrange_k64")) {
        const uint32_t p[3] = { 0, 0, 1 };
        int_fill_rand("i8_rand_k64", 2, I8_WGT_BYTES);
        run_int("i8_badrange_k64", KIND_I8, 64, M_I8_DEEP, p);
    }
    if (case_selected("i8_overrange_k4096")) {
        const uint32_t p[3] = { 0, 0, 2 };
        int_fill_rand("i8_rand_k4096", 128, I8_WGT_BYTES);
        run_int("i8_overrange_k4096", KIND_I8, 4096, M_I8_DEEP, p);
    }
    // The int32 limit: tile pair 0 is all 255 x all 127 (+1036320 for each MAC) or all 255 x all -128
    // (-1044480 for each MAC), tile pair 1 is the other sign. The repeat counts cross 2^31.
    static const struct { const char * name; uint8_t w0, w1; uint32_t n0, n1; } sat[] = {
        { "i8_sat_pos_2072", 0x7F, 0x80, 2072, 0 },     // 2147447040, below 2^31 - 1
        { "i8_sat_pos_2073", 0x7F, 0x80, 2073, 0 },     // 2148483360, above
        { "i8_sat_neg_2056", 0x80, 0x7F, 2056, 0 },     // -2147450880, above -2^31
        { "i8_sat_neg_2057", 0x80, 0x7F, 2057, 0 },     // -2148495360, below
        { "i8_sat_pos_back", 0x7F, 0x80, 2100, 1000 },  // over 2^31, then back to 1131792000
        { "i8_sat_neg_4200", 0x80, 0x7F, 4200, 0 },     // -4386816000, below -2^32
    };
    for (size_t i = 0; i < sizeof(sat) / sizeof(sat[0]); i++) {
        if (!case_selected(sat[i].name)) {
            continue;
        }
        memset(g.act, 0xFF, 2 * I8_ACT_BYTES);
        memset(g.wgt, sat[i].w0, I8_WGT_BYTES);
        memset(g.wgt + I8_WGT_BYTES, sat[i].w1, I8_WGT_BYTES);
        const uint32_t p[3] = { sat[i].n0, sat[i].n1, 0 };
        run_int(sat[i].name, KIND_I8, 32, M_I8_REPEAT, p);
    }
    if (case_selected("i4_rand_k32")) {
        int_fill_rand("i4_rand_k32", 1, I4_WGT_BYTES);
        run_int("i4_rand_k32", KIND_I4, 32, M_I4_DEEP, p0);
    }
    if (case_selected("i4_rand_k128")) {
        int_fill_rand("i4_rand_k128", 4, I4_WGT_BYTES);
        run_int("i4_rand_k128", KIND_I4, 128, M_I4_DEEP, p0);
    }
}

// ---- the paths of a Q8_0 matmul and of an F16 matmul, against the CPU reference (compare.py)
//
// The source data: f32 activations x [64][k] and f32 weights w [k][32], with an approximately
// normal distribution, row and column scales, and outlier channels. The weights become Q8_0 with the
// reference quantization of ggml (quantize_row_q8_0_ref) along k for each column, and the CPU
// reference quantizes the activations the same way. The scalar float unit of the DSP is IEEE with
// round to nearest even, and the loops are not vectorized, thus the quants here are those of the
// reference. compare.py checks them against its own quantization.
//
// The HMX paths on this data:
//   q8a  today's prefill path: f16(x) x f16(q_w * d_w), one deep f16 accumulation
//   q8c  the integer quants as f16 values (exact), one f16 sum for each 32-k block (M_BLOCK)
//   q8b  the integer quants on the int path: u8 activation q_x + 128, s8 weight q_w, one int32 read
//        for each 32-k block (M_I8_BLOCK); the host subtracts 128 * sum(q_w) for each block
//   f16w an F16 weight matrix: f16(x) x f16(w), one deep f16 accumulation

#define Q8_M     64u
#define Q8_N     32u
#define Q8_K_MAX 1024u

static struct {
    float *    x;    // [64][k] activations
    float *    w;    // [k][32] weights
    int8_t *   qx;   // [64][k] activation quants
    uint16_t * dx;   // [64][k / 32] activation scales (f16 bits)
    int8_t *   qw;   // [k][32] weight quants
    uint16_t * dw;   // [k / 32][32] weight scales (f16 bits)
} q8;

// The f16 bits of an f32 value with round to nearest even, as GGML_FP32_TO_FP16 does (integer code,
// thus it does not depend on the conversion instruction of the core). The function is not inlined:
// the HVX vectorizer of hexagon-clang 19.0.07 stops with "Cannot select: v64i1 = truncate" on a
// loop that inlines it.
static __attribute__((noinline)) uint16_t f32_to_f16_rne(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t absx = x & 0x7FFFFFFFu;
    if (absx >= 0x7F800000u) {
        return (uint16_t) (sign | (absx > 0x7F800000u ? 0x7E00u : 0x7C00u));
    }
    const int e = (int) (absx >> 23) - 127;
    if (e > 15) {
        return (uint16_t) (sign | 0x7C00u);
    }
    uint32_t mant = (absx & 0x7FFFFFu) | 0x800000u;
    int      shift;
    uint32_t base;
    if (e >= -14) {
        shift = 13;
        base  = (uint32_t) (e + 15) << 10;
        mant &= 0x7FFFFFu;
    } else {
        shift = -e - 1;  // subnormal: the significand of 2^-24 units
        base  = 0;
        if (shift > 31) {
            return (uint16_t) sign;
        }
    }
    const uint32_t q    = mant >> shift;
    const uint32_t rem  = mant & ((1u << shift) - 1);
    const uint32_t half = 1u << (shift - 1);
    uint32_t       r    = base + q;
    if (rem > half || (rem == half && (q & 1))) {
        r++;  // a carry into the exponent is the correct result
    }
    return (uint16_t) (sign | r);
}

// A value with an approximately normal distribution (mean 0, deviation 1): four uniform values
static float gauss_f32(void) {
    int32_t s = 0;
    for (int i = 0; i < 4; i++) {
        s += (int32_t) (rng_u32() & 0xFFFF);
    }
    return (float) (s - 2 * 65535) / 37837.0f;
}

// quantize_row_q8_0_ref of ggml for k values at the given element stride
static void q8_quant_ref(const float * x, size_t stride, uint32_t k, int8_t * q, size_t qstride, uint16_t * d, size_t dstride) {
    for (uint32_t b = 0; b < k / 32; b++) {
        float amax = 0.0f;
#pragma clang loop vectorize(disable)
        for (uint32_t j = 0; j < 32; j++) {
            const float v = fabsf(x[(size_t) (b * 32 + j) * stride]);
            amax = amax > v ? amax : v;
        }
        const float d_f = amax / ((1 << 7) - 1);
        const float id  = d_f ? 1.0f / d_f : 0.0f;
        d[(size_t) b * dstride] = f32_to_f16_rne(d_f);
#pragma clang loop vectorize(disable)
        for (uint32_t j = 0; j < 32; j++) {
            q[(size_t) (b * 32 + j) * qstride] = (int8_t) roundf(x[(size_t) (b * 32 + j) * stride] * id);
        }
    }
}

// The f32 value of f16 bits (integer code)
static float f16_bits_to_f32(uint16_t h) {
    return lab_hf_to_f32(h);
}

// Makes the source data for k and writes the file q8src_k<k>.bin: A is x (u32 bits [64][k]), W is
// q_w (s8 [k][32]) then d_w (u16 [k/32][32]), the output section is q_x (s8 [64][k]) then d_x
// (u16 [64][k/32]).
static void q8_src_make(uint32_t k, bool write) {
    char name[48];
    snprintf(name, sizeof(name), "q8src_k%u", (unsigned) k);
    rng_seed(name);
    for (uint32_t r = 0; r < Q8_M; r++) {
        const float rs = (float) (1u << (r % 7)) / 8.0f;  // 2^-3 .. 2^3
        for (uint32_t kk = 0; kk < k; kk++) {
            const float ch = (kk % 29 == 3) ? 8.0f : 1.0f;  // outlier channels
            q8.x[(size_t) r * k + kk] = gauss_f32() * rs * ch;
        }
    }
    for (uint32_t kk = 0; kk < k; kk++) {
        for (uint32_t c = 0; c < Q8_N; c++) {
            q8.w[(size_t) kk * Q8_N + c] = gauss_f32() * 0.02f * (float) (4 + c % 5) / 4.0f;
        }
    }
    for (uint32_t c = 0; c < Q8_N; c++) {
        q8_quant_ref(q8.w + c, Q8_N, k, q8.qw + c, Q8_N, q8.dw + c, Q8_N);
    }
    for (uint32_t r = 0; r < Q8_M; r++) {
        q8_quant_ref(q8.x + (size_t) r * k, 1, k, q8.qx + (size_t) r * k, 1, q8.dx + (size_t) r * (k / 32), 1);
    }
    if (!write) {
        return;
    }
    struct case_hdr h;
    hdr_init(&h, KIND_Q8_SRC, 0, Q8_M, k, Q8_N);
    h.a_bytes = Q8_M * k * 4;
    h.w_bytes = k * Q8_N + (k / 32) * Q8_N * 2;
    h.out_bytes = Q8_M * k + Q8_M * (k / 32) * 2;
    uint8_t * wbuf = g.res;
    memcpy(wbuf, q8.qw, (size_t) k * Q8_N);
    memcpy(wbuf + (size_t) k * Q8_N, q8.dw, (size_t) (k / 32) * Q8_N * 2);
    uint8_t * obuf = (uint8_t *) g.w_log;  // free here: the f16 cases fill it again
    memcpy(obuf, q8.qx, (size_t) Q8_M * k);
    memcpy(obuf + (size_t) Q8_M * k, q8.dx, (size_t) Q8_M * (k / 32) * 2);
    case_write(name, &h, NULL, q8.x, wbuf, obuf);
}

static void cases_q8(void) {
    for (uint32_t k = 256; k <= Q8_K_MAX; k *= 4) {
        char n_src[32], n_a[32], n_b[32], n_c[32], n_f[32];
        snprintf(n_src, sizeof(n_src), "q8src_k%u", (unsigned) k);
        snprintf(n_a, sizeof(n_a), "q8a_k%u", (unsigned) k);
        snprintf(n_b, sizeof(n_b), "q8b_k%u", (unsigned) k);
        snprintf(n_c, sizeof(n_c), "q8c_k%u", (unsigned) k);
        snprintf(n_f, sizeof(n_f), "f16w_k%u", (unsigned) k);
        const bool s_src = case_selected(n_src);
        const bool s_a   = case_selected(n_a);
        const bool s_b   = case_selected(n_b);
        const bool s_c   = case_selected(n_c);
        const bool s_f   = case_selected(n_f);
        if (!(s_src || s_a || s_b || s_c || s_f)) {
            continue;
        }
        q8_src_make(k, s_src);
        if (s_a) {
            for (uint32_t r = 0; r < Q8_M; r++) {
                for (uint32_t kk = 0; kk < k; kk++) {
                    *A_at(k, r, kk) = f32_to_f16_rne(q8.x[(size_t) r * k + kk]);
                }
            }
            for (uint32_t kk = 0; kk < k; kk++) {
                for (uint32_t c = 0; c < Q8_N; c++) {
                    const float dq = (float) q8.qw[(size_t) kk * Q8_N + c] * f16_bits_to_f32(q8.dw[(size_t) (kk / 32) * Q8_N + c]);
                    *W_at(Q8_N, kk, c) = f32_to_f16_rne(dq);  // exact product, one rounding to f16
                }
            }
            bias_set(0x3c00, 0, 0);
            run_f16(n_a, Q8_M, k, Q8_N, M_DEEP);
        }
        if (s_c) {
            for (uint32_t r = 0; r < Q8_M; r++) {
                for (uint32_t kk = 0; kk < k; kk++) {
                    *A_at(k, r, kk) = f32_to_f16_rne((float) q8.qx[(size_t) r * k + kk]);
                }
            }
            for (uint32_t kk = 0; kk < k; kk++) {
                for (uint32_t c = 0; c < Q8_N; c++) {
                    *W_at(Q8_N, kk, c) = f32_to_f16_rne((float) q8.qw[(size_t) kk * Q8_N + c]);
                }
            }
            // A block sum reaches 127 * 127 * 32 = 516128, above the f16 range: the output scale
            // 2^-8 keeps it below 2048 and does not change the relative precision
            bias_set(0x1c00, 0, 0);
            run_f16(n_c, Q8_M, k, Q8_N, M_BLOCK);
        }
        if (s_f) {
            for (uint32_t r = 0; r < Q8_M; r++) {
                for (uint32_t kk = 0; kk < k; kk++) {
                    *A_at(k, r, kk) = f32_to_f16_rne(q8.x[(size_t) r * k + kk]);
                }
            }
            for (uint32_t kk = 0; kk < k; kk++) {
                for (uint32_t c = 0; c < Q8_N; c++) {
                    *W_at(Q8_N, kk, c) = f32_to_f16_rne(q8.w[(size_t) kk * Q8_N + c]);
                }
            }
            bias_set(0x3c00, 0, 0);
            run_f16(n_f, Q8_M, k, Q8_N, M_DEEP);
        }
        if (s_b) {
            const uint32_t n_kt = k / 32;
            for (uint32_t t = 0; t < n_kt; t++) {
                for (uint32_t r = 0; r < I8_ROWS; r++) {
                    for (uint32_t kk = 0; kk < 32; kk++) {
                        g.act[(size_t) t * I8_ACT_BYTES + i8_act_off(r, kk)] = (uint8_t) (q8.qx[(size_t) r * k + t * 32 + kk] + 128);
                    }
                }
                for (uint32_t kk = 0; kk < 32; kk++) {
                    for (uint32_t c = 0; c < Q8_N; c++) {
                        g.wgt[(size_t) t * I8_WGT_BYTES + i8_wgt_off(kk, c)] = (uint8_t) q8.qw[(size_t) (t * 32 + kk) * Q8_N + c];
                    }
                }
            }
            const uint32_t p0[3] = { 0, 0, 0 };
            run_int(n_b, KIND_I8, k, M_I8_BLOCK, p0);
        }
    }
}

// Prints the USR value at the start and the USR bits that a write can set, for the silicon probe
static void usr_report(void) {
    const uint32_t usr0 = usr_read();
    uint32_t writable = 0;
    for (int b = 0; b < 32; b++) {
        if ((b >= 25 && b <= 29) || b == 8 || b == 9) {
            continue;  // the IEEE trap enables of the scalar unit, and LPCFG of the hardware loops
        }
        usr_write(usr0 | (1u << b));
        if (usr_read() & (1u << b) & ~usr0) {
            writable |= 1u << b;
        }
        usr_write(usr0);
    }
    printf("lab: hmxisa usr_initial = 0x%08lx usr_writable = 0x%08lx\n", (unsigned long) usr0, (unsigned long) writable);
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
            g.only = argv[++i];
        } else if (strcmp(argv[i], "--list") == 0) {
            g.list = true;
        }
    }
    lab_init();
    crc_init();
    usr_report();

    g.act   = lab_vtcm_alloc(ARENA_ACT, TILE_BYTES);
    g.wgt   = lab_vtcm_alloc(ARENA_WGT, TILE_BYTES);
    g.out   = lab_vtcm_alloc(ARENA_OUT, TILE_BYTES);
    g.cfg   = lab_vtcm_alloc(ARENA_CFG, TILE_BYTES);
    g.a_log = lab_ddr_alloc(ARENA_ACT, 128);
    g.w_log = lab_ddr_alloc(ARENA_WGT, 128);
    g.res   = lab_ddr_alloc(ARENA_OUT, 128);
    q8.x    = lab_ddr_alloc((size_t) Q8_M * Q8_K_MAX * sizeof(float), 128);
    q8.w    = lab_ddr_alloc((size_t) Q8_K_MAX * Q8_N * sizeof(float), 128);
    q8.qx   = lab_ddr_alloc((size_t) Q8_M * Q8_K_MAX, 128);
    q8.dx   = lab_ddr_alloc((size_t) Q8_M * (Q8_K_MAX / 32) * 2, 128);
    q8.qw   = lab_ddr_alloc((size_t) Q8_K_MAX * Q8_N, 128);
    q8.dw   = lab_ddr_alloc((size_t) (Q8_K_MAX / 32) * Q8_N * 2, 128);

    cases_pass();
    cases_special();
    cases_zero();
    cases_round();
    cases_cancel();
    cases_growth();
    cases_rand();
    cases_drift();
    cases_method();
    cases_bias();
    cases_usr();
    cases_int();
    cases_q8();

    printf("lab: hmxisa cases = %u\n", (unsigned) g.n_cases);
    return 0;
}
