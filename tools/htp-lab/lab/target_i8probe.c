// Target i8probe: a bytecode interpreter for the HMX instructions of the int path.
//
// The program reads a stream of operations from a file, runs each one on the HMX, and writes the
// bytes that the DUMP operations select to a second file. Thus a host script can ask the core any
// question about the u8 activation x s8 weight path (the accumulator, the conversion word of the
// bias register, the store forms, the cvt forms) without a new build. The operation words are in
// enum op and in the form tables, thus a short host script can write a stream and decode the dump.
//
// The simulator runs the HMX in functional mode only (refer to run.sh), thus the program gives
// values and not cycles. Run it with MODE=functional.
//
// The stream is a sequence of 32-bit little-endian words. Each operation is one opcode word and a
// fixed number of argument words (refer to enum op). The areas are fixed regions of the VTCM:
//   ACT  1 MiB   activation tiles
//   WGT  512 KiB weight tiles
//   CFG  64 KiB  bias area (the conversion words)
//   OUT  256 KiB store destination
// An offset argument is a byte offset into its area. The program stops with an error when an
// operation reaches outside its area.
//
// The same source builds for the device (tools/hmx-bench/build-i8.sh, LAB_DEVICE=1) with the lab
// runtime of tools/hmx-bench/src/dsp_lab.c. There printf reaches logcat, and the paths are paths of
// the phone file system.
//
// Arguments:
//   --in <path>    The operation stream. Preset value: i8probe.in
//   --out <path>   The dump file. Preset value: i8probe.out
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"

#include "lab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef LAB_DEVICE
void lab_fini(void);
#else
static inline void lab_fini(void) {}
#endif

#define AREA_ACT_BYTES (1024u * 1024u)
#define AREA_WGT_BYTES (512u * 1024u)
#define AREA_CFG_BYTES (64u * 1024u)
#define AREA_OUT_BYTES (256u * 1024u)

enum area { A_ACT = 0, A_WGT = 1, A_CFG = 2, A_OUT = 3, A_COUNT = 4 };

// The operations. The comment gives the argument words.
enum op {
    OP_END   = 0,  // no arguments: the end of the stream
    OP_LOAD  = 1,  // area, offset, n_bytes, then ceil(n_bytes / 4) data words: copy into the area
    OP_FILL  = 2,  // area, offset, n_bytes, byte: fill a range
    OP_DUMP  = 3,  // area, offset, n_bytes: append the range to the dump file
    OP_INSN  = 4,  // form, a0, a1, a2, a3: run one HMX form (refer to run_insn)
    OP_MARK  = 5,  // tag: append the tag word to the dump file (the host script checks the order)
};

static uint8_t * g_area[A_COUNT];
static uint32_t  g_size[A_COUNT] = { AREA_ACT_BYTES, AREA_WGT_BYTES, AREA_CFG_BYTES, AREA_OUT_BYTES };

// Returns the address of [off, off + n) in the area, or stops the program when the range is
// outside the area.
static uint8_t * area_ptr(uint32_t area, uint32_t off, uint32_t n) {
    if (area >= A_COUNT || off > g_size[area] || n > g_size[area] - off) {
        printf("lab: error: the range [%u, %u + %u) is outside area %u\n", (unsigned) off, (unsigned) off, (unsigned) n,
               (unsigned) area);
        exit(2);
    }
    return g_area[area] + off;
}

// The store forms of the accumulator, "mxmem(Rs, Rt)<suffix>". Rs is the OUT address, Rt a value.
#define STORE_FORMS(X)                                      \
    X(0, ":after.ub = acc")                                 \
    X(1, ":after:sat.ub = acc")                             \
    X(2, ":after:cm.ub = acc")                              \
    X(3, ":after:cm:sat.ub = acc")                          \
    X(4, ":after:retain.ub = acc")                          \
    X(5, ":after:retain:sat.ub = acc")                      \
    X(6, ":after:retain:cm.ub = acc")                       \
    X(7, ":after:retain:cm:sat.ub = acc")                   \
    X(8, ":before.ub = acc")                                \
    X(9, ":before:sat.ub = acc")                            \
    X(10, ":before:cm.ub = acc")                            \
    X(11, ":before:cm:sat.ub = acc")                        \
    X(12, ":before:retain.ub = acc")                        \
    X(13, ":before:retain:sat.ub = acc")                    \
    X(14, ":before:retain:cm.ub = acc")                     \
    X(15, ":before:retain:cm:sat.ub = acc")                 \
    X(16, ":after.uh = acc:2x1")                            \
    X(17, ":after:sat.uh = acc:2x1")                        \
    X(18, ":after:retain.uh = acc:2x1")                     \
    X(19, ":after:retain:sat.uh = acc:2x1")                 \
    X(20, ":after.uh = acc:2x2")                            \
    X(21, ":after:sat.uh = acc:2x2")                        \
    X(22, ":after:retain.uh = acc:2x2")                     \
    X(23, ":after:retain:sat.uh = acc:2x2")                 \
    X(24, ":before.uh = acc:2x1")                           \
    X(25, ":before:sat.uh = acc:2x1")                       \
    X(26, ":before:retain.uh = acc:2x1")                    \
    X(27, ":before:retain:sat.uh = acc:2x1")                \
    X(28, ":before.uh = acc:2x2")                           \
    X(29, ":before:sat.uh = acc:2x2")                       \
    X(30, ":before:retain.uh = acc:2x2")                    \
    X(31, ":before:retain:sat.uh = acc:2x2")                \
    X(32, ":after.hf = acc")                                \
    X(33, ":after:pos.hf = acc")                            \
    X(34, ":after:retain.hf = acc")                         \
    X(35, ":after:retain:pos.hf = acc")                     \
    X(36, ":before.hf = acc")                               \
    X(37, ":before:pos.hf = acc")                           \
    X(38, ":before:retain.hf = acc")                        \
    X(39, ":before:retain:pos.hf = acc")                    \
    X(50, " = cvt")                                         \
    X(51, ":2x2 = cvt")                                     \
    X(52, ":cm = cvt")                                      \
    X(53, ":deep = cvt")                                    \
    X(54, ":cm:deep = cvt")

// The conversion forms "cvt.<type> = acc(Rs)<suffix>". Rs is a value.
#define CVT_FORMS(X)                   \
    X(40, "cvt.ub = acc(%0)")          \
    X(41, "cvt.ub = acc(%0):sc0")      \
    X(42, "cvt.ub = acc(%0):sc1")      \
    X(43, "cvt.uh = acc(%0):2x1")      \
    X(44, "cvt.uh = acc(%0):2x2")      \
    X(45, "cvt.hf = acc(%0)")

// The forms with no operand
#define BARE_FORMS(X)          \
    X(60, "mxclracc")          \
    X(61, "mxclracc.hf")       \
    X(62, "mxswapacc")         \
    X(63, "mxswapacc.hf")

// The bias forms. Rs is a CFG address for a load and an OUT address for a store.
#define BIAS_FORMS(X)                          \
    X(64, "bias = mxmem(%0)", A_CFG)           \
    X(65, "bias = mxmem2(%0)", A_CFG)          \
    X(66, "mxmem(%0) = bias", A_OUT)           \
    X(67, "mxmem2(%0) = bias", A_OUT)

// The multiply forms "{ activation = mxmem(ACT + a0, a2)<asuffix> ; weight = mxmem(WGT + a1, a3)<wsuffix> }"
#define MPY_FORMS(X)                                                                   \
    X(70, "activation.ub = mxmem(%0, %2):cm", "weight.b = mxmem(%1, %3)")              \
    X(71, "activation.ub = mxmem(%0, %2):deep:cm", "weight.b = mxmem(%1, %3)")         \
    X(72, "activation.ub = mxmem(%0, %2)", "weight.b = mxmem(%1, %3)")                 \
    X(73, "activation.ub = mxmem(%0, %2):deep", "weight.b = mxmem(%1, %3)")            \
    X(74, "activation.ub = mxmem(%0, %2):cm", "weight.n = mxmem(%1, %3)")              \
    X(75, "activation.ub = mxmem(%0, %2):deep:cm", "weight.n = mxmem(%1, %3)")         \
    X(76, "activation.hf = mxmem(%0, %2)", "weight.hf = mxmem(%1, %3)")                \
    X(77, "activation.hf = mxmem(%0, %2):deep", "weight.hf = mxmem(%1, %3)")           \
    X(78, "activation.ub = mxmem(%0, %2):single:cm", "weight.b = mxmem(%1, %3)")       \
    X(79, "activation.ub = mxmem(%0, %2):cm", "weight.b = mxmem(%1, %3):single")       \
    X(80, "activation.ub = mxmem(%0, %2):above:cm", "weight.b = mxmem(%1, %3)")        \
    X(81, "activation.ub = mxmem(%0, %2):dilate:cm", "weight.b = mxmem(%1, %3)")       \
    X(82, "activation.ub = mxmem(%0, %2):cm", "weight.b = mxmem(%1, %3):drop")         \
    X(83, "activation.ub = mxmem(%0, %2):cm", "weight.b = mxmem(%1, %3):after")        \
    X(84, "activation.ub = mxmem(%0, %2):cm", "weight.b = mxmem(%1, %3):dilate")       \
    X(85, "activation.ub = mxmem(%0, %2):cm", "weight.b = mxmem(%1, %3):deep")

// Runs one HMX form. The meaning of a0 to a3 depends on the form:
//   store forms:  a0 OUT offset, a1 the value of Rt
//   cvt forms:    a0 the value of Rs
//   bias forms:   a0 CFG offset (load) or OUT offset (store)
//   mpy forms:    a0 ACT offset, a1 WGT offset, a2 activation range, a3 weight range
static void run_insn(uint32_t form, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
    switch (form) {
#define X_STORE(id, sfx)                                                                                  \
    case id:                                                                                              \
        asm volatile("mxmem(%0, %1)" sfx "\n" : : "r"(area_ptr(A_OUT, a0, 1)), "r"(a1) : "memory"); \
        break;
        STORE_FORMS(X_STORE)
#undef X_STORE
#define X_CVT(id, text)                                    \
    case id:                                               \
        asm volatile(text "\n" : : "r"(a0) : "memory");    \
        break;
        CVT_FORMS(X_CVT)
#undef X_CVT
#define X_BARE(id, text)                       \
    case id:                                   \
        asm volatile(text "\n" : : : "memory");  \
        break;
        BARE_FORMS(X_BARE)
#undef X_BARE
#define X_BIAS(id, text, area)                                                  \
    case id:                                                                    \
        asm volatile(text "\n" : : "r"(area_ptr(area, a0, 256)) : "memory");    \
        break;
        BIAS_FORMS(X_BIAS)
#undef X_BIAS
#define X_MPY(id, atext, wtext)                                                                        \
    case id:                                                                                           \
        asm volatile("{\n" atext "\n" wtext "\n}\n"                                                    \
                     :                                                                                 \
                     : "r"(area_ptr(A_ACT, a0, 1)), "r"(area_ptr(A_WGT, a1, 1)), "r"(a2), "r"(a3)       \
                     : "memory");                                                                      \
        break;
        MPY_FORMS(X_MPY)
#undef X_MPY
        default:
            printf("lab: error: form %u is not known\n", (unsigned) form);
            exit(2);
    }
}

// Reads the whole file into DDR. O(file size).
static uint32_t * read_stream(const char * path, size_t * n_words) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        printf("lab: error: cannot open the operation stream %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0 || n % 4 != 0) {
        printf("lab: error: the stream %s has %ld bytes, not a positive multiple of 4\n", path, n);
        exit(2);
    }
    uint32_t * w = lab_ddr_alloc((size_t) n, 128);
    if (fread(w, 1, (size_t) n, f) != (size_t) n) {
        printf("lab: error: the read of %s failed\n", path);
        exit(2);
    }
    fclose(f);
    *n_words = (size_t) n / 4;
    return w;
}

// Returns word i of the stream, or stops the program at the end of the stream
static uint32_t word_at(const uint32_t * w, size_t n, size_t i) {
    if (i >= n) {
        printf("lab: error: the stream ends inside an operation at word %lu\n", (unsigned long) i);
        exit(2);
    }
    return w[i];
}

// The operation stream is not a static list of Hexagon forms, and the dump holds the raw bytes,
// thus each opcode is small and the host script does the decoding.
int main(int argc, char ** argv) {
    lab_init();
    const char * in_path  = "i8probe.in";
    const char * out_path = "i8probe.out";
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--in") == 0) {
            in_path = argv[i + 1];
        } else if (strcmp(argv[i], "--out") == 0) {
            out_path = argv[i + 1];
        }
    }
    g_area[A_ACT] = lab_vtcm_alloc(AREA_ACT_BYTES, 4096);
    g_area[A_WGT] = lab_vtcm_alloc(AREA_WGT_BYTES, 4096);
    g_area[A_CFG] = lab_vtcm_alloc(AREA_CFG_BYTES, 4096);
    g_area[A_OUT] = lab_vtcm_alloc(AREA_OUT_BYTES, 4096);

    size_t     n  = 0;
    uint32_t * w  = read_stream(in_path, &n);
    FILE *     fo = fopen(out_path, "wb");
    if (!fo) {
        printf("lab: error: cannot open the dump file %s\n", out_path);
        return 2;
    }

    size_t   i     = 0;
    uint32_t n_ops = 0;
    for (;;) {
        const uint32_t op = word_at(w, n, i++);
        if (op == OP_END) {
            break;
        }
        n_ops++;
        switch (op) {
            case OP_LOAD: {
                const uint32_t area = word_at(w, n, i), off = word_at(w, n, i + 1), nb = word_at(w, n, i + 2);
                i += 3;
                const size_t nw = (nb + 3) / 4;
                if (i + nw > n) {
                    printf("lab: error: the data of a LOAD reaches past the end of the stream\n");
                    return 2;
                }
                memcpy(area_ptr(area, off, nb), &w[i], nb);
                i += nw;
                break;
            }
            case OP_FILL: {
                const uint32_t area = word_at(w, n, i), off = word_at(w, n, i + 1), nb = word_at(w, n, i + 2);
                const uint32_t b = word_at(w, n, i + 3);
                i += 4;
                memset(area_ptr(area, off, nb), (int) (b & 0xFF), nb);
                break;
            }
            case OP_DUMP: {
                const uint32_t area = word_at(w, n, i), off = word_at(w, n, i + 1), nb = word_at(w, n, i + 2);
                i += 3;
                LAB_BARRIER();
                if (fwrite(area_ptr(area, off, nb), 1, nb, fo) != nb) {
                    printf("lab: error: the write of the dump file failed\n");
                    return 2;
                }
                break;
            }
            case OP_INSN: {
                const uint32_t form = word_at(w, n, i);
                const uint32_t a0 = word_at(w, n, i + 1), a1 = word_at(w, n, i + 2);
                const uint32_t a2 = word_at(w, n, i + 3), a3 = word_at(w, n, i + 4);
                i += 5;
                run_insn(form, a0, a1, a2, a3);
                break;
            }
            case OP_MARK: {
                const uint32_t tag = word_at(w, n, i++);
                if (fwrite(&tag, 4, 1, fo) != 1) {
                    printf("lab: error: the write of the dump file failed\n");
                    return 2;
                }
                break;
            }
            default:
                printf("lab: error: opcode %u at word %lu is not known\n", (unsigned) op, (unsigned long) (i - 1));
                return 2;
        }
    }
    fclose(fo);
    printf("lab: i8probe ops = %u words = %lu\n", (unsigned) n_ops, (unsigned long) n);
    lab_fini();
    return 0;
}
