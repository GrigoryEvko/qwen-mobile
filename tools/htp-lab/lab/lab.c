// The common runtime of the kernel lab. Refer to lab.h.
#include "lab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <hexagon_standalone.h>

#define LAB_MAX_THREADS 8
#define LAB_THREAD_STACK (256 * 1024)
#define LAB_MAX_GUARDED 64
#define LAB_MAX_ARG_NAMES 64
#define LAB_MAX_LIMITS 32
#define LAB_GUARD_BYTE 0xA5

static uint8_t * g_vtcm_base;
static size_t    g_vtcm_size;
static size_t    g_vtcm_used;
static uint32_t  g_rand_state = 0x9E3779B9u;

static uint8_t g_stacks[LAB_MAX_THREADS][LAB_THREAD_STACK] __attribute__((aligned(128)));

// One guarded allocation: the block that the allocator gave, the buffer that the caller sees, and
// the bytes of that buffer.
struct lab_guarded {
    uint8_t * block;
    uint8_t * buffer;
    size_t    bytes;
};

static struct lab_guarded g_guarded[LAB_MAX_GUARDED];
static size_t             g_guarded_count;

static const char * g_arg_names[LAB_MAX_ARG_NAMES];
static bool         g_arg_flags[LAB_MAX_ARG_NAMES];
static int          g_argc;
static char **      g_argv;
static bool         g_args_checked;
static size_t       g_arg_name_count;

static const char * g_limits[LAB_MAX_LIMITS];
static size_t       g_limit_count;

static void lab_at_exit(void);

// The standalone runtime maps a page on the first access with the default attributes. The lab
// maps the VTCM pages explicitly with the cacheable attribute (L1 write-back, L2), which is the
// attribute that the read and write tests of the lab use.
static void lab_map_vtcm(void) {
    const uint32_t base = __rdcfg(__vtcm_base) << 16;
    const uint32_t size_kb = __rdcfg(__tcm_size);
    g_vtcm_base = (uint8_t *) (uintptr_t) base;
    g_vtcm_size = (size_t) size_kb * 1024;
    for (size_t off = 0; off < g_vtcm_size; off += HEXAGON_DEFAULT_PAGE_SIZE) {
        add_translation((void *) (base + off), (void *) (base + off), HEXAGON_DEFAULT_PAGE_CACHE);
    }
    g_vtcm_used = 0;
}

void lab_hmx_enable(void) {
    uint32_t ssr;
    __asm__ volatile("%0 = ssr" : "=r"(ssr));
    ssr |= (1u << 26);
    __asm__ volatile("ssr = %0\n isync\n" : : "r"(ssr));
}

// The limits that every lab run has, whatever the target. They come from the build flags and from
// the simulator, thus no target can remove them.
static void lab_record_build_limits(void) {
    lab_limit("A green lab result is necessary and not sufficient. A kernel patch needs one "
              "generation on the device before it enters patches/series.");
#ifdef LAB_LTO
    (void) 0;
#else
    lab_limit("This build has no -flto. The DSP library ships with -flto, thus a defect that only "
              "an inlined copy of a kernel holds does not appear here. PROFILE=release builds with "
              "-flto.");
#endif
#ifdef NDEBUG
    lab_limit("NDEBUG is set, thus assert() does not run. PROFILE=debug keeps the asserts.");
#endif
    lab_limit("The simulator gives an exact zero for a qf product with a zero operand. The v79 "
              "silicon gives the constant qf32 0x00000080, which is 2^-22. An exact zero must not "
              "go into a qf multiply-add chain.");
    lab_limit("The timing model of this SDK does not retire an HMX instruction, thus an HMX path "
              "has a value result and no cycle number.");
    lab_limit("The random data of lab_fill_f32 is uniform. The model reaches values that uniform "
              "data does not: refer to lab_fill_f32_specials and lab_fill_f32_octaves.");
}

void lab_init(void) {
    lab_map_vtcm();
    lab_hmx_enable();
    g_rand_state = 0x9E3779B9u;
    printf("lab: core vtcm_base = 0x%08lx vtcm_size = %lu KB threads = %lu hvx_contexts = %lu\n",
           (unsigned long) (uintptr_t) g_vtcm_base, (unsigned long) (g_vtcm_size / 1024),
           (unsigned long) __builtin_popcount(__rdcfg(__thread_mask)), (unsigned long) __rdcfg(__coproc_ctx));
#ifdef LAB_PROFILE_NAME
    printf("lab: build profile = %s\n", LAB_PROFILE_NAME);
#endif
    lab_record_build_limits();
    atexit(lab_at_exit);
}

uint8_t * lab_vtcm_base(void) {
    return g_vtcm_base;
}

size_t lab_vtcm_size(void) {
    return g_vtcm_size;
}

void * lab_vtcm_alloc(size_t bytes, size_t align) {
    size_t off = (g_vtcm_used + align - 1) & ~(align - 1);
    if (off + bytes > g_vtcm_size) {
        printf("lab: error: the VTCM is full (%lu of %lu bytes used, %lu requested)\n",
               (unsigned long) g_vtcm_used, (unsigned long) g_vtcm_size, (unsigned long) bytes);
        exit(2);
    }
    g_vtcm_used = off + bytes;
    return g_vtcm_base + off;
}

void * lab_ddr_alloc(size_t bytes, size_t align) {
    void * p = NULL;
    if (posix_memalign(&p, align < sizeof(void *) ? sizeof(void *) : align, bytes) != 0 || p == NULL) {
        printf("lab: error: the DDR allocation of %lu bytes failed\n", (unsigned long) bytes);
        exit(2);
    }
    memset(p, 0, bytes);
    return p;
}

void * lab_ddr_alloc_guarded(size_t bytes, size_t align) {
    if (g_guarded_count == LAB_MAX_GUARDED) {
        printf("lab: error: the guarded allocations are %d, which is the maximum\n", LAB_MAX_GUARDED);
        exit(2);
    }
    if (align < LAB_GUARD_BYTES) {
        align = LAB_GUARD_BYTES;
    }
    // The front guard is one alignment unit, thus the buffer keeps the alignment of the caller.
    uint8_t * block = (uint8_t *) lab_ddr_alloc(bytes + 2 * align, align);
    uint8_t * buffer = block + align;
    memset(block, LAB_GUARD_BYTE, align);
    memset(buffer + bytes, LAB_GUARD_BYTE, align);
    memset(buffer, 0, bytes);
    g_guarded[g_guarded_count].block  = block;
    g_guarded[g_guarded_count].buffer = buffer;
    g_guarded[g_guarded_count].bytes  = bytes;
    g_guarded_count++;
    return buffer;
}

// Counts the bytes of one guard region that do not hold the pattern. Complexity O(n).
static size_t lab_guard_scan(const char * what, const char * side, const uint8_t * p, size_t n) {
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (p[i] != LAB_GUARD_BYTE) {
            if (bad == 0) {
                printf("lab: guard %s %s: byte %lu holds 0x%02x and not 0x%02x\n", what, side,
                       (unsigned long) i, (unsigned) p[i], (unsigned) LAB_GUARD_BYTE);
            }
            bad++;
        }
    }
    return bad;
}

size_t lab_guard_check(const char * what, const void * buffer) {
    for (size_t i = 0; i < g_guarded_count; i++) {
        if (g_guarded[i].buffer != (const uint8_t *) buffer) {
            continue;
        }
        const size_t front = (size_t) (g_guarded[i].buffer - g_guarded[i].block);
        size_t       bad   = lab_guard_scan(what, "before", g_guarded[i].block, front);
        bad += lab_guard_scan(what, "after", g_guarded[i].buffer + g_guarded[i].bytes, front);
        if (bad != 0) {
            printf("lab: check guard %s: %lu guard bytes changed\n", what, (unsigned long) bad);
        }
        return bad;
    }
    printf("lab: error: %p is not a guarded buffer\n", buffer);
    exit(2);
}

size_t lab_guard_check_all(void) {
    size_t bad = 0;
    for (size_t i = 0; i < g_guarded_count; i++) {
        if (lab_guard_check("buffer", g_guarded[i].buffer) != 0) {
            bad++;
        }
    }
    return bad;
}

uint32_t lab_rand_u32(void) {
    uint32_t x = g_rand_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rand_state = x;
    return x;
}

float lab_rand_f32(float lo, float hi) {
    const float u = (float) (lab_rand_u32() >> 8) * (1.0f / 16777216.0f);
    return lo + (hi - lo) * u;
}

void lab_fill_f32(float * dst, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = lab_rand_f32(lo, hi);
    }
}

void lab_fill_u8(uint8_t * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = (uint8_t) lab_rand_u32();
    }
}

void lab_fill_f32_specials(float * dst, size_t n, float lo, float hi, size_t period) {
    static const uint32_t bits[] = {
        0x00000000u,  // +0
        0x80000000u,  // -0
        0x00000001u,  // the smallest subnormal
        0x7149F2CAu,  // 1e30
        0x7F800000u,  // +Inf
        0xFF800000u,  // -Inf
        0x7FC00000u,  // a NaN
        0x4167D70Au,  // 14.49, the low end of the band that gives +Inf in a defective int16 path
        0x41748F5Cu,  // 15.285, inside that band
        0x417BA5E3u,  // 15.7275, the high end of that band
    };
    const size_t n_bits = sizeof(bits) / sizeof(bits[0]);
    size_t       k      = 0;
    for (size_t i = 0; i < n; i++) {
        if (period != 0 && i % period == 0) {
            memcpy(&dst[i], &bits[k % n_bits], sizeof(float));
            k++;
        } else {
            dst[i] = lab_rand_f32(lo, hi);
        }
    }
}

void lab_fill_f32_octaves(float * dst, size_t n) {
    // The octaves of the f16 range: the smallest normal is 2^-14 and the largest value is below
    // 2^16. Eight significands of each octave give the values between the powers of two.
    const int octaves = 31;
    for (size_t i = 0; i < n; i++) {
        const int    e    = -14 + (int) (i % (size_t) octaves);
        const double m    = 1.0 + (double) ((i / (size_t) octaves) % 8) / 8.0;
        const double sign = (i & 1) ? -1.0 : 1.0;
        dst[i] = (float) (sign * m * ldexp(1.0, e));
    }
}

size_t lab_tail_sweep(size_t index, size_t n_max) {
    static const size_t table[] = { 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129 };
    const size_t        count   = sizeof(table) / sizeof(table[0]);
    size_t              kept    = 0;
    for (size_t i = 0; i < count; i++) {
        if (table[i] > n_max) {
            continue;
        }
        if (kept == index) {
            return table[i];
        }
        kept++;
    }
    if (index != kept) {
        return 0;
    }
    for (size_t i = 0; i < count; i++) {
        if (table[i] == n_max) {
            return 0;
        }
    }
    return n_max;
}

size_t lab_f16_overflow_count(const float * v, size_t n) {
    size_t over = 0;
    for (size_t i = 0; i < n; i++) {
        if (fabsf(v[i]) > LAB_F16_MAX) {
            over++;
        }
    }
    return over;
}

size_t lab_compare_f32(const char * what, const float * got, const float * ref, size_t n, float abs_tol, float rel_tol) {
    size_t bad = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float d = fabsf(got[i] - ref[i]);
        const float r = fabsf(ref[i]);
        if (d > max_abs) {
            max_abs = d;
        }
        if (r > 0.0f && d / r > max_rel) {
            max_rel = d / r;
        }
        if (d > abs_tol + rel_tol * r || got[i] != got[i]) {
            if (bad < 5) {
                printf("lab: mismatch %s[%lu]: got %g expected %g\n", what, (unsigned long) i, got[i], ref[i]);
            }
            bad++;
        }
    }
    printf("lab: check %s: %lu of %lu outside tolerance, max abs error %g, max rel error %g\n",
           what, (unsigned long) bad, (unsigned long) n, max_abs, max_rel);
    return bad;
}

void lab_metric_start(lab_metric * m, const char * what) {
    memset(m, 0, sizeof(*m));
    m->what = what;
}

void lab_metric_add(lab_metric * m, double got, double ref, size_t index, double input) {
    m->n++;
    if (!isfinite(got)) {
        m->n_nonfinite++;
    }
    m->sum_sq_ref += ref * ref;
    const double err = fabs(got - ref);
    if (isfinite(err)) {
        m->sum_sq_err += err * err;
    }
    if (err > m->worst_abs) {
        m->worst_abs = err;
        m->abs_index = index;
        m->abs_got   = got;
        m->abs_ref   = ref;
        m->abs_input = input;
    }
    if (ref == 0.0) {
        m->n_zero_ref++;
        return;
    }
    const double rel = err / fabs(ref);
    if (rel > m->worst_rel) {
        m->worst_rel = rel;
        m->rel_index = index;
        m->rel_got   = got;
        m->rel_ref   = ref;
        m->rel_input = input;
    }
}

size_t lab_metric_report(const lab_metric * m, const char * target, double abs_tol, double rel_tol) {
    const double nmse = m->sum_sq_ref > 0.0 ? m->sum_sq_err / m->sum_sq_ref : m->sum_sq_err;
    printf("lab: %s %s n = %lu nmse = %.6g worst_abs = %.6g worst_rel = %.6g zero_ref = %lu nonfinite = %lu\n",
           target, m->what, (unsigned long) m->n, nmse, m->worst_abs, m->worst_rel,
           (unsigned long) m->n_zero_ref, (unsigned long) m->n_nonfinite);
    printf("lab: %s %s worst_abs at [%lu]: got %.9g ref %.9g input %.9g\n", target, m->what,
           (unsigned long) m->abs_index, m->abs_got, m->abs_ref, m->abs_input);
    printf("lab: %s %s worst_rel at [%lu]: got %.9g ref %.9g input %.9g\n", target, m->what,
           (unsigned long) m->rel_index, m->rel_got, m->rel_ref, m->rel_input);
    size_t bad = m->n_nonfinite;
    if (m->worst_abs > abs_tol + rel_tol * fabs(m->abs_ref)) {
        bad++;
    }
    if (m->worst_rel > rel_tol && m->worst_abs > abs_tol) {
        bad++;
    }
    return bad;
}

uint64_t lab_fnv1a_update(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ b[i]) * 0x100000001B3ull;
    }
    return h;
}

uint64_t lab_fnv1a(const void * p, size_t n) {
    return lab_fnv1a_update(LAB_FNV1A_BASIS, p, n);
}

void lab_report(const char * target, const char * key, double value, const char * unit) {
    printf("lab: %s %s = %.6g %s\n", target, key, value, unit);
}

void lab_limit(const char * text) {
    for (size_t i = 0; i < g_limit_count; i++) {
        if (g_limits[i] == text || strcmp(g_limits[i], text) == 0) {
            return;
        }
    }
    if (g_limit_count == LAB_MAX_LIMITS) {
        return;
    }
    g_limits[g_limit_count++] = text;
}

void lab_limits_report(void) {
    printf("lab: limits %lu (a green result of this run does not cover them)\n", (unsigned long) g_limit_count);
    for (size_t i = 0; i < g_limit_count; i++) {
        printf("lab: limit %lu: %s\n", (unsigned long) i + 1, g_limits[i]);
    }
}

struct lab_thread_arg {
    lab_thread_fn fn;
    void *        data;
    unsigned int  nth;
    unsigned int  ith;
};

static struct lab_thread_arg g_thread_args[LAB_MAX_THREADS];

static void lab_thread_entry(void * p) {
    struct lab_thread_arg * a = (struct lab_thread_arg *) p;
    if (!acquire_vector_unit(HEXAGON_VECTOR_WAIT)) {
        printf("lab: error: thread %u did not get an HVX context\n", a->ith);
        return;
    }
    a->fn(a->nth, a->ith, a->data);
    release_vector_unit();
}

void lab_run_threads(lab_thread_fn fn, void * data, unsigned int n) {
    if (n <= 1) {
        fn(1, 0, data);
        return;
    }
    if (n > LAB_MAX_THREADS) {
        printf("lab: error: %u threads requested, the maximum is %d\n", n, LAB_MAX_THREADS);
        exit(2);
    }
    int mask = 0;
    for (unsigned int i = 1; i < n; i++) {
        g_thread_args[i].fn   = fn;
        g_thread_args[i].data = data;
        g_thread_args[i].nth  = n;
        g_thread_args[i].ith  = i;
        thread_create(lab_thread_entry, g_stacks[i] + LAB_THREAD_STACK, (int) i, &g_thread_args[i]);
        mask |= 1 << i;
    }
    fn(n, 0, data);
    thread_join(mask);
}

// Records an option name and the argument vector, thus the check of the options knows the names that
// the program reads. A flag is an option with no value.
static void lab_arg_seen(int argc, char ** argv, const char * name, bool flag) {
    if (g_argv == NULL) {
        g_argc = argc;
        g_argv = argv;
    }
    for (size_t i = 0; i < g_arg_name_count; i++) {
        if (strcmp(g_arg_names[i], name) == 0) {
            return;
        }
    }
    if (g_arg_name_count == LAB_MAX_ARG_NAMES) {
        return;
    }
    g_arg_flags[g_arg_name_count]   = flag;
    g_arg_names[g_arg_name_count++] = name;
}

// Returns the value of "--name value", or NULL when the option is not in argv
static const char * lab_arg_find(int argc, char ** argv, const char * name) {
    lab_arg_seen(argc, argv, name, false);
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return NULL;
}

long lab_arg_long(int argc, char ** argv, const char * name, long def) {
    const char * v = lab_arg_find(argc, argv, name);
    return v ? strtol(v, NULL, 0) : def;
}

double lab_arg_double(int argc, char ** argv, const char * name, double def) {
    const char * v = lab_arg_find(argc, argv, name);
    return v ? strtod(v, NULL) : def;
}

const char * lab_arg_str(int argc, char ** argv, const char * name, const char * def) {
    const char * v = lab_arg_find(argc, argv, name);
    return v ? v : def;
}

bool lab_arg_flag(int argc, char ** argv, const char * name) {
    lab_arg_seen(argc, argv, name, true);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) {
            return true;
        }
    }
    return false;
}

// Prints each argument that starts with "--" and that no lab_arg_* call read. Returns their number.
// Complexity O(argc * names).
static size_t lab_args_unknown(int argc, char ** argv) {
    size_t bad = 0;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-' || argv[i][1] != '-') {
            continue;
        }
        long known = -1;
        for (size_t k = 0; k < g_arg_name_count; k++) {
            if (strcmp(g_arg_names[k], argv[i]) == 0) {
                known = (long) k;
                break;
            }
        }
        if (known < 0) {
            printf("lab: error: this program does not read the option %s, thus the run did not use it\n", argv[i]);
            bad++;
        } else if (!g_arg_flags[known]) {
            i++;  // the value of the option
        }
    }
    if (bad != 0) {
        printf("lab: the options that this run of the program reads are:");
        for (size_t k = 0; k < g_arg_name_count; k++) {
            printf(" %s", g_arg_names[k]);
        }
        printf("\n");
    }
    return bad;
}

void lab_args_done(int argc, char ** argv) {
    g_args_checked = true;
    if (lab_args_unknown(argc, argv) != 0) {
        exit(2);
    }
}

// The exit handler of every lab program: the check of the options, then the limits block. A program
// that read its options with lab_arg_* and got one that it did not read ends with the status 2, thus
// a misspelled option name cannot take the preset value in silence.
static void lab_at_exit(void) {
    const size_t bad = (g_argv != NULL && !g_args_checked) ? lab_args_unknown(g_argc, g_argv) : 0;
    lab_limits_report();
    if (bad != 0) {
        fflush(stdout);
        _Exit(2);
    }
}

float lab_hf_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t lab_f32_to_hf(float f) {
    __fp16 h = (__fp16) f;
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}
