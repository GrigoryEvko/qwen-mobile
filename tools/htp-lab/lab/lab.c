// The common runtime of the kernel lab. Refer to lab.h.
#include "lab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <hexagon_standalone.h>

#define LAB_MAX_THREADS 8
#define LAB_THREAD_STACK (256 * 1024)

static uint8_t * g_vtcm_base;
static size_t    g_vtcm_size;
static size_t    g_vtcm_used;
static uint32_t  g_rand_state = 0x9E3779B9u;

static uint8_t g_stacks[LAB_MAX_THREADS][LAB_THREAD_STACK] __attribute__((aligned(128)));

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

void lab_init(void) {
    lab_map_vtcm();
    lab_hmx_enable();
    g_rand_state = 0x9E3779B9u;
    printf("lab: core vtcm_base = 0x%08lx vtcm_size = %lu KB threads = %lu hvx_contexts = %lu\n",
           (unsigned long) (uintptr_t) g_vtcm_base, (unsigned long) (g_vtcm_size / 1024),
           (unsigned long) __builtin_popcount(__rdcfg(__thread_mask)), (unsigned long) __rdcfg(__coproc_ctx));
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

void lab_report(const char * target, const char * key, double value, const char * unit) {
    printf("lab: %s %s = %.6g %s\n", target, key, value, unit);
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

long lab_arg_long(int argc, char ** argv, const char * name, long def) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0) {
            return strtol(argv[i + 1], NULL, 0);
        }
    }
    return def;
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
