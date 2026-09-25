// Target: op_concat of concat-ops.c against a scalar reference, byte for byte.
//
// The row copy (concat_rows) serves each concat whose three tensors have one type and contiguous rows, on one
// device. The other cases keep concat_generic or the transposed path. The program runs op_concat on shapes of
// each dimension (0 to 3), of the types f32, f16 and i32, with strided rows, and with a transposed src1 (which
// must not take the row copy). For each case it compares each byte of dst with the reference and checks a guard
// region of 256 bytes after dst. It also gives the cycles of the MTP concat of the 4B (two rows of 2560 floats
// for each token) at 22 and 1024 tokens, with the DMA shim below (the copies of a descriptor at the push, thus
// the transposed path costs its copies inside the op).
//
// The work queue of the lab runs the jobs of a call one after the other on one thread.
//
// Arguments: --iters 2
#include "lab.h"

#include <stdio.h>
#include <string.h>

// ---- the synchronous DMA shim of target_gdnk.c: the engine of the standalone runtime raises "No Access"
#define HTP_DMA_H

typedef struct {
    void *       dst;
    const void * src;
} dma_ptr;

#define LAB_DMA_CAPACITY 1024

typedef struct dma_queue_s {
    dma_ptr  ptr[LAB_DMA_CAPACITY];
    uint32_t push_idx;
    uint32_t pop_idx;
} dma_queue;
typedef dma_queue * dma_queue_t;

static inline dma_ptr dma_make_ptr(void * dst, const void * src) {
    dma_ptr p = { dst, src };
    return p;
}

static inline bool dma_queue_push(dma_queue * q, dma_ptr p, size_t dst_stride, size_t src_stride,
                                  size_t row_size, size_t nrows) {
    if (q->push_idx - q->pop_idx >= LAB_DMA_CAPACITY) {
        printf("lab: error: the DMA shim queue is full\n");
        return false;
    }
    for (size_t r = 0; r < nrows; r++) {
        memcpy((uint8_t *) p.dst + r * dst_stride, (const uint8_t *) p.src + r * src_stride, row_size);
    }
    q->ptr[q->push_idx & (LAB_DMA_CAPACITY - 1)] = p;
    q->push_idx++;
    return true;
}

static inline dma_ptr dma_queue_pop(dma_queue * q) {
    dma_ptr p = { NULL, NULL };
    if (q->pop_idx == q->push_idx) {
        return p;
    }
    p = q->ptr[q->pop_idx & (LAB_DMA_CAPACITY - 1)];
    q->pop_idx++;
    return p;
}

static inline void dma_queue_flush(dma_queue * q) {
    q->pop_idx = q->push_idx;
}

#include "concat-ops.c"

#define TARGET "concat"
#define GUARD  256

static struct htp_context g_ctx;
static dma_queue          g_dma[HTP_MAX_NTHREADS];

// One case: the type, the dimension, the shapes of src0 and src1, the row padding of src0 in elements, and
// whether src1 is transposed (its first two dimensions swapped in memory).
struct concat_case {
    const char * name;
    uint32_t     type;
    int          dim;
    uint32_t     a[4];
    uint32_t     b[4];
    uint32_t     pad0;
    int          b_transposed;
};

static uint32_t type_size(uint32_t type) {
    return type == HTP_TYPE_F16 ? 2 : 4;
}

// A tensor with the shape ne, rows of ne[0] + pad elements, and the other dimensions contiguous over those rows.
// A transposed tensor stores dimension 1 fastest (nb[1] is the element size). O(1).
static void set_tensor(struct htp_tensor * t, void * data, uint32_t type, const uint32_t ne[4], uint32_t pad,
                       int transposed) {
    const uint32_t ts = type_size(type);
    memset(t, 0, sizeof(*t));
    t->data = (uint32_t) (uintptr_t) data;
    t->type = type;
    for (int i = 0; i < 4; i++) {
        t->ne[i] = ne[i];
    }
    if (transposed) {
        t->nb[1] = ts;
        t->nb[0] = ts * ne[1];
    } else {
        t->nb[0] = ts;
        t->nb[1] = ts * (ne[0] + pad);
    }
    t->nb[2] = transposed ? ts * ne[0] * ne[1] : t->nb[1] * ne[1];
    t->nb[3] = t->nb[2] * ne[2];
    t->size  = t->nb[3] * ne[3];
}

// The byte offset of an element of a tensor. O(1).
static size_t offs(const struct htp_tensor * t, uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
    return (size_t) i0 * t->nb[0] + (size_t) i1 * t->nb[1] + (size_t) i2 * t->nb[2] + (size_t) i3 * t->nb[3];
}

// The reference: each element of dst from src0 or src1 along dim, as ggml_concat defines it. O(elements).
static void ref_concat(uint8_t * out, const struct htp_tensor * d, const struct htp_tensor * s0,
                       const struct htp_tensor * s1, int dim) {
    const uint32_t ts = type_size(d->type);
    for (uint32_t i3 = 0; i3 < d->ne[3]; i3++) {
        for (uint32_t i2 = 0; i2 < d->ne[2]; i2++) {
            for (uint32_t i1 = 0; i1 < d->ne[1]; i1++) {
                for (uint32_t i0 = 0; i0 < d->ne[0]; i0++) {
                    uint32_t idx[4] = { i0, i1, i2, i3 };
                    const struct htp_tensor * s = s0;
                    if (idx[dim] >= s0->ne[dim]) {
                        idx[dim] -= s0->ne[dim];
                        s = s1;
                    }
                    memcpy(out + offs(d, i0, i1, i2, i3),
                           (const uint8_t *) (uintptr_t) s->data + offs(s, idx[0], idx[1], idx[2], idx[3]), ts);
                }
            }
        }
    }
}

// Runs one case. Returns 1 when each byte of dst equals the reference and the guard is intact, else 0.
static int run_case(const struct concat_case * c, uint32_t nth, uint32_t iters, int report_cycles) {
    const uint32_t ts = type_size(c->type);
    uint32_t dne[4];
    for (int i = 0; i < 4; i++) {
        dne[i] = i == c->dim ? c->a[i] + c->b[i] : c->a[i];
    }
    struct htp_tensor s0, s1, d;
    set_tensor(&s0, NULL, c->type, c->a, c->pad0, 0);
    set_tensor(&s1, NULL, c->type, c->b, 0, c->b_transposed);
    set_tensor(&d, NULL, c->type, dne, 0, 0);

    uint8_t * p0 = lab_ddr_alloc(s0.size + GUARD, 128);
    uint8_t * p1 = lab_ddr_alloc(s1.size + GUARD, 128);
    uint8_t * pd = lab_ddr_alloc(d.size + GUARD, 128);
    uint8_t * pr = lab_ddr_alloc(d.size + GUARD, 128);
    lab_fill_u8(p0, s0.size);
    lab_fill_u8(p1, s1.size);
    memset(pd, 0x5a, d.size + GUARD);
    memset(pr, 0x5a, d.size + GUARD);
    s0.data = (uint32_t) (uintptr_t) p0;
    s1.data = (uint32_t) (uintptr_t) p1;
    d.data  = (uint32_t) (uintptr_t) pd;
    ref_concat(pr, &d, &s0, &s1, c->dim);

    struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx           = &g_ctx;
    octx.op            = HTP_OP_CONCAT;
    octx.n_threads     = nth;
    octx.n_threads_div = g_ctx.n_threads_div;
    octx.op_params[0]  = c->dim;
    octx.src[0]        = &s0;
    octx.src[1]        = &s1;
    octx.dst           = &d;

    uint64_t best = UINT64_MAX;
    int status = HTP_STATUS_OK;
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        status = op_concat(&octx);
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        best = (t1 - t0) < best ? (t1 - t0) : best;
        if (status != HTP_STATUS_OK) {
            break;
        }
    }
    size_t n_diff = 0;
    for (size_t i = 0; i < d.size + GUARD; i++) {
        n_diff += pd[i] != pr[i];
    }
    const int pass = status == HTP_STATUS_OK && n_diff == 0;
    printf("lab: %s %-34s dim %d type size %u dst %ux%ux%ux%u: %s (status %d, %zu bytes differ)", TARGET, c->name,
           c->dim, ts, dne[0], dne[1], dne[2], dne[3], pass ? "PASS" : "FAIL", status, n_diff);
    if (report_cycles) {
        printf(" %llu cycles, %.2f bytes of dst per cycle", (unsigned long long) best, (double) d.size / (double) best);
    }
    printf("\n");
    return pass;
}

int main(int argc, char ** argv) {
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 2);
    const uint32_t nth   = 6;

    lab_init();
    memset(&g_ctx, 0, sizeof(g_ctx));
    for (uint32_t i = 0; i < HTP_MAX_NTHREADS; i++) {
        g_ctx.dma[i]        = &g_dma[i];
        g_ctx.dma_cached[i] = &g_dma[i];
    }
    g_ctx.n_threads     = nth;
    g_ctx.n_threads_div = init_fastdiv_values(nth);
    g_ctx.vtcm_size     = lab_vtcm_size() - 4096;
    g_ctx.vtcm_base     = lab_vtcm_alloc(g_ctx.vtcm_size, 2048);
    g_ctx.mdev.count    = 1;

    static const struct concat_case cases[] = {
        { "mtp 1 token",                  HTP_TYPE_F32, 0, { 2560, 1, 1, 1 },  { 2560, 1, 1, 1 },  0, 0 },
        { "mtp 5 tokens",                 HTP_TYPE_F32, 0, { 2560, 5, 1, 1 },  { 2560, 5, 1, 1 },  0, 0 },
        { "dim 0 odd rows 4d",            HTP_TYPE_F32, 0, { 7, 3, 2, 2 },     { 33, 3, 2, 2 },    0, 0 },
        { "dim 0 f16 4d",                 HTP_TYPE_F16, 0, { 5, 4, 3, 1 },     { 64, 4, 3, 1 },    0, 0 },
        { "dim 0 i32 padded src0 rows",   HTP_TYPE_I32, 0, { 40, 6, 1, 1 },    { 24, 6, 1, 1 },    9, 0 },
        { "dim 1 f32",                    HTP_TYPE_F32, 1, { 40, 3, 2, 1 },    { 40, 5, 2, 1 },    0, 0 },
        { "dim 1 f16 padded src0 rows",   HTP_TYPE_F16, 1, { 70, 2, 3, 1 },    { 70, 4, 3, 1 },    3, 0 },
        { "dim 2 f32",                    HTP_TYPE_F32, 2, { 17, 2, 3, 2 },    { 17, 2, 1, 2 },    0, 0 },
        { "dim 3 f32",                    HTP_TYPE_F32, 3, { 129, 2, 2, 1 },   { 129, 2, 2, 3 },   0, 0 },
        { "dim 0 transposed src1 (old path)", HTP_TYPE_F32, 0, { 64, 40, 1, 1 }, { 32, 40, 1, 1 }, 0, 1 },
    };
    int all = 1;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        all &= run_case(&cases[i], nth, 1, 0);
    }
    // The MTP concat of the 4B at the token counts of a short call and of a prefill ubatch
    static const struct concat_case timed[] = {
        { "mtp 22 tokens",   HTP_TYPE_F32, 0, { 2560, 22, 1, 1 },   { 2560, 22, 1, 1 },   0, 0 },
        { "mtp 1024 tokens", HTP_TYPE_F32, 0, { 2560, 1024, 1, 1 }, { 2560, 1024, 1, 1 }, 0, 0 },
    };
    for (size_t i = 0; i < sizeof(timed) / sizeof(timed[0]); i++) {
        all &= run_case(&timed[i], nth, iters, 1);
    }
    printf("lab: %s check %s\n", TARGET, all ? "PASS" : "FAIL");
    return all ? 0 : 1;
}
