// Target 8: the HVX flash attention of flash-attn-ops.c (flash_attn_ext_f16_thread).
//
// The program includes the kernel file verbatim and calls its thread function, thus the
// measurement covers the kernel, its DMA queue and its per-thread row loop, and not the FastRPC
// path or the op batch of the phone. Q, K, V, the mask and the output are in DDR (the simulator
// memory) and the scratchpads are in the VTCM, as on the phone.
//
// On the phone this op measures 31.1 ms of a 512-token prefill of the 4B at 2.11 instructions per
// packet, with IU_NO_PKT 11.0 %, the highest instruction-fetch stall of the whole profile. The
// static cause is the size of the function: flash_attn_ext_f16_thread compiles to 21472 bytes in
// one body, whose inner block loop alone is 8924 bytes. This target measures what that costs.
//
// The HMX path (hmx_flash_attn_ext) is NOT measured here: the simulator runs HMX instructions
// functionally but its timing model never retires them, thus that path has no cycle story.
//
// Arguments: --dk 128 --dv 128 --nkv 256 --heads 4 --kvheads 4 --tokens 1 --threads 1 --iters 3
//            --mask 1 --probe 0
// --probe 1 runs a DMA sanity check only: the kernel needs the user DMA engine of the core, and
// this mode reports whether the simulator completes a descriptor at all.
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

// The synchronous DMA shim of the lab. The kernel moves K, V and the mask with the user DMA
// engine of the core (dmstart / dmpoll inline asm in dma-queue.h). The standalone runtime of the
// simulator does not grant that thread access to the engine: a descriptor raises the exception
// 0x28, "No Access". Thus the lab defines the guard of dma-queue.h and supplies a FIFO whose push
// copies at once and whose pop returns the destinations in the order of the pushes, which is the
// order contract the kernel depends on. The copies then appear as ordinary loads and stores, thus
// the cycle numbers of this target hold the transfer cost in the kernel rather than beside it.
// Every other behaviour, and every instruction of the compute path, is that of the phone.
#define HTP_DMA_H

typedef struct {
    void *       dst;
    const void * src;
} dma_ptr;

#define LAB_DMA_CAPACITY 256

typedef struct dma_queue_s {
    void *   dst[LAB_DMA_CAPACITY];
    uint32_t push_idx;
    uint32_t pop_idx;
} dma_queue;
typedef dma_queue * dma_queue_t;

static inline dma_ptr dma_make_ptr(void * dst, const void * src) {
    dma_ptr p = { dst, src };
    return p;
}

// Copies nrows rows of row_size bytes at once, then records the destination for the pop.
// nrows == 0 is the dummy transfer that the line cache uses for a hit: it records only.
static inline bool dma_queue_push(dma_queue * q, dma_ptr p, size_t dst_stride, size_t src_stride,
                                  size_t row_size, size_t nrows) {
    for (size_t r = 0; r < nrows; r++) {
        memcpy((uint8_t *) p.dst + r * dst_stride, (const uint8_t *) p.src + r * src_stride, row_size);
    }
    q->dst[q->push_idx & (LAB_DMA_CAPACITY - 1)] = p.dst;
    q->push_idx++;
    return true;
}

static inline dma_ptr dma_queue_pop(dma_queue * q) {
    dma_ptr p = { NULL, NULL };
    if (q->pop_idx == q->push_idx) {
        return p;
    }
    p.dst = q->dst[q->pop_idx & (LAB_DMA_CAPACITY - 1)];
    q->pop_idx++;
    return p;
}

#define DMA_CACHE_MAX_SIZE 128

// The line cache of the mask, with the replacement rule of dma-queue.h: a hit refreshes the age
// and pushes a dummy transfer, a miss takes the oldest line and pushes a real one.
typedef struct {
    uint8_t * base;
    uint32_t  line_size;
    uint32_t  capacity;
    uint32_t  src[DMA_CACHE_MAX_SIZE];
    uint16_t  age[DMA_CACHE_MAX_SIZE];
} dma_cache;

static inline void dma_cache_init(dma_cache * c, uint8_t * base, uint32_t line_size, uint32_t capacity) {
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->base      = base;
    c->line_size = line_size;
    for (unsigned i = 0; i < c->capacity; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

static inline bool dma_cache_push(dma_queue * q, dma_cache * c, const uint8_t * src, uint32_t dst_stride,
                                  uint32_t src_stride, uint32_t row_size, uint32_t nrows) {
    uint32_t  o_idx = 0;
    uint16_t  o_age = 0;
    uint8_t * dst   = 0;
    for (unsigned i = 0; i < c->capacity; i++) {
        if (c->src[i] == (uint32_t) (uintptr_t) src) {
            c->age[i] = 0;
            dst = c->base + (i * c->line_size);
            nrows = 0;
        } else {
            c->age[i]++;
            if (c->age[i] > o_age) {
                o_age = c->age[i];
                o_idx = i;
            }
        }
    }
    if (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = (uint32_t) (uintptr_t) src;
        dst = c->base + o_idx * c->line_size;
    }
    return dma_queue_push(q, dma_make_ptr(dst, src), dst_stride, src_stride, row_size, nrows);
}

#include "flash-attn-ops.c"

#define TARGET "fa"

// The scalar reference in double, in the order of ggml_compute_forward_flash_attn_ext_f16.
// One q row: the scores against every key, the softmax, then the weighted sum of the values.
// O(n_kv * (DK + DV)) for each row.
static void ref_fa_row(const __fp16 * q, const __fp16 * k, const __fp16 * v, const __fp16 * mask,
                       float * out, uint32_t DK, uint32_t DV, uint32_t n_kv,
                       size_t k_stride, size_t v_stride, float scale, double * scores) {
    double m = -1e30;
    for (uint32_t j = 0; j < n_kv; j++) {
        double s = 0.0;
        const __fp16 * kr = (const __fp16 *) ((const uint8_t *) k + (size_t) j * k_stride);
        for (uint32_t d = 0; d < DK; d++) {
            s += (double) q[d] * (double) kr[d];
        }
        s *= (double) scale;
        if (mask) {
            s += (double) mask[j];
        }
        scores[j] = s;
        if (s > m) {
            m = s;
        }
    }
    double sum = 0.0;
    for (uint32_t j = 0; j < n_kv; j++) {
        scores[j] = exp(scores[j] - m);
        sum += scores[j];
    }
    for (uint32_t d = 0; d < DV; d++) {
        double acc = 0.0;
        for (uint32_t j = 0; j < n_kv; j++) {
            const __fp16 * vr = (const __fp16 *) ((const uint8_t *) v + (size_t) j * v_stride);
            acc += scores[j] * (double) vr[d];
        }
        out[d] = (float) (acc / sum);
    }
}

// One transfer through the shim: the copy must complete and the pop must give its destination.
static bool dma_probe(struct htp_context * ctx) {
    uint8_t * src = lab_ddr_alloc(256, 128);
    uint8_t * dst = (uint8_t *) lab_vtcm_alloc(256, 128);
    for (int i = 0; i < 256; i++) {
        src[i] = (uint8_t) (i * 7 + 1);
    }
    memset(dst, 0, 256);
    dma_queue_push(ctx->dma[0], dma_make_ptr(dst, src), 256, 256, 256, 1);
    dma_ptr got = dma_queue_pop(ctx->dma[0]);
    if (got.dst != dst) {
        printf("lab: %s dma probe: pop gave %p, expected %p\n", TARGET, got.dst, (void *) dst);
        return false;
    }
    return memcmp(dst, src, 256) == 0;
}

int main(int argc, char ** argv) {
    const uint32_t DK        = (uint32_t) lab_arg_long(argc, argv, "--dk", 128);
    const uint32_t DV        = (uint32_t) lab_arg_long(argc, argv, "--dv", 128);
    const uint32_t n_kv      = (uint32_t) lab_arg_long(argc, argv, "--nkv", 256);
    const uint32_t n_heads   = (uint32_t) lab_arg_long(argc, argv, "--heads", 4);
    const uint32_t n_kvheads = (uint32_t) lab_arg_long(argc, argv, "--kvheads", 4);
    const uint32_t n_tokens  = (uint32_t) lab_arg_long(argc, argv, "--tokens", 1);
    const uint32_t n_threads = (uint32_t) lab_arg_long(argc, argv, "--threads", 1);
    const uint32_t iters     = (uint32_t) lab_arg_long(argc, argv, "--iters", 3);
    const bool     use_mask  = lab_arg_long(argc, argv, "--mask", 1) != 0;
    const bool     probe     = lab_arg_long(argc, argv, "--probe", 0) != 0;

    lab_init();

    if (n_kv % FLASH_ATTN_BLOCK_SIZE) {
        printf("lab: %s needs --nkv a multiple of %d\n", TARGET, FLASH_ATTN_BLOCK_SIZE);
        return 2;
    }
    if (n_heads % n_kvheads) {
        printf("lab: %s needs --heads a multiple of --kvheads\n", TARGET);
        return 2;
    }

    static struct htp_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.vtcm_base = lab_vtcm_base();
    ctx.vtcm_size = lab_vtcm_size();
    ctx.n_threads = n_threads;
    ctx.n_threads_div = init_fastdiv_values(n_threads);

    // one queue of the shim for each thread
    static dma_queue lab_queues[HTP_MAX_NTHREADS];
    for (uint32_t i = 0; i < n_threads && i < HTP_MAX_NTHREADS; i++) {
        memset(&lab_queues[i], 0, sizeof(lab_queues[i]));
        ctx.dma[i] = &lab_queues[i];
    }

    if (probe) {
        const bool ok = dma_probe(&ctx);
        printf("lab: %s dma probe = %s\n", TARGET, ok ? "OK the simulator completes a descriptor" : "FAILED");
        lab_report(TARGET, "dma_probe_ok", ok ? 1 : 0, "");
        return ok ? 0 : 1;
    }

    // the tensors in DDR. Q is [DK, n_tokens, n_heads, 1], K and V are [D, n_kv, n_kvheads, 1].
    const size_t q_row   = (size_t) DK * sizeof(__fp16);
    const size_t k_row   = (size_t) DK * sizeof(__fp16);
    const size_t v_row   = (size_t) DV * sizeof(__fp16);
    const size_t n_q     = (size_t) n_tokens * n_heads;
    const size_t q_bytes = n_q * q_row + 256;
    const size_t k_bytes = (size_t) n_kv * n_kvheads * k_row + 256;
    const size_t v_bytes = (size_t) n_kv * n_kvheads * v_row + 256;
    const size_t m_bytes = (size_t) n_tokens * n_kv * sizeof(__fp16) + 256;
    const size_t o_bytes = n_q * DV * sizeof(float) + 256;

    __fp16 * q_data = lab_ddr_alloc(q_bytes, 128);
    __fp16 * k_data = lab_ddr_alloc(k_bytes, 128);
    __fp16 * v_data = lab_ddr_alloc(v_bytes, 128);
    __fp16 * m_data = use_mask ? lab_ddr_alloc(m_bytes, 128) : NULL;
    float  * o_data = lab_ddr_alloc(o_bytes, 128);
    float  * o_ref  = lab_ddr_alloc(n_q * DV * sizeof(float), 128);
    double * scores = lab_ddr_alloc((size_t) n_kv * sizeof(double), 128);

    // f16 inputs of a moderate range: the softmax of the kernel works in f16, thus a wide range
    // of scores would measure the f16 exponent range and not the kernel.
    for (size_t i = 0; i < n_q * DK; i++) {
        q_data[i] = (__fp16) lab_rand_f32(-1.0f, 1.0f);
    }
    for (size_t i = 0; i < (size_t) n_kv * n_kvheads * DK; i++) {
        k_data[i] = (__fp16) lab_rand_f32(-1.0f, 1.0f);
    }
    for (size_t i = 0; i < (size_t) n_kv * n_kvheads * DV; i++) {
        v_data[i] = (__fp16) lab_rand_f32(-1.0f, 1.0f);
    }
    if (m_data) {
        for (size_t i = 0; i < (size_t) n_tokens * n_kv; i++) {
            m_data[i] = (__fp16) lab_rand_f32(-1.0f, 0.0f);
        }
    }
    memset(o_data, 0, n_q * DV * sizeof(float));

    static struct htp_tensor tq, tk, tv, tm, td;
    memset(&tq, 0, sizeof(tq)); memset(&tk, 0, sizeof(tk)); memset(&tv, 0, sizeof(tv));
    memset(&tm, 0, sizeof(tm)); memset(&td, 0, sizeof(td));

    tq.data = (uint32_t) (uintptr_t) q_data; tq.type = HTP_TYPE_F16;
    tq.ne[0] = DK; tq.ne[1] = n_tokens; tq.ne[2] = n_heads; tq.ne[3] = 1;
    tq.nb[0] = sizeof(__fp16); tq.nb[1] = q_row; tq.nb[2] = q_row * n_tokens; tq.nb[3] = q_row * n_tokens * n_heads;

    tk.data = (uint32_t) (uintptr_t) k_data; tk.type = HTP_TYPE_F16;
    tk.ne[0] = DK; tk.ne[1] = n_kv; tk.ne[2] = n_kvheads; tk.ne[3] = 1;
    tk.nb[0] = sizeof(__fp16); tk.nb[1] = k_row; tk.nb[2] = k_row * n_kv; tk.nb[3] = k_row * n_kv * n_kvheads;

    tv.data = (uint32_t) (uintptr_t) v_data; tv.type = HTP_TYPE_F16;
    tv.ne[0] = DV; tv.ne[1] = n_kv; tv.ne[2] = n_kvheads; tv.ne[3] = 1;
    tv.nb[0] = sizeof(__fp16); tv.nb[1] = v_row; tv.nb[2] = v_row * n_kv; tv.nb[3] = v_row * n_kv * n_kvheads;

    if (m_data) {
        tm.data = (uint32_t) (uintptr_t) m_data; tm.type = HTP_TYPE_F16;
        tm.ne[0] = n_kv; tm.ne[1] = n_tokens; tm.ne[2] = 1; tm.ne[3] = 1;
        tm.nb[0] = sizeof(__fp16); tm.nb[1] = n_kv * sizeof(__fp16);
        tm.nb[2] = tm.nb[1] * n_tokens; tm.nb[3] = tm.nb[2];
    }

    // dst is permuted: [DV, n_heads, n_tokens, 1], thus nb[1] is the head stride
    td.data = (uint32_t) (uintptr_t) o_data; td.type = HTP_TYPE_F32;
    td.ne[0] = DV; td.ne[1] = n_heads; td.ne[2] = n_tokens; td.ne[3] = 1;
    td.nb[0] = sizeof(float); td.nb[1] = DV * sizeof(float);
    td.nb[2] = td.nb[1] * n_heads; td.nb[3] = td.nb[2] * n_tokens;

    static struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx = &ctx;
    octx.n_threads = n_threads;
    octx.n_threads_div = init_fastdiv_values(n_threads);
    octx.src[0] = &tq; octx.src[1] = &tk; octx.src[2] = &tv;
    octx.src[3] = m_data ? &tm : NULL; octx.src[4] = NULL;
    octx.dst = &td;

    const float scale = 1.0f / sqrtf((float) DK);

    static struct htp_fa_context factx;
    memset(&factx, 0, sizeof(factx));
    factx.octx = &octx;
    factx.k = &tk;
    factx.v = &tv;
    factx.src0_div21 = init_fastdiv_values(n_heads * n_tokens);
    factx.src0_div1  = init_fastdiv_values(n_tokens);
    factx.broadcast_rk2 = init_fastdiv_values(n_heads / n_kvheads);
    factx.broadcast_rk3 = init_fastdiv_values(1);
    factx.broadcast_rv2 = init_fastdiv_values(n_heads / n_kvheads);
    factx.broadcast_rv3 = init_fastdiv_values(1);
    factx.src3_div2 = init_fastdiv_values(1);
    factx.src3_div3 = init_fastdiv_values(1);
    factx.is_q_fp32 = false;
    factx.size_q_row_padded = hex_round_up(q_row, 128);
    factx.size_k_row_padded = hex_round_up(k_row, 128);
    factx.size_v_row_padded = hex_round_up(v_row, 128);
    factx.size_k_block = factx.size_k_row_padded * FLASH_ATTN_BLOCK_SIZE;
    factx.size_v_block = factx.size_v_row_padded * FLASH_ATTN_BLOCK_SIZE;
    factx.size_m_block = hex_round_up(FLASH_ATTN_BLOCK_SIZE * sizeof(__fp16), 128);
    factx.n_blocks = (n_kv + FLASH_ATTN_BLOCK_SIZE - 1) / FLASH_ATTN_BLOCK_SIZE;
    factx.scale = scale;
    factx.max_bias = 0.0f;
    factx.logit_softcap = (__fp16) 0.0f;
    factx.n_head_log2 = 1;
    factx.m0 = 1.0f;
    factx.m1 = 1.0f;
    for (uint32_t h = 0; h < n_heads && h < 512; h++) {
        factx.slopes[h] = (__fp16) 1.0f;
    }
    factx.qrows = (uint32_t) n_q;
    factx.qrow_start = 0;
    factx.qrows_per_thread = fastdiv((uint32_t) n_q + n_threads - 1, &octx.n_threads_div);
    factx.size_q_block = factx.size_q_row_padded;
    factx.size_vkq_acc = hex_round_up(DV * sizeof(float), 128);

    uint8_t * vtcm_cur = (uint8_t *) lab_vtcm_alloc(0, 128);
    (void) vtcm_cur;
    factx.spad_q = lab_vtcm_alloc(factx.size_q_block * n_threads, 128);
    factx.spad_k = lab_vtcm_alloc(factx.size_k_block * 2 * n_threads, 128);
    factx.spad_v = lab_vtcm_alloc(factx.size_v_block * 2 * n_threads, 128);
    factx.spad_m = m_data ? lab_vtcm_alloc(factx.size_m_block * HVX_FA_DMA_CACHE_SIZE * n_threads, 128) : NULL;
    factx.spad_a = lab_vtcm_alloc(factx.size_vkq_acc * n_threads, 128);

    printf("lab: %s DK %u DV %u n_kv %u heads %u kvheads %u tokens %u blocks %u threads %u mask %d\n",
           TARGET, DK, DV, n_kv, n_heads, n_kvheads, n_tokens, factx.n_blocks, n_threads, (int) use_mask);

    // The warm-up run fills the caches with the code and the data
    lab_run_threads(flash_attn_ext_f16_thread, &factx, n_threads);

    uint64_t best  = UINT64_MAX;
    uint64_t total = 0;
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        lab_run_threads(flash_attn_ext_f16_thread, &factx, n_threads);
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        const uint64_t d = t1 - t0;
        total += d;
        if (d < best) {
            best = d;
        }
    }

    // the reference of every q row
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t t = 0; t < n_tokens; t++) {
            const uint32_t kh = h / (n_heads / n_kvheads);
            ref_fa_row(q_data + ((size_t) h * n_tokens + t) * DK,
                       k_data + (size_t) kh * n_kv * DK,
                       v_data + (size_t) kh * n_kv * DV,
                       m_data ? m_data + (size_t) t * n_kv : NULL,
                       o_ref + ((size_t) t * n_heads + h) * DV,
                       DK, DV, n_kv, k_row, v_row, scale, scores);
        }
    }

    const size_t n_out = n_q * DV;
    double se = 0.0, sr = 0.0;
    for (size_t i = 0; i < n_out; i++) {
        const double d = (double) o_data[i] - (double) o_ref[i];
        se += d * d;
        sr += (double) o_ref[i] * (double) o_ref[i];
    }
    const size_t bad = lab_compare_f32("out", o_data, o_ref, n_out, 2e-3f, 2e-2f);

    // the work of one call: each q row reads every key and every value
    const double macs  = (double) n_q * n_kv * (DK + DV);
    const double bytes = (double) n_q * n_kv * (DK + DV) * sizeof(__fp16);
    lab_report(TARGET, "n_kv", n_kv, "");
    lab_report(TARGET, "q_rows", (double) n_q, "");
    lab_report(TARGET, "threads", n_threads, "");
    lab_report(TARGET, "cycles_per_call_min", (double) best, "cycles");
    lab_report(TARGET, "cycles_per_call_mean", (double) total / iters, "cycles");
    lab_report(TARGET, "cycles_per_qrow", (double) best / (double) n_q, "cycles");
    lab_report(TARGET, "cycles_per_kv_row", (double) best / ((double) n_q * n_kv), "cycles");
    lab_report(TARGET, "macs_per_cycle", macs / (double) best, "MAC/cycle");
    lab_report(TARGET, "bytes_per_cycle", bytes / (double) best, "B/cycle");
    lab_report(TARGET, "us_at_2112_mhz", (double) best / 2112.0, "us");
    lab_report(TARGET, "nmse_out", sr > 0.0 ? se / sr : 0.0, "");
    lab_report(TARGET, "mismatches", (double) bad, "");
    return 0;
}
