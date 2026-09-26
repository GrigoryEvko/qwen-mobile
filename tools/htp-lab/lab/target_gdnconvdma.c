// Target: the op GDN_CONV_CHUNK of gdn-conv-ops.c (op_gdn_conv_chunk), the path that reads x from DDR
// with vector loads (kernel_params[2] = 0) against the DMA path (kernel_params[2] = 1).
//
// The program includes the kernel file verbatim and calls the op through its context, thus the run
// covers the thread split, the prologue of the planes, the bands and the snapshot walk. The DMA queue
// is the shim of lab-dma.h. With --copy 1 the shim copies the rows at the push (the
// check needs that). With --copy 0 the push only records, thus a timing run measures the compute of the
// DMA path without the cost of a copy that the phone does on the DMA engine.
//
// The check: the scalar reference of target_gdn_chunk.c (the sum order of the CPU op, then SiLU) for
// every channel and token, and the exact new state of each snapshot slot. The tolerance is the one of
// the packed path in that target.
//
// Arguments: --n_ch 8192 --tokens 76 --slots 1 --threads 6 --dma 1 --copy 1 --iters 1 --range 4
// lab-run: mode=functional
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lab-dma.h"

#include "gdn-conv-ops.c"

#define TARGET "gdnconvdma"

static struct htp_context g_ctx;
static dma_queue          g_dma[HTP_MAX_NTHREADS];

// x * sigmoid(x) without an overflow of expf. O(1).
static float ref_silu(float a) {
    if (a >= 0.0f) {
        return a / (1.0f + expf(-a));
    }
    const float e = expf(a);
    return a * e / (1.0f + e);
}

static void set_tensor(struct htp_tensor * t, const void * data, uint32_t type, uint32_t ne0, uint32_t ne1,
                       uint32_t nb0, uint32_t nb1) {
    memset(t, 0, sizeof(*t));
    t->data  = (uint32_t) (uintptr_t) data;
    t->type  = type;
    t->ne[0] = ne0;
    t->ne[1] = ne1;
    t->ne[2] = 1;
    t->ne[3] = 1;
    t->nb[0] = nb0;
    t->nb[1] = nb1;
    t->nb[2] = nb1 * ne1;
    t->nb[3] = t->nb[2];
}

int main(int argc, char ** argv) {
    const uint32_t n_ch    = (uint32_t) lab_arg_long(argc, argv, "--n_ch", 8192);
    const uint32_t T       = (uint32_t) lab_arg_long(argc, argv, "--tokens", 76);
    const uint32_t n_slots = (uint32_t) lab_arg_long(argc, argv, "--slots", 1);
    const uint32_t nth     = (uint32_t) lab_arg_long(argc, argv, "--threads", 6);
    const uint32_t use_dma = (uint32_t) lab_arg_long(argc, argv, "--dma", 1);
    const uint32_t iters   = (uint32_t) lab_arg_long(argc, argv, "--iters", 1);
    const float    range   = (float) lab_arg_long(argc, argv, "--range", 4);
    lab_dma_copy           = (int) lab_arg_long(argc, argv, "--copy", 1);

    if (n_ch % 32 != 0 || T < 4 || nth == 0 || nth > HTP_MAX_NTHREADS || n_slots == 0) {
        printf("lab: %s the shape is not supported\n", TARGET);
        return 2;
    }

    lab_init();
    printf("lab: %s n_ch %u tokens %u slots %u threads %u dma %u copy %d\n", TARGET, n_ch, T, n_slots, nth, use_dma,
           lab_dma_copy);

    const uint32_t row = 3 * n_ch;
    // The projection has the layout of the graph: the token rows of the qkv matmul output, one after
    // the other, and the op reads its transpose (element (t, c) at t * nb0 + c * 4).
    float * states = lab_ddr_alloc((size_t) 2 * row * sizeof(float) + 256, 128);
    float * x      = lab_ddr_alloc((size_t) T * n_ch * sizeof(float) + 256, 128);
    float * w      = lab_ddr_alloc((size_t) 4 * n_ch * sizeof(float) + 256, 128);
    const size_t slot_stride = (size_t) row * sizeof(float) + 512;
    uint8_t * slots = lab_ddr_alloc(slot_stride * n_slots + 256, 128);
    float * y      = lab_ddr_alloc((size_t) T * n_ch * sizeof(float) + 256, 128);
    float * y_ref  = lab_ddr_alloc((size_t) T * n_ch * sizeof(float), 128);
    float * n_ref  = lab_ddr_alloc((size_t) n_slots * row * sizeof(float), 128);
    int32_t * idx  = lab_ddr_alloc(128, 128);

    lab_fill_f32(states, 2 * row, -range, range);
    lab_fill_f32(x, (size_t) T * n_ch, -range, range);
    lab_fill_f32(w, 4 * n_ch, -1.0f, 1.0f);
    idx[0] = 1;

    struct htp_tensor t_states, t_idx, t_x, t_w, t_slot, t_y;
    set_tensor(&t_states, states, HTP_TYPE_F32, row, 2, sizeof(float), row * sizeof(float));
    set_tensor(&t_idx, idx, HTP_TYPE_I32, 1, 1, sizeof(int32_t), sizeof(int32_t));
    set_tensor(&t_x, x, HTP_TYPE_F32, T, n_ch, n_ch * sizeof(float), sizeof(float));
    set_tensor(&t_w, w, HTP_TYPE_F32, 4, n_ch, sizeof(float), 4 * sizeof(float));
    set_tensor(&t_slot, slots, HTP_TYPE_F32, row, n_slots, sizeof(float), (uint32_t) slot_stride);
    set_tensor(&t_y, y, HTP_TYPE_F32, n_ch, T, sizeof(float), n_ch * sizeof(float));

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

    struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx           = &g_ctx;
    octx.op            = HTP_OP_GDN_CONV_CHUNK;
    octx.n_threads     = nth;
    octx.n_threads_div = g_ctx.n_threads_div;
    octx.src[0] = &t_states;
    octx.src[1] = &t_idx;
    octx.src[2] = &t_x;
    octx.src[3] = &t_w;
    octx.src[4] = &t_slot;
    octx.dst    = &t_y;
    octx.kernel_params[0] = 1;
    octx.kernel_params[1] = 1;
    octx.kernel_params[2] = (int32_t) use_dma;

    uint64_t best = UINT64_MAX;
    for (uint32_t it = 0; it < iters + 1; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        const int status = op_gdn_conv_chunk(&octx);
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (status != HTP_STATUS_OK) {
            printf("lab: %s the op returned %d (detail %u)\n", TARGET, status, octx.err_detail);
            return 1;
        }
        if (it > 0 || iters == 0) {
            best = (t1 - t0) < best ? (t1 - t0) : best;
        }
    }
    lab_report(TARGET, "cycles_per_op", (double) best, "cycles");
    lab_report(TARGET, "cycles_per_token", (double) best / T, "cycles");
    lab_report(TARGET, "ms_per_1024_tokens_24_layers", (double) best / 2112.0 / 1000.0 * 24.0 * 1024.0 / T, "ms");
    const uint32_t dma_faults = lab_dma_report(TARGET);

    if (!lab_dma_copy) {
        lab_report(TARGET, "checked", 0, "");
        return dma_faults == 0 ? 0 : 1;
    }

    // ---- the reference
    const float * src = states + row;
    for (uint32_t c = 0; c < n_ch; c++) {
        float h[3] = { src[c * 3 + 0], src[c * 3 + 1], src[c * 3 + 2] };
        for (uint32_t t = 0; t < T; t++) {
            const float xc = x[(size_t) t * n_ch + c];
            float acc = h[0] * w[4 * c];
            acc += h[1] * w[4 * c + 1];
            acc += h[2] * w[4 * c + 2];
            acc += xc * w[4 * c + 3];
            y_ref[(size_t) t * n_ch + c] = ref_silu(acc);
            h[0] = h[1];
            h[1] = h[2];
            h[2] = xc;
            const int64_t g = (int64_t) T - 1 - (int64_t) t;
            if (g >= 0 && g < (int64_t) n_slots) {
                n_ref[(size_t) g * row + c * 3 + 0] = h[0];
                n_ref[(size_t) g * row + c * 3 + 1] = h[1];
                n_ref[(size_t) g * row + c * 3 + 2] = h[2];
            }
        }
    }

    size_t fail = 0, n_nonfinite = 0;
    double se = 0.0, sr = 0.0, worst = 0.0;
    const double abs_tol = 4.0 * (double) range * 2.0 / 2048.0;
    for (size_t i = 0; i < (size_t) T * n_ch; i++) {
        const double g = (double) y[i];
        const double r = (double) y_ref[i];
        if (!isfinite(g)) {
            n_nonfinite++;
            continue;
        }
        const double d = fabs(g - r);
        se += d * d;
        sr += r * r;
        worst = d > worst ? d : worst;
        if (d > abs_tol + 2e-3 * fabs(r)) {
            if (fail < 4) {
                printf("lab: %s y mismatch at token %u channel %u: got %g want %g\n", TARGET,
                       (uint32_t) (i / n_ch), (uint32_t) (i % n_ch), g, r);
            }
            fail++;
        }
    }
    const double nmse = sr > 0.0 ? se / sr : 0.0;
    size_t slot_bad = 0;
    for (uint32_t g = 0; g < n_slots; g++) {
        const float * got = (const float *) (slots + g * slot_stride);
        for (uint32_t i = 0; i < row; i++) {
            if (got[i] != n_ref[(size_t) g * row + i]) {
                if (slot_bad < 4) {
                    printf("lab: %s slot %u mismatch at %u: got %g want %g\n", TARGET, g, i, (double) got[i],
                           (double) n_ref[(size_t) g * row + i]);
                }
                slot_bad++;
            }
        }
    }
    lab_report(TARGET, "nmse", nmse, "");
    lab_report(TARGET, "worst_abs", worst, "");
    lab_report(TARGET, "y_failures", (double) fail, "");
    lab_report(TARGET, "not_finite", (double) n_nonfinite, "");
    lab_report(TARGET, "slot_mismatches", (double) slot_bad, "");
    const bool pass = fail == 0 && n_nonfinite == 0 && slot_bad == 0 && nmse < 1e-5 && dma_faults == 0;
    printf("lab: %s check %s\n", TARGET, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
