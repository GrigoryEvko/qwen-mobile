// Target 5: the fused GDN conv of gdn-conv-ops.c over a batch of tokens (d_conv 4), the prefill path.
//
// The program includes the kernel file verbatim and calls its thread function, thus the
// measurement covers the kernel and its per-thread chunk loop, and not the FastRPC path, the op
// batch, or the work queue wakeup of the phone. The inputs and the outputs are in DDR (the
// simulator memory) and the plane buffers are in the VTCM, as on the phone.
//
// On the phone this op is the largest single op of a 512-token prefill of the 4B (27 % of the DSP
// time at 1.27 instructions per packet, measured 2026-09-20), thus it has its own target.
//
// Arguments: --n_ch 8192 --tokens 16 --threads 1 --iters 3 --range 4 --f16 0
// --f16 selects the precision mode of the proposal (0 f32, 1 f16 SiLU, 2 f16 taps with a 32-bit
// accumulator, 3 f16 taps with an f16 accumulator, 4 the packed path: f16 taps, 32-bit accumulator,
// int16 SiLU, 4 token rows in each iteration). The kernel of the checkout has mode 0 only.
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "gdn-conv-ops.c"

#define TARGET "gdn_chunk"
#define D_CONV 4

// The scalar reference: the sum order of ggml_compute_forward_ssm_conv_f32, then x * sigmoid(x).
// slot[c * 3 + j] is tap j of channel c before the batch, oldest first. O(tokens * n_ch).
static void ref_conv_chunk(const float * slot, const float * x, const float * w, float * y, float * n_slot,
                           uint32_t n_ch, uint32_t n_tokens) {
    for (uint32_t c = 0; c < n_ch; c++) {
        float h0 = slot[c * 3 + 0];
        float h1 = slot[c * 3 + 1];
        float h2 = slot[c * 3 + 2];
        for (uint32_t t = 0; t < n_tokens; t++) {
            const float xc = x[(size_t) t * n_ch + c];
            float acc = h0 * w[4 * c];
            acc += h1 * w[4 * c + 1];
            acc += h2 * w[4 * c + 2];
            acc += xc * w[4 * c + 3];
            y[(size_t) t * n_ch + c] = acc / (1.0f + expf(-acc));
            h0 = h1;
            h1 = h2;
            h2 = xc;
        }
        n_slot[c * 3 + 0] = h0;
        n_slot[c * 3 + 1] = h1;
        n_slot[c * 3 + 2] = h2;
    }
}

int main(int argc, char ** argv) {
    const uint32_t n_ch      = (uint32_t) lab_arg_long(argc, argv, "--n_ch", 8192);
    const uint32_t n_tokens  = (uint32_t) lab_arg_long(argc, argv, "--tokens", 16);
    const uint32_t n_threads = (uint32_t) lab_arg_long(argc, argv, "--threads", 1);
    const uint32_t iters     = (uint32_t) lab_arg_long(argc, argv, "--iters", 3);
    // the range of the conv input. A wide range reaches the saturated ends of the SiLU.
    const float    x_range   = (float) lab_arg_long(argc, argv, "--range", 4);

    if (n_tokens < D_CONV - 1) {
        printf("lab: %s needs at least %d tokens\n", TARGET, D_CONV - 1);
        return 2;
    }

    lab_init();

#ifdef LAB_PROPOSED
    gdn_conv_f16_mode = (int) lab_arg_long(argc, argv, "--f16", 0);
    printf("lab: %s f16_mode = %d\n", TARGET, gdn_conv_f16_mode);
#endif

    const size_t n_xy = (size_t) n_tokens * n_ch;

    // The kernel loads full vectors at unaligned addresses, thus each buffer has a margin at the end.
    float * src_slot = lab_ddr_alloc(3 * n_ch * sizeof(float) + 256, 128);
    float * dst_slot = lab_ddr_alloc(3 * n_ch * sizeof(float) + 256, 128);
    float * x        = lab_ddr_alloc(n_xy * sizeof(float) + 256, 128);
    float * w        = lab_ddr_alloc(4 * n_ch * sizeof(float) + 256, 128);
    float * y        = lab_ddr_alloc(n_xy * sizeof(float) + 256, 128);
    float * y_ref    = lab_ddr_alloc(n_xy * sizeof(float), 128);
    float * n_ref    = lab_ddr_alloc(3 * n_ch * sizeof(float), 128);

    lab_fill_f32(src_slot, 3 * n_ch, -x_range, x_range);
    lab_fill_f32(x, n_xy, -x_range, x_range);
    lab_fill_f32(w, 4 * n_ch, -1.0f, 1.0f);

    static struct htp_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx           = &ctx;
    octx.n_threads     = n_threads;
    octx.n_threads_div = init_fastdiv_values(n_threads);

    struct htp_gdn_conv_chunk_context cctx;
    memset(&cctx, 0, sizeof(cctx));
    cctx.octx     = &octx;
    cctx.src_slot = src_slot;
    cctx.dst_slot = dst_slot;
    cctx.x        = x;
    cctx.y        = y;
    cctx.w        = w;
    cctx.x_stride = n_ch;
    cctx.y_stride = n_ch;
    cctx.n_ch     = n_ch;
    cctx.n_tokens = n_tokens;
    cctx.n_slots  = 1;   // one snapshot slot, as a batch without speculative decoding has
    cctx.chunk    = hex_round_up(fastdiv(n_ch + n_threads - 1, &octx.n_threads_div), VLEN_FP32);
    cctx.vtcm_per_thread = hex_round_up((size_t) 11 * cctx.chunk * sizeof(float) + 512, 128);
    cctx.vtcm     = lab_vtcm_alloc(cctx.vtcm_per_thread * n_threads, 128);

    // The warm-up run fills the caches with the code and the data
    lab_run_threads(gdn_conv_chunk_thread, &cctx, n_threads);

    uint64_t best  = UINT64_MAX;
    uint64_t total = 0;
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        lab_run_threads(gdn_conv_chunk_thread, &cctx, n_threads);
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        const uint64_t d = t1 - t0;
        total += d;
        if (d < best) {
            best = d;
        }
    }

    ref_conv_chunk(src_slot, x, w, y_ref, n_ref, n_ch, n_tokens);
    const size_t bad_y = lab_compare_f32("y", y, y_ref, n_xy, 1e-4f, 2e-3f);
    const size_t bad_n = lab_compare_f32("slot", dst_slot, n_ref, 3 * n_ch, 0.0f, 0.0f);

    // the normalized mean squared error of y, the measure that test-backend-ops uses
    double se = 0.0;
    double sr = 0.0;
    for (size_t i = 0; i < n_xy; i++) {
        const double d = (double) y[i] - (double) y_ref[i];
        se += d * d;
        sr += (double) y_ref[i] * (double) y_ref[i];
    }

    // Where the error sits decides its cause. A lane or a chunk mix-up follows the block parity or the
    // lane index and is of the size of the values. A numeric error follows the value of the sum.
    {
        double se_par[2] = { 0, 0 }, sr_par[2] = { 0, 0 };   // by the parity of the 32-lane block
        double se_sgn[2] = { 0, 0 }, sr_sgn[2] = { 0, 0 };   // by the sign of the reference
        double se_mag[4] = { 0 },    sr_mag[4] = { 0 };      // by |y_ref|: <0.1, <1, <4, the rest
        for (size_t i = 0; i < n_xy; i++) {
            const uint32_t c   = (uint32_t) (i % n_ch);
            const double   d   = (double) y[i] - (double) y_ref[i];
            const double   r   = (double) y_ref[i];
            const int      par = (int) ((c / VLEN_FP32) & 1);
            const int      sgn = r < 0.0;
            const double   a   = fabs(r);
            const int      mag = a < 0.1 ? 0 : a < 1.0 ? 1 : a < 4.0 ? 2 : 3;
            se_par[par] += d * d; sr_par[par] += r * r;
            se_sgn[sgn] += d * d; sr_sgn[sgn] += r * r;
            se_mag[mag] += d * d; sr_mag[mag] += r * r;
        }
        printf("lab: %s nmse by block parity: even %.3g odd %.3g\n", TARGET, se_par[0] / sr_par[0], se_par[1] / sr_par[1]);
        printf("lab: %s nmse by sign of y:    pos %.3g neg %.3g\n", TARGET, se_sgn[0] / sr_sgn[0], se_sgn[1] / sr_sgn[1]);
        printf("lab: %s nmse by |y|: <0.1 %.3g  <1 %.3g  <4 %.3g  rest %.3g\n", TARGET,
               se_mag[0] / sr_mag[0], se_mag[1] / sr_mag[1], se_mag[2] / sr_mag[2], se_mag[3] / sr_mag[3]);
        int shown = 0;
        for (size_t i = 0; i < n_xy && shown < 10; i++) {
            const double d = fabs((double) y[i] - (double) y_ref[i]);
            if (d > 1e-3) {
                const uint32_t c = (uint32_t) (i % n_ch);
                printf("lab: %s bad  ch %5u lane %2u blk %3u  ref % .6f got % .6f\n", TARGET,
                       c, c % VLEN_FP32, c / VLEN_FP32, (double) y_ref[i], (double) y[i]);
                shown++;
            }
        }
    }

    const double blocks = (double) n_xy / VLEN_FP32;
    const double bytes  = (double) n_xy * 2 * sizeof(float);
    lab_report(TARGET, "n_ch", n_ch, "");
    lab_report(TARGET, "tokens", n_tokens, "");
    lab_report(TARGET, "threads", n_threads, "");
    lab_report(TARGET, "cycles_per_call_min", (double) best, "cycles");
    lab_report(TARGET, "cycles_per_call_mean", (double) total / iters, "cycles");
    lab_report(TARGET, "cycles_per_block", (double) best / blocks, "cycles");
    lab_report(TARGET, "bytes_per_cycle", bytes / (double) best, "B/cycle");
    lab_report(TARGET, "us_per_token_at_2112_mhz", (double) best / 2112.0 / n_tokens, "us");
    lab_report(TARGET, "nmse_y", sr > 0.0 ? se / sr : 0.0, "");
    lab_report(TARGET, "mismatches", (double) (bad_y + bad_n), "");
    return (bad_y + bad_n) ? 1 : 0;
}
