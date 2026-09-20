// Target 1: the fused GDN conv step of gdn-conv-ops.c at one token (d_conv 4).
//
// The program includes the kernel file verbatim and calls its thread function, thus the
// measurement covers the kernel and its per-thread chunk loop, and not the FastRPC path, the op
// batch, or the work queue wakeup of the phone. The inputs and the outputs are in DDR (the
// simulator memory), as on the phone.
//
// The program builds the taps of every channel in one canonical array, and then writes them into
// the slot in the layout that the kernel expects: three floats per channel for the kernel of the
// checkout, three planes of n_ch floats for the kernel of the proposal (LAB_PROPOSED). The output
// y does not depend on the layout, thus the base run and the proposal run must give the same y.
//
// Arguments: --n_ch 6144 --threads 1 --iters 20
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "gdn-conv-ops.c"

#define TARGET "gdn_conv"
#define D_CONV 4

// The slot holds the three taps of a channel one after the other, in both programs.
//
// A plane layout (tap t of every channel in one row) would remove almost every permute of this
// kernel, but the slot is the recurrent state cache of llama.cpp: the graph of
// src/models/delta-net-base.cpp builds it with a CONCAT of [d_conv - 1, n_ch], every chain that
// the Hexagon matcher refuses reads it with the generic ops, and
// llama_memory_recurrent::state_write_data writes those bytes into the state file that the app
// stores. Thus a plane layout needs the host, the CPU reference and the state file to change
// together, and at one token the conversion at the two ends costs the permutes again. The
// proposal keeps the layout and changes the schedule instead.
#ifdef LAB_PROPOSED
#define LAYOUT "interleaved (proposal)"
#else
#define LAYOUT "interleaved"
#endif

static inline size_t slot_off(uint32_t c, uint32_t t, uint32_t n_ch) {
    (void) n_ch;
    return (size_t) c * 3 + t;
}

// The scalar reference: the sum order of ggml_compute_forward_ssm_conv_f32, then x * sigmoid(x).
// taps[t * n_ch + c] is tap t of channel c, in time order.
static void ref_conv_step(const float * taps, const float * x, const float * w, float * y, float * n_taps, uint32_t n_ch) {
    for (uint32_t c = 0; c < n_ch; c++) {
        const float s0 = taps[0 * n_ch + c];
        const float s1 = taps[1 * n_ch + c];
        const float s2 = taps[2 * n_ch + c];
        const float xc = x[c];
        float acc = s0 * w[4 * c];
        acc += s1 * w[4 * c + 1];
        acc += s2 * w[4 * c + 2];
        acc += xc * w[4 * c + 3];
        y[c] = acc / (1.0f + expf(-acc));
        n_taps[0 * n_ch + c] = s1;
        n_taps[1 * n_ch + c] = s2;
        n_taps[2 * n_ch + c] = xc;
    }
}

int main(int argc, char ** argv) {
    const uint32_t n_ch      = (uint32_t) lab_arg_long(argc, argv, "--n_ch", 6144);
    const uint32_t n_threads = (uint32_t) lab_arg_long(argc, argv, "--threads", 1);
    const uint32_t iters     = (uint32_t) lab_arg_long(argc, argv, "--iters", 20);

    lab_init();
    printf("lab: %s layout = %s\n", TARGET, LAYOUT);

    // The kernel loads full vectors at unaligned addresses, thus each buffer has a margin at the end.
    float * src_slot = lab_ddr_alloc(3 * n_ch * sizeof(float) + 256, 128);
    float * dst_slot = lab_ddr_alloc(3 * n_ch * sizeof(float) + 256, 128);
    float * x        = lab_ddr_alloc(n_ch * sizeof(float) + 256, 128);
    float * w        = lab_ddr_alloc(4 * n_ch * sizeof(float) + 256, 128);
    float * y        = lab_ddr_alloc(n_ch * sizeof(float) + 256, 128);
    float * taps     = lab_ddr_alloc(3 * n_ch * sizeof(float), 128);
    float * y_ref    = lab_ddr_alloc(n_ch * sizeof(float), 128);
    float * n_ref    = lab_ddr_alloc(3 * n_ch * sizeof(float), 128);
    float * n_got    = lab_ddr_alloc(3 * n_ch * sizeof(float), 128);

    lab_fill_f32(taps, 3 * n_ch, -2.0f, 2.0f);
    lab_fill_f32(x, n_ch, -2.0f, 2.0f);
    lab_fill_f32(w, 4 * n_ch, -1.0f, 1.0f);
    for (uint32_t c = 0; c < n_ch; c++) {
        for (uint32_t t = 0; t < 3; t++) {
            src_slot[slot_off(c, t, n_ch)] = taps[t * n_ch + c];
        }
    }

    static struct htp_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx           = &ctx;
    octx.n_threads     = n_threads;
    octx.n_threads_div = init_fastdiv_values(n_threads);

    struct htp_gdn_conv_context cctx;
    memset(&cctx, 0, sizeof(cctx));
    cctx.octx     = &octx;
    cctx.src_slot = src_slot;
    cctx.dst_slot = dst_slot;
    cctx.x        = x;
    cctx.w        = w;
    cctx.y        = y;
    cctx.n_ch     = n_ch;
    cctx.chunk    = hex_round_up(fastdiv(n_ch + n_threads - 1, &octx.n_threads_div), VLEN_FP32);

    // The warm-up run fills the caches with the code and the data
    lab_run_threads(gdn_conv_step_thread, &cctx, n_threads);

    uint64_t best  = UINT64_MAX;
    uint64_t total = 0;
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        lab_run_threads(gdn_conv_step_thread, &cctx, n_threads);
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        const uint64_t d = t1 - t0;
        total += d;
        if (d < best) {
            best = d;
        }
    }

    ref_conv_step(taps, x, w, y_ref, n_ref, n_ch);
    for (uint32_t c = 0; c < n_ch; c++) {
        for (uint32_t t = 0; t < 3; t++) {
            n_got[t * n_ch + c] = dst_slot[slot_off(c, t, n_ch)];
        }
    }
    const size_t bad_y = lab_compare_f32("y", y, y_ref, n_ch, 1e-4f, 2e-3f);
    const size_t bad_n = lab_compare_f32("slot", n_got, n_ref, 3 * n_ch, 0.0f, 0.0f);

    const double blocks = (double) n_ch / VLEN_FP32;
    const double bytes  = (double) n_ch * (3 + 1 + 4 + 1 + 3) * sizeof(float);
    lab_report(TARGET, "n_ch", n_ch, "");
    lab_report(TARGET, "threads", n_threads, "");
    lab_report(TARGET, "cycles_per_call_min", (double) best, "cycles");
    lab_report(TARGET, "cycles_per_call_mean", (double) total / iters, "cycles");
    lab_report(TARGET, "cycles_per_block", (double) best / blocks, "cycles");
    lab_report(TARGET, "bytes_per_cycle", bytes / (double) best, "B/cycle");
    lab_report(TARGET, "us_at_2112_mhz", (double) best / 2112.0, "us");
    lab_report(TARGET, "us_at_2112_mhz_6_threads", (double) best / 2112.0 / 6.0, "us");
    lab_report(TARGET, "mismatches", (double) (bad_y + bad_n), "");
    return (bad_y + bad_n) ? 1 : 0;
}
