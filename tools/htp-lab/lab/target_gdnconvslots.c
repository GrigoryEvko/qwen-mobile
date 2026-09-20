// Target 13: the rollback snapshot slots of the fused GDN conv over a batch, gdn-conv-ops.c.
//
// Without speculative decoding the conv chain writes one slot of the conv cache, the state after
// the last token. Speculative decoding asks for K slots, slot g the state after token T - 1 - g.
// This target runs gdn_conv_chunk_thread for K of 1 to 5 and checks every slot against a scalar
// reference, and it times the kernel at each K so the cost of the extra slots is visible.
//
// The kernel is HVX only, thus the cycles here are real.
//
// Arguments: --n_ch 8192 --tokens 16 --threads 1 --iters 3 --range 4 --slots 5
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "gdn-conv-ops.c"

#define TARGET "gdnconvslots"
#define D_CONV 4

// Column c of the CONCAT of one channel: the three columns of the cache, then the batch.
static inline float ci_col(const float * slot, const float * x, uint32_t n_ch, uint32_t c, uint32_t ch) {
    return (c < D_CONV - 1) ? slot[(size_t) ch * (D_CONV - 1) + c] : x[(size_t) (c - (D_CONV - 1)) * n_ch + ch];
}

// The reference of slot g: the d_conv - 1 columns of the CONCAT that start at column max(0, T - g).
// That window ends at token T - 1 - g, thus it is the conv state after that token. O(n_ch).
static void ref_conv_slot(const float * slot, const float * x, float * out, uint32_t n_ch, uint32_t T,
                          uint32_t g) {
    const uint32_t c0 = (T > g) ? (T - g) : 0;
    for (uint32_t ch = 0; ch < n_ch; ch++) {
        for (uint32_t j = 0; j < D_CONV - 1; j++) {
            out[(size_t) ch * (D_CONV - 1) + j] = ci_col(slot, x, n_ch, c0 + j, ch);
        }
    }
}

int main(int argc, char ** argv) {
    const uint32_t n_ch      = (uint32_t) lab_arg_long(argc, argv, "--n_ch", 8192);
    const uint32_t n_tokens  = (uint32_t) lab_arg_long(argc, argv, "--tokens", 16);
    const uint32_t n_threads = (uint32_t) lab_arg_long(argc, argv, "--threads", 1);
    const uint32_t iters     = (uint32_t) lab_arg_long(argc, argv, "--iters", 3);
    const float    x_range   = (float) lab_arg_long(argc, argv, "--range", 4);
    const uint32_t max_slots = (uint32_t) lab_arg_long(argc, argv, "--slots", 5);

    if (n_tokens < D_CONV) {
        printf("lab: %s needs at least %d tokens\n", TARGET, D_CONV);
        return 2;
    }

    lab_init();
    printf("lab: %s n_ch %u tokens %u threads %u slots 1..%u\n", TARGET, n_ch, n_tokens, n_threads, max_slots);

    const size_t n_xy  = (size_t) n_tokens * n_ch;
    const size_t row   = (size_t) (D_CONV - 1) * n_ch;

    // The kernel loads and stores full vectors at unaligned addresses, thus each buffer has a margin.
    float * src_slot = lab_ddr_alloc(row * sizeof(float) + 256, 128);
    float * dst      = lab_ddr_alloc(row * max_slots * sizeof(float) + 256, 128);
    float * x        = lab_ddr_alloc(n_xy * sizeof(float) + 256, 128);
    float * w        = lab_ddr_alloc(4 * n_ch * sizeof(float) + 256, 128);
    float * y        = lab_ddr_alloc(n_xy * sizeof(float) + 256, 128);
    float * ref      = lab_ddr_alloc(row * sizeof(float), 128);

    lab_fill_f32(src_slot, row, -x_range, x_range);
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
    cctx.dst_slot = dst;
    cctx.x        = x;
    cctx.y        = y;
    cctx.w        = w;
    cctx.x_stride = n_ch;
    cctx.y_stride = n_ch;
    cctx.n_ch     = n_ch;
    cctx.n_tokens = n_tokens;
    cctx.chunk    = hex_round_up(fastdiv(n_ch + n_threads - 1, &octx.n_threads_div), VLEN_FP32);
    cctx.vtcm_per_thread = hex_round_up((size_t) 11 * cctx.chunk * sizeof(float) + 512, 128);
    cctx.vtcm     = lab_vtcm_alloc(cctx.vtcm_per_thread * n_threads, 128);

    size_t   bad_total = 0;
    uint64_t base_cyc  = 0;

    for (uint32_t slots = 1; slots <= max_slots; slots++) {
        cctx.n_slots     = slots;
        cctx.slot_stride = (uint32_t) (row * sizeof(float));
        memset(dst, 0, row * max_slots * sizeof(float));

        lab_run_threads(gdn_conv_chunk_thread, &cctx, n_threads);

        uint64_t best = UINT64_MAX;
        for (uint32_t it = 0; it < iters; it++) {
            LAB_BARRIER();
            const uint64_t t0 = lab_cycles();
            lab_run_threads(gdn_conv_chunk_thread, &cctx, n_threads);
            const uint64_t t1 = lab_cycles();
            LAB_BARRIER();
            if (t1 - t0 < best) {
                best = t1 - t0;
            }
        }
        if (slots == 1) {
            base_cyc = best;
        }

        size_t bad = 0;
        for (uint32_t g = 0; g < slots; g++) {
            ref_conv_slot(src_slot, x, ref, n_ch, n_tokens, g);
            char what[32];
            snprintf(what, sizeof(what), "slot%u", g);
            bad += lab_compare_f32(what, dst + (size_t) g * row, ref, row, 0.0f, 0.0f);
        }
        bad_total += bad;

        printf("lab: %s slots %u %llu cycles (%+.2f %% of one slot) mismatches %zu\n", TARGET, slots,
               (unsigned long long) best,
               base_cyc ? 100.0 * ((double) best - (double) base_cyc) / (double) base_cyc : 0.0, bad);
        char key[32];
        snprintf(key, sizeof(key), "cycles_slots%u", slots);
        lab_report(TARGET, key, (double) best, "cycles");
        snprintf(key, sizeof(key), "mismatches_slots%u", slots);
        lab_report(TARGET, key, (double) bad, "");
    }

    lab_report(TARGET, "mismatches_total", (double) bad_total, "");
    return bad_total ? 1 : 0;
}
