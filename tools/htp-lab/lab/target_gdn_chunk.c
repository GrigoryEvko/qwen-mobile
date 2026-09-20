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
// The program runs three phases. Each phase reports its own errors, and each one can fail the run.
//   random   Uniform inputs in [-range, range]. This phase gives the cycle numbers.
//   octaves  The weights are (0, 0, 0, 1), thus the conv accumulator is the input. The input walks
//            the f16 octaves, thus the SiLU of the kernel meets every magnitude that an f16 holds
//            and not only the magnitudes that a sum of four uniform products reaches.
//   silu     Every one of the 65536 f16 values through hvx_silu_h_i16, with a check of the range
//            of h. This phase holds the routine to its own contract, 0 <= h <= 0.2785.
//
// Why the two phases after the random one: the packed path of 2026-09-20 gave -Inf for a conv
// accumulator in [14.4922, 15.7266], and a uniform input of [-4, 4] reaches that band with a
// probability near zero. The random phase measured NMSE 2.9e-07 and reported no failure, thus the
// build shipped and the phone answered with one token again and again. A mean square error hides
// a rare value that is not finite when the rare value does not occur.
//
// Arguments: --n_ch 8192 --tokens 16 --threads 1 --iters 3 --range 4 --f16 0
//            --phase all|random|octaves|silu
// --f16 selects the precision mode of the proposal (0 f32, 1 f16 SiLU, 2 f16 taps with a 32-bit
// accumulator, 3 f16 taps with an f16 accumulator, 4 the packed path: f16 taps, a 16-bit qfloat
// accumulator, the int16 SiLU, 3 token rows in each iteration). The kernel of the checkout has
// mode 0 only.
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "gdn-conv-ops.c"

#define TARGET "gdn_chunk"
#define D_CONV 4

// The largest value of h(u) = u / (1 + e^u), at the scale 2^16 of hvx_silu_h_i16, plus a margin
#define SILU_H_MAX 18300

// x * sigmoid(x) without an overflow of expf. The form a / (1 + expf(-a)) overflows for a large
// negative a, and the expf of the standalone runtime then gives a wrong finite number instead of
// an infinity: at a = -65248 it gave -24153 where the value is -0. The octaves phase reaches such
// values, thus the reference needs the two branches. O(1).
static float ref_silu(float a) {
    if (a >= 0.0f) {
        return a / (1.0f + expf(-a));          // expf(-a) is in (0, 1]
    }
    const float e = expf(a);                   // e is in (0, 1), and 0 when it underflows
    return a * e / (1.0f + e);
}

// The scalar reference: the sum order of ggml_compute_forward_ssm_conv_f32, then x * sigmoid(x).
// slot[c * 3 + j] is tap j of channel c before the batch, oldest first. The accumulator goes to
// acc_out, thus the report can name the input of an error and bin the errors by its octave.
// O(tokens * n_ch).
static void ref_conv_chunk(const float * slot, const float * x, const float * w, float * y, float * n_slot,
                           float * acc_out, uint32_t n_ch, uint32_t n_tokens) {
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
            acc_out[(size_t) t * n_ch + c] = acc;
            y[(size_t) t * n_ch + c] = ref_silu(acc);
            h0 = h1;
            h1 = h2;
            h2 = xc;
        }
        n_slot[c * 3 + 0] = h0;
        n_slot[c * 3 + 1] = h1;
        n_slot[c * 3 + 2] = h2;
    }
}

// The input of the octaves phase, as an f16 bit pattern. The even samples walk the octave
// [8, 16) of 1024 values, which is the octave that the int16 SiLU covers last and the octave
// where its polynomial is weakest. The odd samples walk every finite positive f16 in the order
// of a stride that is coprime with 31744, thus a long run covers the full range. One quarter of
// the samples is negative. O(1).
static uint16_t sweep_bits(uint32_t k) {
    uint16_t bits;
    if ((k & 1) == 0) {
        bits = (uint16_t) (0x4800 + ((k / 2) % 0x0400));
    } else {
        bits = (uint16_t) (((k / 2) * 9973u) % 0x7C00u);
    }
    if ((k & 3) >= 2) {
        bits |= 0x8000;
    }
    return bits;
}

// Reports the errors of one phase and returns the number of failures. A value fails when it is
// not finite, or when its error is more than abs_bound + rel_bound * |ref|. The phase also fails
// when the NMSE is more than nmse_bound.
//
// The tolerance holds |ref| and not |y| in its relative part, because the SiLU output of a large
// negative input is exponentially small while its error stays at the size of the SiLU table.
// A bound on the relative error alone rejects that tail, and a bound on the absolute error alone
// accepts a wrong sign of a large value.
//
// The report names the conv accumulator of the worst case of each kind, because the accumulator,
// and not the output, tells which part of the SiLU is weak. The octave table of |acc| shows one
// weak octave that a mean square error over all octaves hides. O(tokens * n_ch).
static size_t report_phase(const char * phase, const float * y, const float * y_ref, const float * acc,
                           size_t n, uint32_t n_ch, double abs_bound, double rel_bound,
                           double nmse_bound) {
    size_t n_bad = 0, n_nonfinite = 0, shown = 0;
    double max_abs = 0.0, max_rel = 0.0, max_over = 0.0;
    size_t arg_abs = 0, arg_rel = 0, arg_over = 0;
    double se = 0.0, sr = 0.0;

    // the worst error and the sample count of each octave of |acc|, plus one bin for zero
    double oct_max[18] = { 0 };
    size_t oct_n[18] = { 0 }, oct_bad[18] = { 0 };

    for (size_t i = 0; i < n; i++) {
        const double g = (double) y[i];
        const double r = (double) y_ref[i];
        const double a = fabs((double) acc[i]);
        const int    o = (a <= 0.0) ? 0 : (int) fmin(17.0, fmax(0.0, floor(log2(a)) + 13.0));

        oct_n[o]++;
        if (!isfinite(g)) {
            n_nonfinite++;
            n_bad++;
            oct_bad[o]++;
            if (shown < 8) {
                printf("lab: %s %s NOT FINITE at ch %5u token %5u: acc % .6f, y = %g, ref = % .6f\n",
                       TARGET, phase, (uint32_t) (i % n_ch), (uint32_t) (i / n_ch),
                       (double) acc[i], g, r);
                shown++;
            }
            continue;
        }

        const double d   = fabs(g - r);
        const double tol = abs_bound + rel_bound * fabs(r);
        se += d * d;
        sr += r * r;
        if (d > oct_max[o]) {
            oct_max[o] = d;
        }
        if (d > max_abs) {
            max_abs = d;
            arg_abs = i;
        }
        if (fabs(r) > 0.0 && d / fabs(r) > max_rel) {
            max_rel = d / fabs(r);
            arg_rel = i;
        }
        if (d / tol > max_over) {
            max_over = d / tol;
            arg_over = i;
        }
        if (d > tol) {
            n_bad++;
            oct_bad[o]++;
        }
    }

    const double nmse = sr > 0.0 ? se / sr : 0.0;
    if (nmse > nmse_bound) {
        printf("lab: %s %s NMSE %.4e is more than the bound %.4e\n", TARGET, phase, nmse, nmse_bound);
        n_bad++;
    }
    printf("lab: %s %s failures %lu (not finite %lu of %lu), tolerance %.1e + %.1e * |ref|\n",
           TARGET, phase, (unsigned long) n_bad, (unsigned long) n_nonfinite, (unsigned long) n,
           abs_bound, rel_bound);
    printf("lab: %s %s worst abs %.4e at acc % .6f (ref % .6f got % .6f)\n", TARGET, phase,
           max_abs, (double) acc[arg_abs], (double) y_ref[arg_abs], (double) y[arg_abs]);
    printf("lab: %s %s worst rel %.4e at acc % .6f (ref % .6e got % .6e)\n", TARGET, phase,
           max_rel, (double) acc[arg_rel], (double) y_ref[arg_rel], (double) y[arg_rel]);
    printf("lab: %s %s worst error over its tolerance %.4f at acc % .6f (ref % .6e got % .6e)\n",
           TARGET, phase, max_over, (double) acc[arg_over], (double) y_ref[arg_over], (double) y[arg_over]);
    printf("lab: %s %s nmse %.4e (bound %.4e)\n", TARGET, phase, nmse, nmse_bound);
    printf("lab: %s %s per octave of |acc| (bin: samples, failures, worst abs)\n", TARGET, phase);
    for (int o = 0; o < 18; o++) {
        if (oct_n[o] == 0) {
            continue;
        }
        printf("lab: %s %s   2^%-4d %8lu %8lu  %.4e\n", TARGET, phase, o - 13,
               (unsigned long) oct_n[o], (unsigned long) oct_bad[o], oct_max[o]);
    }
    return n_bad;
}

// Every f16 value through hvx_silu_h_i16, with a check that h stays in its documented range.
// A negative h breaks every caller that reads the f16 exponent field of h, thus the check is a
// contract check and not a tolerance check. O(65536).
#ifndef HVX_SILU_I16_H
static size_t check_silu_range(void) {
    printf("lab: %s silu phase skipped: the kernel tree has no hvx-silu-i16.h\n", TARGET);
    return 0;
}
#else
static size_t check_silu_range(void) {
    static uint16_t in_h[VLEN_FP16]   __attribute__((aligned(VLEN)));
    static int16_t  out_h[VLEN_FP16]  __attribute__((aligned(VLEN)));
    size_t bad = 0, shown = 0;
    int    lo = 32767, hi = -32768;

    for (uint32_t base = 0; base < 0x10000; base += VLEN_FP16) {
        for (uint32_t j = 0; j < VLEN_FP16; j++) {
            in_h[j] = (uint16_t) (base + j);
        }
        HVX_Vector a[4], h[4];
        a[0] = hvx_vmem(in_h);
        hvx_silu_h_i16(a, h, 1);
        hvx_vmem(out_h) = h[0];

        for (uint32_t j = 0; j < VLEN_FP16; j++) {
            const int v = out_h[j];
            if (v < lo) { lo = v; }
            if (v > hi) { hi = v; }
            if (v < 0 || v > SILU_H_MAX) {
                bad++;
                if (shown < 8) {
                    printf("lab: %s silu h out of range: a = % .6f (bits 0x%04x) h16 = %d\n",
                           TARGET, (double) lab_hf_to_f32(in_h[j]), in_h[j], v);
                    shown++;
                }
            }
        }
    }
    printf("lab: %s silu h16 range [%d, %d], bound [0, %d], failures %lu\n",
           TARGET, lo, hi, SILU_H_MAX, (unsigned long) bad);
    return bad;
}
#endif /* HVX_SILU_I16_H */

int main(int argc, char ** argv) {
    const uint32_t n_ch      = (uint32_t) lab_arg_long(argc, argv, "--n_ch", 8192);
    const uint32_t n_tokens  = (uint32_t) lab_arg_long(argc, argv, "--tokens", 16);
    const uint32_t n_threads = (uint32_t) lab_arg_long(argc, argv, "--threads", 1);
    const uint32_t iters     = (uint32_t) lab_arg_long(argc, argv, "--iters", 3);
    // the range of the conv input of the random phase
    const float    x_range   = (float) lab_arg_long(argc, argv, "--range", 4);
    // The tolerance of the octaves phase, as a part per million. The accumulator of that phase is
    // the input itself, thus the error is the f16 rounding of the path, 2^-11, and the error of
    // the SiLU table, 1.6e-4. The defaults hold the four f16 modes, and mode 0 stays far inside.
    const double   abs_tol   = (double) lab_arg_long(argc, argv, "--abs_tol_ppm", 500) * 1e-6;
    const double   rel_tol   = (double) lab_arg_long(argc, argv, "--rel_tol_ppm", 2000) * 1e-6;
    const double   nmse_tol  = (double) lab_arg_long(argc, argv, "--nmse_tol_ppm", 10) * 1e-6;
    // The tolerance of the random phase. Its four taps cancel, thus the absolute error of an f16
    // accumulator is the size of one tap and not the size of the sum: 4 taps of x_range times
    // 2^-11, with a margin of 2. The tight criterion of this phase is the NMSE.
    const double   rnd_abs   = 4.0 * (double) x_range * 2.0 / 2048.0;

    const char * phase = "all";
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--phase") == 0) {
            phase = argv[i + 1];
        }
    }
    const bool do_random  = strcmp(phase, "all") == 0 || strcmp(phase, "random") == 0;
    const bool do_octaves = strcmp(phase, "all") == 0 || strcmp(phase, "octaves") == 0;
    const bool do_silu    = strcmp(phase, "all") == 0 || strcmp(phase, "silu") == 0;

    if (n_tokens < 1) {
        printf("lab: %s needs at least one token\n", TARGET);
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
    float * acc_ref  = lab_ddr_alloc(n_xy * sizeof(float), 128);
    float * n_ref    = lab_ddr_alloc(3 * n_ch * sizeof(float), 128);

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

    size_t fail = 0;
    uint64_t best = UINT64_MAX, total = 0;

    if (do_random) {
        lab_fill_f32(src_slot, 3 * n_ch, -x_range, x_range);
        lab_fill_f32(x, n_xy, -x_range, x_range);
        lab_fill_f32(w, 4 * n_ch, -1.0f, 1.0f);

        // The warm-up run fills the caches with the code and the data
        lab_run_threads(gdn_conv_chunk_thread, &cctx, n_threads);

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

        ref_conv_chunk(src_slot, x, w, y_ref, n_ref, acc_ref, n_ch, n_tokens);
        fail += lab_compare_f32("slot", dst_slot, n_ref, 3 * n_ch, 0.0f, 0.0f);
        fail += report_phase("random", y, y_ref, acc_ref, n_xy, n_ch, rnd_abs, rel_tol, nmse_tol);
    }

    if (do_octaves) {
        // the weights (0, 0, 0, 1), thus the conv accumulator of each token is the input itself
        memset(src_slot, 0, 3 * n_ch * sizeof(float));
        for (uint32_t c = 0; c < n_ch; c++) {
            w[4 * c + 0] = 0.0f;
            w[4 * c + 1] = 0.0f;
            w[4 * c + 2] = 0.0f;
            w[4 * c + 3] = 1.0f;
        }
        for (size_t i = 0; i < n_xy; i++) {
            x[i] = lab_hf_to_f32(sweep_bits((uint32_t) i));
        }
        memset(y, 0, n_xy * sizeof(float));

        lab_run_threads(gdn_conv_chunk_thread, &cctx, n_threads);
        ref_conv_chunk(src_slot, x, w, y_ref, n_ref, acc_ref, n_ch, n_tokens);
        fail += lab_compare_f32("slot_oct", dst_slot, n_ref, 3 * n_ch, 0.0f, 0.0f);
        // The octaves phase reaches values of the size of the f16 range, thus its NMSE is the
        // error of the largest values only. The per-value tolerance is the criterion here.
        fail += report_phase("octaves", y, y_ref, acc_ref, n_xy, n_ch, abs_tol, rel_tol, 1.0);
    }

    if (do_silu) {
        fail += check_silu_range();
    }

    if (do_random) {
        const double blocks = (double) n_xy / VLEN_FP32;
        const double bytes  = (double) n_xy * 2 * sizeof(float);
        lab_report(TARGET, "cycles_per_call_min", (double) best, "cycles");
        lab_report(TARGET, "cycles_per_call_mean", (double) total / iters, "cycles");
        lab_report(TARGET, "cycles_per_block", (double) best / blocks, "cycles");
        lab_report(TARGET, "bytes_per_cycle", bytes / (double) best, "B/cycle");
        lab_report(TARGET, "us_per_token_at_2112_mhz", (double) best / 2112.0 / n_tokens, "us");
    }
    lab_report(TARGET, "n_ch", n_ch, "");
    lab_report(TARGET, "tokens", n_tokens, "");
    lab_report(TARGET, "threads", n_threads, "");
    lab_report(TARGET, "failures", (double) fail, "");
    return fail ? 1 : 0;
}
