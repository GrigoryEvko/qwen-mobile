// Target 12: one token of the gated delta rule, gdn_step_token_f32 of gated-delta-net-ops.c.
//
// The chunked kernel gives the state at a chunk boundary only. Speculative decoding asks for the
// state after each of the last K tokens of a batch, thus the chunk loop stops K tokens early and
// this function does those tokens. The cost of the tail is the cost of this function times K, and
// that cost decides whether the chunked kernel can take K > 1 at all.
//
// The function is HVX only, thus the timing model of the simulator retires every instruction of it
// and the cycles here are real. The float64 check runs after the timing.
//
// Arguments: --d 128 --iters 8 --kda 0
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "gdn-chunk-ops.c"
#include "gated-delta-net-ops.c"

#define TARGET "gdntail"

// The float64 reference of one token. The state is transposed: state[j * D + i] = S[i][j], i the
// key index and j the value index. O(D * D).
static void ref_step(double * state, const double * q, const double * k, const double * v,
                     const double * g, double beta, double * attn, uint32_t D, bool kda, double scale) {
    double d[HTP_GDN_MAX_SV];

    for (uint32_t j = 0; j < D; j++) {
        double * row = state + (size_t) j * D;
        for (uint32_t i = 0; i < D; i++) {
            row[i] *= exp(kda ? g[i] : g[0]);
        }
        double sum = 0.0;
        for (uint32_t i = 0; i < D; i++) {
            sum += row[i] * k[i];
        }
        d[j] = (v[j] - sum) * beta;
    }
    for (uint32_t j = 0; j < D; j++) {
        double * row = state + (size_t) j * D;
        double   acc = 0.0;
        for (uint32_t i = 0; i < D; i++) {
            row[i] += k[i] * d[j];
            acc += row[i] * q[i];
        }
        attn[j] = acc * scale;
    }
}

int main(int argc, char ** argv) {
    const uint32_t D     = (uint32_t) lab_arg_long(argc, argv, "--d", 128);
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 8);
    const bool     kda   = lab_arg_long(argc, argv, "--kda", 0) != 0;

    if (D == 0 || D > HTP_GDN_MAX_SV) {
        printf("lab: %s head size %u is out of range\n", TARGET, D);
        return 2;
    }

    lab_init();
    printf("lab: %s head size D %u kda %d\n", TARGET, D, (int) kda);

    const size_t dd    = (size_t) D * D;
    const double scale = 1.0 / sqrt((double) D);

    // The state lives in VTCM, as it does in both kernels. The inputs live in DDR, as they do for
    // the tail: the kernel reads full vectors at unaligned addresses, thus each buffer has a margin.
    float * state = lab_vtcm_alloc(dd * sizeof(float), 128);
    float * q     = lab_ddr_alloc(D * sizeof(float) + 256, 128);
    float * k     = lab_ddr_alloc(D * sizeof(float) + 256, 128);
    float * v     = lab_ddr_alloc(D * sizeof(float) + 256, 128);
    float * g     = lab_ddr_alloc(D * sizeof(float) + 256, 128);
    float * attn  = lab_ddr_alloc(D * sizeof(float) + 256, 128);

    lab_fill_f32(state, dd, -0.5f, 0.5f);
    lab_fill_f32(q, D, -1.0f, 1.0f);
    lab_fill_f32(k, D, -1.0f, 1.0f);
    lab_fill_f32(v, D, -0.3f, 5.0f);
    lab_fill_f32(g, D, -2.0f, -1e-4f);   // the log decay of the test of test-backend-ops
    const float beta = 0.5f;

    // q and k of this op come from an L2 norm, thus they have unit length
    float nq = 0.0f, nk = 0.0f;
    for (uint32_t i = 0; i < D; i++) {
        nq += q[i] * q[i];
        nk += k[i] * k[i];
    }
    nq = 1.0f / sqrtf(nq);
    nk = 1.0f / sqrtf(nk);
    for (uint32_t i = 0; i < D; i++) {
        q[i] *= nq;
        k[i] *= nk;
    }

    float * state0 = lab_ddr_alloc(dd * sizeof(float), 128);
    memcpy(state0, state, dd * sizeof(float));

    // ---- timing. One call is one token of one head.
    uint64_t best = UINT64_MAX;
    gdn_step_token_f32(state, q, k, v, g, beta, attn, D, kda, (float) scale);
    for (uint32_t it = 0; it < iters; it++) {
        memcpy(state, state0, dd * sizeof(float));
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        gdn_step_token_f32(state, q, k, v, g, beta, attn, D, kda, (float) scale);
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best) {
            best = t1 - t0;
        }
    }

    const uint32_t blocks = (uint32_t) (dd / VLEN_FP32);
    printf("lab: %s step %llu cycles, %.2f cycles per 32-lane block (%u blocks)\n", TARGET,
           (unsigned long long) best, (double) best / (double) blocks, blocks);
    lab_report(TARGET, "step_cycles", (double) best, "cycles");
    lab_report(TARGET, "step_cycles_per_block", (double) best / (double) blocks, "cycles");
    lab_report(TARGET, "step_us_at_2112_mhz", (double) best / 2112.0, "us");

    // ---- the float64 check, after the timing
    memcpy(state, state0, dd * sizeof(float));
    gdn_step_token_f32(state, q, k, v, g, beta, attn, D, kda, (float) scale);

    double * ref_state = lab_ddr_alloc(dd * sizeof(double), 128);
    double * ref_attn  = lab_ddr_alloc(D * sizeof(double), 128);
    double   dq[HTP_GDN_MAX_SV], dk[HTP_GDN_MAX_SV], dv[HTP_GDN_MAX_SV], dg[HTP_GDN_MAX_SV];
    for (size_t i = 0; i < dd; i++) {
        ref_state[i] = (double) state0[i];
    }
    for (uint32_t i = 0; i < D; i++) {
        dq[i] = (double) q[i];
        dk[i] = (double) k[i];
        dv[i] = (double) v[i];
        dg[i] = (double) g[i];
    }
    ref_step(ref_state, dq, dk, dv, dg, (double) beta, ref_attn, D, kda, scale);

    double se_s = 0.0, sr_s = 0.0, se_a = 0.0, sr_a = 0.0;
    for (size_t i = 0; i < dd; i++) {
        const double d = (double) state[i] - ref_state[i];
        se_s += d * d;
        sr_s += ref_state[i] * ref_state[i];
    }
    for (uint32_t j = 0; j < D; j++) {
        const double d = (double) attn[j] - ref_attn[j];
        se_a += d * d;
        sr_a += ref_attn[j] * ref_attn[j];
    }
    const double nmse_s = sr_s > 0.0 ? se_s / sr_s : 0.0;
    const double nmse_a = sr_a > 0.0 ? se_a / sr_a : 0.0;
    printf("lab: %s nmse state %.3g attn %.3g\n", TARGET, nmse_s, nmse_a);
    lab_report(TARGET, "nmse_state", nmse_s, "");
    lab_report(TARGET, "nmse_attn", nmse_a, "");

    // A tail of K tokens costs K times the step, plus one transposed copy of the state for each
    // slot. The copy is a DMA, which the simulator does not model, thus only the step is timed.
    for (uint32_t K = 2; K <= 5; K++) {
        printf("lab: %s tail of K %u: %llu cycles of step work for one head, %.2f us at 2112 MHz\n",
               TARGET, K, (unsigned long long) (best * K), (double) (best * K) / 2112.0);
    }
    lab_report(TARGET, "head_D", (double) D, "");
    return (nmse_s < 1e-12 && nmse_a < 1e-12) ? 0 : 1;
}
