// The flash attention plans of the 4B shapes and of its vision encoder, and their modeled times.
//
// Build and run on the box, against a llama.cpp tree with the patches of patches/hexagon-fa:
//
//   T=third_party/llama.cpp/ggml/src/ggml-hexagon
//   g++ -std=c++17 -O1 -D__fp16=_Float16 -I$T/htp -I$T -I$T/../../include tools/stages/fak/plan.cpp -o /tmp/fak-plan
//   /tmp/fak-plan
//
// The program prints three tables:
//   1. The prefill shapes of the 4B (16 query heads, 4 KV heads, head size 256, Q8_0 K and V, F32 Q, 6
//      threads, 8 MB VTCM): the plan and the modeled time of one op for A (hmx_fa_find_chunk_size), B
//      (hmx_fa_find_chunk_size_v2), C (HTP_FA_KERNEL_HMX2 streaming) and D (HTP_FA_KERNEL_HMX2 with the
//      resident form). The model counts core cycles, and the time is at 2112 MHz.
//   2. The decode shapes: the plans of A, B and the spans of HTP_FA_KERNEL_HMX2.
//   3. The count of the KV lengths from 1 to 512 where A and B give different plans, for 1 to 8 queries
//      and for 512 queries.
// The time is O(shapes * Br_max * Bc_limit / 64) for the searches.
#include <cstdio>

#include "flash-attn-ops.h"

namespace {

constexpr size_t kG = 4, kD = 256, kVtcm = 8u << 20, kThreads = 6, kKvHeads = 4;

// The modeled ms of one op of HTP_FA_KERNEL_HMX with the plan (Br, Bc), for all KV heads.
double ms_hmx(size_t n, size_t kv, size_t Br, size_t Bc) {
    uint64_t c = (n / Br) * hmx_fa_cost_q_block(Br * kG, kv, Bc, kD, kD, kThreads, true);
    if (n % Br) {
        c += hmx_fa_cost_q_block((n % Br) * kG, kv, Bc, kD, kD, kThreads, true);
    }
    return (double) c * kKvHeads / 2112e3;
}

// The modeled ms of one op of HTP_FA_KERNEL_HMX2 with the plan (Br, Bc) and the form, for all KV heads.
double ms_fa2(size_t n, size_t kv, size_t Br, size_t Bc, bool resident) {
    uint64_t c = resident ? hmx_fa_cost_conv(kv, kD, kD, true) / kThreads + FA_COST_PHASE : 0;
    c += (n / Br) * hmx_fa2_cost_q_block(Br * kG, kv, Bc, kD, kD, kThreads, resident, true);
    if (n % Br) {
        c += hmx_fa2_cost_q_block((n % Br) * kG, kv, Bc, kD, kD, kThreads, resident, true);
    }
    return (double) c * kKvHeads / 2112e3;
}

}  // namespace

int main() {
    std::printf("1. prefill: queries x KV rows | A Br x Bc, ms | B | C streaming | D\n");
    const size_t prefill[][2] = {{512, 512}, {1024, 1024}, {1024, 2048}, {1024, 3072}, {1024, 4096},
                                 {512, 4608}, {1024, 8192}, {1024, 16384}};
    for (const auto & sh : prefill) {
        const size_t n = sh[0], kv = sh[1];
        size_t a_br = 0, a_bc = 0, b_br = 0, b_bc = 0, c_br = 0, c_bc = 0, d_br = 0, d_bc = 0;
        bool   c_res = false, d_res = false;
        hmx_fa_find_chunk_size(&a_br, &a_bc, kG, kD, kD, n, kv, kVtcm, kThreads, true);
        hmx_fa_find_chunk_size_v2(&b_br, &b_bc, kG, kD, kD, n, kv, kVtcm, kThreads, true, true);
        hmx_fa2_find_chunk_size(&c_br, &c_bc, &c_res, kG, kD, kD, n, kv, kVtcm, kThreads, true, true, false);
        hmx_fa2_find_chunk_size(&d_br, &d_bc, &d_res, kG, kD, kD, n, kv, kVtcm, kThreads, true, true, true);
        std::printf("  %4zu x %5zu | %3zu x %-4zu %6.2f | %3zu x %-4zu %6.2f | %3zu x %-4zu %6.2f | %s %3zu x %-4zu %6.2f\n",
                    n, kv, a_br, a_bc, ms_hmx(n, kv, a_br, a_bc), b_br, b_bc, ms_hmx(n, kv, b_br, b_bc), c_br, c_bc,
                    ms_fa2(n, kv, c_br, c_bc, false), d_res ? "res " : "strm", d_br, d_bc,
                    ms_fa2(n, kv, d_br, d_bc, d_res));
    }

    std::printf("2. decode: queries x KV rows | A Br x Bc | B Br x Bc | spans Br x Bc, span count\n");
    const size_t decode[][2] = {{1, 512}, {1, 4096}, {1, 16384}, {4, 4096}};
    for (const auto & sh : decode) {
        const size_t n = sh[0], kv = sh[1];
        size_t a_br = 0, a_bc = 0, b_br = 0, b_bc = 0, s_br = 0, s_bc = 0;
        hmx_fa_find_chunk_size(&a_br, &a_bc, kG, kD, kD, n, kv, kVtcm, kThreads, true);
        hmx_fa_find_chunk_size_v2(&b_br, &b_bc, kG, kD, kD, n, kv, kVtcm, kThreads, true, true);
        hmx_fa2_find_span_size(&s_br, &s_bc, kG, kD, kD, n, kv, kVtcm, kThreads, true);
        std::printf("  %4zu x %5zu | %2zu x %-4zu | %2zu x %-4zu | %2zu x %-4zu, %zu\n", n, kv, a_br, a_bc, b_br, b_bc,
                    s_br, s_bc, (kv + s_bc - 1) / s_bc);
    }

    std::printf("3. KV lengths 1 to 512 where A and B give different plans\n");
    const size_t queries[] = {1, 2, 4, 8, 512};
    for (size_t n : queries) {
        int  diff = 0;
        bool same_512 = false;
        for (size_t kv = 1; kv <= 512; ++kv) {
            size_t a_br = 0, a_bc = 0, b_br = 0, b_bc = 0;
            const int  ra   = hmx_fa_find_chunk_size(&a_br, &a_bc, kG, kD, kD, n, kv, kVtcm, kThreads, true);
            const int  rb   = hmx_fa_find_chunk_size_v2(&b_br, &b_bc, kG, kD, kD, n, kv, kVtcm, kThreads, true, true);
            const bool same = ra == rb && a_br == b_br && a_bc == b_bc;
            diff += same ? 0 : 1;
            same_512 = kv == 512 ? same : same_512;
        }
        std::printf("  %3zu queries: %d of 512 differ, the plans at 512 KV rows are %s\n", n, diff,
                    same_512 ? "the same" : "different");
    }
    return 0;
}
