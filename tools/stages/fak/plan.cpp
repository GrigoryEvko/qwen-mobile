// The flash attention plans of the 4B shapes and their modeled times, for the kernels of patches/hexagon-fa.
//
// Build and run on the box, against the llama.cpp submodule with the patch series:
//
//   T=third_party/llama.cpp/ggml/src/ggml-hexagon
//   g++ -std=c++17 -O1 -D__fp16=_Float16 -I$T/htp -I$T -I$T/../../include tools/stages/fak/plan.cpp -o /tmp/fak-plan
//   /tmp/fak-plan [n_threads]
//
// The program prints two tables for 16 query heads, 4 KV heads, head size 256, Q8_0 K and V, F32 Q and 8 MB
// of VTCM, with the thread count of the argument (6, the preset: the HVX threads of v79):
//   1. The prefill shapes: the plan of HTP_FA_KERNEL_HMX (hmx_fa_find_chunk_size) and the plan of
//      HTP_FA_KERNEL_HMX2 (hmx_fa2_find_chunk_size) with its modeled time of one op. The model counts core
//      cycles, and the time is at 2112 MHz. The model has the constants of v79.
//   2. The decode shapes: the plan of HTP_FA_KERNEL_HMX and the spans of HTP_FA_KERNEL_HMX2.
// The time is O(shapes * Br_max * Bc_limit / 64) for the searches.
#include <cstdio>
#include <cstdlib>

#include "flash-attn-ops.h"

namespace {

constexpr size_t kG = 4, kD = 256, kVtcm = 8u << 20, kKvHeads = 4;

// The modeled ms of one op of HTP_FA_KERNEL_HMX2 with the plan (Br, Bc), for all KV heads.
double ms_fa2(size_t n, size_t kv, size_t Br, size_t Bc, size_t n_threads) {
    uint64_t c = (n / Br) * hmx_fa2_cost_q_block(Br * kG, kv, Bc, kD, kD, n_threads, true);
    if (n % Br) {
        c += hmx_fa2_cost_q_block((n % Br) * kG, kv, Bc, kD, kD, n_threads, true);
    }
    return (double) c * kKvHeads / 2112e3;
}

}  // namespace

int main(int argc, char ** argv) {
    const size_t n_thr = argc > 1 ? (size_t) std::strtoul(argv[1], nullptr, 10) : 6;
    if (n_thr == 0) {
        std::fprintf(stderr, "fak-plan: the thread count must be 1 or more\n");
        return 2;
    }
    std::printf("1. prefill, %zu threads: queries x KV rows | HTP_FA_KERNEL_HMX Br x Bc | HTP_FA_KERNEL_HMX2 Br x Bc, ms\n",
                n_thr);
    const size_t prefill[][2] = {{512, 512}, {1024, 1024}, {1024, 2048}, {1024, 3072}, {1024, 4096},
                                 {512, 4608}, {1024, 8192}, {1024, 16384}};
    for (const auto & sh : prefill) {
        const size_t n = sh[0], kv = sh[1];
        size_t a_br = 0, a_bc = 0, t_br = 0, t_bc = 0;
        hmx_fa_find_chunk_size(&a_br, &a_bc, kG, kD, kD, n, kv, kVtcm, n_thr, true);
        hmx_fa2_find_chunk_size(&t_br, &t_bc, kG, kD, kD, n, kv, kVtcm, n_thr, true, true);
        std::printf("  %4zu x %5zu | %3zu x %-4zu | %3zu x %-4zu %6.2f\n", n, kv, a_br, a_bc, t_br, t_bc,
                    ms_fa2(n, kv, t_br, t_bc, n_thr));
    }

    std::printf("2. decode: queries x KV rows | HTP_FA_KERNEL_HMX Br x Bc | spans Br x Bc, span count\n");
    const size_t decode[][2] = {{1, 512}, {1, 4096}, {1, 16384}, {4, 4096}, {8, 4096}};
    for (const auto & sh : decode) {
        const size_t n = sh[0], kv = sh[1];
        size_t a_br = 0, a_bc = 0, s_br = 0, s_bc = 0;
        hmx_fa_find_chunk_size(&a_br, &a_bc, kG, kD, kD, n, kv, kVtcm, n_thr, true);
        hmx_fa2_find_span_size(&s_br, &s_bc, kG, kD, kD, n, kv, kVtcm, n_thr, true);
        std::printf("  %4zu x %5zu | %2zu x %-4zu | %2zu x %-4zu, %zu\n", n, kv, a_br, a_bc, s_br, s_bc,
                    (kv + s_bc - 1) / s_bc);
    }
    return 0;
}
