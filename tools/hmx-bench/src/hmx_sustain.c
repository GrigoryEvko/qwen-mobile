// The sustained rate of the HMX f16 path on the device, with no DMA and no HVX work.
//
// A long prefill matmul becomes slower during a sustained run while the core clock stays constant,
// and it gets its rate back after a pause of about 40 ms (the roofline measurement of 2026-09-24 in
// the 4B model). This program asks if the HMX alone shows that behavior. It runs the pattern of the
// production f16 matmul (htp/hmx-mm-kernels-tiled.h core_dot_chunk_fp16): for each output tile a
// clear, one deep MAC chain of K_TILES k-tiles and one store, for N_COLS output tiles. All operands
// are in VTCM, thus only the core and the HMX run.
//
//   1. The series: RUN_MS of continuous passes. Each window of WIN_US gives one line with the passes,
//      the wall time and the core pcycles of the window (integers only: FARF takes no float). One pass
//      is N_COLS x K_TILES tile MACs of 2 x 32^3 FLOP each. A slower HMX gives fewer passes in a window,
//      and more pcycles for each pass when the core clock stays constant.
//   2. The recovery: for each pause of PAUSES_MS, HEAT_MS of continuous passes, then qurt_sleep for the
//      pause, then REC_WINDOWS windows. The first window after the pause shows the recovered rate.
//
// Each result line starts with "sustain:". Arguments: --out <path> writes every line to a file as well
// (the log of the phone can drop lines). Build: tools/hmx-bench/build-sustain.sh. Run: from its directory,
// "ADSP_LIBRARY_PATH=<dir> ./run_main_on_hexagon 3 hmx_sustain.so --out <file>". The stage sweep
// (tools/stages/sweep/stage.py) runs it as the run hs and prints its table.
//
// Complexity: RUN_MS + len(PAUSES_MS) x HEAT_MS of HMX work, about 5 s on the device.

#include "lab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HAP_farf.h"
#include "HAP_perf.h"
#include "qurt.h"

#ifdef LAB_DEVICE
void lab_fini(void);
#else
static inline void lab_fini(void) {}
#endif

static FILE * g_out;

#define LOG(...)                                   \
    do {                                           \
        printf("sustain: " __VA_ARGS__);           \
        printf("\n");                              \
        FARF(ALWAYS, "sustain: " __VA_ARGS__);     \
        if (g_out) {                               \
            fprintf(g_out, "sustain: " __VA_ARGS__); \
            fprintf(g_out, "\n");                  \
            fflush(g_out);                         \
        }                                          \
    } while (0)

#define K_TILES     80u    // the 2560 reduction of the 4B, in tiles of 32
#define N_COLS      8u     // output tiles of one pass
#define TILE        2048u  // bytes of one f16 tile
#define WIN_US      1000u  // one window of the series
#define RUN_MS      800u   // the continuous run of the series
#define HEAT_MS     400u   // the continuous run before each pause
#define REC_WINDOWS 5u     // the windows after each pause
#define MAX_WIN     1024u

static const uint32_t PAUSES_MS[] = { 1, 2, 5, 10, 20, 40, 80, 160 };

static uint8_t * g_act;  // K_TILES activation tiles
static uint8_t * g_wgt;  // N_COLS x K_TILES weight tiles
static uint8_t * g_dst;  // N_COLS output tiles
static uint8_t * g_tab;  // the f16 bias area: scale 1.0 and bias 0 for each column

static inline void clr_f16(void) { asm volatile("mxclracc.hf\n" ::: "memory"); }
static inline void bias_f16(const uint8_t * t) { asm volatile("bias = mxmem2(%0)\n" : : "r"(t) : "memory"); }
static inline void st_f16(uint8_t * o) { asm volatile("mxmem(%0, %1):after.hf = acc\n" : : "r"(o), "r"(0) : "memory"); }
static inline void sync_mem(void) { asm volatile("syncht\n" ::: "memory"); }

static inline void mac_f16_deep(const uint8_t * a, const uint8_t * w, uint32_t r) {
    asm volatile("{ activation.hf = mxmem(%0, %2):deep\n weight.hf = mxmem(%1, %2) }\n"
                 : : "r"(a), "r"(w), "r"(r) : "memory");
}

// One pass: N_COLS output tiles, each a chain of K_TILES tiles (at most 32 tiles for each
// instruction, because the range operand holds 16 bits). O(N_COLS x K_TILES).
static inline void pass(void) {
    for (uint32_t c = 0; c < N_COLS; c++) {
        const uint8_t * a = g_act;
        const uint8_t * w = g_wgt + (size_t) c * K_TILES * TILE;
        uint32_t        k = K_TILES;
        clr_f16();
        while (k >= 32) {
            mac_f16_deep(a, w, 32 * TILE - 1);
            a += 32 * TILE;
            w += 32 * TILE;
            k -= 32;
        }
        if (k) {
            mac_f16_deep(a, w, k * TILE - 1);
        }
        st_f16(g_dst + c * TILE);
    }
}

struct window {
    uint32_t passes;
    uint32_t us;
    uint64_t pcycles;
};

// Runs passes for one window of at least WIN_US and waits for the last store. O(passes).
static struct window run_window(void) {
    struct window  w  = { 0, 0, 0 };
    const uint64_t t0 = HAP_perf_get_time_us();
    const uint64_t c0 = HAP_perf_get_pcycles();
    uint64_t       t  = t0;
    bias_f16(g_tab);
    while (t - t0 < WIN_US) {
        pass();
        w.passes++;
        t = HAP_perf_get_time_us();
    }
    sync_mem();
    w.us      = (uint32_t) (HAP_perf_get_time_us() - t0);
    w.pcycles = HAP_perf_get_pcycles() - c0;
    return w;
}

static struct window g_series[MAX_WIN];

int main(int argc, char ** argv) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--out") == 0) {
            g_out = fopen(argv[i + 1], "w");
        }
    }
    lab_init();

    g_act = lab_vtcm_alloc(K_TILES * TILE, 4096);
    g_wgt = lab_vtcm_alloc((size_t) N_COLS * K_TILES * TILE, 4096);
    g_dst = lab_vtcm_alloc(N_COLS * TILE, 4096);
    g_tab = lab_vtcm_alloc(4096, 4096);

    // Small f16 values (the byte mask keeps each value finite), scale 1.0 (0x3c00) and bias 0
    for (uint32_t i = 0; i < K_TILES * TILE; i++) {
        g_act[i] = (uint8_t) ((i * 7u + 1u) & 0x3F);
    }
    for (uint32_t i = 0; i < N_COLS * K_TILES * TILE; i++) {
        g_wgt[i] = (uint8_t) ((i * 13u + 3u) & 0x3F);
    }
    uint32_t * fb = (uint32_t *) g_tab;
    for (uint32_t i = 0; i < 32; i++) {
        fb[i]      = 0x3c00u;
        fb[32 + i] = 0;
    }
    sync_mem();
    bias_f16(g_tab);

    LOG("config k_tiles=%u n_cols=%u win_us=%u run_ms=%u heat_ms=%u", (unsigned) K_TILES, (unsigned) N_COLS,
        (unsigned) WIN_US, (unsigned) RUN_MS, (unsigned) HEAT_MS);

    // 1. The series after an idle start
    qurt_sleep(200000);
    const uint32_t n_win = RUN_MS * 1000u / WIN_US < MAX_WIN ? RUN_MS * 1000u / WIN_US : MAX_WIN;
    const uint64_t s0    = HAP_perf_get_time_us();
    for (uint32_t i = 0; i < n_win; i++) {
        g_series[i] = run_window();
    }
    const uint32_t total_us = (uint32_t) (HAP_perf_get_time_us() - s0);
    uint32_t       t_us     = 0;
    for (uint32_t i = 0; i < n_win; i++) {
        const struct window * w = &g_series[i];
        LOG("series i=%u t_us=%u passes=%u us=%u pcycles=%llu", (unsigned) i, (unsigned) t_us, (unsigned) w->passes,
            (unsigned) w->us,
            (unsigned long long) w->pcycles);
        t_us += w->us;
    }
    LOG("series total_us=%u windows=%u", (unsigned) total_us, (unsigned) n_win);

    // 2. The recovery after each pause
    for (uint32_t p = 0; p < sizeof(PAUSES_MS) / sizeof(PAUSES_MS[0]); p++) {
        const uint64_t h0 = HAP_perf_get_time_us();
        struct window  last = { 0, 0, 0 };
        while (HAP_perf_get_time_us() - h0 < HEAT_MS * 1000u) {
            last = run_window();
        }
        qurt_sleep(PAUSES_MS[p] * 1000u);
        for (uint32_t r = 0; r < REC_WINDOWS; r++) {
            const struct window w = run_window();
            LOG("recovery pause_ms=%u window=%u passes=%u us=%u pcycles=%llu heat_passes=%u heat_us=%u",
                (unsigned) PAUSES_MS[p], (unsigned) r, (unsigned) w.passes, (unsigned) w.us, (unsigned long long) w.pcycles,
                (unsigned) last.passes, (unsigned) last.us);
        }
    }

    lab_fini();
    LOG("done");
    if (g_out) {
        fclose(g_out);
    }
    return 0;
}
