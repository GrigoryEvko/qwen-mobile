// The HMX rate of the f16 mode against the int8 mode, measured on the device.
//
// The question: our matmul expands every weight to f16 and drives the HMX in its
// f16 mode. The vendor block geometry is 32 x 32 x 32 for f16 and 64 x 32 x 32 for
// int8, thus one int8 issue does 65536 multiply-accumulate operations where one f16
// issue does 32768. This program measures whether the silicon gives that factor.
//
// It reports five things:
//   1. hmx_fp16_rate, the multiply-accumulate rate per cycle that the hardware
//      declares. A value of 0 means the f16 mode is absent on this part.
//   2. A check of each mode against a reference matmul on the scalar unit. Without
//      it a call that fails every time still gives a cycle count and a plausible
//      ratio, because the timed loops cannot report a status.
//   3. The issue rate of each mode: the cycles of one hexkl_micro_hmx_mm_* call in a
//      chain of GDN_K_TILES calls, which is the inner loop of a real kernel.
//   4. The cost of one accumulator read of each mode, measured on its own. This is
//      the number that decides which weight formats the int8 path can carry. The
//      int32 accumulator has no f16 store and no f32 store, thus a full int32 read
//      is four byte-plane stores where the f16 read is one store.
//   5. The break-even k tile count: the k tiles that one int8 accumulator read must
//      cover before the faster multiply pays for the slower read. A weight format
//      whose scale changes more often than that cannot use the int8 path.
//
// The shape is the one the 4B model runs: a reduction of 2560 gives 80 tiles of 32.
//
// Licence: this program links libhexkl_micro.a from the Qualcomm HexKL package,
// which is click-through licensed. Build and run it locally. Never commit the
// archive and never put it in a published image.

#include "hexkl_micro.h"
#include "hexkl_test.h"

#include "HAP_perf.h"

#define GDN_K_TILES 80U   // the 2560 reduction of the 4B, in tiles of 32
#define N_COL_TILES 8U    // output column tiles per timed pass
#define N_REPS      20U   // timed passes

#define I8_TILE_ROWS 64U  // activation rows of one int8 HMX tile
#define I8_TILE_COLS 32U  // channels of one int8 HMX tile
#define I8_TILE_K    32U  // reduction of one int8 HMX tile

/** The u8 activation of the correctness case, row major. */
static uint8_t g_check_act[I8_TILE_ROWS * I8_TILE_K];

/** The s8 weight of the correctness case, row major. */
static int8_t g_check_wt[I8_TILE_K * I8_TILE_COLS];

/** The int32 product that the HMX gives. */
static int32_t g_check_hmx[I8_TILE_ROWS * I8_TILE_COLS];

/** The int32 product that the scalar unit gives. */
static int32_t g_check_ref[I8_TILE_ROWS * I8_TILE_COLS];

/**
 * The offsets of one VTCM layout that holds the activation tiles, one weight tile
 * and the HMX configuration.
 *
 * The activation tiles sit at the base. The weight tile and the configuration sit
 * at the top, because the configuration size is a runtime value.
 */
struct vtcm_plan {
    uint32_t weight;   /**< byte offset of the one weight tile */
    uint32_t result;   /**< byte offset of the accumulator read destination */
    uint32_t config;   /**< byte offset of the HMX configuration block */
};

/**
 * Build the VTCM layout for a chain of k tiles.
 *
 * @param vtcm_size The bytes of the VTCM region
 * @param k_tiles The activation tiles that the chain holds
 * @param act_bytes The bytes of one activation tile slot
 * @return The offsets, all inside the region
 */
static struct vtcm_plan plan_vtcm(uint32_t vtcm_size, uint32_t k_tiles, uint32_t act_bytes) {
    struct vtcm_plan p;
    p.config = vtcm_size - hexkl_micro_hmx_config_size();
    p.config &= ~(HEXKL_HMX_CONFIG_ALIGNMENT - 1U);
    p.result = (p.config - 64U * 32U * 4U) & ~(HEXKL_HMX_ACTIVATION_ALIGNMENT - 1U);
    p.weight = (p.result - 4096U) & ~(HEXKL_HMX_WEIGHTS_ALIGNMENT - 1U);
    (void) k_tiles;
    (void) act_bytes;
    return p;
}

/**
 * Do a check of the int8 mode against a reference matmul on the scalar unit.
 *
 * One 64 by 32 activation tile multiplies one 32 by 32 weight tile. The function
 * converts both operands into the layout that the HMX wants, runs one multiply,
 * reads the int32 accumulator, converts the result back, and compares it with
 * hexkl_test_matmul_cm_u8i8_i32.
 *
 * The function writes the configuration region and the operand region of the VTCM.
 * Call it before the timed passes fill those regions.
 *
 * Complexity: O(rows * cols * k) on the scalar unit, which is 65536 operations.
 *
 * @param vtcm_base The base of the VTCM region
 * @param p The VTCM layout
 * @return AEE_SUCCESS when every element agrees, an error code otherwise
 */
static int check_i8_against_reference(uint8_t * vtcm_base, const struct vtcm_plan * p) {
    for (uint32_t i = 0; i < I8_TILE_ROWS * I8_TILE_K; i++) {
        g_check_act[i] = (uint8_t) (i * 7U + 1U);
    }
    for (uint32_t i = 0; i < I8_TILE_K * I8_TILE_COLS; i++) {
        g_check_wt[i] = (int8_t) (int32_t) ((i * 13U + 3U) % 251U) - 125;
    }

    int res = hexkl_micro_hmx_copy_submatrix_to_8b_activation(
        vtcm_base, 0U, g_check_act, 0U, 0U, I8_TILE_ROWS, I8_TILE_K);
    if (res != AEE_SUCCESS) {
        HEXKL_TEST_ERROR_PRINT("copy_submatrix_to_8b_activation failed 0x%x", res);
        return res;
    }

    res = hexkl_micro_hmx_rm_to_wh_i8(vtcm_base, p->weight, g_check_wt, 0U, 0U, I8_TILE_COLS);
    if (res != AEE_SUCCESS) {
        HEXKL_TEST_ERROR_PRINT("rm_to_wh_i8 failed 0x%x", res);
        return res;
    }

    res = hexkl_micro_hmx_setup_acc_read_int32(vtcm_base, p->config);
    if (res != AEE_SUCCESS) {
        HEXKL_TEST_ERROR_PRINT("setup_acc_read_int32 failed 0x%x", res);
        return res;
    }

    hexkl_micro_hmx_acc_clear_int32();
    res = hexkl_micro_hmx_mm_u8i8(vtcm_base, 0U, p->weight);
    if (res != AEE_SUCCESS) {
        HEXKL_TEST_ERROR_PRINT("mm_u8i8 failed 0x%x", res);
        return res;
    }

    res = hexkl_micro_hmx_acc_read_int32(vtcm_base, p->config, p->result);
    if (res != AEE_SUCCESS) {
        HEXKL_TEST_ERROR_PRINT("acc_read_int32 failed 0x%x", res);
        return res;
    }

    res = hexkl_micro_hmx_copy_32b_to_submatrix(
        vtcm_base, p->result, g_check_hmx, 0U, 0U, I8_TILE_ROWS, I8_TILE_COLS);
    if (res != AEE_SUCCESS) {
        HEXKL_TEST_ERROR_PRINT("copy_32b_to_submatrix failed 0x%x", res);
        return res;
    }

    res = hexkl_test_matmul_cm_u8i8_i32(I8_TILE_ROWS, I8_TILE_COLS, I8_TILE_K, I8_TILE_COLS,
                                        g_check_ref, g_check_act, g_check_wt);
    if (res != AEE_SUCCESS) {
        HEXKL_TEST_ERROR_PRINT("reference matmul failed 0x%x", res);
        return res;
    }

    return hexkl_test_vector_check_i32(I8_TILE_ROWS * I8_TILE_COLS, g_check_ref, g_check_hmx);
}

/**
 * The cycles of one timed pass of the f16 mode.
 *
 * One pass runs N_COL_TILES output column tiles. Each column tile clears the
 * accumulator, issues GDN_K_TILES multiplies against resident tiles, then reads the
 * accumulator. No weight layout runs inside the timed region, thus the result is the
 * issue rate of the matrix engine and not of the data movement around it.
 *
 * A k_tiles of 0 leaves the clear and the read only, thus it measures the read.
 *
 * @param vtcm_base The base of the VTCM region
 * @param p The VTCM layout
 * @param k_tiles The multiplies of one column tile
 * @return The pcycles of the pass
 */
static uint64_t time_f16_issue(uint8_t * vtcm_base, const struct vtcm_plan * p, uint32_t k_tiles) {
    const uint64_t t0 = HAP_perf_get_pcycles();
    for (uint32_t c = 0; c < N_COL_TILES; c++) {
        hexkl_micro_hmx_acc_clear_f16();
        for (uint32_t i = 0; i < k_tiles; i++) {
            hexkl_micro_hmx_mm_f16(vtcm_base, HEXKL_HMX_ACTIVATION_ALIGNMENT * i, p->weight);
        }
        hexkl_micro_hmx_acc_read_f16(vtcm_base, p->config, p->result);
    }
    return HAP_perf_get_pcycles() - t0;
}

/**
 * The cycles of one timed pass of the int8 mode. Refer to time_f16_issue.
 *
 * @param vtcm_base The base of the VTCM region
 * @param p The VTCM layout
 * @param k_tiles The multiplies of one column tile
 * @return The pcycles of the pass
 */
static uint64_t time_i8_issue(uint8_t * vtcm_base, const struct vtcm_plan * p, uint32_t k_tiles) {
    const uint64_t t0 = HAP_perf_get_pcycles();
    for (uint32_t c = 0; c < N_COL_TILES; c++) {
        hexkl_micro_hmx_acc_clear_int32();
        for (uint32_t i = 0; i < k_tiles; i++) {
            hexkl_micro_hmx_mm_u8i8(vtcm_base, HEXKL_HMX_ACTIVATION_ALIGNMENT * i, p->weight);
        }
        hexkl_micro_hmx_acc_read_int32(vtcm_base, p->config, p->result);
    }
    return HAP_perf_get_pcycles() - t0;
}

/**
 * The cycles of one timed pass that clears and reads the accumulator and multiplies
 * nothing. The difference against a pass with multiplies is not necessary here,
 * because the clear and the read are the whole body.
 *
 * @param vtcm_base The base of the VTCM region
 * @param p The VTCM layout
 * @param f16 True for the f16 mode, false for the int8 mode
 * @return The pcycles of N_COL_TILES clear-and-read pairs
 */
static uint64_t time_read_only(uint8_t * vtcm_base, const struct vtcm_plan * p, int f16) {
    return f16 ? time_f16_issue(vtcm_base, p, 0U) : time_i8_issue(vtcm_base, p, 0U);
}

/**
 * The best pcycles of several passes at one tile count.
 *
 * @param vtcm_base The base of the VTCM region
 * @param p The VTCM layout
 * @param k_tiles The multiplies of one column tile
 * @param f16 True for the f16 mode, false for the int8 mode
 * @return The smallest pcycle count of N_REPS passes
 */
static uint64_t best_of(uint8_t * vtcm_base, const struct vtcm_plan * p, uint32_t k_tiles, int f16) {
    uint64_t best = (uint64_t) -1;
    if (f16) {
        (void) time_f16_issue(vtcm_base, p, k_tiles);
    } else {
        (void) time_i8_issue(vtcm_base, p, k_tiles);
    }
    for (uint32_t r = 0; r < N_REPS; r++) {
        const uint64_t c = f16 ? time_f16_issue(vtcm_base, p, k_tiles)
                               : time_i8_issue(vtcm_base, p, k_tiles);
        if (c < best) {
            best = c;
        }
    }
    return best;
}

/**
 * The best pcycles of several clear-and-read passes.
 *
 * @param vtcm_base The base of the VTCM region
 * @param p The VTCM layout
 * @param f16 True for the f16 mode, false for the int8 mode
 * @return The smallest pcycle count of N_REPS passes
 */
static uint64_t best_read_of(uint8_t * vtcm_base, const struct vtcm_plan * p, int f16) {
    uint64_t best = (uint64_t) -1;
    (void) time_read_only(vtcm_base, p, f16);
    for (uint32_t r = 0; r < N_REPS; r++) {
        const uint64_t c = time_read_only(vtcm_base, p, f16);
        if (c < best) {
            best = c;
        }
    }
    return best;
}

int main(void) {
    uint8_t * vtcm_base     = NULL;
    uint32_t  vtcm_size     = 0;
    uint32_t  hmx_fp16_rate = 0;
    int       res           = AEE_SUCCESS;
    int       locked        = 0;

    HEXKL_TEST_INFO_PRINT("hmx_rate start");

    res = hexkl_micro_hw_init(&vtcm_base, &vtcm_size, &hmx_fp16_rate);
    if (res != AEE_SUCCESS) {
        HEXKL_TEST_ERROR_PRINT("hw_init failed 0x%x", res);
        goto done;
    }
    HEXKL_TEST_INFO_PRINT("vtcm_size %u bytes", (unsigned) vtcm_size);
    HEXKL_TEST_INFO_PRINT("hmx_fp16_rate %u  (0 means the f16 mode is absent)", (unsigned) hmx_fp16_rate);

    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        HEXKL_TEST_ERROR_PRINT("hmx_lock failed 0x%x", res);
        goto done;
    }
    locked = 1;

    const struct vtcm_plan p = plan_vtcm(vtcm_size, GDN_K_TILES, HEXKL_HMX_ACTIVATION_ALIGNMENT);
    HEXKL_TEST_INFO_PRINT("layout: weight 0x%x result 0x%x config 0x%x",
                          (unsigned) p.weight, (unsigned) p.result, (unsigned) p.config);

    // The check runs first, because it writes the same operand region that the timed
    // passes fill with a pattern afterwards.
    res = check_i8_against_reference(vtcm_base, &p);
    if (res != AEE_SUCCESS) {
        HEXKL_TEST_ERROR_PRINT("the int8 mode disagrees with the reference, 0x%x", res);
        goto done;
    }
    HEXKL_TEST_INFO_PRINT("int8 check: 64x32 by 32x32 agrees with the scalar reference");

    // The tiles hold whatever the region holds. The rate does not depend on the
    // values, because the check above covers the arithmetic.
    for (uint32_t i = 0; i < GDN_K_TILES * HEXKL_HMX_ACTIVATION_ALIGNMENT; i++) {
        vtcm_base[i] = (uint8_t) (i * 7U + 1U);
    }
    for (uint32_t i = 0; i < 4096U; i++) {
        vtcm_base[p.weight + i] = (uint8_t) (i * 13U + 3U);
    }

    // Two tile counts. The difference of the two costs divides by the difference of
    // the issue counts, thus every fixed cost of a pass cancels and the result is the
    // marginal cost of one multiply.
    static const uint32_t K_LO = 10U;
    static const uint32_t K_HI = 80U;

    hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, p.config);
    const uint64_t f16_lo   = best_of(vtcm_base, &p, K_LO, 1);
    const uint64_t f16_hi   = best_of(vtcm_base, &p, K_HI, 1);
    const uint64_t f16_read = best_read_of(vtcm_base, &p, 1);

    hexkl_micro_hmx_setup_acc_read_int32(vtcm_base, p.config);
    const uint64_t i8_lo   = best_of(vtcm_base, &p, K_LO, 0);
    const uint64_t i8_hi   = best_of(vtcm_base, &p, K_HI, 0);
    const uint64_t i8_read = best_read_of(vtcm_base, &p, 0);

    const uint64_t d_issues = (uint64_t) N_COL_TILES * (K_HI - K_LO);
    // Scale by 1000 to print a fraction without a floating point format.
    const uint64_t f16_marginal = (f16_hi - f16_lo) * 1000U / d_issues;
    const uint64_t i8_marginal  = (i8_hi  - i8_lo)  * 1000U / d_issues;

    HEXKL_TEST_INFO_PRINT("f16  k=%u %llu pcyc, k=%u %llu pcyc", (unsigned) K_LO,
                          (unsigned long long) f16_lo, (unsigned) K_HI,
                          (unsigned long long) f16_hi);
    HEXKL_TEST_INFO_PRINT("int8 k=%u %llu pcyc, k=%u %llu pcyc", (unsigned) K_LO,
                          (unsigned long long) i8_lo, (unsigned) K_HI,
                          (unsigned long long) i8_hi);
    HEXKL_TEST_INFO_PRINT("marginal cycles per multiply: f16 %llu.%03llu  int8 %llu.%03llu",
                          (unsigned long long) (f16_marginal / 1000U),
                          (unsigned long long) (f16_marginal % 1000U),
                          (unsigned long long) (i8_marginal / 1000U),
                          (unsigned long long) (i8_marginal % 1000U));

    // One pass holds N_COL_TILES clear-and-read pairs, thus divide to get one read.
    const uint64_t f16_read_one = f16_read / N_COL_TILES;
    const uint64_t i8_read_one  = i8_read / N_COL_TILES;
    HEXKL_TEST_INFO_PRINT("accumulator read, one tile: f16 %llu pcyc  int8 %llu pcyc",
                          (unsigned long long) f16_read_one,
                          (unsigned long long) i8_read_one);

    if (f16_marginal > 0U && i8_marginal > 0U) {
        // f16 moves 32768 multiply-accumulate operations, int8 moves 65536.
        const uint64_t f16_mac_per_cyc = 32768U * 1000U / f16_marginal;
        const uint64_t i8_mac_per_cyc  = 65536U * 1000U / i8_marginal;
        HEXKL_TEST_INFO_PRINT("marginal MAC per cycle: f16 %llu  int8 %llu",
                              (unsigned long long) f16_mac_per_cyc,
                              (unsigned long long) i8_mac_per_cyc);
        HEXKL_TEST_INFO_PRINT("int8 throughput against f16: %llu.%02llux",
                              (unsigned long long) (i8_mac_per_cyc / f16_mac_per_cyc),
                              (unsigned long long) ((i8_mac_per_cyc * 100U / f16_mac_per_cyc) % 100U));

        // The break-even point. One int8 tile covers 64 activation rows where one f16
        // tile covers 32, thus k f16 tiles of one 64-row output need 2k f16 issues
        // against k int8 issues. The int8 path wins when
        //     k * i8_marginal + i8_read  <  2k * f16_marginal + f16_read
        // Scale cancels because both marginals carry the same factor of 1000.
        if (2U * f16_marginal > i8_marginal) {
            const uint64_t gain_per_tile = 2U * f16_marginal - i8_marginal;
            const uint64_t extra_read    = (i8_read_one > f16_read_one)
                                               ? (i8_read_one - f16_read_one) * 1000U
                                               : 0U;
            const uint64_t break_even    = (extra_read + gain_per_tile - 1U) / gain_per_tile;
            HEXKL_TEST_INFO_PRINT("break-even: one int8 accumulator read must cover %llu k tiles",
                                  (unsigned long long) break_even);
            HEXKL_TEST_INFO_PRINT("Q8_0 carries one scale for each k tile, thus it covers 1.");
        }
    }

done:
    if (locked) {
        const int unlock_res = hexkl_micro_hmx_unlock();
        if (res == AEE_SUCCESS) {
            res = unlock_res;
        }
    }
    HEXKL_TEST_PRINT_PASS_STATUS(res);
    return res;
}
