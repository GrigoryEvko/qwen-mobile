// Stub of htp/hex-utils.h for the x86 build of htp/htp-tensor.c in fuzz_dirty.
// The constants are those of the DSP. fuzz_dirty.cpp gives hex_l2flush, which
// records the flushed lines, with the line rounding of the real function.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "qurt.h"
#include "hex-fastdiv.h"
#include "hex-common.h"

#define HEX_L2_LINE_SIZE           128
#define HEX_L2_BLOCK_SIZE          (HEX_L2_LINE_SIZE * 4)
#define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

#ifdef __cplusplus
extern "C" {
#endif

void hex_l2flush(void * addr, size_t size);

#ifdef __cplusplus
}
#endif

static inline void hex_l2fetch_block(const void * addr, size_t size) {
    (void) addr;
    (void) size;
}

static inline uint64_t hex_get_cycles(void) {
    return 0;
}

static inline void hex_pause(void) {}
