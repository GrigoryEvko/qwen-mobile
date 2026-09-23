// Stub of htp/htp-ctx.h for the x86 build of htp/htp-tensor.c in fuzz_dirty. It
// gives the fields of the DSP context that htp-tensor.c reads, with the same
// names and the same limits as the real header.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "hex-fastdiv.h"
#include "hex-profile.h"
#include "htp-ops.h"
#include "work-queue.h"

#define HTP_MAX_DIRTY_RANGES 32

struct htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

struct htp_context {
    struct htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_queue_t            work_queue;
    uint32_t                n_threads;
    struct fastdiv_values   n_threads_div;
    struct htp_dirty_range  dirty_ranges[HTP_MAX_DIRTY_RANGES];
    size_t                  footprint;
};
