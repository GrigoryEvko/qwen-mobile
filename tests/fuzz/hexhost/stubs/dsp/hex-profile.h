// Stub of htp/hex-profile.h for the x86 build of htp/htp-tensor.c: the trace events do nothing.
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "htp-ops.h"

struct htp_thread_trace {
    uint32_t                count;
    uint32_t                max_events;
    struct htp_trace_desc * events;
};

static inline void htp_trace_event_start(struct htp_thread_trace * tr, uint16_t id, uint16_t info) {
    (void) tr;
    (void) id;
    (void) info;
}

static inline void htp_trace_event_stop(struct htp_thread_trace * tr, uint16_t id, uint16_t info) {
    (void) tr;
    (void) id;
    (void) info;
}
