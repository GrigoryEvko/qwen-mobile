// Stub of the Hexagon SDK header dspqueue.h for the x86 fuzz harness.
// The declarations follow the SDK 6.6.0.0. The fake DSP of the harness
// (common/fake_dsp.cpp) gives the implementation.
#pragma once

#include <stdint.h>
#include <stdlib.h>

#include "AEEStdDef.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DSPQUEUE_TIMEOUT_NONE 0xffffffff

enum dspqueue_buffer_flags {
    DSPQUEUE_BUFFER_FLAG_REF                  = 0x00000004,
    DSPQUEUE_BUFFER_FLAG_DEREF                = 0x00000008,
    DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER         = 0x00000010,
    DSPQUEUE_BUFFER_FLAG_INVALIDATE_SENDER    = 0x00000020,
    DSPQUEUE_BUFFER_FLAG_FLUSH_RECIPIENT      = 0x00000040,
    DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT = 0x00000080,
};

struct dspqueue;
typedef struct dspqueue * dspqueue_t;

struct dspqueue_buffer {
    uint32_t fd;
    uint32_t size;
    uint32_t offset;
    uint32_t flags;
    union {
        void *   ptr;
        uint64_t address;
    };
};

typedef void (*dspqueue_callback_t)(dspqueue_t queue, AEEResult error, void * context);

AEEResult dspqueue_create(int domain, uint32_t flags, uint32_t req_queue_size, uint32_t resp_queue_size,
                          dspqueue_callback_t packet_callback, dspqueue_callback_t error_callback,
                          void * callback_context, dspqueue_t * queue);
AEEResult dspqueue_close(dspqueue_t queue);
AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id);
AEEResult dspqueue_write(dspqueue_t queue, uint32_t flags, uint32_t num_buffers, struct dspqueue_buffer * buffers,
                         uint32_t message_length, const uint8_t * message, uint32_t timeout_us);
AEEResult dspqueue_read(dspqueue_t queue, uint32_t * flags, uint32_t max_buffers, uint32_t * num_buffers,
                        struct dspqueue_buffer * buffers, uint32_t max_message_length, uint32_t * message_length,
                        uint8_t * message, uint32_t timeout_us);

#ifdef __cplusplus
}
#endif
