// Stub of htp/work-queue.h for the x86 build of htp/htp-tensor.c: work_queue_run
// calls the function for each thread index, one after the other.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef void (*work_queue_func_t)(unsigned int n, unsigned int i, void *);

struct work_queue_s;
typedef struct work_queue_s * work_queue_t;

static inline bool work_queue_run(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    (void) q;
    for (unsigned int i = 0; i < (n ? n : 1); i++) {
        func(n, i, data);
    }
    return true;
}
