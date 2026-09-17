// Stubs of the DSP runtime services that the kernels call at link time.
//
// The stubs change the timing as follows:
//   work_queue_run_async: the real work queue wakes sleeping QuRT threads through a futex. The
//   stub creates the threads on the standalone runtime for each call, thus each call pays the
//   thread creation (some hundred cycles) instead of the wakeup.
#include "lab.h"

#include <stdio.h>

#include "work-queue.h"

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    (void) q;
    lab_run_threads(func, data, n);
    return true;
}

size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t stack_size) {
    (void) n_threads;
    (void) capacity;
    (void) stack_size;
    return 0;
}

size_t work_queue_alignof(void) {
    return 128;
}

work_queue_t work_queue_init(void * ptr, uint32_t n_threads, uint32_t capacity, uint32_t stack_size) {
    (void) n_threads;
    (void) capacity;
    (void) stack_size;
    return (work_queue_t) ptr;
}

void work_queue_free(work_queue_t q) {
    (void) q;
}

void work_queue_wakeup(work_queue_t q) {
    (void) q;
}

void work_queue_suspend(work_queue_t q) {
    (void) q;
}

int qurt_futex_wait(void * addr, int value) {
    (void) addr;
    (void) value;
    return 0;
}

int qurt_futex_wake(void * addr, int count) {
    (void) addr;
    (void) count;
    return 0;
}
