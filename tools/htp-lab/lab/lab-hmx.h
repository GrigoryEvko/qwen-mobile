// The HMX queue shim of the kernel lab: the interface of htp/hmx-queue.h without a second thread.
//
// Include this header BEFORE the kernel file. It defines HMX_QUEUE_H, thus the kernel takes this
// shim in place of the real queue. A push runs the job at once on the thread of the push, and a pop
// returns the descriptor of the matching push.
//
// What the shim finds and what it cannot find:
//   - It finds a wrong tile layout, a wrong lane order and a wrong output buffer, because the job
//     does the real HMX instructions.
//   - It cannot find a defect of the overlap. On the phone the HMX thread runs beside the HVX
//     threads. Here a job ends before the thread of the push continues, thus a kernel that gives a
//     job an output buffer that another thread still reads gives the right values here.
//   - The timing model of the simulator does not retire an HMX instruction. Run a target that uses
//     this shim in the functional mode of the simulator (MODE=functional).
//
// The chunked gated delta net kernel of version 2 reads the ring fields of the real queue
// (desc, idx_write, idx_pop, idx_mask) and needs a job that runs at its wait. That kernel thus
// keeps its own shim in its target.
#ifndef LAB_HMX_H
#define LAB_HMX_H

#define HMX_QUEUE_H

#include "lab.h"

#include <stdio.h>

// The descriptors that the shim records for the pops. A push over this depth without a pop is an
// error of the kernel, not of the shim.
#ifndef LAB_HMX_CAPACITY
#define LAB_HMX_CAPACITY 64
#endif

typedef void (*hmx_queue_func)(void *);

struct hmx_queue_desc {
    hmx_queue_func func;
    void *         data;
};

struct hmx_queue_s {
    struct hmx_queue_desc ring[LAB_HMX_CAPACITY];
    uint32_t              pushed;
    uint32_t              popped;
};

typedef struct hmx_queue_s * hmx_queue_t;

static inline struct hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, void * data) {
    struct hmx_queue_desc d = { func, data };
    return d;
}

// The jobs that a pop did not take
static inline uint32_t hmx_queue_pending(hmx_queue_t q) {
    return q->pushed - q->popped;
}

// Runs the job and records its descriptor. Complexity: the cost of the job.
static inline bool hmx_queue_push(hmx_queue_t q, struct hmx_queue_desc d) {
    if (hmx_queue_pending(q) >= LAB_HMX_CAPACITY) {
        printf("lab: error: the HMX shim queue holds %d jobs that no pop took\n", LAB_HMX_CAPACITY);
        return false;
    }
    q->ring[q->pushed & (LAB_HMX_CAPACITY - 1)] = d;
    q->pushed++;
    d.func(d.data);
    return true;
}

// Returns the descriptor of the oldest push that no pop took. A pop without a job is an error of
// the kernel: the real pop would wait forever. Complexity O(1).
static inline struct hmx_queue_desc hmx_queue_pop(hmx_queue_t q) {
    struct hmx_queue_desc d = { NULL, NULL };
    if (q->popped == q->pushed) {
        printf("lab: error: an HMX queue pop without a job\n");
        return d;
    }
    d = q->ring[q->popped & (LAB_HMX_CAPACITY - 1)];
    q->popped++;
    return d;
}

static inline bool hmx_queue_empty(hmx_queue_t q) {
    return q->pushed == q->popped;
}

static inline uint32_t hmx_queue_depth(hmx_queue_t q) {
    return hmx_queue_pending(q);
}

// Takes every job that no pop took
static inline void hmx_queue_flush(hmx_queue_t q) {
    q->popped = q->pushed;
}

// Clears the queue. Call it before a case, thus the job count of the case starts at 0.
static inline void lab_hmx_reset(hmx_queue_t q) {
    q->pushed = 0;
    q->popped = 0;
}

// Prints the job count of the run and records the limit of the shim. Returns the jobs that no pop
// took, thus a caller can fail the run.
static inline uint32_t lab_hmx_report(const char * target, hmx_queue_t q) {
    lab_report(target, "hmx_jobs", (double) q->pushed, "");
    lab_report(target, "hmx_jobs_not_popped", (double) hmx_queue_pending(q), "");
    lab_limit("The HMX shim runs a job at the push, thus it cannot find a defect of the overlap of "
              "the HMX thread with the HVX threads.");
    return hmx_queue_pending(q);
}

#endif
