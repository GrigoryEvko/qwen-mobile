// The DMA shim of the kernel lab: the interface of htp/dma-queue.h over ordinary loads and stores.
//
// Include this header BEFORE the kernel file. It defines HTP_DMA_H, thus the kernel takes this
// shim in place of the real queue. The standalone runtime of the simulator gives no user DMA
// engine: a real descriptor raises "No Access". Thus a push of this shim copies the rows at once
// and records the descriptor, and a pop returns the descriptors in the order of the pushes. That
// order is the contract that every kernel of the backend depends on.
//
// What the shim finds and what it cannot find:
//   - It finds a wrong geometry of a descriptor: a wrong stride or row size puts wrong bytes in
//     the destination, and the output of the kernel changes.
//   - It finds a wrong order of the pops, because the pop order is the push order.
//   - It finds a push to a second ring of one hardware thread. Refer to lab_dma_note_push.
//   - It cannot find a timing defect. The copy is complete at the push, thus a kernel that reads
//     a destination before its pop still reads the right bytes here and the wrong bytes on the
//     chip.
//
// A target that needs a deeper queue defines LAB_DMA_CAPACITY before this include. The value must
// be a power of two.
#ifndef LAB_DMA_H
#define LAB_DMA_H

#define HTP_DMA_H

#include "lab.h"

#include <stdio.h>
#include <string.h>

#include <hexagon_standalone.h>

#ifndef LAB_DMA_CAPACITY
#define LAB_DMA_CAPACITY 256
#endif

// The hardware threads of the core. The core of the phone has 8.
#define LAB_DMA_THREADS 32

// The lines of the descriptor cache of a mask. The value of htp/dma-queue.h.
#define DMA_CACHE_MAX_SIZE 128

typedef struct {
    void *       dst;
    const void * src;
} dma_ptr;

typedef struct dma_queue_s {
    dma_ptr  ptr[LAB_DMA_CAPACITY];
    uint32_t push_idx;
    uint32_t pop_idx;
} dma_queue;
typedef dma_queue * dma_queue_t;

// The transfers that the queue holds. Each push of the shim is complete at once, thus the value is
// the number of descriptors that no pop took.
static inline uint32_t dma_queue_depth(dma_queue * q) {
    return q->push_idx - q->pop_idx;
}

// ---- one DMA ring for each hardware thread ----

// The last queue that each hardware thread pushed to. A thread writes its own slot only, thus the
// array needs no lock.
static dma_queue * lab_dma_last[LAB_DMA_THREADS];
static uint32_t    lab_dma_ring_violations;
static uint64_t    lab_dma_push_count[LAB_DMA_THREADS];
static uint64_t    lab_dma_row_count[LAB_DMA_THREADS];

// 1 (the preset value): a push copies the rows, thus a value check sees the bytes that the phone
// sees. 0: a push records the descriptor and copies nothing. A timing run of a kernel whose real
// transfers run on the DMA engine sets 0, because the copy of the shim runs on the vector unit and
// the phone does not pay it there. A value check needs the copy.
static int lab_dma_copy = 1;

// Prints the first push to a second ring and counts every one of them. The function stays out of
// the inline push, thus the push pays one compare and no call in the usual case.
static void lab_dma_ring_fault(unsigned int tid, dma_queue * pushed, dma_queue * busy) {
    if (__atomic_fetch_add(&lab_dma_ring_violations, 1, __ATOMIC_RELAXED) == 0) {
        printf("lab: dma: thread %u pushes to queue %p while queue %p has %u transfers in flight\n", tid,
               (void *) pushed, (void *) busy, dma_queue_depth(busy));
    }
}

// The DMA engine of a hardware thread follows one descriptor chain: a push links its descriptor to
// the tail of its own queue (dmlink of htp/dma-queue.h). On the chip a push to a queue while a
// different queue of the same thread has transfers in flight links to a chain that the engine does
// not read. The transfer never starts and its pop waits forever: the op hangs. This shim copies at
// the push, thus it cannot hang. It counts the push instead.
//
// The rule that a kernel must obey: the op thread uses ctx->dma[0] only, and a worker i uses
// ctx->dma[i] only. Order the pushes so that the pops stay in the order of the pushes.
// Complexity O(1).
static inline void lab_dma_note_push(dma_queue * q, size_t nrows) {
    const unsigned int tid  = (unsigned int) thread_get_tnum() & (LAB_DMA_THREADS - 1);
    dma_queue * const  last = lab_dma_last[tid];
    if (last != q) {
        if (last != NULL && dma_queue_depth(last) > 0) {
            lab_dma_ring_fault(tid, q, last);
        }
        lab_dma_last[tid] = q;
    }
    lab_dma_push_count[tid]++;
    lab_dma_row_count[tid] += nrows;
}

// The pushes of every thread. Complexity O(LAB_DMA_THREADS).
static inline uint64_t lab_dma_pushes(void) {
    uint64_t n = 0;
    for (int i = 0; i < LAB_DMA_THREADS; i++) {
        n += lab_dma_push_count[i];
    }
    return n;
}

// The rows that the pushes of every thread moved. Complexity O(LAB_DMA_THREADS).
static inline uint64_t lab_dma_rows(void) {
    uint64_t n = 0;
    for (int i = 0; i < LAB_DMA_THREADS; i++) {
        n += lab_dma_row_count[i];
    }
    return n;
}

// Clears the counters and the ring state of every thread. Call it before a case.
static inline void lab_dma_reset(void) {
    for (int i = 0; i < LAB_DMA_THREADS; i++) {
        lab_dma_last[i]       = NULL;
        lab_dma_push_count[i] = 0;
        lab_dma_row_count[i]  = 0;
    }
    lab_dma_ring_violations = 0;
}

// Prints the DMA counters of the run and records the limit of the shim. Returns the number of
// pushes to a second ring of one thread, thus a caller can fail the run.
static inline uint32_t lab_dma_report(const char * target) {
    lab_report(target, "dma_pushes", (double) lab_dma_pushes(), "");
    lab_report(target, "dma_rows", (double) lab_dma_rows(), "");
    lab_report(target, "dma_ring_violations", (double) lab_dma_ring_violations, "");
    lab_limit("The DMA shim copies at the push, thus a kernel that reads a destination before its "
              "pop reads the right bytes here and the wrong bytes on the chip.");
    if (!lab_dma_copy) {
        lab_limit("lab_dma_copy is 0, thus the DMA shim moved no byte. The values of this run come "
                  "from the buffers that the program filled, and not from a transfer.");
    }
    return lab_dma_ring_violations;
}

// ---- the interface of htp/dma-queue.h ----

static inline dma_ptr dma_make_ptr(void * dst, const void * src) {
    dma_ptr p = { dst, src };
    return p;
}

// Copies nrows rows of row_size bytes at once, then records the descriptor for the pop. nrows of 0
// is the dummy transfer that a hit of the descriptor cache pushes: it records only.
// Complexity O(nrows * row_size).
static inline bool dma_queue_push(dma_queue * q, dma_ptr p, size_t dst_stride, size_t src_stride,
                                  size_t row_size, size_t nrows) {
    if (dma_queue_depth(q) >= LAB_DMA_CAPACITY) {
        printf("lab: error: the DMA shim queue %p is full at %d descriptors. Define LAB_DMA_CAPACITY "
               "larger before the include of lab-dma.h.\n", (void *) q, LAB_DMA_CAPACITY);
        return false;
    }
    lab_dma_note_push(q, nrows);
    if (lab_dma_copy) {
        for (size_t r = 0; r < nrows; r++) {
            memcpy((uint8_t *) p.dst + r * dst_stride, (const uint8_t *) p.src + r * src_stride, row_size);
        }
    }
    q->ptr[q->push_idx & (LAB_DMA_CAPACITY - 1)] = p;
    q->push_idx++;
    return true;
}

// Returns the descriptor of the oldest push that no pop took, or two null pointers when the queue
// is empty. Complexity O(1).
static inline dma_ptr dma_queue_pop(dma_queue * q) {
    dma_ptr p = { NULL, NULL };
    if (q->pop_idx == q->push_idx) {
        return p;
    }
    p = q->ptr[q->pop_idx & (LAB_DMA_CAPACITY - 1)];
    q->pop_idx++;
    return p;
}

// Each push of the shim is complete at once, thus a pop that does not wait is an ordinary pop.
static inline dma_ptr dma_queue_pop_nowait(dma_queue * q) {
    return dma_queue_pop(q);
}

static inline bool dma_queue_empty(dma_queue * q) {
    return q->push_idx == q->pop_idx;
}

static inline uint32_t dma_queue_capacity(dma_queue * q) {
    (void) q;
    return LAB_DMA_CAPACITY;
}

// Takes every descriptor that no pop took. A push of the shim is complete at once, thus the queue
// needs no wait and a drop of the indexes is equal to a loop of pops.
static inline void dma_queue_flush(dma_queue * q) {
    q->pop_idx = q->push_idx;
}

static inline bool dma_queue_push_single_1d(dma_queue * q, dma_ptr p, size_t size) {
    return dma_queue_push(q, p, size, size, size, 1);
}

static inline bool dma_queue_push_single_2d(dma_queue * q, dma_ptr p, size_t dst_stride, size_t src_stride,
                                            size_t row_size, size_t nrows) {
    return dma_queue_push(q, p, dst_stride, src_stride, row_size, nrows);
}

static inline bool dma_queue_push_ddr_to_vtcm(dma_queue * q, dma_ptr p, size_t dst_row_size, size_t src_row_size,
                                              size_t nrows) {
    return dma_queue_push(q, p, dst_row_size, src_row_size, src_row_size, nrows);
}

static inline bool dma_queue_push_vtcm_to_ddr(dma_queue * q, dma_ptr p, size_t dst_row_size, size_t src_row_size,
                                              size_t nrows) {
    return dma_queue_push(q, p, dst_row_size, src_row_size, dst_row_size, nrows);
}

// The shim holds no VTCM range, thus it reports every pointer as DDR. A kernel that selects a path
// by this answer takes the DDR path here.
static inline bool dma_is_vtcm(const dma_queue * q, const void * ptr) {
    (void) q;
    (void) ptr;
    return false;
}

static inline size_t dma_queue_sizeof(size_t capacity) {
    (void) capacity;
    return sizeof(dma_queue);
}

static inline size_t dma_queue_alignof(void) {
    return 128;
}

// Makes a queue in the buffer of the caller. The buffer must hold dma_queue_sizeof bytes.
static inline dma_queue_t dma_queue_init(void * ptr, size_t capacity, uintptr_t vtcm_base, size_t vtcm_size,
                                         void * trace) {
    (void) capacity;
    (void) vtcm_base;
    (void) vtcm_size;
    (void) trace;
    memset(ptr, 0, sizeof(dma_queue));
    return (dma_queue_t) ptr;
}

static inline void dma_queue_free(dma_queue_t q) {
    (void) q;
}

// ---- the descriptor cache of a mask ----

// The line cache of htp/dma-queue.h: a hit refreshes the age and pushes a dummy transfer, and a
// miss takes the oldest line and pushes a real one.
typedef struct {
    uint8_t * base;
    uint32_t  line_size;
    uint32_t  capacity;
    uint32_t  src[DMA_CACHE_MAX_SIZE];
    uint16_t  age[DMA_CACHE_MAX_SIZE];
} dma_cache;

static inline void dma_cache_init(dma_cache * c, uint8_t * base, uint32_t line_size, uint32_t capacity) {
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->base      = base;
    c->line_size = line_size;
    for (unsigned i = 0; i < c->capacity; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

// Complexity O(capacity).
static inline bool dma_cache_push(dma_queue * q, dma_cache * c, const uint8_t * src, uint32_t dst_stride,
                                  uint32_t src_stride, uint32_t row_size, uint32_t nrows) {
    uint32_t  o_idx = 0;
    uint16_t  o_age = 0;
    uint8_t * dst   = 0;
    for (unsigned i = 0; i < c->capacity; i++) {
        if (c->src[i] == (uint32_t) (uintptr_t) src) {
            c->age[i] = 0;
            dst = c->base + (i * c->line_size);
            nrows = 0;
        } else {
            c->age[i]++;
            if (c->age[i] > o_age) {
                o_age = c->age[i];
                o_idx = i;
            }
        }
    }
    if (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = (uint32_t) (uintptr_t) src;
        dst = c->base + o_idx * c->line_size;
    }
    return dma_queue_push(q, dma_make_ptr(dst, src), dst_stride, src_stride, row_size, nrows);
}

#endif
