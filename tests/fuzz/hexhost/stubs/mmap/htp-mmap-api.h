// The interface of fuzz_mmap to the copy of the mapping table of the DSP (prep_op_bufs and its helpers of
// htp/main.c). CMakeLists.txt writes the copy into the build directory (dsp/htp-mmap.c), and the copy
// includes htp-mmap-host.h first and htp-mmap-api.inc last.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct htp_buf_desc;

#ifdef __cplusplus
extern "C" {
#endif

// The VA of the DSP (HAP_mmap2 and HAP_munmap2 of htp_mmap and htp_munmap in main.c). fuzz_mmap.cpp gives
// a fake VA: a first-fit allocator with a capacity. htp_mmap gives NULL when no range is free.
void * htp_mmap(uint32_t fd, uint32_t size);
void   htp_munmap(void * va, uint32_t size);

// The bytes of the context of the copy (the table and max_vmem)
size_t hexmmap_ctx_size(void);

// The slot count of the table (HTP_MAX_MMAPS)
uint32_t hexmmap_slots(void);

// Clears the table, as htp_iface_start does with the context block, and sets max_vmem
void hexmmap_init(void * ctx, uint64_t max_vmem);

// prep_op_bufs of main.c for one batch: seq is the sequence number of the batch, lru the flag
// HTP_OPBATCH_MMAP_LRU of the request
void hexmmap_prep(void * ctx, struct htp_buf_desc * bufs, uint32_t n_bufs, uint32_t seq, bool lru);

// Slot i of the table: its fd, its size (0 for a free slot) and its base
void hexmmap_slot(const void * ctx, uint32_t i, uint32_t * fd, uint64_t * size, uint64_t * base);

#ifdef __cplusplus
}
#endif
