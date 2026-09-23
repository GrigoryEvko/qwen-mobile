// Stub of the QuRT headers for the x86 build of htp/htp-tensor.c in fuzz_dirty.
// fuzz_dirty.cpp gives qurt_mem_cache_clean, which records a flush of the whole cache.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t qurt_addr_t;
typedef uint32_t qurt_size_t;
typedef int      qurt_thread_t;

enum { QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL = 5 };
enum { QURT_MEM_DCACHE = 1 };

int qurt_mem_cache_clean(qurt_addr_t addr, qurt_size_t size, int op, int type);

#ifdef __cplusplus
}
#endif
