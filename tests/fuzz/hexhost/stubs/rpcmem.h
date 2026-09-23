// Stub of the Hexagon SDK header rpcmem.h for the x86 fuzz harness.
// The fake DSP of the harness (common/fake_dsp.cpp) gives the implementation.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS  1

void * rpcmem_alloc(int heapid, uint32_t flags, int size);
void * rpcmem_alloc2(int heapid, uint32_t flags, size_t size);
void   rpcmem_free(void * po);
int    rpcmem_to_fd(void * po);

#ifdef __cplusplus
}
#endif
