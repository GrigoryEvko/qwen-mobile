// The run-time loader of the FastRPC library of the compute DSP (libcdsprpc.so) for ddrbw.
//
// The NDK has no libcdsprpc.so, and each phone has its own copy in /vendor. Thus the program loads
// it with dlopen, as ggml-hexagon/htp-drv.cpp of llama.cpp does. rpc.c defines the FastRPC, rpcmem,
// fastrpc_mmap and dspqueue functions that the qaic stub and ddrbw.c call, and each definition
// calls the function of the library.
#ifndef DDRBW_RPC_H
#define DDRBW_RPC_H

// Load libcdsprpc.so and find its functions. Returns 0, or -1 after an error message on stderr.
// Call it one time before the first FastRPC call. A function that the library does not have
// gives AEE_EUNSUPPORTED (or NULL for an allocation) when the program calls it.
int rpc_load(void);

#endif
