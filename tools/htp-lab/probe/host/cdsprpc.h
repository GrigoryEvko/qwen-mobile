// The run-time loader of the FastRPC library of the compute DSP (libcdsprpc.so).
//
// The NDK has no libcdsprpc.so, and each phone has its own copy in /vendor. Thus the program loads
// it with dlopen, as ggml-hexagon/htp-drv.cpp of llama.cpp does, and cdsprpc.c defines the
// FastRPC functions that the qaic stub calls. Each definition calls the function of the library.
#ifndef ISAPROBE_CDSPRPC_H
#define ISAPROBE_CDSPRPC_H

// Load libcdsprpc.so and find its functions. Returns 0, or -1 after an error message on stderr.
// Call it one time before a FastRPC call.
int cdsprpc_load(void);

#endif
