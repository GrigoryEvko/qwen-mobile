// The common runtime of the kernel lab: cycle counter, VTCM, random data, comparison, threads.
//
// The lab programs run on the standalone runtime of the Hexagon simulator (no QuRT). The runtime
// starts main() in monitor mode, thus the program reads the processor cycle counter (PCYCLE) and
// sets the HMX enable bit of the thread itself.
#ifndef LAB_H
#define LAB_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Prevents the compiler from a move of a kernel call across the measurement points
#define LAB_BARRIER() __asm__ volatile("" ::: "memory")

typedef void (*lab_thread_fn)(unsigned int nth, unsigned int ith, void * data);

// Reads the processor cycle counter (PCYCLE, monitor mode)
static inline uint64_t lab_cycles(void) {
    uint64_t c;
    __asm__ volatile("%0 = s31:30" : "=r"(c));
    return c;
}

// Maps the VTCM, enables the HMX for the main thread, and seeds the random generator
void lab_init(void);

// Returns the VTCM base and size that the core configuration reports
uint8_t * lab_vtcm_base(void);
size_t    lab_vtcm_size(void);

// Bump allocator in VTCM. The alignment is a power of two. The function stops the program when
// the VTCM is full.
void * lab_vtcm_alloc(size_t bytes, size_t align);

// Aligned allocation in DDR (the simulator memory). The function stops the program on failure.
void * lab_ddr_alloc(size_t bytes, size_t align);

// Fixed-seed random numbers (xorshift32), the same sequence in each run
uint32_t lab_rand_u32(void);
float    lab_rand_f32(float lo, float hi);
void     lab_fill_f32(float * dst, size_t n, float lo, float hi);
void     lab_fill_u8(uint8_t * dst, size_t n);

// Compares two float arrays. Prints the first mismatches and the maximum errors.
// Returns the number of elements outside abs_tol + rel_tol * |ref|.
size_t lab_compare_f32(const char * what, const float * got, const float * ref, size_t n, float abs_tol, float rel_tol);

// Prints one result line in the form "lab: <target> <key> = <value> <unit>". The report script
// collects these lines.
void lab_report(const char * target, const char * key, double value, const char * unit);

// Runs fn(n, ith, data) on n hardware threads (thread 0 is the caller) and waits for all of them.
// Each worker thread acquires an HVX context first. The maximum is 8 threads.
void lab_run_threads(lab_thread_fn fn, void * data, unsigned int n);

// Enables the HMX (SSR bit XE2) for the calling thread
void lab_hmx_enable(void);

// Parses "--name value" pairs. Returns the value of the option or the default.
long lab_arg_long(int argc, char ** argv, const char * name, long def);

// Converts half floats
float    lab_hf_to_f32(uint16_t h);
uint16_t lab_f32_to_hf(float f);

#endif
