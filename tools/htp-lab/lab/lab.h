// The common runtime of the kernel lab: cycle counter, VTCM, random data, comparison, threads,
// error metrics, output hashes, guard regions, and the limits of a run.
//
// The lab programs run on the standalone runtime of the Hexagon simulator (no QuRT). The runtime
// starts main() in monitor mode, thus the program reads the processor cycle counter (PCYCLE) and
// sets the HMX enable bit of the thread itself.
//
// A green lab result is necessary and not sufficient. lab_init records the limits of the build and
// of the simulator, and it prints them at the exit of the program. Refer to lab_limit.
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

// Maps the VTCM, enables the HMX for the main thread, seeds the random generator, and records the
// limits of the build and of the simulator. At the exit of the program it does the check of the
// options (refer to lab_args_done) and prints the limits.
void lab_init(void);

// Returns the VTCM base and size that the core configuration reports
uint8_t * lab_vtcm_base(void);
size_t    lab_vtcm_size(void);

// Bump allocator in VTCM. The alignment is a power of two. The function stops the program when
// the VTCM is full.
void * lab_vtcm_alloc(size_t bytes, size_t align);

// Aligned allocation in DDR (the simulator memory). The function stops the program on failure.
void * lab_ddr_alloc(size_t bytes, size_t align);

// ---- guard regions ----

// The bytes of the guard region before and after a guarded allocation
#define LAB_GUARD_BYTES 256

// Aligned allocation in DDR with a guard region of LAB_GUARD_BYTES bytes before and after the
// buffer. The guard bytes hold a fixed pattern. Use lab_guard_check to find a write past the
// buffer. The returned pointer has the requested alignment, thus a kernel sees an ordinary
// buffer. The function stops the program on failure.
void * lab_ddr_alloc_guarded(size_t bytes, size_t align);

// Counts the guard bytes of one guarded buffer that do not hold the pattern, and prints the first
// of them. The pointer is the one that lab_ddr_alloc_guarded returned. The function stops the
// program when the pointer is not a guarded buffer. Returns 0 when no byte changed.
// Complexity O(LAB_GUARD_BYTES).
size_t lab_guard_check(const char * what, const void * buffer);

// Does a check of the guard region of every guarded buffer. Returns the number of buffers that
// hold a changed guard byte. Complexity O(buffers).
size_t lab_guard_check_all(void);

// ---- random data ----

// Fixed-seed random numbers (xorshift32), the same sequence in each run
uint32_t lab_rand_u32(void);
float    lab_rand_f32(float lo, float hi);
void     lab_fill_f32(float * dst, size_t n, float lo, float hi);
void     lab_fill_u8(uint8_t * dst, size_t n);

// Writes n values in [lo, hi] and puts one special value at each period-th index: +0, -0, a
// subnormal, 1e30, +Inf, -Inf, a NaN, and a value in the band [14.49, 15.73]. That band gives
// +Inf when an int16 activation path drops the sign of the exponent. A period of 0 writes no
// special value. Uniform data reaches none of these values, thus a kernel that only sees uniform
// data hides a defect of its special cases.
void lab_fill_f32_specials(float * dst, size_t n, float lo, float hi, size_t period);

// Writes n values that walk the octaves of the f16 range, from 2^-14 to 2^15, with the two signs.
// A sum of four products of uniform values reaches a narrow band of magnitudes only, thus a
// sweep of the octaves is the way to reach every magnitude that an f16 holds.
void lab_fill_f32_octaves(float * dst, size_t n);

// ---- the element-count sweep of a row ----

// The number of element counts of the canonical sweep
#define LAB_TAIL_SWEEP_COUNT 19

// Returns the element count with the given index of the canonical sweep, or 0 when the index is
// past the end. The sweep holds 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128,
// 129 and n_max. 16 and 32 are the row lengths that the gates of the 2B and of the 4B give, thus
// the model runs the tail block of a kernel and no other block. A count that the sweep gives more
// than one time comes back one time only, thus the caller sees each count one time.
size_t lab_tail_sweep(size_t index, size_t n_max);

// ---- the F16 range ----

// The largest finite value of an f16
#define LAB_F16_MAX 65504.0f

// Counts the values of the row whose magnitude is more than LAB_F16_MAX. An f16 store of such a
// value gives an infinity. The residual stream of the Qwen3.5-4B vision encoder reaches 91410,
// thus a candidate that stores an activation as f16 needs this count. Complexity O(n).
size_t lab_f16_overflow_count(const float * v, size_t n);

// ---- comparison ----

// Compares two float arrays. Prints the first mismatches and the maximum errors.
// Returns the number of elements outside abs_tol + rel_tol * |ref|.
size_t lab_compare_f32(const char * what, const float * got, const float * ref, size_t n, float abs_tol, float rel_tol);

// The error of a result against a reference. lab_metric_add takes one pair at a time, thus a
// caller needs no buffer for the reference. The relative error divides by |ref| and by nothing
// else: a denominator such as |ref| + 1e-6 floors the relative error of a small reference, and
// that floor hid a kernel that wrote nothing at all.
typedef struct {
    const char * what;          // the name of the value that the metric holds
    size_t       n;             // the pairs of the metric
    size_t       n_zero_ref;    // the pairs whose reference is exactly 0
    size_t       n_nonfinite;   // the results that are not finite
    double       sum_sq_err;    // the sum of (got - ref)^2
    double       sum_sq_ref;    // the sum of ref^2
    double       worst_abs;     // the worst |got - ref|
    double       worst_rel;     // the worst |got - ref| / |ref| over the pairs whose ref is not 0
    size_t       abs_index;     // the index of the worst absolute error
    double       abs_got;       // the result of the worst absolute error
    double       abs_ref;       // the reference of the worst absolute error
    double       abs_input;     // the input of the worst absolute error
    size_t       rel_index;     // the index of the worst relative error
    double       rel_got;       // the result of the worst relative error
    double       rel_ref;       // the reference of the worst relative error
    double       rel_input;     // the input of the worst relative error
} lab_metric;

// Clears the metric and gives it its name
void lab_metric_start(lab_metric * m, const char * what);

// Adds one pair. The index and the input go into the report of the worst case, thus a reader gets
// the input that caused the worst error and not the error alone. Give 0 for the input when the
// value has no single input. Complexity O(1).
void lab_metric_add(lab_metric * m, double got, double ref, size_t index, double input);

// Prints the metric: the NMSE, the worst absolute error and its input, the worst relative error
// and its input, the pairs whose reference is 0, and the results that are not finite. Returns the
// number of pairs outside abs_tol + rel_tol * |ref|, and a result that is not finite always
// counts. Complexity O(1).
size_t lab_metric_report(const lab_metric * m, const char * target, double abs_tol, double rel_tol);

// ---- output hashes ----

// The offset basis of the FNV-1a hash
#define LAB_FNV1A_BASIS 0xCBF29CE484222325ull

// The FNV-1a hash of n bytes, offset basis LAB_FNV1A_BASIS, prime 0x100000001B3. Two kernel trees
// give the same bits when they give the same hash for the same inputs. Complexity O(n).
uint64_t lab_fnv1a(const void * p, size_t n);

// Adds n bytes to a hash that a previous call gave, thus one hash covers a list of buffers. Start
// the chain with LAB_FNV1A_BASIS. Complexity O(n).
uint64_t lab_fnv1a_update(uint64_t h, const void * p, size_t n);

// ---- the result lines ----

// Prints one result line in the form "lab: <target> <key> = <value> <unit>". The report script
// collects these lines.
void lab_report(const char * target, const char * key, double value, const char * unit);

// ---- the limits of a run ----

// Records a limit of this run that a green result does not cover. The text goes into the limits
// block that the program prints at its exit. A limit that the framework cannot prevent must be
// visible, thus a reader cannot take a green result for a proof. The text must stay valid for the
// life of the program, thus give a string literal.
void lab_limit(const char * text);

// Prints the limits block. The exit handler of lab_init calls this function, thus a program prints
// its limits without a call of its own.
void lab_limits_report(void);

// ---- threads ----

// Runs fn(n, ith, data) on n hardware threads (thread 0 is the caller) and waits for all of them.
// Each worker thread acquires an HVX context first. The maximum is 8 threads.
void lab_run_threads(lab_thread_fn fn, void * data, unsigned int n);

// Enables the HMX (SSR bit XE2) for the calling thread
void lab_hmx_enable(void);

// ---- arguments ----

// Parses "--name value" pairs. Returns the value of the option or the default. The function
// records the name and the argument vector, thus the exit of the program finds an option that no
// call of the program read.
long        lab_arg_long(int argc, char ** argv, const char * name, long def);
double      lab_arg_double(int argc, char ** argv, const char * name, double def);
const char * lab_arg_str(int argc, char ** argv, const char * name, const char * def);

// Returns true when argv holds the flag, an option with no value
bool lab_arg_flag(int argc, char ** argv, const char * name);

// The check of the options. An argument that starts with "--" and that no lab_arg_* call read is an
// error: without the check a misspelled option name takes the preset value in silence, and the run
// then measures a case that the reader did not ask for.
//
// The exit handler of lab_init does this check for every program that read an option with lab_arg_*.
// It prints the unknown options and the limits block, and the program ends with the status 2. Thus an
// option that a mode of the program does not read also fails, because the run did not use it.
//
// lab_args_done does the same check at once and stops the program with the status 2. Call it after
// the last lab_arg_* call when the program must not start work with a wrong option.
// Complexity O(argc * names).
void lab_args_done(int argc, char ** argv);

// ---- half floats ----

float    lab_hf_to_f32(uint16_t h);
uint16_t lab_f32_to_hf(float f);

#endif
