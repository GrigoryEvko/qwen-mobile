/*
 * A preload library of the test suite: the count of the processors that a
 * test sees.
 *
 * test-opt of llama.cpp chooses its thread count itself, as
 * std::thread::hardware_concurrency() / 2, with no option. libstdc++ gets the
 * count from get_nprocs(), libc++ from sysconf(_SC_NPROCESSORS_ONLN). On a
 * build server with 376 hardware threads, test-opt starts 188 threads, and
 * under ThreadSanitizer it runs for more than one hour. With this library in
 * LD_PRELOAD, the three functions give SUITE_NPROCS (preset 16), thus
 * test-opt starts 8 threads, as on a small CI runner.
 *
 * tests/suite/llama-ctest.sh uses it only for test-opt in the tsan
 * configuration. The library has no sanitizer instrumentation: the TSan
 * runtime calls sysconf during its start, before its own initialization.
 * The library code of llama.cpp does not change.
 */
#define _GNU_SOURCE
#include <stdlib.h>
#include <sys/sysinfo.h>
#include <unistd.h>

/* Return the processor count that the test sees: SUITE_NPROCS if it is a
 * positive integer, else 16. */
static int shim_count(void) {
    const char * s = getenv("SUITE_NPROCS");
    const int n = s != NULL ? atoi(s) : 0;
    return n > 0 ? n : 16;
}

/* Replace get_nprocs of glibc (the source of libstdc++). */
int get_nprocs(void) {
    return shim_count();
}

/* Replace get_nprocs_conf of glibc. */
int get_nprocs_conf(void) {
    return shim_count();
}

/* The sysconf of glibc under its internal name. dlsym is not used, because
 * the TSan runtime calls sysconf before dlsym works. */
extern long __sysconf(int name);

/* Replace sysconf of glibc (the source of libc++) for the two processor
 * counts, and give each other name to glibc. */
long sysconf(int name) {
    if (name == _SC_NPROCESSORS_ONLN || name == _SC_NPROCESSORS_CONF) {
        return shim_count();
    }
    return __sysconf(name);
}
