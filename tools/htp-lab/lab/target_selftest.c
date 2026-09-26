// Target selftest: the self-test of the verdict of tools/htp-lab/run.sh. Each case ends in one way
// that the lab must report as a failure, and the preset case passes. Thus "run.sh all" runs the
// target as an ordinary one, and tools/htp-lab/test-verdict.sh runs each case and makes sure that
// run.sh gives a non-zero exit code for each failure.
//
// hexagon-sim gives the exit code 0 for each program, thus run.sh reads the verdict from the
// output of the program: the check lines, the exit status line of the lab runtime, and the
// messages of the simulator.
//
// Arguments: --case pass|check|status|stop|abort|exception|hang
//   pass       one check with no value outside its tolerance, and the status 0
//   check      one check with one value outside its tolerance, and the status 0
//   status     no failed check line, and the status 1
//   stop       the program ends before its checks and writes no result line
//   abort      abort() before the checks
//   exception  a word load from an address that is not a multiple of 4: the simulator stops the
//              program with an exception
//   hang       a loop with no end: the cycle limit of the simulator (--plimit) stops it
// lab-run: mode=functional
#include "lab.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TARGET "selftest"

// Does one comparison of four values. The last value of the result differs from its reference by 1
// when bad is true, thus the check reports one value outside the tolerance.
static size_t one_check(bool bad) {
    static const float ref[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float              got[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    if (bad) {
        got[3] += 1.0f;
    }
    return lab_compare_f32(TARGET, got, ref, 4, 0.0f, 0.0f);
}

int main(int argc, char ** argv) {
    const char * const which = lab_arg_str(argc, argv, "--case", "pass");
    lab_args_done(argc, argv);
    lab_init();
    printf("lab: %s case = %s\n", TARGET, which);

    if (strcmp(which, "stop") == 0) {
        fflush(stdout);
        _Exit(0);
    }
    if (strcmp(which, "abort") == 0) {
        fflush(stdout);
        abort();
    }
    if (strcmp(which, "exception") == 0) {
        static uint32_t words[2] __attribute__((aligned(8)));
        const uintptr_t odd = (uintptr_t) words + 1;
        uint32_t        v;
        fflush(stdout);
        __asm__ volatile("%0 = memw(%1)" : "=r"(v) : "r"(odd) : "memory");
        printf("lab: %s the load gave 0x%08lx\n", TARGET, (unsigned long) v);
    }
    if (strcmp(which, "hang") == 0) {
        fflush(stdout);
        for (volatile uint32_t i = 0;; i++) {
        }
    }

    const size_t bad = one_check(strcmp(which, "check") == 0);
    lab_report(TARGET, "values_outside_tolerance", (double) bad, "");
    if (strcmp(which, "status") == 0) {
        return 1;
    }
    return 0;
}
