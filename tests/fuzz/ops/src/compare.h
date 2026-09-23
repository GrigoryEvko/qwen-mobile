// The comparison of backend outputs with the oracle outputs, with the per-element bounds of the kind.

#pragma once

#include "case.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fo {

// The verdict of one case, from the best to the worst.
enum verdict : int {
    V_PASS      = 0, // every element inside the strict bound
    V_SUBNORMAL = 1, // differences only where the oracle and the backend are below FLT_MIN
    V_STRICT    = 2, // an element above the strict bound, inside the loose bound
    V_LOOSE     = 3, // an element above the loose bound
    V_NONFINITE = 4, // a NaN or Inf where the other side has a different value
};

// The comparison of one output.
struct out_cmp {
    int64_t n_cmp       = 0;  // compared elements
    int64_t n_strict    = 0;  // above the strict bound
    int64_t n_loose     = 0;  // above the loose bound
    int64_t n_nonfinite = 0;
    int64_t n_subnormal = 0;
    double  max_err     = 0.0;
    double  max_ratio   = 0.0; // max of err / strict (err / half an ulp where strict is 0)
    double  max_ratio_l = 0.0; // max of err / loose
    double  max_ulp     = 0.0; // max of err / ulp(ref)
    int64_t worst_i     = -1;
    float   worst_ref   = 0.0f;
    float   worst_got   = 0.0f;
    double  worst_strict = 0.0;
    double  worst_loose  = 0.0;
    int64_t nf_i        = -1;  // the first non-finite mismatch
    float   nf_ref      = 0.0f;
    float   nf_got      = 0.0f;
};

// The comparison of one case.
struct case_cmp {
    std::vector<out_cmp> outs;
    verdict              v     = V_PASS;
    double               ratio = 0.0; // the max strict ratio over the outputs
    double               ulp   = 0.0; // the max ulp error over the outputs
    std::string          text;        // a readable summary
};

// Compare the outputs of a backend with the outputs of the oracle. The time is O(elements) plus
// the cost of the bound rule of the kind.
//
// `fast` gives the bound of a backend that is built with -ffp-model=fast (the CPU backend in both
// profiles, as the app ships it): strict' = 2 strict + 4u|y| and loose' = max(loose, strict').
// Reason: fast math lets the compiler turn a division into a product with a reciprocal (one more
// rounding), contract a product and a sum into an FMA (one rounding less, a different result by up
// to one rounding), and reassociate a sum. The strict rules already allow any order of a sum once,
// and the factor 2 allows the second order of the oracle side; 4u|y| covers the extra roundings at
// the end. The Hexagon DSP code has its own compiler flags, thus its results keep the plain rule.
case_cmp compare_case(const built_case & c, const std::vector<std::vector<uint8_t>> & got,
                      const std::vector<std::vector<uint8_t>> & ref, bool fast = false);

// Return the name of a verdict.
const char * verdict_name(int v);

// Return the severity of a verdict: none, low, medium or high.
const char * verdict_severity(int v, bool special);

} // namespace fo
