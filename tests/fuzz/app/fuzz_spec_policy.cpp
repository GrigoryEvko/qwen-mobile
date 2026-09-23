/**
 * The fuzzer of the draft length policy (spec_policy.h). A program is a
 * sequence of steps as spec_step in llama_jni.cpp makes them, and of the
 * calls that the engine does not make (an observe with no record, negative
 * and huge values), because a wrong call must not corrupt the estimates.
 *
 * The checks after each call:
 *
 * - next_draft and best_draft are in [0, kDraftMax].
 * - predicted_tokens is finite and 1.0 or more, and it does not decrease
 *   with the depth, because each position adds a product of acceptances.
 * - predicted_us is finite, and 1.0 or more, or -1.0 before two lengths hold a sample.
 * - acceptance is in [0, 1], or -1.0 before a step reached the position.
 * - rate is in [0, 1], and the window sums are in their ranges.
 */
#include "spec_policy.h"

#include "fuzz_death.h"
#include "harness/crash_input.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include <fuzzer/FuzzedDataProvider.h>

namespace {

[[noreturn]] void fail(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void fail(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "\n==FUZZ-SPEC-POLICY== ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    crash_input::stop();
}

void check(const SpecPolicy & p) {
    const int best = p.best_draft();
    if (best < 0 || best > SpecPolicy::kDraftMax) {
        fail("best_draft gave %d", best);
    }
    double last = 0.0;
    for (int d = 0; d <= SpecPolicy::kDraftMax; ++d) {
        const double t = p.predicted_tokens(d);
        if (!std::isfinite(t) || t < 1.0 || t < last - 1e-12) {
            fail("predicted_tokens(%d) is %g after %g", d, t, last);
        }
        last = t;
        const double us = p.predicted_us(d);
        if (!std::isfinite(us) || (us < 1.0 && us != -1.0)) {
            fail("predicted_us(%d) is %g", d, us);
        }
    }
    for (int i = 1; i <= SpecPolicy::kDraftMax; ++i) {
        const double a = p.acceptance(i);
        if (!(a == -1.0 || (a >= 0.0 && a <= 1.0))) {
            fail("acceptance(%d) is %g", i, a);
        }
    }
    const double r = p.rate();
    if (!(r >= 0.0 && r <= 1.0)) {
        fail("rate is %g", r);
    }
    if (p.window_drafted() < 0 || p.window_accepted() < 0 || p.window_accepted() > p.window_drafted() ||
        p.window_drafts() < 0 || p.window_drafts() > SpecPolicy::kWindow || p.window_drafted() > 255 * SpecPolicy::kWindow) {
        fail("the window holds %d drafts, %d drafted, %d accepted", p.window_drafts(), p.window_drafted(), p.window_accepted());
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
    crash_input::remember(data, size);
    FuzzedDataProvider fdp(data, size);
    SpecPolicy p;
    int ops = 0;
    while (fdp.remaining_bytes() > 0 && ops++ < 4096) {
        switch (fdp.ConsumeIntegralInRange<int>(0, 9)) {
            case 0: case 1: case 2: case 3: case 4: {
                // One step as spec_step makes it: the policy selects, the room of the context can shorten it.
                const int want = p.next_draft();
                if (want < 0 || want > SpecPolicy::kDraftMax) {
                    fail("next_draft gave %d", want);
                }
                const int drafted = std::min(want, fdp.ConsumeIntegralInRange<int>(0, SpecPolicy::kDraftMax));
                const int accepted = fdp.ConsumeIntegralInRange<int>(0, drafted);
                p.record(drafted, accepted, accepted + 1);
                p.observe(fdp.ConsumeIntegralInRange<int64_t>(1, 400000) + drafted * 27000);
                break;
            }
            case 5:
                p.record(fdp.ConsumeIntegral<int>(), fdp.ConsumeIntegral<int>(), fdp.ConsumeIntegral<int>());
                break;
            case 6:
                p.observe(fdp.ConsumeIntegral<int64_t>());
                break;
            case 7:
                p.reset();
                break;
            case 8:
                if (fdp.ConsumeIntegralInRange<int>(0, 15) == 0) {
                    p.reset_all();
                }
                break;
            default:
                p.next_draft();
                break;
        }
        check(p);
    }
    return 0;
}
