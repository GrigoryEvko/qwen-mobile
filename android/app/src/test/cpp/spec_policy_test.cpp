/**
 * The host test of the draft length policy: the thresholds, the probe
 * cadence, the window, and the range checks. No llama.cpp is necessary.
 *
 *   cmake -S android/app/src/test/cpp -B /tmp/qwen_tests && cmake --build /tmp/qwen_tests
 *   /tmp/qwen_tests/spec_policy_test
 */
#include "spec_policy.h"

#include <cstdio>

namespace {

int g_failed = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                                   \
        }                                                                                 \
    } while (0)

/** Record n steps that draft d tokens and get a accepted ones. */
void steps(SpecPolicy & p, int n, int d, int a) {
    for (int i = 0; i < n; ++i) {
        p.record(d, a, a + 1);
    }
}

void test_start_drafts() {
    SpecPolicy p;
    CHECK(p.next_draft() == SpecPolicy::kDraftMax);
    CHECK(p.rate() == 0.0);
    CHECK(p.window_drafts() == 0);
    // Two drafts of 3 tokens are less than kMinDrafted, thus the policy still measures.
    steps(p, 2, 3, 0);
    CHECK(p.window_drafted() == 6 && p.next_draft() == SpecPolicy::kDraftMax);
}

void test_high_rate_keeps_the_long_draft() {
    SpecPolicy p;
    steps(p, 4, 3, 3);
    CHECK(p.rate() == 1.0);
    CHECK(p.next_draft() == SpecPolicy::kDraftMax);
    // 8 of 12 is 0.67, above kRateHigh.
    SpecPolicy q;
    steps(q, 2, 3, 3);
    steps(q, 2, 3, 1);
    CHECK(q.window_drafted() == 12 && q.window_accepted() == 8);
    CHECK(q.next_draft() == SpecPolicy::kDraftMax);
}

void test_middle_rate_drafts_one_token() {
    SpecPolicy p;
    // 5 of 12 is 0.42: between kRateLow and kRateHigh.
    p.record(3, 2, 3);
    p.record(3, 1, 2);
    p.record(3, 2, 3);
    p.record(3, 0, 1);
    CHECK(p.window_drafted() == 12 && p.window_accepted() == 5);
    CHECK(p.next_draft() == SpecPolicy::kDraftMin);
}

void test_low_rate_stops_the_drafts_and_probes() {
    SpecPolicy p;
    // 4 of 12 is 0.33, below kRateLow.
    steps(p, 4, 3, 1);
    CHECK(p.next_draft() == 0);
    // The plain steps count. The probe comes after kProbeTokens tokens.
    for (int i = 0; i < SpecPolicy::kProbeTokens - 1; ++i) {
        p.record(0, 0, 1);
        CHECK(p.next_draft() == 0);
    }
    p.record(0, 0, 1);
    CHECK(p.tokens_since_draft() == SpecPolicy::kProbeTokens);
    CHECK(p.next_draft() == SpecPolicy::kDraftMin);
    // The probe that finds no acceptance starts the interval again.
    p.record(1, 0, 1);
    CHECK(p.tokens_since_draft() == 0 && p.next_draft() == 0);
}

void test_probes_that_accept_start_the_drafts_again() {
    SpecPolicy p;
    steps(p, 4, 3, 1);
    CHECK(p.next_draft() == 0);
    // Formulaic text: each probe accepts its token. The rate climbs above
    // kRateLow, and then above kRateHigh, thus the long drafts come back.
    bool one = false;
    for (int i = 0; i < 40; ++i) {
        p.record(1, 1, 2);
        if (p.next_draft() == SpecPolicy::kDraftMin) {
            one = true;
        }
        if (p.next_draft() == SpecPolicy::kDraftMax) {
            break;
        }
    }
    CHECK(one);
    CHECK(p.next_draft() == SpecPolicy::kDraftMax);
}

void test_the_window_forgets() {
    SpecPolicy p;
    steps(p, SpecPolicy::kWindow, 3, 3);
    CHECK(p.window_drafts() == SpecPolicy::kWindow && p.rate() == 1.0);
    // The window holds kWindow drafts, thus the next kWindow drafts replace all of them.
    steps(p, SpecPolicy::kWindow, 3, 0);
    CHECK(p.window_drafts() == SpecPolicy::kWindow);
    CHECK(p.window_drafted() == 3 * SpecPolicy::kWindow && p.window_accepted() == 0);
    CHECK(p.rate() == 0.0 && p.next_draft() == 0);
    p.reset();
    CHECK(p.window_drafts() == 0 && p.next_draft() == SpecPolicy::kDraftMax);
}

void test_out_of_range_values() {
    SpecPolicy p;
    // More accepted tokens than drafted ones cannot occur. The policy holds the rate at 1.0.
    steps(p, 3, 3, 9);
    CHECK(p.window_accepted() == 9 && p.rate() == 1.0);
    // A negative count does not change the window.
    p.record(-1, -1, -1);
    CHECK(p.window_drafts() == 3 && p.tokens_since_draft() == 0);
}

} // namespace

int main() {
    test_start_drafts();
    test_high_rate_keeps_the_long_draft();
    test_middle_rate_drafts_one_token();
    test_low_rate_stops_the_drafts_and_probes();
    test_probes_that_accept_start_the_drafts_again();
    test_the_window_forgets();
    test_out_of_range_values();
    if (g_failed != 0) {
        std::fprintf(stderr, "%d checks failed\n", g_failed);
        return 1;
    }
    std::printf("spec_policy_test: all checks passed\n");
    return 0;
}
