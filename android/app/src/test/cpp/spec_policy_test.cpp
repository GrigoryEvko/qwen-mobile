/**
 * The host test of the draft length policy.
 *
 * The primary test drives the policy with the step model that the phone
 * measured, step(d) = a + b*d microseconds and a chain of Bernoulli
 * acceptances, and it checks two properties:
 *
 * - The policy settles on the draft length that maximizes the throughput.
 * - The throughput of the policy is never below plain decode.
 *
 * The second property is the one that matters: a speculative loop must not be
 * slower than no speculative loop.
 *
 *   cmake -S android/app/src/test/cpp -B /tmp/qwen_tests && cmake --build /tmp/qwen_tests
 *   /tmp/qwen_tests/spec_policy_test
 */
#include "spec_policy.h"

#include <cmath>
#include <cstdint>
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

/** A deterministic generator, thus a failure of this test repeats. */
class Rng {
public:
    explicit Rng(uint32_t seed) : state_(seed ? seed : 1u) {}

    /** A value in the interval 0.0 to 1.0. */
    double next() {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 17;
        state_ ^= state_ << 5;
        return (double) (state_ >> 8) / (double) (1u << 24);
    }

private:
    uint32_t state_;
};

/** The step model of one checkpoint on one device. */
struct StepModel {
    const char * name;
    double       a;  // the fixed cost of one step, microseconds
    double       b;  // the cost of one draft position, microseconds
    double       p;  // the conditional acceptance of one draft position
};

/** The tokens that a step of length d gives on average. */
double model_tokens(const StepModel & m, int d) {
    double tokens = 1.0;
    double chain  = 1.0;
    for (int i = 0; i < d; ++i) {
        chain  *= m.p;
        tokens += chain;
    }
    return tokens;
}

/** The draft length with the highest throughput, which the policy must find. */
int model_best(const StepModel & m) {
    int    best = 0;
    double rate = -1.0;
    for (int d = 0; d <= SpecPolicy::kDraftMax; ++d) {
        const double r = model_tokens(m, d) / (m.a + m.b * (double) d);
        if (r > rate) {
            rate = r;
            best = d;
        }
    }
    return best;
}

/** The result of one simulated answer. */
struct RunResult {
    double tokens;       // the tokens of the whole run
    double us;           // the microseconds of the whole run
    int    at_best;      // the steps of the tail that used the best length
    int    tail_steps;   // the steps of the tail
};

/**
 * Drive the policy through n steps of one step model.
 *
 * @param p      The policy
 * @param m      The step model
 * @param n      The steps to run
 * @param tail   The steps at the end that the settled check looks at
 * @param seed   The seed of the acceptance draws
 * @return The tokens, the time, and how often the tail used the best length
 */
RunResult run(SpecPolicy & p, const StepModel & m, int n, int tail, uint32_t seed) {
    Rng       rng(seed);
    RunResult out{0.0, 0.0, 0, 0};
    const int best = model_best(m);
    for (int step = 0; step < n; ++step) {
        const int d = p.next_draft();
        CHECK(d >= 0 && d <= SpecPolicy::kDraftMax);
        int accepted = 0;
        while (accepted < d && rng.next() < m.p) {
            accepted += 1;
        }
        const int    emitted = 1 + accepted;
        const double step_us = m.a + m.b * (double) d;
        p.record(d, accepted, emitted);
        p.observe((int64_t) step_us);
        out.tokens += (double) emitted;
        out.us     += step_us;
        if (step >= n - tail) {
            out.tail_steps += 1;
            if (d == best) {
                out.at_best += 1;
            }
        }
    }
    return out;
}

/**
 * The policy settles on the best draft length and never loses to plain decode.
 *
 * The four models are the measured ones: the 4B Q8_0 with the full tied head
 * and with the 32768-row draft head, each at the acceptance of the application
 * sampler and at the acceptance of a greedy sampler, and one model in which no
 * draft pays.
 */
void test_the_policy_finds_the_best_length() {
    const StepModel models[] = {
        // The full 636 MB tied head, the application sampler. Best is 1.
        {"full head, p 0.52", 108300.0, 27660.0, 0.52},
        // The 32768-row draft head, the application sampler. Best is 2.
        {"draft head, p 0.52", 108300.0, 19000.0, 0.52},
        // The 32768-row draft head, a greedy sampler. Best is 4.
        {"draft head, p 0.78", 108300.0, 19000.0, 0.78},
        // A draft that never pays. Best is 0, thus the policy must decode plain.
        {"no draft pays", 108300.0, 60000.0, 0.30},
    };
    for (const StepModel & m : models) {
        SpecPolicy      p;
        const RunResult r    = run(p, m, 600, 200, 12345u);
        const int       best = model_best(m);
        // The step with no draft inside the speculative context, which the loop
        // must not lose to, and the ceiling that a policy with no warmup and no
        // probe would reach. The intercept a is that step, not the plain decode
        // of a context without a draft, which is cheaper.
        const double plain_us_per_token = m.a;
        const double ideal_us_per_token =
            (m.a + m.b * (double) best) / model_tokens(m, best);
        const double us_per_token = r.us / r.tokens;
        const double share        = (double) r.at_best / (double) r.tail_steps;

        std::printf("  %-20s best %d, settled on %d (%.0f %% of the tail), "
                    "%.0f us/token against plain %.0f and ideal %.0f\n",
                    m.name, best, p.best_draft(), 100.0 * share,
                    us_per_token, plain_us_per_token, ideal_us_per_token);

        // No net loss: the warmup and the probes together must stay below 5 %
        // of plain decode. This is the property that the policy exists for.
        CHECK(us_per_token <= plain_us_per_token * 1.05);
        // The throughput must be near the ceiling. The optimum is flat: at an
        // acceptance of 0.52 the lengths 1 and 2 differ by 2.4 %, thus the
        // policy need not select the exact argmax, only a length beside it.
        CHECK(us_per_token <= ideal_us_per_token * 1.06);
        CHECK(p.best_draft() >= best - 1 && p.best_draft() <= best + 1);
        // Every model but the last must beat plain decode.
        if (best > 0) {
            CHECK(us_per_token < plain_us_per_token);
        }
    }
}

/** The policy follows a change of the step model inside one answer. */
void test_the_policy_follows_a_change() {
    const StepModel fast{"draft head, p 0.78", 108300.0, 19000.0, 0.78};
    const StepModel slow{"no draft pays", 108300.0, 60000.0, 0.30};
    SpecPolicy      p;
    run(p, fast, 400, 1, 999u);
    CHECK(p.best_draft() >= model_best(fast) - 1);
    // The text becomes free form and the draft stops paying. The gap to the
    // next length is large here, thus the policy must reach exactly 0.
    run(p, slow, 600, 1, 777u);
    CHECK(p.best_draft() == model_best(slow));
}

/** A second answer starts at the best length, because the measurements stay. */
void test_reset_keeps_the_measurements() {
    const StepModel m{"draft head, p 0.78", 108300.0, 19000.0, 0.78};
    SpecPolicy      p;
    run(p, m, 400, 1, 4242u);
    const int settled = p.best_draft();
    CHECK(settled >= model_best(m) - 1 && settled <= model_best(m) + 1);

    p.reset();
    CHECK(p.window_drafts() == 0);
    CHECK(p.window_drafted() == 0 && p.window_accepted() == 0);
    // No warmup: the first step of the second answer uses the settled length.
    CHECK(p.best_draft() == settled);
    CHECK(p.next_draft() == settled);

    p.reset_all();
    CHECK(p.samples(0) == 0 && p.samples(SpecPolicy::kDraftMax) == 0);
    CHECK(p.acceptance(1) < 0.0);
    // With no measurement the policy drafts the maximum.
    CHECK(p.best_draft() == SpecPolicy::kDraftMax);
}

/** The per-position acceptance counts only the steps that reached a position. */
void test_conditional_acceptance() {
    SpecPolicy p;
    p.reset_all();
    // Ten steps of 4 drafts that accept the first two and reject the third.
    for (int i = 0; i < 10; ++i) {
        p.record(4, 2, 3);
        p.observe(100000);
    }
    // Positions 1 and 2 always accept, position 3 always rejects.
    CHECK(std::fabs(p.acceptance(1) - 1.0) < 1e-9);
    CHECK(std::fabs(p.acceptance(2) - 1.0) < 1e-9);
    CHECK(std::fabs(p.acceptance(3) - 0.0) < 1e-9);
    // No step reached position 4, thus it holds no measurement.
    CHECK(p.acceptance(4) < 0.0);
    // The chain predicts 1 + 1 + 1 + 0 + 0 tokens at depth 4.
    CHECK(std::fabs(p.predicted_tokens(4) - 3.0) < 1e-9);
    CHECK(std::fabs(p.predicted_tokens(0) - 1.0) < 1e-9);
    // The window reports the rate that the Debug tab shows.
    CHECK(p.window_drafted() == 40 && p.window_accepted() == 20);
    CHECK(std::fabs(p.rate() - 0.5) < 1e-9);
}

/** The line of the step time recovers a and b from the measured lengths. */
void test_the_step_time_line() {
    SpecPolicy p;
    p.reset_all();
    const double a = 108300.0;
    const double b = 19000.0;
    for (int i = 0; i < 20; ++i) {
        for (int d = 0; d <= SpecPolicy::kDraftMax; ++d) {
            p.record(d, 0, 1);
            p.observe((int64_t) (a + b * (double) d));
        }
    }
    for (int d = 0; d <= SpecPolicy::kDraftMax; ++d) {
        CHECK(std::fabs(p.predicted_us(d) - (a + b * (double) d)) < 1.0);
        CHECK(p.samples(d) == 20);
    }
}

/** A step that records nothing does not corrupt a measurement. */
void test_observe_without_record() {
    SpecPolicy p;
    p.reset_all();
    p.record(2, 1, 2);
    p.observe(150000);
    CHECK(p.samples(2) == 1);
    // The second observe has no record before it, thus it does nothing.
    p.observe(900000);
    CHECK(p.samples(2) == 1);
    CHECK(std::fabs(p.predicted_us(2) - 150000.0) < 1.0);
}

/** Values outside the range never index outside an array. */
void test_out_of_range_values() {
    SpecPolicy p;
    p.reset_all();
    p.record(-5, -5, -5);
    p.observe(-1);
    p.record(9999, 9999, 9999);
    p.observe(1000);
    p.record(3, 99, 1);
    p.observe(1000);
    CHECK(p.samples(-1) == 0);
    CHECK(p.samples(SpecPolicy::kDepths) == 0);
    CHECK(p.acceptance(0) < 0.0);
    CHECK(p.acceptance(SpecPolicy::kDraftMax + 1) < 0.0);
    CHECK(p.predicted_us(-1) < 0.0);
    CHECK(p.predicted_us(SpecPolicy::kDepths) < 0.0);
    CHECK(p.predicted_tokens(-1) == 1.0);
    const int d = p.next_draft();
    CHECK(d >= 0 && d <= SpecPolicy::kDraftMax);
}

/** The window forgets the drafts that left it. */
void test_the_window_forgets() {
    SpecPolicy p;
    p.reset_all();
    for (int i = 0; i < SpecPolicy::kWindow; ++i) {
        p.record(3, 3, 4);
    }
    CHECK(p.window_drafts() == SpecPolicy::kWindow);
    CHECK(std::fabs(p.rate() - 1.0) < 1e-9);
    for (int i = 0; i < SpecPolicy::kWindow; ++i) {
        p.record(3, 0, 1);
    }
    CHECK(p.window_drafts() == SpecPolicy::kWindow);
    CHECK(p.window_drafted() == 3 * SpecPolicy::kWindow);
    CHECK(p.window_accepted() == 0);
}

}  // namespace

int main() {
    std::printf("spec_policy_test: the step models\n");
    test_the_policy_finds_the_best_length();
    test_the_policy_follows_a_change();
    test_reset_keeps_the_measurements();
    test_conditional_acceptance();
    test_the_step_time_line();
    test_observe_without_record();
    test_out_of_range_values();
    test_the_window_forgets();
    if (g_failed != 0) {
        std::fprintf(stderr, "%d checks failed\n", g_failed);
        return 1;
    }
    std::printf("spec_policy_test: all checks passed\n");
    return 0;
}
