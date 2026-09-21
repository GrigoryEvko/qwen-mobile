/**
 * The draft length policy of the speculative decode loop.
 *
 * A step that drafts d tokens costs approximately a + b*d microseconds and
 * gives 1 + accepted tokens. The policy must select the d with the highest
 * tokens per microsecond.
 *
 * An earlier version selected d from the acceptance rate against two fixed
 * thresholds. That rule is wrong, because the break-even acceptance depends on
 * b, and b is a property of the checkpoint: on the OnePlus 13 the 4B Q8_0 with
 * the full tied head measures b = 27.66 ms, and the same model with the
 * 32768-row draft head measures b = 19.0 ms. At an acceptance of 0.52 the first
 * wants d = 1 and the second wants d = 2, thus no fixed threshold is correct
 * for both.
 *
 * The policy also cannot simply measure the tokens per microsecond of each
 * candidate, because that measurement is too noisy to be useful. At d = 4 and
 * an acceptance of 0.52 the emitted count has a standard deviation of 1.24 on a
 * mean of 2.0, thus a rank of two candidates that differ by 10 % needs
 * approximately 150 steps of each, and one answer does not hold that many.
 *
 * Thus the policy measures the two quantities that have a low variance, and it
 * predicts the third:
 *
 * - The time of a step at each draft length. This is nearly deterministic. A
 *   least-squares line over the measured lengths gives a and b, thus the policy
 *   also has a time for a length it did not measure.
 * - The conditional acceptance of each draft position: how often position i is
 *   accepted, counted only over the steps that reached position i. Each count
 *   is a mean of Bernoulli trials, and position 1 gets a sample at every step.
 * - The predicted tokens of a step of length d, which is
 *   1 + sum over i of the product of the acceptances up to i.
 *
 * Candidate 0 is a step with no draft, thus the policy can always fall back to
 * plain decode and the loop has no net loss against it.
 *
 * The caller must call record() one time for each step, and then observe() with
 * the time of that step.
 *
 * No llama.cpp and no Android dependency: a host test compiles this file.
 * All calls come from the engine thread.
 */
#pragma once

#include <cstdint>

class SpecPolicy {
public:
    /**
     * The longest draft that the policy can select.
     *
     * Measured on the phone on 2026-09-21, 4B Q8_0 on HTP0 with the 32768-row
     * draft head and a greedy sampler: depth 4 gives 17.44 t/s, against 17.22
     * at depth 3, 17.28 at 5, 16.95 at 6 and 16.01 at 7. Depth 4 is also where
     * the verify batch reaches 5 rows and crosses HTP_MM_HMX_MIN_NROWS, thus
     * the fourth position costs less than the third.
     */
    static constexpr int kDraftMax = 4;

    /** The number of candidates: a draft of 0 tokens up to kDraftMax tokens. */
    static constexpr int kDepths = kDraftMax + 1;

    /** The number of drafts that the reported acceptance rate looks back on. */
    static constexpr int kWindow = 32;

    /**
     * The weight that the history of a measurement keeps when a new sample
     * arrives, once it holds 1/(1 - kDecay) samples.
     *
     * Before that the weight is 1 - 1/n, thus the first samples count in full
     * and the estimate converges quickly. A decay of 0.9 gives a settled
     * estimate a memory of approximately 10 samples, thus the policy follows a
     * change of the clock, of the context length or of the text.
     */
    static constexpr double kDecay = 0.9;

    /**
     * The steps between two probes of a draft length that the policy does not
     * use.
     *
     * The probe drafts one token more than the length in use, because that is
     * the only step that measures the acceptance of the next draft position.
     * A policy that already drafts kDraftMax probes 0 instead. The measured
     * cost of the probe is 1.6 % of the throughput or less.
     */
    static constexpr int kProbeEvery = 16;

    /** The steps that the policy measures before it predicts. */
    static constexpr int kWarmupSteps = 2 * kDepths;

    SpecPolicy() { reset_all(); }

    /**
     * Forget the acceptance window. Call at the start of an answer.
     *
     * The measured step time and the measured acceptance stay, because they
     * describe the model and the device and not the answer. Thus the second
     * answer starts at the best known draft length and pays no warmup.
     */
    void reset();

    /** Forget the window and every measurement. */
    void reset_all();

    /**
     * The number of tokens that the next step must draft, 0 for a plain step.
     *
     * The call is not const, because it advances the warmup cursor and the
     * probe cursor. Call it one time for each step. O(kDepths).
     */
    int next_draft();

    /**
     * Record the result of one step. Call observe() after it with the time.
     *
     * @param drafted  The number of tokens that the step drafted, 0 for a plain step
     * @param accepted The number of drafted tokens that the target sampler accepted
     * @param emitted  The number of tokens that the step gives to the caller
     */
    void record(int drafted, int accepted, int emitted);

    /**
     * Give the step that record() took its measured time.
     *
     * A call with no record() before it does nothing, thus a step that ends
     * early and records nothing does not corrupt a measurement. O(1).
     *
     * @param step_us The microseconds of the whole step
     */
    void observe(int64_t step_us);

    /** The acceptance rate of the window, 0.0 when it holds no draft. */
    double rate() const;

    /**
     * The conditional acceptance of one draft position.
     *
     * @param position The position in the draft, 1 to kDraftMax
     * @return The acceptance, or -1.0 when no step reached that position
     */
    double acceptance(int position) const;

    /**
     * The predicted microseconds of one step, from the line of the step time.
     *
     * @param depth The draft length, 0 to kDraftMax
     * @return The time, or -1.0 while fewer than two lengths hold a measurement
     */
    double predicted_us(int depth) const;

    /**
     * The predicted tokens of one step.
     *
     * @param depth The draft length, 0 to kDraftMax
     * @return The tokens, which is 1.0 or more
     */
    double predicted_tokens(int depth) const;

    /** The draft length with the highest predicted tokens per microsecond. */
    int best_draft() const;

    /** The drafted and the accepted tokens of the window. */
    int window_drafted() const { return sum_drafted_; }
    int window_accepted() const { return sum_accepted_; }

    /** The number of drafts in the window. */
    int window_drafts() const { return count_; }

    /** The number of steps that measured the time of one draft length. */
    int samples(int depth) const;

private:
    /**
     * Fold one sample into a decayed mean.
     *
     * @param mean  The mean, which the call changes
     * @param n     The sample count, which the call increments
     * @param value The new sample
     */
    static void fold(double & mean, int & n, double value);

    /** The weight that a history of n samples keeps when sample n + 1 arrives. */
    static double weight_of(int n);

    /** The drafted and the accepted tokens of each draft in the window, oldest at next_. */
    uint8_t drafted_[kWindow];
    uint8_t accepted_[kWindow];

    /** The number of used entries, and the entry that the next draft replaces. */
    int count_;
    int next_;

    /** The sums of the window, thus rate() is O(1). */
    int sum_drafted_;
    int sum_accepted_;

    /** The decayed mean microseconds of a step of each draft length. */
    double time_us_[kDepths];

    /** The number of steps that measured each draft length. */
    int time_n_[kDepths];

    /** The decayed conditional acceptance of draft position i + 1. */
    double accept_p_[kDraftMax];

    /** The number of steps that reached draft position i + 1. */
    int accept_n_[kDraftMax];

    /** The steps of the warmup that remain. */
    int warmup_left_;

    /** The draft length that the warmup measures next. */
    int warm_cursor_;

    /** The steps since the last probe. */
    int since_probe_;

    /** The draft length of the step that record() took, or -1 when none is open. */
    int pending_depth_;
};
