/**
 * The draft length policy of the speculative decode loop.
 *
 * A draft step costs more than a plain step: on the OnePlus 13 a draft of 3
 * tokens costs approximately 2.1 plain steps and gives 1 + 3 x acceptance
 * tokens. Thus the step is a gain only above an acceptance of approximately
 * 0.37. Free-form text accepts 39 to 42 %, formulaic text 89 %. The policy
 * measures the acceptance of the last kWindow drafts and selects the draft
 * length of the next step from it.
 *
 * No llama.cpp and no Android dependency: a host test compiles this file.
 * All calls come from the engine thread.
 */
#pragma once

#include <cstdint>

class SpecPolicy {
public:
    /** The number of drafts that the acceptance rate looks back on. */
    static constexpr int kWindow = 32;

    /**
     * The draft length above kRateHigh, and the length of a probe.
     *
     * Measured on the phone on 2026-09-21, 4B Q8_0 on HTP0 with the 32768-row
     * draft head: depth 4 gives 17.44 t/s, against 17.22 at depth 3, 17.28 at
     * 5, 16.95 at 6 and 16.01 at 7. Depth 4 is also where the verify batch
     * reaches 5 rows and crosses HTP_MM_HMX_MIN_NROWS, thus the fourth
     * position costs less than the third. This value is the ceiling, and the
     * rate window below takes the draft down when the acceptance falls.
     */
    static constexpr int kDraftMax = 4;
    static constexpr int kDraftMin = 1;

    /** Above this acceptance rate a step drafts kDraftMax tokens. */
    static constexpr double kRateHigh = 0.55;

    /** Below this acceptance rate a step does not draft. */
    static constexpr double kRateLow = 0.35;

    /** The number of tokens between two probes while the policy drafts nothing. */
    static constexpr int kProbeTokens = 16;

    /**
     * The number of drafted tokens that the window must hold before the rate
     * decides. Before that, each step drafts kDraftMax tokens.
     */
    static constexpr int kMinDrafted = 8;

    SpecPolicy() { reset(); }

    /** Forget the window and the probe counter. Call at the start of an answer. */
    void reset();

    /**
     * The number of tokens that the next step must draft, or 0 for a plain
     * decode. O(1).
     */
    int next_draft() const;

    /**
     * Record the result of one step.
     *
     * @param drafted  The number of tokens that the step drafted, 0 for a plain step
     * @param accepted The number of drafted tokens that the target sampler accepted
     * @param emitted  The number of tokens that the step gives to the caller
     */
    void record(int drafted, int accepted, int emitted);

    /** The acceptance rate of the window, 0.0 when it holds no draft. */
    double rate() const;

    /** The drafted and the accepted tokens of the window. */
    int window_drafted() const { return sum_drafted_; }
    int window_accepted() const { return sum_accepted_; }

    /** The number of drafts in the window. */
    int window_drafts() const { return count_; }

    /** The number of tokens since the last draft. */
    int tokens_since_draft() const { return since_draft_; }

private:
    /** The drafted and the accepted tokens of each draft in the window, oldest at next_. */
    uint8_t drafted_[kWindow];
    uint8_t accepted_[kWindow];

    /** The number of used entries, and the entry that the next draft replaces. */
    int count_;
    int next_;

    /** The sums of the window, thus rate() is O(1). */
    int sum_drafted_;
    int sum_accepted_;

    /** The tokens that the caller received since the last draft. */
    int since_draft_;
};
