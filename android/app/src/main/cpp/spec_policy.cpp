#include "spec_policy.h"

#include <algorithm>

void SpecPolicy::reset() {
    std::fill(drafted_, drafted_ + kWindow, (uint8_t) 0);
    std::fill(accepted_, accepted_ + kWindow, (uint8_t) 0);
    count_       = 0;
    next_        = 0;
    sum_drafted_ = 0;
    sum_accepted_ = 0;
    // The first step of an answer drafts, thus a formulaic answer is fast from its first token.
    since_draft_ = kProbeTokens;
}

int SpecPolicy::next_draft() const {
    // Too few samples to decide: draft and measure.
    if (sum_drafted_ < kMinDrafted) {
        return kDraftMax;
    }
    const double r = rate();
    if (r > kRateHigh) {
        return kDraftMax;
    }
    if (r >= kRateLow) {
        return kDraftMin;
    }
    // The drafts do not pay at this time. One probe every kProbeTokens tokens
    // measures the acceptance again, because the text can become formulaic.
    return since_draft_ >= kProbeTokens ? kDraftMin : 0;
}

void SpecPolicy::record(int drafted, int accepted, int emitted) {
    if (drafted <= 0) {
        since_draft_ += std::max(0, emitted);
        return;
    }
    const int d = std::min(drafted, 255);
    const int a = std::min(std::max(accepted, 0), d);
    // The window is full: the oldest entry goes away.
    if (count_ == kWindow) {
        sum_drafted_  -= drafted_[next_];
        sum_accepted_ -= accepted_[next_];
    } else {
        count_ += 1;
    }
    drafted_[next_]  = (uint8_t) d;
    accepted_[next_] = (uint8_t) a;
    next_            = (next_ + 1) % kWindow;
    sum_drafted_    += d;
    sum_accepted_   += a;
    since_draft_     = 0;
}

double SpecPolicy::rate() const {
    return sum_drafted_ > 0 ? (double) sum_accepted_ / (double) sum_drafted_ : 0.0;
}
