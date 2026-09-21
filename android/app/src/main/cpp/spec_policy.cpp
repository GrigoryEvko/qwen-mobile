#include "spec_policy.h"

#include <algorithm>

namespace {

/** The largest value that a sample counter holds. It prevents an overflow. */
constexpr int kSampleCap = 1 << 24;

/** The acceptance that a draft position gets before any step reaches it. */
constexpr double kAcceptPrior = 0.5;

}  // namespace

double SpecPolicy::weight_of(int n) {
    // Before the history holds 1/(1 - kDecay) samples, the weight is that of a
    // plain mean, thus the estimate converges in a few samples.
    if (n <= 0) {
        return 0.0;
    }
    const double as_mean = 1.0 - 1.0 / (double) (n + 1);
    return as_mean < kDecay ? as_mean : kDecay;
}

void SpecPolicy::fold(double & mean, int & n, double value) {
    const double w = weight_of(n);
    mean = mean * w + value * (1.0 - w);
    if (n < kSampleCap) {
        n += 1;
    }
}

void SpecPolicy::reset() {
    // The acceptance window describes one answer, thus it goes away here. The
    // step time and the per-position acceptance describe the model and the
    // device, thus they stay: the second answer starts at the best known draft
    // length and pays no warmup.
    std::fill(drafted_, drafted_ + kWindow, (uint8_t) 0);
    std::fill(accepted_, accepted_ + kWindow, (uint8_t) 0);
    count_         = 0;
    next_          = 0;
    sum_drafted_   = 0;
    sum_accepted_  = 0;
    pending_depth_ = -1;
    // The first step of an answer uses the best known length, not a probe.
    since_probe_ = 0;
}

void SpecPolicy::reset_all() {
    reset();
    std::fill(time_us_, time_us_ + kDepths, 0.0);
    std::fill(time_n_, time_n_ + kDepths, 0);
    std::fill(accept_p_, accept_p_ + kDraftMax, 0.0);
    std::fill(accept_n_, accept_n_ + kDraftMax, 0);
    warmup_left_ = kWarmupSteps;
    // The warmup starts at the longest draft, thus a formulaic answer is fast
    // from its first step.
    warm_cursor_ = kDraftMax;
}

int SpecPolicy::samples(int depth) const {
    return (depth < 0 || depth >= kDepths) ? 0 : time_n_[depth];
}

double SpecPolicy::acceptance(int position) const {
    const int i = position - 1;
    if (i < 0 || i >= kDraftMax || accept_n_[i] <= 0) {
        return -1.0;
    }
    return accept_p_[i];
}

double SpecPolicy::predicted_us(int depth) const {
    if (depth < 0 || depth >= kDepths) {
        return -1.0;
    }
    // A least-squares line over the draft lengths that hold a measurement. The
    // step time is nearly deterministic, thus two points already give a and b,
    // and the line covers a length that no step measured.
    double n = 0.0;
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    for (int d = 0; d < kDepths; ++d) {
        if (time_n_[d] <= 0) {
            continue;
        }
        const double x = (double) d;
        const double y = time_us_[d];
        n   += 1.0;
        sx  += x;
        sy  += y;
        sxx += x * x;
        sxy += x * y;
    }
    if (n < 2.0) {
        // One point gives no slope. The measured length keeps its own time.
        return time_n_[depth] > 0 ? time_us_[depth] : -1.0;
    }
    const double denom = n * sxx - sx * sx;
    if (denom <= 0.0) {
        return time_n_[depth] > 0 ? time_us_[depth] : -1.0;
    }
    double slope = (n * sxy - sx * sy) / denom;
    // A draft position cannot make a step faster. Noise can give a negative
    // slope, and a negative slope would select kDraftMax always.
    if (slope < 0.0) {
        slope = 0.0;
    }
    const double base = (sy - slope * sx) / n;
    const double us   = base + slope * (double) depth;
    return us > 1.0 ? us : 1.0;
}

double SpecPolicy::predicted_tokens(int depth) const {
    if (depth <= 0) {
        return 1.0;
    }
    const int d = depth > kDraftMax ? kDraftMax : depth;
    // The draft is a chain: position i is verified only when every position
    // before it was accepted. Thus the expected tokens are the sum of the
    // products of the conditional acceptances.
    double tokens = 1.0;
    double chain  = 1.0;
    double last   = kAcceptPrior;
    for (int i = 0; i < d; ++i) {
        // A position that no step reached inherits the last measured one, thus
        // a deep position is neither free nor impossible before its first sample.
        const double p = accept_n_[i] > 0 ? accept_p_[i] : last;
        last   = p;
        chain *= p;
        tokens += chain;
    }
    return tokens;
}

int SpecPolicy::best_draft() const {
    int    best  = 0;
    double best_rate = -1.0;
    for (int d = 0; d < kDepths; ++d) {
        const double us = predicted_us(d);
        if (us <= 0.0) {
            continue;
        }
        const double r = predicted_tokens(d) / us;
        if (r > best_rate) {
            best_rate = r;
            best      = d;
        }
    }
    // No length holds a measurement: draft the maximum, thus a formulaic answer
    // is fast from its first step.
    return best_rate < 0.0 ? kDraftMax : best;
}

int SpecPolicy::next_draft() {
    // The warmup walks every draft length twice, thus the line of the step time
    // and the acceptance of every position hold a sample before the first
    // prediction.
    if (warmup_left_ > 0) {
        warmup_left_ -= 1;
        const int d  = warm_cursor_;
        warm_cursor_ = (warm_cursor_ + 1) % kDepths;
        return d;
    }

    const int best = best_draft();

    // The probe drafts one token more than the length in use. That is the only
    // step that reaches draft position best + 1, and no other step measures the
    // acceptance of that position. The line of the step time already predicts
    // every other length, thus no other probe is necessary.
    //
    // One neighbour is enough, because the throughput is unimodal in the draft
    // length: the predicted tokens are concave in the length, because each term
    // of the chain is the one before it times an acceptance of 1.0 or less, and
    // the time is linear. Thus a climb to the better neighbour reaches the
    // maximum and cannot stop at a different length.
    //
    // A probe that drafts kDraftMax instead would cost far more. At the step
    // model a = 108.3 ms, b = 60 ms and an acceptance of 0.30 the best length
    // is 0, and a probe of 4 tokens costs 2.26 times the plain step per token,
    // which is a loss of 11 % of the throughput. The neighbour probe costs 1.6 %.
    if (since_probe_ >= kProbeEvery) {
        since_probe_ = 0;
        return best < kDraftMax ? best + 1 : 0;
    }
    since_probe_ += 1;
    return best;
}

void SpecPolicy::record(int drafted, int accepted, int emitted) {
    (void) emitted;
    // The step that ran is the one that observe() must charge, whatever the
    // caller selected: the context limit can make the step shorter.
    pending_depth_ = std::min(std::max(drafted, 0), kDraftMax);
    if (drafted <= 0) {
        return;
    }
    const int d = std::min(drafted, 255);
    const int a = std::min(std::max(accepted, 0), d);

    // The conditional acceptance of each position. Position i + 1 is reached
    // only when positions 1 to i were accepted, thus the loop stops at the
    // first position that the target rejected.
    const int reached = std::min(d, kDraftMax);
    for (int i = 0; i < reached; ++i) {
        if (i > a) {
            break;
        }
        fold(accept_p_[i], accept_n_[i], i < a ? 1.0 : 0.0);
    }

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
}

void SpecPolicy::observe(int64_t step_us) {
    const int d    = pending_depth_;
    pending_depth_ = -1;
    if (d < 0 || d >= kDepths || step_us <= 0) {
        return;
    }
    fold(time_us_[d], time_n_[d], (double) step_us);
}

double SpecPolicy::rate() const {
    return sum_drafted_ > 0 ? (double) sum_accepted_ / (double) sum_drafted_ : 0.0;
}
