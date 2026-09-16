#pragma once

#include <cstdint>
#include <vector>

/**
 * One ADPF performance hint session over the inference threads.
 *
 * The session tells the Power HAL the target duration of one unit of work
 * (one decode call) and the measured duration after each call. The HAL then
 * holds the clocks for these threads. The API is in libandroid.so from
 * Android 13 (API 33). The class loads it with dlsym, thus the app keeps
 * minSdk 28, and the session is a no-op on an older phone.
 */
class PerfHintSession {
public:
    /**
     * Open a session for the specified threads.
     *
     * @param tids       Kernel thread identifiers of the inference threads
     * @param target_ns  The target duration of one decode call, in nanoseconds
     */
    PerfHintSession(const std::vector<int32_t> & tids, int64_t target_ns);

    ~PerfHintSession();

    PerfHintSession(const PerfHintSession &) = delete;
    PerfHintSession & operator=(const PerfHintSession &) = delete;

    /** Tell if the phone supports the API and the session is open. */
    bool ok() const { return session_ != nullptr; }

    /** Change the target duration of one decode call. */
    void set_target(int64_t target_ns);

    /** Report the measured duration of the last decode call. */
    void report(int64_t actual_ns);

private:
    void * session_ = nullptr;
};
