#pragma once

/**
 * One section of a system trace, for Perfetto and systrace.
 *
 * The section opens in the constructor and closes in the destructor, on
 * the same thread. The functions ATrace_beginSection, ATrace_endSection and
 * ATrace_isEnabled live in libandroid.so from API 23. The class loads them
 * with dlsym, thus the library has no hard dependency on them, and a section
 * costs one atomic read when no trace records.
 */
class TraceSection {
public:
    /** Open a section with the name. The name must stay valid until the end of the call. */
    explicit TraceSection(const char * name);

    ~TraceSection();

    TraceSection(const TraceSection &) = delete;
    TraceSection & operator=(const TraceSection &) = delete;

    /** Tell if the API is available on this phone. */
    static bool available();

private:
    bool open_ = false;
};
