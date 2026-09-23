// Helpers of the hexhost fuzz targets. Refer to harness.h.

#include "harness.h"

#include "dsp_model.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

namespace harness {

thread_local jmp_buf * g_guard = nullptr;

namespace {

bool g_valid_slots = false;

// Writes the ggml log to stderr only when HEXHOST_LOG is set, and finds the fallback of a fused
// recurrent op (ggml-hexagon.cpp:3815) when every slot index is valid.
void log_filter(enum ggml_log_level level, const char * text, void * user) {
    (void) user;
    static const bool on = getenv("HEXHOST_LOG") != nullptr;
    if (on || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
    if (g_valid_slots && text && strstr(text, "the unfused ops run")) {
        fakedsp::violation("gdn-slot-stale", "the host reads a slot index that is not valid, but the harness wrote a valid one: %s",
                           text);
    }
}

// Returns to the active guard, or lets ggml abort the process (a finding).
void on_abort(const char * message) {
    if (g_guard) {
        longjmp(*g_guard, 1);
    }
    fprintf(stderr, "\nhexhost: GGML ABORT: %s\n", message);
    fflush(stderr);
}

} // namespace

void init() {
    ggml_log_set(log_filter, nullptr);
    ggml_set_abort_callback(on_abort);
}

void expect_valid_slots(bool on) {
    g_valid_slots = on;
}

fakedsp::op_record to_record(const hexhost::packed_op & p) {
    fakedsp::op_record r;
    r.opcode = p.opcode;
    memcpy(r.params, p.params, sizeof(r.params));
    memcpy(r.kparams, p.kparams, sizeof(r.kparams));
    auto conv = [](const hexhost::packed_tensor & s, fakedsp::tensor_ref & t) {
        t.present = s.present;
        t.addr    = s.addr;
        t.size    = s.size;
        t.type    = s.type;
        t.flags   = s.flags;
        memcpy(t.ne, s.ne, sizeof(t.ne));
        memcpy(t.nb, s.nb, sizeof(t.nb));
    };
    for (int i = 0; i < 10; i++) {
        conv(p.src[i], r.src[i]);
    }
    for (int i = 0; i < 4; i++) {
        conv(p.dst[i], r.dst[i]);
    }
    return r;
}

dsp_limits limits_of(hexhost::device * d) {
    dsp_limits l;
    l.n_threads = hexhost::session_n_threads(d);
    l.n_hmx     = hexhost::get_options().nhmx ? hexhost::session_n_hmx(d) : 0;
    l.vtcm      = hexhost::session_vtcm(d);
    return l;
}

void check_packed(hexhost::device * d, const hexhost::packed_op & p, const char * context) {
    const dsp_limits   l = limits_of(d);
    fakedsp::dsp_ctx   ctx;
    ctx.n_threads = l.n_threads;
    ctx.n_hmx     = l.n_hmx;
    ctx.vtcm_size = l.vtcm;
    const fakedsp::op_record  r = to_record(p);
    const fakedsp::op_verdict v  = fakedsp::model_op(ctx, r);
    const std::string         id = fakedsp::verdict_id(v, fakedsp::opcode_name(p.opcode));
    if (v.status != 1) {
        fakedsp::violation(id.c_str(), "%s: the host packs %s and the DSP returns status %u: %s", context,
                           fakedsp::opcode_name(p.opcode), v.status, v.why.c_str());
    } else if (v.silent) {
        fakedsp::violation(id.c_str(), "%s: the host packs %s and the DSP returns OK with a wrong or no result: %s",
                           context, fakedsp::opcode_name(p.opcode), v.why.c_str());
    } else if (v.vtcm_overflow) {
        fakedsp::violation(id.c_str(), "%s: %s: %s", context, fakedsp::opcode_name(p.opcode), v.why.c_str());
    }
}

} // namespace harness

// The preset options of UBSan: each report that no suppression skips stops the process. The
// suppressions come from the shared file tests/sanitizers/ubsan.supp (rule R5) when it exists.
// UBSAN_OPTIONS of the environment adds to these options and can replace them.
// The sanitizer runtime calls this function before the C++ runtime starts (the ASan runtime also
// calls it), thus it must not allocate memory. access() is a plain system call.
extern "C" const char * __ubsan_default_options() {
#ifdef HEXHOST_UBSAN_SUPP
    if (access(HEXHOST_UBSAN_SUPP, R_OK) == 0) {
        return "halt_on_error=1:print_stacktrace=1:report_error_type=1:suppressions=" HEXHOST_UBSAN_SUPP;
    }
#endif
    return "halt_on_error=1:print_stacktrace=1:report_error_type=1";
}

// The preset options of TSan: the first report stops the process, thus libFuzzer keeps the input.
extern "C" const char * __tsan_default_options() {
    return "halt_on_error=1:second_deadlock_stack=1";
}
