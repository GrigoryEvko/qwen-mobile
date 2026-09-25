/**
 * The interpreter of the app fuzzers (refer to app_harness.h).
 *
 * This file includes llama_jni.cpp, thus the checks read the Engine of a
 * handle directly. Four functions of the JNI layer go through hooks, with a
 * macro before the include:
 *
 * - common_speculative_process: a fault plan makes the draft context refuse
 *   a batch, as a failed decode on the NPU does.
 * - setpriority and closedir: the check of the thread priorities.
 * - ggml_threadpool_free: the thread that destroys an engine. The app calls
 *   free on the engine thread, and only requestStop on another thread. Thus
 *   a destruction on another thread in a call that is not free (the frees
 *   of the second thread of --cross-free) would stop the thread of the
 *   caller, in the app the display thread, for the time of the destructor.
 *
 * The checks after each operation, for each live engine:
 *
 * - The out queue: out_next <= out_queue.size().
 * - The memory: the last position of sequence 0 is n_past - 1, and a memory
 *   with no items is empty. A text-only memory has one position per item.
 *   The same for the prefill context of the hybrid backend.
 * - The draft context holds no position at or after n_past, and the prompt
 *   of the draft driver is the text of the items.
 * - A token that went to the app is in the memory, or it is the token of the
 *   last step (id_last), which the next step decodes.
 * - The exactness oracle: the state of the engine, restored
 *   into a new context, gives the same logits for a probe token as a new
 *   context that decodes the items of the engine from the start.
 *
 * A C++ exception that leaves a JNI function, and each error of the fake VM
 * (fake_jni.h), abort the process with a message.
 */
#include "app_harness.h"

#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fuzzer/FuzzedDataProvider.h>

#include "cache_io.h"
#include "chat.h"
#include "common.h"
#include "crash_input.h"
#include "fake_jni.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "image_cache.h"
#include "llama.h"
// The memory breakdown of a context: the path of common/speculative.cpp, through the include directory common.
#include "../src/llama-ext.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "perf_hint.h"
#include "spec_policy.h"
#include "speculative.h"
#include "state_cache.h"
#include "trace.h"

// The hooks. They have C++ linkage and live outside of every namespace, thus the macros below reach them.
int  harness_setpriority(int which, id_t who, int prio);
int  harness_closedir(DIR * dir);
bool harness_spec_process(common_speculative * spec, const llama_batch & batch);
void harness_threadpool_free(ggml_threadpool * tp);

#define setpriority                harness_setpriority
#define closedir                   harness_closedir
#define common_speculative_process harness_spec_process
#define ggml_threadpool_free       harness_threadpool_free
#include "llama_jni.cpp"
#undef setpriority
#undef closedir
#undef common_speculative_process
#undef ggml_threadpool_free

namespace {

using fakejni::fail;

/** The fault plan of the draft follow: the countdown-th call from now returns false. -1 is off. */
std::atomic<int> g_spec_countdown{-1};
std::atomic<uint64_t> g_spec_faults{0};

/** The setpriority calls of the process: the target thread and the value. */
std::mutex g_prio_mutex;
std::vector<std::pair<int32_t, int>> g_prio_calls;

/** The priority check: a thread that starts after the first listing of /proc/self/task in a load. */
std::atomic<bool> g_bystander_armed{false};
std::atomic<int32_t> g_bystander_tid{0};
std::mutex g_bystander_mutex;
std::condition_variable g_bystander_cv;
bool g_bystander_release = false;
std::thread g_bystander;

/**
 * The engine thread of the harness, and the counts of the engines that
 * another thread destroyed: in a free of that thread, and in another call.
 */
std::thread::id g_engine_thread;
std::atomic<uint64_t> g_off_thread_frees{0};
std::atomic<uint64_t> g_off_thread_other{0};

/** The name of the JNI call that runs on this thread, or null. */
thread_local const char * t_jni_call = nullptr;

}  // namespace

int harness_setpriority(int which, id_t who, int prio) {
    {
        std::lock_guard<std::mutex> lock(g_prio_mutex);
        g_prio_calls.emplace_back((int32_t) who, prio);
    }
    // glibc types the first argument as an enum, bionic as an int.
    return ::setpriority(static_cast<decltype(PRIO_PROCESS)>(which), who, prio);
}

int harness_closedir(DIR * dir) {
    const int rc = ::closedir(dir);
    if (g_bystander_armed.exchange(false)) {
        // The first listing of the load is complete. A thread that starts now is a
        // thread of the app that the load did not make, for example a thread of
        // the IO dispatcher of Kotlin.
        std::atomic<bool> started{false};
        g_bystander = std::thread([&started] {
            // ART names each new Java thread, as here a thread of the IO dispatcher of Kotlin.
            pthread_setname_np(pthread_self(), "DefaultDispatch");
            g_bystander_tid = (int32_t) syscall(SYS_gettid);
            started = true;
            std::unique_lock<std::mutex> lock(g_bystander_mutex);
            g_bystander_cv.wait(lock, [] { return g_bystander_release; });
        });
        while (!started) {
            std::this_thread::yield();
        }
    }
    return rc;
}

bool harness_spec_process(common_speculative * spec, const llama_batch & batch) {
    int c = g_spec_countdown.load();
    while (c >= 0) {
        if (g_spec_countdown.compare_exchange_weak(c, c - 1)) {
            if (c == 0) {
                g_spec_faults += 1;
                return false;
            }
            break;
        }
    }
    return common_speculative_process(spec, batch);
}

void harness_threadpool_free(ggml_threadpool * tp) {
    if (std::this_thread::get_id() != g_engine_thread) {
        if (t_jni_call != nullptr && strcmp(t_jni_call, "free") == 0) {
            g_off_thread_frees += 1;
        } else {
            g_off_thread_other += 1;
            fakejni::fail("an engine was destroyed off the engine thread in %s: in the app that call runs on the "
                          "display thread, which then stops for the time of the destructor",
                          t_jni_call != nullptr ? t_jni_call : "no JNI call");
        }
    }
    ggml_threadpool_free(tp);
}

namespace harness {

namespace {

constexpr const char * kNativeClass = "ai/airi/qwenmobile/LlamaNative";

Counters g_counters;

/** The distinct messages of the Java exceptions, with the digits removed, and their counts. */
std::mutex g_exc_mutex;
std::map<std::string, uint64_t> g_exceptions;

/** One message of a conversation as the Kotlin side has it. */
struct Msg {
    std::string    role;
    std::u16string text;
    int            image = -1;  // An index into Program::images, or -1.
};

/** The parameters of one load. */
struct LoadSpec {
    std::string model;
    std::string mmproj;
    std::string device;
    std::string prefill;
    std::string vision;
    int         gpu_layers = 0;
    int         threads = 2;
    int         n_ctx = 2048;
    int         image_max_tokens = 256;
    bool        speculative = false;
    std::string cache;
};

/** An engine that the program loaded and did not free. */
struct LiveEngine {
    jlong            handle = 0;
    LoadSpec         spec;
    std::vector<Msg> convo;
    std::string      answer;
    bool             answer_open = false;
    /** True when the memory changed after the last run of the oracle. */
    bool             dirty = false;
    /** The record of the turn for check_delivered: the tokens the app received, and the items of the prompt. */
    bool                     tracking = false;
    std::vector<llama_token> delivered;
    size_t                   prompt_items = 0;
};

/** The state of one program. */
struct Program {
    const Options *                   opt = nullptr;
    FuzzedDataProvider *              fdp = nullptr;
    std::vector<LiveEngine>           live;
    std::vector<jlong>                dead;
    std::vector<std::vector<uint8_t>> images;
    std::vector<uint8_t>              tape;
    size_t                            tape_pos = 0;
    std::string                       dir;
    std::string                       cache_a;
    std::string                       cache_b;
    std::atomic<jlong>                current{0};
};

thread_local Program * g_prog = nullptr;

void trace(const Options & opt, const char * fmt, ...) __attribute__((format(printf, 2, 3)));
void trace(const Options & opt, const char * fmt, ...) {
    if (!opt.trace) {
        return;
    }
    static const auto t0 = std::chrono::steady_clock::now();
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[harness %8.3f] ",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

/** Record the message of a Java exception. */
void note_exception(const fakejni::CallOutcome & out, const char * call) {
    if (!out.threw) {
        return;
    }
    g_counters.java_exceptions += 1;
    std::string key = std::string(call) + ": " + out.exception_class + ": ";
    for (char c : out.exception_message) {
        if (key.size() > 200) {
            break;
        }
        key.push_back(c >= '0' && c <= '9' ? '#' : c);
    }
    std::lock_guard<std::mutex> lock(g_exc_mutex);
    g_exceptions[key] += 1;
}

/** Run one native call in a frame of the fake VM. A C++ exception that leaves it is an error. */
template <typename F>
fakejni::CallOutcome jni_call(const char * name, F && body) {
    JNIEnv * env = fakejni::env();
    fakejni::begin_call(env, name);
    const char * outer = t_jni_call;
    t_jni_call = name;
    struct Restore {
        const char * name;
        ~Restore() { t_jni_call = name; }
    } restore{outer};
    try {
        body(env);
    } catch (const std::exception & ex) {
        fail("a C++ exception crossed the JNI boundary in %s: %s (the process calls std::terminate)", name, ex.what());
    } catch (...) {
        fail("a C++ exception crossed the JNI boundary in %s (the process calls std::terminate)", name);
    }
    fakejni::CallOutcome out = fakejni::end_call(env);
    note_exception(out, name);
    return out;
}

jclass native_class(JNIEnv * env) {
    return reinterpret_cast<jclass>(fakejni::arg(env, fakejni::class_object(kNativeClass)));
}

jstring jstr(JNIEnv * env, const std::string & s, bool null_if_empty) {
    if (null_if_empty && s.empty()) {
        return nullptr;
    }
    return reinterpret_cast<jstring>(fakejni::arg(env, fakejni::new_string_utf8(s)));
}

// --- The JNI calls as LlamaNative declares them. ---

jlong api_load(const LoadSpec & s) {
    jlong h = 0;
    jni_call("load", [&](JNIEnv * env) {
        h = Java_ai_airi_qwenmobile_LlamaNative_load(
            env, native_class(env), jstr(env, s.model, false), jstr(env, s.mmproj, true), jstr(env, s.device, true),
            jstr(env, s.prefill, true), jstr(env, s.vision, true), s.gpu_layers, s.threads, s.n_ctx, s.image_max_tokens,
            s.speculative ? JNI_TRUE : JNI_FALSE, jstr(env, s.cache, true));
    });
    return h;
}

void api_free(jlong h) {
    jni_call("free", [&](JNIEnv * env) { Java_ai_airi_qwenmobile_LlamaNative_free(env, native_class(env), h); });
}

void api_request_stop(jlong h) {
    jni_call("requestStop", [&](JNIEnv * env) { Java_ai_airi_qwenmobile_LlamaNative_requestStop(env, native_class(env), h); });
}

/** chatStart. The shape flags break the contract of the Kotlin side on purpose. Returns the result, or -2 on an exception. */
int api_chat_start(Program & p, jlong h, const std::vector<Msg> & msgs, bool thinking, float temp, float top_p,
                   bool images_null, bool short_contents) {
    int rc = -2;
    const fakejni::CallOutcome out = jni_call("chatStart", [&](JNIEnv * env) {
        std::vector<fakejni::Obj *> roles, contents, imgs;
        for (const Msg & m : msgs) {
            roles.push_back(fakejni::new_string_utf8(m.role));
            contents.push_back(fakejni::new_string(m.text));
            imgs.push_back(m.image >= 0 ? fakejni::new_byte_array(p.images[(size_t) m.image].data(),
                                                                  p.images[(size_t) m.image].size())
                                        : nullptr);
        }
        if (short_contents && !contents.empty()) {
            contents.pop_back();
        }
        jobject jroles    = fakejni::arg(env, fakejni::new_object_array("java/lang/String", roles));
        jobject jcontents = fakejni::arg(env, fakejni::new_object_array("java/lang/String", contents));
        jobject jimages   = images_null ? nullptr : fakejni::arg(env, fakejni::new_object_array("[B", imgs));
        rc = Java_ai_airi_qwenmobile_LlamaNative_chatStart(env, native_class(env), h, (jobjectArray) jroles,
                                                           (jobjectArray) jcontents, (jobjectArray) jimages,
                                                           thinking ? JNI_TRUE : JNI_FALSE, temp, top_p);
    });
    return out.threw ? -2 : rc;
}

/** stageImage. Returns the result, or -2 on an exception. */
int api_stage_image(Program & p, jlong h, const std::vector<Msg> & msgs, bool thinking) {
    int rc = -2;
    const fakejni::CallOutcome out = jni_call("stageImage", [&](JNIEnv * env) {
        std::vector<fakejni::Obj *> roles, contents, imgs;
        for (const Msg & m : msgs) {
            roles.push_back(fakejni::new_string_utf8(m.role));
            contents.push_back(fakejni::new_string(m.text));
            imgs.push_back(m.image >= 0 ? fakejni::new_byte_array(p.images[(size_t) m.image].data(),
                                                                  p.images[(size_t) m.image].size())
                                        : nullptr);
        }
        jobject jroles    = fakejni::arg(env, fakejni::new_object_array("java/lang/String", roles));
        jobject jcontents = fakejni::arg(env, fakejni::new_object_array("java/lang/String", contents));
        jobject jimages   = fakejni::arg(env, fakejni::new_object_array("[B", imgs));
        rc = Java_ai_airi_qwenmobile_LlamaNative_stageImage(env, native_class(env), h, (jobjectArray) jroles,
                                                            (jobjectArray) jcontents, (jobjectArray) jimages,
                                                            thinking ? JNI_TRUE : JNI_FALSE);
    });
    return out.threw ? -2 : rc;
}

/** prepareVision. Returns 1 when the projector is ready, 0 when not, -2 on an exception. */
int api_prepare_vision(jlong h) {
    jboolean r = JNI_FALSE;
    const fakejni::CallOutcome out = jni_call("prepareVision", [&](JNIEnv * env) {
        r = Java_ai_airi_qwenmobile_LlamaNative_prepareVision(env, native_class(env), h);
    });
    return out.threw ? -2 : (r == JNI_TRUE ? 1 : 0);
}

/** generateNext. Returns 1 with the piece, 0 for null, -1 for an exception. */
int api_generate_next(jlong h, std::vector<int8_t> & piece) {
    jbyteArray res = nullptr;
    fakejni::Obj * obj = nullptr;
    const fakejni::CallOutcome out = jni_call("generateNext", [&](JNIEnv * env) {
        res = Java_ai_airi_qwenmobile_LlamaNative_generateNext(env, native_class(env), h);
        obj = res != nullptr ? fakejni::object_of(env, res, "generateNext result") : nullptr;
    });
    if (out.threw) {
        return -1;
    }
    if (obj == nullptr) {
        return 0;
    }
    if (obj->kind != fakejni::Kind::ByteArray || obj->bytes.empty()) {
        fail("generateNext returned an object that is not a byte array of at least one byte");
    }
    piece = obj->bytes;
    return 1;
}

std::string api_string_call(const char * name, jlong h) {
    std::string text;
    jni_call(name, [&](JNIEnv * env) {
        jstring s = nullptr;
        if (strcmp(name, "stats") == 0) {
            s = Java_ai_airi_qwenmobile_LlamaNative_stats(env, native_class(env), h);
        } else if (strcmp(name, "modelInfo") == 0) {
            s = Java_ai_airi_qwenmobile_LlamaNative_modelInfo(env, native_class(env), h);
        } else {
            s = Java_ai_airi_qwenmobile_LlamaNative_devices(env, native_class(env));
        }
        fakejni::Obj * o = s != nullptr ? fakejni::object_of(env, s, name) : nullptr;
        if (o != nullptr) {
            text = fakejni::to_modified_utf8(o->chars);
        }
    });
    return text;
}

bool api_has_mtp(jlong h) {
    bool r = false;
    jni_call("hasMtp", [&](JNIEnv * env) { r = Java_ai_airi_qwenmobile_LlamaNative_hasMtp(env, native_class(env), h) == JNI_TRUE; });
    return r;
}

void api_reset(jlong h) {
    jni_call("resetChat", [&](JNIEnv * env) { Java_ai_airi_qwenmobile_LlamaNative_resetChat(env, native_class(env), h); });
}

std::string api_bench(jlong h, int pp, int tg, int reps) {
    std::string text;
    jni_call("bench", [&](JNIEnv * env) {
        jstring s = Java_ai_airi_qwenmobile_LlamaNative_bench(env, native_class(env), h, pp, tg, reps);
        fakejni::Obj * o = s != nullptr ? fakejni::object_of(env, s, "bench") : nullptr;
        if (o != nullptr) {
            text = fakejni::to_modified_utf8(o->chars);
        }
    });
    return text;
}

// --- The images. ---

/** A 24-bit BMP of w x h pixels with a pattern from the seed. stb_image decodes it. */
std::vector<uint8_t> make_bmp(int w, int h, uint32_t seed) {
    const int row = (w * 3 + 3) & ~3;
    const uint32_t data = (uint32_t) (row * h);
    std::vector<uint8_t> out(54 + data, 0);
    auto put32 = [&](size_t at, uint32_t v) { memcpy(out.data() + at, &v, 4); };
    auto put16 = [&](size_t at, uint16_t v) { memcpy(out.data() + at, &v, 2); };
    out[0] = 'B';
    out[1] = 'M';
    put32(2, (uint32_t) out.size());
    put32(10, 54);
    put32(14, 40);
    put32(18, (uint32_t) w);
    put32(22, (uint32_t) h);
    put16(26, 1);
    put16(28, 24);
    put32(34, data);
    std::mt19937 rng(seed);
    for (uint32_t i = 54; i < out.size(); ++i) {
        out[i] = (uint8_t) rng();
    }
    return out;
}

/** The next byte of the behavior tape of decodeImage. */
uint8_t tape_next(Program & p) {
    if (p.tape.empty()) {
        return 0;
    }
    return p.tape[p.tape_pos++ % p.tape.size()];
}

/**
 * LlamaNative.decodeImage as the fuzzer wants it: a correct decode, no
 * decode (the engine then uses stb_image), an exception, a layout that
 * does not agree with the buffer, or an object that is not a direct buffer.
 */
jobject java_decode_image(JNIEnv * env, va_list args) {
    jobject image = va_arg(args, jobject);
    const jint max_tokens = va_arg(args, jint);
    jobject dims = va_arg(args, jobject);
    fakejni::Obj * img = fakejni::object_of(env, image, "decodeImage bytes");
    fakejni::Obj * d   = fakejni::object_of(env, dims, "decodeImage dims");
    if (img == nullptr || img->kind != fakejni::Kind::ByteArray || d == nullptr || d->kind != fakejni::Kind::IntArray ||
        d->ints.size() != 3) {
        fail("decodeImage: the engine broke the contract of LlamaNative.decodeImage (bytes, dims[3])");
    }
    if (max_tokens < kImageTokensMin || max_tokens > kImageTokensMax) {
        fail("decodeImage: the token limit %d is out of the range of the engine", max_tokens);
    }
    // The tape can give two sizes for the same bytes, as a decode that fails one time and not the next.
    Program * p = g_prog;
    const uint8_t mode = p != nullptr ? tape_next(*p) : 0;
    const int     w    = 1 + (p != nullptr ? tape_next(*p) : 31) % 96;
    const int     h    = 1 + (p != nullptr ? tape_next(*p) : 31) % 96;
    std::vector<uint8_t> px;
    switch (mode % 10) {
        case 0: case 1: case 2: case 3: {
            const int stride = w * 4 + (mode & 1) * 16;
            px.assign((size_t) stride * h, (uint8_t) mode);
            d->ints = {w, h, stride};
            return fakejni::new_local(env, fakejni::new_direct_buffer(px.data(), px.size()), true);
        }
        case 4:
            return nullptr;
        case 5:
            fakejni::throw_exception(env, "java/lang/IllegalArgumentException", "the platform decoder refused the image");
            return nullptr;
        case 6: {
            // The stride and the capacity are one row short.
            px.assign((size_t) w * 4 * (h - 1 > 0 ? h - 1 : 1), 7);
            d->ints = {w, h, w * 4 - 1};
            return fakejni::new_local(env, fakejni::new_direct_buffer(px.data(), px.size()), true);
        }
        case 7:
            px.assign(64, 9);
            d->ints = {w, h, w * 4};
            return fakejni::new_local(env, fakejni::new_byte_array(px.data(), px.size()), true);
        case 8:
            px.assign(64, 3);
            d->ints = {(1 << 29) + 1, 1, 4};
            return fakejni::new_local(env, fakejni::new_direct_buffer(px.data(), px.size()), true);
        default:
            px.assign(16, 1);
            d->ints = {-w, h, 0};
            return fakejni::new_local(env, fakejni::new_direct_buffer(px.data(), px.size()), true);
    }
}

// --- The program input. ---

const char16_t * const kWords[] = {
    u"the", u"cat", u"What is in the picture?", u"Привет", u"日本語", u"\U0001F600", u"\U0001F468‍\U0001F469",
    u"<|im_start|>", u"<|im_end|>", u"<think>", u"</think>", u"<__media__>", u"<|vision_start|>", u"<|image_pad|>",
    u"<tool_call>", u"</tool_call>", u"<|endoftext|>", u"{{ messages }}", u"{% if %}", u"\\n", u"é́",
    u"<tools>", u"<function=x>", u"</function>", u"<parameter=y>", u"assistant", u"user\n", u"‮", u"﻿",
};

std::u16string gen_text(FuzzedDataProvider & fdp, size_t max_units) {
    const size_t n = fdp.ConsumeIntegralInRange<size_t>(0, max_units);
    std::u16string s;
    while (s.size() < n && fdp.remaining_bytes() > 0) {
        const uint8_t k = fdp.ConsumeIntegral<uint8_t>();
        switch (k % 16) {
            case 0: case 1: case 2: case 3: case 4:
                s.push_back((char16_t) ('a' + k % 26));
                break;
            case 5:
                s.push_back(u' ');
                break;
            case 6: case 7:
                s += kWords[fdp.ConsumeIntegralInRange<size_t>(0, sizeof(kWords) / sizeof(kWords[0]) - 1)];
                break;
            case 8:
                s.push_back((char16_t) fdp.ConsumeIntegral<uint16_t>());
                break;
            case 9:
                s.push_back(u'\0');
                break;
            case 10:
                s.push_back(u'\n');
                break;
            case 11:
                s.push_back((char16_t) (0xD800 + k));
                break;
            default:
                s.push_back((char16_t) (0x20 + k % 95));
                break;
        }
    }
    return s;
}

std::string pick_role(FuzzedDataProvider & fdp) {
    static const char * const roles[] = {"user", "user", "user", "assistant", "assistant", "system", "tool", "", "developer", "USER"};
    return roles[fdp.ConsumeIntegralInRange<size_t>(0, sizeof(roles) / sizeof(roles[0]) - 1)];
}

/**
 * A temperature or a top-p: the usual value, the edges, and values out of
 * range. NaN and the infinities go to the check of rebuild_sampler: NaN
 * logits stop llama_sampler_dist_apply.
 */
float pick_float(FuzzedDataProvider & fdp, float usual) {
    switch (fdp.ConsumeIntegralInRange<int>(0, 10)) {
        case 0: return 0.0f;
        case 1: return -1.0f;
        case 2: return std::numeric_limits<float>::quiet_NaN();
        case 3: return std::numeric_limits<float>::infinity();
        case 10: return -std::numeric_limits<float>::infinity();
        case 4: return 1e30f;
        case 5: return fdp.ConsumeFloatingPointInRange<float>(0.0f, 2.0f);
        default: return usual;
    }
}

LoadSpec gen_load(Program & p) {
    FuzzedDataProvider & fdp = *p.fdp;
    const Options & opt = *p.opt;
    LoadSpec s;
    const std::string f32  = opt.model_dir + "/tiny-qwen35-f32.gguf";
    const std::string q8   = opt.model_dir + "/tiny-qwen35-q8_0.gguf";
    const std::string mmp  = opt.model_dir + "/tiny-qwen35-mmproj.gguf";
    // Each parameter is valid in most loads: an invalid one ends the load, and the turns after it test nothing.
    const bool real_only = !opt.real_model.empty() && getenv("FUZZ_APP_REAL_ONLY") != nullptr;
    switch (fdp.ConsumeIntegralInRange<int>(0, 19)) {
        case 0: s.model = p.dir + "/missing.gguf"; break;
        case 1: s.model = mmp; break;
        case 2: s.model = p.dir + "/garbage.gguf"; break;
        case 3: case 4: case 5: case 6: s.model = q8; break;
        case 7: case 8: s.model = opt.real_model.empty() ? f32 : opt.real_model; break;
        default: s.model = f32; break;
    }
    const bool real = !opt.real_model.empty() && s.model == opt.real_model;
    switch (fdp.ConsumeIntegralInRange<int>(0, 15)) {
        case 0: case 1: s.mmproj = ""; break;
        case 2: s.mmproj = p.dir + "/missing-mmproj.gguf"; break;
        case 3: s.mmproj = f32; break;
        default: s.mmproj = real && !opt.real_mmproj.empty() ? opt.real_mmproj : mmp; break;
    }
    if (real && opt.real_mmproj.empty() && s.mmproj == mmp) {
        s.mmproj = "";
    }
    // The device of the options (HTP0 on the phone) takes most loads when it is set.
    auto dev = [&](int p_dev, int p_cpu, int p_bad) {
        const int k = fdp.ConsumeIntegralInRange<int>(0, 99);
        if (k < p_bad) return std::string("NoSuchDevice");
        if (k < p_bad + p_cpu) return std::string("CPU");
        if (k < p_bad + p_cpu + p_dev) return opt.device;
        return std::string();
    };
    const bool has_dev = !opt.device.empty();
    s.device  = dev(has_dev ? 60 : 0, 10, 3);
    s.prefill = dev(has_dev ? 8 : 0, 8, 2);
    s.vision  = dev(has_dev ? 30 : 0, 20, 3);
    s.gpu_layers = s.device.empty() || s.device == "CPU" ? 0 : 999;
    s.threads = fdp.ConsumeIntegralInRange<int>(0, 15) == 0 ? -1 : fdp.ConsumeIntegralInRange<int>(1, 4);
    static const int ctxs[] = {1024, 1024, 2048, 2048, 4096, 1023, 512, 2048};
    s.n_ctx = ctxs[fdp.ConsumeIntegralInRange<int>(0, 15) == 0 ? fdp.ConsumeIntegralInRange<int>(5, 6)
                                                             : fdp.ConsumeIntegralInRange<int>(0, 4)];
    static const int toks[] = {64, 256, 576, 768, 63, 769};
    s.image_max_tokens = toks[fdp.ConsumeIntegralInRange<int>(0, 15) == 0 ? fdp.ConsumeIntegralInRange<int>(4, 5)
                                                                        : fdp.ConsumeIntegralInRange<int>(0, 3)];
    s.speculative = fdp.ConsumeBool();
    switch (fdp.ConsumeIntegralInRange<int>(0, 3)) {
        case 0: s.cache = ""; break;
        case 1: s.cache = p.cache_b; break;
        default: s.cache = p.cache_a; break;
    }
    if (real_only) {
        // The rule of the phone for a real model: the Hexagon backend copies the
        // weights into rpcmem, which the kernel cannot reclaim. Thus the process
        // loads the real model one time, with the shortest context of the engine,
        // no prefill model, and a projector only when FUZZ_APP_REAL_MMPROJ names
        // one. The later loads of the process take the tiny model.
        static std::atomic<int> real_loads{0};
        if (real_loads.fetch_add(1) == 0) {
            s.model   = opt.real_model;
            s.mmproj  = opt.real_mmproj;
            s.device  = s.device == "NoSuchDevice" ? opt.device : s.device;
            s.prefill = "";
            s.vision  = s.vision == "NoSuchDevice" ? std::string() : s.vision;
            s.n_ctx   = 1024;
        } else {
            s.model  = f32;
            s.mmproj = mmp;
        }
    }
    return s;
}

// --- The checks. ---

/** The white-box invariants of one engine. The caller holds no lock of the engine. */
void check_engine(jlong h, const char * after) {
    const std::shared_ptr<Engine> sp = engine_of(h);
    if (!sp) {
        return;
    }
    Engine & e = *sp;
    std::lock_guard<std::mutex> lock(e.mutex);
    if (e.out_next > e.out_queue.size()) {
        fail("after %s: out_next %zu is past the queue of %zu tokens", after, e.out_next, e.out_queue.size());
    }
    const uint32_t n_ctx = llama_n_ctx(e.ctx);
    if (e.n_past < 0 || (uint32_t) e.n_past > n_ctx) {
        fail("after %s: n_past %d is out of [0, %u]", after, (int) e.n_past, n_ctx);
    }
    auto memory_agrees = [&](llama_context * ctx, const std::vector<MemItem> & items, llama_pos n_past, const char * what) {
        const llama_pos pmax = llama_memory_seq_pos_max(llama_get_memory(ctx), kSeqMain);
        if (items.empty()) {
            if (n_past != 0 || pmax != -1) {
                fail("after %s: the %s holds no items but n_past is %d and the last position is %d", after, what,
                     (int) n_past, (int) pmax);
            }
            return;
        }
        bool text_only = true;
        for (const MemItem & it : items) {
            text_only = text_only && it.token != kMemTokenNull;
        }
        if (text_only && (size_t) n_past != items.size()) {
            fail("after %s: the %s holds %zu text items at %d positions", after, what, items.size(), (int) n_past);
        }
        if (items.back().token != kMemTokenNull && pmax != n_past - 1) {
            fail("after %s: the %s must end at position %d, its memory ends at %d (%zu items)", after, what,
                 (int) n_past - 1, (int) pmax, items.size());
        }
    };
    memory_agrees(e.ctx, e.cache, e.n_past, "decode context");
    if (e.ctx_pf != nullptr) {
        memory_agrees(e.ctx_pf, e.pf_cache, e.pf_n_past, "prefill context");
    }
    if (e.spec != nullptr) {
        std::vector<llama_token> text;
        for (const MemItem & it : e.cache) {
            if (it.token != kMemTokenNull) {
                text.push_back(it.token);
            }
        }
        if (!e.cache.empty() && text != e.spec_prompt) {
            fail("after %s: the draft prompt (%zu tokens) is not the text of the memory (%zu tokens)", after,
                 e.spec_prompt.size(), text.size());
        }
        const llama_pos dmax = llama_memory_seq_pos_max(llama_get_memory(e.ctx_dft), kSeqMain);
        if (dmax >= e.n_past && e.n_past > 0) {
            fail("after %s: the draft context holds position %d, at or after n_past %d", after, (int) dmax, (int) e.n_past);
        }
    }
}

/**
 * The tokens that generateNext gave to the app must be the tokens of the
 * answer in the memory, in order: the memory holds each token that the app
 * received, and the last token of a step (id_last) until the next
 * step decodes it. The end token is in the memory and does not go to the
 * app. The check holds while the memory holds the prompt of the turn; a
 * benchmark or a reset ends the turn. O(answer).
 */
void check_delivered(LiveEngine & le, const char * after) {
    if (!le.tracking) {
        return;
    }
    const std::shared_ptr<Engine> sp = engine_of(le.handle);
    if (!sp) {
        le.tracking = false;
        return;
    }
    Engine & e = *sp;
    std::lock_guard<std::mutex> lock(e.mutex);
    // After the end of an answer the memory can hold tokens of a step that
    // the app did not receive, by design: no later turn reads them.
    if (e.cache.size() < le.prompt_items || e.answer_done) {
        le.tracking = false;
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(e.model);
    std::vector<llama_token> memory;
    for (size_t i = le.prompt_items; i < e.cache.size(); ++i) {
        if (!llama_vocab_is_eog(vocab, e.cache[i].token)) {
            memory.push_back(e.cache[i].token);
        }
    }
    if (e.id_last != LLAMA_TOKEN_NULL) {
        memory.push_back(e.id_last);
    }
    const bool prefix = le.delivered.size() <= memory.size() &&
                        std::equal(le.delivered.begin(), le.delivered.end(), memory.begin());
    if (!prefix) {
        size_t at = 0;
        while (at < le.delivered.size() && at < memory.size() && le.delivered[at] == memory[at]) {
            ++at;
        }
        fail("after %s: the app received %zu tokens of the answer, the memory holds %zu (id_last %d, draft driver %s); "
             "they differ at token %zu: the app got %d, the memory holds %d",
             after, le.delivered.size(), memory.size(), e.id_last, e.spec ? "on" : "off", at,
             at < le.delivered.size() ? le.delivered[at] : -1, at < memory.size() ? memory[at] : -1);
    }
}

/** The count of tokens that the engine gave out in this turn (TurnStats::gen_tokens), or -1 without an engine. */
int64_t given_tokens(const LiveEngine & le) {
    if (const std::shared_ptr<Engine> sp = engine_of(le.handle)) {
        std::lock_guard<std::mutex> lock(sp->mutex);
        return sp->turn.gen_tokens;
    }
    return -1;
}

/**
 * Record the token that the last generateNext took from the queue, when it
 * took one: the count of the turn went up. A call that took a token and then
 * did not return its piece (an OutOfMemoryError from NewByteArray) took it all
 * the same, and a call that gave only the close of the thinking took none.
 */
void note_delivered(LiveEngine & le, int64_t given_before) {
    if (!le.tracking) {
        return;
    }
    if (const std::shared_ptr<Engine> sp = engine_of(le.handle)) {
        std::lock_guard<std::mutex> lock(sp->mutex);
        if (sp->turn.gen_tokens > given_before && sp->out_next > 0 && sp->out_next <= sp->out_queue.size()) {
            le.delivered.push_back(sp->out_queue[sp->out_next - 1]);
        }
    }
}

/** Start the record of a turn after a chatStart that decoded its prompt. */
void start_tracking(LiveEngine & le) {
    le.tracking = false;
    le.delivered.clear();
    if (const std::shared_ptr<Engine> sp = engine_of(le.handle)) {
        std::lock_guard<std::mutex> lock(sp->mutex);
        le.prompt_items = sp->cache.size();
        le.tracking     = !sp->answer_done;
    }
}

/**
 * The exactness oracle. Returns false with the reason. The oracle skips an
 * engine with an image in its memory, and an engine with no room for the probe.
 */
bool oracle(jlong h, std::string & why) {
    const std::shared_ptr<Engine> sp = engine_of(h);
    if (!sp) {
        return true;
    }
    Engine & e = *sp;
    std::lock_guard<std::mutex> lock(e.mutex);
    if (e.cache.empty() || e.n_past + 2 >= (llama_pos) llama_n_ctx(e.ctx)) {
        return true;
    }
    for (const MemItem & it : e.cache) {
        if (it.token == kMemTokenNull) {
            return true;
        }
    }
    g_counters.oracle_checks += 1;
    std::string err;
    const std::shared_ptr<const cache_io::Blob> blob = take_state(e.ctx, err);
    if (!blob) {
        why = "the state did not copy out: " + err;
        return false;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = llama_n_ctx(e.ctx);
    cp.n_batch    = kBatch;
    cp.n_ubatch   = kBatch;
    cp.n_seq_max  = 1;
    cp.kv_unified = true;
    cp.n_threads  = 1;
    cp.n_threads_batch = 1;
    cp.n_rs_seq   = llama_n_rs_seq(e.ctx);
    cp.no_perf    = true;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(e.model));
    const llama_token probe = e.cache.front().token;
    llama_batch b = llama_batch_init(kBatch, 0, 1);
    auto decode = [&](llama_context * ctx, const llama_token * toks, int n, llama_pos pos0, bool logits_last) {
        for (int i = 0; i < n; i += kBatch) {
            const int cnt = std::min(kBatch, n - i);
            for (int j = 0; j < cnt; ++j) {
                b.token[j] = toks[i + j];
                b.pos[j] = pos0 + i + j;
                b.n_seq_id[j] = 1;
                b.seq_id[j][0] = kSeqMain;
                b.logits[j] = logits_last && i + j == n - 1;
            }
            b.n_tokens = cnt;
            if (llama_decode(ctx, b) != 0) {
                return false;
            }
        }
        return true;
    };
    std::vector<float> lb, lc;
    bool ok = true;
    llama_context * ctx_b = llama_init_from_model(e.model, cp);
    if (ctx_b == nullptr || !restore_state(ctx_b, *blob) || !decode(ctx_b, &probe, 1, e.n_past, true)) {
        why = "the state of the engine did not restore into a new context, or the probe did not decode";
        ok = false;
    } else {
        const float * l = llama_get_logits_ith(ctx_b, -1);
        lb.assign(l, l + n_vocab);
    }
    llama_context * ctx_c = ok ? llama_init_from_model(e.model, cp) : nullptr;
    if (ok) {
        std::vector<llama_token> toks;
        for (const MemItem & it : e.cache) {
            toks.push_back(it.token);
        }
        if (ctx_c == nullptr || !decode(ctx_c, toks.data(), (int) toks.size(), 0, false) ||
            !decode(ctx_c, &probe, 1, e.n_past, true)) {
            why = "the reference decode failed";
            ok = false;
        } else {
            const float * l = llama_get_logits_ith(ctx_c, -1);
            lc.assign(l, l + n_vocab);
        }
    }
    if (ok) {
        float worst = 0.0f;
        int at = 0;
        for (int i = 0; i < n_vocab; ++i) {
            const float d = std::fabs(lb[(size_t) i] - lc[(size_t) i]);
            if (!(d <= worst)) {
                worst = d;
                at = i;
            }
        }
        // Batched and single-token decodes differ in rounding only. A wrong
        // rollback leaves a different state, which moves the logits by 0.1 and more.
        if (!(worst <= 2e-2f)) {
            why = "the logits after the state of the engine differ from a clean decode of its " +
                  std::to_string(e.cache.size()) + " tokens by " + std::to_string(worst) + " at token " +
                  std::to_string(at) + " (spec " + (e.spec ? "on" : "off") + ", n_rs_seq " + std::to_string(cp.n_rs_seq) + ")";
            ok = false;
        }
    }
    if (ctx_c != nullptr) llama_free(ctx_c);
    if (ctx_b != nullptr) llama_free(ctx_b);
    llama_batch_free(b);
    return ok;
}

void check_all(Program & p, const char * after, bool with_oracle) {
    for (LiveEngine & le : p.live) {
        check_engine(le.handle, after);
        // The oracle makes two contexts, thus it runs one time for each change of the memory.
        if (with_oracle && p.opt->oracle && le.dirty) {
            le.dirty = false;
            std::string why;
            if (!oracle(le.handle, why)) {
                fail("after %s: exactness oracle: %s", after, why.c_str());
            }
        }
    }
}

// --- The cache directories. ---

/** The regular files below a directory, recursively. */
void list_tree(const std::string & dir, std::vector<std::string> & out) {
    for (const std::string & d : cache_io::list_dirs(dir)) {
        list_tree(dir + "/" + d, out);
    }
    for (const std::string & f : cache_io::list_files(dir, "")) {
        out.push_back(dir + "/" + f);
    }
}

/** Change one file of the cache directories as a damaged disk, or the system, does. */
void corrupt_cache(Program & p) {
    FuzzedDataProvider & fdp = *p.fdp;
    std::vector<std::string> files;
    list_tree(p.cache_a, files);
    list_tree(p.cache_b, files);
    if (files.empty()) {
        return;
    }
    std::sort(files.begin(), files.end());
    const std::string & path = files[fdp.ConsumeIntegralInRange<size_t>(0, files.size() - 1)];
    std::vector<uint8_t> bytes;
    if (!cache_io::read_file(path, bytes, 64u << 20)) {
        return;
    }
    const int mode = fdp.ConsumeIntegralInRange<int>(0, 6);
    trace(*p.opt, "corrupt %s mode %d (%zu bytes)", path.c_str(), mode, bytes.size());
    switch (mode) {
        case 0:
            cache_io::remove_file(path);
            return;
        case 1:
            bytes.resize(fdp.ConsumeIntegralInRange<size_t>(0, bytes.size()));
            break;
        case 2: {
            // Flip bytes after the fixed header: the items of a snapshot or its state bytes, or the floats of an image.
            const int n = fdp.ConsumeIntegralInRange<int>(1, 16);
            for (int i = 0; i < n && bytes.size() > 40; ++i) {
                bytes[fdp.ConsumeIntegralInRange<size_t>(40, bytes.size() - 1)] ^= (uint8_t) (1 + fdp.ConsumeIntegral<uint8_t>() % 255);
            }
            break;
        }
        case 3: {
            const int n = fdp.ConsumeIntegralInRange<int>(1, 4);
            for (int i = 0; i < n && !bytes.empty(); ++i) {
                bytes[fdp.ConsumeIntegralInRange<size_t>(0, std::min<size_t>(39, bytes.size() - 1))] ^= fdp.ConsumeIntegral<uint8_t>();
            }
            break;
        }
        case 4: {
            // The encoder output of an image with a new shape of the same size: n_tokens and n_embd trade a factor.
            if (bytes.size() >= 32 && memcmp(bytes.data(), "QMIE", 4) == 0) {
                uint32_t nt = 0, ne = 0;
                memcpy(&nt, bytes.data() + 16, 4);
                memcpy(&ne, bytes.data() + 20, 4);
                if (ne % 2 == 0) {
                    nt *= 2;
                    ne /= 2;
                } else if (nt % 2 == 0) {
                    nt /= 2;
                    ne *= 2;
                }
                memcpy(bytes.data() + 16, &nt, 4);
                memcpy(bytes.data() + 20, &ne, 4);
            }
            break;
        }
        case 5: {
            std::vector<uint8_t> extra = fdp.ConsumeBytes<uint8_t>(fdp.ConsumeIntegralInRange<size_t>(1, 64));
            bytes.insert(bytes.end(), extra.begin(), extra.end());
            break;
        }
        default:
            bytes = fdp.ConsumeBytes<uint8_t>(fdp.ConsumeIntegralInRange<size_t>(0, 128));
            break;
    }
    cache_io::write_file_atomic(path, {{bytes.data(), bytes.size()}});
}

/**
 * Write an encoder output file for an image of the program before a load,
 * with a shape from the input. The load scans it, and a later turn with the
 * image uses it. A file of an earlier build of the app, or a damaged file,
 * has such a shape.
 */
void plant_image(Program & p) {
    FuzzedDataProvider & fdp = *p.fdp;
    if (p.images.empty()) {
        return;
    }
    const size_t k = fdp.ConsumeIntegralInRange<size_t>(0, p.images.size() - 1);
    LoadSpec s = gen_load(p);
    const std::string ns = cache_namespace(s.model, s.mmproj, s.image_max_tokens);
    const std::string dir = (fdp.ConsumeBool() ? p.cache_a : p.cache_b) + "/engine/" + ns + "/images";
    cache_io::make_dirs(dir);
    const std::string id = cache_io::sha256_hex(p.images[k].data(), p.images[k].size());
    const uint32_t fields[5] = {1, fdp.ConsumeIntegralInRange<uint32_t>(1, 128), fdp.ConsumeIntegralInRange<uint32_t>(1, 128),
                                fdp.ConsumeIntegralInRange<uint32_t>(1, 16), fdp.ConsumeBool() ? 64u : fdp.ConsumeIntegralInRange<uint32_t>(1, 128)};
    std::vector<uint8_t> file(32, 0);
    memcpy(file.data(), "QMIE", 4);
    memcpy(file.data() + 4, fields, sizeof(fields));
    file.resize(32 + (size_t) fields[3] * fields[4] * 4, 0);
    cache_io::write_file_atomic(dir + "/" + id + ".embd", {{file.data(), file.size()}});
    trace(*p.opt, "plant image %zu: %ux%u, %u tokens x %u", k, fields[1], fields[2], fields[3], fields[4]);
}

// --- The operations. ---

jlong pick_handle(Program & p) {
    FuzzedDataProvider & fdp = *p.fdp;
    const int k = fdp.ConsumeIntegralInRange<int>(0, 15);
    if (k < 13 && !p.live.empty()) {
        return p.live[(size_t) k % p.live.size()].handle;
    }
    if (k == 13 && !p.dead.empty()) {
        return p.dead.back();
    }
    return k == 14 ? 0 : (jlong) fdp.ConsumeIntegral<int32_t>();
}

LiveEngine * live_of(Program & p, jlong h) {
    for (LiveEngine & le : p.live) {
        if (le.handle == h) {
            return &le;
        }
    }
    return nullptr;
}

void op_load(Program & p) {
    // The app releases the engine before a load. Two engines at a time test the table with a second engine.
    if (p.live.size() >= 2 || (!p.live.empty() && p.fdp->ConsumeBool())) {
        const jlong h = p.live.front().handle;
        api_free(h);
        p.dead.push_back(h);
        p.live.erase(p.live.begin());
    }
    const LoadSpec s = gen_load(p);
    trace(*p.opt, "load %s mmproj=%s dev=%s pf=%s vis=%s ctx=%d img=%d spec=%d threads=%d cache=%s", s.model.c_str(),
          s.mmproj.c_str(), s.device.c_str(), s.prefill.c_str(), s.vision.c_str(), s.n_ctx, s.image_max_tokens,
          (int) s.speculative, s.threads, s.cache.c_str());
    const jlong h = api_load(s);
    if (h == 0) {
        g_counters.loads_failed += 1;
        return;
    }
    g_counters.loads_ok += 1;
    LiveEngine le;
    le.handle = h;
    le.spec   = s;
    p.live.push_back(std::move(le));
    p.current = h;
}

void op_chat(Program & p) {
    FuzzedDataProvider & fdp = *p.fdp;
    const jlong h = pick_handle(p);
    LiveEngine * le = live_of(p, h);
    std::vector<Msg> msgs;
    const int mode = fdp.ConsumeIntegralInRange<int>(0, 5);
    if (le != nullptr && !le->convo.empty() && mode <= 2) {
        // The app sends the history with the answer and a new message.
        msgs = le->convo;
        if (!le->answer.empty() || fdp.ConsumeBool()) {
            // The app shows the answer as a String, thus a byte that is not UTF-8 becomes U+FFFD.
            msgs.push_back(Msg{"assistant", fakejni::new_string_utf8(le->answer)->chars, -1});
        }
        if (mode != 2) {
            Msg m{"user", gen_text(fdp, 48), -1};
            if (!p.images.empty() && fdp.ConsumeIntegralInRange<int>(0, 3) == 0) {
                m.image = fdp.ConsumeIntegralInRange<int>(0, (int) p.images.size() - 1);
            }
            msgs.push_back(std::move(m));
        }
    } else if (le != nullptr && !le->convo.empty() && mode == 3) {
        msgs.assign(le->convo.begin(), le->convo.begin() + (ptrdiff_t) fdp.ConsumeIntegralInRange<size_t>(1, le->convo.size()));
    } else if (fdp.ConsumeIntegralInRange<int>(0, 7) != 0) {
        // A conversation as the app makes it: a system message or none, pairs of turns, and a user message last.
        auto user = [&](size_t max_units) {
            Msg m{"user", gen_text(fdp, max_units), -1};
            if (!p.images.empty() && fdp.ConsumeIntegralInRange<int>(0, 3) == 0) {
                m.image = fdp.ConsumeIntegralInRange<int>(0, (int) p.images.size() - 1);
            }
            return m;
        };
        if (fdp.ConsumeIntegralInRange<int>(0, 4) == 0) {
            msgs.push_back(Msg{"system", gen_text(fdp, 64), -1});
        }
        const int turns = fdp.ConsumeIntegralInRange<int>(0, 3);
        for (int i = 0; i < turns; ++i) {
            msgs.push_back(user(48));
            msgs.push_back(Msg{"assistant", gen_text(fdp, 48), -1});
        }
        msgs.push_back(user(fdp.ConsumeIntegralInRange<int>(0, 15) == 0 ? 3000 : 64));
    } else {
        const int n = fdp.ConsumeIntegralInRange<int>(0, 5);
        for (int i = 0; i < n; ++i) {
            Msg m{pick_role(fdp), gen_text(fdp, fdp.ConsumeIntegralInRange<int>(0, 15) == 0 ? 3000 : 64), -1};
            if (!p.images.empty() && fdp.ConsumeIntegralInRange<int>(0, 3) == 0) {
                m.image = fdp.ConsumeIntegralInRange<int>(0, (int) p.images.size() - 1);
            }
            msgs.push_back(std::move(m));
        }
    }
    const bool thinking = fdp.ConsumeBool();
    const float temp    = pick_float(fdp, 0.7f);
    const float top_p   = pick_float(fdp, 0.8f);
    const int   shape   = fdp.ConsumeIntegralInRange<int>(0, 31);
    for (const Msg & m : msgs) {
        g_counters.images += m.image >= 0 ? 1 : 0;
    }
    trace(*p.opt, "chatStart h=%lld %zu messages thinking=%d temp=%g top_p=%g", (long long) h, msgs.size(), (int) thinking,
          temp, top_p);
    g_counters.turns += 1;
    // chatStart resets the counters of the turn, thus the draft counts of the last turn go into the totals here.
    if (const std::shared_ptr<Engine> e = engine_of(h)) {
        g_counters.drafted  += (uint64_t) e->turn.drafted;
        g_counters.accepted += (uint64_t) e->turn.accepted;
    }
    const int rc = api_chat_start(p, h, msgs, thinking, temp, top_p, shape == 0, shape == 1);
    if (le != nullptr) {
        le->convo       = msgs;
        le->answer.clear();
        le->answer_open = rc >= 0;
        le->dirty       = true;
        le->tracking    = false;
        if (rc >= 0) {
            start_tracking(*le);
        }
    }
    check_all(p, "chatStart", false);
}

/**
 * The stage of an attached image, as the app calls it before the send: the
 * history of the engine with its answer, or a new chat, and a user message
 * with an image of the program (or without one, which stages nothing).
 */
void op_stage(Program & p) {
    FuzzedDataProvider & fdp = *p.fdp;
    const jlong h = pick_handle(p);
    LiveEngine * le = live_of(p, h);
    std::vector<Msg> msgs;
    if (le != nullptr && !le->convo.empty() && fdp.ConsumeIntegralInRange<int>(0, 3) != 0) {
        msgs = le->convo;
        if (!le->answer.empty() || fdp.ConsumeBool()) {
            msgs.push_back(Msg{"assistant", fakejni::new_string_utf8(le->answer)->chars, -1});
        }
    }
    Msg m{"user", gen_text(fdp, 32), -1};
    if (!p.images.empty() && fdp.ConsumeIntegralInRange<int>(0, 7) != 0) {
        m.image = fdp.ConsumeIntegralInRange<int>(0, (int) p.images.size() - 1);
    }
    msgs.push_back(std::move(m));
    const bool thinking = fdp.ConsumeBool();
    const int  rc       = api_stage_image(p, h, msgs, thinking);
    trace(*p.opt, "stageImage h=%lld %zu messages thinking=%d: %d", (long long) h, msgs.size(), (int) thinking, rc);
    if (le != nullptr) {
        // The stage ends an answer that the app did not read to its end, and it changes the memory.
        le->dirty    = true;
        le->tracking = false;
    }
    check_all(p, "stageImage", false);
}

void op_generate(Program & p) {
    FuzzedDataProvider & fdp = *p.fdp;
    const jlong h = pick_handle(p);
    LiveEngine * le = live_of(p, h);
    if (le != nullptr) {
        le->dirty = true;
    }
    const int n = fdp.ConsumeIntegralInRange<int>(1, p.opt->max_gen);
    const int stop_at = fdp.ConsumeBool() ? -1 : fdp.ConsumeIntegralInRange<int>(0, n);
    bool ended = false;
    int nulls = 0;
    for (int i = 0; i < n; ++i) {
        if (i == stop_at) {
            g_counters.stops += 1;
            api_request_stop(h);
        }
        std::vector<int8_t> piece;
        const int64_t given = le != nullptr ? given_tokens(*le) : -1;
        const int r = api_generate_next(h, piece);
        check_engine(h, "one generateNext");
        if (le != nullptr) {
            note_delivered(*le, given);
            check_delivered(*le, "one generateNext");
        }
        if (r == 1) {
            g_counters.tokens += 1;
            if (nulls > 0) {
                fail("generateNext gave a piece after it gave null, with no chatStart between");
            }
            if (piece[0] < 0 || piece[0] > 2) {
                fail("generateNext gave the token kind %d", piece[0]);
            }
            if (le != nullptr) {
                le->answer.append(reinterpret_cast<const char *>(piece.data() + 1), piece.size() - 1);
            }
            continue;
        }
        ended = true;
        nulls += r == 0 ? 1 : 0;
        if (r == 0 && nulls >= 2) {
            break;
        }
        if (r < 0) {
            break;
        }
    }
    trace(*p.opt, "generateNext h=%lld: %d calls, stop at %d, ended %d", (long long) h, n, stop_at, (int) ended);
    check_all(p, "generateNext", ended);
}

void op_bench(Program & p) {
    FuzzedDataProvider & fdp = *p.fdp;
    static const int pps[] = {0, 1, 7, 64, 1100, -5};
    static const int tgs[] = {0, 1, 3, -1};
    const jlong h = pick_handle(p);
    const int pp = pps[fdp.ConsumeIntegralInRange<int>(0, 5)];
    const int tg = tgs[fdp.ConsumeIntegralInRange<int>(0, 3)];
    const int reps = fdp.ConsumeIntegralInRange<int>(-1, 2);
    trace(*p.opt, "bench h=%lld pp=%d tg=%d reps=%d", (long long) h, pp, tg, reps);
    api_bench(h, pp, tg, reps);
    check_all(p, "bench", false);
}

void op_free(Program & p) {
    FuzzedDataProvider & fdp = *p.fdp;
    if (!p.live.empty() && fdp.ConsumeIntegralInRange<int>(0, 3) != 0) {
        const size_t k = fdp.ConsumeIntegralInRange<size_t>(0, p.live.size() - 1);
        const jlong h = p.live[k].handle;
        api_free(h);
        p.dead.push_back(h);
        p.live.erase(p.live.begin() + (ptrdiff_t) k);
        return;
    }
    api_free(pick_handle(p));
}

/** Run the stop requests and the frees of the second thread, one schedule for each program. */
void ui_thread(Program * p, std::vector<uint8_t> schedule, std::atomic<bool> * done) {
    size_t i = 0;
    while (!done->load() && !schedule.empty()) {
        const uint8_t a = schedule[i++ % schedule.size()];
        std::this_thread::sleep_for(std::chrono::microseconds(50 + (a >> 2) * 40));
        const jlong h = p->current.load();
        switch (a & 3) {
            case 0: case 1:
                api_request_stop(h);
                break;
            case 2:
                api_has_mtp(h);
                break;
            default:
                if (p->opt->cross_free) {
                    api_free(h);
                } else {
                    api_request_stop(h + 1000);
                }
                break;
        }
    }
}

}  // namespace

Options options_from_env() {
    Options o;
    auto str = [](const char * k, const char * d) {
        const char * v = getenv(k);
        return std::string(v != nullptr ? v : d);
    };
    auto num = [](const char * k, int d) {
        const char * v = getenv(k);
        return v != nullptr ? atoi(v) : d;
    };
    o.model_dir   = str("FUZZ_APP_MODEL_DIR", FUZZ_APP_DEFAULT_MODEL_DIR);
    o.real_model  = str("FUZZ_APP_REAL_MODEL", "");
    o.real_mmproj = str("FUZZ_APP_REAL_MMPROJ", "");
    o.device      = str("FUZZ_APP_DEVICE", "");
    o.work_dir    = str("FUZZ_APP_WORK", "/tmp/fuzz-app-work");
    o.oracle      = num("FUZZ_APP_ORACLE", 1) != 0;
    o.threads     = num("FUZZ_APP_THREADS", 0) != 0;
    o.cross_free  = num("FUZZ_APP_CROSS_FREE", 0) != 0;
    o.max_ops     = num("FUZZ_APP_MAX_OPS", 48);
    o.max_gen     = num("FUZZ_APP_MAX_GEN", 96);
    o.trace       = num("FUZZ_APP_TRACE", 0) != 0;
    return o;
}

void init_once(const Options & opt) {
    static std::once_flag once;
    std::call_once(once, [&] {
        g_engine_thread = std::this_thread::get_id();
        cache_io::make_dirs(opt.work_dir);
        fakejni::register_static(kNativeClass, "decodeImage", "([BI[I)Ljava/nio/ByteBuffer;", java_decode_image);
        const char * libdir = getenv("FUZZ_APP_LIBDIR");
        jni_call("init", [&](JNIEnv * env) {
            Java_ai_airi_qwenmobile_LlamaNative_init(env, native_class(env), jstr(env, libdir ? libdir : "", true));
        });
        const std::string devices = api_string_call("devices", 0);
        trace(opt, "devices:\n%s", devices.c_str());
        atexit([] {
            const char * path = getenv("FUZZ_APP_EXC_LOG");
            if (path == nullptr) {
                return;
            }
            FILE * f = fopen(path, "a");
            if (f == nullptr) {
                return;
            }
            std::lock_guard<std::mutex> lock(g_exc_mutex);
            for (const auto & kv : g_exceptions) {
                fprintf(f, "%8llu %s\n", (unsigned long long) kv.second, kv.first.c_str());
            }
            fprintf(f,
                    "engines destroyed off the engine thread: %llu in a free, %llu in another call, draft follow faults: %llu\n",
                    (unsigned long long) g_off_thread_frees.load(), (unsigned long long) g_off_thread_other.load(),
                    (unsigned long long) g_spec_faults.load());
            fclose(f);
        });
    });
}

const Counters & counters() {
    return g_counters;
}

void print_summary() {
    const Counters & c = g_counters;
    fprintf(stderr,
            "summary: %llu programs, %llu ops, %llu loads (%llu failed), %llu turns, %llu tokens, %llu stops, "
            "%llu images, %llu oracle checks, %llu Java exceptions, %llu JNI faults, %llu draft faults, "
            "drafted %llu accepted %llu, engines destroyed off the engine thread: %llu in a free, %llu in another call\n",
            (unsigned long long) c.programs, (unsigned long long) c.ops, (unsigned long long) (c.loads_ok + c.loads_failed),
            (unsigned long long) c.loads_failed, (unsigned long long) c.turns, (unsigned long long) c.tokens,
            (unsigned long long) c.stops, (unsigned long long) c.images, (unsigned long long) c.oracle_checks,
            (unsigned long long) c.java_exceptions, (unsigned long long) c.faults,
            (unsigned long long) g_spec_faults.load(), (unsigned long long) c.drafted, (unsigned long long) c.accepted,
            (unsigned long long) g_off_thread_frees.load(), (unsigned long long) g_off_thread_other.load());
    std::lock_guard<std::mutex> lock(g_exc_mutex);
    for (const auto & kv : g_exceptions) {
        fprintf(stderr, "  %8llu %s\n", (unsigned long long) kv.second, kv.first.c_str());
    }
}

namespace {

/** Write an encoder output file for an image, before a load, with the shape of the arguments. */
void write_embd(const std::string & dir, const std::vector<uint8_t> & image, uint32_t nx, uint32_t ny, uint32_t n_tokens,
                uint32_t n_embd) {
    cache_io::make_dirs(dir);
    const uint32_t fields[5] = {1, nx, ny, n_tokens, n_embd};
    std::vector<uint8_t> file(32, 0);
    memcpy(file.data(), "QMIE", 4);
    memcpy(file.data() + 4, fields, sizeof(fields));
    file.resize(32 + (size_t) n_tokens * n_embd * 4, 0);
    const std::string id = cache_io::sha256_hex(image.data(), image.size());
    cache_io::write_file_atomic(dir + "/" + id + ".embd", {{file.data(), file.size()}});
}

/**
 * Generate pieces until null, an exception, or the limit, with the checks
 * after each one. With text, the bytes of the pieces go there. Returns the
 * count of pieces.
 */
int drain_answer(LiveEngine & le, int limit, std::string * text = nullptr) {
    int n = 0;
    for (; n < limit; ++n) {
        std::vector<int8_t> piece;
        const int64_t given = given_tokens(le);
        const int r = api_generate_next(le.handle, piece);
        check_engine(le.handle, "one generateNext of a scenario");
        note_delivered(le, given);
        check_delivered(le, "one generateNext of a scenario");
        if (r != 1) {
            break;
        }
        if (text != nullptr) {
            text->append(reinterpret_cast<const char *>(piece.data()), piece.size());
        }
    }
    return n;
}

/**
 * The greedy answer of the tiny model to one message, as bytes. With
 * speculative, the draft follow fails at the fault_at-th call after the
 * first piece (-1 for no fault). O(limit) generateNext calls and one load.
 */
std::string greedy_answer(Program & p, LoadSpec s, bool speculative, int fault_at, int limit) {
    s.speculative = speculative;
    const jlong h = api_load(s);
    if (h == 0) {
        fail("scenario spec-parity: the tiny model did not load");
    }
    const std::vector<Msg> chat = {Msg{"user", u"Hello, how are you? Tell me about the cat.", -1}};
    if (api_chat_start(p, h, chat, false, 0.0f, 0.8f, false, false) < 0) {
        fail("scenario spec-parity: chatStart failed");
    }
    LiveEngine le;
    le.handle = h;
    start_tracking(le);
    std::string text;
    drain_answer(le, 1, &text);
    g_spec_countdown = fault_at;
    drain_answer(le, limit - 1, &text);
    g_spec_countdown = -1;
    api_free(h);
    return text;
}

/**
 * The greedy answer of one load to a conversation, as bytes: load, chatStart,
 * up to limit pieces, free. The free waits for the queued file writes of the
 * caches. O(limit) generateNext calls and one load.
 */
std::string answer_of(Program & p, const LoadSpec & s, const std::vector<Msg> & chat, int limit, const char * what) {
    const jlong h = api_load(s);
    if (h == 0) {
        fail("%s: the tiny model did not load", what);
    }
    if (api_chat_start(p, h, chat, false, 0.0f, 0.8f, false, false) < 0) {
        fail("%s: chatStart failed", what);
    }
    LiveEngine le;
    le.handle = h;
    start_tracking(le);
    std::string text;
    drain_answer(le, limit, &text);
    api_free(h);
    return text;
}

/**
 * Write 0xFF over 256 bytes in the last quarter of each file of the tree whose
 * name ends with suffix, and over its last 256 bytes. A float of 0xFFFFFFFF is
 * NaN. The headers stay intact, thus only a check of the data finds the
 * damage. Returns the number of files.
 */
int damage_data(const std::string & dir, const std::string & suffix) {
    std::vector<std::string> files;
    list_tree(dir, files);
    int n = 0;
    for (const std::string & path : files) {
        if (path.size() < suffix.size() || path.compare(path.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        std::vector<uint8_t> bytes;
        if (!cache_io::read_file(path, bytes, 256u << 20) || bytes.size() < 1024) {
            continue;
        }
        const size_t mid = bytes.size() * 3 / 4;
        std::fill(bytes.begin() + (ptrdiff_t) mid, bytes.begin() + (ptrdiff_t) (mid + 256), 0xFF);
        std::fill(bytes.end() - 256, bytes.end(), 0xFF);
        cache_io::write_file_atomic(path, {{bytes.data(), bytes.size()}});
        n += 1;
    }
    return n;
}

/**
 * The render of the chat template with the empty think block after each
 * "<|im_start|>assistant\n" that has no think block: the live form of the
 * Qwen3.5 template, made without chat_prompt.cpp. O(length).
 */
std::string empty_blocks_in(const std::string & stock) {
    static const std::string open  = "<|im_start|>assistant\n";
    static const std::string block = "<think>\n\n</think>\n\n";
    std::string out;
    size_t at = 0;
    for (;;) {
        const size_t k = stock.find(open, at);
        if (k == std::string::npos) {
            out += stock.substr(at);
            return out;
        }
        out += stock.substr(at, k + open.size() - at);
        at = k + open.size();
        if (stock.compare(at, 7, "<think>") != 0) {
            out += block;
        }
    }
}

/** The first index where two item sequences differ, or the length of the shorter one. */
size_t first_difference(const std::vector<MemItem> & a, const std::vector<MemItem> & b) {
    size_t at = 0;
    while (at < a.size() && at < b.size() && a[at] == b[at]) {
        ++at;
    }
    return at;
}

/**
 * The scenario render-live: the items of the prompt of each turn
 * (plan_prompt) must equal the tokens of the full render of the form of the
 * turn, with every image. A turn that follows a complete answer continues
 * the memory, and the full render of the live form must equal the render of
 * the template with the empty think blocks. The chat has a system message,
 * three images, an interrupted turn (the app removes its question) and a turn
 * with thinking. The memory after an answer is the prompt, the tokens of the
 * answer text and the end token, without a decode. With FUZZ_APP_REAL_MODEL
 * and FUZZ_APP_REAL_MMPROJ the scenario takes that model on the CPU.
 */
int check_render_live(Program & p, LoadSpec s) {
    const Options & opt = *p.opt;
    if (!opt.real_model.empty()) {
        s.model  = opt.real_model;
        s.mmproj = opt.real_mmproj;
        s.device.clear();
        s.n_ctx  = 8192;
    }
    s.speculative = false;
    for (int i = 0; i < 3; ++i) {
        p.images.push_back(make_bmp(24 + 8 * i, 24, 21 + (uint32_t) i));
    }
    // Each decode of an image in the app gives 96 x 64 pixels: the tape of decodeImage repeats one decode.
    p.tape = {0, 95, 63};
    const jlong h = api_load(s);
    if (h == 0) {
        fail("scenario render-live: the model did not load");
    }
    const std::shared_ptr<Engine> sp = engine_of(h);
    struct Turn {
        const char * user;
        int          image;
        const char * answer;
        bool         thinking;
        bool         interrupted;
    };
    const Turn turns[] = {
        {"Describe the first picture.", 0, "A small square with a pattern of colored dots.", false, false},
        {"What colors are there?", -1, "Red, green and blue.\n\n- red\n- green\n- blue", false, false},
        {"And this one?", 1, "Another pattern: rows of gray and white.", false, false},
        {"Write a word in Russian.", -1, "Привет! 日本語 is Japanese.", false, false},
        {"Stop here.", -1, "This answer stops", false, true},
        {"Think about the third picture.", 2, "It shows stripes.", true, false},
        {"Now without thinking, one line.", -1, "One line.", false, false},
        {"The last question?", -1, "Yes, the last one.", false, false},
    };
    int checks = 0;
    int continued = 0;
    jni_call("render-live", [&](JNIEnv * env) {
        Engine & e = *sp;
        std::lock_guard<std::mutex> lock(e.mutex);
        if (e.prompt->form() != HistoryForm::kLive) {
            fail("scenario render-live: the live form is off: %s", e.prompt->refusal().c_str());
        }
        const llama_vocab * vocab = llama_model_get_vocab(e.model);
        const jclass        cls   = native_class(env);
        std::vector<jbyteArray>  arrays;
        std::vector<std::string> ids;
        for (const std::vector<uint8_t> & img : p.images) {
            arrays.push_back(reinterpret_cast<jbyteArray>(fakejni::arg(env, fakejni::new_byte_array(img.data(), img.size()))));
            ids.push_back(cache_io::sha256_hex(img.data(), img.size()));
        }
        std::vector<PromptMessage> history = {PromptMessage{"system", "You are a helpful assistant.", ""}};
        std::vector<jbyteArray>    history_images = {nullptr};
        const llama_token          eog = e.prompt->eog_token();
        bool                       expect_cont = false;
        for (size_t k = 0; k < sizeof(turns) / sizeof(turns[0]); ++k) {
            const Turn & t = turns[k];
            std::vector<PromptMessage> msgs     = history;
            std::vector<jbyteArray>    image_of = history_images;
            PromptMessage u{"user", t.user, ""};
            jbyteArray    arr = nullptr;
            if (t.image >= 0) {
                u.content  = std::string(mtmd_default_marker()) + "\n" + u.content;
                u.image_id = ids[(size_t) t.image];
                arr        = arrays[(size_t) t.image];
            }
            msgs.push_back(u);
            image_of.push_back(arr);
            // The references: the full render of the form of the turn, and its tokens with every image.
            std::string ref_text;
            std::string tail;
            if (t.thinking) {
                e.prompt->render_template(msgs, true, ref_text, tail);
            } else {
                if (!e.prompt->render_live(msgs, ref_text)) {
                    fail("scenario render-live: turn %zu has no full render of the live form", k + 1);
                }
                std::string stock;
                e.prompt->render_template(msgs, false, stock, tail);
                if (empty_blocks_in(stock) != ref_text) {
                    fail("scenario render-live: turn %zu: the full render of the live form differs from the render of "
                         "the template with the empty think blocks",
                         k + 1);
                }
            }
            std::vector<jbyteArray>  all_images;
            std::vector<std::string> all_ids;
            for (size_t i = 0; i < msgs.size(); ++i) {
                if (image_of[i] != nullptr) {
                    all_images.push_back(image_of[i]);
                    all_ids.push_back(msgs[i].image_id);
                }
            }
            mtmd::input_chunks                    ref_chunks;
            std::vector<MemItem>                  ref_items;
            std::vector<const mtmd_input_chunk *> ref_chunk_of;
            std::string                           error;
            if (!tokenize_prompt(env, cls, e, ref_text, all_images, all_ids, true, ref_chunks, ref_items, ref_chunk_of,
                                 error)) {
                fail("scenario render-live: turn %zu: the reference did not tokenize: %s", k + 1, error.c_str());
            }
            // The items of the engine.
            PromptPlan plan;
            if (!plan_prompt(env, cls, e, msgs, image_of, t.thinking, plan, error)) {
                fail("scenario render-live: turn %zu: plan_prompt failed: %s", k + 1, error.c_str());
            }
            if (plan.tp.continuation != expect_cont || plan.tp.base_snapshot != t.thinking) {
                fail("scenario render-live: turn %zu: continuation %d and base snapshot %d, want %d and %d", k + 1,
                     (int) plan.tp.continuation, (int) plan.tp.base_snapshot, (int) expect_cont, (int) t.thinking);
            }
            if (plan.items != ref_items) {
                const size_t at = first_difference(plan.items, ref_items);
                fail("scenario render-live: turn %zu (continuation %d): %zu items against %zu of the full render, first "
                     "difference at %zu: %d against %d",
                     k + 1, (int) plan.tp.continuation, plan.items.size(), ref_items.size(), at,
                     at < plan.items.size() ? plan.items[at].token : -2, at < ref_items.size() ? ref_items[at].token : -2);
            }
            const int64_t ref_tokens = count_tokens(ref_chunk_of, 0, ref_chunk_of.size());
            if (plan.n_tokens != ref_tokens) {
                fail("scenario render-live: turn %zu: %lld tokens against %lld of the full render", k + 1,
                     (long long) plan.n_tokens, (long long) ref_tokens);
            }
            checks += 1;
            continued += plan.tp.continuation ? 1 : 0;
            // The memory after the answer, as the decode leaves it: the prompt, the answer, and the end token of a
            // complete answer. The thinking of a thinking turn is part of the answer tokens.
            const std::string        generated = t.thinking ? std::string("Some thought.\n</think>\n\n") + t.answer : t.answer;
            const std::vector<llama_token> ans = common_tokenize(vocab, generated, false, true);
            e.cache = plan.items;
            for (const llama_token tok : ans) {
                e.cache.push_back(MemItem{tok, {}});
            }
            if (!t.interrupted) {
                e.cache.push_back(MemItem{eog, {}});
            }
            e.mem_tokens = plan.n_tokens + (int64_t) ans.size() + (t.interrupted ? 0 : 1);
            e.n_past     = (llama_pos) e.cache.size();
            if (!t.interrupted) {
                const bool recorded = e.prompt->answer_ended(t.answer, !t.thinking, e.cache.size(), e.cache.back().token);
                if (recorded == t.thinking) {
                    fail("scenario render-live: turn %zu: the record after the answer is %d", k + 1, (int) recorded);
                }
                history.push_back(u);
                history_images.push_back(arr);
                history.push_back(PromptMessage{"assistant", t.answer, ""});
                history_images.push_back(nullptr);
            }
            expect_cont = !t.interrupted && !t.thinking;
        }
        // The memory of the context is empty: the record and the items go.
        clear_all(e);
    });
    api_free(h);
    if (continued < 4) {
        fail("scenario render-live: only %d of %d turns continued the memory", continued, checks);
    }
    fprintf(stderr, "scenario render-live: %d turns, %d continued, the items of each turn equal the full render of its "
                    "form (%s)\n",
            checks, continued, opt.real_model.empty() ? "tiny model" : opt.real_model.c_str());
    return 0;
}

/**
 * Generate the answer as the app keeps it (AnswerStream in Thinking.kt): the
 * text after the thinking, without the white space at its start. Returns the
 * count of pieces. O(limit) generateNext calls.
 */
int drain_content(LiveEngine & le, int limit, std::string & content) {
    bool thinking = false;
    int  n        = 0;
    content.clear();
    for (; n < limit; ++n) {
        std::vector<int8_t> piece;
        const int64_t given = given_tokens(le);
        const int r = api_generate_next(le.handle, piece);
        check_engine(le.handle, "one generateNext of a scenario");
        note_delivered(le, given);
        check_delivered(le, "one generateNext of a scenario");
        if (r != 1) {
            break;
        }
        if (piece.size() > 1 && !thinking) {
            content.append(reinterpret_cast<const char *>(piece.data()) + 1, piece.size() - 1);
        }
        if (piece[0] == 1) {
            if (content.empty()) {
                thinking = true;
            } else {
                content += "<think>";
            }
        } else if (piece[0] == 2) {
            thinking = false;
        }
    }
    const size_t first = content.find_first_not_of(" \t\n\r\f\v");
    content = first == std::string::npos ? std::string() : content.substr(first);
    return n;
}

/**
 * Put a complete answer into the engine as the model gives it: the tokens of
 * text and the end token of the template decode (decode_one, which the draft
 * context follows), and the next generateNext ends the answer
 * (finish_answer). The tiny model does not frequently end an answer without
 * a thinking tag, thus the scenario gives some answers this way.
 */
void inject_answer(jlong h, const std::string & text, const char * what) {
    const std::shared_ptr<Engine> sp = engine_of(h);
    {
        std::lock_guard<std::mutex> lock(sp->mutex);
        Engine & e = *sp;
        const llama_vocab * vocab = llama_model_get_vocab(e.model);
        for (const llama_token t : common_tokenize(vocab, text, false, true)) {
            if (decode_one(e, t) != 0) {
                fail("%s: a token of the injected answer did not decode", what);
            }
        }
        if (decode_one(e, e.prompt->eog_token()) != 0) {
            fail("%s: the end token of the injected answer did not decode", what);
        }
        e.answer_text = text;
        e.answer_ends = true;
        e.id_last     = LLAMA_TOKEN_NULL;
    }
    std::vector<int8_t> piece;
    if (api_generate_next(h, piece) != 0) {
        fail("%s: the injected answer did not end", what);
    }
}

/**
 * The scenario live-turns: a chat in the live form with decodes. A turn after
 * a complete answer continues the memory, the state after each complete
 * answer goes into the store on the background thread, the exactness oracle
 * holds after each turn, and a new engine on the same cache directory (a
 * restart of the app) restores the snapshot after the last complete answer.
 * The even turns get an answer that inject_answer puts into the engine,
 * with the end token at its end. The odd turns get the answer of the model.
 */
int check_live_turns(Program & p, LoadSpec s, bool speculative) {
    s.speculative = speculative;
    s.cache       = p.dir + (speculative ? "/live-s" : "/live-n");
    if (p.images.empty()) {
        p.images.push_back(make_bmp(32, 32, 31));
    }
    const char * what = speculative ? "scenario live-turns (draft)" : "scenario live-turns";
    jlong h = api_load(s);
    if (h == 0) {
        fail("%s: the tiny model did not load", what);
    }
    const char16_t * questions[] = {u"Hello, how are you? Tell me about the cat.", u"And the dog?",
                                    u"What is in the picture?", u"Thanks. One more word?", u"Bye.", u"Really?"};
    const char * answers[] = {"The cat sits on the mat.", "", "A square of gray dots.", "", "Goodbye, and thank you.", ""};
    std::vector<Msg> chat;
    int continued = 0;
    int snapshots = 0;
    bool last_complete = false;
    for (int k = 0; k < 6; ++k) {
        const std::shared_ptr<Engine> sp = engine_of(h);
        bool expect = false;
        {
            std::lock_guard<std::mutex> lock(sp->mutex);
            expect = sp->prompt->has_record();
        }
        chat.push_back(Msg{"user", questions[k], k == 2 ? 0 : -1});
        if (api_chat_start(p, h, chat, false, 0.0f, 0.8f, false, false) < 0) {
            fail("%s: chatStart of turn %d failed", what, k + 1);
        }
        {
            std::lock_guard<std::mutex> lock(sp->mutex);
            if (sp->turn.continued != expect) {
                fail("%s: turn %d continued %d, the record said %d", what, k + 1, (int) sp->turn.continued, (int) expect);
            }
            continued += sp->turn.continued ? 1 : 0;
        }
        std::string content;
        if (k % 2 == 0) {
            content = answers[k];
            inject_answer(h, content, what);
        } else {
            LiveEngine le;
            le.handle = h;
            start_tracking(le);
            drain_content(le, 64, content);
        }
        // The snapshot task of a complete answer runs on the background thread.
        sp->worker->drain();
        {
            std::lock_guard<std::mutex> lock(sp->mutex);
            last_complete = sp->prompt->has_record();
            if (last_complete) {
                if (sp->states->find(sp->cache) == nullptr) {
                    fail("%s: turn %d: no snapshot of the memory after a complete answer", what, k + 1);
                }
                snapshots += 1;
            }
        }
        std::string why;
        if (!oracle(h, why)) {
            fail("%s: turn %d: exactness oracle: %s", what, k + 1, why.c_str());
        }
        chat.push_back(Msg{"assistant", fakejni::new_string_utf8(content)->chars, -1});
    }
    // A restart: a new engine on the same directory restores the state after the last complete answer.
    api_free(h);
    h = api_load(s);
    if (h == 0) {
        fail("%s: the second load failed", what);
    }
    chat.push_back(Msg{"user", u"After the restart?", -1});
    if (api_chat_start(p, h, chat, false, 0.0f, 0.8f, false, false) < 0) {
        fail("%s: chatStart after the restart failed", what);
    }
    bool restored = false;
    {
        const std::shared_ptr<Engine> sp = engine_of(h);
        std::lock_guard<std::mutex> lock(sp->mutex);
        restored = sp->turn.from_snapshot;
    }
    if (snapshots > 0 && !restored) {
        fail("%s: the turn after the restart did not restore the snapshot after the last complete answer", what);
    }
    if (continued < 3) {
        fail("%s: %d turns continued the memory, and each of the three injected answers must give one", what,
             continued);
    }
    LiveEngine le;
    le.handle = h;
    start_tracking(le);
    std::string content;
    drain_content(le, 32, content);
    std::string why;
    if (!oracle(h, why)) {
        fail("%s: after the restart: exactness oracle: %s", what, why.c_str());
    }
    api_free(h);
    fprintf(stderr, "%s: 6 turns, %d continued, %d snapshots after an answer, restart %s (last answer %s)\n", what,
            continued, snapshots, restored ? "restored" : "decoded the prompt", last_complete ? "complete" : "not complete");
    return 0;
}

/**
 * The scenario spec-limit: the MTP drafter stops at the draft length of the
 * call (the n_max of common_speculative_draft_params). A draft of n tokens
 * runs n passes of the MTP block, each writes one cell of the draft context,
 * and it gives the first n tokens of the longest draft.
 */
int check_spec_limit(Program & p, LoadSpec s) {
    s.speculative = true;
    s.mmproj.clear();
    const jlong h = api_load(s);
    if (h == 0) {
        fail("scenario spec-limit: the tiny model did not load");
    }
    if (api_chat_start(p, h, {Msg{"user", u"Hello, how are you? Tell me about the cat.", -1}}, false, 0.0f, 0.8f, false,
                       false) < 0) {
        fail("scenario spec-limit: chatStart failed");
    }
    LiveEngine le;
    le.handle = h;
    start_tracking(le);
    // The first call samples the first token: the draft of the next step starts from it.
    drain_answer(le, 1);
    int passes[SpecPolicy::kDraftMax + 1] = {};
    std::vector<llama_tokens> drafts(SpecPolicy::kDraftMax + 1);
    {
        const std::shared_ptr<Engine> sp = engine_of(h);
        Engine & e = *sp;
        std::lock_guard<std::mutex> lock(e.mutex);
        if (e.spec == nullptr || e.id_last == LLAMA_TOKEN_NULL) {
            fail("scenario spec-limit: no draft driver, or no token to draft from");
        }
        llama_memory_t  mem_dft = llama_get_memory(e.ctx_dft);
        const llama_pos pos0    = e.n_past;
        for (int n = SpecPolicy::kDraftMax; n >= 1; --n) {
            common_speculative_get_draft_params(e.spec, kSeqMain) = {
                /* .drafting = */ true,
                /* .n_max    = */ n,
                /* .pos0     = */ pos0,
                /* .id_last  = */ e.id_last,
                /* .prompt   = */ &e.spec_prompt,
                /* .result   = */ &drafts[(size_t) n],
            };
            common_speculative_draft(e.spec);
            const llama_pos dmax = llama_memory_seq_pos_max(mem_dft, kSeqMain);
            passes[n] = dmax >= pos0 ? (int) (dmax - pos0 + 1) : 0;
            llama_memory_seq_rm(mem_dft, kSeqMain, pos0, -1);
        }
    }
    const llama_tokens & longest = drafts[SpecPolicy::kDraftMax];
    for (int n = 1; n <= SpecPolicy::kDraftMax; ++n) {
        const llama_tokens & d = drafts[(size_t) n];
        if (passes[n] > n) {
            fail("scenario spec-limit: a draft of at most %d tokens ran %d passes of the MTP block", n, passes[n]);
        }
        if (d.size() > (size_t) n || !std::equal(d.begin(), d.end(), longest.begin())) {
            fail("scenario spec-limit: the draft of at most %d tokens (%zu tokens) is not a prefix of the draft of %d",
                 n, d.size(), SpecPolicy::kDraftMax);
        }
    }
    // The answer continues after the drafts.
    drain_answer(le, 24);
    api_free(h);
    fprintf(stderr, "scenario spec-limit: passes %d %d %d %d for the limits 1 to 4, each draft a prefix of the longest\n",
            passes[1], passes[2], passes[3], passes[4]);
    return 0;
}

/**
 * One flow of the scenario image-stage. A flow starts from an empty memory,
 * makes the first turn of the chat when turn1 is set, stages the images of
 * stages (stageImage, in order), and then sends the second message with
 * send_image (-1 for none).
 */
struct StageFlowSpec {
    const char * name;
    /** The flow whose answer this one must give, or null for a reference flow. */
    const char * ref;
    bool         turn1;
    std::vector<int> stages;
    /** resetChat after the stages, before the send. */
    bool         reset_before_send;
    int          send_image;
    /** The thinking of the stages and of the send. */
    bool         stage_thinking;
    bool         send_thinking;
};

/** The result of one flow. */
struct StageFlowResult {
    std::vector<llama_token> tokens;
    std::vector<float>       logits;
    double                   stage_ms    = 0.0;
    double                   chat_ms     = 0.0;
    double                   ttft_ms     = 0.0;
    int                      staged      = 0;
    bool                     first_image = false;
    /** The compute buffers of the decode context at the start and at the end of the flow, in MiB. */
    double                   compute0_mib = 0.0;
    double                   compute1_mib = 0.0;
    /** The items of the last staged part, and the first item that the send decoded after it, as text. */
    std::string              part;
    std::string              next;
    std::string              stats;
};

/**
 * The flows of the scenario. The first image is A (0), the second B (1).
 * "warm" is a send of B that no other flow refers to: on the phone it runs
 * first, thus the first image of the engine (the load of the projector and the
 * first image graph) is not part of a measured flow.
 */
const std::vector<StageFlowSpec> & stage_flows() {
    static const std::vector<StageFlowSpec> flows = {
        // Not a flow: prepareVision alone (the user opens the photo menu). The loop of the scenario runs it.
        {"vision", nullptr, false, {}, false, -1, false, false},
        {"warm", nullptr, true, {}, false, 1, false, false},
        {"send", nullptr, true, {}, false, 0, false, false},
        {"send-again", "send", true, {}, false, 0, false, false},
        {"stage-send", "send", true, {0}, false, 0, false, false},
        {"stage-twice", "send", true, {0, 0}, false, 0, false, false},
        {"text", nullptr, true, {}, false, -1, false, false},
        {"stage-drop", "text", true, {0}, false, -1, false, false},
        {"send-b", nullptr, true, {}, false, 1, false, false},
        {"stage-other", "send-b", true, {0}, false, 1, false, false},
        {"stage-swap", "send-b", true, {0, 1}, false, 1, false, false},
        {"send-think", nullptr, true, {}, false, 0, true, true},
        {"stage-think", "send-think", true, {0}, false, 0, true, true},
        {"stage-mixed", "send-think", true, {0}, false, 0, false, true},
        {"send-first", nullptr, false, {}, false, 0, false, false},
        {"stage-first", "send-first", false, {0}, false, 0, false, false},
        {"reset-send", nullptr, true, {}, true, 0, false, false},
        {"stage-reset", "reset-send", true, {0}, true, 0, false, false},
    };
    return flows;
}

/** The logits of the last position of the decode context, which chatStart leaves for the first sample. */
std::vector<float> prompt_logits(jlong h) {
    const std::shared_ptr<Engine> sp = engine_of(h);
    std::lock_guard<std::mutex> lock(sp->mutex);
    const int     n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(sp->model));
    const float * l       = llama_get_logits_ith(sp->ctx, -1);
    return l != nullptr ? std::vector<float>(l, l + n_vocab) : std::vector<float>();
}

/** The compute buffers of the decode context of the engine h, over all buffer types, in MiB. */
double compute_mib(jlong h) {
    const std::shared_ptr<Engine> sp = engine_of(h);
    std::lock_guard<std::mutex> lock(sp->mutex);
    size_t bytes = 0;
    for (const auto & [buft, mb] : llama_get_memory_breakdown(sp->ctx)) {
        (void) buft;
        bytes += mb.compute;
    }
    return bytes / 1048576.0;
}

/** The items [from, to) of the memory of the engine h as text: the piece of each token, an image as [image]. */
std::string items_text(jlong h, size_t from, size_t to) {
    const std::shared_ptr<Engine> sp = engine_of(h);
    std::lock_guard<std::mutex> lock(sp->mutex);
    const llama_vocab * vocab = llama_model_get_vocab(sp->model);
    std::string out;
    for (size_t i = from; i < to && i < sp->cache.size(); ++i) {
        const MemItem & it = sp->cache[i];
        std::string piece = it.token == kMemTokenNull ? "[image]" : common_token_to_piece(vocab, it.token, true);
        // No white space, thus the text is one field of a STAGE line.
        for (char c : piece) {
            switch (c) {
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case ' ':  out += "_"; break;
                default:   out += c; break;
            }
        }
        out += '|';
    }
    return out;
}

/**
 * The Kullback-Leibler divergence KL(P || Q) of the distributions softmax(p)
 * and softmax(q), in double, or infinity for vectors of different lengths.
 * O(vocabulary).
 */
double kl_divergence(const std::vector<float> & p, const std::vector<float> & q) {
    if (p.size() != q.size() || p.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    const double max_p = *std::max_element(p.begin(), p.end());
    const double max_q = *std::max_element(q.begin(), q.end());
    double sum_p = 0.0, sum_q = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        sum_p += std::exp((double) p[i] - max_p);
        sum_q += std::exp((double) q[i] - max_q);
    }
    const double log_zp = max_p + std::log(sum_p);
    const double log_zq = max_q + std::log(sum_q);
    double kl = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        const double lp = (double) p[i] - log_zp;
        const double lq = (double) q[i] - log_zq;
        kl += std::exp(lp) * (lp - lq);
    }
    return std::max(kl, 0.0);
}

/** The index of the largest logit, the token that a greedy sample takes. */
size_t top_token(const std::vector<float> & logits) {
    return logits.empty() ? 0 : (size_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
}

/** The largest absolute difference of two logit vectors, or infinity for vectors of different lengths. */
float logit_distance(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return std::numeric_limits<float>::infinity();
    }
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (!(d <= worst)) {
            worst = d;
        }
    }
    return worst;
}

/**
 * Run one flow on the engine h: the memory starts empty, then the first turn,
 * the stages and the send. The answer is the first n_tokens tokens of the
 * sampler of the app (temperature 0.7, top-p 0.8, the seed of
 * QWEN_SAMPLER_SEED). O(prompt + n_tokens) decodes.
 */
StageFlowResult run_stage_flow(Program & p, jlong h, const StageFlowSpec & f, const std::u16string & first_text,
                               int n_tokens, const char * what) {
    const std::u16string first_answer = u"I read the log. Every step finished with no error.";
    const std::u16string send_text    = u"What is in the picture? Answer in two sentences.";
    const std::u16string stage_text   = u"Describe the image.";
    StageFlowResult r;
    const std::shared_ptr<Engine> sp = engine_of(h);
    api_reset(h);
    {
        std::lock_guard<std::mutex> lock(sp->mutex);
        r.first_image = sp->mctx == nullptr;
    }
    r.compute0_mib = compute_mib(h);
    std::vector<Msg> history;
    if (f.turn1) {
        history.push_back(Msg{"user", first_text, -1});
        if (api_chat_start(p, h, history, false, 0.7f, 0.8f, false, false) < 0) {
            fail("%s: flow %s: the first turn did not start", what, f.name);
        }
        const std::string answer = fakejni::to_modified_utf8(first_answer);
        inject_answer(h, answer, what);
        // The snapshot after a complete answer runs on the background thread.
        sp->worker->drain();
        history.push_back(Msg{"assistant", first_answer, -1});
    }
    for (const int image : f.stages) {
        std::vector<Msg> msgs = history;
        msgs.push_back(Msg{"user", stage_text, image});
        const auto t0 = std::chrono::steady_clock::now();
        r.staged = api_stage_image(p, h, msgs, f.stage_thinking);
        r.stage_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        check_engine(h, "stageImage of a scenario");
        if (r.staged <= 0) {
            fail("%s: flow %s: stageImage gave %d, it must stage the image", what, f.name, r.staged);
        }
    }
    // The staged part and the end of it: the send decodes from there when it extends the part.
    size_t part_end = 0;
    if (!f.stages.empty()) {
        size_t from = 0;
        {
            std::lock_guard<std::mutex> lock(sp->mutex);
            from     = sp->staged.active() ? sp->staged.from : sp->cache.size();
            part_end = sp->cache.size();
        }
        r.part = items_text(h, from, part_end);
    }
    if (f.reset_before_send) {
        api_reset(h);
    }
    std::vector<Msg> msgs = history;
    msgs.push_back(Msg{"user", send_text, f.send_image});
    const auto t0 = std::chrono::steady_clock::now();
    if (api_chat_start(p, h, msgs, f.send_thinking, 0.7f, 0.8f, false, false) < 0) {
        fail("%s: flow %s: chatStart failed", what, f.name);
    }
    const auto t1 = std::chrono::steady_clock::now();
    r.logits = prompt_logits(h);
    bool extended = false;
    {
        std::lock_guard<std::mutex> lock(sp->mutex);
        extended = sp->turn.staged_tokens > 0;
    }
    if (extended) {
        // The send kept the staged part: its first own item follows the image.
        r.next = items_text(h, part_end, part_end + 1);
    }
    LiveEngine le;
    le.handle = h;
    start_tracking(le);
    std::vector<int8_t> piece;
    const int64_t given = given_tokens(le);
    const int     rc    = api_generate_next(h, piece);
    const auto    t2    = std::chrono::steady_clock::now();
    note_delivered(le, given);
    check_engine(h, "the first generateNext of a scenario");
    if (rc == 1) {
        drain_answer(le, n_tokens - 1);
    }
    r.tokens  = le.delivered;
    r.chat_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.ttft_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
    r.stats   = api_string_call("stats", h);
    r.compute1_mib = compute_mib(h);
    return r;
}

/**
 * The scenario image-stage: stageImage of an attached image before its send
 * gives the answer of the send alone. Each flow of stage_flows() with a
 * reference must give the tokens of that reference, and the logits after its
 * prompt must be those of the reference when two runs of the same send give
 * the same logits ("send-again"). The flows cover a continuation of the live
 * form, the same image staged two times, a send without the image, a
 * different image, a second stage of another image, the template form of a
 * thinking turn, a stage without thinking and a send with it, the first
 * message of a chat, and a reset after the stage.
 *
 * With FUZZ_APP_REAL_MODEL and FUZZ_APP_REAL_MMPROJ the scenario takes that
 * model on FUZZ_APP_DEVICE with the settings of the app (n_ctx 8192, four
 * threads, FUZZ_APP_IMAGE_TOKENS image tokens, the draft when
 * FUZZ_APP_SPECULATIVE is not 0), the photo of FUZZ_APP_IMAGE as the first
 * image, a first message of about 1700 tokens, and the flows of
 * FUZZ_APP_STAGE_FLOWS (names separated by commas, in order). Each flow
 * prints one "STAGE" line with its times, the compute buffers of the context
 * at its start and end, the staged part and the first item after it, and
 * against its reference the largest logit difference, the KL divergence and
 * the top token of the logits after the prompt, and the count of equal
 * leading tokens. FUZZ_APP_STAGE_TOKENS is the length of each answer (the
 * preset is 32). FUZZ_APP_STAGE_STRICT=0 turns the checks into reports: the
 * kernels of HTP0 round differently for other batch shapes. O(flows x
 * (prompt + answer)) decodes.
 */
int check_image_stage(Program & p, LoadSpec s, bool speculative) {
    const Options & opt  = *p.opt;
    const bool      real = !opt.real_model.empty() && !opt.real_mmproj.empty();
    const char *    what = speculative ? "scenario image-stage (draft)" : "scenario image-stage";
    auto env_int = [](const char * name, int preset) {
        const char * v = getenv(name);
        return v != nullptr ? atoi(v) : preset;
    };
    s.speculative = speculative;
    s.cache.clear();
    std::u16string first_text = u"Here is the log of the tool. ";
    // About 9 tokens for each line: the first message of the real model is 1517 tokens, and the memory after the
    // image turn holds about 1590 positions, near the 1647 of the chat of the user.
    const int lines = real ? 160 : 6;
    for (int i = 0; i < lines; ++i) {
        first_text += u"Step " + fakejni::new_string_utf8(std::to_string(i + 1))->chars + u" finished with no error. ";
    }
    first_text += u"Did any step fail?";
    p.images.clear();
    if (real) {
        s.model            = opt.real_model;
        s.mmproj           = opt.real_mmproj;
        s.vision           = opt.device;
        s.gpu_layers       = opt.device.empty() ? 0 : 999;
        // Four threads as the app. A host run on the CPU takes more with FUZZ_APP_REAL_THREADS.
        s.threads          = std::max(1, env_int("FUZZ_APP_REAL_THREADS", 4));
        s.n_ctx            = 8192;
        s.image_max_tokens = env_int("FUZZ_APP_IMAGE_TOKENS", 768);
        std::vector<uint8_t> photo;
        const char * path = getenv("FUZZ_APP_IMAGE");
        if (path == nullptr || !cache_io::read_file(path, photo, 64u << 20) || photo.empty()) {
            fail("%s: FUZZ_APP_IMAGE must name a readable photo", what);
        }
        p.images.push_back(std::move(photo));
        p.images.push_back(make_bmp(640, 480, 5));
    } else {
        p.images.push_back(make_bmp(48, 40, 3));
        p.images.push_back(make_bmp(40, 48, 4));
    }
    // decodeImage gives no pixels, thus the engine decodes each image itself (stb_image): the same pixels each time.
    p.tape = {4};
    const int n_tokens = std::max(1, env_int("FUZZ_APP_STAGE_TOKENS", 32));
    // FUZZ_APP_STAGE_STRICT=0 (the phone) reports the KL and the top token of each comparison and does not stop: the
    // kernels of HTP0 round differently for other batch shapes, thus the tokens of an answer can differ.
    const bool strict = env_int("FUZZ_APP_STAGE_STRICT", 1) != 0;
    std::vector<const StageFlowSpec *> order;
    if (const char * list = getenv("FUZZ_APP_STAGE_FLOWS")) {
        std::string names = list;
        size_t at = 0;
        while (at <= names.size()) {
            const size_t comma = std::min(names.find(',', at), names.size());
            const std::string name = names.substr(at, comma - at);
            const auto it = std::find_if(stage_flows().begin(), stage_flows().end(),
                                         [&](const StageFlowSpec & f) { return name == f.name; });
            if (it == stage_flows().end()) {
                fail("%s: FUZZ_APP_STAGE_FLOWS names the unknown flow '%s'", what, name.c_str());
            }
            order.push_back(&*it);
            at = comma + 1;
        }
    } else {
        for (const StageFlowSpec & f : stage_flows()) {
            order.push_back(&f);
        }
    }
    setenv("QWEN_SAMPLER_SEED", "20260925", 1);
    const auto t_load = std::chrono::steady_clock::now();
    const jlong h = api_load(s);
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load).count();
    unsetenv("QWEN_SAMPLER_SEED");
    if (h == 0) {
        fail("%s: the model did not load", what);
    }
    printf("STAGE-LOAD ms=%.1f model=%s draft=%d image_tokens=%d\n", load_ms, s.model.c_str(), (int) speculative,
           s.image_max_tokens);
    // The first result of each reference name with the flow that gave it, the logit distance of two sends
    // without a stage (the noise of the device), and the distance of each comparison with a stage flow.
    struct FirstResult {
        StageFlowResult result;
        std::string     flow;
        bool            staged = false;
    };
    std::map<std::string, FirstResult> first;
    std::vector<std::pair<std::string, float>> staged_dist;
    float noise = -1.0f;
    int   compared = 0;
    // FUZZ_APP_STAGE_DUMP=<prefix> writes the logits after the prompt of flow k to <prefix>-<k>-<flow>.f32 (raw
    // floats), thus a comparison across processes (with and without a switch) can read them.
    const char * dump = getenv("FUZZ_APP_STAGE_DUMP");
    int          index = 0;
    for (const StageFlowSpec * f : order) {
        if (std::string(f->name) == "vision") {
            const auto t0 = std::chrono::steady_clock::now();
            const int  ok = api_prepare_vision(h);
            printf("STAGE-VISION ms=%.1f ready=%d\n",
                   std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), ok);
            if (ok != 1) {
                fail("%s: prepareVision gave %d, the projector must load", what, ok);
            }
            ++index;
            continue;
        }
        const StageFlowResult r = run_stage_flow(p, h, *f, first_text, n_tokens, what);
        if (dump != nullptr) {
            const std::string path = std::string(dump) + "-" + std::to_string(index) + "-" + f->name + ".f32";
            if (!cache_io::write_file_atomic(path, {{r.logits.data(), r.logits.size() * sizeof(float)}})) {
                fail("%s: the logits did not write to %s", what, path.c_str());
            }
        }
        ++index;
        const std::string ref = f->ref != nullptr ? f->ref : f->name;
        const bool        is_stage = !f->stages.empty();
        std::string same = "-";
        float       dist = -1.0f;
        double      kl   = -1.0;
        int         top1 = -1;
        size_t      lead = 0;
        const auto  it   = first.find(ref);
        if (it != first.end()) {
            const StageFlowResult & want = it->second.result;
            dist = logit_distance(want.logits, r.logits);
            kl   = kl_divergence(want.logits, r.logits);
            top1 = top_token(want.logits) == top_token(r.logits) ? 1 : 0;
            while (lead < r.tokens.size() && lead < want.tokens.size() && r.tokens[lead] == want.tokens[lead]) {
                ++lead;
            }
            same = want.tokens == r.tokens ? "yes" : "no";
            compared += 1;
            if (!is_stage && !it->second.staged) {
                noise = std::max(noise, dist);
            } else {
                staged_dist.emplace_back(is_stage ? f->name : it->second.flow, dist);
            }
        }
        printf("STAGE flow=%s ref=%s first_image=%d stage_ms=%.1f staged=%d chat_ms=%.1f ttft_ms=%.1f tokens=%zu same=%s "
               "dlogit=%g kl=%.6g top1=%d lead=%zu compute0_mib=%.1f compute1_mib=%.1f part=%s next=%s stats=%s\n",
               f->name, ref.c_str(), (int) r.first_image, r.stage_ms, r.staged, r.chat_ms, r.ttft_ms, r.tokens.size(),
               same.c_str(), dist, kl, top1, lead, r.compute0_mib, r.compute1_mib, r.part.empty() ? "-" : r.part.c_str(),
               r.next.empty() ? "-" : r.next.c_str(), r.stats.c_str());
        fflush(stdout);
        if (same == "no" && strict) {
            const std::vector<llama_token> & want = it->second.result.tokens;
            size_t at = 0;
            while (at < r.tokens.size() && at < want.size() && r.tokens[at] == want[at]) {
                ++at;
            }
            fail("%s: flow %s gave other tokens than flow %s: they differ at token %zu of %zu (logits %g apart)", what,
                 f->name, it->second.flow.c_str(), at, r.tokens.size(), dist);
        }
        if (it == first.end()) {
            first.emplace(ref, FirstResult{r, f->name, is_stage});
        }
    }
    // Two runs of the same send gave the same logits: then each stage flow must give exactly the logits of its reference.
    if (strict && noise == 0.0f) {
        for (const auto & [name, dist] : staged_dist) {
            if (dist != 0.0f) {
                fail("%s: flow %s gave logits %g apart from its reference, and two runs of one send gave the same logits",
                     what, name.c_str(), dist);
            }
        }
    }
    api_free(h);
    fprintf(stderr, "%s: %zu flows, %d compared with their reference, %s, logit noise of a repeated send %g\n", what,
            order.size(), compared, strict ? "the same tokens in each" : "the STAGE lines report each comparison", noise);
    return 0;
}

}  // namespace

int run_scenario(const Options & opt, const std::string & name) {
    init_once(opt);
    Program p;
    p.opt = &opt;
    p.dir = opt.work_dir + "/scenario-" + name + "-" + std::to_string(getpid());
    p.cache_a = p.dir + "/cache";
    cache_io::make_dirs(p.dir);
    g_prog = &p;
    LoadSpec s;
    s.model = opt.model_dir + "/tiny-qwen35-f32.gguf";
    s.mmproj = opt.model_dir + "/tiny-qwen35-mmproj.gguf";
    s.n_ctx = 1024;
    s.threads = 2;
    s.image_max_tokens = 256;
    // FUZZ_APP_DEVICE=HTP0 runs the scenario on the NPU of the phone.
    s.device = opt.device;
    int result = 0;
    auto user = [](const char16_t * text, int image) { return Msg{"user", text, image}; };
    if (name == "image-shape") {
        // An encoder output file of the right namespace with 1 token, where the image gives more.
        p.images.push_back(make_bmp(32, 32, 7));
        s.cache = p.cache_a;
        write_embd(s.cache + "/engine/" + cache_namespace(s.model, s.mmproj, s.image_max_tokens) + "/images",
                   p.images[0], 32, 32, 1, 64);
        const jlong h = api_load(s);
        fprintf(stderr, "scenario image-shape: load %s, then a turn with the image\n", h ? "ok" : "failed");
        const int rc = api_chat_start(p, h, {user(u"What is in the picture?", 0)}, false, 0.0f, 0.8f, false, false);
        fprintf(stderr, "scenario image-shape: chatStart gave %d, no memory error\n", rc);
        api_free(h);
    } else if (name == "image-twice") {
        // The same image bytes in two messages, and the decoder of the app gives two sizes (as when one decode fails).
        p.images.push_back(make_bmp(40, 40, 9));
        // 16 x 16 pixels give 9 tokens, 96 x 8 pixels give 10.
        p.tape = {0, 15, 15, 0, 95, 7};
        const jlong h = api_load(s);
        const int rc = api_chat_start(p, h, {user(u"one", 0), Msg{"assistant", u"ok", -1}, user(u"two", 0)}, false, 0.0f,
                                      0.8f, false, false);
        fprintf(stderr, "scenario image-twice: chatStart gave %d, no memory error\n", rc);
        api_free(h);
    } else if (name == "jni-pending") {
        // The first allocation of load fails: GetStringUTFChars of the model path.
        fakejni::arm_alloc_failure(fakejni::env(), 0);
        const jlong h = api_load(s);
        fakejni::arm_alloc_failure(fakejni::env(), -1);
        fprintf(stderr, "scenario jni-pending: load gave %lld, no JNI call with a pending exception\n", (long long) h);
        api_free(h);
    } else if (name == "spec-disable") {
        // The draft context refuses the batch of a step, as a failed decode on the NPU does.
        s.speculative = true;
        s.mmproj.clear();
        const jlong h = api_load(s);
        api_chat_start(p, h, {user(u"Hello, how are you?", -1)}, false, 0.0f, 0.8f, false, false);
        LiveEngine le;
        le.handle = h;
        start_tracking(le);
        drain_answer(le, 1);
        g_spec_countdown = 0;
        const int n = drain_answer(le, 64);
        g_spec_countdown = -1;
        fprintf(stderr, "scenario spec-disable: %d more pieces after the fault, %llu faults, no lost token\n", n,
                (unsigned long long) g_spec_faults.load());
        api_free(h);
    } else if (name == "spec-parity") {
        // A forced draft failure must give the text of a run with speculation off.
        s.mmproj.clear();
        const std::string want = greedy_answer(p, s, false, -1, 48);
        for (const int fault_at : {-1, 0, 1, 2, 5}) {
            const uint64_t faults = g_spec_faults.load();
            const std::string got = greedy_answer(p, s, true, fault_at, 48);
            size_t at = 0;
            while (at < want.size() && at < got.size() && want[at] == got[at]) {
                ++at;
            }
            if (got != want) {
                fail("scenario spec-parity: with the draft fault at %d, the answer has %zu bytes and differs from the "
                     "%zu bytes without speculation at byte %zu",
                     fault_at, got.size(), want.size(), at);
            }
            fprintf(stderr, "scenario spec-parity: fault at %d (%llu faults), %zu bytes, the same text\n", fault_at,
                    (unsigned long long) (g_spec_faults.load() - faults), got.size());
        }
    } else if (name == "spec-image") {
        // A speculative engine with the projector: a text turn with drafts, then a turn with an
        // image. The verify batch of a speculative step asks for the most logits rows of all the
        // decodes of the engine, and the image chunks ask for none: each one must stay inside the
        // output limit of the contexts.
        s.speculative = true;
        p.images.push_back(make_bmp(32, 32, 11));
        const jlong h = api_load(s);
        if (h == 0) {
            fail("scenario spec-image: the tiny model did not load");
        }
        LiveEngine le;
        le.handle = h;
        std::vector<Msg> chat = {user(u"Hello, how are you? Tell me about the cat.", -1)};
        if (api_chat_start(p, h, chat, false, 0.0f, 0.8f, false, false) < 0) {
            fail("scenario spec-image: the text turn did not start");
        }
        start_tracking(le);
        const int n_text = drain_answer(le, 48);
        const std::shared_ptr<Engine> e = engine_of(h);
        const int64_t drafted = e != nullptr ? e->turn.drafted : 0;
        if (drafted == 0) {
            fail("scenario spec-image: the text turn made no draft, thus it did not test the verify batch");
        }
        chat.push_back(Msg{"assistant", u"ok", -1});
        chat.push_back(user(u"What is in the picture?", 0));
        if (api_chat_start(p, h, chat, false, 0.0f, 0.8f, false, false) < 0) {
            fail("scenario spec-image: the image turn did not start");
        }
        start_tracking(le);
        const int n_image = drain_answer(le, 48);
        fprintf(stderr, "scenario spec-image: text turn %d pieces with %lld drafted, image turn %d pieces, no abort\n",
                n_text, (long long) drafted, n_image);
        api_free(h);
    } else if (name == "snapshot-damage" || name == "image-damage") {
        // A file of the cache with damaged data and an intact header: the next load
        // reads it, and its answer must be the answer of a load without a cache.
        // The template form writes the snapshot before the generation prompt of
        // each turn. The live form writes one only after a complete answer, which
        // the tiny model does not frequently give: live-turns tests that store.
        setenv("QWEN_HISTORY_FORM", "template", 1);
        const bool image = name == "image-damage";
        if (image) {
            p.images.push_back(make_bmp(32, 32, 13));
        } else {
            s.mmproj.clear();
        }
        std::vector<Msg> chat = {user(u"Hello, how are you? Tell me about the cat.", image ? 0 : -1)};
        LoadSpec cached = s;
        cached.cache = p.cache_a;
        LoadSpec plain = s;
        plain.cache.clear();
        const std::string what = "scenario " + name;
        answer_of(p, cached, chat, 16, what.c_str());
        if (image) {
            // Without the snapshots, the next prefill decodes the image from its cache entry.
            std::vector<std::string> files;
            list_tree(p.cache_a, files);
            for (const std::string & path : files) {
                if (path.size() > 5 && path.compare(path.size() - 5, 5, ".snap") == 0) {
                    cache_io::remove_file(path);
                }
            }
        }
        const int n = damage_data(p.cache_a, image ? ".embd" : ".snap");
        if (n == 0) {
            fail("%s: the first load wrote no cache file", what.c_str());
        }
        const std::string got  = answer_of(p, cached, chat, 48, what.c_str());
        const std::string want = answer_of(p, plain, chat, 48, what.c_str());
        if (got != want) {
            fail("%s: the answer after the damage of %d files has %zu bytes and differs from the %zu bytes of a load "
                 "without a cache",
                 what.c_str(), n, got.size(), want.size());
        }
        fprintf(stderr, "scenario %s: %d files damaged, %zu bytes, the same text as without a cache\n", name.c_str(), n,
                got.size());
        unsetenv("QWEN_HISTORY_FORM");
    } else if (name == "sampler-nan") {
        // A NaN temperature (a damaged settings file can hold one) reaches the sampler chain.
        s.mmproj.clear();
        const jlong h = api_load(s);
        const int rc = api_chat_start(p, h, {user(u"Hello", -1)}, false, std::numeric_limits<float>::quiet_NaN(), 0.8f,
                                      false, false);
        LiveEngine le;
        le.handle = h;
        start_tracking(le);
        const int n = drain_answer(le, 8);
        fprintf(stderr, "scenario sampler-nan: chatStart gave %d, %d pieces, no assert\n", rc, n);
        api_free(h);
    } else if (name == "stop-free") {
        // The display thread of the app calls requestStop while the engine thread frees the
        // model. The destructor of an engine takes hundreds of milliseconds on the phone, thus
        // it must run in free, and never in requestStop or hasMtp on the other thread. Thus
        // those two calls must not own the engine at any time: a call that owns it can hold
        // the last owner when a free runs, and a free in that window is too rare to wait for.
        s.mmproj.clear();
        constexpr int kRounds = 50;
        constexpr int kSamples = 20000;
        const uint64_t before = g_off_thread_other.load();
        uint64_t requests = 0;
        uint64_t owned = 0;
        for (int round = 0; round < kRounds; ++round) {
            const jlong h = api_load(s);
            if (h == 0) {
                fail("scenario stop-free: the tiny model did not load");
            }
            std::atomic<bool> done{false};
            std::atomic<uint64_t> calls{0};
            std::thread ui([&] {
                while (!done.load()) {
                    api_request_stop(h);
                    api_has_mtp(h);
                    calls += 1;
                }
            });
            while (calls.load() == 0) {
                std::this_thread::yield();
            }
            {
                // The owners of the engine: the table and this sample. A third one is a call of the other thread.
                const std::shared_ptr<Engine> e = engine_of(h);
                for (int i = 0; i < kSamples; ++i) {
                    owned += e.use_count() > 2 ? 1 : 0;
                }
            }
            api_free(h);
            done = true;
            ui.join();
            requests += calls.load();
        }
        const uint64_t off = g_off_thread_other.load() - before;
        if (owned != 0 || off != 0) {
            fail("scenario stop-free: requestStop or hasMtp owned the engine in %llu of %d samples, and %llu of %d "
                 "engines were destroyed on the other thread, not in free",
                 (unsigned long long) owned, kRounds * kSamples, (unsigned long long) off, kRounds);
        }
        fprintf(stderr,
                "scenario stop-free: %llu stop requests, no owner of the engine in them, %d frees, each engine "
                "destroyed in free\n",
                (unsigned long long) requests, kRounds);
    } else if (name == "priority") {
        result = check_priority(opt);
    } else if (name == "render-live") {
        result = check_render_live(p, s);
    } else if (name == "live-turns") {
        result = check_live_turns(p, s, false);
        if (result == 0) {
            result = check_live_turns(p, s, true);
        }
    } else if (name == "spec-limit") {
        result = check_spec_limit(p, s);
    } else if (name == "image-stage") {
        // FUZZ_APP_SPECULATIVE selects one mode (the phone), else the scenario runs without and with the draft.
        if (const char * mode = getenv("FUZZ_APP_SPECULATIVE")) {
            result = check_image_stage(p, s, atoi(mode) != 0);
        } else {
            result = check_image_stage(p, s, false);
            if (result == 0) {
                result = check_image_stage(p, s, true);
            }
        }
    } else {
        fprintf(stderr,
                "unknown scenario %s: image-shape, image-twice, jni-pending, spec-disable, spec-parity, spec-image, "
                "snapshot-damage, image-damage, sampler-nan, stop-free, priority, render-live, live-turns, spec-limit, "
                "image-stage\n",
                name.c_str());
        result = 2;
    }
    g_prog = nullptr;
    cache_io::remove_tree(p.dir);
    fakejni::reset();
    return result;
}

int check_priority(const Options & opt) {
    init_once(opt);
    {
        std::lock_guard<std::mutex> lock(g_prio_mutex);
        g_prio_calls.clear();
    }
    const int32_t self = (int32_t) syscall(SYS_gettid);
    const int nice_before = getpriority(PRIO_PROCESS, (id_t) self);
    LoadSpec s;
    s.model = opt.model_dir + "/tiny-qwen35-f32.gguf";
    s.n_ctx = 1024;
    s.threads = 3;
    s.image_max_tokens = 256;
    g_bystander_armed = true;
    const jlong h = api_load(s);
    g_bystander_armed = false;
    const int32_t by = g_bystander_tid.load();
    int result = 0;
    fprintf(stderr, "check-priority: load %s, caller thread %d (nice %d before the load, %d after), bystander thread %d (nice %d)\n",
            h != 0 ? "ok" : "failed", self, nice_before, getpriority(PRIO_PROCESS, (id_t) self), by,
            by != 0 ? getpriority(PRIO_PROCESS, (id_t) by) : 0);
    {
        std::lock_guard<std::mutex> lock(g_prio_mutex);
        for (const auto & c : g_prio_calls) {
            const bool is_bystander = by != 0 && c.first == by;
            fprintf(stderr, "check-priority: setpriority(tid %d, %d)%s%s\n", c.first, c.second,
                    c.first == 0 ? " (the calling thread)" : "", is_bystander ? " <- the bystander thread" : "");
            result |= is_bystander ? 1 : 0;
        }
    }
    const std::shared_ptr<Engine> e = engine_of(h);
    if (e && e->hint) {
        fprintf(stderr, "check-priority: ADPF session %s\n", e->hint->ok() ? "open" : "not open (no libandroid.so)");
    }
    {
        std::lock_guard<std::mutex> lock(g_bystander_mutex);
        g_bystander_release = true;
    }
    g_bystander_cv.notify_all();
    if (g_bystander.joinable()) {
        g_bystander.join();
    }
    api_free(h);
    fprintf(stderr, "check-priority: %s\n", result != 0 ? "FAIL: the load lowered the priority of a thread that it did not make"
                                                        : "PASS");
    return result;
}

int run_program(const Options & opt, const uint8_t * data, size_t size) {
    static std::atomic<uint64_t> serial{0};
    crash_input::remember(data, size);
    FuzzedDataProvider fdp(data, size);
    Program p;
    p.opt = &opt;
    p.fdp = &fdp;
    p.dir = opt.work_dir + "/p" + std::to_string(getpid()) + "-" + std::to_string(serial++);
    p.cache_a = p.dir + "/cache-a";
    p.cache_b = p.dir + "/cache-b";
    cache_io::make_dirs(p.dir);
    {
        const std::string junk = "GGUF\x03\0\0\0 not a model";
        cache_io::write_file_atomic(p.dir + "/garbage.gguf", {{junk.data(), junk.size()}});
    }
    g_prog = &p;
    g_counters.programs += 1;

    const int n_images = fdp.ConsumeIntegralInRange<int>(0, 3);
    for (int i = 0; i < n_images; ++i) {
        if (fdp.ConsumeIntegralInRange<int>(0, 3) == 0) {
            p.images.push_back(fdp.ConsumeBytes<uint8_t>(fdp.ConsumeIntegralInRange<size_t>(0, 64)));
        } else {
            p.images.push_back(make_bmp(fdp.ConsumeIntegralInRange<int>(1, 64), fdp.ConsumeIntegralInRange<int>(1, 64),
                                        fdp.ConsumeIntegral<uint32_t>()));
        }
    }
    p.tape = fdp.ConsumeBytes<uint8_t>(fdp.ConsumeIntegralInRange<size_t>(0, 12));

    std::atomic<bool> ui_done{false};
    std::thread ui;
    if (opt.threads) {
        std::vector<uint8_t> schedule = fdp.ConsumeBytes<uint8_t>(fdp.ConsumeIntegralInRange<size_t>(1, 16));
        ui = std::thread(ui_thread, &p, std::move(schedule), &ui_done);
    }

    // Most programs start with a load: without an engine most operations test only the handle check.
    if (fdp.ConsumeIntegralInRange<int>(0, 7) != 0) {
        op_load(p);
    }
    int ops = 0;
    while (fdp.remaining_bytes() > 0 && ops < opt.max_ops) {
        ++ops;
        g_counters.ops += 1;
        const int op = fdp.ConsumeIntegralInRange<int>(0, 31);
        // Without a live engine most operations only test the check of the handle: load one first, mostly.
        if (p.live.empty() && op >= 3 && op < 24 && fdp.ConsumeIntegralInRange<int>(0, 3) != 0) {
            op_load(p);
        }
        if (op < 3) {
            op_load(p);
        } else if (op < 11) {
            op_chat(p);
        } else if (op < 19) {
            op_generate(p);
        } else if (op == 19) {
            api_request_stop(pick_handle(p));
            g_counters.stops += 1;
        } else if (op == 20) {
            api_string_call("stats", pick_handle(p));
        } else if (op == 21) {
            api_string_call("modelInfo", pick_handle(p));
            api_has_mtp(pick_handle(p));
        } else if (op == 22) {
            api_reset(pick_handle(p));
            check_all(p, "resetChat", false);
        } else if (op == 23) {
            op_bench(p);
        } else if (op == 24) {
            op_free(p);
        } else if (op == 25) {
            fakejni::arm_alloc_failure(fakejni::env(), fdp.ConsumeIntegralInRange<int>(0, 24));
        } else if (op == 26) {
            g_spec_countdown = fdp.ConsumeIntegralInRange<int>(0, 12);
        } else if (op == 27) {
            corrupt_cache(p);
        } else if (op == 28) {
            plant_image(p);
        } else if (op == 29) {
            for (LiveEngine & le : p.live) {
                le.dirty = true;
            }
            check_all(p, "check", true);
        } else if (op == 30) {
            // The user opens the photo menu: the projector loads before the image.
            api_prepare_vision(pick_handle(p));
            check_all(p, "prepareVision", false);
        } else if (op == 31) {
            op_stage(p);
        } else {
            op_generate(p);
        }
        trace(opt, "op %d done", op);
    }

    if (opt.threads) {
        ui_done = true;
        ui.join();
    }
    fakejni::arm_alloc_failure(fakejni::env(), -1);
    g_spec_countdown = -1;
    check_all(p, "the end of the program", true);
    for (const LiveEngine & le : p.live) {
        if (const std::shared_ptr<Engine> e = engine_of(le.handle)) {
            g_counters.drafted  += (uint64_t) e->turn.drafted;
            g_counters.accepted += (uint64_t) e->turn.accepted;
        }
        api_free(le.handle);
    }
    g_counters.faults = fakejni::injected_faults(fakejni::env());
    g_prog = nullptr;
    cache_io::remove_tree(p.dir);
    fakejni::reset();
    return 0;
}

int speed_check(const Options & opt, int reps) {
    init_once(opt);
    Program p;
    p.opt = &opt;
    p.dir = opt.work_dir + "/speed-" + std::to_string(getpid());
    cache_io::make_dirs(p.dir);
    g_prog = &p;
    LoadSpec s;
    s.model   = opt.model_dir + "/tiny-qwen35-f32.gguf";
    s.n_ctx   = 1024;
    s.threads = 1;
    const std::vector<Msg> chat = {Msg{"user", u"Hello, how are you? Tell me about the cat.", -1}};
    for (const bool speculative : {false, true}) {
        s.speculative = speculative;
        const jlong h = api_load(s);
        if (h == 0) {
            fail("speed: the tiny model did not load");
        }
        std::vector<double> us_per_call;
        size_t tokens = 0;
        for (int r = 0; r < reps; ++r) {
            // Greedy sampling: each repetition gives the same answer.
            if (api_chat_start(p, h, chat, false, 0.0f, 0.8f, false, false) < 0) {
                fail("speed: chatStart failed");
            }
            std::vector<int8_t> piece;
            int n = 0;
            const auto t0 = std::chrono::steady_clock::now();
            while (n < 64 && api_generate_next(h, piece) == 1) {
                ++n;
            }
            const auto t1 = std::chrono::steady_clock::now();
            if (n > 0) {
                us_per_call.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / n);
                tokens = (size_t) n;
            }
        }
        api_free(h);
        if (us_per_call.empty()) {
            fail("speed: no answer gave a token");
        }
        std::sort(us_per_call.begin(), us_per_call.end());
        fprintf(stderr, "speed %s: %zu tokens, %zu runs, one generateNext median %.1f us, min %.1f us, max %.1f us\n",
                speculative ? "speculative" : "plain", tokens, us_per_call.size(), us_per_call[us_per_call.size() / 2],
                us_per_call.front(), us_per_call.back());
    }
    g_prog = nullptr;
    cache_io::remove_tree(p.dir);
    fakejni::reset();
    return 0;
}

}  // namespace harness
