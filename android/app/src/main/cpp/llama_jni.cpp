/**
 * The JNI layer between ai.airi.qwenmobile.LlamaNative and llama.cpp.
 *
 * One Engine holds one model, one context, one sampler, and one thread pool.
 * All calls for one engine come from one Kotlin thread. A mutex guards the
 * engine against a second caller.
 */

#include <android/log.h>
#include <dirent.h>
#include <jni.h>
#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <vector>

#ifndef QWEN_NO_OPENCL_INFO
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#endif

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "perf_hint.h"

#define TAG "QwenMobile"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

/** The target duration of one decode call for the ADPF session, 40 ms. */
constexpr int64_t kHintTargetNs = 40'000'000;

/** The number of tokens that the presence penalty looks back on. */
constexpr int32_t kPenaltyLastN = 256;

/** The maximum number of vision tokens of one image. 1024 tokens is a 1024 x 1024 image. */
constexpr int32_t kImageMaxTokens = 1024;

/** The nice value of the compute threads. The display thread keeps its priority. */
constexpr int kComputeNice = 10;

/** The number of new prompt tokens from which the hybrid backend uses the prefill context. */
constexpr int kHybridMinTokens = 128;

/** The batch of the prefill context. The Hexagon backend wants 512. */
constexpr int kPrefillBatch = 512;

/** Give a thread the compute priority. tid 0 is the calling thread. */
void lower_priority(int32_t tid) {
    if (setpriority(PRIO_PROCESS, tid, kComputeNice) != 0) {
        LOGE("setpriority for thread %d failed", tid);
    }
}

struct Engine {
    llama_model *     model = nullptr;
    llama_context *   ctx   = nullptr;
    llama_sampler *   smpl  = nullptr;
    ggml_threadpool * tp    = nullptr;
    /** The vision projector. It loads on the first image, on the device of the model. */
    mtmd_context *    mctx  = nullptr;
    std::string       mmproj;
    ggml_backend_dev_t device = nullptr;

    /**
     * The hybrid backend: a second copy of the model on the prefill device
     * (the NPU) with its own context. The prompt goes there, the state moves
     * to the decode context, and the answer comes from the decode device.
     */
    llama_model *      model_pf  = nullptr;
    llama_context *    ctx_pf    = nullptr;
    ggml_backend_dev_t device_pf = nullptr;
    /** True when the first answer token of the turn was an end token on the prefill context. */
    bool first_eog = false;
    /** The kind of the first answer token of a hybrid turn, for the next generateNext. */
    int first_kind = 0;
    /** The thinking tags of the vocabulary, or LLAMA_TOKEN_NULL. */
    llama_token tok_think_open  = LLAMA_TOKEN_NULL;
    llama_token tok_think_close = LLAMA_TOKEN_NULL;
    /** The state copies of the current turn, in microseconds. */
    int64_t transfer_us = 0;
    common_chat_templates_ptr tmpls;
    std::unique_ptr<PerfHintSession> hint;
    std::mutex mutex;

    /** The tokens that the model memory holds, in order. Empty when the memory holds images. */
    std::vector<llama_token> cache;
    /** True when the memory holds image chunks. A token prefix cannot extend such a memory. */
    bool memory_has_media = false;
    /** Bytes of an incomplete UTF-8 sequence from the last token. */
    std::string utf8_pending;

    int  n_threads  = 4;
    int  n_batch    = 512;
    int  gpu_layers = 0;
    bool thinking   = false;

    // Timings of the current turn, in microseconds.
    int64_t prefill_tokens = 0;
    int64_t prefill_us     = 0;
    int64_t gen_tokens     = 0;
    int64_t gen_us         = 0;
};

/** Current time in microseconds. */
int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

/** Send the ggml log to logcat with the matching priority. */
void log_to_logcat(ggml_log_level level, const char * text, void *) {
    int prio = ANDROID_LOG_INFO;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: prio = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  prio = ANDROID_LOG_WARN;  break;
        case GGML_LOG_LEVEL_DEBUG: prio = ANDROID_LOG_DEBUG; break;
        default: break;
    }
    __android_log_write(prio, "llama.cpp", text);
}

/** Throw a java.lang.RuntimeException with the message. */
void throw_java(JNIEnv * env, const std::string & msg) {
    LOGE("%s", msg.c_str());
    jclass cls = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(cls, msg.c_str());
}

/** Read the kernel thread identifiers of this process from /proc/self/task. */
std::set<int32_t> list_tids() {
    std::set<int32_t> tids;
    DIR * dir = opendir("/proc/self/task");
    if (dir == nullptr) {
        return tids;
    }
    while (dirent * entry = readdir(dir)) {
        if (entry->d_name[0] >= '0' && entry->d_name[0] <= '9') {
            tids.insert(atoi(entry->d_name));
        }
    }
    closedir(dir);
    return tids;
}

/** The id of a special token, or LLAMA_TOKEN_NULL when the text is not one token. */
llama_token single_token(const llama_vocab * vocab, const char * text) {
    const std::vector<llama_token> ids = common_tokenize(vocab, text, false, true);
    return ids.size() == 1 ? ids[0] : LLAMA_TOKEN_NULL;
}

/** The kind of a generated token for the app: 0 text, 1 the thinking opens, 2 the thinking closes. */
int token_kind(const Engine & e, llama_token token) {
    if (token == e.tok_think_open)  return 1;
    if (token == e.tok_think_close) return 2;
    return 0;
}

/** Length of the longest prefix of s that is complete UTF-8. */
size_t utf8_complete_prefix(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = s[i];
        size_t len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        if (i + len > s.size()) {
            break;
        }
        i += len;
    }
    return i;
}

/**
 * Build the sampler chain. Qwen3.5 recommends temperature 1.0 and top-p 0.95
 * with thinking, 0.7 and 0.8 without. A temperature of 0 is greedy.
 */
void rebuild_sampler(Engine & e, bool thinking, float temp, float top_p) {
    if (e.smpl != nullptr) {
        llama_sampler_free(e.smpl);
    }
    auto params = llama_sampler_chain_default_params();
    params.no_perf = true;
    e.smpl = llama_sampler_chain_init(params);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(e.model));
    llama_sampler_chain_add(e.smpl, llama_sampler_init_penalties(n_vocab, kPenaltyLastN, 1.0f, 0.0f, 1.5f));
    if (temp <= 0.0f) {
        llama_sampler_chain_add(e.smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(e.smpl, llama_sampler_init_top_k(20));
        llama_sampler_chain_add(e.smpl, llama_sampler_init_top_p(std::min(std::max(top_p, 0.05f), 1.0f), 1));
        llama_sampler_chain_add(e.smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(e.smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    e.thinking = thinking;
}

/**
 * Decode tokens in chunks of n_batch. Each chunk goes to the ADPF session
 * with its measured duration. Returns the llama_decode code, 0 on success.
 */
int decode_tokens(Engine & e, llama_context * ctx, const llama_token * tokens, int n) {
    const int n_batch = ctx == e.ctx_pf ? kPrefillBatch : e.n_batch;
    for (int i = 0; i < n; i += n_batch) {
        const int count = std::min(n_batch, n - i);
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens) + i, count);
        const int64_t t0 = now_us();
        const int rc = llama_decode(ctx, batch);
        if (e.hint) {
            e.hint->report((now_us() - t0) * 1000);
        }
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}

/** Decode one token on the decode context and append it to the memory cache. */
int decode_one(Engine & e, llama_token token) {
    const int rc = decode_tokens(e, e.ctx, &token, 1);
    if (rc == 0) {
        e.cache.push_back(token);
    }
    return rc;
}

/** Empty the model memory of both contexts and the record of what it holds. */
void clear_memory(Engine & e) {
    llama_memory_clear(llama_get_memory(e.ctx), true);
    if (e.ctx_pf != nullptr) {
        llama_memory_clear(llama_get_memory(e.ctx_pf), true);
    }
    e.cache.clear();
    e.memory_has_media = false;
}

/**
 * Copy the sequence state (the KV cache of the attention layers and the
 * recurrent states) from one context to the other. The destination loses
 * its own state first. Returns false with the error text set.
 */
bool transfer_state(llama_context * src, llama_context * dst, std::string & error) {
    const size_t size = llama_state_seq_get_size(src, 0);
    std::vector<uint8_t> buffer(size);
    const size_t got = llama_state_seq_get_data(src, buffer.data(), size, 0);
    llama_memory_seq_rm(llama_get_memory(dst), 0, -1, -1);
    if (got == 0 || llama_state_seq_set_data(dst, buffer.data(), got, 0) == 0) {
        error = "The state transfer between the prefill and the decode context failed (" + std::to_string(size) + " bytes)";
        return false;
    }
    return true;
}

/**
 * The end of a hybrid prefill: sample the first answer token on the prefill
 * context, move the state to the decode context, and decode that token
 * there. Thus generateNext samples the second token from the decode context.
 */
bool hybrid_finish(Engine & e, std::string & error) {
    const llama_token first = llama_sampler_sample(e.smpl, e.ctx_pf, -1);
    const int64_t t0 = now_us();
    if (!transfer_state(e.ctx_pf, e.ctx, error)) {
        return false;
    }
    e.transfer_us += now_us() - t0;
    if (llama_vocab_is_eog(llama_model_get_vocab(e.model), first)) {
        e.first_eog = true;
        return true;
    }
    e.first_kind = token_kind(e, first);
    if (e.first_kind == 0) {
        e.utf8_pending += common_token_to_piece(e.ctx, first, true);
    }
    const int64_t t1 = now_us();
    if (decode_one(e, first) != 0) {
        error = "llama_decode failed on the first answer token";
        return false;
    }
    e.gen_tokens = 1;
    e.gen_us     = now_us() - t1;
    return true;
}

/**
 * Decode new prompt tokens on the prefill context. With ``extend`` the
 * decode context holds the earlier turns, and its state moves to the
 * prefill context first.
 */
bool hybrid_prefill_tokens(Engine & e, const llama_token * tokens, int n, bool extend, std::string & error) {
    const int64_t t0 = now_us();
    if (extend) {
        if (!transfer_state(e.ctx, e.ctx_pf, error)) {
            return false;
        }
    } else {
        llama_memory_clear(llama_get_memory(e.ctx_pf), true);
    }
    e.transfer_us = now_us() - t0;
    const int rc = decode_tokens(e, e.ctx_pf, tokens, n);
    if (rc != 0) {
        error = "llama_decode failed on the prompt on the prefill device with code " + std::to_string(rc);
        return false;
    }
    return hybrid_finish(e, error);
}

/** Load the vision projector. Returns false when the model has none or it does not load. */
bool ensure_vision(Engine & e) {
    if (e.mctx != nullptr) {
        return true;
    }
    if (e.mmproj.empty()) {
        return false;
    }
    // The encoder runs on the prefill device when there is one, else on the device of the model.
    ggml_backend_dev_t dev = e.device_pf != nullptr ? e.device_pf : e.device;
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu          = dev != nullptr;
    mp.device           = dev;
    mp.n_threads        = e.n_threads;
    mp.print_timings    = false;
    mp.warmup           = false;
    mp.image_max_tokens = kImageMaxTokens;
    const int64_t t0 = now_us();
    e.mctx = mtmd_init_from_file(e.mmproj.c_str(), e.model, mp);
    if (e.mctx == nullptr) {
        LOGE("the vision projector did not load: %s", e.mmproj.c_str());
        return false;
    }
    LOGI("vision projector loaded in %.0f ms: %s", (now_us() - t0) / 1000.0, e.mmproj.c_str());
    return true;
}

/**
 * Decode a prompt that holds images into an empty memory. The prompt has
 * one media marker per image, in order. Returns the number of prompt
 * tokens, or -1 with the error text set.
 */
int64_t prefill_with_images(JNIEnv * env, Engine & e, const std::string & prompt,
                            const std::vector<jbyteArray> & images, std::string & error) {
    if (!ensure_vision(e)) {
        error = "This model has no vision projector (mmproj) next to it";
        return -1;
    }
    std::vector<mtmd_bitmap *> bitmaps;
    auto free_bitmaps = [&bitmaps] {
        for (mtmd_bitmap * b : bitmaps) mtmd_bitmap_free(b);
        bitmaps.clear();
    };
    for (jbyteArray image : images) {
        const jsize len = env->GetArrayLength(image);
        jbyte * bytes = env->GetByteArrayElements(image, nullptr);
        const mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_buf(
            e.mctx, reinterpret_cast<const unsigned char *>(bytes), (size_t) len, false,
            mtmd_helper_init_opt_default());
        env->ReleaseByteArrayElements(image, bytes, JNI_ABORT);
        if (w.bitmap == nullptr) {
            free_bitmaps();
            error = "The image did not decode";
            return -1;
        }
        bitmaps.push_back(w.bitmap);
    }

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text text;
    text.text          = prompt.c_str();
    text.text_len      = prompt.size();
    text.add_special   = true;
    text.parse_special = true;
    const int32_t tk = mtmd_tokenize(e.mctx, chunks, &text, bitmaps.data(), bitmaps.size());
    free_bitmaps();
    if (tk != 0) {
        mtmd_input_chunks_free(chunks);
        error = tk == 1 ? "The number of images differs from the number of markers in the prompt"
                        : "The image preprocessing failed";
        return -1;
    }
    const llama_pos n_pos = mtmd_helper_get_n_pos(chunks);
    if ((uint32_t) n_pos + 8 >= llama_n_ctx(e.ctx)) {
        mtmd_input_chunks_free(chunks);
        error = "The conversation is longer than the context (" + std::to_string(n_pos) + " positions)";
        return -1;
    }

    clear_memory(e);
    e.memory_has_media = true;
    const bool hybrid = e.ctx_pf != nullptr;
    llama_context * lctx = hybrid ? e.ctx_pf : e.ctx;
    llama_pos new_n_past = 0;
    const int64_t t0 = now_us();
    const int32_t rc = mtmd_helper_eval_chunks(e.mctx, lctx, chunks, 0, 0, hybrid ? kPrefillBatch : e.n_batch, true,
                                               &new_n_past);
    if (e.hint) {
        e.hint->report((now_us() - t0) * 1000);
    }
    const size_t n_tokens = mtmd_helper_get_n_tokens(chunks);
    mtmd_input_chunks_free(chunks);
    if (rc != 0) {
        clear_memory(e);
        error = "The prompt with images did not decode, code " + std::to_string(rc);
        return -1;
    }
    if (hybrid && !hybrid_finish(e, error)) {
        clear_memory(e);
        return -1;
    }
    LOGI("image prefill: %zu images, %zu tokens, %d positions, %.0f ms on %s", images.size(), n_tokens, (int) new_n_past,
         (now_us() - t0) / 1000.0, hybrid ? ggml_backend_dev_name(e.device_pf) : e.device ? ggml_backend_dev_name(e.device) : "CPU");
    return (int64_t) n_tokens;
}

/** Mean and sample standard deviation of the values. */
std::pair<double, double> mean_std(const std::vector<double> & v) {
    if (v.empty()) {
        return {0.0, 0.0};
    }
    double sum = 0.0;
    for (double x : v) sum += x;
    const double mean = sum / v.size();
    if (v.size() < 2) {
        return {mean, 0.0};
    }
    double sq = 0.0;
    for (double x : v) sq += (x - mean) * (x - mean);
    return {mean, std::sqrt(sq / (v.size() - 1))};
}

Engine * engine_of(jlong handle) {
    return reinterpret_cast<Engine *>(handle);
}

/**
 * The result of generateNext: byte 0 is the kind of the token (0 text,
 * 1 the thinking opens, 2 the thinking closes), then the complete UTF-8
 * bytes of the pending text. The tag tokens add no text.
 */
jbyteArray pack_piece(JNIEnv * env, Engine & e, int kind) {
    const size_t complete = utf8_complete_prefix(e.utf8_pending);
    jbyteArray out = env->NewByteArray((jsize) (1 + complete));
    const jbyte k = (jbyte) kind;
    env->SetByteArrayRegion(out, 0, 1, &k);
    env->SetByteArrayRegion(out, 1, (jsize) complete, reinterpret_cast<const jbyte *>(e.utf8_pending.data()));
    e.utf8_pending.erase(0, complete);
    return out;
}

std::string jstring_to_std(JNIEnv * env, jstring s) {
    if (s == nullptr) {
        return {};
    }
    const char * chars = env->GetStringUTFChars(s, nullptr);
    std::string out(chars);
    env->ReleaseStringUTFChars(s, chars);
    return out;
}

} // namespace

extern "C" {

/**
 * Initialize the backends. With a library directory, the dynamic backends
 * load from it, and the Hexagon backend finds the DSP library there through
 * ADSP_LIBRARY_PATH.
 */
JNIEXPORT void JNICALL
Java_ai_airi_qwenmobile_LlamaNative_init(JNIEnv * env, jclass, jstring jlibdir) {
    llama_log_set(log_to_logcat, nullptr);
    // The op fusion of the Hexagon backend breaks the single-token decode
    // (correct prefill, garbage after the first tokens). It stays off.
    setenv("GGML_HEXAGON_OPFUSION", "0", 0);
    const std::string libdir = jstring_to_std(env, jlibdir);
    if (!libdir.empty()) {
        setenv("ADSP_LIBRARY_PATH", libdir.c_str(), 1);
        ggml_backend_load_all_from_path(libdir.c_str());
    }
    llama_backend_init();
    LOGI("llama.cpp initialized, %zu backend devices, libdir %s", ggml_backend_dev_count(), libdir.c_str());
}

/**
 * Make the directory the current one. The OpenCL profiling build writes
 * cl_profiling.csv into the current directory when the backend closes.
 */
JNIEXPORT void JNICALL
Java_ai_airi_qwenmobile_LlamaNative_setWorkingDirectory(JNIEnv * env, jclass, jstring jpath) {
    const std::string path = jstring_to_std(env, jpath);
    if (chdir(path.c_str()) != 0) {
        LOGE("chdir to %s failed", path.c_str());
    }
}

/** The driver version and the extensions of the first OpenCL GPU, or an empty string. */
static std::string opencl_device_info() {
#ifdef QWEN_NO_OPENCL_INFO
    return {};
#else
    cl_platform_id platform = nullptr;
    cl_device_id   dev      = nullptr;
    if (clGetPlatformIDs(1, &platform, nullptr) != CL_SUCCESS ||
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) != CL_SUCCESS) {
        return {};
    }
    auto str = [&](cl_device_info what) {
        size_t size = 0;
        clGetDeviceInfo(dev, what, 0, nullptr, &size);
        std::string s(size, '\0');
        clGetDeviceInfo(dev, what, size, s.data(), nullptr);
        while (!s.empty() && (s.back() == '\0' || s.back() == ' ')) s.pop_back();
        return s;
    };
    cl_uint  cu = 0, clk = 0;
    cl_ulong lmem = 0, cache = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clk), &clk, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cache), &cache, nullptr);
    std::string out = "opencl: " + str(CL_DEVICE_VERSION) + ", driver " + str(CL_DRIVER_VERSION) +
                      ", " + std::to_string(cu) + " CU, " + std::to_string(clk) + " MHz, local " +
                      std::to_string(lmem >> 10) + " KB, cache " + std::to_string(cache >> 10) + " KB\n";
    out += "extensions: " + str(CL_DEVICE_EXTENSIONS) + "\n";
    return out;
#endif
}

JNIEXPORT jstring JNICALL
Java_ai_airi_qwenmobile_LlamaNative_devices(JNIEnv * env, jclass) {
    std::string out = opencl_device_info();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        size_t mem_free = 0, mem_total = 0;
        ggml_backend_dev_memory(dev, &mem_free, &mem_total);
        const char * type = "other";
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   type = "CPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   type = "GPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: type = "ACCEL"; break;
            default: break;
        }
        char line[512];
        snprintf(line, sizeof(line), "%s: %s (%s), %zu MB free of %zu MB\n",
                 ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), type,
                 mem_free >> 20, mem_total >> 20);
        out += line;
    }
    // One log line per item, because logcat cuts a line at 4 KB.
    for (size_t pos = 0; pos < out.size();) {
        size_t end = out.find_first_of(" \n", pos);
        if (end == std::string::npos) end = out.size();
        if (end > pos) LOGI("%s", out.substr(pos, end - pos).c_str());
        pos = end + 1;
    }
    return env->NewStringUTF(out.c_str());
}

JNIEXPORT jlong JNICALL
Java_ai_airi_qwenmobile_LlamaNative_load(JNIEnv * env, jclass, jstring jpath, jstring jmmproj,
                                          jstring jdevice, jstring jprefill, jint gpu_layers, jint n_threads,
                                          jint n_ctx) {
    const std::string path    = jstring_to_std(env, jpath);
    const std::string device  = jstring_to_std(env, jdevice);
    const std::string prefill = jstring_to_std(env, jprefill);
    auto e = std::make_unique<Engine>();
    e->n_threads  = std::max(1, (int) n_threads);
    e->gpu_layers = gpu_layers;
    e->mmproj     = jstring_to_std(env, jmmproj);

    // The device by its ggml name: GPUOpenCL for the Adreno, HTP0 for the Hexagon NPU.
    std::vector<ggml_backend_dev_t> devices;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    if (!device.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
        if (dev == nullptr) {
            throw_java(env, "The device is not available: " + device);
            return 0;
        }
        devices = {dev, nullptr};
        mp.devices = devices.data();
        e->device  = dev;
    }
    e->model = llama_model_load_from_file(path.c_str(), mp);
    if (e->model == nullptr) {
        throw_java(env, "The model did not load: " + path);
        return 0;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = e->n_batch;
    cp.n_ubatch        = e->n_batch;
    cp.n_threads       = e->n_threads;
    cp.n_threads_batch = e->n_threads;
    cp.no_perf         = false;
    e->ctx = llama_init_from_model(e->model, cp);
    if (e->ctx == nullptr) {
        llama_model_free(e->model);
        throw_java(env, "The context did not initialize (n_ctx=" + std::to_string(n_ctx) + ")");
        return 0;
    }

    // The hybrid backend: the same file again on the prefill device, with the prefill batch.
    if (!prefill.empty()) {
        e->device_pf = ggml_backend_dev_by_name(prefill.c_str());
        if (e->device_pf == nullptr) {
            llama_free(e->ctx);
            llama_model_free(e->model);
            throw_java(env, "The prefill device is not available: " + prefill);
            return 0;
        }
        std::vector<ggml_backend_dev_t> devices_pf = {e->device_pf, nullptr};
        llama_model_params mp_pf = llama_model_default_params();
        mp_pf.n_gpu_layers = 999;
        mp_pf.devices      = devices_pf.data();
        e->model_pf = llama_model_load_from_file(path.c_str(), mp_pf);
        llama_context_params cp_pf = cp;
        cp_pf.n_batch  = kPrefillBatch;
        cp_pf.n_ubatch = kPrefillBatch;
        e->ctx_pf = e->model_pf ? llama_init_from_model(e->model_pf, cp_pf) : nullptr;
        if (e->ctx_pf == nullptr) {
            if (e->model_pf) llama_model_free(e->model_pf);
            llama_free(e->ctx);
            llama_model_free(e->model);
            throw_java(env, "The prefill model did not load on " + prefill);
            return 0;
        }
    }

    // The thread pool exists before the first decode, thus its thread ids are
    // known and go into the ADPF session together with the caller thread.
    // The workers block between graphs and run at the compute priority, thus
    // the display thread keeps the cores it needs.
    const std::set<int32_t> before = list_tids();
    ggml_threadpool_params tpp = ggml_threadpool_params_default(e->n_threads);
    tpp.prio       = GGML_SCHED_PRIO_LOW;
    tpp.poll       = 0;
    tpp.strict_cpu = false;
    e->tp = ggml_threadpool_new(&tpp);
    llama_attach_threadpool(e->ctx, e->tp, e->tp);
    if (e->ctx_pf != nullptr) {
        llama_attach_threadpool(e->ctx_pf, e->tp, e->tp);
    }
    std::vector<int32_t> tids;
    for (int32_t tid : list_tids()) {
        if (before.count(tid) == 0) {
            tids.push_back(tid);
            lower_priority(tid);
        }
    }
    lower_priority(0);
    tids.push_back(gettid());
    e->hint = std::make_unique<PerfHintSession>(tids, kHintTargetNs);

    e->tmpls = common_chat_templates_init(e->model, "");
    e->tok_think_open  = single_token(llama_model_get_vocab(e->model), "<think>");
    e->tok_think_close = single_token(llama_model_get_vocab(e->model), "</think>");
    rebuild_sampler(*e, false, 0.7f, 0.8f);

    LOGI("model loaded: %s, device=%s, prefill=%s, gpu_layers=%d, threads=%d, n_ctx=%u, mmproj=%s",
         path.c_str(), device.empty() ? "cpu" : device.c_str(), prefill.empty() ? "same" : prefill.c_str(),
         gpu_layers, e->n_threads, llama_n_ctx(e->ctx), e->mmproj.empty() ? "none" : e->mmproj.c_str());
    return reinterpret_cast<jlong>(e.release());
}

JNIEXPORT void JNICALL
Java_ai_airi_qwenmobile_LlamaNative_free(JNIEnv *, jclass, jlong handle) {
    Engine * e = engine_of(handle);
    if (e == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(e->mutex);
        e->hint.reset();
        if (e->smpl) llama_sampler_free(e->smpl);
        if (e->mctx) mtmd_free(e->mctx);
        if (e->ctx_pf) {
            llama_detach_threadpool(e->ctx_pf);
            llama_free(e->ctx_pf);
        }
        if (e->model_pf) llama_model_free(e->model_pf);
        if (e->ctx) {
            llama_detach_threadpool(e->ctx);
            llama_free(e->ctx);
        }
        if (e->tp) ggml_threadpool_free(e->tp);
        if (e->model) llama_model_free(e->model);
    }
    delete e;
}

JNIEXPORT jstring JNICALL
Java_ai_airi_qwenmobile_LlamaNative_modelInfo(JNIEnv * env, jclass, jlong handle) {
    Engine * e = engine_of(handle);
    char desc[256];
    llama_model_desc(e->model, desc, sizeof(desc));
    char line[512];
    snprintf(line, sizeof(line), "%s, %.2f GiB, %.2f B params, n_ctx %u, gpu layers %d, threads %d, ADPF %s, vision %s",
             desc, llama_model_size(e->model) / 1073741824.0, llama_model_n_params(e->model) / 1e9,
             llama_n_ctx(e->ctx), e->gpu_layers, e->n_threads, e->hint && e->hint->ok() ? "on" : "off",
             e->mmproj.empty() ? "none" : (e->mctx ? "loaded" : "ready"));
    return env->NewStringUTF(line);
}

JNIEXPORT jint JNICALL
Java_ai_airi_qwenmobile_LlamaNative_chatStart(JNIEnv * env, jclass, jlong handle,
                                               jobjectArray roles, jobjectArray contents,
                                               jobjectArray images, jboolean thinking,
                                               jfloat temperature, jfloat top_p) {
    Engine * e = engine_of(handle);
    std::lock_guard<std::mutex> lock(e->mutex);

    // A message with an image starts with the media marker. mtmd replaces
    // the marker with the vision tokens of that image, in message order.
    common_chat_templates_inputs inputs;
    std::vector<jbyteArray> image_refs;
    const jsize n = env->GetArrayLength(roles);
    for (jsize i = 0; i < n; ++i) {
        common_chat_msg msg;
        msg.role    = jstring_to_std(env, (jstring) env->GetObjectArrayElement(roles, i));
        msg.content = jstring_to_std(env, (jstring) env->GetObjectArrayElement(contents, i));
        jbyteArray image = images ? (jbyteArray) env->GetObjectArrayElement(images, i) : nullptr;
        if (image != nullptr) {
            msg.content = std::string(mtmd_default_marker()) + "\n" + msg.content;
            image_refs.push_back(image);
        }
        inputs.messages.push_back(std::move(msg));
    }
    inputs.add_generation_prompt = true;
    inputs.use_jinja             = true;
    inputs.enable_thinking       = thinking;

    std::string prompt;
    try {
        prompt = common_chat_templates_apply(e->tmpls.get(), inputs).prompt;
    } catch (const std::exception & ex) {
        throw_java(env, std::string("The chat template failed: ") + ex.what());
        return -1;
    }

    rebuild_sampler(*e, thinking, temperature, top_p);
    e->utf8_pending.clear();
    e->gen_tokens  = 0;
    e->gen_us      = 0;
    e->first_eog   = false;
    e->first_kind  = 0;
    e->transfer_us = 0;

    if (!image_refs.empty()) {
        std::string error;
        const int64_t t0 = now_us();
        const int64_t n_tokens = prefill_with_images(env, *e, prompt, image_refs, error);
        if (n_tokens < 0) {
            throw_java(env, error);
            return -1;
        }
        e->prefill_us     = now_us() - t0;
        e->prefill_tokens = n_tokens;
        return (jint) n_tokens;
    }

    const std::vector<llama_token> tokens = common_tokenize(llama_model_get_vocab(e->model), prompt, true, true);
    if (tokens.size() + 8 >= llama_n_ctx(e->ctx)) {
        throw_java(env, "The conversation is longer than the context (" + std::to_string(tokens.size()) + " tokens)");
        return -1;
    }

    // The memory keeps the previous turns when the new prompt extends them.
    // A recurrent state cannot roll back, thus any other case starts again.
    if (e->memory_has_media) {
        clear_memory(*e);
    }
    size_t common = 0;
    while (common < e->cache.size() && common < tokens.size() && e->cache[common] == tokens[common]) {
        ++common;
    }
    size_t start = 0;
    if (common == e->cache.size() && common < tokens.size()) {
        start = common;
    } else {
        clear_memory(*e);
    }

    const int64_t t0 = now_us();
    const int n_new = (int) (tokens.size() - start);
    std::string error;
    bool ok;
    if (e->ctx_pf != nullptr && n_new >= kHybridMinTokens) {
        // Both contexts hold these tokens after the prefill. The first answer token joins the cache.
        e->cache = tokens;
        ok = hybrid_prefill_tokens(*e, tokens.data() + start, n_new, start > 0, error);
    } else {
        const int rc = decode_tokens(*e, e->ctx, tokens.data() + start, n_new);
        ok = rc == 0;
        if (!ok) {
            error = "llama_decode failed on the prompt with code " + std::to_string(rc);
        }
        e->cache = tokens;
    }
    if (!ok) {
        clear_memory(*e);
        throw_java(env, error);
        return -1;
    }
    e->prefill_us     = now_us() - t0;
    e->prefill_tokens = n_new;
    return (jint) e->prefill_tokens;
}

JNIEXPORT jbyteArray JNICALL
Java_ai_airi_qwenmobile_LlamaNative_generateNext(JNIEnv * env, jclass, jlong handle) {
    Engine * e = engine_of(handle);
    std::lock_guard<std::mutex> lock(e->mutex);
    const llama_vocab * vocab = llama_model_get_vocab(e->model);
    if (e->first_eog) {
        // The prefill context sampled the end token as the first answer token.
        e->first_eog = false;
        return nullptr;
    }
    if (e->first_kind != 0) {
        // The first token of a hybrid turn was a thinking tag: report it before the next sample.
        const int kind = e->first_kind;
        e->first_kind = 0;
        return pack_piece(env, *e, kind);
    }

    const int64_t t0 = now_us();
    const llama_token token = llama_sampler_sample(e->smpl, e->ctx, -1);
    if (llama_vocab_is_eog(vocab, token)) {
        // The end token goes into the memory, thus the next turn extends this one.
        decode_one(*e, token);
        return nullptr;
    }
    const int kind = token_kind(*e, token);
    if (kind == 0) {
        e->utf8_pending += common_token_to_piece(e->ctx, token, true);
    }
    if (decode_one(*e, token) != 0) {
        throw_java(env, "llama_decode failed during generation");
        return nullptr;
    }
    e->gen_tokens += 1;
    e->gen_us     += now_us() - t0;
    return pack_piece(env, *e, kind);
}

JNIEXPORT jstring JNICALL
Java_ai_airi_qwenmobile_LlamaNative_stats(JNIEnv * env, jclass, jlong handle) {
    Engine * e = engine_of(handle);
    std::lock_guard<std::mutex> lock(e->mutex);
    char line[256];
    const double pp = e->prefill_us > 0 ? e->prefill_tokens * 1e6 / e->prefill_us : 0.0;
    const double tg = e->gen_us > 0 ? e->gen_tokens * 1e6 / e->gen_us : 0.0;
    const llama_pos n_past = llama_memory_seq_pos_max(llama_get_memory(e->ctx), 0) + 1;
    const char * pf_dev = e->device_pf ? ggml_backend_dev_name(e->device_pf) : e->device ? ggml_backend_dev_name(e->device) : "CPU";
    snprintf(line, sizeof(line), "prefill %lld tok in %.0f ms (%.1f t/s on %s%s), generate %lld tok (%.1f t/s), memory %d pos",
             (long long) e->prefill_tokens, e->prefill_us / 1000.0, pp, pf_dev,
             e->device_pf ? (", transfer " + std::to_string(e->transfer_us / 1000) + " ms").c_str() : "",
             (long long) e->gen_tokens, tg, (int) n_past);
    return env->NewStringUTF(line);
}

JNIEXPORT void JNICALL
Java_ai_airi_qwenmobile_LlamaNative_resetChat(JNIEnv *, jclass, jlong handle) {
    Engine * e = engine_of(handle);
    std::lock_guard<std::mutex> lock(e->mutex);
    clear_memory(*e);
    e->utf8_pending.clear();
    e->prefill_tokens = e->prefill_us = e->gen_tokens = e->gen_us = 0;
}

/**
 * The llama-bench method: pp tokens in batches of n_batch, then tg tokens
 * one at a time, each on a clean memory, reps times. The chat memory is
 * empty after the benchmark.
 */
JNIEXPORT jstring JNICALL
Java_ai_airi_qwenmobile_LlamaNative_bench(JNIEnv * env, jclass, jlong handle,
                                           jint pp, jint tg, jint reps) {
    Engine * e = engine_of(handle);
    std::lock_guard<std::mutex> lock(e->mutex);
    llama_memory_t mem = llama_get_memory(e->ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(e->model));
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> pick(0, n_vocab - 1);

    std::vector<double> pp_tps, tg_tps;
    std::string log;
    for (int r = 0; r < reps; ++r) {
        if (pp > 0) {
            std::vector<llama_token> tokens(pp);
            for (auto & t : tokens) t = pick(rng);
            // The hybrid backend prefills on its prefill context.
            llama_context * pctx = e->ctx_pf ? e->ctx_pf : e->ctx;
            llama_memory_clear(llama_get_memory(pctx), true);
            const int64_t t0 = now_us();
            const int rc = decode_tokens(*e, pctx, tokens.data(), pp);
            llama_synchronize(pctx);
            const int64_t dt = now_us() - t0;
            if (rc != 0) {
                log += "pp decode failed with code " + std::to_string(rc) + "\n";
                break;
            }
            pp_tps.push_back(pp * 1e6 / dt);
        }
        if (tg > 0) {
            llama_memory_clear(mem, true);
            const int64_t t0 = now_us();
            int rc = 0;
            for (int i = 0; i < tg && rc == 0; ++i) {
                llama_token t = pick(rng);
                rc = decode_tokens(*e, e->ctx, &t, 1);
                llama_synchronize(e->ctx);
            }
            const int64_t dt = now_us() - t0;
            if (rc != 0) {
                log += "tg decode failed with code " + std::to_string(rc) + "\n";
                break;
            }
            tg_tps.push_back(tg * 1e6 / dt);
        }
    }
    clear_memory(*e);

    char desc[128];
    llama_model_desc(e->model, desc, sizeof(desc));
    char line[256];
    snprintf(line, sizeof(line), "%s | gpu layers %d | threads %d | reps %d\n", desc, e->gpu_layers, e->n_threads, reps);
    log += line;
    if (!pp_tps.empty()) {
        auto [m, s] = mean_std(pp_tps);
        snprintf(line, sizeof(line), "pp%d: %.2f ± %.2f t/s\n", pp, m, s);
        log += line;
    }
    if (!tg_tps.empty()) {
        auto [m, s] = mean_std(tg_tps);
        snprintf(line, sizeof(line), "tg%d: %.2f ± %.2f t/s\n", tg, m, s);
        log += line;
    }
    return env->NewStringUTF(log.c_str());
}

} // extern "C"
