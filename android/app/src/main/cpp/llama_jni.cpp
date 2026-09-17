/**
 * The JNI layer between ai.airi.qwenmobile.LlamaNative and llama.cpp.
 *
 * One Engine holds one model, one context, one sampler, and one thread pool.
 * All calls for one engine come from one Kotlin thread. A mutex guards the
 * engine against a second caller.
 *
 * The model memory of Qwen3.5 is a recurrent state plus a KV cache. A
 * recurrent state cannot roll back, thus the engine keeps a snapshot of the
 * state at the end of the prompt without the generation prompt. The chat
 * template renders the previous answer differently from the generated
 * tokens, thus the next prompt extends that snapshot and not the answer.
 * The hybrid backend keeps the snapshot on the prefill context. The other
 * backends keep it in a second sequence of the decode context, which
 * shares its KV cells and copies the recurrent state on the next write.
 */

#include <android/log.h>
#include <dirent.h>
#include <jni.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
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

/** The maximum number of vision tokens of one image. 576 tokens is a 768 x 768 image. */
constexpr int32_t kImageMaxTokens = 576;

/** The maximum size of the cache of encoded images, in bytes. One image is 4 to 10 MB. */
constexpr size_t kImageCacheBytes = 32u << 20;

/** The nice value of the compute threads. The display thread keeps its priority. */
constexpr int kComputeNice = 10;

/** The batch of the contexts. The Hexagon backend wants 512. */
constexpr int kBatch = 512;

/** The number of free positions that a prompt must leave in the context. */
constexpr size_t kContextHeadroom = 8;

/** The sequence of the conversation, and the sequence of the prompt snapshot. */
constexpr llama_seq_id kSeqMain = 0;
constexpr llama_seq_id kSeqSnap = 1;

/** Give a thread the compute priority. tid 0 is the calling thread. */
void lower_priority(int32_t tid) {
    if (setpriority(PRIO_PROCESS, tid, kComputeNice) != 0) {
        LOGE("setpriority for thread %d failed", tid);
    }
}

/** One unit of the model memory: a text token, or an image chunk identified by the hash of its bitmap. */
struct MemItem {
    llama_token token = LLAMA_TOKEN_NULL;
    std::string image_id;

    bool operator==(const MemItem & o) const { return token == o.token && image_id == o.image_id; }
    bool operator!=(const MemItem & o) const { return !(*this == o); }
};

/** The output of the vision encoder for one image. */
struct ImageEmbd {
    std::string        id;
    std::vector<float> embd;
};

struct Engine {
    llama_model *     model = nullptr;
    llama_context *   ctx   = nullptr;
    llama_sampler *   smpl  = nullptr;
    ggml_threadpool * tp    = nullptr;
    /** The vision projector. It loads on the first image, on the OpenCL GPU. */
    mtmd_context *    mctx  = nullptr;
    std::string       mmproj;
    ggml_backend_dev_t device = nullptr;
    /** The batch of the text decodes, kBatch tokens, allocated one time. */
    llama_batch batch = {};

    /**
     * The hybrid backend: a second copy of the model on the prefill device
     * (the NPU) with its own context. The prompt goes there, the state moves
     * to the decode context, and the answer comes from the decode device.
     */
    llama_model *      model_pf  = nullptr;
    llama_context *    ctx_pf    = nullptr;
    ggml_backend_dev_t device_pf = nullptr;
    /** The thinking tags of the vocabulary, or LLAMA_TOKEN_NULL. */
    llama_token tok_think_open  = LLAMA_TOKEN_NULL;
    llama_token tok_think_close = LLAMA_TOKEN_NULL;
    common_chat_templates_ptr tmpls;
    std::unique_ptr<PerfHintSession> hint;
    std::mutex mutex;

    /** The items that the conversation sequence of the decode context holds, in order. */
    std::vector<MemItem> cache;
    /** The number of positions that the conversation sequence holds. */
    llama_pos n_past = 0;
    /**
     * The snapshot: the items of the last prompt without its generation
     * prompt. The hybrid backend holds this state on the prefill context,
     * the other backends in sequence kSeqSnap of the decode context.
     */
    std::vector<MemItem> snap;
    llama_pos snap_n_past = 0;
    /** The encoded images, oldest first. The total is less than kImageCacheBytes. */
    std::vector<ImageEmbd> image_cache;
    size_t image_cache_bytes = 0;
    /** Bytes of an incomplete UTF-8 sequence from the last token. */
    std::string utf8_pending;

    int  n_threads  = 4;
    int  gpu_layers = 0;
    bool thinking   = false;

    // Timings of the current turn, in microseconds.
    int64_t prefill_tokens = 0;
    int64_t prefill_us     = 0;
    int64_t transfer_us    = 0;
    int64_t gen_tokens     = 0;
    int64_t gen_us         = 0;

    Engine() = default;
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    /** Release every native object, in the order of their dependencies. */
    ~Engine() {
        hint.reset();
        if (smpl != nullptr) llama_sampler_free(smpl);
        if (mctx != nullptr) mtmd_free(mctx);
        if (ctx_pf != nullptr) {
            llama_detach_threadpool(ctx_pf);
            llama_free(ctx_pf);
        }
        if (model_pf != nullptr) llama_model_free(model_pf);
        if (ctx != nullptr) {
            llama_detach_threadpool(ctx);
            llama_free(ctx);
        }
        if (tp != nullptr) ggml_threadpool_free(tp);
        if (model != nullptr) llama_model_free(model);
        if (batch.token != nullptr) llama_batch_free(batch);
    }
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

/** Append the text of a token to the pending UTF-8 bytes, without a heap allocation for the usual token. */
void append_piece(Engine & e, llama_token token) {
    const llama_vocab * vocab = llama_model_get_vocab(e.model);
    char buf[128];
    int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n >= 0) {
        e.utf8_pending.append(buf, (size_t) n);
        return;
    }
    // The piece is longer than the buffer: the negative value is its length.
    std::string big((size_t) -n, '\0');
    n = llama_token_to_piece(vocab, token, big.data(), (int32_t) big.size(), 0, true);
    if (n > 0) {
        e.utf8_pending.append(big.data(), (size_t) n);
    }
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

/** Tell the ADPF session the duration of one unit of work that started at t0. */
void report_hint(Engine & e, int64_t t0) {
    if (e.hint) {
        e.hint->report((now_us() - t0) * 1000);
    }
}

/**
 * Decode text tokens into sequence kSeqMain of a context, in chunks of
 * kBatch, at explicit positions from pos0. Only the last token gives logits,
 * and only with logits_last. Returns the llama_decode code, 0 on success.
 */
int decode_text(Engine & e, llama_context * lctx, const llama_token * tokens, int n, llama_pos pos0, bool logits_last) {
    llama_batch & b = e.batch;
    for (int i = 0; i < n; i += kBatch) {
        const int count = std::min(kBatch, n - i);
        for (int j = 0; j < count; ++j) {
            b.token[j]     = tokens[i + j];
            b.pos[j]       = pos0 + i + j;
            b.n_seq_id[j]  = 1;
            b.seq_id[j][0] = kSeqMain;
            b.logits[j]    = 0;
        }
        if (logits_last && i + count == n) {
            b.logits[count - 1] = 1;
        }
        b.n_tokens = count;
        const int64_t t0 = now_us();
        const int rc = llama_decode(lctx, b);
        report_hint(e, t0);
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}

/** Decode one token of the answer on the decode context. Returns the llama_decode code. */
int decode_one(Engine & e, llama_token token) {
    const int rc = decode_text(e, e.ctx, &token, 1, e.n_past, true);
    if (rc == 0) {
        e.cache.push_back(MemItem{token, {}});
        e.n_past += 1;
    }
    return rc;
}

/** Empty the model memory of both contexts, the snapshot, and the record of what they hold. */
void clear_all(Engine & e) {
    llama_memory_clear(llama_get_memory(e.ctx), true);
    if (e.ctx_pf != nullptr) {
        llama_memory_clear(llama_get_memory(e.ctx_pf), true);
    }
    e.cache.clear();
    e.n_past = 0;
    e.snap.clear();
    e.snap_n_past = 0;
}

/**
 * Copy the sequence state (the KV cache of the attention layers and the
 * recurrent states) from one context to the other. The destination loses
 * its own state first. Returns false with the error text set.
 */
bool transfer_state(llama_context * src, llama_context * dst, std::string & error) {
    const size_t size = llama_state_seq_get_size(src, kSeqMain);
    // A plain array: a vector would fill tens of megabytes with zeros first.
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[size]);
    const size_t got = llama_state_seq_get_data(src, buffer.get(), size, kSeqMain);
    if (got == 0 || llama_state_seq_set_data(dst, buffer.get(), got, kSeqMain) == 0) {
        error = "The state transfer between the prefill and the decode context failed (" + std::to_string(size) + " bytes)";
        return false;
    }
    return true;
}

/** Load the vision projector. Returns false with the error text set. */
bool ensure_vision(Engine & e, std::string & error) {
    if (e.mctx != nullptr) {
        return true;
    }
    if (e.mmproj.empty()) {
        error = "This model has no vision projector (mmproj) next to it";
        return false;
    }
    // The encoder runs on the OpenCL GPU. The Hexagon backend lacks operators of the CLIP
    // graph and gives wrong image features. Without the GPU it runs on the CPU.
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("GPUOpenCL");
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
        error = "The vision projector did not load: " + e.mmproj;
        return false;
    }
    LOGI("vision projector loaded in %.0f ms: %s", (now_us() - t0) / 1000.0, e.mmproj.c_str());
    return true;
}

/**
 * The encoder output of an image chunk: from the cache by the hash of the
 * bitmap, or from one run of the vision encoder. The pointer is valid
 * until the next call. Returns nullptr with the error text set.
 */
float * image_embd(Engine & e, const mtmd_input_chunk * chunk, std::string & error) {
    const std::string id = mtmd_input_chunk_get_id(chunk);
    for (ImageEmbd & entry : e.image_cache) {
        if (!id.empty() && entry.id == id) {
            return entry.embd.data();
        }
    }
    const int64_t t0 = now_us();
    const int32_t rc = mtmd_encode_chunk(e.mctx, chunk);
    report_hint(e, t0);
    if (rc != 0) {
        error = "The vision encoder failed with code " + std::to_string(rc);
        return nullptr;
    }
    const size_t n_floats = mtmd_input_chunk_get_n_tokens(chunk) * (size_t) llama_model_n_embd_inp(e.model);
    const float * out = mtmd_get_output_embd(e.mctx);
    const size_t bytes = n_floats * sizeof(float);
    while (!e.image_cache.empty() && e.image_cache_bytes + bytes > kImageCacheBytes) {
        e.image_cache_bytes -= e.image_cache.front().embd.size() * sizeof(float);
        e.image_cache.erase(e.image_cache.begin());
    }
    e.image_cache.push_back(ImageEmbd{id, std::vector<float>(out, out + n_floats)});
    e.image_cache_bytes += bytes;
    LOGI("image %s encoded in %.0f ms, %zu tokens", id.substr(0, 12).c_str(), (now_us() - t0) / 1000.0,
         mtmd_input_chunk_get_n_tokens(chunk));
    return e.image_cache.back().embd.data();
}

/**
 * Decode the items [start, end) into sequence kSeqMain of a context. Text
 * runs go in batches, an image chunk goes through its encoder output with
 * the M-RoPE positions of the helper. pos advances. Returns false with the
 * error text set.
 */
bool decode_items(Engine & e, llama_context * lctx, const std::vector<MemItem> & items,
                  const std::vector<const mtmd_input_chunk *> & chunk_of, size_t start, size_t end,
                  llama_pos & pos, bool logits_last, std::string & error) {
    std::vector<llama_token> run;
    run.reserve(end - start);
    auto flush = [&](bool last) {
        if (run.empty()) {
            return true;
        }
        const int rc = decode_text(e, lctx, run.data(), (int) run.size(), pos, last);
        if (rc != 0) {
            error = "llama_decode failed on the prompt with code " + std::to_string(rc);
            return false;
        }
        pos += (llama_pos) run.size();
        run.clear();
        return true;
    };
    for (size_t i = start; i < end; ++i) {
        const mtmd_input_chunk * chunk = chunk_of[i];
        if (chunk == nullptr) {
            run.push_back(items[i].token);
            continue;
        }
        if (!flush(false)) {
            return false;
        }
        float * embd = image_embd(e, chunk, error);
        if (embd == nullptr) {
            return false;
        }
        llama_pos new_pos = pos;
        const int64_t t0 = now_us();
        const int32_t rc = mtmd_helper_decode_image_chunk(e.mctx, lctx, chunk, embd, pos, kSeqMain, kBatch, &new_pos,
                                                          nullptr, nullptr);
        report_hint(e, t0);
        if (rc != 0) {
            error = "The image did not decode, code " + std::to_string(rc);
            return false;
        }
        pos = new_pos;
    }
    return flush(logits_last);
}

/** True when a is a prefix of b. */
bool is_prefix(const std::vector<MemItem> & a, const std::vector<MemItem> & b) {
    return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}

/** The number of tokens of the items [start, end): one per text item, the token count of an image chunk. */
int64_t count_tokens(const std::vector<const mtmd_input_chunk *> & chunk_of, size_t start, size_t end) {
    int64_t n = 0;
    for (size_t i = start; i < end; ++i) {
        n += chunk_of[i] == nullptr ? 1 : (int64_t) mtmd_input_chunk_get_n_tokens(chunk_of[i]);
    }
    return n;
}

/**
 * Decode a prompt. The items [0, base_len) are the prompt without the
 * generation prompt, the rest is the generation prompt. The longest reusable
 * prefix comes from the snapshot, or on the other backends also from the
 * conversation sequence. The new snapshot is taken before the generation
 * prompt. Returns false with the error text set, and the memory empty.
 */
bool prefill(Engine & e, const std::vector<MemItem> & items, const std::vector<const mtmd_input_chunk *> & chunk_of,
             size_t base_len, std::string & error) {
    const bool hybrid = e.ctx_pf != nullptr;
    llama_memory_t mem = llama_get_memory(e.ctx);

    // The prefix that the memory holds already.
    size_t start   = 0;
    bool   restore = false;
    if (!e.snap.empty() && is_prefix(e.snap, items)) {
        start   = e.snap.size();
        restore = true;
    }
    if (!hybrid && e.cache.size() > start && is_prefix(e.cache, items)) {
        start   = e.cache.size();
        restore = false;
    }
    if (start >= items.size() || start > base_len) {
        // Nothing new gives no logits, and a prefix past the generation prompt cannot keep a snapshot.
        start   = 0;
        restore = false;
    }
    const size_t reused = start;

    llama_pos pos = 0;
    if (start == 0) {
        clear_all(e);
    } else if (restore) {
        if (hybrid) {
            pos = e.snap_n_past;
        } else {
            if (!llama_memory_seq_rm(mem, kSeqMain, -1, -1)) {
                error = "The memory did not release the conversation sequence";
                clear_all(e);
                return false;
            }
            llama_memory_seq_cp(mem, kSeqSnap, kSeqMain, -1, -1);
            e.cache  = e.snap;
            e.n_past = e.snap_n_past;
            pos      = e.n_past;
        }
    } else {
        pos = e.n_past;
    }

    const int64_t t0 = now_us();
    llama_context * pctx = hybrid ? e.ctx_pf : e.ctx;
    if (!decode_items(e, pctx, items, chunk_of, start, base_len, pos, false, error)) {
        clear_all(e);
        return false;
    }

    // The snapshot before the generation prompt.
    const bool snap_same = e.snap.size() == base_len && std::equal(e.snap.begin(), e.snap.end(), items.begin());
    if (!snap_same && base_len > 0) {
        if (!hybrid) {
            llama_memory_seq_rm(mem, kSeqSnap, -1, -1);
            llama_memory_seq_cp(mem, kSeqMain, kSeqSnap, -1, -1);
        }
        e.snap.assign(items.begin(), items.begin() + (ptrdiff_t) base_len);
        e.snap_n_past = pos;
    }

    if (hybrid) {
        const int64_t t1 = now_us();
        if (!transfer_state(e.ctx_pf, e.ctx, error)) {
            clear_all(e);
            return false;
        }
        e.transfer_us += now_us() - t1;
    }
    if (!decode_items(e, e.ctx, items, chunk_of, base_len, items.size(), pos, true, error)) {
        clear_all(e);
        return false;
    }
    e.cache  = items;
    e.n_past = pos;
    e.prefill_tokens = count_tokens(chunk_of, reused, items.size());
    e.prefill_us     = now_us() - t0 - e.transfer_us;
    LOGI("prefill: %zu of %zu items reused, %lld tokens decoded in %.0f ms on %s, transfer %.0f ms, memory %d positions",
         reused, items.size(), (long long) e.prefill_tokens, e.prefill_us / 1000.0,
         hybrid ? ggml_backend_dev_name(e.device_pf) : e.device ? ggml_backend_dev_name(e.device) : "CPU",
         e.transfer_us / 1000.0, (int) e.n_past);
    return true;
}

/**
 * Tokenize the prompt. With images, mtmd replaces each media marker with
 * the chunk of the image in order, and chunk_of points at the image chunks.
 * Returns false with the error text set.
 */
bool tokenize_prompt(JNIEnv * env, Engine & e, const std::string & prompt, const std::vector<jbyteArray> & images,
                     mtmd::input_chunks & chunks, std::vector<MemItem> & items,
                     std::vector<const mtmd_input_chunk *> & chunk_of, std::string & error) {
    items.clear();
    chunk_of.clear();
    if (images.empty()) {
        const std::vector<llama_token> tokens = common_tokenize(llama_model_get_vocab(e.model), prompt, true, true);
        items.reserve(tokens.size());
        for (llama_token t : tokens) {
            items.push_back(MemItem{t, {}});
        }
        chunk_of.assign(items.size(), nullptr);
        return true;
    }

    if (!ensure_vision(e, error)) {
        return false;
    }
    mtmd::bitmaps bitmaps;
    for (jbyteArray image : images) {
        const jsize len = env->GetArrayLength(image);
        jbyte * bytes = env->GetByteArrayElements(image, nullptr);
        if (bytes == nullptr) {
            error = "The image bytes are not readable";
            return false;
        }
        const mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_buf(
            e.mctx, reinterpret_cast<const unsigned char *>(bytes), (size_t) len, false,
            mtmd_helper_init_opt_default());
        env->ReleaseByteArrayElements(image, bytes, JNI_ABORT);
        if (w.bitmap == nullptr) {
            error = "The image did not decode";
            return false;
        }
        bitmaps.entries.emplace_back(w.bitmap);
    }

    chunks.ptr.reset(mtmd_input_chunks_init());
    mtmd_input_text text;
    text.text          = prompt.c_str();
    text.text_len      = prompt.size();
    text.add_special   = true;
    text.parse_special = true;
    const std::vector<const mtmd_bitmap *> ptrs = bitmaps.c_ptr();
    const int32_t tk = mtmd_tokenize(e.mctx, chunks.ptr.get(), &text, ptrs.data(), ptrs.size());
    if (tk != 0) {
        error = tk == 1 ? "The number of images differs from the number of markers in the prompt"
                        : "The image preprocessing failed";
        return false;
    }
    for (size_t c = 0; c < chunks.size(); ++c) {
        const mtmd_input_chunk * chunk = chunks[c];
        switch (mtmd_input_chunk_get_type(chunk)) {
            case MTMD_INPUT_CHUNK_TYPE_TEXT: {
                size_t n = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n);
                for (size_t i = 0; i < n; ++i) {
                    items.push_back(MemItem{tokens[i], {}});
                    chunk_of.push_back(nullptr);
                }
                break;
            }
            case MTMD_INPUT_CHUNK_TYPE_IMAGE:
                items.push_back(MemItem{LLAMA_TOKEN_NULL, mtmd_input_chunk_get_id(chunk)});
                chunk_of.push_back(chunk);
                break;
            default:
                error = "The prompt holds a media type that this engine does not decode";
                return false;
        }
    }
    return true;
}

/**
 * The number of items of the prompt without the generation prompt. The
 * tokens of the generation prompt must be the tail of the items, or the
 * whole prompt counts as the base.
 */
size_t base_length(const Engine & e, const std::string & prompt, const std::string & tail,
                   const std::vector<MemItem> & items) {
    if (tail.empty() || prompt.size() < tail.size() ||
        prompt.compare(prompt.size() - tail.size(), tail.size(), tail) != 0) {
        return items.size();
    }
    const std::vector<llama_token> tail_tokens = common_tokenize(llama_model_get_vocab(e.model), tail, false, true);
    if (tail_tokens.empty() || tail_tokens.size() > items.size()) {
        return items.size();
    }
    const size_t base = items.size() - tail_tokens.size();
    for (size_t i = 0; i < tail_tokens.size(); ++i) {
        if (items[base + i] != MemItem{tail_tokens[i], {}}) {
            return items.size();
        }
    }
    return base;
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

/** The string element i of a Java array, with its local reference released. */
std::string array_string(JNIEnv * env, jobjectArray array, jsize i) {
    jstring s = (jstring) env->GetObjectArrayElement(array, i);
    std::string out = jstring_to_std(env, s);
    if (s != nullptr) {
        env->DeleteLocalRef(s);
    }
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
    if (n_ctx < kBatch) {
        throw_java(env, "The context length must be at least " + std::to_string(kBatch) + " tokens, not " + std::to_string(n_ctx));
        return 0;
    }
    // The destructor of the engine releases what loaded when a later step fails.
    auto e = std::make_unique<Engine>();
    e->n_threads  = std::max(1, (int) n_threads);
    e->gpu_layers = gpu_layers;
    e->mmproj     = jstring_to_std(env, jmmproj);
    const bool hybrid = !prefill.empty();

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

    // The decode context. Without the hybrid backend, a second sequence holds
    // the prompt snapshot: the KV cache is unified, thus the sequences share
    // cells, and the recurrent memory has one cell for each sequence.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) n_ctx;
    cp.n_batch         = kBatch;
    cp.n_ubatch        = kBatch;
    cp.n_seq_max       = hybrid ? 1 : 2;
    cp.kv_unified      = true;
    cp.n_threads       = e->n_threads;
    cp.n_threads_batch = e->n_threads;
    cp.no_perf         = false;
    e->ctx = llama_init_from_model(e->model, cp);
    if (e->ctx == nullptr) {
        throw_java(env, "The context did not initialize (n_ctx=" + std::to_string(n_ctx) + ")");
        return 0;
    }

    // The hybrid backend: the same file again on the prefill device, one sequence.
    if (hybrid) {
        e->device_pf = ggml_backend_dev_by_name(prefill.c_str());
        if (e->device_pf == nullptr) {
            throw_java(env, "The prefill device is not available: " + prefill);
            return 0;
        }
        std::vector<ggml_backend_dev_t> devices_pf = {e->device_pf, nullptr};
        llama_model_params mp_pf = llama_model_default_params();
        mp_pf.n_gpu_layers = 999;
        mp_pf.devices      = devices_pf.data();
        e->model_pf = llama_model_load_from_file(path.c_str(), mp_pf);
        llama_context_params cp_pf = cp;
        cp_pf.n_seq_max = 1;
        e->ctx_pf = e->model_pf ? llama_init_from_model(e->model_pf, cp_pf) : nullptr;
        if (e->ctx_pf == nullptr) {
            throw_java(env, "The prefill model did not load on " + prefill);
            return 0;
        }
    }
    e->batch = llama_batch_init(kBatch, 0, 1);

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
    if (e->tp == nullptr) {
        throw_java(env, "The thread pool did not start (" + std::to_string(e->n_threads) + " threads)");
        return 0;
    }
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

    try {
        e->tmpls = common_chat_templates_init(e->model, "");
    } catch (const std::exception & ex) {
        throw_java(env, std::string("The chat template of the model did not parse: ") + ex.what());
        return 0;
    }
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
    delete engine_of(handle);
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
    // Each local reference goes away at once: the table holds 512.
    common_chat_templates_inputs inputs;
    std::vector<jbyteArray> image_refs;
    const jsize n = env->GetArrayLength(roles);
    if (env->GetArrayLength(contents) != n || (images != nullptr && env->GetArrayLength(images) != n)) {
        throw_java(env, "The roles, contents and images arrays have different lengths");
        return -1;
    }
    for (jsize i = 0; i < n; ++i) {
        common_chat_msg msg;
        msg.role    = array_string(env, roles, i);
        msg.content = array_string(env, contents, i);
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

    // The prompt, and the generation prompt at its end. The template gives
    // the tail directly, or the render without it gives the common prefix.
    std::string prompt;
    std::string tail;
    try {
        common_chat_params params = common_chat_templates_apply(e->tmpls.get(), inputs);
        prompt = std::move(params.prompt);
        tail   = std::move(params.generation_prompt);
        if (tail.empty() || prompt.size() < tail.size() ||
            prompt.compare(prompt.size() - tail.size(), tail.size(), tail) != 0) {
            inputs.add_generation_prompt = false;
            const std::string base = common_chat_templates_apply(e->tmpls.get(), inputs).prompt;
            size_t k = 0;
            while (k < base.size() && k < prompt.size() && base[k] == prompt[k]) {
                ++k;
            }
            tail = prompt.substr(k);
        }
    } catch (const std::exception & ex) {
        throw_java(env, std::string("The chat template failed: ") + ex.what());
        return -1;
    }

    rebuild_sampler(*e, thinking, temperature, top_p);
    e->utf8_pending.clear();
    e->prefill_tokens = 0;
    e->prefill_us     = 0;
    e->transfer_us    = 0;
    e->gen_tokens     = 0;
    e->gen_us         = 0;

    std::string error;
    mtmd::input_chunks chunks;
    std::vector<MemItem> items;
    std::vector<const mtmd_input_chunk *> chunk_of;
    if (!tokenize_prompt(env, *e, prompt, image_refs, chunks, items, chunk_of, error)) {
        throw_java(env, error);
        return -1;
    }
    const int64_t n_tokens = count_tokens(chunk_of, 0, items.size());
    if (items.empty() || (uint64_t) n_tokens + kContextHeadroom >= llama_n_ctx(e->ctx)) {
        throw_java(env, "The conversation is longer than the context (" + std::to_string(n_tokens) + " tokens)");
        return -1;
    }
    size_t base_len = base_length(*e, prompt, tail, items);
    if (base_len == items.size() && e->ctx_pf != nullptr) {
        // The decode context must decode the last token itself: the state transfer carries no logits.
        base_len -= 1;
    }
    if (chunk_of.back() != nullptr) {
        throw_java(env, "The prompt ends with an image, thus it gives no logits");
        return -1;
    }

    if (!prefill(*e, items, chunk_of, base_len, error)) {
        throw_java(env, error);
        return -1;
    }
    return (jint) e->prefill_tokens;
}

JNIEXPORT jbyteArray JNICALL
Java_ai_airi_qwenmobile_LlamaNative_generateNext(JNIEnv * env, jclass, jlong handle) {
    Engine * e = engine_of(handle);
    std::lock_guard<std::mutex> lock(e->mutex);
    const llama_vocab * vocab = llama_model_get_vocab(e->model);
    if ((uint32_t) e->n_past >= llama_n_ctx(e->ctx)) {
        throw_java(env, "The context is full (" + std::to_string(llama_n_ctx(e->ctx)) + " tokens). Start a new chat.");
        return nullptr;
    }

    const int64_t t0 = now_us();
    const llama_token token = llama_sampler_sample(e->smpl, e->ctx, -1);
    if (llama_vocab_is_eog(vocab, token)) {
        // The end token goes into the memory, thus a template that renders it lets the next turn extend this one.
        decode_one(*e, token);
        return nullptr;
    }
    const int kind = token_kind(*e, token);
    if (kind == 0) {
        append_piece(*e, token);
    }
    const int rc = decode_one(*e, token);
    if (rc != 0) {
        throw_java(env, "llama_decode failed during generation with code " + std::to_string(rc));
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
    const llama_pos n_past = llama_memory_seq_pos_max(llama_get_memory(e->ctx), kSeqMain) + 1;
    const char * pf_dev = e->device_pf ? ggml_backend_dev_name(e->device_pf) : e->device ? ggml_backend_dev_name(e->device) : "CPU";
    const std::string transfer = e->device_pf ? ", transfer " + std::to_string(e->transfer_us / 1000) + " ms" : "";
    snprintf(line, sizeof(line), "prefill %lld tok in %.0f ms (%.1f t/s on %s%s), generate %lld tok (%.1f t/s), memory %d pos",
             (long long) e->prefill_tokens, e->prefill_us / 1000.0, pp, pf_dev, transfer.c_str(),
             (long long) e->gen_tokens, tg, (int) n_past);
    return env->NewStringUTF(line);
}

JNIEXPORT void JNICALL
Java_ai_airi_qwenmobile_LlamaNative_resetChat(JNIEnv *, jclass, jlong handle) {
    Engine * e = engine_of(handle);
    std::lock_guard<std::mutex> lock(e->mutex);
    clear_all(*e);
    e->image_cache.clear();
    e->image_cache_bytes = 0;
    e->utf8_pending.clear();
    e->prefill_tokens = e->prefill_us = e->transfer_us = e->gen_tokens = e->gen_us = 0;
}

/**
 * The llama-bench method: pp tokens in batches of kBatch, then tg tokens
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
            const int rc = decode_text(*e, pctx, tokens.data(), pp, 0, true);
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
                rc = decode_text(*e, e->ctx, &t, 1, i, true);
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
    clear_all(*e);

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
