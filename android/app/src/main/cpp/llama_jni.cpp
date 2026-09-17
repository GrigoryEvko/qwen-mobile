/**
 * The JNI layer between ai.airi.qwenmobile.LlamaNative and llama.cpp.
 *
 * One Engine holds one model, one context, one sampler, and one thread pool.
 * All calls for one engine come from one Kotlin thread. A mutex guards the
 * engine against a second caller.
 *
 * The model memory of Qwen3.5 is a recurrent state plus a KV cache. A
 * recurrent state cannot roll back, thus the engine keeps snapshots of the
 * state: after each prompt, before its generation prompt, the state of
 * sequence 0 goes as bytes into a store (state_cache.h) with the exact
 * items that it holds. The chat template renders the previous answer
 * differently from the generated tokens, thus the next prompt extends the
 * snapshot of the previous prompt and not the answer: only the rendered
 * answer and the new message decode. An earlier turn or a repeated
 * question restores its snapshot without a decode. The hybrid backend
 * prefills on the prefill context and moves the same bytes to the decode
 * context.
 *
 * The output of the vision encoder goes into a second store
 * (image_cache.h) by the hash of the image file, in RAM and on disk. A
 * known image tokenizes as a placeholder, thus its JPEG is not decoded
 * again.
 */

#include <android/log.h>
#include <dirent.h>
#include <jni.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef QWEN_NO_OPENCL_INFO
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#endif

#include "cache_io.h"
#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "image_cache.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "perf_hint.h"
#include "state_cache.h"

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

/** The budgets of the cache of encoded images, in bytes. One image is 2 to 5 MB at 2048 x 576 floats. */
constexpr size_t kImageRamBytes  = 64u << 20;
constexpr size_t kImageDiskBytes = 256u << 20;

/**
 * The budgets of the store of sequence states, in bytes. One snapshot of
 * the 2B model is 20 MB of recurrent state plus 12 KB for each token in
 * the KV cache: 44 MB at 2K tokens, thus the RAM holds 5 to 10 of them.
 */
constexpr size_t kStateRamBytes  = 256u << 20;
constexpr size_t kStateDiskBytes = 256u << 20;

/** The nice value of the compute threads. The display thread keeps its priority. */
constexpr int kComputeNice = 10;

/** The batch of the contexts. The Hexagon backend wants 512. */
constexpr int kBatch = 512;

/** The number of free positions that a prompt must leave in the context. */
constexpr size_t kContextHeadroom = 8;

/** The sequence of the conversation in each context. */
constexpr llama_seq_id kSeqMain = 0;

static_assert(kMemTokenNull == LLAMA_TOKEN_NULL, "The null token of the state store must be LLAMA_TOKEN_NULL");

/** Give a thread the compute priority. tid 0 is the calling thread. */
void lower_priority(int32_t tid) {
    if (setpriority(PRIO_PROCESS, tid, kComputeNice) != 0) {
        LOGE("setpriority for thread %d failed", tid);
    }
}

/** The counters of one turn, for the stats line. Times are in microseconds. */
struct TurnStats {
    /** The tokens of the prompt that decoded, and the time of these decodes. */
    int64_t prefill_tokens = 0;
    int64_t prefill_us     = 0;
    /** The tokens of the prompt that the memory held already. From a snapshot, or from the live memory. */
    int64_t reused_tokens  = 0;
    bool    from_snapshot  = false;
    /** The restore of a snapshot into the prefill context, the snapshot of the new prompt state, the move to the decode context. */
    int64_t restore_us     = 0;
    int64_t snapshot_us    = 0;
    int64_t transfer_us    = 0;
    /** The images of the prompt: with a placeholder bitmap, with a cached encoder output, and encoded. */
    int     images_total   = 0;
    int     images_known   = 0;
    int     images_cached  = 0;
    int     images_encoded = 0;
    int64_t gen_tokens     = 0;
    int64_t gen_us         = 0;
};

struct Engine {
    llama_model *     model = nullptr;
    llama_context *   ctx   = nullptr;
    llama_sampler *   smpl  = nullptr;
    ggml_threadpool * tp    = nullptr;
    /** The vision projector. It loads on the first image, on the device of vision_device. */
    mtmd_context *    mctx  = nullptr;
    std::string       mmproj;
    /** The ggml device name of the image encoder. Empty takes the OpenCL GPU when it is present, else the CPU. */
    std::string       vision_device;
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
    /** The hybrid backend: the items that sequence 0 of the prefill context holds, and its positions. */
    std::vector<MemItem> pf_cache;
    llama_pos pf_n_past = 0;
    /** The snapshots of the prompt states, and the outputs of the vision encoder. */
    std::unique_ptr<StateCache> states;
    std::unique_ptr<ImageCache> images;
    /** The dimensions of the images that this turn decoded from their bytes, by id. The encoder output of a new image goes into the cache with them. */
    std::unordered_map<std::string, ImageInfo> turn_images;
    /** Bytes of an incomplete UTF-8 sequence from the last token. */
    std::string utf8_pending;
    /** Set by requestStop from another thread. generateNext reads it before the sample. */
    std::atomic<bool> stop_requested{false};
    /** True while the thinking of the answer is open: the generation prompt or a generated tag opened it. */
    bool think_open = false;
    /** True after the end of the answer: the end token, a stop, or the context limit. */
    bool answer_done = false;

    int  n_threads  = 4;
    int  gpu_layers = 0;
    bool thinking   = false;

    TurnStats turn;

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

/** Empty the model memory of both contexts and the record of what they hold. The snapshot store stays. */
void clear_all(Engine & e) {
    llama_memory_clear(llama_get_memory(e.ctx), true);
    if (e.ctx_pf != nullptr) {
        llama_memory_clear(llama_get_memory(e.ctx_pf), true);
    }
    e.cache.clear();
    e.n_past = 0;
    e.pf_cache.clear();
    e.pf_n_past = 0;
}

/**
 * The state of sequence kSeqMain of a context as bytes: the KV cache of
 * the attention layers and the recurrent states, independent of the
 * device. Returns null with the error text set.
 */
std::shared_ptr<const cache_io::Blob> take_state(llama_context * lctx, std::string & error) {
    const size_t size = llama_state_seq_get_size(lctx, kSeqMain);
    auto blob = std::make_shared<cache_io::Blob>(size);
    const size_t got = llama_state_seq_get_data(lctx, blob->data.get(), size, kSeqMain);
    if (got == 0 || got > size) {
        error = "The state of the sequence did not copy out of the context (" + std::to_string(size) + " bytes)";
        return nullptr;
    }
    blob->size = got;
    return blob;
}

/**
 * Put a state into sequence kSeqMain of a context in the place of what it
 * holds. Returns false when the bytes do not fit the context, and then the
 * memory of the context is empty.
 */
bool restore_state(llama_context * lctx, const cache_io::Blob & bytes) {
    llama_memory_t mem = llama_get_memory(lctx);
    llama_memory_seq_rm(mem, kSeqMain, -1, -1);
    if (llama_state_seq_set_data(lctx, bytes.data.get(), bytes.size, kSeqMain) == 0) {
        llama_memory_clear(mem, true);
        return false;
    }
    return true;
}

/** The namespace of the disk caches: the model and projector files with their sizes and times, and the image token limit. */
std::string cache_namespace(const std::string & model, const std::string & mmproj) {
    std::string key;
    for (const std::string & path : {model, mmproj}) {
        cache_io::FileStat st;
        key += path + "|";
        if (!path.empty() && cache_io::stat_file(path, st)) {
            key += std::to_string(st.size) + "|" + std::to_string(st.mtime);
        }
        key += "|";
    }
    key += std::to_string(kImageMaxTokens);
    return cache_io::hex64(cache_io::fnv1a64(key.data(), key.size()));
}

/**
 * Make the two stores. With a cache directory, the files go below
 * <dir>/engine/<namespace>, and the directories of other namespaces go
 * away, thus the disk holds the caches of one model.
 */
void open_stores(Engine & e, const std::string & cache_dir, const std::string & model_path) {
    std::string state_dir;
    std::string image_dir;
    if (!cache_dir.empty()) {
        const std::string root = cache_dir + "/engine";
        const std::string ns   = cache_namespace(model_path, e.mmproj);
        for (const std::string & name : cache_io::list_dirs(root)) {
            if (name != ns) {
                cache_io::remove_tree(root + "/" + name);
            }
        }
        state_dir = root + "/" + ns + "/state";
        image_dir = root + "/" + ns + "/images";
    }
    e.states = std::make_unique<StateCache>(kStateRamBytes, kStateDiskBytes, state_dir);
    e.images = std::make_unique<ImageCache>(kImageRamBytes, kImageDiskBytes, image_dir);
    LOGI("caches: %zu snapshots (%.0f MB) and %zu images (%.0f MB) on disk in %s",
         e.states->count(), e.states->disk_bytes() / 1048576.0, e.images->count(), e.images->disk_bytes() / 1048576.0,
         cache_dir.empty() ? "no directory" : cache_dir.c_str());
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
    // The encoder runs on the device the app selected. The Hexagon NPU encodes a photo in
    // less than one second, the OpenCL GPU in some seconds. A device that the app named
    // and that is not available is an error, not a fallback to the CPU.
    ggml_backend_dev_t dev = nullptr;
    if (!e.vision_device.empty()) {
        dev = ggml_backend_dev_by_name(e.vision_device.c_str());
        if (dev == nullptr) {
            error = "The image encoder device is not available: " + e.vision_device;
            return false;
        }
    } else {
        dev = ggml_backend_dev_by_name("GPUOpenCL");
    }
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
    LOGI("vision projector loaded in %.0f ms on %s: %s", (now_us() - t0) / 1000.0,
         dev ? ggml_backend_dev_name(dev) : "CPU", e.mmproj.c_str());
    return true;
}

/**
 * The encoder output of an image chunk: from the cache by the hash of the
 * image file, or from one run of the vision encoder. The pointer is valid
 * until the next call. Returns nullptr with the error text set.
 */
const float * image_embd(Engine & e, const mtmd_input_chunk * chunk, std::string & error) {
    const std::string id = mtmd_input_chunk_get_id(chunk);
    if (const float * hit = e.images->get(id)) {
        e.turn.images_cached += 1;
        return hit;
    }
    const auto known = e.turn_images.find(id);
    if (known == e.turn_images.end()) {
        // The bitmap is a placeholder, because the cache knew the image. Its file is gone since.
        error = "The encoder output of image " + id.substr(0, 12) + " is not in the cache. Send the message again.";
        return nullptr;
    }
    const int64_t t0 = now_us();
    const int32_t rc = mtmd_encode_chunk(e.mctx, chunk);
    report_hint(e, t0);
    if (rc != 0) {
        error = "The vision encoder failed with code " + std::to_string(rc);
        return nullptr;
    }
    e.turn.images_encoded += 1;
    ImageInfo info = known->second;
    info.n_tokens  = (uint32_t) mtmd_input_chunk_get_n_tokens(chunk);
    info.n_embd    = (uint32_t) llama_model_n_embd_inp(e.model);
    const float * out = mtmd_get_output_embd(e.mctx);
    e.images->put(id, info, out);
    LOGI("image %s encoded in %.0f ms, %u tokens, %.1f MB, cache %zu images %.0f MB in RAM, %.0f MB on disk",
         id.substr(0, 12).c_str(), (now_us() - t0) / 1000.0, info.n_tokens, info.n_floats() * 4 / 1048576.0,
         e.images->count(), e.images->ram_bytes() / 1048576.0, e.images->disk_bytes() / 1048576.0);
    // The output buffer of the encoder stays valid until the next encode, thus it serves when the cache did not keep the copy.
    const float * kept = e.images->get(id);
    return kept != nullptr ? kept : out;
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
        const float * embd = image_embd(e, chunk, error);
        if (embd == nullptr) {
            return false;
        }
        llama_pos new_pos = pos;
        const int64_t t0 = now_us();
        // The helper takes a non-const pointer and only reads the floats.
        const int32_t rc = mtmd_helper_decode_image_chunk(e.mctx, lctx, chunk, const_cast<float *>(embd), pos, kSeqMain,
                                                          kBatch, &new_pos, nullptr, nullptr);
        report_hint(e, t0);
        if (rc != 0) {
            error = "The image did not decode, code " + std::to_string(rc);
            return false;
        }
        pos = new_pos;
    }
    return flush(logits_last);
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
 * generation prompt, the rest is the generation prompt. The longest
 * reusable prefix comes from the live memory of the prefill context, or
 * from the snapshot store, and only the remaining items decode. The state
 * before the generation prompt goes into the store, and on the hybrid
 * backend also into the decode context. Returns false with the error
 * text set, and the memory empty.
 *
 * Exactness: a snapshot holds the state after exactly its items, with the
 * positions of each KV cell (the M-RoPE x and y of an image token
 * included) and the count of positions. A restore puts that state back,
 * and the tail decodes at the same positions as without the store. Thus
 * the logits after the prompt are the same as after one decode of the
 * whole prompt on the same device.
 */
bool prefill(Engine & e, const std::vector<MemItem> & items, const std::vector<const mtmd_input_chunk *> & chunk_of,
             size_t base_len, std::string & error) {
    const bool hybrid = e.ctx_pf != nullptr;
    llama_context * pctx = hybrid ? e.ctx_pf : e.ctx;
    // What sequence kSeqMain of the prefill context holds at this time.
    std::vector<MemItem> & live     = hybrid ? e.pf_cache : e.cache;
    llama_pos &            live_pos = hybrid ? e.pf_n_past : e.n_past;
    // At least one item decodes on the decode context, thus it has logits.
    const size_t limit = std::min(base_len, items.size() - 1);
    const int64_t t0   = now_us();

    // The longest prefix that exists already: the live memory when it is at least as long as the best snapshot.
    size_t    start = 0;
    llama_pos pos   = 0;
    const Snapshot * snap = e.states->best_prefix(items, limit);
    const bool live_ok = !live.empty() && live.size() <= limit && is_item_prefix(live, items);
    if (live_ok && (snap == nullptr || live.size() >= snap->items.size())) {
        start = live.size();
        pos   = live_pos;
    } else if (snap != nullptr) {
        const size_t    n_items = snap->items.size();
        const llama_pos n_pos   = snap->n_pos;
        std::shared_ptr<const cache_io::Blob> bytes = e.states->bytes(snap);
        if (bytes && restore_state(pctx, *bytes)) {
            start = n_items;
            pos   = n_pos;
            e.turn.from_snapshot = true;
        } else if (bytes) {
            LOGE("snapshot of %zu items did not restore (%zu bytes), it is dropped", n_items, bytes->size);
            e.states->drop(snap);
        }
        e.turn.restore_us = now_us() - t0;
    }
    if (start == 0) {
        llama_memory_clear(llama_get_memory(pctx), true);
        pos = 0;
    }
    live.assign(items.begin(), items.begin() + (ptrdiff_t) start);
    live_pos = pos;
    e.turn.reused_tokens = count_tokens(chunk_of, 0, start);

    const int64_t t1 = now_us();
    if (!decode_items(e, pctx, items, chunk_of, start, base_len, pos, false, error)) {
        clear_all(e);
        return false;
    }
    live.assign(items.begin(), items.begin() + (ptrdiff_t) base_len);
    live_pos = pos;

    // The state before the generation prompt: into the store when it is not there, and to the decode context.
    std::shared_ptr<const cache_io::Blob> base_bytes;
    if (base_len > 0) {
        std::vector<MemItem> key(items.begin(), items.begin() + (ptrdiff_t) base_len);
        const Snapshot * have = e.states->find(key);
        if (have != nullptr && hybrid) {
            base_bytes = e.states->bytes(have);
        }
        if (have == nullptr || (hybrid && !base_bytes)) {
            const int64_t t2 = now_us();
            base_bytes = take_state(pctx, error);
            if (!base_bytes) {
                clear_all(e);
                return false;
            }
            e.turn.snapshot_us = now_us() - t2;
            e.states->put(std::move(key), pos, base_bytes);
            LOGI("snapshot: %zu items, %d positions, %.1f MB in %.0f ms, store %zu snapshots, %.0f MB in RAM, %.0f MB on disk",
                 base_len, (int) pos, base_bytes->size / 1048576.0, e.turn.snapshot_us / 1000.0, e.states->count(),
                 e.states->ram_bytes() / 1048576.0, e.states->disk_bytes() / 1048576.0);
        }
    }
    if (hybrid) {
        const int64_t t3 = now_us();
        if (base_len == 0) {
            llama_memory_clear(llama_get_memory(e.ctx), true);
        } else if (!base_bytes || !restore_state(e.ctx, *base_bytes)) {
            error = "The state did not move from the prefill context to the decode context (" +
                    std::to_string(base_bytes ? base_bytes->size : 0) + " bytes)";
            clear_all(e);
            return false;
        }
        e.turn.transfer_us = now_us() - t3;
        e.cache.assign(items.begin(), items.begin() + (ptrdiff_t) base_len);
        e.n_past = pos;
    }
    if (!decode_items(e, e.ctx, items, chunk_of, base_len, items.size(), pos, true, error)) {
        clear_all(e);
        return false;
    }
    e.cache  = items;
    e.n_past = pos;
    e.turn.prefill_tokens = count_tokens(chunk_of, start, items.size());
    e.turn.prefill_us     = now_us() - t1 - e.turn.snapshot_us - e.turn.transfer_us;
    LOGI("prefill: %zu of %zu items %s (%lld tokens, %.0f ms), %lld tokens decoded in %.0f ms on %s, "
         "transfer %.0f ms, images %d (%d known, %d cached, %d encoded), memory %d positions",
         start, items.size(), e.turn.from_snapshot ? "restored" : "kept", (long long) e.turn.reused_tokens,
         e.turn.restore_us / 1000.0, (long long) e.turn.prefill_tokens, e.turn.prefill_us / 1000.0,
         hybrid ? ggml_backend_dev_name(e.device_pf) : e.device ? ggml_backend_dev_name(e.device) : "CPU",
         e.turn.transfer_us / 1000.0, e.turn.images_total, e.turn.images_known, e.turn.images_cached,
         e.turn.images_encoded, (int) e.n_past);
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
    // A known image gives a placeholder bitmap with its dimensions and its
    // id: the same chunk as its decode gives, without the decode of the JPEG
    // and its preprocessing. The id is the SHA-256 of the file bytes, the
    // same value that the helper gives a decoded bitmap.
    mtmd::bitmaps bitmaps;
    e.turn_images.clear();
    e.turn.images_total = (int) images.size();
    for (jbyteArray image : images) {
        const jsize len = env->GetArrayLength(image);
        jbyte * bytes = env->GetByteArrayElements(image, nullptr);
        if (bytes == nullptr) {
            error = "The image bytes are not readable";
            return false;
        }
        const std::string id = cache_io::sha256_hex(bytes, (size_t) len);
        ImageInfo info;
        mtmd_bitmap * bitmap = nullptr;
        if (e.images->info(id, info)) {
            bitmap = mtmd_bitmap_init(info.nx, info.ny, nullptr);
            mtmd_bitmap_set_id(bitmap, id.c_str());
            e.turn.images_known += 1;
        } else {
            bitmap = mtmd_helper_bitmap_init_from_buf(e.mctx, reinterpret_cast<const unsigned char *>(bytes),
                                                      (size_t) len, false, mtmd_helper_init_opt_default()).bitmap;
            if (bitmap != nullptr) {
                info.nx = mtmd_bitmap_get_nx(bitmap);
                info.ny = mtmd_bitmap_get_ny(bitmap);
                e.turn_images[mtmd_bitmap_get_id(bitmap)] = info;
            }
        }
        env->ReleaseByteArrayElements(image, bytes, JNI_ABORT);
        if (bitmap == nullptr) {
            error = "The image did not decode";
            return false;
        }
        bitmaps.entries.emplace_back(bitmap);
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
                                          jstring jdevice, jstring jprefill, jstring jvision, jint gpu_layers,
                                          jint n_threads, jint n_ctx, jstring jcache) {
    const std::string path    = jstring_to_std(env, jpath);
    const std::string device  = jstring_to_std(env, jdevice);
    const std::string prefill = jstring_to_std(env, jprefill);
    const std::string vision  = jstring_to_std(env, jvision);
    const std::string cache   = jstring_to_std(env, jcache);
    if (n_ctx < kBatch) {
        throw_java(env, "The context length must be at least " + std::to_string(kBatch) + " tokens, not " + std::to_string(n_ctx));
        return 0;
    }
    // The destructor of the engine releases what loaded when a later step fails.
    auto e = std::make_unique<Engine>();
    e->n_threads  = std::max(1, (int) n_threads);
    e->gpu_layers = gpu_layers;
    e->mmproj     = jstring_to_std(env, jmmproj);
    e->vision_device = vision;
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

    // The decode context. One sequence: the snapshots of the prompt states are bytes in the store.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) n_ctx;
    cp.n_batch         = kBatch;
    cp.n_ubatch        = kBatch;
    cp.n_seq_max       = 1;
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
    open_stores(*e, cache, path);

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
    e->turn           = TurnStats{};
    e->answer_done    = false;
    e->stop_requested = false;

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
    // The thinking is open when the last tag of the generation prompt opens it.
    e->think_open = false;
    for (size_t i = base_len; i < items.size() && e->tok_think_open != LLAMA_TOKEN_NULL; ++i) {
        if (items[i].token == e->tok_think_open) {
            e->think_open = true;
        } else if (items[i].token == e->tok_think_close) {
            e->think_open = false;
        }
    }

    if (!prefill(*e, items, chunk_of, base_len, error)) {
        throw_java(env, error);
        return -1;
    }
    return (jint) e->turn.prefill_tokens;
}

JNIEXPORT jbyteArray JNICALL
Java_ai_airi_qwenmobile_LlamaNative_generateNext(JNIEnv * env, jclass, jlong handle) {
    Engine * e = engine_of(handle);
    std::lock_guard<std::mutex> lock(e->mutex);
    const llama_vocab * vocab = llama_model_get_vocab(e->model);
    if (e->answer_done) {
        return nullptr;
    }
    if (e->stop_requested.exchange(false)) {
        e->answer_done = true;
        return nullptr;
    }
    if ((uint32_t) e->n_past >= llama_n_ctx(e->ctx)) {
        e->answer_done = true;
        throw_java(env, "The context is full (" + std::to_string(llama_n_ctx(e->ctx)) + " tokens). Start a new chat.");
        return nullptr;
    }

    const int64_t t0 = now_us();
    const llama_token token = llama_sampler_sample(e->smpl, e->ctx, -1);
    if (llama_vocab_is_eog(vocab, token)) {
        // The end token goes into the memory, thus a template that renders it lets the next turn extend this one.
        decode_one(*e, token);
        e->answer_done = true;
        if (e->think_open) {
            // The answer ends inside the thinking: the app gets the close of the thinking one time, then the end.
            e->think_open = false;
            e->utf8_pending.clear();
            return pack_piece(env, *e, 2);
        }
        return nullptr;
    }
    const int kind = token_kind(*e, token);
    if (kind == 0) {
        append_piece(*e, token);
    } else {
        e->think_open = kind == 1;
    }
    const int rc = decode_one(*e, token);
    if (rc != 0) {
        e->answer_done = true;
        throw_java(env, "llama_decode failed during generation with code " + std::to_string(rc));
        return nullptr;
    }
    e->turn.gen_tokens += 1;
    e->turn.gen_us     += now_us() - t0;
    return pack_piece(env, *e, kind);
}

/**
 * Ask the running answer to stop. Any thread can call this: the flag is
 * atomic and the call takes no lock. The next generateNext gives the end
 * of the answer without a sample.
 */
JNIEXPORT void JNICALL
Java_ai_airi_qwenmobile_LlamaNative_requestStop(JNIEnv *, jclass, jlong handle) {
    engine_of(handle)->stop_requested = true;
}

JNIEXPORT jstring JNICALL
Java_ai_airi_qwenmobile_LlamaNative_stats(JNIEnv * env, jclass, jlong handle) {
    Engine * e = engine_of(handle);
    std::lock_guard<std::mutex> lock(e->mutex);
    const TurnStats & t = e->turn;
    char line[512];
    const double pp = t.prefill_us > 0 ? t.prefill_tokens * 1e6 / t.prefill_us : 0.0;
    const double tg = t.gen_us > 0 ? t.gen_tokens * 1e6 / t.gen_us : 0.0;
    const char * pf_dev = e->device_pf ? ggml_backend_dev_name(e->device_pf) : e->device ? ggml_backend_dev_name(e->device) : "CPU";
    std::string extra;
    if (t.reused_tokens > 0) {
        extra += ", " + std::to_string(t.reused_tokens) + " tok " + (t.from_snapshot ? "restored" : "kept");
        if (t.from_snapshot) {
            extra += " in " + std::to_string(t.restore_us / 1000) + " ms";
        }
    }
    if (t.snapshot_us > 0) {
        extra += ", snapshot " + std::to_string(t.snapshot_us / 1000) + " ms";
    }
    if (e->device_pf) {
        extra += ", transfer " + std::to_string(t.transfer_us / 1000) + " ms";
    }
    if (t.images_total > 0) {
        // Known: the JPEG was not decoded. Cached: the encoder output came from the cache. The rest of the prompt images sat in the reused prefix.
        extra += ", images " + std::to_string(t.images_total) + " (" + std::to_string(t.images_known) + " known, " +
                 std::to_string(t.images_cached) + " cached, " + std::to_string(t.images_encoded) + " encoded)";
    }
    snprintf(line, sizeof(line), "prefill %lld tok in %.0f ms (%.1f t/s on %s)%s, generate %lld tok (%.1f t/s), memory %d pos",
             (long long) t.prefill_tokens, t.prefill_us / 1000.0, pp, pf_dev, extra.c_str(),
             (long long) t.gen_tokens, tg, (int) e->n_past);
    return env->NewStringUTF(line);
}

JNIEXPORT void JNICALL
Java_ai_airi_qwenmobile_LlamaNative_resetChat(JNIEnv *, jclass, jlong handle) {
    Engine * e = engine_of(handle);
    std::lock_guard<std::mutex> lock(e->mutex);
    clear_all(*e);
    // The RAM tiers go, the files stay: a conversation on disk continues after a load of the same model.
    e->states->clear(false);
    e->images->clear(false);
    e->turn_images.clear();
    e->utf8_pending.clear();
    e->turn           = TurnStats{};
    e->think_open     = false;
    e->answer_done    = true;
    e->stop_requested = false;
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
