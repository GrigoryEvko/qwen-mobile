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
 * known image tokenizes as a placeholder, thus its file is not decoded
 * again. A new image decodes in the app (LlamaNative.decodeImage) at the
 * size that the preprocessor selects, thus mtmd gets RGB pixels and
 * copies them without a resize.
 *
 * With speculative decoding (load with speculative = true), the engine
 * holds a second context on the same model with the graph of the MTP
 * block (common/speculative.h, type draft-mtp). One step drafts up to
 * kSpecDraftMax tokens there, decodes the drafted tokens together with
 * the last sampled token on the target context, and gives the tokens that
 * the target sampler also selects. The draft length follows the measured
 * acceptance (spec_policy.h). The step rolls the rejected tokens back with
 * llama_memory_seq_rm: the target context holds n_rs_seq recurrent state
 * snapshots, one for each position of the verified batch, thus the
 * rollback needs no state copy. The model memory holds exactly the tokens
 * that the answer gives, thus the snapshot store stays exact.
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
#include "spec_policy.h"
#include "speculative.h"
#include "state_cache.h"
#include "trace.h"

#define TAG "QwenMobile"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

/** The target duration of one decode call for the ADPF session, 40 ms. */
constexpr int64_t kHintTargetNs = 40'000'000;

/** The number of tokens that the presence penalty looks back on. */
constexpr int32_t kPenaltyLastN = 256;

/**
 * The range of the token limit of one image, which the app sets for each
 * load. 576 tokens is a 768 x 768 image. 768 tokens (3072 patches) is the
 * ceiling until the encoder is correct on the NPU at 4096 patches: a
 * 1024-token photo gave an empty answer there.
 */
constexpr int32_t kImageTokensMin = 64;
constexpr int32_t kImageTokensMax = 768;

/** The budgets of the cache of encoded images, in bytes. One image is 2 to 6 MB at 2048 floats for each of up to 768 tokens. */
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

/**
 * The batch of the contexts. A prompt longer than this value goes to the
 * backend as more than one ubatch, and each ubatch pays the dispatch cost
 * one time. No Hexagon kernel puts an upper bound on the tokens of one op.
 */
constexpr int kBatch = 1024;
/** Not a llama_decode code: a stop request arrived between two batches of a prompt. */
constexpr int kDecodeStopped = 1 << 16;

/** The number of free positions that a prompt must leave in the context. */
constexpr size_t kContextHeadroom = 8;

/** Working local references of a call, above the one that each image takes. */
constexpr jint kLocalRefHeadroom = 16;

/** The sequence of the conversation in each context. */
constexpr llama_seq_id kSeqMain = 0;

/**
 * The maximum number of tokens that one step drafts. The target context
 * keeps this number of recurrent state snapshots, thus a step that accepts
 * no drafted token rolls its state back without a state copy. The policy
 * (spec_policy.h) selects the length of each step between 0 and this value.
 */
constexpr int32_t kSpecDraftMax = SpecPolicy::kDraftMax;

static_assert(kMemTokenNull == LLAMA_TOKEN_NULL, "The null token of the state store must be LLAMA_TOKEN_NULL");

/** Give a thread the compute priority. tid 0 is the calling thread. */
void lower_priority(int32_t tid) {
    if (setpriority(PRIO_PROCESS, tid, kComputeNice) != 0) {
        LOGE("setpriority for thread %d failed", tid);
    }
}

/** The counters of one turn, for the stats line. Times are in microseconds. */
struct TurnStats {
    /**
     * The tokens of the prompt that decoded in batches, and the time of those
     * batches. The generation prompt is not here: it decodes after the
     * snapshot, at the speed of a single token, and one rate over the two
     * would depend on the length of the prompt.
     */
    int64_t prefill_tokens = 0;
    int64_t prefill_us     = 0;
    /** The generation prompt: its tokens and the time of its decode. */
    int64_t tail_tokens    = 0;
    int64_t tail_us        = 0;
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
    /** The vision tokens of all images of the prompt. */
    int64_t image_tokens   = 0;
    int64_t gen_tokens     = 0;
    int64_t gen_us         = 0;
    /** The part of gen_us in the sampler chain: one call for each position with logits. */
    int64_t sample_us      = 0;
    /** The tokens that the MTP draft context proposed, and the ones that the target sampler accepted. */
    int64_t drafted        = 0;
    int64_t accepted       = 0;
    /**
     * The steps of the answer, with and without a draft.
     *
     * The mean draft length is drafted / gen_steps. Without this field a reader
     * must compute the steps as gen_tokens - accepted, which is not obvious.
     */
    int64_t gen_steps      = 0;
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
    /** The token limit of one image, from the load, in [kImageTokensMin, kImageTokensMax]. */
    int32_t           image_max_tokens = 0;
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
    /**
     * Speculative decoding. The draft context runs the graph of the MTP
     * block of the same model, thus it holds no second copy of the weights:
     * it adds the KV cache of one attention layer and the compute buffers of
     * that graph. spec_init owns the context, spec owns the driver.
     */
    common_speculative_init_result_ptr spec_init;
    llama_context *      ctx_dft = nullptr;
    common_speculative * spec    = nullptr;
    /** True when the model holds the MTP tensors and the backend can use them. The app asks with hasMtp. */
    bool mtp_ready = false;
    /** False while the benchmark runs: the draft context must not follow those decodes. */
    bool spec_feed = false;
    /** The draft of the step that runs, and the tokens that the target context holds (text only). */
    llama_tokens draft;
    llama_tokens spec_prompt;
    /** The draft length policy of this answer. */
    SpecPolicy policy;
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
    /**
     * True until chatStart decodes a prompt, and again after the end of the
     * answer: the end token, a stop, an error, or the context limit. While
     * it is true the context can hold no logits, thus generateNext samples
     * nothing.
     */
    bool answer_done = true;
    /**
     * The tokens of the answer that the app did not receive, and the next of
     * them. One speculative step gives up to kSpecDraftMax + 1 tokens, and
     * generateNext gives one token for each call.
     */
    std::vector<llama_token> out_queue;
    size_t                   out_next = 0;
    /**
     * The token that a step sampled last: the app received it, and the next
     * step decodes it together with its draft. LLAMA_TOKEN_NULL before the
     * first step of an answer.
     */
    llama_token id_last = LLAMA_TOKEN_NULL;
    /** True when the queue holds the last tokens of the answer. */
    bool answer_ends = false;

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
        // The driver releases the backend samplers of the draft context, thus it goes first.
        if (spec != nullptr) common_speculative_free(spec);
        if (ctx_dft != nullptr) llama_detach_threadpool(ctx_dft);
        spec_init.reset();
        ctx_dft = nullptr;
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

std::string jstring_to_std(JNIEnv * env, jstring s);

/**
 * Throw a java.lang.RuntimeException with the message.
 *
 * A pending exception stays: it carries the first failure, and the class
 * lookup below would fail again while it stands. FindClass returns null
 * when the Java heap is full, and ThrowNew on a null class is a crash,
 * thus the result is checked.
 */
void throw_java(JNIEnv * env, const std::string & msg) {
    LOGE("%s", msg.c_str());
    if (env->ExceptionCheck()) {
        return;
    }
    jclass cls = env->FindClass("java/lang/RuntimeException");
    if (cls == nullptr) {
        return;
    }
    env->ThrowNew(cls, msg.c_str());
    env->DeleteLocalRef(cls);
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
        // 0xF8 and above, and 0xC0 and 0xC1, start no sequence: a byte-fallback
        // token carries them alone, thus they must not hold later bytes back.
        if (c >= 0xF8 || c < 0xC2) len = 1;
        else if (c >= 0xF0) len = 4;
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
    // A value that is not a finite number (NaN from a damaged settings file)
    // takes the recommended value of the mode. With NaN logits the dist
    // sampler finds no token: an assert in a build with assertions, and the
    // last candidate in a release build.
    if (!std::isfinite(temp)) {
        temp = thinking ? 1.0f : 0.7f;
    }
    if (!std::isfinite(top_p)) {
        top_p = thinking ? 0.95f : 0.8f;
    }
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
 * Let the MTP draft context follow a batch that the target context decoded.
 * The draft block reads the hidden state of each token of the batch from the
 * target context and writes its own KV cells at the same positions, thus the
 * next draft starts from the state of the answer. Returns false when the
 * draft context failed: the caller then decodes without a draft.
 */
bool spec_follow(Engine & e, const llama_batch & b) {
    if (e.spec == nullptr || !e.spec_feed) {
        return true;
    }
    TraceSection trace("spec-follow");
    if (common_speculative_process(e.spec, b)) {
        return true;
    }
    LOGE("the MTP draft context did not follow a batch of %d tokens, the answer continues without a draft", b.n_tokens);
    return false;
}

/**
 * Stop the speculative decoding of this engine after a failure of the draft
 * context. The target context and the answer are not affected.
 */
void spec_disable(Engine & e) {
    if (e.spec != nullptr) {
        common_speculative_free(e.spec);
        e.spec = nullptr;
    }
    e.draft.clear();
}

/**
 * Decode text tokens into sequence kSeqMain of a context, in chunks of
 * kBatch, at explicit positions from pos0. Only the last token gives logits,
 * and only with logits_last. With stoppable, a stop request ends the call
 * before its next chunk with kDecodeStopped: the chunks before it are in the
 * memory. Returns the llama_decode code, 0 on success.
 */
int decode_text(Engine & e, llama_context * lctx, const llama_token * tokens, int n, llama_pos pos0, bool logits_last,
                bool stoppable) {
    llama_batch & b = e.batch;
    for (int i = 0; i < n; i += kBatch) {
        if (stoppable && e.stop_requested.load()) {
            return kDecodeStopped;
        }
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
        // Only the decode context holds the answer. The prefill context of the
        // hybrid backend has its own model, thus the draft block cannot read it.
        if (lctx == e.ctx && !spec_follow(e, b)) {
            spec_disable(e);
        }
    }
    return 0;
}

/** Record a token that the decode context holds at the end of its sequence. */
void commit_token(Engine & e, llama_token token) {
    e.cache.push_back(MemItem{token, {}});
    e.n_past += 1;
    if (e.spec != nullptr) {
        e.spec_prompt.push_back(token);
    }
}

/** Decode one token of the answer on the decode context. Returns the llama_decode code. */
int decode_one(Engine & e, llama_token token) {
    TraceSection trace("decode-step");
    const int rc = decode_text(e, e.ctx, &token, 1, e.n_past, true, false);
    if (rc == 0) {
        commit_token(e, token);
    }
    return rc;
}

/** Empty the model memory of every context and the record of what they hold. The snapshot store stays. */
void clear_all(Engine & e) {
    llama_memory_clear(llama_get_memory(e.ctx), true);
    if (e.ctx_pf != nullptr) {
        llama_memory_clear(llama_get_memory(e.ctx_pf), true);
    }
    if (e.ctx_dft != nullptr) {
        llama_memory_clear(llama_get_memory(e.ctx_dft), false);
    }
    e.cache.clear();
    e.n_past = 0;
    e.pf_cache.clear();
    e.pf_n_past = 0;
    e.spec_prompt.clear();
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

/**
 * The namespace of the disk caches: the model and projector files with
 * their sizes and times, and the image token limit. The limit is part of
 * the key because the state after an image depends on its token count,
 * and the items of a snapshot name the image only by its file hash.
 */
std::string cache_namespace(const std::string & model, const std::string & mmproj, int32_t image_max_tokens) {
    std::string key;
    for (const std::string & path : {model, mmproj}) {
        cache_io::FileStat st;
        key += path + "|";
        if (!path.empty() && cache_io::stat_file(path, st)) {
            key += std::to_string(st.size) + "|" + std::to_string(st.mtime);
        }
        key += "|";
    }
    key += std::to_string(image_max_tokens);
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
        const std::string ns   = cache_namespace(model_path, e.mmproj, e.image_max_tokens);
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

/**
 * Make the MTP draft context and the driver of the speculative decoding.
 * The context runs the graph of the MTP block on the weights of the loaded
 * model, thus it adds the KV cache of one attention layer and the compute
 * buffers of that graph, and no second copy of the weights.
 *
 * Returns false with the reason in the log. The engine then decodes one
 * token at a time, and the answers are the same.
 */
bool setup_speculative(Engine & e, const std::string & model_path) {
    // A rejected draft token rolls the recurrent state back by one position
    // for each rejected token. Without the snapshots the step would have to
    // copy the whole state of the sequence, which is 20 MB and more.
    if (llama_n_rs_seq(e.ctx) < (uint32_t) kSpecDraftMax) {
        LOGE("the context keeps %u recurrent state snapshots of the %d that a draft needs, thus the engine does not draft",
             llama_n_rs_seq(e.ctx), kSpecDraftMax);
        return false;
    }
    common_params params;
    params.model.path                = model_path;
    params.n_ctx                     = (int32_t) llama_n_ctx(e.ctx);
    params.n_batch                   = kBatch;
    params.n_ubatch                  = kBatch;
    params.n_parallel                = 1;
    params.kv_unified                = true;
    params.no_perf                   = true;
    params.cpuparams.n_threads       = e.n_threads;
    params.cpuparams_batch.n_threads = e.n_threads;
    params.speculative.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max   = kSpecDraftMax;

    const int64_t t0 = now_us();
    try {
        // The context of the draft comes from the target model: the same
        // weights, the graph of the MTP block, and its own memory.
        common_params params_dft = common_base_params_to_speculative(params);
        e.spec_init = common_speculative_init_from_params(params_dft, e.model, e.ctx);
    } catch (const std::exception & ex) {
        LOGE("the MTP draft context did not initialize: %s", ex.what());
        e.spec_init.reset();
        return false;
    }
    e.ctx_dft = e.spec_init ? e.spec_init->context() : nullptr;
    if (e.ctx_dft == nullptr) {
        LOGE("the MTP draft context did not initialize");
        e.spec_init.reset();
        return false;
    }
    params.speculative.draft.ctx_tgt = e.ctx;
    params.speculative.draft.ctx_dft = e.ctx_dft;
    try {
        e.spec = common_speculative_init(params.speculative, 1);
    } catch (const std::exception & ex) {
        LOGE("the speculative driver did not initialize: %s", ex.what());
        e.spec = nullptr;
    }
    if (e.spec == nullptr) {
        e.spec_init.reset();
        e.ctx_dft = nullptr;
        return false;
    }
    if (e.tp != nullptr) {
        llama_attach_threadpool(e.ctx_dft, e.tp, e.tp);
    }
    LOGI("speculative decoding on: %d MTP layers, draft of %d tokens, %u state snapshots, ready in %.0f ms",
         llama_model_n_layer_nextn(e.model), kSpecDraftMax, llama_n_rs_seq(e.ctx), (now_us() - t0) / 1000.0);
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
    mp.image_max_tokens = e.image_max_tokens;
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
 * image file and the shape of the chunk, or from one run of the vision
 * encoder. mtmd_helper_decode_image_chunk reads n_tokens x n_embd floats
 * from the pointer, thus an entry of another shape is not a hit (the cache
 * drops it). The pointer is valid until the next call. Returns nullptr
 * with the error text set.
 */
const float * image_embd(Engine & e, const mtmd_input_chunk * chunk, std::string & error) {
    const std::string id       = mtmd_input_chunk_get_id(chunk);
    const uint32_t    n_tokens = (uint32_t) mtmd_input_chunk_get_n_tokens(chunk);
    const uint32_t    n_embd   = (uint32_t) llama_model_n_embd_inp(e.model);
    if (const float * hit = e.images->get(id, n_tokens, n_embd)) {
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
    int32_t rc = 0;
    {
        TraceSection trace("image-encode");
        rc = mtmd_encode_chunk(e.mctx, chunk);
    }
    report_hint(e, t0);
    if (rc != 0) {
        error = "The vision encoder failed with code " + std::to_string(rc);
        return nullptr;
    }
    e.turn.images_encoded += 1;
    ImageInfo info = known->second;
    info.n_tokens  = n_tokens;
    info.n_embd    = n_embd;
    const float * out = mtmd_get_output_embd(e.mctx);
    e.images->put(id, info, out);
    LOGI("image %s encoded in %.0f ms, %u tokens, %.1f MB, cache %zu images %.0f MB in RAM, %.0f MB on disk",
         id.substr(0, 12).c_str(), (now_us() - t0) / 1000.0, info.n_tokens, info.n_floats() * 4 / 1048576.0,
         e.images->count(), e.images->ram_bytes() / 1048576.0, e.images->disk_bytes() / 1048576.0);
    // The output buffer of the encoder stays valid until the next encode, thus it serves when the cache did not keep the copy.
    const float * kept = e.images->get(id, n_tokens, n_embd);
    return kept != nullptr ? kept : out;
}

/** The outcome of a prompt decode. */
enum class DecodeOutcome { kDone, kStopped, kFailed };

/**
 * Decode the items [start, end) into sequence kSeqMain of a context. Text
 * runs go in batches, an image chunk goes through its encoder output with
 * the M-RoPE positions of the helper. pos advances. A stop request stops
 * the decode before the next batch or image and gives kStopped. Gives
 * kFailed with the error text set.
 */
DecodeOutcome decode_items(Engine & e, llama_context * lctx, const std::vector<MemItem> & items,
                           const std::vector<const mtmd_input_chunk *> & chunk_of, size_t start, size_t end,
                           llama_pos & pos, bool logits_last, std::string & error) {
    std::vector<llama_token> run;
    run.reserve(end - start);
    auto flush = [&](bool last) {
        if (run.empty()) {
            return DecodeOutcome::kDone;
        }
        const int rc = decode_text(e, lctx, run.data(), (int) run.size(), pos, last, true);
        if (rc == kDecodeStopped) {
            return DecodeOutcome::kStopped;
        }
        if (rc != 0) {
            error = "llama_decode failed on the prompt with code " + std::to_string(rc);
            return DecodeOutcome::kFailed;
        }
        pos += (llama_pos) run.size();
        run.clear();
        return DecodeOutcome::kDone;
    };
    for (size_t i = start; i < end; ++i) {
        const mtmd_input_chunk * chunk = chunk_of[i];
        if (chunk == nullptr) {
            run.push_back(items[i].token);
            continue;
        }
        const DecodeOutcome flushed = flush(false);
        if (flushed != DecodeOutcome::kDone) {
            return flushed;
        }
        // The encoder runs for up to a second, thus a stop request takes effect before it and not after.
        if (e.stop_requested.load()) {
            return DecodeOutcome::kStopped;
        }
        const float * embd = image_embd(e, chunk, error);
        if (embd == nullptr) {
            return DecodeOutcome::kFailed;
        }
        llama_pos new_pos = pos;
        const int64_t t0 = now_us();
        // The helper takes a non-const pointer and only reads the floats.
        const int32_t rc = mtmd_helper_decode_image_chunk(e.mctx, lctx, chunk, const_cast<float *>(embd), pos, kSeqMain,
                                                          kBatch, &new_pos, nullptr, nullptr);
        report_hint(e, t0);
        if (rc != 0) {
            error = "The image did not decode, code " + std::to_string(rc);
            return DecodeOutcome::kFailed;
        }
        pos = new_pos;
    }
    const DecodeOutcome last = flush(logits_last);
    if (last == DecodeOutcome::kDone) {
        // A backend can run the batch asynchronously: the wait belongs to the prefill time, not to the snapshot after it.
        llama_synchronize(lctx);
    }
    return last;
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
 * backend also into the decode context. Gives kFailed with the error
 * text set, or kStopped after a stop request, and then the memory is
 * empty.
 *
 * Exactness: a snapshot holds the state after exactly its items, with the
 * positions of each KV cell (the M-RoPE x and y of an image token
 * included) and the count of positions. A restore puts that state back,
 * and the tail decodes at the same positions as without the store. Thus
 * the logits after the prompt are the same as after one decode of the
 * whole prompt on the same device.
 */
DecodeOutcome prefill(Engine & e, const std::vector<MemItem> & items,
                      const std::vector<const mtmd_input_chunk *> & chunk_of, size_t base_len, std::string & error) {
    TraceSection trace("prefill");
    // The draft context cannot follow a prompt that a snapshot restores or
    // that an image gives, because it reads the hidden state of each token
    // from a decode of the target context. Thus it starts each turn empty and
    // follows the decodes of this turn: its attention then reads only cells
    // that hold tokens of this conversation.
    if (e.ctx_dft != nullptr) {
        llama_memory_clear(llama_get_memory(e.ctx_dft), false);
    }
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
    const DecodeOutcome base = decode_items(e, pctx, items, chunk_of, start, base_len, pos, false, error);
    if (base != DecodeOutcome::kDone) {
        clear_all(e);
        return base;
    }
    // The batches of the prompt end here. What follows is the generation
    // prompt, a handful of tokens that decode at the speed of one token and
    // not at the speed of a batch, thus the two carry their own times.
    e.turn.prefill_tokens = count_tokens(chunk_of, start, base_len);
    e.turn.prefill_us     = now_us() - t1;
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
                return DecodeOutcome::kFailed;
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
            return DecodeOutcome::kFailed;
        }
        e.turn.transfer_us = now_us() - t3;
        e.cache.assign(items.begin(), items.begin() + (ptrdiff_t) base_len);
        e.n_past = pos;
    }
    const int64_t t4 = now_us();
    const DecodeOutcome tail = decode_items(e, e.ctx, items, chunk_of, base_len, items.size(), pos, true, error);
    if (tail != DecodeOutcome::kDone) {
        clear_all(e);
        return tail;
    }
    e.cache  = items;
    e.n_past = pos;
    e.turn.tail_tokens = count_tokens(chunk_of, base_len, items.size());
    e.turn.tail_us     = now_us() - t4;
    LOGI("prefill: %zu of %zu items %s (%lld tokens, %.0f ms), %lld tokens decoded in %.0f ms on %s "
         "(generation prompt %lld tokens in %.0f ms), "
         "transfer %.0f ms, images %d (%d known, %d cached, %d encoded), memory %d positions",
         start, items.size(), e.turn.from_snapshot ? "restored" : "kept", (long long) e.turn.reused_tokens,
         e.turn.restore_us / 1000.0, (long long) e.turn.prefill_tokens, e.turn.prefill_us / 1000.0,
         hybrid ? ggml_backend_dev_name(e.device_pf) : e.device ? ggml_backend_dev_name(e.device) : "CPU",
         (long long) e.turn.tail_tokens, e.turn.tail_us / 1000.0,
         e.turn.transfer_us / 1000.0, e.turn.images_total, e.turn.images_known, e.turn.images_cached,
         e.turn.images_encoded, (int) e.n_past);
    return DecodeOutcome::kDone;
}

/**
 * Sample the token of each position of a verified batch, and accept the
 * drafted tokens that the sampler selects itself. The result holds at least
 * one token: the tokens that the draft got right, and then the token of the
 * sampler. The sampler accepts each token that it returns, thus its
 * penalties see the same answer as a decode of one token at a time. O(n) in
 * the length of the draft.
 */
std::vector<llama_token> sample_and_accept(Engine & e, const llama_tokens & draft) {
    std::vector<llama_token> out;
    out.reserve(draft.size() + 1);
    size_t i = 0;
    for (; i < draft.size(); ++i) {
        const llama_token id = llama_sampler_sample(e.smpl, e.ctx, (int32_t) i);
        out.push_back(id);
        if (draft[i] != id) {
            break;
        }
    }
    if (i == draft.size()) {
        out.push_back(llama_sampler_sample(e.smpl, e.ctx, (int32_t) i));
    }
    return out;
}

/** The error text of an answer that has no free position in the context. */
std::string context_full_text(const Engine & e) {
    return "The context is full (" + std::to_string(llama_n_ctx(e.ctx)) + " tokens). Start a new chat.";
}

/**
 * One step of the answer without a draft: sample the next token from the
 * logits of the context, decode it, and put it in the queue. Returns false
 * with the error text set.
 */
bool plain_step(Engine & e, std::string & error) {
    const llama_vocab * vocab = llama_model_get_vocab(e.model);
    // The draft driver went off (spec_disable) in a step that gave its last
    // token to the app. That token is not in the memory, and the logits of
    // the context belong to the last position of the verified batch. Thus
    // the token decodes first, and the sample reads its logits: the answer
    // is the answer of a decode without a draft.
    if (e.id_last != LLAMA_TOKEN_NULL) {
        const llama_token pending = e.id_last;
        e.id_last = LLAMA_TOKEN_NULL;
        const int rc = decode_one(e, pending);
        if (rc != 0) {
            error = "llama_decode failed during generation with code " + std::to_string(rc);
            return false;
        }
        if ((uint32_t) e.n_past >= llama_n_ctx(e.ctx)) {
            error = context_full_text(e);
            return false;
        }
    }
    llama_token token = LLAMA_TOKEN_NULL;
    {
        TraceSection trace("sample");
        const int64_t t0 = now_us();
        token = llama_sampler_sample(e.smpl, e.ctx, -1);
        e.turn.sample_us += now_us() - t0;
    }
    if (llama_vocab_is_eog(vocab, token)) {
        // The end token goes into the memory, thus a template that renders it lets the next turn extend this one.
        decode_one(e, token);
        e.answer_ends = true;
        return true;
    }
    const int rc = decode_one(e, token);
    if (rc != 0) {
        error = "llama_decode failed during generation with code " + std::to_string(rc);
        return false;
    }
    e.out_queue.push_back(token);
    e.policy.record(0, 0, 1);
    return true;
}

/**
 * One step of the answer with a draft. The MTP block proposes up to
 * n_draft tokens, and one decode of the target context verifies all of
 * them: the sampler of the engine samples each position, and a drafted
 * token that it selects itself is part of the answer. The step gives 1 to
 * n_draft + 1 tokens.
 *
 * Exactness: the tokens come from the same sampler chain in the same order
 * as a decode of one token at a time. The positions of the rejected draft
 * go away before the step returns, thus the model memory holds exactly the
 * tokens of the answer and the snapshot of the next turn is exact.
 *
 * Returns false with the error text set.
 */
bool spec_step(Engine & e, std::string & error) {
    TraceSection trace("spec-step");
    const llama_vocab * vocab   = llama_model_get_vocab(e.model);
    llama_memory_t      mem     = llama_get_memory(e.ctx);
    llama_memory_t      mem_dft = llama_get_memory(e.ctx_dft);

    // The first step of an answer samples from the logits of the prompt.
    if (e.id_last == LLAMA_TOKEN_NULL) {
        llama_token first = LLAMA_TOKEN_NULL;
        {
            TraceSection sample("sample");
            const int64_t t0 = now_us();
            first = llama_sampler_sample(e.smpl, e.ctx, -1);
            e.turn.sample_us += now_us() - t0;
        }
        if (llama_vocab_is_eog(vocab, first)) {
            decode_one(e, first);
            e.answer_ends = true;
            return true;
        }
        e.out_queue.push_back(first);
        e.id_last = first;
    }

    const llama_pos pos0 = e.n_past;
    // The token of the last step and its draft must fit in the context.
    const int room    = (int) llama_n_ctx(e.ctx) - (int) pos0 - 2;
    const int n_draft = std::min(e.policy.next_draft(), std::max(0, room));
    e.draft.clear();
    if (n_draft > 0) {
        TraceSection draft_trace("spec-draft");
        common_speculative_get_draft_params(e.spec, kSeqMain) = {
            /* .drafting = */ true,
            /* .n_max    = */ n_draft,
            /* .pos0     = */ pos0,
            /* .id_last  = */ e.id_last,
            /* .prompt   = */ &e.spec_prompt,
            /* .result   = */ &e.draft,
        };
        common_speculative_draft(e.spec);
        // The draft wrote its own cells from pos0. They go away, thus the
        // step that follows writes each of these positions one time.
        llama_memory_seq_rm(mem_dft, kSeqMain, pos0, -1);
    }

    // The verified batch: the token of the last step, then the draft. Each
    // position gives logits, thus the sampler reads each of them.
    llama_batch & b = e.batch;
    b.n_tokens = 1 + (int) e.draft.size();
    for (int i = 0; i < b.n_tokens; ++i) {
        b.token[i]     = i == 0 ? e.id_last : e.draft[i - 1];
        b.pos[i]       = pos0 + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = kSeqMain;
        b.logits[i]    = 1;
    }
    const int64_t t0 = now_us();
    const int rc = llama_decode(e.ctx, b);
    report_hint(e, t0);
    if (rc != 0) {
        error = "llama_decode failed during generation with code " + std::to_string(rc);
        return false;
    }
    const bool followed = spec_follow(e, b);

    const int64_t ts = now_us();
    const std::vector<llama_token> ids = sample_and_accept(e, e.draft);
    e.turn.sample_us += now_us() - ts;
    const int accepted = (int) ids.size() - 1;
    common_speculative_accept(e.spec, kSeqMain, (uint16_t) accepted);
    e.turn.drafted  += (int64_t) e.draft.size();
    e.turn.accepted += accepted;
    e.policy.record((int) e.draft.size(), accepted, (int) ids.size());

    // The memory holds the token of the last step at pos0, and each accepted
    // token after it. The last token of ids is the token of the next step.
    commit_token(e, e.id_last);
    for (int i = 0; i < accepted; ++i) {
        commit_token(e, ids[i]);
    }

    // An end token stops the answer. It gives no text, and the tokens after it are not part of the answer.
    int eog_at = -1;
    for (int i = 0; i < (int) ids.size(); ++i) {
        if (llama_vocab_is_eog(vocab, ids[i])) {
            eog_at = i;
            break;
        }
    }
    const int n_emit = eog_at >= 0 ? eog_at : (int) ids.size();
    for (int i = 0; i < n_emit; ++i) {
        e.out_queue.push_back(ids[i]);
    }
    if (eog_at < 0) {
        e.id_last = ids.back();
    } else {
        e.answer_ends = true;
        e.id_last     = LLAMA_TOKEN_NULL;
        // The end token is in the memory: the positions after it go away with the rejected draft.
        if (eog_at < accepted) {
            const size_t drop = (size_t) (accepted - eog_at - 1);
            e.cache.resize(e.cache.size() - drop);
            e.spec_prompt.resize(e.spec_prompt.size() - drop);
            e.n_past -= (llama_pos) drop;
        }
    }

    // One call removes every position that the answer does not hold. A
    // recurrent rollback is single use, thus the step calls it one time.
    if (e.n_past <= pos0 + (llama_pos) e.draft.size()) {
        if (!llama_memory_seq_rm(mem, kSeqMain, e.n_past, -1)) {
            error = "The rejected draft did not roll back at position " + std::to_string(e.n_past);
            return false;
        }
    }
    // The draft context rolls back before the end token decodes. Its memory
    // holds every position of the verified batch, and the end token writes one
    // of those positions again: a decode of it while the draft still holds the
    // position gives the draft context a batch that its memory refuses, and
    // that one refusal turns the drafting off for the life of the engine.
    if (followed) {
        llama_memory_seq_rm(mem_dft, kSeqMain, e.n_past, -1);
    } else {
        spec_disable(e);
    }
    if (eog_at >= 0 && eog_at == accepted) {
        const int rc_eog = decode_one(e, ids.back());
        if (rc_eog != 0) {
            error = "llama_decode failed on the end token with code " + std::to_string(rc_eog);
            return false;
        }
    }
    return true;
}

/** The message of the pending Java exception, which this call clears. */
std::string take_exception_message(JNIEnv * env) {
    jthrowable ex = env->ExceptionOccurred();
    if (ex == nullptr) {
        return "no exception";
    }
    env->ExceptionClear();
    jclass cls = env->GetObjectClass(ex);
    jmethodID to_string = env->GetMethodID(cls, "toString", "()Ljava/lang/String;");
    std::string text = "no message";
    if (to_string != nullptr) {
        jstring s = (jstring) env->CallObjectMethod(ex, to_string);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        } else if (s != nullptr) {
            text = jstring_to_std(env, s);
            env->DeleteLocalRef(s);
        }
    }
    env->DeleteLocalRef(cls);
    env->DeleteLocalRef(ex);
    return text;
}

/**
 * The bitmap of a new image: the pixels come from the app
 * (LlamaNative.decodeImage) at the size that the preprocessor selects for
 * the token limit of the engine, as RGBA rows in a direct buffer. The
 * decoder of the platform samples the file in the DCT domain and applies
 * the EXIF orientation, thus mtmd gets the bitmap it would make itself,
 * without its own decode and resize. A file that the platform does not
 * decode goes through the stb decoder of mtmd, and then the preprocessor
 * resizes it. Returns null with the error text set.
 */
mtmd_bitmap * decode_image(JNIEnv * env, jclass native_class, Engine & e, jbyteArray image, std::string & error) {
    TraceSection trace("image-decode");
    jmethodID method = env->GetStaticMethodID(native_class, "decodeImage", "([BI[I)Ljava/nio/ByteBuffer;");
    if (method == nullptr) {
        error = "LlamaNative.decodeImage is missing: " + take_exception_message(env);
        return nullptr;
    }
    jintArray dims = env->NewIntArray(3);
    if (dims == nullptr) {
        error = "The image dimensions did not allocate: " + take_exception_message(env);
        return nullptr;
    }
    jobject buffer = env->CallStaticObjectMethod(native_class, method, image, (jint) e.image_max_tokens, dims);
    if (env->ExceptionCheck()) {
        error = "The image did not decode in the app: " + take_exception_message(env);
        env->DeleteLocalRef(dims);
        return nullptr;
    }
    mtmd_bitmap * bitmap = nullptr;
    if (buffer != nullptr) {
        jint d[3] = {0, 0, 0};
        env->GetIntArrayRegion(dims, 0, 3, d);
        const int   nx     = d[0];
        const int   ny     = d[1];
        const int   stride = d[2];
        const auto * px    = static_cast<const unsigned char *>(env->GetDirectBufferAddress(buffer));
        const jlong  bytes = env->GetDirectBufferCapacity(buffer);
        // The dimensions cross the JNI boundary, thus the products go in 64
        // bits and each side has a limit: nx * 4 as an int overflows at
        // nx = 2^29 and the guard would then accept a buffer of any length.
        constexpr jlong kMaxSide = 1 << 16;
        if (px != nullptr && nx > 0 && ny > 0 && nx <= kMaxSide && ny <= kMaxSide &&
            (jlong) stride >= (jlong) nx * 4 && bytes >= (jlong) stride * ny) {
            // RGBA rows with the stride to packed RGB, the layout of mtmd.
            std::vector<unsigned char> rgb((size_t) nx * ny * 3);
            for (int y = 0; y < ny; ++y) {
                const unsigned char * row = px + (size_t) y * stride;
                unsigned char *       out = rgb.data() + (size_t) y * nx * 3;
                for (int x = 0; x < nx; ++x) {
                    out[x * 3 + 0] = row[x * 4 + 0];
                    out[x * 3 + 1] = row[x * 4 + 1];
                    out[x * 3 + 2] = row[x * 4 + 2];
                }
            }
            bitmap = mtmd_bitmap_init((uint32_t) nx, (uint32_t) ny, rgb.data());
        } else {
            LOGE("the decoded image has an unusable layout: %d x %d, stride %d, %lld bytes", nx, ny, stride, (long long) bytes);
        }
        env->DeleteLocalRef(buffer);
    }
    env->DeleteLocalRef(dims);
    if (bitmap == nullptr) {
        const jsize len = env->GetArrayLength(image);
        jbyte * bytes = env->GetByteArrayElements(image, nullptr);
        if (bytes == nullptr) {
            error = "The image bytes are not readable";
            return nullptr;
        }
        bitmap = mtmd_helper_bitmap_init_from_buf(e.mctx, reinterpret_cast<const unsigned char *>(bytes), (size_t) len,
                                                  false, mtmd_helper_init_opt_default()).bitmap;
        env->ReleaseByteArrayElements(image, bytes, JNI_ABORT);
        if (bitmap == nullptr) {
            error = "The image did not decode";
            return nullptr;
        }
        LOGI("image decoded by mtmd, %u x %u", mtmd_bitmap_get_nx(bitmap), mtmd_bitmap_get_ny(bitmap));
    }
    return bitmap;
}

/**
 * Tokenize the prompt. With images, mtmd replaces each media marker with
 * the chunk of the image in order, and chunk_of points at the image chunks.
 * native_class is LlamaNative, for the decode of a new image in the app.
 * Returns false with the error text set.
 */
bool tokenize_prompt(JNIEnv * env, jclass native_class, Engine & e, const std::string & prompt,
                     const std::vector<jbyteArray> & images, mtmd::input_chunks & chunks, std::vector<MemItem> & items,
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
    const uint32_t n_embd = (uint32_t) llama_model_n_embd_inp(e.model);
    // A known image gives a placeholder bitmap with its dimensions and its
    // id: the same chunk as its decode gives, without the decode of the file
    // and its preprocessing. The id is the SHA-256 of the file bytes. The
    // dimensions of a new image are the target of the preprocessor, thus
    // the placeholder of a later turn gives the same target. The same bytes
    // two times in one prompt decode one time and give two copies of one
    // bitmap, thus the chunks of one id have one shape.
    //
    // A cache entry whose shape is not the shape of its chunk (a file of an
    // earlier build, or a damaged file) goes, and the second pass decodes
    // the bytes of its image. The first pass drops each such entry, thus
    // the second pass finds none.
    for (int pass = 0; pass < 2; ++pass) {
        items.clear();
        chunk_of.clear();
        chunks.ptr.reset();
        mtmd::bitmaps bitmaps;
        // The entries of the images that tokenize as placeholders, by id.
        std::unordered_map<std::string, ImageInfo> known;
        // The bitmaps that this pass decoded, by id. bitmaps owns them.
        std::unordered_map<std::string, const mtmd_bitmap *> decoded;
        e.turn_images.clear();
        e.turn.images_total = (int) images.size();
        e.turn.images_known = 0;
        for (jbyteArray image : images) {
            const jsize len = env->GetArrayLength(image);
            jbyte * bytes = env->GetByteArrayElements(image, nullptr);
            if (bytes == nullptr) {
                error = "The image bytes are not readable";
                return false;
            }
            const std::string id = cache_io::sha256_hex(bytes, (size_t) len);
            env->ReleaseByteArrayElements(image, bytes, JNI_ABORT);
            ImageInfo info;
            mtmd_bitmap * bitmap = nullptr;
            const auto seen = decoded.find(id);
            if (seen != decoded.end()) {
                bitmap = mtmd_bitmap_init(mtmd_bitmap_get_nx(seen->second), mtmd_bitmap_get_ny(seen->second),
                                          mtmd_bitmap_get_data(seen->second));
            } else if (e.images->info(id, info)) {
                bitmap = mtmd_bitmap_init(info.nx, info.ny, nullptr);
                known[id] = info;
                e.turn.images_known += 1;
            } else {
                bitmap = decode_image(env, native_class, e, image, error);
                if (bitmap == nullptr) {
                    return false;
                }
                info.nx = mtmd_bitmap_get_nx(bitmap);
                info.ny = mtmd_bitmap_get_ny(bitmap);
                e.turn_images[id] = info;
                decoded[id] = bitmap;
            }
            mtmd_bitmap_set_id(bitmap, id.c_str());
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
        bool stale = false;
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
                case MTMD_INPUT_CHUNK_TYPE_IMAGE: {
                    const std::string id = mtmd_input_chunk_get_id(chunk);
                    const auto k = known.find(id);
                    const uint32_t n_tokens = (uint32_t) mtmd_input_chunk_get_n_tokens(chunk);
                    if (k != known.end() && (k->second.n_tokens != n_tokens || k->second.n_embd != n_embd)) {
                        LOGE("image %s: the cache holds %u tokens x %u, the prompt needs %u x %u, "
                             "the entry goes and the image decodes again",
                             id.substr(0, 12).c_str(), k->second.n_tokens, k->second.n_embd, n_tokens, n_embd);
                        e.images->drop(id);
                        known.erase(k);
                        stale = true;
                    }
                    items.push_back(MemItem{LLAMA_TOKEN_NULL, id});
                    chunk_of.push_back(chunk);
                    break;
                }
                default:
                    error = "The prompt holds a media type that this engine does not decode";
                    return false;
            }
        }
        if (!stale) {
            return true;
        }
    }
    error = "The cache of encoded images did not agree with the prompt. Send the message again.";
    return false;
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

/**
 * The table of live engines. A handle is a number that this table gives out
 * and not an address, thus a handle of a released engine matches nothing and
 * a late call cannot reach freed memory.
 *
 * requestStop runs on the thread of the caller without the mutex of the
 * engine, and a model switch can release the engine at the same moment. The
 * table therefore hands out a shared pointer: the engine stays alive while a
 * call holds it, and the destructor runs when the last holder lets go.
 * Complexity: O(1) for each operation.
 */
class EngineTable {
public:
    /** Registers the engine and returns its handle. Never returns 0. */
    jlong add(std::shared_ptr<Engine> engine) {
        std::lock_guard<std::mutex> lock(mutex_);
        const jlong handle = next_++;
        engines_.emplace(handle, std::move(engine));
        return handle;
    }

    /** The engine of the handle, or null when the handle is not live. */
    std::shared_ptr<Engine> get(jlong handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = engines_.find(handle);
        return it == engines_.end() ? nullptr : it->second;
    }

    /** Drops the engine of the handle. A second call with it does nothing. */
    void remove(jlong handle) {
        // The engine leaves the lock before it is destroyed: the destructor
        // joins the writer thread of the state store and must not hold the
        // table against a concurrent lookup.
        std::shared_ptr<Engine> engine;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = engines_.find(handle);
            if (it == engines_.end()) {
                return;
            }
            engine = std::move(it->second);
            engines_.erase(it);
        }
    }

private:
    std::mutex mutex_;
    std::unordered_map<jlong, std::shared_ptr<Engine>> engines_;
    jlong next_ = 1;  // The app reads 0 as "no model", thus a handle starts at 1.
};

EngineTable & engine_table() {
    static EngineTable table;
    return table;
}

/** The engine of the handle, or null when no model of that handle is live. */
std::shared_ptr<Engine> engine_of(jlong handle) {
    return engine_table().get(handle);
}

/**
 * Runs the body of an entry point and turns a C++ exception into a Java one.
 * An exception that leaves a JNI function calls std::terminate, which kills
 * the process with no Java stack. The state snapshot of a turn allocates tens
 * of megabytes on every answer, thus a failed allocation is a real case and
 * must not reach the boundary.
 */
template <typename T, typename Body>
T jni_guard(JNIEnv * env, T fallback, Body && body) {
    try {
        return body();
    } catch (const std::exception & ex) {
        throw_java(env, std::string("Native error: ") + ex.what());
    } catch (...) {
        throw_java(env, "Native error: an exception without a message");
    }
    return fallback;
}

/** The same for an entry point that returns nothing. */
template <typename Body>
void jni_guard_void(JNIEnv * env, Body && body) {
    try {
        body();
    } catch (const std::exception & ex) {
        throw_java(env, std::string("Native error: ") + ex.what());
    } catch (...) {
        throw_java(env, "Native error: an exception without a message");
    }
}

/**
 * Releases the local references that a call collected, at the end of its
 * scope. The reference table of a thread is small, thus a conversation with
 * an image in every message overflows it without this.
 */
class LocalRefs {
public:
    explicit LocalRefs(JNIEnv * env) : env_(env) {}
    ~LocalRefs() {
        for (jobject ref : refs_) {
            env_->DeleteLocalRef(ref);
        }
    }
    LocalRefs(const LocalRefs &)             = delete;
    LocalRefs & operator=(const LocalRefs &) = delete;

    void keep(jobject ref) { refs_.push_back(ref); }

private:
    JNIEnv *             env_;
    std::vector<jobject> refs_;
};

/**
 * The result of generateNext: byte 0 is the kind of the token (0 text,
 * 1 the thinking opens, 2 the thinking closes), then the complete UTF-8
 * bytes of the pending text. The tag tokens add no text.
 */
jbyteArray pack_piece(JNIEnv * env, Engine & e, int kind) {
    const size_t complete = utf8_complete_prefix(e.utf8_pending);
    jbyteArray out = env->NewByteArray((jsize) (1 + complete));
    if (out == nullptr) {
        // An OutOfMemoryError is pending. The bytes stay for the next piece.
        return nullptr;
    }
    const jbyte k = (jbyte) kind;
    env->SetByteArrayRegion(out, 0, 1, &k);
    env->SetByteArrayRegion(out, 1, (jsize) complete, reinterpret_cast<const jbyte *>(e.utf8_pending.data()));
    e.utf8_pending.erase(0, complete);
    return out;
}

/**
 * The last result of an answer: the close of the thinking one time when the
 * answer ended inside it, else null.
 */
jbyteArray finish_answer(JNIEnv * env, Engine & e) {
    e.answer_done = true;
    e.id_last     = LLAMA_TOKEN_NULL;
    if (e.think_open) {
        // The answer ends inside the thinking: the app gets the close of the thinking one time, then the end.
        e.think_open = false;
        e.utf8_pending.clear();
        return pack_piece(env, e, 2);
    }
    return nullptr;
}

/** Forget the tokens of the answer that the app did not receive. */
void clear_queue(Engine & e) {
    e.out_queue.clear();
    e.out_next = 0;
}

std::string jstring_to_std(JNIEnv * env, jstring s) {
    // With a pending exception, JNI permits only some calls, and CheckJNI stops
    // the app on each other call. The caller examines the exception after its reads.
    if (s == nullptr || env->ExceptionCheck()) {
        return {};
    }
    const char * chars = env->GetStringUTFChars(s, nullptr);
    if (chars == nullptr) {
        // An OutOfMemoryError is pending: the caller throws nothing more, and the empty text goes nowhere.
        return {};
    }
    std::string out(chars);
    env->ReleaseStringUTFChars(s, chars);
    return out;
}

/** The string element i of a Java array, with its local reference released. Empty with a pending exception. */
std::string array_string(JNIEnv * env, jobjectArray array, jsize i) {
    if (env->ExceptionCheck()) {
        return {};
    }
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
static void init_impl(JNIEnv * env, jstring jlibdir) {
    llama_log_set(log_to_logcat, nullptr);
    // Op fusion is correct with the patch of the fused matvec add (patches/hexagon-fusion/0001)
    // and gives the F16 file a faster prefill. The fused recurrent state step is verified: a
    // one-token KL run is at the same floor as the unfused path (patches/hexagon-fusion/0002 to 0004).
    setenv("GGML_HEXAGON_OPFUSION", "1", 0);
    setenv("GGML_HEXAGON_OPFUSION_STATE", "1", 0);
    const std::string libdir = jstring_to_std(env, jlibdir);
    if (env->ExceptionCheck()) {
        // The directory did not read (OutOfMemoryError). The exception goes to the app, and no backend loads without it.
        return;
    }
    if (!libdir.empty()) {
        setenv("ADSP_LIBRARY_PATH", libdir.c_str(), 1);
        ggml_backend_load_all_from_path(libdir.c_str());
    }
    llama_backend_init();
    LOGI("llama.cpp initialized, %zu backend devices, libdir %s, trace sections %s", ggml_backend_dev_count(),
         libdir.c_str(), TraceSection::available() ? "available" : "unavailable");
}

JNIEXPORT void JNICALL
Java_ai_airi_qwenmobile_LlamaNative_init(JNIEnv * env, jclass, jstring jlibdir) {
    jni_guard_void(env, [&] { init_impl(env, jlibdir); });
}

/**
 * Make the directory the current one. The OpenCL profiling build writes
 * cl_profiling.csv into the current directory when the backend closes.
 */
JNIEXPORT void JNICALL
Java_ai_airi_qwenmobile_LlamaNative_setWorkingDirectory(JNIEnv * env, jclass, jstring jpath) {
    jni_guard_void(env, [&] {
        const std::string path = jstring_to_std(env, jpath);
        if (env->ExceptionCheck()) {
            return;
        }
        if (chdir(path.c_str()) != 0) {
            LOGE("chdir to %s failed", path.c_str());
        }
    });
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

static jstring devices_impl(JNIEnv * env) {
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

JNIEXPORT jstring JNICALL
Java_ai_airi_qwenmobile_LlamaNative_devices(JNIEnv * env, jclass) {
    return jni_guard<jstring>(env, nullptr, [&] { return devices_impl(env); });
}

static jlong load_impl(JNIEnv * env, jstring jpath, jstring jmmproj,
                       jstring jdevice, jstring jprefill, jstring jvision, jint gpu_layers,
                       jint n_threads, jint n_ctx, jint image_max_tokens, jboolean jspeculative,
                       jstring jcache) {
    const std::string path    = jstring_to_std(env, jpath);
    const std::string device  = jstring_to_std(env, jdevice);
    const std::string prefill = jstring_to_std(env, jprefill);
    const std::string vision  = jstring_to_std(env, jvision);
    const std::string cache   = jstring_to_std(env, jcache);
    const std::string mmproj  = jstring_to_std(env, jmmproj);
    if (env->ExceptionCheck()) {
        // A string did not read (OutOfMemoryError). That exception goes to the app.
        return 0;
    }
    if (n_ctx < kBatch) {
        throw_java(env, "The context length must be at least " + std::to_string(kBatch) + " tokens, not " + std::to_string(n_ctx));
        return 0;
    }
    // The limit is a hard range of the engine: the encoder is not correct at 4096 patches on the NPU.
    if (image_max_tokens < kImageTokensMin || image_max_tokens > kImageTokensMax) {
        throw_java(env, "The image token limit must be between " + std::to_string(kImageTokensMin) + " and " +
                            std::to_string(kImageTokensMax) + ", not " + std::to_string(image_max_tokens));
        return 0;
    }
    // The destructor of the engine releases what loaded when a later step fails.
    auto e = std::make_shared<Engine>();
    e->n_threads  = std::max(1, (int) n_threads);
    e->gpu_layers = gpu_layers;
    e->mmproj     = mmproj;
    e->vision_device    = vision;
    e->image_max_tokens = image_max_tokens;
    const bool hybrid = !prefill.empty();

    // The MTP block of the file loads only for a speculative engine: it is one
    // decoder layer more, and no other path of the engine reads it. The hybrid
    // backend prefills on a second model, whose hidden states the draft block
    // cannot read, thus it decodes one token at a time.
    const bool want_spec = jspeculative == JNI_TRUE && !hybrid;

    // The device by its ggml name: GPUOpenCL for the Adreno, HTP0 for the Hexagon NPU.
    std::vector<ggml_backend_dev_t> devices;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    mp.load_mtp     = want_spec;
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
        // A file that names MTP layers but holds no MTP tensors fails only with want_spec.
        throw_java(env, "The model did not load: " + path +
                            (want_spec ? " (with its MTP block. Set the speculative switch to off for this file.)" : ""));
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
    // The recurrent layers keep one state snapshot for each drafted position
    // of a verified batch, thus a rejected draft rolls back without a copy of
    // the state. The architecture must support the rollback, and
    // setup_speculative reads the value that the context gave.
    if (want_spec) {
        cp.n_rs_seq = (uint32_t) kSpecDraftMax;
    }
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

    // The MTP tensors of the file, and a backend that can use them. The app
    // asks with hasMtp, thus its switch is available for this model.
    e->mtp_ready = llama_model_n_layer_nextn(e->model) > 0 && !hybrid;
    if (want_spec && e->mtp_ready && !setup_speculative(*e, path)) {
        LOGE("the engine decodes without a draft");
    }

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

    LOGI("model loaded: %s, device=%s, prefill=%s, gpu_layers=%d, threads=%d, n_ctx=%u, mmproj=%s, image tokens %d, draft %s",
         path.c_str(), device.empty() ? "cpu" : device.c_str(), prefill.empty() ? "same" : prefill.c_str(),
         gpu_layers, e->n_threads, llama_n_ctx(e->ctx), e->mmproj.empty() ? "none" : e->mmproj.c_str(),
         e->image_max_tokens, e->spec != nullptr ? "on" : e->mtp_ready ? "off" : "absent");
    return engine_table().add(std::move(e));
}

JNIEXPORT jlong JNICALL
Java_ai_airi_qwenmobile_LlamaNative_load(JNIEnv * env, jclass, jstring jpath, jstring jmmproj,
                                          jstring jdevice, jstring jprefill, jstring jvision, jint gpu_layers,
                                          jint n_threads, jint n_ctx, jint image_max_tokens, jboolean jspeculative,
                                          jstring jcache) {
    return jni_guard<jlong>(env, 0, [&] {
        return load_impl(env, jpath, jmmproj, jdevice, jprefill, jvision, gpu_layers,
                         n_threads, n_ctx, image_max_tokens, jspeculative, jcache);
    });
}

JNIEXPORT void JNICALL
Java_ai_airi_qwenmobile_LlamaNative_free(JNIEnv * env, jclass, jlong handle) {
    jni_guard_void(env, [&] { engine_table().remove(handle); });
}

JNIEXPORT jstring JNICALL
Java_ai_airi_qwenmobile_LlamaNative_modelInfo(JNIEnv * env, jclass, jlong handle) {
    return jni_guard<jstring>(env, nullptr, [&]() -> jstring {
        const std::shared_ptr<Engine> e = engine_of(handle);
        if (e == nullptr) {
            throw_java(env, "No model is loaded");
            return nullptr;
        }
        char desc[256];
        llama_model_desc(e->model, desc, sizeof(desc));
        char line[512];
        snprintf(line, sizeof(line),
                 "%s, %.2f GiB, %.2f B params, n_ctx %u, gpu layers %d, threads %d, ADPF %s, vision %s, %d image tokens, MTP %s",
                 desc, llama_model_size(e->model) / 1073741824.0, llama_model_n_params(e->model) / 1e9,
                 llama_n_ctx(e->ctx), e->gpu_layers, e->n_threads, e->hint && e->hint->ok() ? "on" : "off",
                 e->mmproj.empty() ? "none" : (e->mctx ? "loaded" : "ready"), e->image_max_tokens,
                 e->spec != nullptr ? "on" : e->mtp_ready ? "off" : "absent");
        return env->NewStringUTF(line);
    });
}

/**
 * True when the model file holds the MTP tensors and the backend of this
 * engine can draft with them. The app enables its switch with this answer,
 * and a change of the switch loads the model again.
 */
JNIEXPORT jboolean JNICALL
Java_ai_airi_qwenmobile_LlamaNative_hasMtp(JNIEnv *, jclass, jlong handle) {
    const std::shared_ptr<Engine> e = engine_of(handle);
    return e != nullptr && e->mtp_ready ? JNI_TRUE : JNI_FALSE;
}

static jint chat_start_impl(JNIEnv * env, jclass native_class, jlong handle,
                            jobjectArray roles, jobjectArray contents,
                            jobjectArray images, jboolean thinking,
                            jfloat temperature, jfloat top_p) {
    const std::shared_ptr<Engine> engine = engine_of(handle);
    if (engine == nullptr) {
        throw_java(env, "No model is loaded");
        return -1;
    }
    Engine * e = engine.get();
    std::lock_guard<std::mutex> lock(e->mutex);
    // A stop request of the last answer does not stop this one. A request
    // that arrives after this line stops the prompt decode or the first sample.
    e->stop_requested = false;

    // A message with an image starts with the media marker. mtmd replaces
    // the marker with the vision tokens of that image, in message order.
    // The reference of an image lives until tokenize_prompt has read its
    // bytes, thus LocalRefs releases them all at the end of this call. The
    // roles and the contents release theirs inside array_string.
    common_chat_templates_inputs inputs;
    std::vector<jbyteArray> image_refs;
    LocalRefs refs(env);
    const jsize n = env->GetArrayLength(roles);
    if (env->GetArrayLength(contents) != n || (images != nullptr && env->GetArrayLength(images) != n)) {
        throw_java(env, "The roles, contents and images arrays have different lengths");
        return -1;
    }
    // The table of a thread holds a small number of references. A conversation
    // with an image in every message needs one for each, thus the capacity is
    // requested before the loop.
    if (images != nullptr && env->EnsureLocalCapacity(n + kLocalRefHeadroom) != JNI_OK) {
        throw_java(env, "The local reference table cannot hold " + std::to_string(n) + " images");
        return -1;
    }
    for (jsize i = 0; i < n; ++i) {
        common_chat_msg msg;
        msg.role    = array_string(env, roles, i);
        msg.content = array_string(env, contents, i);
        if (env->ExceptionCheck()) {
            // A string did not read (OutOfMemoryError). That exception goes to the app.
            return -1;
        }
        jbyteArray image = images ? (jbyteArray) env->GetObjectArrayElement(images, i) : nullptr;
        if (image != nullptr) {
            msg.content = std::string(mtmd_default_marker()) + "\n" + msg.content;
            image_refs.push_back(image);
            refs.keep(image);
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
    // The answer stays done until the prompt is in the memory: a failure below leaves no logits to sample.
    e->answer_done    = true;
    e->answer_ends    = false;
    e->id_last        = LLAMA_TOKEN_NULL;
    e->spec_feed      = true;
    e->draft.clear();
    e->policy.reset();
    clear_queue(*e);

    std::string error;
    mtmd::input_chunks chunks;
    std::vector<MemItem> items;
    std::vector<const mtmd_input_chunk *> chunk_of;
    if (!tokenize_prompt(env, native_class, *e, prompt, image_refs, chunks, items, chunk_of, error)) {
        throw_java(env, error);
        return -1;
    }
    const int64_t n_tokens = count_tokens(chunk_of, 0, items.size());
    if (items.empty() || (uint64_t) n_tokens + kContextHeadroom >= llama_n_ctx(e->ctx)) {
        throw_java(env, "The conversation is longer than the context (" + std::to_string(n_tokens) + " tokens)");
        return -1;
    }
    e->turn.image_tokens = n_tokens - (int64_t) std::count(chunk_of.begin(), chunk_of.end(), nullptr);
    size_t base_len = base_length(*e, prompt, tail, items);
    if (base_len == items.size()) {
        // The decode context must decode the last item itself: the state
        // transfer of the hybrid backend carries no logits, and on every
        // backend an empty tail leaves a context that holds no logits, which
        // the sampler reads as a null pointer and aborts the process.
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

    switch (prefill(*e, items, chunk_of, base_len, error)) {
        case DecodeOutcome::kFailed:
            throw_java(env, error);
            return -1;
        case DecodeOutcome::kStopped:
            // The memory is empty and the answer is done: the next generateNext gives null.
            return 0;
        case DecodeOutcome::kDone:
            break;
    }
    if (e->spec != nullptr) {
        // The text tokens of the context, for the draft implementations that read them.
        e->spec_prompt.clear();
        e->spec_prompt.reserve(items.size());
        for (const MemItem & item : items) {
            if (item.token != kMemTokenNull) {
                e->spec_prompt.push_back(item.token);
            }
        }
        common_speculative_begin(e->spec, kSeqMain, e->spec_prompt);
    }
    e->answer_done = false;
    return (jint) e->turn.prefill_tokens;
}

JNIEXPORT jint JNICALL
Java_ai_airi_qwenmobile_LlamaNative_chatStart(JNIEnv * env, jclass native_class, jlong handle,
                                               jobjectArray roles, jobjectArray contents,
                                               jobjectArray images, jboolean thinking,
                                               jfloat temperature, jfloat top_p) {
    return jni_guard<jint>(env, -1, [&] {
        return chat_start_impl(env, native_class, handle, roles, contents, images,
                               thinking, temperature, top_p);
    });
}

static jbyteArray generate_next_impl(JNIEnv * env, jlong handle) {
    const std::shared_ptr<Engine> engine = engine_of(handle);
    if (engine == nullptr) {
        throw_java(env, "No model is loaded");
        return nullptr;
    }
    Engine * e = engine.get();
    std::lock_guard<std::mutex> lock(e->mutex);
    if (e->answer_done) {
        return nullptr;
    }
    if (e->stop_requested.exchange(false)) {
        // The answer stops here. A speculative step can hold tokens that the
        // app did not receive: they stay in the model memory, which the
        // snapshot of the next prompt does not read.
        clear_queue(*e);
        e->answer_done = true;
        e->id_last     = LLAMA_TOKEN_NULL;
        return nullptr;
    }
    if ((uint32_t) e->n_past >= llama_n_ctx(e->ctx)) {
        e->answer_done = true;
        throw_java(env, context_full_text(*e));
        return nullptr;
    }
    if (e->cache.empty()) {
        // No prompt is in the memory, thus the context holds no logits and a sample would read a null pointer.
        e->answer_done = true;
        throw_java(env, "No prompt is decoded. Call chatStart first.");
        return nullptr;
    }

    if (e->out_next == e->out_queue.size()) {
        clear_queue(*e);
        if (e->answer_ends) {
            return finish_answer(env, *e);
        }
        // One step gives one token, or up to kSpecDraftMax + 1 tokens with a draft.
        const int64_t t0 = now_us();
        std::string error;
        const bool ok = e->spec != nullptr ? spec_step(*e, error) : plain_step(*e, error);
        const int64_t step_us = now_us() - t0;
        e->turn.gen_us += step_us;
        if (!ok) {
            e->answer_done = true;
            throw_java(env, error);
            return nullptr;
        }
        // The policy predicts the throughput of each draft length from the
        // measured step time and acceptance, thus the step that ran must give
        // it its time.
        e->policy.observe(step_us);
        e->turn.gen_steps += 1;
        if (e->out_queue.empty()) {
            return finish_answer(env, *e);
        }
    }

    const llama_token token = e->out_queue[e->out_next];
    e->out_next += 1;
    const int kind = token_kind(*e, token);
    if (kind == 0) {
        append_piece(*e, token);
    } else {
        e->think_open = kind == 1;
    }
    e->turn.gen_tokens += 1;
    return pack_piece(env, *e, kind);
}

JNIEXPORT jbyteArray JNICALL
Java_ai_airi_qwenmobile_LlamaNative_generateNext(JNIEnv * env, jclass, jlong handle) {
    return jni_guard<jbyteArray>(env, nullptr, [&] { return generate_next_impl(env, handle); });
}

/**
 * Ask the running answer to stop. Any thread can call this: the flag is
 * atomic and the call takes no lock. The next generateNext gives the end
 * of the answer without a sample.
 */
JNIEXPORT void JNICALL
Java_ai_airi_qwenmobile_LlamaNative_requestStop(JNIEnv *, jclass, jlong handle) {
    // Runs on the thread of the caller, without the mutex of the engine and
    // possibly while the engine thread releases the model. The shared pointer
    // keeps the engine alive for the store below, and a handle that is no
    // longer live gives null instead of a freed address.
    const std::shared_ptr<Engine> e = engine_of(handle);
    if (e != nullptr) {
        e->stop_requested = true;
    }
}

static jstring stats_impl(JNIEnv * env, jlong handle) {
    const std::shared_ptr<Engine> engine = engine_of(handle);
    if (engine == nullptr) {
        throw_java(env, "No model is loaded");
        return nullptr;
    }
    Engine * e = engine.get();
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
        // Known: the file was not decoded. Cached: the encoder output came from the cache. The rest of the prompt images sat in the reused prefix.
        extra += ", images " + std::to_string(t.images_total) + " (" + std::to_string(t.images_known) + " known, " +
                 std::to_string(t.images_cached) + " cached, " + std::to_string(t.images_encoded) + " encoded, " +
                 std::to_string(t.image_tokens) + " tok)";
    }
    if (e->spec != nullptr || t.drafted > 0) {
        const int percent = t.drafted > 0 ? (int) ((t.accepted * 100 + t.drafted / 2) / t.drafted) : 0;
        extra += ", drafted " + std::to_string(t.drafted) + ", accepted " + std::to_string(t.accepted) +
                 " (" + std::to_string(percent) + " %)";
        // The mean draft length and the length that the policy settled on. The
        // two differ when the policy changed its choice inside the answer, and
        // together they say why the drafted count is what it is.
        if (t.gen_steps > 0) {
            char mean[32];
            snprintf(mean, sizeof(mean), "%.2f", (double) t.drafted / (double) t.gen_steps);
            extra += " over " + std::to_string(t.gen_steps) + " steps (mean draft " + mean +
                     ", policy " + std::to_string(e->policy.best_draft()) + ")";
        }
    }
    // The generation prompt has its own entry: those few tokens decode at the
    // speed of one token, thus a rate over the prompt and them together would
    // fall with the length of the prompt and would not be a prefill rate.
    if (t.tail_tokens > 0) {
        extra += ", generation prompt " + std::to_string(t.tail_tokens) + " tok in " +
                 std::to_string(t.tail_us / 1000) + " ms";
    }
    snprintf(line, sizeof(line),
             "prefill %lld tok in %.0f ms (%.1f t/s on %s)%s, generate %lld tok (%.1f t/s, sample %.0f ms), memory %d pos",
             (long long) t.prefill_tokens, t.prefill_us / 1000.0, pp, pf_dev, extra.c_str(),
             (long long) t.gen_tokens, tg, t.sample_us / 1000.0, (int) e->n_past);
    return env->NewStringUTF(line);
}

JNIEXPORT jstring JNICALL
Java_ai_airi_qwenmobile_LlamaNative_stats(JNIEnv * env, jclass, jlong handle) {
    return jni_guard<jstring>(env, nullptr, [&] { return stats_impl(env, handle); });
}

static void reset_chat_impl(jlong handle) {
    const std::shared_ptr<Engine> engine = engine_of(handle);
    if (engine == nullptr) {
        return;
    }
    Engine * e = engine.get();
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
    e->answer_ends    = false;
    e->stop_requested = false;
    e->id_last        = LLAMA_TOKEN_NULL;
    e->draft.clear();
    e->policy.reset();
    clear_queue(*e);
}

JNIEXPORT void JNICALL
Java_ai_airi_qwenmobile_LlamaNative_resetChat(JNIEnv * env, jclass, jlong handle) {
    jni_guard_void(env, [&] { reset_chat_impl(handle); });
}

/**
 * The llama-bench method: pp tokens in batches of kBatch, then tg tokens
 * one at a time, each on a clean memory, reps times. The chat memory is
 * empty after the benchmark.
 */
static jstring bench_impl(JNIEnv * env, jlong handle, jint pp, jint tg, jint reps) {
    const std::shared_ptr<Engine> engine = engine_of(handle);
    if (engine == nullptr) {
        throw_java(env, "No model is loaded");
        return nullptr;
    }
    Engine * e = engine.get();
    std::lock_guard<std::mutex> lock(e->mutex);
    llama_memory_t mem = llama_get_memory(e->ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(e->model));
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> pick(0, n_vocab - 1);
    // The benchmark measures the target model. The draft context must not
    // follow these decodes: it would add its own graph to each of them.
    e->spec_feed = false;

    std::vector<double> pp_tps, tg_tps;
    std::string log;
    // A warmup that no rate holds, as llama-bench does: the first pass of a
    // fresh memory runs at the boost clocks of an idle phone and measured
    // 10 % above the passes that follow, thus a mean that holds it reads
    // about 3 % high at three reps.
    {
        llama_context * pctx = e->ctx_pf ? e->ctx_pf : e->ctx;
        const int n_warm = pp > 0 ? std::min(pp, 32) : 1;
        std::vector<llama_token> tokens(n_warm);
        for (auto & t : tokens) t = pick(rng);
        llama_memory_clear(llama_get_memory(pctx), true);
        decode_text(*e, pctx, tokens.data(), n_warm, 0, true, false);
        llama_synchronize(pctx);
        if (tg > 0) {
            llama_memory_clear(mem, true);
            llama_token t = pick(rng);
            decode_text(*e, e->ctx, &t, 1, 0, true, false);
            llama_synchronize(e->ctx);
        }
    }
    for (int r = 0; r < reps; ++r) {
        if (pp > 0) {
            std::vector<llama_token> tokens(pp);
            for (auto & t : tokens) t = pick(rng);
            // The hybrid backend prefills on its prefill context.
            llama_context * pctx = e->ctx_pf ? e->ctx_pf : e->ctx;
            llama_memory_clear(llama_get_memory(pctx), true);
            const int64_t t0 = now_us();
            const int rc = decode_text(*e, pctx, tokens.data(), pp, 0, true, false);
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
                rc = decode_text(*e, e->ctx, &t, 1, i, true, false);
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
    // The configuration belongs with the numbers: a record that names neither
    // device nor the context length cannot be compared with a llama-bench run.
    snprintf(line, sizeof(line), "%s | %s%s%s | gpu layers %d | threads %d | ctx %d | reps %d + warmup\n", desc,
             e->device ? ggml_backend_dev_name(e->device) : "CPU",
             e->device_pf ? " prefill " : "", e->device_pf ? ggml_backend_dev_name(e->device_pf) : "",
             e->gpu_layers, e->n_threads, (int) llama_n_ctx(e->ctx), reps);
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

JNIEXPORT jstring JNICALL
Java_ai_airi_qwenmobile_LlamaNative_bench(JNIEnv * env, jclass, jlong handle,
                                           jint pp, jint tg, jint reps) {
    return jni_guard<jstring>(env, nullptr, [&] { return bench_impl(env, handle, pp, tg, reps); });
}

} // extern "C"
