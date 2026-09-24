/**
 * memprobe: the memory map of one engine with the configuration of the app.
 *
 * The tool loads a model and makes the contexts as android/app/src/main/cpp/llama_jni.cpp
 * does (n_batch = n_ubatch = 1024, one sequence, a unified KV cache, n_rs_seq = 4 and the
 * MTP draft context with --spec). At each stage it prints the process memory from
 * /proc/self/status, /proc/self/smaps_rollup and /proc/self/smaps (grouped by mapping),
 * and the llama.cpp buffers by buffer type. Then it decodes a prompt, measures the
 * decode rate at that depth, and measures the state of the sequence: its size, the time of
 * llama_state_seq_get_data and _set_data, and the time to write it to a file and read it
 * back. It is a measurement tool, not part of the app.
 *
 *   memprobe -m MODEL [-dev NAME|none] [-c N_CTX] [-b N_BATCH] [-ctk TYPE] [-ctv TYPE]
 *            [-fa on|off|auto] [--spec] [-p N_PROMPT] [-n N_GEN] [-t THREADS]
 *            [--state-file PATH] [--smaps] [--outputs-max N]
 *            [--mmproj PATH --image WxH --image-tokens N --vision-dev NAME|none [--vision-warmup]]
 *            [--grow] [--mmap | --no-mmap] [--embd-advise none|random|willneed|touch] [--lazy off|auto|on]
 *            [--hash] [--drop-cache] [--reps N [--rest-ms MS]] [--log-ts] [--therm]
 *            [--cold N] [--sweep N1,N2,.. [--sweep-depths D1,D2,..] [--sweep-calls C] [--sweep-logits last|none]]
 *            [--turns K [--turn-first N] [--turn-message N] [--turn-answer N] [--turn-idle-ms MS] [--state-dir PATH]]
 *
 * With --reps N the tool decodes the prompt N times, each time into a cleared memory with the same
 * tokens, and prints one "TIME prefill" line with rep=I for each pass. The first pass also holds the
 * costs of a first decode (the rate of the first prompt after a load). The passes after it give the
 * rate that the benchmark of the app (bench_impl in llama_jni.cpp) measures after its warmup.
 *
 * With --grow, after the prompt the tool moves the sequence into a context of twice the n_ctx:
 * it copies the state out, frees the context, makes the larger one, copies the state in, and
 * decodes n_gen tokens there. It prints the time of each step, which is the cost of a context
 * that grows with the conversation.
 *
 * --mmap loads the model with LLAMA_LOAD_MODE_MMAP, --no-mmap with LLAMA_LOAD_MODE_NONE (the load
 * of the app on HTP0 on a host without that device). The AUTO mode reads the whole file into
 * memory when one device of the model cannot load from a mapping, thus the tensors of the CPU
 * (the token embedding) take anonymous memory. --hash prints an FNV-1a 64 hash of the logits
 * after the prompt and after each decode, thus two load modes can be compared. --drop-cache asks
 * the kernel to drop the cached pages of the model file before the load (posix_fadvise
 * DONTNEED, no root needed), thus the load reads from the flash.
 *
 * --embd-advise applies to the mappings of the model file that stay after the load (with --mmap
 * the loader unmaps the ranges of the device tensors, thus the rest is the token embedding of the
 * CPU). "random" gives MADV_RANDOM: a page fault reads one page and not a read-around window.
 * "willneed" gives MADV_WILLNEED: the kernel starts to read the ranges into the page cache.
 * "touch" reads one byte of each page, thus the pages are in memory before the first decode.
 * The pages stay file-backed and reclaimable in the three modes. The tool prints the bytes and
 * the time as "TIME embd-advise".
 *
 * --lazy sets llama_model_params.lazy_mode (the preset value of llama.cpp is auto). With "on", a
 * tensor that the architecture marks with TENSOR_READ_LAZY stays in a file mapping with
 * POSIX_MADV_RANDOM whatever the load mode, and the other tensors load as the load mode says.
 *
 * With --mmproj the tool loads the vision projector as ensure_vision in llama_jni.cpp does
 * (warmup off unless --vision-warmup, the image token limit of the app) and encodes one gray
 * image of W x H pixels, which the preprocessor resizes to fit the token limit.
 *
 * --rest-ms MS waits MS milliseconds before each pass of --reps after the first, outside the time of
 * the pass, thus the NPU cools between the passes. --therm prints the highest temperature of the NPU
 * thermal zones (type nsp*) before and after each pass, and before each group of the app modes.
 *
 * --log-ts puts the time since the start of the tool in front of each log line on stderr, in the form
 * of common/log.cpp ("M.SS.mmm.uuu L "), thus tools/trace/htp_trace.py puts each DSP batch of
 * GGML_HEXAGON_PROFILE on the host clock. The tool then also writes STAMP lines on the same clock
 * around each step, thus a script assigns the LLAMA_HOSTPROF lines and the profile lines to the steps.
 *
 * The app modes --cold, --sweep and --turns measure the fixed costs of a chat turn in the engine of
 * the app (load_impl, prefill, plain_step and spec_step in llama_jni.cpp). They make the thread pool of
 * the app, and with --spec also the MTP draft driver of setup_speculative, and the draft context then
 * follows each decode of the target context, as in the app. (The --spec of the other modes makes only
 * the draft context.) The app modes turn --log-ts on and do not go with -p, -n, --grow or --mmproj.
 *
 * --cold N decodes N tokens at depth 0 as the first decode of the process: the first use of the DSP
 * buffers, the first graph and the first pages of the token embedding are in its time.
 *
 * --sweep decodes each size of the list at each depth of --sweep-depths (preset 0) C times (preset 4),
 * with the logits of the last token (--sweep-logits last, the preset) or no logits (none, as the prompt
 * batch of the app). A depth above 0 comes from a restore of a state blob before each call, as the app
 * restores a snapshot. The first call of a size follows a call of another size, thus llama.cpp builds a
 * new graph and the Hexagon session packs a new batch, as for each prompt batch of the app. The other
 * calls reuse the graph and replay the packed batch.
 *
 * --turns K runs a chat of K turns as chat_start_impl and generate_next_impl in llama_jni.cpp: the chat
 * template of the model, the tokenization of the whole conversation, the prefix of the live memory or of
 * the snapshot store (state_cache.cpp, with its disk tier in --state-dir), the prompt batch without the
 * generation prompt, the snapshot, the generation prompt, and the answer with the sampler chain of the
 * app (without thinking) and with the draft policy of the app (spec_policy.cpp). The first message has
 * --turn-first tokens (preset 500), each later message --turn-message tokens (preset 40), and an answer
 * stops at the end token or after --turn-answer tokens (preset 32). The answer text goes into the
 * history of the next turn. --turn-idle-ms waits before each turn, outside the times, as a user who reads
 * and writes. The dist sampler has a fixed seed, thus two rounds give the same answers.
 *
 * Output lines start with a tag, thus a host script can parse them:
 *   MEM <stage> key=value ...        process memory in KiB
 *   MAP <stage> <group> rss pss      smaps totals by mapping group, in KiB
 *   BIGMAP <stage> <range> ...       one mapping of 32 MiB or more, in KiB
 *   DMABUF <stage> <exporter> ...    the DMA buffers of the process by exporter, in MiB
 *   BUF <ctx> <buffer type> model context compute   in MiB
 *   TIME <what> <ms> [extra]
 *   LOADMEM <progress> ...           /proc/meminfo and the process memory during the load, in KiB
 *   HASH <what> <hex>                with --hash, the FNV-1a 64 of a logits vector
 *   THERM <label> nsp=<mC>            with --therm, the highest NPU zone temperature in millidegrees C, -1 if none
 *   TIME cold|sweep|sweep-fill|turn|turn-store key=value ...   the app modes, times in ms
 *   (stderr) memprobe: STAMP <event> key=value ...           with --log-ts, the steps of the app modes and of --reps
 *
 * The app modes compile the sources of the app that have no llama.cpp and no Android dependency
 * (state_cache.cpp, cache_io.cpp and spec_policy.cpp in android/app/src/main/cpp).
 */

#include "chat.h"
#include "common.h"
#include "fit.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-ext.h"
#include "llama.h"
#include "mtmd.h"
#include "spec_policy.h"
#include "speculative.h"
#include "state_cache.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

/** Current time in milliseconds on the steady clock. */
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/** The value in KiB of a "Key:   123 kB" line of a /proc file, or -1. */
long proc_kb(const std::string & path, const std::string & key) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, key.size(), key) == 0 && line.size() > key.size() && line[key.size()] == ':') {
            return std::atol(line.c_str() + key.size() + 1);
        }
    }
    return -1;
}

/**
 * The group of a mapping name of /proc/self/smaps: the model file, the DMA buffers of
 * the DSP and the GPU, the anonymous memory, the libraries, and the rest.
 */
std::string map_group(const std::string & name, const std::string & model_path) {
    if (!model_path.empty() && name.find(model_path) != std::string::npos) return "model-file";
    if (name.find(".gguf") != std::string::npos) return "other-gguf";
    if (name.find("dmabuf") != std::string::npos || name.find("dma_buf") != std::string::npos) return "dmabuf";
    if (name.find("kgsl") != std::string::npos) return "kgsl";
    if (name.find("/dev/") == 0) return "dev:" + name.substr(5, name.find_first_of(" /", 5) - 5);
    if (name.find(".so") != std::string::npos) return "libs";
    if (name.empty() || name == "[heap]" || name.find("[anon") == 0) return "anon";
    if (name == "[stack]") return "stack";
    return "other:" + name.substr(0, 40);
}

/**
 * Print the DMA buffers that this process holds by file descriptor: their count and bytes by
 * exporter (/proc/self/fdinfo "size:" and "exp_name:"). The rpcmem buffers of the Hexagon
 * backend are system-heap DMA buffers, and a PFN mapping does not count in VmRSS, thus this
 * is the only per-process measure of them without root. O(open descriptors).
 */
void print_dmabuf(const char * stage) {
    std::map<std::string, std::pair<long, long long>> by_exp;   // count, bytes
    for (int fd = 0; fd < 4096; ++fd) {
        char link[64], target[256];
        snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
        const ssize_t n = readlink(link, target, sizeof(target) - 1);
        if (n <= 0) continue;
        target[n] = '\0';
        if (strstr(target, "dmabuf") == nullptr) continue;
        char info[64];
        snprintf(info, sizeof(info), "/proc/self/fdinfo/%d", fd);
        std::ifstream f(info);
        std::string line, exp = "unknown";
        long long size = 0;
        while (std::getline(f, line)) {
            if (line.compare(0, 5, "size:") == 0) size = std::atoll(line.c_str() + 5);
            if (line.compare(0, 9, "exp_name:") == 0) {
                exp = line.substr(9);
                exp.erase(0, exp.find_first_not_of(" \t"));
            }
        }
        by_exp[exp].first += 1;
        by_exp[exp].second += size;
    }
    for (const auto & [exp, v] : by_exp) {
        printf("DMABUF %s %s count=%ld mib=%.2f\n", stage, exp.c_str(), v.first, v.second / 1048576.0);
    }
    if (by_exp.empty()) printf("DMABUF %s none\n", stage);
}

/** Print the MEM line and the MAP lines of one stage. O(number of mappings). */
void print_memory(const char * stage, const std::string & model_path, bool smaps) {
    print_dmabuf(stage);
    const char * st = "/proc/self/status";
    const char * sr = "/proc/self/smaps_rollup";
    printf("MEM %s VmRSS=%ld VmHWM=%ld RssAnon=%ld RssFile=%ld RssShmem=%ld Pss=%ld Pss_Anon=%ld Pss_File=%ld "
           "Pss_Shmem=%ld Swap=%ld MemAvailable=%ld\n",
           stage, proc_kb(st, "VmRSS"), proc_kb(st, "VmHWM"), proc_kb(st, "RssAnon"), proc_kb(st, "RssFile"),
           proc_kb(st, "RssShmem"), proc_kb(sr, "Pss"), proc_kb(sr, "Pss_Anon"), proc_kb(sr, "Pss_File"),
           proc_kb(sr, "Pss_Shmem"), proc_kb(sr, "Swap"), proc_kb("/proc/meminfo", "MemAvailable"));
    if (!smaps) {
        fflush(stdout);
        return;
    }
    // smaps: a header line "start-end perms offset dev inode [name]", then "Key: N kB" lines.
    std::ifstream f("/proc/self/smaps");
    std::string line;
    std::string group;
    std::map<std::string, std::pair<long, long>> totals;   // rss, pss
    std::map<std::string, long> vsize;
    // Each mapping of 32 MiB or more gets its own BIGMAP line: the host allocations of the
    // scheduler (a malloc of hundreds of MiB) and the DMA buffers show there one by one.
    std::string cur_name, cur_range;
    long cur_vsize = 0, cur_rss = 0, cur_pss = 0;
    auto flush_big = [&]() {
        if (cur_vsize >= (32l << 10)) {
            printf("BIGMAP %s %s vsize=%ld rss=%ld pss=%ld name=%s\n", stage, cur_range.c_str(), cur_vsize, cur_rss,
                   cur_pss, cur_name.empty() ? "[anon]" : cur_name.c_str());
        }
    };
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const bool header = line.find(':') == std::string::npos || (line[0] >= '0' && line[0] <= '9') ||
                            (line[0] >= 'a' && line[0] <= 'f' && line.find('-') < 17);
        if (header && line.find(" kB") == std::string::npos) {
            std::istringstream is(line);
            std::string range, perms, offset, dev, inode, name;
            is >> range >> perms >> offset >> dev >> inode;
            std::getline(is, name);
            const size_t p = name.find_first_not_of(' ');
            name = p == std::string::npos ? "" : name.substr(p);
            group = map_group(name, model_path);
            const size_t dash = range.find('-');
            const unsigned long long a = std::strtoull(range.substr(0, dash).c_str(), nullptr, 16);
            const unsigned long long b = std::strtoull(range.substr(dash + 1).c_str(), nullptr, 16);
            vsize[group] += (long) ((b - a) >> 10);
            flush_big();
            cur_name  = name;
            cur_range = range;
            cur_vsize = (long) ((b - a) >> 10);
            cur_rss = cur_pss = 0;
            continue;
        }
        if (line.compare(0, 4, "Rss:") == 0) {
            totals[group].first += std::atol(line.c_str() + 4);
            cur_rss = std::atol(line.c_str() + 4);
        }
        if (line.compare(0, 4, "Pss:") == 0) {
            totals[group].second += std::atol(line.c_str() + 4);
            cur_pss = std::atol(line.c_str() + 4);
        }
    }
    flush_big();
    for (const auto & [g, v] : totals) {
        printf("MAP %s %s vsize=%ld rss=%ld pss=%ld\n", stage, g.c_str(), vsize[g], v.first, v.second);
    }
    fflush(stdout);
}

/** Print the llama.cpp buffers of a context by buffer type, in MiB. */
void print_buffers(const char * tag, const llama_context * ctx) {
    const double mib = 1024.0 * 1024.0;
    for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
        printf("BUF %s %s model=%.2f context=%.2f compute=%.2f\n", tag, ggml_backend_buft_name(buft), mb.model / mib,
               mb.context / mib, mb.compute / mib);
    }
    fflush(stdout);
}

ggml_type parse_type(const std::string & s) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * name = ggml_type_name((ggml_type) i);
        if (name != nullptr && s == name) return (ggml_type) i;
    }
    fprintf(stderr, "unknown type %s\n", s.c_str());
    std::exit(2);
}

/** FNV-1a 64 of n bytes. O(n). */
uint64_t fnv_bytes(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const auto * b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/**
 * Apply one advice to each mapping of the model file in /proc/self/maps: "random" (MADV_RANDOM),
 * "willneed" (MADV_WILLNEED) or "touch" (a read of one byte of each page). Print the bytes, the
 * range count and the time. O(mappings + pages of the mappings for "touch").
 */
void advise_model_mappings(const std::string & model_path, const std::string & mode) {
    const double t0 = now_ms();
    std::ifstream f("/proc/self/maps");
    std::string line;
    size_t bytes = 0, ranges = 0;
    int failed = 0;
    volatile uint8_t sink = 0;
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    while (std::getline(f, line)) {
        if (line.find(model_path) == std::string::npos) continue;
        const size_t dash = line.find('-');
        const size_t space = line.find(' ');
        const uintptr_t a = (uintptr_t) std::strtoull(line.substr(0, dash).c_str(), nullptr, 16);
        const uintptr_t b = (uintptr_t) std::strtoull(line.substr(dash + 1, space - dash - 1).c_str(), nullptr, 16);
        const size_t len = (size_t) (b - a);
        if (mode == "random") {
            failed += madvise((void *) a, len, MADV_RANDOM) != 0;
        } else if (mode == "willneed") {
            failed += madvise((void *) a, len, MADV_WILLNEED) != 0;
        } else if (mode == "touch") {
            for (uintptr_t p = a; p < b; p += page) sink = sink + *(const volatile uint8_t *) p;
        }
        bytes += len;
        ranges += 1;
    }
    (void) sink;
    printf("TIME embd-advise %.1f mode=%s ranges=%zu mib=%.2f failed=%d\n", now_ms() - t0, mode.c_str(), ranges,
           bytes / 1048576.0, failed);
    fflush(stdout);
}

void stamp(const char * fmt, ...) __attribute__((format(printf, 1, 2)));

/** Decode tokens in batches of n_batch into sequence 0 from pos0. Logits only for the last token. */
int decode_tokens(llama_context * ctx, llama_batch & b, const std::vector<llama_token> & toks, int pos0, int n_batch) {
    for (size_t i = 0; i < toks.size(); i += n_batch) {
        const int count = (int) std::min<size_t>(n_batch, toks.size() - i);
        for (int j = 0; j < count; ++j) {
            b.token[j]     = toks[i + j];
            b.pos[j]       = pos0 + (int) i + j;
            b.n_seq_id[j]  = 1;
            b.seq_id[j][0] = 0;
            b.logits[j]    = (i + j + 1 == toks.size());
        }
        b.n_tokens = count;
        stamp("decode-begin tokens=%d pos=%d", count, pos0 + (int) i);
        const int rc = llama_decode(ctx, b);
        stamp("decode-end rc=%d", rc);
        if (rc != 0) return rc;
    }
    return 0;
}

// ---- The time stamps of the log (--log-ts) and the STAMP lines ----

/** The start of the tool on the steady clock, in microseconds: the zero of the log time stamps. */
int64_t g_t0_us = 0;
/** True after --log-ts installs log_ts_callback. The STAMP lines exist only then. */
bool g_log_ts = false;
/** The lock of the stderr lines, the state of the line that is open, and the level letter of that line. */
std::mutex g_log_mutex;
bool       g_log_line_start = true;
char       g_log_level      = 'I';

/** Current time in microseconds on the steady clock. */
int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/** Write a text to stderr with the time stamp at the start of each line. The caller holds g_log_mutex. O(length). */
void log_write_locked(const char * text) {
    for (const char * p = text; *p != '\0';) {
        const char * nl = strchr(p, '\n');
        const size_t n  = nl != nullptr ? (size_t) (nl - p) + 1 : strlen(p);
        if (g_log_line_start) {
            const int64_t t = now_us() - g_t0_us;
            fprintf(stderr, "%d.%02d.%03d.%03d %c ", (int) (t / 60000000), (int) (t / 1000000 % 60),
                    (int) (t / 1000 % 1000), (int) (t % 1000), g_log_level);
        }
        fwrite(p, 1, n, stderr);
        g_log_line_start = nl != nullptr;
        p += n;
    }
}

/**
 * The log callback of --log-ts. Each line on stderr starts with the time since the start of the tool
 * in the form of common/log.cpp, "M.SS.mmm.uuu L ". A text of the level CONT continues the line before
 * it. The lock keeps the lines of two threads apart.
 */
void log_ts_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (level != GGML_LOG_LEVEL_CONT) {
        g_log_level = level == GGML_LOG_LEVEL_ERROR ? 'E'
                    : level == GGML_LOG_LEVEL_WARN  ? 'W'
                    : level == GGML_LOG_LEVEL_DEBUG ? 'D'
                                                    : 'I';
    }
    log_write_locked(text);
}

/**
 * With --log-ts, write one line "memprobe: STAMP <text>" on stderr on the clock of the log lines, thus
 * a script finds the lines of llama.cpp and of the Hexagon backend between two stamps. A line that the
 * log did not end gets its end first. Without --log-ts it writes nothing.
 */
void stamp(const char * fmt, ...) {
    if (!g_log_ts) {
        return;
    }
    char    buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (!g_log_line_start) {
        log_write_locked("\n");
    }
    g_log_level = 'I';
    log_write_locked("memprobe: STAMP ");
    log_write_locked(buf);
    log_write_locked("\n");
}

/**
 * The highest temperature of the thermal zones of the NPU (type nsp*) in millidegrees C, or -1 when the
 * phone gives no readable zone of that type. The shell user can read these files. O(thermal zones).
 */
long nsp_millideg() {
    long  best = -1;
    DIR * dir  = opendir("/sys/class/thermal");
    if (dir == nullptr) {
        return best;
    }
    while (const dirent * ent = readdir(dir)) {
        if (strncmp(ent->d_name, "thermal_zone", 12) != 0) {
            continue;
        }
        const std::string base = std::string("/sys/class/thermal/") + ent->d_name;
        std::ifstream     type_file(base + "/type");
        std::string       type;
        if (!(type_file >> type) || type.compare(0, 3, "nsp") != 0) {
            continue;
        }
        std::ifstream temp_file(base + "/temp");
        long          value = 0;
        if (temp_file >> value) {
            best = std::max(best, value);
        }
    }
    closedir(dir);
    return best;
}

/** Print the line "THERM <label> nsp=<millidegrees C>". */
void print_therm(const std::string & label) {
    printf("THERM %s nsp=%ld\n", label.c_str(), nsp_millideg());
    fflush(stdout);
}

/** The values of a comma-separated list of integers from min_value to 1000000. Exits with a message on a bad list. O(length). */
std::vector<int> parse_int_list(const std::string & text, const char * option, int min_value) {
    std::vector<int> out;
    size_t           pos = 0;
    while (true) {
        const size_t      comma = text.find(',', pos);
        const std::string item  = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        char *            end   = nullptr;
        const long        v     = std::strtol(item.c_str(), &end, 10);
        if (item.empty() || end == nullptr || *end != '\0' || v < min_value || v > 1000000) {
            fprintf(stderr, "%s takes a comma-separated list of integers from %d to 1000000, not \"%s\"\n", option,
                    min_value, text.c_str());
            std::exit(2);
        }
        out.push_back((int) v);
        if (comma == std::string::npos) {
            return out;
        }
        pos = comma + 1;
    }
}

// ---- The engine of the app for the modes --cold, --sweep and --turns ----

/**
 * The parts of the engine of llama_jni.cpp that a prompt and an answer use: the decode context, the
 * MTP draft context and its driver (setup_speculative), and the batch of the text decodes.
 */
struct AppEngine {
    llama_model *        model   = nullptr;
    llama_context *      ctx     = nullptr;
    llama_context *      ctx_dft = nullptr;
    common_speculative * spec    = nullptr;  // null without --spec, or after a failure of the draft context
    llama_batch *        batch   = nullptr;
    int                  n_batch = 0;
};

/** The wall times of the parts of one prompt decode or of the steps of one answer, in milliseconds. */
struct DecodeTimes {
    double decode = 0.0;  // the llama_decode calls of the target context
    double follow = 0.0;  // the follows of the draft context (common_speculative_process)
    double draft  = 0.0;  // the drafts of the MTP block (common_speculative_draft)
    double sync   = 0.0;  // llama_synchronize
};

/** Stop the drafts after a failure of the draft context, as spec_disable in llama_jni.cpp does. */
void app_spec_disable(AppEngine & e) {
    if (e.spec != nullptr) {
        common_speculative_free(e.spec);
        e.spec = nullptr;
    }
}

/**
 * Let the draft context follow a batch that the target context decoded (spec_follow in llama_jni.cpp)
 * and add the wall time to follow_ms. Returns false when the draft context failed. The caller stops
 * the drafts then, as the app does.
 */
bool app_follow(AppEngine & e, const llama_batch & b, double & follow_ms) {
    if (e.spec == nullptr) {
        return true;
    }
    stamp("follow-begin tokens=%d", b.n_tokens);
    const int64_t t0 = now_us();
    const bool    ok = common_speculative_process(e.spec, b);
    follow_ms += (now_us() - t0) / 1000.0;
    stamp("follow-end ok=%d", ok ? 1 : 0);
    return ok;
}

/**
 * Decode text tokens into sequence 0 from pos0 in chunks of n_batch, as decode_text in llama_jni.cpp:
 * only the last token gives logits, and only with logits_last, and the draft context follows each chunk.
 * A failed follow stops the drafts. Adds the wall times to t. Returns the llama_decode code, 0 on
 * success. O(n) tokens.
 */
int app_decode(AppEngine & e, const llama_token * tokens, int n, llama_pos pos0, bool logits_last, DecodeTimes & t) {
    llama_batch & b = *e.batch;
    for (int i = 0; i < n; i += e.n_batch) {
        const int count = std::min(e.n_batch, n - i);
        for (int j = 0; j < count; ++j) {
            b.token[j]     = tokens[i + j];
            b.pos[j]       = pos0 + i + j;
            b.n_seq_id[j]  = 1;
            b.seq_id[j][0] = 0;
            b.logits[j]    = 0;
        }
        if (logits_last && i + count == n) {
            b.logits[count - 1] = 1;
        }
        b.n_tokens = count;
        stamp("decode-begin tokens=%d pos=%d", count, (int) (pos0 + i));
        const int64_t t0 = now_us();
        const int     rc = llama_decode(e.ctx, b);
        t.decode += (now_us() - t0) / 1000.0;
        stamp("decode-end rc=%d", rc);
        if (rc != 0) {
            return rc;
        }
        if (!app_follow(e, b, t.follow)) {
            stamp("follow-failed tokens=%d", count);
            app_spec_disable(e);
        }
    }
    return 0;
}

/** Wait for the device work of the target context and add the wait to t.sync. The draft context shares the device, thus its work ends too. */
void app_sync(AppEngine & e, DecodeTimes & t) {
    const int64_t t0 = now_us();
    llama_synchronize(e.ctx);
    t.sync += (now_us() - t0) / 1000.0;
    stamp("sync-end");
}

/** Put a state into sequence 0 in the place of what it holds (restore_state in llama_jni.cpp). Returns false when the bytes do not fit, and then the memory is empty. */
bool app_restore(llama_context * ctx, const uint8_t * data, size_t size) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);
    if (llama_state_seq_set_data(ctx, data, size, 0) == 0) {
        llama_memory_clear(mem, true);
        return false;
    }
    return true;
}

/** The state of sequence 0 as bytes (take_state in llama_jni.cpp), or null. The allocation is part of the time, as in the app. */
std::shared_ptr<const cache_io::Blob> app_take_state(llama_context * ctx) {
    const size_t size = llama_state_seq_get_size(ctx, 0);
    auto         blob = std::make_shared<cache_io::Blob>(size);
    const size_t got  = llama_state_seq_get_data(ctx, blob->data.get(), size, 0);
    if (got == 0 || got > size) {
        return nullptr;
    }
    blob->size = got;
    return blob;
}

/** Empty the memory of the draft context without its data, as the app does at the start of each prompt. */
void app_clear_draft(AppEngine & e) {
    if (e.ctx_dft != nullptr) {
        llama_memory_clear(llama_get_memory(e.ctx_dft), false);
    }
}

/**
 * The first prompt decode of the process (--cold): the first n tokens of the pool at depth 0 with the
 * logits of the last one, the follow of the draft context and the synchronize. Its time holds the costs
 * of a first decode after the load: the first use of each DSP buffer, the first graph and the first
 * pages of the token embedding. The memory is empty after it. Prints "TIME cold".
 */
void run_cold(AppEngine & e, const std::vector<llama_token> & pool, int n) {
    DecodeTimes t;
    stamp("call-begin mode=cold depth=0 tokens=%d call=0", n);
    const int64_t t0 = now_us();
    const int     rc = app_decode(e, pool.data(), n, 0, true, t);
    app_sync(e, t);
    const double ms = (now_us() - t0) / 1000.0;
    stamp("call-end");
    printf("TIME cold tokens=%d ms=%.1f decode_ms=%.1f follow_ms=%.1f sync_ms=%.1f rc=%d\n", n, ms, t.decode, t.follow,
           t.sync, rc);
    fflush(stdout);
    llama_memory_seq_rm(llama_get_memory(e.ctx), 0, -1, -1);
    app_clear_draft(e);
}

/**
 * The prefill time against the prompt length (--sweep). For each depth, the memory gets that depth one
 * time (a decode of random tokens) and keeps its state as a blob. Then each size decodes the first
 * tokens of the pool `calls` times, each time after a restore of the blob (an empty sequence at depth
 * 0), as the app restores a snapshot before a prompt. The restore is outside the time of the call and
 * has its own field. The time of a call is the target decodes, the follows of the draft context and the
 * synchronize. Prints "TIME sweep-fill" and "TIME sweep". O(depths x sizes x calls) decodes.
 */
void run_sweep(AppEngine & e, const std::vector<llama_token> & pool, const std::vector<int> & depths,
               const std::vector<int> & sizes, int calls, bool logits_last, bool therm, std::mt19937 & rng, int n_vocab) {
    llama_memory_t                     mem = llama_get_memory(e.ctx);
    std::uniform_int_distribution<int> pick(0, n_vocab - 1);
    for (const int depth : depths) {
        std::vector<uint8_t> state;
        llama_memory_clear(mem, true);
        app_clear_draft(e);
        if (depth > 0) {
            std::vector<llama_token> fill((size_t) depth);
            for (auto & tok : fill) {
                tok = pick(rng);
            }
            DecodeTimes tf;
            stamp("fill-begin depth=%d", depth);
            const int64_t t0 = now_us();
            const int     rc = app_decode(e, fill.data(), depth, 0, false, tf);
            app_sync(e, tf);
            const double ms = (now_us() - t0) / 1000.0;
            state.resize(llama_state_seq_get_size(e.ctx, 0));
            const size_t got = rc == 0 ? llama_state_seq_get_data(e.ctx, state.data(), state.size(), 0) : 0;
            state.resize(got);
            stamp("fill-end depth=%d", depth);
            printf("TIME sweep-fill depth=%d ms=%.1f bytes=%zu rc=%d\n", depth, ms, got, rc);
            fflush(stdout);
            if (rc != 0 || got == 0) {
                fprintf(stderr, "memprobe: the state of depth %d did not form (rc %d, %zu bytes)\n", depth, rc, got);
                return;
            }
        }
        for (const int n : sizes) {
            if (therm) {
                print_therm("sweep-d" + std::to_string(depth) + "-n" + std::to_string(n));
            }
            for (int c = 0; c < calls; ++c) {
                const int64_t tr = now_us();
                bool          ok = true;
                if (depth > 0) {
                    ok = app_restore(e.ctx, state.data(), state.size());
                } else {
                    llama_memory_seq_rm(mem, 0, -1, -1);
                }
                app_clear_draft(e);
                const double restore_ms = (now_us() - tr) / 1000.0;
                if (!ok) {
                    fprintf(stderr, "memprobe: the state of depth %d did not restore\n", depth);
                    return;
                }
                DecodeTimes t;
                stamp("call-begin mode=sweep depth=%d tokens=%d call=%d", depth, n, c);
                const int64_t t0 = now_us();
                const int     rc = app_decode(e, pool.data(), n, depth, logits_last, t);
                app_sync(e, t);
                const double ms = (now_us() - t0) / 1000.0;
                stamp("call-end");
                printf("TIME sweep depth=%d tokens=%d call=%d ms=%.1f decode_ms=%.1f follow_ms=%.1f sync_ms=%.1f "
                       "restore_ms=%.1f draft=%d rc=%d\n",
                       depth, n, c, ms, t.decode, t.follow, t.sync, restore_ms, e.spec != nullptr ? 1 : 0, rc);
                fflush(stdout);
                if (rc != 0) {
                    return;
                }
            }
        }
    }
    llama_memory_clear(mem, true);
    app_clear_draft(e);
}

/** The options of the turn mode (--turns). */
struct TurnOptions {
    int         turns   = 0;    // the number of chat turns
    int         first   = 500;  // the tokens of the first message
    int         message = 40;   // the tokens of each later message
    int         answer  = 32;   // the most tokens of an answer
    int         idle_ms = 0;    // the wait before each turn, outside the times
    bool        therm   = false;
    std::string state_dir;      // the disk tier of the snapshot store, or empty for none
};

/** The words of the filler text of the chat messages. */
const char * const kFillerWords[] = {
    "the",    "river",  "north",  "house",   "light",   "stone",  "garden", "winter", "market", "window",
    "train",  "letter", "bridge", "forest",  "music",   "paper",  "silver", "harbor", "field",  "clock",
    "island", "valley", "school", "doctor",  "summer",  "engine", "number", "yellow", "and",    "of",
    "in",     "with",   "near",   "after",   "before",  "small",  "large",  "old",    "new",    "morning",
};

/**
 * A message of about n_tokens tokens: filler words from the random generator, cut to the tokens that the
 * question at its end leaves, then the question. In the template the tokens at the joints can differ by
 * one or two. O(n_tokens).
 */
std::string message_text(const llama_vocab * vocab, int n_tokens, const std::string & question, std::mt19937 & rng) {
    const int n_question = (int) common_tokenize(vocab, question, false, false).size();
    const int n_filler   = std::max(1, n_tokens - n_question);
    std::uniform_int_distribution<size_t> pick(0, sizeof(kFillerWords) / sizeof(kFillerWords[0]) - 1);
    std::string              text;
    std::vector<llama_token> toks;
    while ((int) toks.size() < n_filler) {
        for (int i = 0; i < n_filler; ++i) {
            if (!text.empty()) {
                text += ' ';
            }
            text += kFillerWords[pick(rng)];
        }
        toks = common_tokenize(vocab, text, false, false);
    }
    toks.resize((size_t) n_filler);
    return common_detokenize(vocab, toks, false) + question;
}

/**
 * The sampler chain of rebuild_sampler in llama_jni.cpp without thinking (the preset of the app):
 * penalties, top-k 20, top-p 0.8, temperature 0.7, dist. The app seeds the dist sampler from the clock
 * (LLAMA_DEFAULT_SEED), this tool with a fixed seed, thus two rounds give the same answers.
 */
llama_sampler * app_sampler(const llama_model * model) {
    auto params    = llama_sampler_chain_default_params();
    params.no_perf = true;
    llama_sampler * s       = llama_sampler_chain_init(params);
    const int32_t   n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_sampler_chain_add(s, llama_sampler_init_penalties(n_vocab, 256, 1.0f, 0.0f, 1.5f));
    llama_sampler_chain_add(s, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(s, llama_sampler_init_top_p(0.8f, 1));
    llama_sampler_chain_add(s, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(s, llama_sampler_init_dist(1234));
    return s;
}

/**
 * The number of items of the prompt without the generation prompt (base_length in llama_jni.cpp). The
 * tokens of the generation prompt must be the tail of the items, or the whole prompt counts as the base.
 */
size_t app_base_length(const llama_vocab * vocab, const std::string & prompt, const std::string & tail,
                       const std::vector<MemItem> & items) {
    if (tail.empty() || prompt.size() < tail.size() || prompt.compare(prompt.size() - tail.size(), tail.size(), tail) != 0) {
        return items.size();
    }
    const std::vector<llama_token> tail_tokens = common_tokenize(vocab, tail, false, true);
    if (tail_tokens.empty() || tail_tokens.size() > items.size()) {
        return items.size();
    }
    const size_t base = items.size() - tail_tokens.size();
    for (size_t i = 0; i < tail_tokens.size(); ++i) {
        if (items[base + i] != MemItem{ tail_tokens[i], {} }) {
            return items.size();
        }
    }
    return base;
}

/** The state of the conversation of the turn mode: the fields of Engine in llama_jni.cpp that a turn uses. */
struct AppChat {
    std::vector<MemItem>     cache;  // the items that sequence 0 of the context holds, in order
    llama_pos                n_past = 0;
    llama_tokens             spec_prompt;
    llama_tokens             draft;
    llama_token              id_last = LLAMA_TOKEN_NULL;
    std::vector<llama_token> out_queue;
    bool                     answer_ends = false;
    llama_sampler *          smpl        = nullptr;
    SpecPolicy               policy;
    int64_t                  sample_us      = 0;
    int64_t                  drafted        = 0;
    int64_t                  accepted       = 0;
    int64_t                  t_first_sample = 0;  // the time of the first sample of the answer, 0 before it
    DecodeTimes              steps;               // the decodes, follows and drafts of the answer

    AppChat() = default;
    AppChat(const AppChat &)             = delete;
    AppChat & operator=(const AppChat &) = delete;

    ~AppChat() {
        if (smpl != nullptr) {
            llama_sampler_free(smpl);
        }
    }
};

/** Record a token that the context holds at the end of its sequence (commit_token). */
void app_commit(AppEngine & e, AppChat & c, llama_token token) {
    c.cache.push_back(MemItem{ token, {} });
    c.n_past += 1;
    if (e.spec != nullptr) {
        c.spec_prompt.push_back(token);
    }
}

/** Decode one token of the answer (decode_one). Returns the llama_decode code. */
int app_decode_one(AppEngine & e, AppChat & c, llama_token token) {
    const int rc = app_decode(e, &token, 1, c.n_past, true, c.steps);
    if (rc == 0) {
        app_commit(e, c, token);
    }
    return rc;
}

/** Sample the token of the logits row idx, and keep the time of the first sample of the answer. */
llama_token app_sample(AppChat & c, llama_context * ctx, int32_t idx) {
    const int64_t     t0 = now_us();
    const llama_token id = llama_sampler_sample(c.smpl, ctx, idx);
    const int64_t     t1 = now_us();
    c.sample_us += t1 - t0;
    if (c.t_first_sample == 0) {
        c.t_first_sample = t1;
    }
    return id;
}

/** One step of the answer without a draft (plain_step in llama_jni.cpp). Returns false on a decode error. */
bool app_plain_step(AppEngine & e, AppChat & c) {
    const llama_vocab * vocab = llama_model_get_vocab(e.model);
    // The drafts stopped in a step that gave its last token to the caller: that token decodes first.
    if (c.id_last != LLAMA_TOKEN_NULL) {
        const llama_token pending = c.id_last;
        c.id_last                 = LLAMA_TOKEN_NULL;
        if (app_decode_one(e, c, pending) != 0) {
            return false;
        }
    }
    const llama_token token = app_sample(c, e.ctx, -1);
    if (llama_vocab_is_eog(vocab, token)) {
        app_decode_one(e, c, token);
        c.answer_ends = true;
        return true;
    }
    if (app_decode_one(e, c, token) != 0) {
        return false;
    }
    c.out_queue.push_back(token);
    c.policy.record(0, 0, 1);
    return true;
}

/**
 * One step of the answer with a draft (spec_step in llama_jni.cpp): the MTP block drafts the length that
 * the policy selects, one decode verifies the draft, the sampler accepts the drafted tokens that it
 * selects itself, and the rejected positions roll back. Returns false on a decode or rollback error.
 */
bool app_spec_step(AppEngine & e, AppChat & c) {
    const llama_vocab * vocab   = llama_model_get_vocab(e.model);
    llama_memory_t      mem     = llama_get_memory(e.ctx);
    llama_memory_t      mem_dft = llama_get_memory(e.ctx_dft);

    if (c.id_last == LLAMA_TOKEN_NULL) {
        const llama_token first = app_sample(c, e.ctx, -1);
        if (llama_vocab_is_eog(vocab, first)) {
            app_decode_one(e, c, first);
            c.answer_ends = true;
            return true;
        }
        c.out_queue.push_back(first);
        c.id_last = first;
    }

    const llama_pos pos0    = c.n_past;
    const int       room    = (int) llama_n_ctx(e.ctx) - (int) pos0 - 2;
    const int       n_draft = std::min(c.policy.next_draft(), std::max(0, room));
    c.draft.clear();
    if (n_draft > 0) {
        stamp("draft-begin n=%d", n_draft);
        const int64_t t0 = now_us();
        common_speculative_get_draft_params(e.spec, 0) = {
            /* .drafting = */ true,
            /* .n_max    = */ n_draft,
            /* .pos0     = */ pos0,
            /* .id_last  = */ c.id_last,
            /* .prompt   = */ &c.spec_prompt,
            /* .result   = */ &c.draft,
        };
        common_speculative_draft(e.spec);
        llama_memory_seq_rm(mem_dft, 0, pos0, -1);
        c.steps.draft += (now_us() - t0) / 1000.0;
        stamp("draft-end got=%zu", c.draft.size());
    }

    llama_batch & b = *e.batch;
    b.n_tokens      = 1 + (int) c.draft.size();
    for (int i = 0; i < b.n_tokens; ++i) {
        b.token[i]     = i == 0 ? c.id_last : c.draft[(size_t) i - 1];
        b.pos[i]       = pos0 + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 1;
    }
    stamp("decode-begin tokens=%d pos=%d", b.n_tokens, (int) pos0);
    const int64_t t0 = now_us();
    const int     rc = llama_decode(e.ctx, b);
    c.steps.decode += (now_us() - t0) / 1000.0;
    stamp("decode-end rc=%d", rc);
    if (rc != 0) {
        return false;
    }
    const bool followed = app_follow(e, b, c.steps.follow);

    // sample_and_accept
    std::vector<llama_token> ids;
    size_t                   i = 0;
    for (; i < c.draft.size(); ++i) {
        const llama_token id = app_sample(c, e.ctx, (int32_t) i);
        ids.push_back(id);
        if (c.draft[i] != id) {
            break;
        }
    }
    if (i == c.draft.size()) {
        ids.push_back(app_sample(c, e.ctx, (int32_t) i));
    }
    const int accepted = (int) ids.size() - 1;
    common_speculative_accept(e.spec, 0, (uint16_t) accepted);
    c.drafted += (int64_t) c.draft.size();
    c.accepted += accepted;
    c.policy.record((int) c.draft.size(), accepted, (int) ids.size());

    app_commit(e, c, c.id_last);
    for (int k = 0; k < accepted; ++k) {
        app_commit(e, c, ids[(size_t) k]);
    }
    int eog_at = -1;
    for (int k = 0; k < (int) ids.size(); ++k) {
        if (llama_vocab_is_eog(vocab, ids[(size_t) k])) {
            eog_at = k;
            break;
        }
    }
    const int n_emit = eog_at >= 0 ? eog_at : (int) ids.size();
    for (int k = 0; k < n_emit; ++k) {
        c.out_queue.push_back(ids[(size_t) k]);
    }
    if (eog_at < 0) {
        c.id_last = ids.back();
    } else {
        c.answer_ends = true;
        c.id_last     = LLAMA_TOKEN_NULL;
        if (eog_at < accepted) {
            const size_t drop = (size_t) (accepted - eog_at - 1);
            c.cache.resize(c.cache.size() - drop);
            c.spec_prompt.resize(c.spec_prompt.size() - drop);
            c.n_past -= (llama_pos) drop;
        }
    }
    if (c.n_past <= pos0 + (llama_pos) c.draft.size()) {
        if (!llama_memory_seq_rm(mem, 0, c.n_past, -1)) {
            fprintf(stderr, "memprobe: the rejected draft did not roll back at position %d\n", (int) c.n_past);
            return false;
        }
    }
    if (followed) {
        llama_memory_seq_rm(mem_dft, 0, c.n_past, -1);
    } else {
        stamp("follow-failed tokens=%d", b.n_tokens);
        app_spec_disable(e);
    }
    if (eog_at >= 0 && eog_at == accepted) {
        if (app_decode_one(e, c, ids.back()) != 0) {
            return false;
        }
    }
    return true;
}

/** The questions at the end of the first message and of each later message. */
const char * const kFirstQuestion = " Summarize the text above in one sentence.";
const char * const kQuestion      = " Answer in one short sentence.";

/**
 * A chat of o.turns turns in the engine of the app (--turns). Each turn follows chat_start_impl, prefill
 * and generate_next_impl in llama_jni.cpp and prints one "TIME turn" line with the time of each part.
 * ttft_ms is the time from the start of the turn to the end of the first step, when the app gives the
 * first token. first_sample_ms is the time to the first sample of the answer, when the token is known.
 * tokenize_ms holds the new sampler chain, the tokens of the whole conversation and the base length, in
 * the order of chat_start_impl.
 * An answer that reaches o.answer tokens stops there, and its text still goes into the history. At the
 * end it prints "TIME turn-store". O(turns x (prompt + answer)) decodes.
 */
void run_turns(AppEngine & e, const TurnOptions & o, std::mt19937 & rng) {
    const llama_vocab *       vocab = llama_model_get_vocab(e.model);
    common_chat_templates_ptr tmpls;
    try {
        tmpls = common_chat_templates_init(e.model, "");
    } catch (const std::exception & ex) {
        fprintf(stderr, "memprobe: the chat template of the model did not parse: %s\n", ex.what());
        return;
    }
    StateCache                   states(256u << 20, o.state_dir.empty() ? 0 : 256u << 20, o.state_dir);
    AppChat                      c;
    std::vector<common_chat_msg> msgs;
    llama_memory_t               mem = llama_get_memory(e.ctx);

    for (int k = 1; k <= o.turns; ++k) {
        if (o.idle_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(o.idle_ms));
        }
        if (o.therm) {
            print_therm("turn-" + std::to_string(k));
        }
        common_chat_msg user;
        user.role    = "user";
        user.content = message_text(vocab, k == 1 ? o.first : o.message, k == 1 ? kFirstQuestion : kQuestion, rng);
        msgs.push_back(user);

        stamp("turn-begin k=%d", k);
        const int64_t t_start = now_us();

        // chat_start_impl: the prompt, and the generation prompt at its end.
        common_chat_templates_inputs inputs;
        inputs.messages              = msgs;
        inputs.add_generation_prompt = true;
        inputs.use_jinja             = true;
        inputs.enable_thinking       = false;
        std::string prompt;
        std::string tail;
        try {
            common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);
            prompt                    = std::move(params.prompt);
            tail                      = std::move(params.generation_prompt);
            if (tail.empty() || prompt.size() < tail.size() ||
                prompt.compare(prompt.size() - tail.size(), tail.size(), tail) != 0) {
                inputs.add_generation_prompt = false;
                const std::string base       = common_chat_templates_apply(tmpls.get(), inputs).prompt;
                size_t            m          = 0;
                while (m < base.size() && m < prompt.size() && base[m] == prompt[m]) {
                    ++m;
                }
                tail = prompt.substr(m);
            }
        } catch (const std::exception & ex) {
            fprintf(stderr, "memprobe: the chat template failed: %s\n", ex.what());
            return;
        }
        const int64_t t_template = now_us();
        if (c.smpl != nullptr) {
            llama_sampler_free(c.smpl);
        }
        c.smpl        = app_sampler(e.model);
        c.answer_ends = false;
        c.id_last     = LLAMA_TOKEN_NULL;
        c.draft.clear();
        c.out_queue.clear();
        c.policy.reset();
        c.sample_us      = 0;
        c.drafted        = 0;
        c.accepted       = 0;
        c.t_first_sample = 0;
        c.steps          = DecodeTimes();

        const std::vector<llama_token> toks = common_tokenize(vocab, prompt, true, true);
        std::vector<MemItem>           items;
        items.reserve(toks.size());
        for (const llama_token t : toks) {
            items.push_back(MemItem{ t, {} });
        }
        size_t base_len = app_base_length(vocab, prompt, tail, items);
        if (base_len == items.size()) {
            base_len -= 1;
        }
        const int64_t t_tokenize = now_us();
        if (items.size() + 8 >= llama_n_ctx(e.ctx)) {
            fprintf(stderr, "memprobe: turn %d has %zu tokens, the context holds %u\n", k, items.size(), llama_n_ctx(e.ctx));
            return;
        }

        // prefill: the longest prefix from the live memory or from the snapshot store.
        app_clear_draft(e);
        const size_t limit = std::min(base_len, items.size() - 1);
        size_t       start = 0;
        llama_pos    pos   = 0;
        const char * reuse = "none";
        stamp("restore-begin");
        const int64_t    t_r0    = now_us();
        const Snapshot * snap    = states.best_prefix(items, limit);
        const bool       live_ok = !c.cache.empty() && c.cache.size() <= limit && is_item_prefix(c.cache, items);
        if (live_ok && (snap == nullptr || c.cache.size() >= snap->items.size())) {
            start = c.cache.size();
            pos   = c.n_past;
            reuse = "live";
        } else if (snap != nullptr) {
            const size_t                          n_items = snap->items.size();
            const llama_pos                       n_pos   = snap->n_pos;
            std::shared_ptr<const cache_io::Blob> bytes   = states.bytes(snap);
            if (bytes && app_restore(e.ctx, bytes->data.get(), bytes->size)) {
                start = n_items;
                pos   = n_pos;
                reuse = "snapshot";
            } else if (bytes) {
                states.drop(snap);
            }
        }
        const int64_t t_r1 = now_us();
        stamp("restore-end reuse=%s", reuse);
        if (start == 0) {
            llama_memory_clear(mem, true);
            pos = 0;
        }
        const int64_t t_clear = now_us();
        c.cache.assign(items.begin(), items.begin() + (ptrdiff_t) start);
        c.n_past = pos;

        // The prompt without the generation prompt: no logits.
        std::vector<llama_token> base_toks;
        for (size_t i = start; i < base_len; ++i) {
            base_toks.push_back(items[i].token);
        }
        DecodeTimes tb;
        stamp("base-begin tokens=%zu pos=%d", base_toks.size(), (int) pos);
        const int64_t t_b0 = now_us();
        int rc = base_toks.empty() ? 0 : app_decode(e, base_toks.data(), (int) base_toks.size(), pos, false, tb);
        if (rc == 0) {
            app_sync(e, tb);
        }
        const int64_t t_b1 = now_us();
        stamp("base-end rc=%d", rc);
        if (rc != 0) {
            fprintf(stderr, "memprobe: the prompt of turn %d did not decode (rc %d)\n", k, rc);
            return;
        }
        pos += (llama_pos) base_toks.size();
        c.cache.assign(items.begin(), items.begin() + (ptrdiff_t) base_len);
        c.n_past = pos;

        // The snapshot of the state before the generation prompt, when the store does not hold it.
        double snap_ms = 0.0, snap_get_ms = 0.0;
        size_t snap_bytes = 0;
        {
            std::vector<MemItem> key(items.begin(), items.begin() + (ptrdiff_t) base_len);
            if (base_len > 0 && states.find(key) == nullptr) {
                stamp("snapshot-begin");
                const int64_t                         t0   = now_us();
                std::shared_ptr<const cache_io::Blob> blob = app_take_state(e.ctx);
                const int64_t                         t1   = now_us();
                if (!blob) {
                    fprintf(stderr, "memprobe: the state of turn %d did not copy out\n", k);
                    return;
                }
                snap_bytes = blob->size;
                states.put(std::move(key), pos, blob);
                snap_get_ms = (t1 - t0) / 1000.0;
                snap_ms     = (now_us() - t0) / 1000.0;
                stamp("snapshot-end bytes=%zu", snap_bytes);
            }
        }

        // The generation prompt: the logits of its last token.
        std::vector<llama_token> tail_toks;
        for (size_t i = base_len; i < items.size(); ++i) {
            tail_toks.push_back(items[i].token);
        }
        DecodeTimes tt;
        stamp("tail-begin tokens=%zu pos=%d", tail_toks.size(), (int) pos);
        const int64_t t_t0 = now_us();
        rc                 = app_decode(e, tail_toks.data(), (int) tail_toks.size(), pos, true, tt);
        if (rc == 0) {
            app_sync(e, tt);
        }
        const int64_t t_t1 = now_us();
        stamp("tail-end rc=%d", rc);
        if (rc != 0) {
            fprintf(stderr, "memprobe: the generation prompt of turn %d did not decode (rc %d)\n", k, rc);
            return;
        }
        pos += (llama_pos) tail_toks.size();
        c.cache  = items;
        c.n_past = pos;
        if (e.spec != nullptr) {
            c.spec_prompt.clear();
            for (const MemItem & item : items) {
                if (item.token != kMemTokenNull) {
                    c.spec_prompt.push_back(item.token);
                }
            }
            common_speculative_begin(e.spec, 0, c.spec_prompt);
        }
        const int64_t t_ready = now_us();
        stamp("prompt-end");

        // generate_next_impl: one step for each call that finds the queue empty.
        std::vector<llama_token> answer;
        int                      steps         = 0;
        int64_t                  gen_us        = 0;
        int64_t                  first_step_us = 0;
        int64_t                  t_first_token = 0;
        bool                     failed        = false;
        while (!c.answer_ends && (int) answer.size() < o.answer && (uint32_t) c.n_past < llama_n_ctx(e.ctx)) {
            const bool with_draft = e.spec != nullptr;
            stamp("step-begin i=%d draft=%d", steps, with_draft ? 1 : 0);
            const int64_t t0 = now_us();
            const bool    ok = with_draft ? app_spec_step(e, c) : app_plain_step(e, c);
            const int64_t t1 = now_us();
            stamp("step-end i=%d", steps);
            if (!ok) {
                failed = true;
                break;
            }
            c.policy.observe(t1 - t0);
            gen_us += t1 - t0;
            if (steps == 0) {
                first_step_us = t1 - t0;
            }
            steps += 1;
            if (!c.out_queue.empty() && t_first_token == 0) {
                t_first_token = t1;
            }
            answer.insert(answer.end(), c.out_queue.begin(), c.out_queue.end());
            c.out_queue.clear();
        }
        stamp("turn-end k=%d", k);
        printf("TIME turn k=%d draft=%d items=%zu reuse=%s start=%zu depth=%d template_ms=%.1f tokenize_ms=%.1f "
               "restore_ms=%.1f clear_ms=%.1f base_tokens=%zu base_ms=%.1f base_decode_ms=%.1f base_follow_ms=%.1f "
               "base_sync_ms=%.1f snapshot_ms=%.1f snapshot_get_ms=%.1f snapshot_mib=%.2f tail_tokens=%zu tail_ms=%.1f "
               "tail_decode_ms=%.1f tail_follow_ms=%.1f tail_sync_ms=%.1f prompt_ms=%.1f first_sample_ms=%.1f "
               "first_step_ms=%.1f ttft_ms=%.1f gen_tokens=%zu gen_ms=%.1f steps=%d step_decode_ms=%.1f "
               "step_follow_ms=%.1f step_draft_ms=%.1f sample_ms=%.1f drafted=%lld accepted=%lld eog=%d failed=%d\n",
               k, e.spec != nullptr ? 1 : 0, items.size(), reuse, start, (int) c.n_past, (t_template - t_start) / 1000.0,
               (t_tokenize - t_template) / 1000.0, (t_r1 - t_r0) / 1000.0, (t_clear - t_r1) / 1000.0, base_toks.size(),
               (t_b1 - t_b0) / 1000.0, tb.decode, tb.follow, tb.sync, snap_ms, snap_get_ms, snap_bytes / 1048576.0,
               tail_toks.size(), (t_t1 - t_t0) / 1000.0, tt.decode, tt.follow, tt.sync, (t_ready - t_start) / 1000.0,
               c.t_first_sample > 0 ? (c.t_first_sample - t_start) / 1000.0 : -1.0, first_step_us / 1000.0,
               t_first_token > 0 ? (t_first_token - t_start) / 1000.0 : -1.0, answer.size(), gen_us / 1000.0, steps,
               c.steps.decode, c.steps.follow, c.steps.draft, c.sample_us / 1000.0, (long long) c.drafted,
               (long long) c.accepted, c.answer_ends ? 1 : 0, failed ? 1 : 0);
        fflush(stdout);
        if (failed) {
            return;
        }
        common_chat_msg reply;
        reply.role    = "assistant";
        reply.content = common_detokenize(vocab, answer, false);
        msgs.push_back(reply);
    }
    printf("TIME turn-store snapshots=%zu ram_mib=%.1f disk_mib=%.1f\n", states.count(), states.ram_bytes() / 1048576.0,
           states.disk_bytes() / 1048576.0);
    fflush(stdout);
}

} // namespace

int main(int argc, char ** argv) {
    g_t0_us = now_us();
    std::string model_path, dev_name = "HTP0", state_file, fa = "auto";
    int n_ctx = 8192, n_batch = 1024, n_prompt = 0, n_gen = 0, n_threads = 4, n_outputs_max = 0, reps = 1;
    ggml_type tk = GGML_TYPE_F16, tv = GGML_TYPE_F16;
    bool spec = false, smaps = false, vision_warmup = false, grow = false, use_mmap = false, hash = false;
    bool drop_cache = false, no_mmap = false;
    std::string mmproj, vision_dev = "HTP0", embd_advise = "none", lazy = "auto";
    int image_w = 0, image_h = 0, image_tokens = 576;
    // The options of --reps, of the log and of the app modes.
    bool log_ts = false, therm = false;
    int rest_ms = 0, cold = 0, sweep_calls = 4;
    std::string sweep_sizes, sweep_depths = "0", sweep_logits = "last";
    TurnOptions turn_opts;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value of %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m") model_path = next();
        else if (a == "-dev") dev_name = next();
        else if (a == "-c") n_ctx = std::atoi(next().c_str());
        else if (a == "-b") n_batch = std::atoi(next().c_str());
        else if (a == "-ctk") tk = parse_type(next());
        else if (a == "-ctv") tv = parse_type(next());
        else if (a == "-fa") fa = next();
        else if (a == "--spec") spec = true;
        else if (a == "-p") n_prompt = std::atoi(next().c_str());
        else if (a == "-n") n_gen = std::atoi(next().c_str());
        else if (a == "-t") n_threads = std::atoi(next().c_str());
        else if (a == "--state-file") state_file = next();
        else if (a == "--smaps") smaps = true;
        else if (a == "--outputs-max") n_outputs_max = std::atoi(next().c_str());
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--image") { const std::string v = next(); image_w = std::atoi(v.c_str()); image_h = std::atoi(v.c_str() + v.find('x') + 1); }
        else if (a == "--image-tokens") image_tokens = std::atoi(next().c_str());
        else if (a == "--vision-dev") vision_dev = next();
        else if (a == "--vision-warmup") vision_warmup = true;
        else if (a == "--grow") grow = true;
        else if (a == "--mmap") use_mmap = true;
        else if (a == "--no-mmap") no_mmap = true;
        else if (a == "--hash") hash = true;
        else if (a == "--drop-cache") drop_cache = true;
        else if (a == "--embd-advise") embd_advise = next();
        else if (a == "--lazy") lazy = next();
        else if (a == "--reps") reps = std::atoi(next().c_str());
        else if (a == "--rest-ms") rest_ms = std::atoi(next().c_str());
        else if (a == "--log-ts") log_ts = true;
        else if (a == "--therm") { therm = true; turn_opts.therm = true; }
        else if (a == "--cold") cold = std::atoi(next().c_str());
        else if (a == "--sweep") sweep_sizes = next();
        else if (a == "--sweep-depths") sweep_depths = next();
        else if (a == "--sweep-calls") sweep_calls = std::atoi(next().c_str());
        else if (a == "--sweep-logits") sweep_logits = next();
        else if (a == "--turns") turn_opts.turns = std::atoi(next().c_str());
        else if (a == "--turn-first") turn_opts.first = std::atoi(next().c_str());
        else if (a == "--turn-message") turn_opts.message = std::atoi(next().c_str());
        else if (a == "--turn-answer") turn_opts.answer = std::atoi(next().c_str());
        else if (a == "--turn-idle-ms") turn_opts.idle_ms = std::atoi(next().c_str());
        else if (a == "--state-dir") turn_opts.state_dir = next();
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (reps < 1) {
        fprintf(stderr, "--reps takes a count of 1 or more\n");
        return 2;
    }
    if (rest_ms < 0 || cold < 0 || sweep_calls < 1 || turn_opts.turns < 0 || turn_opts.first < 1 ||
        turn_opts.message < 1 || turn_opts.answer < 1 || turn_opts.idle_ms < 0) {
        fprintf(stderr, "--rest-ms, --cold, --turns and --turn-idle-ms take 0 or more, --sweep-calls, --turn-first, "
                        "--turn-message and --turn-answer take 1 or more\n");
        return 2;
    }
    if (sweep_logits != "last" && sweep_logits != "none") {
        fprintf(stderr, "--sweep-logits takes last or none, not %s\n", sweep_logits.c_str());
        return 2;
    }
    const std::vector<int> sizes  = sweep_sizes.empty() ? std::vector<int>() : parse_int_list(sweep_sizes, "--sweep", 1);
    const std::vector<int> depths = parse_int_list(sweep_depths, "--sweep-depths", 0);
    const bool app_mode = cold > 0 || !sizes.empty() || turn_opts.turns > 0;
    if (app_mode && (n_prompt > 0 || n_gen > 0 || grow || !mmproj.empty())) {
        fprintf(stderr, "--cold, --sweep and --turns do not go with -p, -n, --grow or --mmproj\n");
        return 2;
    }
    if (!sizes.empty() &&
        *std::max_element(depths.begin(), depths.end()) + *std::max_element(sizes.begin(), sizes.end()) + 8 > n_ctx) {
        fprintf(stderr, "the largest depth plus the largest size of --sweep must leave 8 positions of the %d of -c\n", n_ctx);
        return 2;
    }
    log_ts = log_ts || app_mode;
    if (lazy != "off" && lazy != "auto" && lazy != "on") {
        fprintf(stderr, "--lazy takes off, auto or on, not %s\n", lazy.c_str());
        return 2;
    }
    if (embd_advise != "none" && embd_advise != "random" && embd_advise != "willneed" && embd_advise != "touch") {
        fprintf(stderr, "--embd-advise takes none, random, willneed or touch, not %s\n", embd_advise.c_str());
        return 2;
    }
    if (use_mmap && no_mmap) {
        fprintf(stderr, "--mmap and --no-mmap exclude each other\n");
        return 2;
    }
    if (grow && spec) {
        fprintf(stderr, "--grow frees the target context, thus it does not go with --spec\n");
        return 2;
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: memprobe -m MODEL [-dev NAME|none] [-c N] [-b N] [-ctk T] [-ctv T] [-fa on|off|auto] "
                        "[--spec] [-p N] [-n N] [-t N] [--state-file PATH] [--smaps] [--outputs-max N] [--reps N]\n");
        return 2;
    }
    // The environment of the app (init_impl in llama_jni.cpp).
    setenv("GGML_HEXAGON_OPFUSION", "1", 0);
    setenv("GGML_HEXAGON_OPFUSION_STATE", "1", 0);

    if (drop_cache) {
        const long before = proc_kb("/proc/meminfo", "Cached");
        const int fd = open(model_path.c_str(), O_RDONLY);
        const int rc = fd >= 0 ? posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) : -1;
        if (fd >= 0) close(fd);
        printf("DROPCACHE rc=%d Cached=%ld -> %ld\n", rc, before, proc_kb("/proc/meminfo", "Cached"));
    }
    print_memory("start", model_path, smaps);
    if (log_ts) {
        g_log_ts = true;
        llama_log_set(log_ts_callback, nullptr);
    }
    ggml_backend_load_all();
    llama_backend_init();

    std::vector<ggml_backend_dev_t> devices;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = dev_name == "none" ? 0 : 999;
    mp.load_mtp     = spec;
    if (use_mmap) mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    if (no_mmap) mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = lazy == "on" ? LLAMA_LAZY_MODE_ON : lazy == "off" ? LLAMA_LAZY_MODE_OFF : LLAMA_LAZY_MODE_AUTO;
    // The memory during the load, at each tenth of the progress: the page cache that the file
    // fills (Cached) and MemAvailable are what the gate of the app and lmkd see.
    int load_next = 1;
    mp.progress_callback = [](float progress, void * ud) -> bool {
        int & next = *static_cast<int *>(ud);
        if (progress * 10.0f >= (float) next) {
            next = (int) (progress * 10.0f) + 1;
            printf("LOADMEM %.1f MemAvailable=%ld Cached=%ld RssAnon=%ld RssFile=%ld\n", progress,
                   proc_kb("/proc/meminfo", "MemAvailable"), proc_kb("/proc/meminfo", "Cached"),
                   proc_kb("/proc/self/status", "RssAnon"), proc_kb("/proc/self/status", "RssFile"));
            fflush(stdout);
        }
        return true;
    };
    mp.progress_callback_user_data = &load_next;
    if (dev_name != "none") {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
        if (dev == nullptr) { fprintf(stderr, "no device %s\n", dev_name.c_str()); return 1; }
        devices = {dev, nullptr};
        mp.devices = devices.data();
    }
    double t0 = now_ms();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (model == nullptr) { fprintf(stderr, "the model did not load\n"); return 1; }
    printf("TIME model-load %.1f\n", now_ms() - t0);
    if (embd_advise != "none") advise_model_mappings(model_path, embd_advise);
    print_memory("model", model_path, smaps);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) n_ctx;
    cp.n_batch         = (uint32_t) n_batch;
    cp.n_ubatch        = (uint32_t) n_batch;
    cp.n_seq_max       = 1;
    cp.kv_unified      = true;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.type_k          = tk;
    cp.type_v          = tv;
    cp.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                                                                : LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (spec) cp.n_rs_seq = 4;
    cp.n_outputs_max   = (uint32_t) n_outputs_max;
    // The app gives each sequence the whole output limit: a verify batch has a row for each position.
    if (app_mode) cp.n_outputs_max_per_seq = cp.n_outputs_max;
    t0 = now_ms();
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) { fprintf(stderr, "the context did not initialize\n"); return 1; }
    printf("TIME ctx-init %.1f n_ctx=%u n_rs_seq=%u\n", now_ms() - t0, llama_n_ctx(ctx), llama_n_rs_seq(ctx));
    print_memory("ctx", model_path, smaps);
    print_buffers("main", ctx);

    // The MTP draft context of setup_speculative in llama_jni.cpp.
    common_speculative_init_result_ptr spec_init;
    llama_context * ctx_dft = nullptr;
    common_params params;
    if (spec && llama_model_n_layer_nextn(model) > 0) {
        params.model.path                = model_path;
        params.n_ctx                     = (int32_t) llama_n_ctx(ctx);
        params.n_batch                   = n_batch;
        params.n_ubatch                  = n_batch;
        params.n_parallel                = 1;
        params.kv_unified                = true;
        params.no_perf                   = true;
        params.cpuparams.n_threads       = n_threads;
        params.cpuparams_batch.n_threads = n_threads;
        params.speculative.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max   = SpecPolicy::kDraftMax;
        t0 = now_ms();
        common_params params_dft = common_base_params_to_speculative(params);
        spec_init = common_speculative_init_from_params(params_dft, model, ctx);
        ctx_dft = spec_init ? spec_init->context() : nullptr;
        printf("TIME draft-init %.1f ok=%d\n", now_ms() - t0, ctx_dft != nullptr);
        print_memory("draft", model_path, smaps);
        if (ctx_dft) print_buffers("draft", ctx_dft);
    }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> pick(0, n_vocab - 1);

    // The app modes: the thread pool of load_impl (low priority, no polling) on the two contexts, and
    // the draft driver of setup_speculative in llama_jni.cpp.
    ggml_threadpool * app_pool = nullptr;
    AppEngine         app;
    if (app_mode) {
        ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
        tpp.prio                   = GGML_SCHED_PRIO_LOW;
        tpp.poll                   = 0;
        tpp.strict_cpu             = false;
        app_pool                   = ggml_threadpool_new(&tpp);
        if (app_pool == nullptr) { fprintf(stderr, "the thread pool of %d threads did not start\n", n_threads); return 1; }
        llama_attach_threadpool(ctx, app_pool, app_pool);
        app.model   = model;
        app.ctx     = ctx;
        app.ctx_dft = ctx_dft;
        app.batch   = &batch;
        app.n_batch = n_batch;
        if (ctx_dft != nullptr) {
            params.speculative.draft.ctx_tgt = ctx;
            params.speculative.draft.ctx_dft = ctx_dft;
            t0 = now_ms();
            app.spec = common_speculative_init(params.speculative, 1);
            printf("TIME driver-init %.1f ok=%d\n", now_ms() - t0, app.spec != nullptr);
            fflush(stdout);
            if (app.spec == nullptr) { fprintf(stderr, "the speculative driver did not initialize\n"); return 1; }
            llama_attach_threadpool(ctx_dft, app_pool, app_pool);
        }
        // The prompt tokens of --cold and --sweep: each size decodes the first tokens of one pool.
        const int max_size = sizes.empty() ? 0 : *std::max_element(sizes.begin(), sizes.end());
        std::vector<llama_token> pool((size_t) std::max(cold, max_size));
        for (auto & t : pool) t = pick(rng);
        if (cold > 0) run_cold(app, pool, cold);
        if (!sizes.empty()) run_sweep(app, pool, depths, sizes, sweep_calls, sweep_logits == "last", therm, rng, n_vocab);
        if (turn_opts.turns > 0) run_turns(app, turn_opts, rng);
        print_memory("app", model_path, smaps);
    }

    if (n_prompt > 0) {
        std::vector<llama_token> toks(n_prompt);
        for (auto & t : toks) t = pick(rng);
        // Each pass after the first starts from a cleared memory, as each pass of bench_impl does.
        for (int r = 0; r < reps; ++r) {
            if (r > 0) llama_memory_clear(llama_get_memory(ctx), true);
            if (r > 0 && rest_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(rest_ms));
            if (therm) print_therm("pass-" + std::to_string(r) + "-before");
            stamp("pass-begin rep=%d tokens=%d", r, n_prompt);
            t0 = now_ms();
            const int rc = decode_tokens(ctx, batch, toks, 0, n_batch);
            llama_synchronize(ctx);
            const double dt = now_ms() - t0;
            stamp("pass-end rep=%d", r);
            printf("TIME prefill %.1f tokens=%d rate=%.1f rc=%d", dt, n_prompt, n_prompt * 1000.0 / dt, rc);
            if (reps > 1) printf(" rep=%d", r);
            printf("\n");
            fflush(stdout);
            if (therm) print_therm("pass-" + std::to_string(r) + "-after");
            if (rc != 0) return 1;
        }
        if (hash) {
            printf("HASH prefill %016llx\n", (unsigned long long) fnv_bytes(llama_get_logits_ith(ctx, -1), (size_t) n_vocab * sizeof(float)));
        }
        print_memory("prefill", model_path, smaps);
    }
    if (n_gen > 0) {
        std::vector<double> steps;
        t0 = now_ms();
        for (int i = 0; i < n_gen; ++i) {
            stamp("step-begin i=%d draft=0", i);
            const double ts = now_ms();
            std::vector<llama_token> one = { pick(rng) };
            if (decode_tokens(ctx, batch, one, n_prompt + i, n_batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            llama_synchronize(ctx);
            steps.push_back(now_ms() - ts);
            stamp("step-end i=%d", i);
            if (i == 0) {
                printf("TIME first-decode %.1f\n", now_ms() - t0);
            }
            if (hash) {
                printf("HASH decode %d %016llx\n", i, (unsigned long long) fnv_bytes(llama_get_logits_ith(ctx, -1), (size_t) n_vocab * sizeof(float)));
            }
        }
        const double dt = now_ms() - t0;
        printf("TIME decode %.1f tokens=%d depth=%d rate=%.2f\n", dt, n_gen, n_prompt, n_gen * 1000.0 / dt);
        // The time of each step after the first: a page fault of a new embedding row shows in the maximum.
        std::vector<double> rest(steps.begin() + 1, steps.end());
        if (!rest.empty()) {
            std::sort(rest.begin(), rest.end());
            printf("TIME decode-steps min=%.1f median=%.1f max=%.1f n=%zu\n", rest.front(), rest[rest.size() / 2],
                   rest.back(), rest.size());
        }
        print_memory("decode", model_path, smaps);
    }

    // A context that grows: the state moves into a context of twice the length.
    if (grow && n_prompt > 0) {
        const double tg0 = now_ms();
        std::vector<uint8_t> st(llama_state_seq_get_size(ctx, 0));
        const size_t got = llama_state_seq_get_data(ctx, st.data(), st.size(), 0);
        const double tg1 = now_ms();
        llama_free(ctx);
        const double tg2 = now_ms();
        print_memory("grow-freed", model_path, smaps);
        cp.n_ctx = (uint32_t) n_ctx * 2;
        const double tg3 = now_ms();
        ctx = llama_init_from_model(model, cp);
        if (ctx == nullptr) { fprintf(stderr, "the larger context did not initialize\n"); return 1; }
        const double tg4 = now_ms();
        const size_t set = llama_state_seq_set_data(ctx, st.data(), got, 0);
        const double tg5 = now_ms();
        printf("TIME grow get=%.1f free=%.1f init=%.1f set=%.1f total=%.1f bytes=%zu ok=%d n_ctx=%u\n", tg1 - tg0,
               tg2 - tg1, tg4 - tg3, tg5 - tg4, tg5 - tg0, got, set != 0, llama_n_ctx(ctx));
        st.clear();
        st.shrink_to_fit();
        print_memory("grow-ctx", model_path, smaps);
        print_buffers("grown", ctx);
        const double td = now_ms();
        for (int i = 0; i < n_gen; ++i) {
            std::vector<llama_token> one = { pick(rng) };
            if (decode_tokens(ctx, batch, one, n_prompt + n_gen + i, n_batch) != 0) { fprintf(stderr, "decode after the growth failed\n"); return 1; }
            llama_synchronize(ctx);
        }
        if (n_gen > 0) {
            printf("TIME grow-decode %.1f tokens=%d rate=%.2f\n", now_ms() - td, n_gen, n_gen * 1000.0 / (now_ms() - td));
        }
    }

    // The snapshot of the prompt state store: size, copy out, file write, file read, copy in.
    if (n_prompt > 0) {
        const size_t size = llama_state_seq_get_size(ctx, 0);
        std::vector<uint8_t> blob(size);
        t0 = now_ms();
        const size_t got = llama_state_seq_get_data(ctx, blob.data(), size, 0);
        printf("TIME state-get %.1f bytes=%zu mib=%.2f\n", now_ms() - t0, got, got / 1048576.0);
        if (!state_file.empty()) {
            t0 = now_ms();
            const int fd = open(state_file.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
            size_t off = 0;
            while (fd >= 0 && off < got) {
                const ssize_t w = write(fd, blob.data() + off, got - off);
                if (w <= 0) break;
                off += (size_t) w;
            }
            if (fd >= 0) { fsync(fd); close(fd); }
            printf("TIME state-write-fsync %.1f bytes=%zu\n", now_ms() - t0, off);
            {
                std::vector<uint8_t> back(got);
                t0 = now_ms();
                const int rfd = open(state_file.c_str(), O_RDONLY);
                off = 0;
                while (rfd >= 0 && off < got) {
                    const ssize_t r = read(rfd, back.data() + off, got - off);
                    if (r <= 0) break;
                    off += (size_t) r;
                }
                if (rfd >= 0) close(rfd);
                printf("TIME state-read-cached %.1f bytes=%zu\n", now_ms() - t0, off);
            }
            // O_DIRECT bypasses the page cache, thus this read measures the flash itself.
            const size_t align = 4096;
            const size_t cap   = (got + align - 1) / align * align;
            void * dbuf = nullptr;
            if (posix_memalign(&dbuf, align, cap) == 0) {
                t0 = now_ms();
                const int dfd = open(state_file.c_str(), O_RDONLY | O_DIRECT);
                off = 0;
                while (dfd >= 0 && off < cap) {
                    const ssize_t r = read(dfd, (uint8_t *) dbuf + off, std::min<size_t>(cap - off, 16u << 20));
                    if (r <= 0) break;
                    off += (size_t) r;
                }
                if (dfd >= 0) close(dfd);
                const double dt = now_ms() - t0;
                printf("TIME state-read-direct %.1f bytes=%zu gbps=%.2f ok=%d\n", dt, off, off / dt / 1e6,
                       dfd >= 0 && off >= got && memcmp(dbuf, blob.data(), got) == 0);
                free(dbuf);
            }
            unlink(state_file.c_str());
        }
        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        t0 = now_ms();
        const size_t set = llama_state_seq_set_data(ctx, blob.data(), got, 0);
        printf("TIME state-set %.1f ok=%d\n", now_ms() - t0, set != 0);
        print_memory("state", model_path, smaps);
    }

    // The vision projector of ensure_vision and one image encode of image_embd in llama_jni.cpp.
    if (!mmproj.empty()) {
        ggml_backend_dev_t vdev = vision_dev == "none" ? nullptr : ggml_backend_dev_by_name(vision_dev.c_str());
        mtmd_context_params vp = mtmd_context_params_default();
        vp.use_gpu          = vdev != nullptr;
        vp.device           = vdev;
        vp.n_threads        = n_threads;
        vp.print_timings    = false;
        vp.warmup           = vision_warmup;
        vp.image_max_tokens = image_tokens;
        t0 = now_ms();
        mtmd_context * vctx = mtmd_init_from_file(mmproj.c_str(), model, vp);
        printf("TIME vision-init %.1f ok=%d dev=%s tokens=%d warmup=%d\n", now_ms() - t0, vctx != nullptr,
               vdev ? ggml_backend_dev_name(vdev) : "CPU", image_tokens, vision_warmup);
        print_memory("vision-init", model_path, smaps);
        if (vctx != nullptr && image_w > 0 && image_h > 0) {
            std::vector<unsigned char> rgb((size_t) image_w * image_h * 3);
            for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = (unsigned char) ((i * 2654435761u) >> 24);
            mtmd_bitmap * bmp = mtmd_bitmap_init((uint32_t) image_w, (uint32_t) image_h, rgb.data());
            mtmd_input_chunks * chunks = mtmd_input_chunks_init();
            const std::string prompt = std::string(mtmd_default_marker()) + "\nDescribe the image.";
            mtmd_input_text text = { prompt.c_str(), prompt.size(), true, true };
            const mtmd_bitmap * bmps[1] = { bmp };
            const int32_t tk = mtmd_tokenize(vctx, chunks, &text, bmps, 1);
            for (size_t c = 0; tk == 0 && c < mtmd_input_chunks_size(chunks); ++c) {
                const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, c);
                if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
                t0 = now_ms();
                const int32_t rc = mtmd_encode_chunk(vctx, chunk);
                printf("TIME vision-encode %.1f rc=%d image=%dx%d tokens=%zu\n", now_ms() - t0, rc, image_w, image_h,
                       mtmd_input_chunk_get_n_tokens(chunk));
            }
            print_memory("vision-encode", model_path, smaps);
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bmp);
        }
        mtmd_free(vctx);
        print_memory("vision-free", model_path, smaps);
    }

    common_memory_breakdown_print(ctx);
    llama_batch_free(batch);
    // The order of ~Engine in llama_jni.cpp: the driver, then the draft context, then the target context, then the pool.
    app_spec_disable(app);
    if (app_pool != nullptr && ctx_dft != nullptr) llama_detach_threadpool(ctx_dft);
    spec_init.reset();
    if (app_pool != nullptr) llama_detach_threadpool(ctx);
    llama_free(ctx);
    if (app_pool != nullptr) ggml_threadpool_free(app_pool);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
