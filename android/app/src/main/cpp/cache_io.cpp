#include "cache_io.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace cache_io {

uint64_t fnv1a64(const void * data, size_t len, uint64_t seed) {
    const auto * p = static_cast<const uint8_t *>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

std::string hex64(uint64_t value) {
    static const char digits[] = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[(size_t) i] = digits[value & 0xF];
        value >>= 4;
    }
    return out;
}

namespace {

/** The round constants of SHA-256, FIPS 180-4. */
constexpr uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/** One 64-byte block of SHA-256 into the state h. */
void sha256_block(uint32_t h[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t) block[4 * i] << 24 | (uint32_t) block[4 * i + 1] << 16 |
               (uint32_t) block[4 * i + 2] << 8 | (uint32_t) block[4 * i + 3];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1  = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch  = (e & f) ^ (~e & g);
        const uint32_t t1  = k + s1 + ch + kSha256K[i] + w[i];
        const uint32_t s0  = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2  = s0 + maj;
        k = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
}

/** The path of the temporary file of an atomic write. */
std::string temp_path(const std::string & path) {
    return path + ".tmp";
}

} // namespace

std::string sha256_hex(const void * data, size_t len) {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const auto * p = static_cast<const uint8_t *>(data);
    size_t i = 0;
    for (; i + 64 <= len; i += 64) {
        sha256_block(h, p + i);
    }
    // The last block: the remaining bytes, the 0x80 marker, zeros, and the bit length.
    uint8_t tail[128] = {};
    const size_t rest = len - i;
    // An empty input can come with a null pointer, and memcpy from null is undefined also for 0 bytes.
    if (rest > 0) {
        memcpy(tail, p + i, rest);
    }
    tail[rest] = 0x80;
    const size_t total = rest + 1 + 8 <= 64 ? 64 : 128;
    const uint64_t bits = (uint64_t) len * 8;
    for (int b = 0; b < 8; ++b) {
        tail[total - 1 - (size_t) b] = (uint8_t) (bits >> (8 * b));
    }
    sha256_block(h, tail);
    if (total == 128) {
        sha256_block(h, tail + 64);
    }
    static const char digits[] = "0123456789abcdef";
    std::string out(64, '0');
    for (int k = 0; k < 8; ++k) {
        for (int b = 0; b < 8; ++b) {
            out[(size_t) (8 * k + b)] = digits[(h[k] >> (28 - 4 * b)) & 0xF];
        }
    }
    return out;
}

bool is_hex(const std::string & text) {
    if (text.empty()) {
        return false;
    }
    for (char c : text) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool make_dirs(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    for (size_t pos = 1; pos <= path.size(); ++pos) {
        if (pos == path.size() || path[pos] == '/') {
            const std::string part = path.substr(0, pos);
            if (mkdir(part.c_str(), 0700) != 0 && errno != EEXIST) {
                return false;
            }
        }
    }
    struct stat st = {};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool stat_file(const std::string & path, FileStat & out) {
    struct stat st = {};
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        return false;
    }
    out.size  = (uint64_t) st.st_size;
    out.mtime = (int64_t) st.st_mtime;
    return true;
}

namespace {

/** The entries of a directory that satisfy the predicate on the name and the file type. */
std::vector<std::string> list_entries(const std::string & dir, bool want_dirs, const std::string & suffix) {
    std::vector<std::string> out;
    DIR * d = opendir(dir.c_str());
    if (d == nullptr) {
        return out;
    }
    while (dirent * entry = readdir(d)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        if (name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        struct stat st = {};
        if (stat((dir + "/" + name).c_str(), &st) != 0) {
            continue;
        }
        if (want_dirs ? S_ISDIR(st.st_mode) : S_ISREG(st.st_mode)) {
            out.push_back(name);
        }
    }
    closedir(d);
    return out;
}

} // namespace

std::vector<std::string> list_files(const std::string & dir, const std::string & suffix) {
    return list_entries(dir, false, suffix);
}

std::vector<std::string> list_dirs(const std::string & dir) {
    return list_entries(dir, true, "");
}

bool read_range(const std::string & path, uint64_t offset, void * out, size_t n) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    auto * dst = static_cast<uint8_t *>(out);
    size_t done = 0;
    bool ok = true;
    while (done < n) {
        const ssize_t got = pread(fd, dst + done, n - done, (off_t) (offset + done));
        if (got <= 0) {
            ok = false;
            break;
        }
        done += (size_t) got;
    }
    close(fd);
    return ok;
}

bool read_file(const std::string & path, std::vector<uint8_t> & out, size_t max_bytes) {
    FileStat st;
    if (!stat_file(path, st) || st.size > max_bytes) {
        return false;
    }
    out.resize((size_t) st.size);
    return read_range(path, 0, out.data(), out.size());
}

bool write_file_atomic(const std::string & path, const std::vector<Part> & parts) {
    const std::string tmp = temp_path(path);
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
        return false;
    }
    bool ok = true;
    for (const Part & part : parts) {
        const auto * src = static_cast<const uint8_t *>(part.first);
        size_t done = 0;
        while (ok && done < part.second) {
            const ssize_t put = write(fd, src + done, part.second - done);
            if (put <= 0) {
                ok = false;
                break;
            }
            done += (size_t) put;
        }
    }
    // The bytes reach the disk before the rename, thus a power loss keeps the previous file.
    ok = ok && fsync(fd) == 0;
    close(fd);
    ok = ok && rename(tmp.c_str(), path.c_str()) == 0;
    if (!ok) {
        unlink(tmp.c_str());
    }
    return ok;
}

bool remove_file(const std::string & path) {
    return unlink(path.c_str()) == 0 || errno == ENOENT;
}

void remove_tree(const std::string & path) {
    for (const std::string & name : list_dirs(path)) {
        remove_tree(path + "/" + name);
    }
    for (const std::string & name : list_files(path, "")) {
        unlink((path + "/" + name).c_str());
    }
    rmdir(path.c_str());
}

AsyncWriter::AsyncWriter() : thread_([this] { run(); }) {}

AsyncWriter::~AsyncWriter() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
}

void AsyncWriter::write(std::string path, std::shared_ptr<const std::vector<uint8_t>> head,
                        std::shared_ptr<const Blob> body) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_.push_back(Job{std::move(path), std::move(head), std::move(body), false});
    }
    cv_.notify_all();
}

void AsyncWriter::remove(std::string path) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_.push_back(Job{std::move(path), nullptr, nullptr, true});
    }
    cv_.notify_all();
}

void AsyncWriter::drain() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return jobs_.empty() && active_ == 0; });
}

std::vector<std::string> AsyncWriter::take_failed() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    out.swap(failed_);
    return out;
}

void AsyncWriter::run() {
    // A snapshot file holds 20 to 44 MB, and its write takes CPU time after
    // each prompt. The display thread must keep the cores, thus this thread
    // runs at the nice value of the compute threads of the engine
    // (kComputeNice in llama_jni.cpp). On Linux the nice value belongs to
    // the thread, and a higher value needs no permission.
    constexpr int kWriterNice = 10;
    setpriority(PRIO_PROCESS, 0, kWriterNice);
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        cv_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
        if (jobs_.empty()) {
            return;
        }
        Job job = std::move(jobs_.front());
        jobs_.pop_front();
        active_ = 1;
        lock.unlock();
        bool ok = true;
        if (job.remove) {
            remove_file(job.path);
        } else {
            std::vector<Part> parts;
            if (job.head) {
                parts.emplace_back(job.head->data(), job.head->size());
            }
            if (job.body) {
                parts.emplace_back(job.body->data.get(), job.body->size);
            }
            ok = write_file_atomic(job.path, parts);
        }
        lock.lock();
        if (!ok) {
            failed_.push_back(std::move(job.path));
        }
        active_ = 0;
        cv_.notify_all();
    }
}

} // namespace cache_io
