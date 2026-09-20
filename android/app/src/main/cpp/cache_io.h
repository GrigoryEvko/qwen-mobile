/**
 * File and hash helpers of the engine caches. No llama.cpp and no Android
 * dependency, thus a host test compiles this file with g++.
 */
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cache_io {

/** A byte buffer without zero fill. The state of one sequence is tens of megabytes. */
struct Blob {
    std::unique_ptr<uint8_t[]> data;
    size_t                     size = 0;

    explicit Blob(size_t n) : data(new uint8_t[n]), size(n) {}
};

/** The FNV-1a hash of the bytes, 64 bits. O(len). */
uint64_t fnv1a64(const void * data, size_t len, uint64_t seed = 14695981039346656037ull);

/** The lowercase hexadecimal text of a 64-bit value, 16 characters. */
std::string hex64(uint64_t value);

/** The lowercase hexadecimal SHA-256 of the bytes, 64 characters. O(len). */
std::string sha256_hex(const void * data, size_t len);

/** True when the text is lowercase hexadecimal and not empty. */
bool is_hex(const std::string & text);

/** The size and the modification time of a file. */
struct FileStat {
    uint64_t size  = 0;
    int64_t  mtime = 0;
};

/** Make the directory and its parents. Returns true when the directory exists at the end. */
bool make_dirs(const std::string & path);

/** The size and the modification time of a regular file. Returns false when there is no such file. */
bool stat_file(const std::string & path, FileStat & out);

/** The names (not the paths) of the regular files in a directory that end with the suffix. */
std::vector<std::string> list_files(const std::string & dir, const std::string & suffix);

/** The names of the subdirectories of a directory. */
std::vector<std::string> list_dirs(const std::string & dir);

/** Read the bytes [offset, offset + n) of a file. Returns false when the file is shorter. */
bool read_range(const std::string & path, uint64_t offset, void * out, size_t n);

/** Read the whole file. Returns false when the file is not readable or larger than max_bytes. */
bool read_file(const std::string & path, std::vector<uint8_t> & out, size_t max_bytes);

/** One part of a file write: a pointer and its length. */
using Part = std::pair<const void *, size_t>;

/**
 * Write the parts to a temporary file next to the path, then rename it
 * onto the path. A reader sees the old file or the new file, never a
 * partial one. Returns false when a step fails, with the temporary file
 * removed.
 */
bool write_file_atomic(const std::string & path, const std::vector<Part> & parts);

/** Remove a file. Returns true when the file is gone at the end. */
bool remove_file(const std::string & path);

/** Remove a directory with everything in it. */
void remove_tree(const std::string & path);

/**
 * A thread that writes and removes files in the order of the requests,
 * thus the caller does not wait for the disk. The destructor completes
 * the queued requests. Each request holds its bytes, thus the caller can
 * release its own reference at once.
 */
class AsyncWriter {
public:
    AsyncWriter();
    ~AsyncWriter();

    AsyncWriter(const AsyncWriter &) = delete;
    AsyncWriter & operator=(const AsyncWriter &) = delete;

    /** Queue a write of head then body to the path, through write_file_atomic. Body can be null. */
    void write(std::string path, std::shared_ptr<const std::vector<uint8_t>> head, std::shared_ptr<const Blob> body);

    /** Queue the removal of a file. It runs after the writes that came before it. */
    void remove(std::string path);

    /** Wait until every queued request is done. */
    void drain();

    /** The paths of the writes that failed since the last call. */
    std::vector<std::string> take_failed();

private:
    struct Job {
        std::string                                 path;
        std::shared_ptr<const std::vector<uint8_t>> head;
        std::shared_ptr<const Blob>                 body;
        bool                                        remove = false;
    };

    void run();

    // The thread is the last member: a member starts before the members that
    // follow it, thus a thread declared first runs while the mutex, the queue
    // and the flags of this object are still raw memory.
    std::mutex               mutex_;
    std::condition_variable  cv_;
    std::deque<Job>          jobs_;
    std::vector<std::string> failed_;
    size_t                   active_ = 0;
    bool                     stop_   = false;
    std::thread              thread_;
};

} // namespace cache_io
