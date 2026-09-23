#include "image_cache.h"

#include "cache_io.h"

#include <cstring>

namespace {

/**
 * The file of an image: a fixed header, then the floats. Every number is
 * little-endian.
 *
 *   "QMIE"     4 bytes
 *   version    u32, kFileVersion
 *   nx, ny     u32 each, the bitmap dimensions
 *   n_tokens   u32
 *   n_embd     u32
 *   checksum   u64, the checksum of the file
 *   floats     n_tokens x n_embd x 4 bytes
 *
 * The checksum is cache_io::checksum64 of the header bytes before it, then of
 * the floats with the first checksum as the seed. Thus it covers each byte of
 * the file but its own 8 bytes. A damaged float goes into the prompt, and a
 * damaged side of the bitmap gives the image tokens wrong positions.
 */
constexpr char     kMagic[4]    = {'Q', 'M', 'I', 'E'};
constexpr uint32_t kFileVersion = 2;
constexpr size_t   kHeaderBytes = 32;
constexpr size_t   kChecksumAt  = kHeaderBytes - sizeof(uint64_t);
constexpr const char * kSuffix  = ".embd";
/** The limits of one encoder output: the token budget of an image and the width of the model. */
constexpr uint32_t kMaxTokens   = 1u << 16;
constexpr uint32_t kMaxEmbd     = 1u << 16;
/** The limit of each side of a bitmap, the same as the limit of the decode in llama_jni.cpp. */
constexpr uint32_t kMaxSide     = 1u << 16;

/** The header of an entry as bytes, with the checksum of the file, thus of the floats too. O(floats). */
std::vector<uint8_t> encode_head(const ImageInfo & info, const float * data) {
    std::vector<uint8_t> out(kHeaderBytes, 0);
    memcpy(out.data(), kMagic, 4);
    const uint32_t fields[5] = {kFileVersion, info.nx, info.ny, info.n_tokens, info.n_embd};
    memcpy(out.data() + 4, fields, sizeof(fields));
    uint64_t sum = cache_io::checksum64(out.data(), kChecksumAt);
    sum          = cache_io::checksum64(data, info.n_floats() * sizeof(float), sum);
    memcpy(out.data() + kChecksumAt, &sum, sizeof(sum));
    return out;
}

/**
 * Read the header of a file. When they are not null, checksum gets the
 * checksum of the file, and head_sum gets the checksum of the header bytes
 * before it: the seed of the checksum of the floats. Returns false when the
 * file is not an image file of this version, or its length is not the header
 * plus the floats.
 */
bool read_head(const std::string & path, ImageInfo & info, uint64_t * checksum = nullptr,
               uint64_t * head_sum = nullptr) {
    cache_io::FileStat st;
    uint8_t head[kHeaderBytes];
    if (!cache_io::stat_file(path, st) || st.size < kHeaderBytes || !cache_io::read_range(path, 0, head, kHeaderBytes) ||
        memcmp(head, kMagic, 4) != 0) {
        return false;
    }
    uint32_t fields[5];
    memcpy(fields, head + 4, sizeof(fields));
    if (fields[0] != kFileVersion) {
        return false;
    }
    // The shape must stay inside the limits of the encoder, thus n_tokens x
    // n_embd cannot overflow the length check below. Without this, a 32-byte
    // file of two huge dimensions passes the check and every later read of it
    // throws on the allocation of the vector.
    if (fields[3] == 0 || fields[3] > kMaxTokens || fields[4] == 0 || fields[4] > kMaxEmbd) {
        return false;
    }
    // The engine tokenizes a placeholder bitmap of these dimensions, and the
    // preprocessor casts each side to int: zero or a side above the limit of
    // a decoded image is a damaged file.
    if (fields[1] == 0 || fields[1] > kMaxSide || fields[2] == 0 || fields[2] > kMaxSide) {
        return false;
    }
    info.nx       = fields[1];
    info.ny       = fields[2];
    info.n_tokens = fields[3];
    info.n_embd   = fields[4];
    if (checksum != nullptr) {
        memcpy(checksum, head + kChecksumAt, sizeof(*checksum));
    }
    if (head_sum != nullptr) {
        *head_sum = cache_io::checksum64(head, kChecksumAt);
    }
    return info.n_tokens > 0 && info.n_embd > 0 && st.size == kHeaderBytes + info.n_floats() * sizeof(float);
}

} // namespace

ImageCache::ImageCache(size_t ram_budget, size_t disk_budget, std::string dir)
    : ram_budget_(ram_budget), disk_budget_(disk_budget), dir_(std::move(dir)) {
    if (dir_.empty() || disk_budget_ == 0 || !cache_io::make_dirs(dir_)) {
        dir_.clear();
        return;
    }
    scan_dir();
}

std::string ImageCache::path_of(const std::string & id) const {
    return dir_ + "/" + id + kSuffix;
}

void ImageCache::scan_dir() {
    for (const std::string & name : cache_io::list_files(dir_, ".tmp")) {
        cache_io::remove_file(dir_ + "/" + name);
    }
    for (const std::string & name : cache_io::list_files(dir_, kSuffix)) {
        const std::string path = dir_ + "/" + name;
        Entry entry;
        entry.id = name.substr(0, name.size() - strlen(kSuffix));
        cache_io::FileStat st;
        // A file of an entry that does not fit in RAM comes from a build with
        // another budget. It could never be served, thus it goes.
        if (!cache_io::is_hex(entry.id) || !read_head(path, entry.info) || !cache_io::stat_file(path, st) ||
            entry.bytes() > ram_budget_) {
            // A file of an older version, a damaged file, or a file of another budget.
            cache_io::remove_file(path);
            removed_at_scan_ += 1;
            continue;
        }
        entry.on_disk    = true;
        entry.disk_stamp = st.mtime;
        disk_bytes_     += entry.bytes();
        entries_.push_back(std::move(entry));
    }
    entries_.sort([](const Entry & a, const Entry & b) { return a.disk_stamp > b.disk_stamp; });
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        index_[it->id] = it;
    }
    if (!entries_.empty()) {
        stamp_ = entries_.front().disk_stamp;
    }
    evict_disk(nullptr);
}

bool ImageCache::info(const std::string & id, ImageInfo & out) {
    const auto found = index_.find(id);
    if (found == index_.end()) {
        return false;
    }
    auto it = found->second;
    if (it->data.empty()) {
        // The system or the user can clear the cache directory while the index stands.
        ImageInfo on_file;
        if (!it->on_disk || !read_head(path_of(it->id), on_file) || on_file.n_floats() != it->info.n_floats()) {
            erase(it, true);
            return false;
        }
    }
    entries_.splice(entries_.begin(), entries_, it);
    if (it->on_disk) {
        it->disk_stamp = ++stamp_;
    }
    out = it->info;
    return true;
}

bool ImageCache::read_file(Entry & entry) {
    ImageInfo info;
    uint64_t checksum = 0;
    uint64_t head_sum = 0;
    const std::string path = path_of(entry.id);
    if (!read_head(path, info, &checksum, &head_sum) || info.n_floats() != entry.info.n_floats()) {
        return false;
    }
    std::vector<float> data(info.n_floats());
    if (!cache_io::read_range(path, kHeaderBytes, data.data(), data.size() * sizeof(float))) {
        return false;
    }
    if (cache_io::checksum64(data.data(), data.size() * sizeof(float), head_sum) != checksum) {
        damaged_.push_back(entry.id);
        return false;
    }
    entry.data = std::move(data);
    return true;
}

bool ImageCache::write_file(const Entry & entry) const {
    const std::vector<uint8_t> head = encode_head(entry.info, entry.data.data());
    return cache_io::write_file_atomic(path_of(entry.id), {{head.data(), head.size()},
                                                          {entry.data.data(), entry.data.size() * sizeof(float)}});
}

const float * ImageCache::get(const std::string & id, uint32_t n_tokens, uint32_t n_embd) {
    const auto found = index_.find(id);
    if (found == index_.end()) {
        return nullptr;
    }
    auto it = found->second;
    if (it->info.n_tokens != n_tokens || it->info.n_embd != n_embd) {
        // A file of an earlier build, a damaged file, or the same bytes at
        // another size: the caller would read past the floats or read wrong ones.
        erase(it, true);
        return nullptr;
    }
    if (it->data.empty()) {
        if (!it->on_disk || !read_file(*it)) {
            erase(it, true);
            return nullptr;
        }
        ram_bytes_ += it->bytes();
        evict_ram(&*it);
    }
    entries_.splice(entries_.begin(), entries_, it);
    if (it->on_disk) {
        // A read counts as a use for the disk tier too, thus the eviction of
        // the files takes the entry that no turn asked for.
        it->disk_stamp = ++stamp_;
    }
    return it->data.data();
}

void ImageCache::put(const std::string & id, const ImageInfo & info, const float * data) {
    // The limits of read_head: an entry that its own file cannot give back does not enter.
    if (!cache_io::is_hex(id) || info.n_floats() == 0 || data == nullptr ||
        info.n_tokens > kMaxTokens || info.n_embd > kMaxEmbd ||
        info.nx == 0 || info.nx > kMaxSide || info.ny == 0 || info.ny > kMaxSide) {
        return;
    }
    // An entry that does not fit in RAM cannot be given to a caller, because
    // get returns a pointer into the entry. Such an entry on disk would say
    // yes to info and null to get on every turn, thus it never enters.
    if (info.n_floats() * sizeof(float) > ram_budget_) {
        return;
    }
    auto found = index_.find(id);
    if (found != index_.end()) {
        // The same key is the same id and the same shape (refer to get).
        if (found->second->info.n_tokens != info.n_tokens || found->second->info.n_embd != info.n_embd) {
            erase(found->second, true);
        } else {
            entries_.splice(entries_.begin(), entries_, found->second);
            return;
        }
    }
    Entry entry;
    entry.id   = id;
    entry.info = info;
    entry.data.assign(data, data + info.n_floats());
    entries_.push_front(std::move(entry));
    auto it = entries_.begin();
    index_[id] = it;
    ram_bytes_ += it->bytes();
    if (!dir_.empty() && it->bytes() <= disk_budget_ && write_file(*it)) {
        it->on_disk    = true;
        it->disk_stamp = ++stamp_;
        disk_bytes_   += it->bytes();
        evict_disk(&*it);
    }
    evict_ram(&*it);
    if (it->data.empty() && !it->on_disk) {
        erase(it, false);
    }
}

void ImageCache::evict_ram(const Entry * keep) {
    while (ram_bytes_ > ram_budget_) {
        auto victim = entries_.end();
        for (auto it = entries_.end(); it != entries_.begin();) {
            --it;
            if (!it->data.empty() && &*it != keep) {
                victim = it;
                break;
            }
        }
        if (victim == entries_.end()) {
            // Only the kept entry is in RAM. A single entry above the budget is not kept in RAM.
            if (keep != nullptr && !keep->data.empty() && keep->bytes() > ram_budget_) {
                auto it = index_.find(keep->id)->second;
                ram_bytes_ -= it->bytes();
                it->data.clear();
                it->data.shrink_to_fit();
            }
            return;
        }
        ram_bytes_ -= victim->bytes();
        victim->data.clear();
        victim->data.shrink_to_fit();
        if (!victim->on_disk) {
            erase(victim, false);
        }
    }
}

void ImageCache::evict_disk(const Entry * keep) {
    while (disk_bytes_ > disk_budget_) {
        auto victim = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->on_disk && &*it != keep && (victim == entries_.end() || it->disk_stamp < victim->disk_stamp)) {
                victim = it;
            }
        }
        if (victim == entries_.end()) {
            return;
        }
        cache_io::remove_file(path_of(victim->id));
        disk_bytes_ -= victim->bytes();
        victim->on_disk = false;
        if (victim->data.empty()) {
            erase(victim, false);
        }
    }
}

void ImageCache::erase(List::iterator it, bool remove_file) {
    if (!it->data.empty()) {
        ram_bytes_ -= it->bytes();
    }
    if (it->on_disk) {
        disk_bytes_ -= it->bytes();
        if (remove_file) {
            cache_io::remove_file(path_of(it->id));
        }
    }
    index_.erase(it->id);
    entries_.erase(it);
}

void ImageCache::drop(const std::string & id) {
    const auto found = index_.find(id);
    if (found != index_.end()) {
        erase(found->second, true);
    }
}

void ImageCache::clear(bool disk_too) {
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (disk_too || !it->on_disk) {
            auto next = std::next(it);
            erase(it, disk_too);
            it = next;
        } else {
            if (!it->data.empty()) {
                ram_bytes_ -= it->bytes();
                it->data.clear();
                it->data.shrink_to_fit();
            }
            ++it;
        }
    }
}

std::vector<std::string> ImageCache::take_damaged() {
    std::vector<std::string> out;
    out.swap(damaged_);
    return out;
}
