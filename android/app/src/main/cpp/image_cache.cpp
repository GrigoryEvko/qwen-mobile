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
 *   reserved   u64, zero
 *   floats     n_tokens x n_embd x 4 bytes
 */
constexpr char     kMagic[4]    = {'Q', 'M', 'I', 'E'};
constexpr uint32_t kFileVersion = 1;
constexpr size_t   kHeaderBytes = 32;
constexpr const char * kSuffix  = ".embd";
/** The limits of one encoder output: the token budget of an image and the width of the model. */
constexpr uint32_t kMaxTokens   = 1u << 16;
constexpr uint32_t kMaxEmbd     = 1u << 16;

/** The header of an entry as bytes. */
std::vector<uint8_t> encode_head(const ImageInfo & info) {
    std::vector<uint8_t> out(kHeaderBytes, 0);
    memcpy(out.data(), kMagic, 4);
    const uint32_t fields[5] = {kFileVersion, info.nx, info.ny, info.n_tokens, info.n_embd};
    memcpy(out.data() + 4, fields, sizeof(fields));
    return out;
}

/** Read the header of a file. Returns false when the file is not an image file, or its length is not the header plus the floats. */
bool read_head(const std::string & path, ImageInfo & info) {
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
    info.nx       = fields[1];
    info.ny       = fields[2];
    info.n_tokens = fields[3];
    info.n_embd   = fields[4];
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
            cache_io::remove_file(path);
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

bool ImageCache::read_file(Entry & entry) const {
    ImageInfo info;
    const std::string path = path_of(entry.id);
    if (!read_head(path, info) || info.n_floats() != entry.info.n_floats()) {
        return false;
    }
    std::vector<float> data(info.n_floats());
    if (!cache_io::read_range(path, kHeaderBytes, data.data(), data.size() * sizeof(float))) {
        return false;
    }
    entry.data = std::move(data);
    return true;
}

bool ImageCache::write_file(const Entry & entry) const {
    const std::vector<uint8_t> head = encode_head(entry.info);
    return cache_io::write_file_atomic(path_of(entry.id), {{head.data(), head.size()},
                                                          {entry.data.data(), entry.data.size() * sizeof(float)}});
}

const float * ImageCache::get(const std::string & id) {
    const auto found = index_.find(id);
    if (found == index_.end()) {
        return nullptr;
    }
    auto it = found->second;
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
    if (!cache_io::is_hex(id) || info.n_floats() == 0 || data == nullptr ||
        info.n_tokens > kMaxTokens || info.n_embd > kMaxEmbd) {
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
        if (found->second->info.n_floats() != info.n_floats()) {
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
