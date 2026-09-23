#include "state_cache.h"

#include <algorithm>
#include <cstring>

namespace {

/**
 * The file of a snapshot: a fixed header, the items, then the state
 * bytes. Every number is little-endian.
 *
 *   "QMST"        4 bytes
 *   version       u32, kFileVersion
 *   key           u64, hash_items(items)
 *   n_items       u32
 *   n_pos         i32
 *   items_bytes   u32, the length of the items block
 *   reserved      u32, zero
 *   body_size     u64, the length of the state bytes
 *   checksum      u64, the checksum of the file
 *   items block:  for each item, token i32, and for an image item id_len u16 then the id
 *   state bytes
 *
 * The checksum is cache_io::checksum64 of the header bytes before it, then of
 * the items block, then of the state bytes. Each part has the checksum of the
 * part before it as its seed, thus the checksum covers each byte of the file
 * but its own 8 bytes. A damaged float in a restored state gives NaN in the
 * next decode, and a damaged n_pos gives the next tokens wrong positions.
 */
constexpr char     kMagic[4]     = {'Q', 'M', 'S', 'T'};
constexpr uint32_t kFileVersion  = 2;
constexpr size_t   kHeaderBytes  = 48;
constexpr size_t   kChecksumAt   = kHeaderBytes - sizeof(uint64_t);
constexpr size_t   kMaxItemBytes = 64u << 20;
constexpr const char * kSuffix   = ".snap";

/** A little-endian byte writer. */
struct Writer {
    std::vector<uint8_t> out;

    template <typename T> void put(T v) {
        uint8_t b[sizeof(T)];
        memcpy(b, &v, sizeof(T));
        out.insert(out.end(), b, b + sizeof(T));
    }

    void put_bytes(const void * p, size_t n) {
        const auto * s = static_cast<const uint8_t *>(p);
        out.insert(out.end(), s, s + n);
    }
};

/** A little-endian byte reader that reports an overrun with ok = false. */
struct Reader {
    const uint8_t * p;
    size_t          n;
    size_t          pos = 0;
    bool            ok  = true;

    template <typename T> T get() {
        T v = T();
        if (pos + sizeof(T) > n) {
            ok = false;
            return v;
        }
        memcpy(&v, p + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }

    std::string get_string(size_t len) {
        if (pos + len > n) {
            ok = false;
            return {};
        }
        std::string s(reinterpret_cast<const char *>(p + pos), len);
        pos += len;
        return s;
    }
};

/**
 * The header and the items block of a snapshot, without the state bytes, but
 * with the checksum of the file, thus of the state bytes too. O(items + bytes).
 */
std::shared_ptr<const std::vector<uint8_t>> encode_head(const Snapshot & snap, const cache_io::Blob & bytes) {
    Writer items;
    for (const MemItem & item : snap.items) {
        items.put<int32_t>(item.token);
        if (item.token == kMemTokenNull) {
            items.put<uint16_t>((uint16_t) std::min<size_t>(item.image_id.size(), 0xFFFF));
            items.put_bytes(item.image_id.data(), std::min<size_t>(item.image_id.size(), 0xFFFF));
        }
    }
    Writer w;
    w.put_bytes(kMagic, 4);
    w.put<uint32_t>(kFileVersion);
    w.put<uint64_t>(snap.key);
    w.put<uint32_t>((uint32_t) snap.items.size());
    w.put<int32_t>(snap.n_pos);
    w.put<uint32_t>((uint32_t) items.out.size());
    w.put<uint32_t>(0);
    w.put<uint64_t>((uint64_t) snap.byte_size);
    w.put<uint64_t>(0);
    w.out.insert(w.out.end(), items.out.begin(), items.out.end());
    uint64_t sum = cache_io::checksum64(w.out.data(), kChecksumAt);
    sum          = cache_io::checksum64(items.out.data(), items.out.size(), sum);
    sum          = cache_io::checksum64(bytes.data.get(), bytes.size, sum);
    memcpy(w.out.data() + kChecksumAt, &sum, sizeof(sum));
    return std::make_shared<const std::vector<uint8_t>>(std::move(w.out));
}

/**
 * Read the header and the items of a snapshot file into snap, without its
 * bytes. checksum gets the checksum of the file, and head_sum gets the
 * checksum of the header and the items: the seed of the checksum of the
 * state bytes. Returns false when the file is not a snapshot file of this
 * version, or its length does not agree with its header. O(items).
 */
bool read_head(const std::string & path, Snapshot & snap, uint64_t & body_offset, uint64_t & checksum,
               uint64_t & head_sum) {
    cache_io::FileStat st;
    if (!cache_io::stat_file(path, st) || st.size < kHeaderBytes) {
        return false;
    }
    uint8_t head[kHeaderBytes];
    if (!cache_io::read_range(path, 0, head, kHeaderBytes)) {
        return false;
    }
    Reader r{head, kHeaderBytes};
    if (memcmp(head, kMagic, 4) != 0) {
        return false;
    }
    r.pos = 4;
    const uint32_t version     = r.get<uint32_t>();
    const uint64_t key         = r.get<uint64_t>();
    const uint32_t n_items     = r.get<uint32_t>();
    const int32_t  n_pos       = r.get<int32_t>();
    const uint32_t items_bytes = r.get<uint32_t>();
    r.get<uint32_t>();
    const uint64_t body_size   = r.get<uint64_t>();
    checksum                   = r.get<uint64_t>();
    // Each number of the header limits an allocation, thus each one must agree
    // with the length of the file before it is used. An item takes
    // at least its token, thus n_items has a limit of items_bytes / 4: without
    // it, a damaged count reserves gigabytes. The subtraction below cannot wrap,
    // where the sum of a huge body_size can.
    if (!r.ok || version != kFileVersion || items_bytes > kMaxItemBytes ||
        n_items > items_bytes / sizeof(int32_t) || st.size < kHeaderBytes + items_bytes ||
        body_size != st.size - kHeaderBytes - items_bytes) {
        return false;
    }
    std::vector<uint8_t> block(items_bytes);
    if (!cache_io::read_range(path, kHeaderBytes, block.data(), block.size())) {
        return false;
    }
    Reader items{block.data(), block.size()};
    std::vector<MemItem> parsed;
    parsed.reserve(n_items);
    for (uint32_t i = 0; i < n_items && items.ok; ++i) {
        MemItem item;
        item.token = items.get<int32_t>();
        if (item.token == kMemTokenNull) {
            const uint16_t len = items.get<uint16_t>();
            item.image_id = items.get_string(len);
        }
        parsed.push_back(std::move(item));
    }
    if (!items.ok || items.pos != items.n || hash_items(parsed) != key) {
        return false;
    }
    head_sum        = cache_io::checksum64(head, kChecksumAt);
    head_sum        = cache_io::checksum64(block.data(), block.size(), head_sum);
    snap.items      = std::move(parsed);
    snap.n_pos      = n_pos;
    snap.key        = key;
    snap.byte_size  = (size_t) body_size;
    snap.on_disk    = true;
    snap.disk_stamp = st.mtime;
    body_offset     = kHeaderBytes + items_bytes;
    return true;
}

} // namespace

bool is_item_prefix(const std::vector<MemItem> & a, const std::vector<MemItem> & b) {
    return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}

uint64_t hash_items(const std::vector<MemItem> & items) {
    uint64_t h = cache_io::fnv1a64(nullptr, 0);
    for (const MemItem & item : items) {
        h = cache_io::fnv1a64(&item.token, sizeof(item.token), h);
        if (item.token == kMemTokenNull) {
            h = cache_io::fnv1a64(item.image_id.data(), item.image_id.size(), h);
        }
    }
    return h;
}

StateCache::StateCache(size_t ram_budget, size_t disk_budget, std::string dir)
    : ram_budget_(ram_budget), disk_budget_(disk_budget), dir_(std::move(dir)) {
    if (dir_.empty() || disk_budget_ == 0) {
        dir_.clear();
        return;
    }
    if (!cache_io::make_dirs(dir_)) {
        dir_.clear();
        return;
    }
    writer_ = std::make_unique<cache_io::AsyncWriter>();
    scan_dir();
}

StateCache::~StateCache() = default;

std::string StateCache::path_of(const Snapshot & snap) const {
    return dir_ + "/" + cache_io::hex64(snap.key) + kSuffix;
}

void StateCache::scan_dir() {
    for (const std::string & name : cache_io::list_files(dir_, ".tmp")) {
        cache_io::remove_file(dir_ + "/" + name);
    }
    for (const std::string & name : cache_io::list_files(dir_, kSuffix)) {
        const std::string path = dir_ + "/" + name;
        Snapshot snap;
        uint64_t body_offset = 0;
        uint64_t checksum    = 0;
        uint64_t head_sum    = 0;
        if (!read_head(path, snap, body_offset, checksum, head_sum) || path_of(snap) != path) {
            // A file of an older version or a damaged file.
            cache_io::remove_file(path);
            removed_at_scan_ += 1;
            continue;
        }
        disk_bytes_ += snap.byte_size;
        entries_.push_back(std::move(snap));
    }
    // The newest file first: it is the most recently used one of the last run.
    entries_.sort([](const Snapshot & a, const Snapshot & b) { return a.disk_stamp > b.disk_stamp; });
    if (!entries_.empty()) {
        stamp_ = entries_.front().disk_stamp;
    }
    evict_disk(nullptr);
}

size_t StateCache::resident() const {
    size_t n = 0;
    for (const Snapshot & s : entries_) {
        n += s.bytes ? 1 : 0;
    }
    return n;
}

const Snapshot * StateCache::best_prefix(const std::vector<MemItem> & items, size_t limit) {
    forget_failed_writes();
    const Snapshot * best = nullptr;
    for (const Snapshot & s : entries_) {
        const size_t n = s.items.size();
        if (n == 0 || n > limit || (best != nullptr && n <= best->items.size())) {
            continue;
        }
        if (is_item_prefix(s.items, items)) {
            best = &s;
        }
    }
    return best;
}

const Snapshot * StateCache::find(const std::vector<MemItem> & items) const {
    const uint64_t key = hash_items(items);
    for (const Snapshot & s : entries_) {
        if (s.key == key && s.items == items) {
            return &s;
        }
    }
    return nullptr;
}

StateCache::List::iterator StateCache::iterator_of(const Snapshot * snap) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (&*it == snap) {
            return it;
        }
    }
    return entries_.end();
}

std::shared_ptr<const cache_io::Blob> StateCache::bytes(const Snapshot * snap) {
    auto it = iterator_of(snap);
    if (it == entries_.end()) {
        return nullptr;
    }
    if (!it->bytes) {
        const std::string path = path_of(*it);
        bool damaged = false;
        auto read = [&]() -> std::shared_ptr<cache_io::Blob> {
            Snapshot head;
            uint64_t body_offset = 0;
            uint64_t checksum    = 0;
            uint64_t head_sum    = 0;
            if (!it->on_disk || !read_head(path, head, body_offset, checksum, head_sum) || head.key != it->key ||
                head.byte_size != it->byte_size) {
                return nullptr;
            }
            auto blob = std::make_shared<cache_io::Blob>(it->byte_size);
            if (!cache_io::read_range(path, body_offset, blob->data.get(), blob->size)) {
                return nullptr;
            }
            damaged = cache_io::checksum64(blob->data.get(), blob->size, head_sum) != checksum;
            return damaged ? nullptr : blob;
        };
        std::shared_ptr<cache_io::Blob> blob = read();
        if (!blob && it->on_disk && writer_) {
            // A snapshot that left RAM before the writer thread wrote its file
            // has no file yet: the read waits for the queued writes one time.
            writer_->drain();
            blob = read();
        }
        if (!blob) {
            if (damaged) {
                damaged_.push_back(it->key);
            }
            erase(it, true);
            return nullptr;
        }
        make_resident(it, blob);
        entries_.splice(entries_.begin(), entries_, it);
        return blob;
    }
    entries_.splice(entries_.begin(), entries_, it);
    return it->bytes;
}

void StateCache::make_resident(List::iterator it, std::shared_ptr<const cache_io::Blob> bytes) {
    // The RAM counter holds the items too, thus the admission compares the same
    // sum with the budget.
    if (it->bytes || !bytes || bytes->size + it->items.size() * sizeof(MemItem) > ram_budget_) {
        return;
    }
    // byte_size stays the one of the entry: the file, the header and the two
    // byte counters carry that value already, and the caller of put dropped
    // an entry whose length differs.
    it->bytes      = std::move(bytes);
    ram_bytes_    += it->byte_size + it->items.size() * sizeof(MemItem);
    evict_ram(&*it);
}

void StateCache::put(std::vector<MemItem> items, int32_t n_pos, std::shared_ptr<const cache_io::Blob> bytes) {
    if (items.empty() || !bytes) {
        return;
    }
    forget_failed_writes();
    const uint64_t key = hash_items(items);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->key == key && it->items == items) {
            if (it->n_pos != n_pos || it->byte_size != bytes->size) {
                // The same items give the same positions and the same number of
                // state bytes. A different value comes from another model or
                // another context length, thus the entry and its file are stale.
                erase(it, true);
                break;
            }
            make_resident(it, bytes);
            entries_.splice(entries_.begin(), entries_, it);
            if (!it->on_disk) {
                write_file(it, bytes);
            }
            return;
        }
    }
    Snapshot snap;
    snap.items     = std::move(items);
    snap.n_pos     = n_pos;
    snap.key       = key;
    snap.byte_size = bytes->size;
    entries_.push_front(std::move(snap));
    auto it = entries_.begin();
    make_resident(it, bytes);
    write_file(it, bytes);
    if (!it->bytes && !it->on_disk) {
        entries_.erase(it);
    }
}

void StateCache::write_file(List::iterator it, const std::shared_ptr<const cache_io::Blob> & bytes) {
    if (dir_.empty() || !bytes || bytes->size > disk_budget_) {
        return;
    }
    it->on_disk    = true;
    it->disk_stamp = ++stamp_;
    disk_bytes_   += it->byte_size;
    writer_->write(path_of(*it), encode_head(*it, *bytes), bytes);
    evict_disk(&*it);
}

void StateCache::evict_ram(const Snapshot * keep) {
    while (ram_bytes_ > ram_budget_) {
        auto victim = entries_.end();
        for (auto it = entries_.end(); it != entries_.begin();) {
            --it;
            if (it->bytes && &*it != keep) {
                victim = it;
                break;
            }
        }
        if (victim == entries_.end()) {
            return;
        }
        ram_bytes_ -= victim->byte_size + victim->items.size() * sizeof(MemItem);
        victim->bytes.reset();
        if (!victim->on_disk) {
            entries_.erase(victim);
        }
    }
}

void StateCache::evict_disk(const Snapshot * keep) {
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
        writer_->remove(path_of(*victim));
        disk_bytes_ -= victim->byte_size;
        victim->on_disk = false;
        if (!victim->bytes) {
            entries_.erase(victim);
        }
    }
}

StateCache::List::iterator StateCache::erase(List::iterator it, bool remove_file) {
    if (it->bytes) {
        ram_bytes_ -= it->byte_size + it->items.size() * sizeof(MemItem);
    }
    if (it->on_disk) {
        disk_bytes_ -= it->byte_size;
        if (remove_file && writer_) {
            writer_->remove(path_of(*it));
        }
    }
    return entries_.erase(it);
}

void StateCache::forget_failed_writes() {
    if (!writer_) {
        return;
    }
    for (const std::string & path : writer_->take_failed()) {
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->on_disk && path_of(*it) == path) {
                disk_bytes_ -= it->byte_size;
                it->on_disk = false;
                if (!it->bytes) {
                    entries_.erase(it);
                }
                break;
            }
        }
    }
}

void StateCache::drop(const Snapshot * snap) {
    auto it = iterator_of(snap);
    if (it != entries_.end()) {
        erase(it, true);
    }
}

void StateCache::clear(bool disk_too) {
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (disk_too || !it->on_disk) {
            it = erase(it, disk_too);
        } else {
            if (it->bytes) {
                ram_bytes_ -= it->byte_size + it->items.size() * sizeof(MemItem);
                it->bytes.reset();
            }
            ++it;
        }
    }
}

void StateCache::drain() {
    if (writer_) {
        writer_->drain();
    }
}

std::vector<uint64_t> StateCache::take_damaged() {
    std::vector<uint64_t> out;
    out.swap(damaged_);
    return out;
}
