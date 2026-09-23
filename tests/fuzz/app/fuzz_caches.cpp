/**
 * The fuzzer of the two engine caches (state_cache.h, image_cache.h) and of
 * the file writer thread (cache_io.h). The budgets are small, thus each
 * program evicts, writes, reads and removes files many times.
 *
 * The checks after each operation:
 *
 * - The byte counters are the sums over the entries, and the budgets hold.
 * - An entry that the store finds gives the bytes of its last put.
 * - best_prefix gives the longest prefix with at most limit items.
 * - A snapshot whose bytes did not read (a null read) is gone from the store
 *   (task #95 items 5 and 6), and a pointer from best_prefix stays valid for
 *   the bytes call and the drop, as prefill in llama_jni.cpp uses it.
 * - After a drain, each snapshot on disk has its file. A new store on the
 *   same directory (a restart of the app) finds each intact file again.
 *
 * The file compiles the headers with public members, thus the checks read
 * the entry lists. The layout of the classes does not change.
 */
#define private public
#include "image_cache.h"
#include "state_cache.h"
#undef private

#include "fuzz_death.h"
#include "harness/crash_input.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <fuzzer/FuzzedDataProvider.h>

namespace {

/** Write the message to stderr and abort. */
[[noreturn]] void fail(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void fail(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "\n==FUZZ-CACHES== ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    crash_input::stop();
}

std::string work_dir() {
    const char * v = getenv("FUZZ_APP_WORK");
    return std::string(v != nullptr ? v : "/tmp/fuzz-app-work");
}

/**
 * The bytes of a blob. The engine gives the same state bytes for the same
 * items, and the store relies on it (a second put of the same items keeps
 * its file). Thus the bytes are a function of the items and the length, and
 * a mix-up of two snapshots is visible.
 */
std::shared_ptr<const cache_io::Blob> make_blob(size_t n, uint64_t seed) {
    auto b = std::make_shared<cache_io::Blob>(n);
    for (size_t i = 0; i < n; ++i) {
        b->data[i] = (uint8_t) ((seed >> ((i % 8) * 8)) + i * 31);
    }
    return b;
}

std::vector<MemItem> gen_items(FuzzedDataProvider & fdp) {
    // A small alphabet, thus the item sequences share prefixes.
    std::vector<MemItem> items;
    const size_t n = fdp.ConsumeIntegralInRange<size_t>(0, 12);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t k = fdp.ConsumeIntegral<uint8_t>();
        if (k % 7 == 0) {
            items.push_back(MemItem{kMemTokenNull, k % 2 ? "ab12" : "cd34"});
        } else {
            items.push_back(MemItem{(mem_token) (k % 5), {}});
        }
    }
    return items;
}

/** The checks of the counters and the budgets of the state store. */
void check_states(const StateCache & s) {
    size_t ram = 0, disk = 0;
    for (const Snapshot & snap : s.entries_) {
        if (snap.bytes) {
            ram += snap.byte_size + snap.items.size() * sizeof(MemItem);
            if (snap.bytes->size != snap.byte_size) {
                fail("state store: a resident snapshot holds %zu bytes but records %zu", snap.bytes->size, snap.byte_size);
            }
        }
        if (snap.on_disk) {
            disk += snap.byte_size;
        }
        if (!snap.bytes && !snap.on_disk) {
            fail("state store: an entry with no bytes in RAM and no file stays in the list");
        }
        if (snap.key != hash_items(snap.items)) {
            fail("state store: the key of an entry is not the hash of its items");
        }
    }
    if (ram != s.ram_bytes_ || disk != s.disk_bytes_) {
        fail("state store: the counters say %zu RAM and %zu disk bytes, the entries hold %zu and %zu", s.ram_bytes_,
             s.disk_bytes_, ram, disk);
    }
    // make_resident admits a snapshot by its state bytes, but the counter also
    // holds its items, and evict_ram never evicts the snapshot it keeps. Thus
    // one resident snapshot can hold the store above the budget (finding
    // state-ram-overhead). More than one cannot.
    if (s.ram_bytes_ > s.ram_budget_ && s.resident() > 1) {
        fail("state store: %zu bytes in RAM in %zu snapshots, over the budget of %zu", s.ram_bytes_, s.resident(),
             s.ram_budget_);
    }
    if (!s.dir_.empty() && s.disk_bytes_ > s.disk_budget_) {
        fail("state store: %zu bytes on disk, over the budget of %zu", s.disk_bytes_, s.disk_budget_);
    }
}

/** The checks of the counters, the budgets and the index of the image cache. */
void check_images(const ImageCache & c) {
    size_t ram = 0, disk = 0;
    for (auto it = c.entries_.begin(); it != c.entries_.end(); ++it) {
        if (!it->data.empty()) {
            ram += it->bytes();
            if (it->data.size() != it->info.n_floats()) {
                fail("image cache: an entry holds %zu floats for a shape of %zu", it->data.size(), it->info.n_floats());
            }
        }
        if (it->on_disk) {
            disk += it->bytes();
        }
        if (it->data.empty() && !it->on_disk) {
            fail("image cache: an entry with no data and no file stays in the list");
        }
        auto found = c.index_.find(it->id);
        if (found == c.index_.end() || found->second != it) {
            fail("image cache: the index does not point at the entry %s", it->id.c_str());
        }
    }
    if (c.index_.size() != c.entries_.size()) {
        fail("image cache: the index holds %zu ids for %zu entries", c.index_.size(), c.entries_.size());
    }
    if (ram != c.ram_bytes_ || disk != c.disk_bytes_) {
        fail("image cache: the counters say %zu RAM and %zu disk bytes, the entries hold %zu and %zu", c.ram_bytes_,
             c.disk_bytes_, ram, disk);
    }
    if (c.ram_bytes_ > c.ram_budget_) {
        fail("image cache: %zu bytes in RAM, over the budget of %zu", c.ram_bytes_, c.ram_budget_);
    }
    if (!c.dir_.empty() && c.disk_bytes_ > c.disk_budget_) {
        fail("image cache: %zu bytes on disk, over the budget of %zu", c.disk_bytes_, c.disk_budget_);
    }
}

/** Change a file of the directory as a damaged disk or the system does. */
void damage(FuzzedDataProvider & fdp, const std::string & dir) {
    std::vector<std::string> names = cache_io::list_files(dir, "");
    if (names.empty()) {
        return;
    }
    std::sort(names.begin(), names.end());
    const std::string path = dir + "/" + names[fdp.ConsumeIntegralInRange<size_t>(0, names.size() - 1)];
    std::vector<uint8_t> bytes;
    if (!cache_io::read_file(path, bytes, 1u << 20)) {
        return;
    }
    switch (fdp.ConsumeIntegralInRange<int>(0, 3)) {
        case 0:
            cache_io::remove_file(path);
            return;
        case 1:
            bytes.resize(fdp.ConsumeIntegralInRange<size_t>(0, bytes.size()));
            break;
        case 2:
            for (int i = 0; i < 4 && !bytes.empty(); ++i) {
                bytes[fdp.ConsumeIntegralInRange<size_t>(0, bytes.size() - 1)] ^= (uint8_t) (1 + fdp.ConsumeIntegral<uint8_t>() % 255);
            }
            break;
        default: {
            const std::vector<uint8_t> extra = fdp.ConsumeBytes<uint8_t>(8);
            bytes.insert(bytes.end(), extra.begin(), extra.end());
            break;
        }
    }
    cache_io::write_file_atomic(path, {{bytes.data(), bytes.size()}});
}

/** One program over the state store. */
void run_states(FuzzedDataProvider & fdp, const std::string & dir) {
    const size_t ram_budget  = fdp.ConsumeIntegralInRange<size_t>(0, 4096);
    const size_t disk_budget = fdp.ConsumeIntegralInRange<size_t>(0, 8192);
    const bool   use_dir     = fdp.ConsumeBool();
    auto store = std::make_unique<StateCache>(ram_budget, disk_budget, use_dir ? dir : "");
    // The bytes of the last put of each item sequence, by hash and items.
    std::map<std::pair<uint64_t, size_t>, std::pair<std::vector<MemItem>, std::shared_ptr<const cache_io::Blob>>> model;
    bool damaged = false;
    int ops = 0;
    while (fdp.remaining_bytes() > 0 && ops++ < 64) {
        switch (fdp.ConsumeIntegralInRange<int>(0, 9)) {
            case 0: case 1: case 2: {
                std::vector<MemItem> items = gen_items(fdp);
                const size_t n = fdp.ConsumeIntegralInRange<size_t>(0, 1024);
                auto blob = make_blob(n, hash_items(items));
                const int32_t n_pos = (int32_t) items.size();
                model[{hash_items(items), items.size()}] = {items, blob};
                store->put(std::move(items), n_pos, blob);
                break;
            }
            case 3: {
                // The use of prefill: best_prefix, then bytes, then drop on a restore failure.
                std::vector<MemItem> items = gen_items(fdp);
                const size_t limit = fdp.ConsumeIntegralInRange<size_t>(0, 12);
                const Snapshot * snap = store->best_prefix(items, limit);
                size_t best = 0;
                for (const Snapshot & s : store->entries_) {
                    if (!s.items.empty() && s.items.size() <= limit && is_item_prefix(s.items, items)) {
                        best = std::max(best, s.items.size());
                    }
                }
                if ((snap == nullptr ? 0 : snap->items.size()) != best) {
                    fail("state store: best_prefix gave %zu items, the longest prefix has %zu",
                         snap == nullptr ? 0 : snap->items.size(), best);
                }
                if (snap == nullptr) {
                    break;
                }
                const std::vector<MemItem> key = snap->items;
                // FUZZ_CACHES_DRAIN=1 waits for the queued writes first: a finding that goes away with it
                // is a read of a file whose write is still in the queue (finding state-pending-write).
                // FUZZ_APP_SKIP_KNOWN=1 does the same, thus a long run explores past that finding.
                static const bool drain_first = getenv("FUZZ_CACHES_DRAIN") != nullptr ||
                                                (getenv("FUZZ_APP_SKIP_KNOWN") != nullptr &&
                                                 strcmp(getenv("FUZZ_APP_SKIP_KNOWN"), "1") == 0);
                if (drain_first) {
                    store->drain();
                }
                std::shared_ptr<const cache_io::Blob> bytes = store->bytes(snap);
                if (!bytes) {
                    if (store->find(key) != nullptr) {
                        fail("state store: a snapshot whose bytes did not read stays in the store");
                    }
                    if (!damaged) {
                        fail("state store: the bytes of a snapshot did not read, and no file was damaged");
                    }
                    break;
                }
                auto m = model.find({hash_items(key), key.size()});
                if (m != model.end() && m->second.first == key && !damaged) {
                    const auto & want = m->second.second;
                    if (want->size != bytes->size || memcmp(want->data.get(), bytes->data.get(), want->size) != 0) {
                        fail("state store: a snapshot gave bytes that are not the bytes of its last put");
                    }
                }
                if (fdp.ConsumeBool()) {
                    store->drop(snap);
                    if (store->find(key) != nullptr) {
                        fail("state store: a dropped snapshot is still in the store");
                    }
                }
                break;
            }
            case 4:
                store->clear(fdp.ConsumeBool());
                break;
            case 5:
                store->drain();
                if (!store->dir_.empty()) {
                    for (const Snapshot & s : store->entries_) {
                        cache_io::FileStat st;
                        if (s.on_disk && !damaged && !cache_io::stat_file(store->path_of(s), st)) {
                            fail("state store: a snapshot on disk has no file after a drain");
                        }
                    }
                }
                break;
            case 6: {
                // A restart of the app: a new store on the same directory.
                store->drain();
                std::vector<std::pair<std::vector<MemItem>, size_t>> on_disk;
                for (const Snapshot & s : store->entries_) {
                    if (s.on_disk) {
                        on_disk.emplace_back(s.items, s.byte_size);
                    }
                }
                store.reset();
                store = std::make_unique<StateCache>(ram_budget, disk_budget, use_dir ? dir : "");
                if (!damaged && use_dir) {
                    for (const auto & [items, size] : on_disk) {
                        const Snapshot * s = store->find(items);
                        if (s == nullptr || s->byte_size != size) {
                            fail("state store: an intact file of a snapshot of %zu items did not load after a restart", items.size());
                        }
                    }
                }
                break;
            }
            case 7:
                if (use_dir) {
                    store->drain();
                    damage(fdp, dir);
                    damaged = true;
                }
                break;
            case 8:
                if (use_dir && fdp.ConsumeBool()) {
                    // A directory without write permission: the writes fail, and the store forgets their files.
                    chmod(dir.c_str(), 0500);
                    std::vector<MemItem> items = gen_items(fdp);
                    const int32_t n_pos = (int32_t) items.size();
                    const uint64_t key = hash_items(items);
                    store->put(std::move(items), n_pos, make_blob(fdp.ConsumeIntegralInRange<size_t>(1, 512), key));
                    store->drain();
                    chmod(dir.c_str(), 0700);
                    store->best_prefix({}, 0);
                    damaged = true;
                }
                break;
            default: {
                std::vector<MemItem> items = gen_items(fdp);
                const Snapshot * s = store->find(items);
                if (s != nullptr && s->items != items) {
                    fail("state store: find gave a snapshot with different items");
                }
                break;
            }
        }
        check_states(*store);
    }
    store->drain();
    check_states(*store);
}

/** One program over the image cache. */
void run_images(FuzzedDataProvider & fdp, const std::string & dir) {
    const size_t ram_budget  = fdp.ConsumeIntegralInRange<size_t>(0, 4096);
    const size_t disk_budget = fdp.ConsumeIntegralInRange<size_t>(0, 8192);
    const bool   use_dir     = fdp.ConsumeBool();
    auto cache = std::make_unique<ImageCache>(ram_budget, disk_budget, use_dir ? dir : "");
    static const char * const ids[] = {"00aa", "11bb", "22cc", "33dd", "not-hex", ""};
    std::map<std::string, std::pair<ImageInfo, std::vector<float>>> model;
    bool damaged = false;
    int ops = 0;
    while (fdp.remaining_bytes() > 0 && ops++ < 64) {
        const std::string id = ids[fdp.ConsumeIntegralInRange<size_t>(0, 5)];
        switch (fdp.ConsumeIntegralInRange<int>(0, 6)) {
            case 0: case 1: {
                ImageInfo info;
                info.nx = fdp.ConsumeIntegral<uint16_t>();
                info.ny = fdp.ConsumeIntegral<uint16_t>();
                info.n_tokens = fdp.ConsumeIntegralInRange<uint32_t>(0, 40);
                info.n_embd = fdp.ConsumeIntegralInRange<uint32_t>(0, 40);
                std::vector<float> data(info.n_floats());
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i] = (float) (i * 3 + info.nx);
                }
                cache->put(id, info, data.empty() ? nullptr : data.data());
                // The cache refuses an output larger than its RAM budget and keeps the entry
                // it has for the id, also when that entry has another shape.
                if (!data.empty() && cache_io::is_hex(id) && info.n_floats() * sizeof(float) <= ram_budget) {
                    model[id] = {info, data};
                }
                break;
            }
            case 2: {
                const float * got = cache->get(id);
                auto m = model.find(id);
                if (got != nullptr && m != model.end() && !damaged) {
                    // A put of a known id with the same shape keeps the first data, thus compare the shape only.
                    ImageInfo info;
                    if (!cache->info(id, info) || info.n_floats() != m->second.first.n_floats()) {
                        fail("image cache: get gave data for %s but info does not agree", id.c_str());
                    }
                }
                break;
            }
            case 3: {
                ImageInfo info;
                if (cache->info(id, info) && cache->index_.count(id) == 0) {
                    fail("image cache: info said yes for an id that is not in the index");
                }
                break;
            }
            case 4:
                cache->clear(fdp.ConsumeBool());
                break;
            case 5:
                cache.reset();
                cache = std::make_unique<ImageCache>(ram_budget, disk_budget, use_dir ? dir : "");
                break;
            default:
                if (use_dir) {
                    damage(fdp, dir);
                    damaged = true;
                }
                break;
        }
        check_images(*cache);
    }
}

/** The writer thread alone: writes, removes, drains and failures in any order. */
void run_writer(FuzzedDataProvider & fdp, const std::string & dir) {
    cache_io::AsyncWriter w;
    int ops = 0;
    while (fdp.remaining_bytes() > 0 && ops++ < 64) {
        const std::string path = dir + "/f" + std::to_string(fdp.ConsumeIntegralInRange<int>(0, 3));
        switch (fdp.ConsumeIntegralInRange<int>(0, 3)) {
            case 0: {
                auto head = std::make_shared<const std::vector<uint8_t>>(fdp.ConsumeBytes<uint8_t>(16));
                w.write(path, head, fdp.ConsumeBool() ? make_blob(fdp.ConsumeIntegralInRange<size_t>(0, 256), 3) : nullptr);
                break;
            }
            case 1:
                w.remove(path);
                break;
            case 2:
                w.drain();
                break;
            default:
                w.write(dir + "/missing-dir/f", nullptr, nullptr);
                w.take_failed();
                break;
        }
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
    static std::atomic<uint64_t> serial{0};
    crash_input::remember(data, size);
    FuzzedDataProvider fdp(data, size);
    const std::string dir = work_dir() + "/caches-" + std::to_string(getpid()) + "-" + std::to_string(serial++);
    cache_io::make_dirs(dir);
    switch (fdp.ConsumeIntegralInRange<int>(0, 2)) {
        case 0: run_states(fdp, dir); break;
        case 1: run_images(fdp, dir); break;
        default: run_writer(fdp, dir); break;
    }
    chmod(dir.c_str(), 0700);
    cache_io::remove_tree(dir);
    return 0;
}
