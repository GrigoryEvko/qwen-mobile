/**
 * The host test of the engine caches: the prefix match and the LRU of the
 * snapshot store, its disk tier, the image cache, the SHA-256 that gives
 * the image ids, and the checksum of the cache files. No llama.cpp is
 * necessary.
 *
 *   cmake -S android/app/src/test/cpp -B /tmp/cache_test && cmake --build /tmp/cache_test && /tmp/cache_test/cache_test
 */
#include "cache_io.h"
#include "image_cache.h"
#include "state_cache.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                                   \
        }                                                                                 \
    } while (0)

/** A temporary directory that the destructor removes. */
struct TempDir {
    std::string path;

    TempDir() {
        char pattern[] = "/tmp/qwen_cache_test_XXXXXX";
        path = mkdtemp(pattern);
    }

    ~TempDir() { cache_io::remove_tree(path); }
};

/** Text tokens t0, t0 + 1, ... t0 + n - 1. */
std::vector<MemItem> tokens(int t0, int n) {
    std::vector<MemItem> out;
    for (int i = 0; i < n; ++i) {
        out.push_back(MemItem{t0 + i, {}});
    }
    return out;
}

/** A blob of n bytes with the value v in each byte. */
std::shared_ptr<const cache_io::Blob> blob(size_t n, uint8_t v) {
    auto b = std::make_shared<cache_io::Blob>(n);
    memset(b->data.get(), v, n);
    return b;
}

/** Add 1 to the byte at the offset of the file. A negative offset counts from the end. */
bool damage_byte(const std::string & path, long offset) {
    std::vector<uint8_t> raw;
    if (!cache_io::read_file(path, raw, 1u << 20) || raw.empty()) {
        return false;
    }
    const long at = offset < 0 ? (long) raw.size() + offset : offset;
    if (at < 0 || at >= (long) raw.size()) {
        return false;
    }
    raw[(size_t) at] += 1;
    return cache_io::write_file_atomic(path, {{raw.data(), raw.size()}});
}

void test_checksum() {
    // The vectors of the reference XXH64 for three seeds: the empty input, a
    // short input, and 100 bytes, which take the steps of 32 bytes.
    std::vector<uint8_t> d(100);
    for (size_t i = 0; i < d.size(); ++i) {
        d[i] = (uint8_t) (i * 7 + 3);
    }
    CHECK(cache_io::checksum64(nullptr, 0) == 0xef46db3751d8e999ull);
    CHECK(cache_io::checksum64("123456789", 9) == 0x8cb841db40e6ae83ull);
    CHECK(cache_io::checksum64(d.data(), d.size()) == 0xa61f8d4c170fe531ull);
    CHECK(cache_io::checksum64(nullptr, 0, 1) == 0xd5afba1336a3be4bull);
    CHECK(cache_io::checksum64("123456789", 9, 1) == 0x1a4cc2c9e8079790ull);
    CHECK(cache_io::checksum64(d.data(), d.size(), 1) == 0x8d8957e68f02c7ceull);
    const uint64_t seed = 0x0123456789abcdefull;
    CHECK(cache_io::checksum64(nullptr, 0, seed) == 0x51e24c0e9077a48cull);
    CHECK(cache_io::checksum64("123456789", 9, seed) == 0x3b48dc2448b0c9efull);
    CHECK(cache_io::checksum64(d.data(), d.size(), seed) == 0xfe1fce732c97c212ull);
}

void test_sha256() {
    // The vectors of FIPS 180-4, and a two-block message.
    CHECK(cache_io::sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(cache_io::sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const char * two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(cache_io::sha256_hex(two, strlen(two)) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // 56 bytes: the length no longer fits in the first block.
    std::string s56(56, 'a');
    CHECK(cache_io::sha256_hex(s56.data(), 56) == "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
    std::string s1000(1000, 'a');
    CHECK(cache_io::sha256_hex(s1000.data(), 1000) == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    CHECK(cache_io::is_hex("0af9") && !cache_io::is_hex("") && !cache_io::is_hex("0AF9") && !cache_io::is_hex("../x"));
}

void test_prefix_match() {
    StateCache cache(1u << 20, 0, "");
    cache.put(tokens(0, 10), 10, blob(100, 1));
    cache.put(tokens(0, 20), 20, blob(100, 2));
    cache.put(tokens(0, 15), 15, blob(100, 3));
    // A sequence that shares 12 items with all three snapshots, then differs.
    std::vector<MemItem> other = tokens(0, 12);
    for (int i = 0; i < 20; ++i) other.push_back(MemItem{1000 + i, {}});
    const Snapshot * s = cache.best_prefix(other, other.size());
    CHECK(s != nullptr && s->items.size() == 10);
    // The full sequence prefers the longest snapshot.
    s = cache.best_prefix(tokens(0, 30), 30);
    CHECK(s != nullptr && s->items.size() == 20 && s->n_pos == 20);
    // The limit cuts the longer snapshots.
    s = cache.best_prefix(tokens(0, 30), 15);
    CHECK(s != nullptr && s->items.size() == 15);
    s = cache.best_prefix(tokens(0, 30), 9);
    CHECK(s == nullptr);
    // A different first token matches nothing.
    s = cache.best_prefix(tokens(5, 30), 30);
    CHECK(s == nullptr);
    // An image item is part of the key.
    std::vector<MemItem> with_image = tokens(0, 5);
    with_image.push_back(MemItem{kMemTokenNull, "abc"});
    with_image.push_back(MemItem{7, {}});
    cache.put(with_image, 6, blob(100, 4));
    std::vector<MemItem> query = tokens(0, 5);
    query.push_back(MemItem{kMemTokenNull, "abd"});
    query.push_back(MemItem{7, {}});
    query.push_back(MemItem{8, {}});
    s = cache.best_prefix(query, query.size());
    CHECK(s == nullptr);
    query[5].image_id = "abc";
    s = cache.best_prefix(query, query.size());
    CHECK(s != nullptr && s->items == with_image);
    CHECK(cache.find(with_image) == s);
    CHECK(cache.find(tokens(0, 11)) == nullptr);
    CHECK(cache.count() == 4);
}

void test_lru() {
    // Each snapshot costs 1000 bytes plus its items. The budget holds two.
    const size_t item = 10 * sizeof(MemItem);
    StateCache cache(2 * (1000 + item), 0, "");
    cache.put(tokens(0, 10), 10, blob(1000, 1));
    cache.put(tokens(100, 10), 10, blob(1000, 2));
    CHECK(cache.count() == 2 && cache.resident() == 2);
    // A use of the first one makes the second one the least recently used.
    const Snapshot * first = cache.best_prefix(tokens(0, 10), 10);
    CHECK(first != nullptr);
    auto bytes = cache.bytes(first);
    CHECK(bytes && bytes->data[0] == 1);
    cache.put(tokens(200, 10), 10, blob(1000, 3));
    CHECK(cache.count() == 2);
    CHECK(cache.best_prefix(tokens(100, 10), 10) == nullptr);
    CHECK(cache.best_prefix(tokens(0, 10), 10) != nullptr);
    CHECK(cache.best_prefix(tokens(200, 10), 10) != nullptr);
    // The same key again does not add an entry.
    cache.put(tokens(0, 10), 10, blob(1000, 1));
    CHECK(cache.count() == 2 && cache.ram_bytes() <= 2 * (1000 + item));
    // A snapshot above the budget is not kept.
    cache.put(tokens(300, 10), 10, blob(5000, 4));
    CHECK(cache.count() == 2);
    // A drop removes the entry.
    cache.drop(cache.best_prefix(tokens(0, 10), 10));
    CHECK(cache.count() == 1 && cache.best_prefix(tokens(0, 10), 10) == nullptr);
    cache.clear(false);
    CHECK(cache.count() == 0 && cache.ram_bytes() == 0);
}

void test_disk() {
    TempDir dir;
    std::vector<MemItem> with_image = tokens(0, 5);
    with_image.push_back(MemItem{kMemTokenNull, std::string(64, 'a')});
    with_image.push_back(MemItem{9, {}});
    {
        StateCache cache(1u << 20, 1u << 20, dir.path);
        cache.put(tokens(0, 10), 10, blob(3000, 1));
        cache.put(with_image, 6, blob(2000, 2));
        cache.drain();
        CHECK(cache.disk_bytes() == 5000);
        CHECK(cache_io::list_files(dir.path, ".snap").size() == 2);
    }
    {
        // A new store on the same directory holds the snapshots without their bytes.
        StateCache cache(1u << 20, 1u << 20, dir.path);
        CHECK(cache.count() == 2 && cache.resident() == 0 && cache.disk_bytes() == 5000);
        const Snapshot * s = cache.best_prefix(with_image, with_image.size());
        CHECK(s != nullptr && s->n_pos == 6 && s->items == with_image);
        auto bytes = cache.bytes(s);
        CHECK(bytes && bytes->size == 2000 && bytes->data[1999] == 2);
        CHECK(cache.resident() == 1);
        // A damaged file goes away on its read.
        const Snapshot * t = cache.best_prefix(tokens(0, 10), 10);
        CHECK(t != nullptr);
        const std::string path = dir.path + "/" + cache_io::hex64(t->key) + ".snap";
        std::vector<uint8_t> raw;
        CHECK(cache_io::read_file(path, raw, 1u << 20));
        raw.resize(raw.size() - 1);
        CHECK(cache_io::write_file_atomic(path, {{raw.data(), raw.size()}}));
        // The scan of a new store rejects the short file.
        StateCache again(1u << 20, 1u << 20, dir.path);
        CHECK(again.count() == 1);
        CHECK(cache_io::list_files(dir.path, ".snap").size() == 1);
    }
    {
        // The disk budget releases the oldest file.
        StateCache cache(1u << 20, 4500, dir.path);
        cache.put(tokens(50, 10), 10, blob(3000, 3));
        cache.drain();
        CHECK(cache.disk_bytes() == 3000 && cache.count() == 1);
        CHECK(cache_io::list_files(dir.path, ".snap").size() == 1);
        cache.clear(true);
        cache.drain();
        CHECK(cache.count() == 0 && cache_io::list_files(dir.path, ".snap").empty());
    }
    {
        // A RAM tier smaller than a snapshot still writes the file, and a read gives the bytes.
        StateCache cache(100, 1u << 20, dir.path);
        cache.put(tokens(0, 10), 10, blob(3000, 5));
        cache.drain();
        CHECK(cache.count() == 1 && cache.resident() == 0 && cache.disk_bytes() == 3000);
        auto bytes = cache.bytes(cache.best_prefix(tokens(0, 10), 10));
        CHECK(bytes && bytes->size == 3000 && bytes->data[0] == 5);
        CHECK(cache.resident() == 0);
    }
}

void test_images() {
    TempDir dir;
    const std::string id_a(64, 'a');
    const std::string id_b(64, 'b');
    const std::string id_c(64, 'c');
    ImageInfo info;
    info.nx = 640; info.ny = 480; info.n_tokens = 100; info.n_embd = 8;
    std::vector<float> a(800, 1.0f), b(800, 2.0f), c(800, 3.0f);
    {
        ImageCache cache(2 * 3200, 1u << 20, dir.path);
        cache.put(id_a, info, a.data());
        cache.put(id_b, info, b.data());
        CHECK(cache.count() == 2 && cache.ram_bytes() == 6400 && cache.disk_bytes() == 6400);
        CHECK(cache.get(id_a, 100, 8) != nullptr && cache.get(id_a, 100, 8)[0] == 1.0f);
        // The third entry pushes b out of RAM, and b stays on disk.
        cache.put(id_c, info, c.data());
        CHECK(cache.count() == 3 && cache.ram_bytes() == 6400);
        const float * pb = cache.get(id_b, 100, 8);
        CHECK(pb != nullptr && pb[799] == 2.0f);
        ImageInfo got;
        CHECK(cache.info(id_c, got) && got.nx == 640 && got.n_tokens == 100);
        CHECK(!cache.info(std::string(64, 'd'), got));
        cache.put("../evil", info, a.data());
        CHECK(cache.count() == 3);
    }
    {
        ImageCache cache(1u << 20, 1u << 20, dir.path);
        CHECK(cache.count() == 3 && cache.ram_bytes() == 0);
        // A request of another shape gets no data, and the entry goes.
        CHECK(cache.get(id_a, 50, 16) == nullptr && cache.count() == 2);
        const float * pc = cache.get(id_c, 100, 8);
        CHECK(pc != nullptr && pc[0] == 3.0f && cache.ram_bytes() == 3200);
        cache.clear(true);
        CHECK(cache.count() == 0 && cache_io::list_files(dir.path, ".embd").empty());
    }
}

void test_damaged_files() {
    TempDir dir;
    const std::vector<MemItem> items = tokens(0, 10);
    const std::string snap_path = dir.path + "/" + cache_io::hex64(hash_items(items)) + ".snap";
    // The scan of a directory reads only the header and the items of a file. Thus a damage
    // of n_pos (offset 20), of the reserved field (28) or of the state bytes (the last byte)
    // shows at the read of the bytes.
    for (long offset : {20L, 28L, -1L}) {
        {
            StateCache cache(1u << 20, 1u << 20, dir.path);
            cache.put(items, 10, blob(3000, 1));
        }
        CHECK(damage_byte(snap_path, offset));
        StateCache cache(1u << 20, 1u << 20, dir.path);
        const Snapshot * s = cache.find(items);
        CHECK(s != nullptr && !cache.bytes(s));
        CHECK(cache.find(items) == nullptr);
        const std::vector<uint64_t> damaged = cache.take_damaged();
        CHECK(damaged.size() == 1 && damaged[0] == hash_items(items));
        CHECK(cache.take_damaged().empty());
        cache.drain();
        CHECK(cache_io::list_files(dir.path, ".snap").empty());
    }
    {
        // A file of another version goes at the scan.
        {
            StateCache cache(1u << 20, 1u << 20, dir.path);
            cache.put(items, 10, blob(3000, 1));
        }
        CHECK(damage_byte(snap_path, 4));
        StateCache cache(1u << 20, 1u << 20, dir.path);
        CHECK(cache.count() == 0 && cache.removed_at_scan() == 1);
        CHECK(cache_io::list_files(dir.path, ".snap").empty());
    }
    // A damage of the side nx of the bitmap (offset 8) or of the last float.
    const std::string id(64, 'e');
    ImageInfo info;
    info.nx = 640; info.ny = 480; info.n_tokens = 100; info.n_embd = 8;
    const std::vector<float> data(800, 1.5f);
    for (long offset : {8L, -1L}) {
        {
            ImageCache cache(1u << 20, 1u << 20, dir.path);
            cache.put(id, info, data.data());
        }
        CHECK(damage_byte(dir.path + "/" + id + ".embd", offset));
        ImageCache cache(1u << 20, 1u << 20, dir.path);
        CHECK(cache.count() == 1 && cache.get(id, 100, 8) == nullptr);
        CHECK(cache.count() == 0 && cache_io::list_files(dir.path, ".embd").empty());
        const std::vector<std::string> damaged = cache.take_damaged();
        CHECK(damaged.size() == 1 && damaged[0] == id);
    }
}

} // namespace

int main() {
    test_sha256();
    test_checksum();
    test_prefix_match();
    test_lru();
    test_disk();
    test_images();
    test_damaged_files();
    if (g_failed != 0) {
        std::fprintf(stderr, "%d checks failed\n", g_failed);
        return 1;
    }
    std::printf("cache_test: all checks passed\n");
    return 0;
}
