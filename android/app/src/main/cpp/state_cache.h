/**
 * The store of sequence states. A snapshot is the byte state of sequence 0
 * of a context (llama_state_seq_get_data) with the exact item sequence
 * that the state holds. The store keeps the most recently used snapshots
 * in RAM inside a byte budget, and the same snapshots on disk inside a
 * second budget, thus a conversation continues after a restart of the app.
 *
 * The bytes are device independent: the prefill context and the decode
 * context of the hybrid backend read the same snapshot.
 *
 * No llama.cpp and no Android dependency: a host test compiles this file.
 * All calls come from one thread. The disk writes run on a second thread
 * that only the destructor waits for.
 */
#pragma once

#include "cache_io.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <vector>

/** A token identifier, the same type as llama_token. */
using mem_token = int32_t;

/** The token of an image item. The same value as LLAMA_TOKEN_NULL. */
constexpr mem_token kMemTokenNull = -1;

/** One unit of the model memory: a text token, or an image chunk identified by the hash of its file bytes. */
struct MemItem {
    mem_token   token = kMemTokenNull;
    std::string image_id;

    bool operator==(const MemItem & o) const { return token == o.token && image_id == o.image_id; }
    bool operator!=(const MemItem & o) const { return !(*this == o); }
};

/** True when a is a prefix of b. O(a). */
bool is_item_prefix(const std::vector<MemItem> & a, const std::vector<MemItem> & b);

/** The hash of an item sequence. Two sequences with the same hash are compared item by item. O(n). */
uint64_t hash_items(const std::vector<MemItem> & items);

/** One snapshot of the store. The pointer to it stays valid until the next call that changes the store. */
struct Snapshot {
    /** The exact items that the state holds, in order. */
    std::vector<MemItem> items;
    /** The number of positions that the state holds. After an image, M-RoPE gives fewer positions than tokens. */
    int32_t n_pos = 0;
    /** hash_items(items). */
    uint64_t key = 0;
    /** The state bytes when the snapshot is in RAM, else null. */
    std::shared_ptr<const cache_io::Blob> bytes;
    /** True when the file of the snapshot exists, or its write is queued. */
    bool on_disk = false;
    /** The size of the state bytes, also when only the file holds them. */
    size_t byte_size = 0;
    /** The write order on disk: the file modification time at the scan, or a later counter. The oldest goes first. */
    int64_t disk_stamp = 0;
};

class StateCache {
public:
    /**
     * @param ram_budget   The maximum bytes of the snapshots in RAM
     * @param disk_budget  The maximum bytes of the snapshot files, 0 for no disk tier
     * @param dir          The directory of the files, or empty for no disk tier. The
     *                     files that are there already become snapshots of the store.
     */
    StateCache(size_t ram_budget, size_t disk_budget, std::string dir);
    ~StateCache();

    StateCache(const StateCache &) = delete;
    StateCache & operator=(const StateCache &) = delete;

    /**
     * The snapshot with the most items that is a prefix of the items and
     * has at most limit items, or null. Among equal lengths, the most
     * recently used one. O(snapshots x limit).
     */
    const Snapshot * best_prefix(const std::vector<MemItem> & items, size_t limit);

    /**
     * The bytes of a snapshot, from RAM or read from its file into RAM.
     * When the file is not there yet, the read waits one time for the
     * queued writes. The snapshot becomes the most recently used one.
     * Returns null when the file is not readable, and then the snapshot is
     * gone from the store.
     */
    std::shared_ptr<const cache_io::Blob> bytes(const Snapshot * snap);

    /** The snapshot with exactly these items, or null. O(1) on the usual case. */
    const Snapshot * find(const std::vector<MemItem> & items) const;

    /**
     * Keep a snapshot. A snapshot with the same items keeps its file and
     * takes these bytes into RAM. The RAM tier and the disk tier release
     * their least recently used snapshots to stay inside their budgets.
     */
    void put(std::vector<MemItem> items, int32_t n_pos, std::shared_ptr<const cache_io::Blob> bytes);

    /** Remove a snapshot from RAM and disk, for example after its bytes did not restore. */
    void drop(const Snapshot * snap);

    /** Remove every snapshot from RAM, and with disk_too also every file. */
    void clear(bool disk_too);

    /** Wait until the queued file writes are done. For tests. */
    void drain();

    size_t ram_bytes() const { return ram_bytes_; }
    size_t disk_bytes() const { return disk_bytes_; }
    size_t count() const { return entries_.size(); }
    /** The number of snapshots in RAM. */
    size_t resident() const;

private:
    using List = std::list<Snapshot>;

    std::string path_of(const Snapshot & snap) const;
    void        scan_dir();
    void        make_resident(List::iterator it, std::shared_ptr<const cache_io::Blob> bytes);
    void        evict_ram(const Snapshot * keep);
    void        evict_disk(const Snapshot * keep);
    void        write_file(List::iterator it, const std::shared_ptr<const cache_io::Blob> & bytes);
    void        forget_failed_writes();
    List::iterator iterator_of(const Snapshot * snap);
    List::iterator erase(List::iterator it, bool remove_file);

    /** The snapshots, the most recently used first. */
    List        entries_;
    size_t      ram_budget_;
    size_t      disk_budget_;
    std::string dir_;
    size_t      ram_bytes_  = 0;
    size_t      disk_bytes_ = 0;
    int64_t     stamp_      = 0;
    std::unique_ptr<cache_io::AsyncWriter> writer_;
};
