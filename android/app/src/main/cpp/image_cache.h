/**
 * The cache of encoded images. An entry is the output of the vision
 * encoder for one image (n_tokens x n_embd floats) with the dimensions of
 * the bitmap, keyed by the SHA-256 of the image file bytes, which is also
 * the id that mtmd gives the bitmap, and by the shape of the output. One id
 * has one entry: an entry of another shape goes. The most recently used entries stay
 * in RAM inside a byte budget, and every entry stays on disk inside a
 * second budget.
 *
 * With the dimensions of a known image, the engine tokenizes a
 * placeholder bitmap, thus the JPEG is not decoded and not preprocessed
 * again.
 *
 * No llama.cpp and no Android dependency. All calls come from one thread.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

/** What the engine needs about an image without its bytes. */
struct ImageInfo {
    uint32_t nx       = 0;
    uint32_t ny       = 0;
    uint32_t n_tokens = 0;
    uint32_t n_embd   = 0;

    /** The number of floats of the encoder output. */
    size_t n_floats() const { return (size_t) n_tokens * n_embd; }
};

class ImageCache {
public:
    /**
     * @param ram_budget   The maximum bytes of the encoder outputs in RAM
     * @param disk_budget  The maximum bytes of the files, 0 for no disk tier
     * @param dir          The directory of the files, or empty for no disk tier. The
     *                     files that are there already become entries of the cache.
     */
    ImageCache(size_t ram_budget, size_t disk_budget, std::string dir);

    /**
     * The dimensions of a known image, from RAM or from a file that is
     * still there. The entry becomes the most recent of the two tiers,
     * thus the evictions of the turn take other entries first. Returns
     * false for an unknown id, or when the file is gone, and then the
     * entry is gone too.
     */
    bool info(const std::string & id, ImageInfo & out);

    /**
     * The encoder output of an image, from RAM or read from its file into
     * RAM. The shape is part of the key: the caller gives the n_tokens and
     * n_embd of its image chunk, because it reads that number of floats
     * from the pointer. The pointer stays valid until the next call that
     * changes the cache. Returns null for an unknown id, for an entry of
     * another shape, or when the file is not readable. In the last two
     * conditions, the entry and its file are gone.
     */
    const float * get(const std::string & id, uint32_t n_tokens, uint32_t n_embd);

    /** Keep the encoder output of an image. The data is copied. An output larger than the RAM budget is not kept. */
    void put(const std::string & id, const ImageInfo & info, const float * data);

    /** Remove the entry of the id from RAM and its file from the disk. An unknown id does nothing. */
    void drop(const std::string & id);

    /** Remove every entry from RAM, and with disk_too also every file. */
    void clear(bool disk_too);

    size_t ram_bytes() const { return ram_bytes_; }
    size_t disk_bytes() const { return disk_bytes_; }
    size_t count() const { return entries_.size(); }

private:
    struct Entry {
        std::string        id;
        ImageInfo          info;
        std::vector<float> data;
        bool               on_disk    = false;
        int64_t            disk_stamp = 0;

        size_t bytes() const { return info.n_floats() * sizeof(float); }
    };
    using List = std::list<Entry>;

    std::string path_of(const std::string & id) const;
    void        scan_dir();
    bool        read_file(Entry & entry) const;
    bool        write_file(const Entry & entry) const;
    void        evict_ram(const Entry * keep);
    void        evict_disk(const Entry * keep);
    void        erase(List::iterator it, bool remove_file);

    /** The entries, the most recently used first. */
    List                                            entries_;
    std::unordered_map<std::string, List::iterator> index_;
    size_t                                          ram_budget_;
    size_t                                          disk_budget_;
    std::string                                     dir_;
    size_t                                          ram_bytes_  = 0;
    size_t                                          disk_bytes_ = 0;
    int64_t                                         stamp_      = 0;
};
