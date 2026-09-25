/**
 * The rows of the token embedding of a GGUF model file (token_embd.weight),
 * read from the file for each call and converted to float with the
 * to_float function of the type: the values that the GET_ROWS op of the token
 * path and the host lookup of llama.cpp (LLAMA_EMBD_LOOKUP_HOST) give. The
 * engine puts the text tokens before an image into the embedding batch of
 * that image with them, thus the text and the image take one pass over the
 * weights.
 *
 * No Android dependency. All calls come from one thread.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

class TokenRows {
public:
    /**
     * Open the file and find its token embedding. ok() is false when the file
     * does not open or has no token embedding of a type with a float
     * conversion. O(size of the metadata of the file).
     */
    explicit TokenRows(const std::string & path);
    ~TokenRows();

    TokenRows(const TokenRows &)             = delete;
    TokenRows & operator=(const TokenRows &) = delete;

    /** True when read() can give rows. */
    bool ok() const { return fd_ >= 0; }

    /** The floats of one row, and the number of rows (the vocabulary). */
    int64_t n_embd() const { return n_embd_; }
    int64_t n_rows() const { return n_rows_; }

    /** The reason when ok() is false. */
    const std::string & error() const { return error_; }

    /**
     * Write the row of each token to out: n rows of n_embd() floats, with
     * stride floats from the start of one row to the next (stride >= n_embd()).
     * Returns false when a token is out of range or a read fails, and then out
     * holds no complete result. O(n x row size).
     */
    bool read(const int32_t * tokens, size_t n, float * out, size_t stride) const;

private:
    int         fd_       = -1;
    int         type_     = -1;  // the ggml_type of the rows
    int64_t     n_embd_   = 0;
    int64_t     n_rows_   = 0;
    size_t      row_size_ = 0;   // bytes of one row in the file
    uint64_t    offset_   = 0;   // the file offset of row 0
    std::string error_;
};
