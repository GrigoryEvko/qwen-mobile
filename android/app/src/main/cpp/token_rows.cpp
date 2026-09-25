#include "token_rows.h"

#include "ggml.h"
#include "gguf.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <vector>

TokenRows::TokenRows(const std::string & path) {
    ggml_context *    meta   = nullptr;
    gguf_init_params  params = {/* .no_alloc = */ true, /* .ctx = */ &meta};
    gguf_context *    gguf   = gguf_init_from_file(path.c_str(), params);
    if (gguf == nullptr) {
        error_ = "the file " + path + " is not a GGUF file";
        return;
    }
    const int64_t id = gguf_find_tensor(gguf, "token_embd.weight");
    const ggml_tensor * t = id >= 0 && meta != nullptr ? ggml_get_tensor(meta, "token_embd.weight") : nullptr;
    if (t == nullptr || ggml_n_dims(t) != 2) {
        error_ = "the file " + path + " has no two-dimensional token_embd.weight";
    } else if (t->type != GGML_TYPE_F32 && ggml_get_type_traits(t->type)->to_float == nullptr) {
        error_ = std::string("the token embedding has the type ") + ggml_type_name(t->type) + ", which has no float conversion";
    } else {
        type_     = (int) t->type;
        n_embd_   = t->ne[0];
        n_rows_   = t->ne[1];
        row_size_ = ggml_row_size(t->type, t->ne[0]);
        offset_   = (uint64_t) gguf_get_data_offset(gguf) + (uint64_t) gguf_get_tensor_offset(gguf, id);
    }
    gguf_free(gguf);
    if (meta != nullptr) {
        ggml_free(meta);
    }
    if (type_ < 0) {
        return;
    }
    fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        error_ = "the file " + path + " did not open: " + strerror(errno);
    }
}

TokenRows::~TokenRows() {
    if (fd_ >= 0) {
        close(fd_);
    }
}

bool TokenRows::read(const int32_t * tokens, size_t n, float * out, size_t stride) const {
    if (fd_ < 0 || stride < (size_t) n_embd_) {
        return false;
    }
    const ggml_type type = (ggml_type) type_;
    std::vector<uint8_t> raw(row_size_);
    for (size_t i = 0; i < n; ++i) {
        if (tokens[i] < 0 || tokens[i] >= n_rows_) {
            return false;
        }
        // A read can return fewer bytes than asked, for example after a signal.
        const uint64_t at   = offset_ + (uint64_t) tokens[i] * row_size_;
        size_t         done = 0;
        while (done < row_size_) {
            const ssize_t got = pread(fd_, raw.data() + done, row_size_ - done, (off_t) (at + done));
            if (got < 0 && errno == EINTR) {
                continue;
            }
            if (got <= 0) {
                return false;
            }
            done += (size_t) got;
        }
        float * dst = out + i * stride;
        if (type == GGML_TYPE_F32) {
            memcpy(dst, raw.data(), row_size_);
        } else {
            ggml_get_type_traits(type)->to_float(raw.data(), dst, n_embd_);
        }
    }
    return true;
}
