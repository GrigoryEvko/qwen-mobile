// hexhost_repack_bytes: the repacked bytes of the K-quant forms of the tensors of a model.
//
// The tool reads each 2D F16, BF16 or F32 tensor of a GGUF file whose rows are a whole number of
// 256-element blocks, quantizes it to Q4_K and to Q6_K with ggml_quantize_chunk, writes it into a
// weight buffer of the host part of the Hexagon backend (the repack into the tiled layout of the
// DSP), and reads it back. It prints one line for each tensor and type: a hash of the tile bytes
// and a hash of the read back. Two builds of the tool against two llama.cpp trees give the same
// lines when a change of the repack keeps the bytes of real weights.
//
//   hexhost_repack_bytes MODEL.gguf [MAX_ELEMENTS]
//
// MAX_ELEMENTS skips a tensor with more elements (default: no limit). O(total elements).

#include "fake_dsp.h"
#include "hexhost.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// FNV-1a over a byte range. O(n).
uint64_t fnv1a(const uint8_t * p, size_t n, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; i++) {
        h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}

// Reads the values of one tensor of the file as floats. Gives false when the file cannot be read.
bool read_floats(FILE * f, size_t offset, ggml_type type, size_t n, std::vector<float> & out) {
    std::vector<uint8_t> raw(n * ggml_type_size(type));
    if (fseeko(f, (off_t) offset, SEEK_SET) != 0 || fread(raw.data(), 1, raw.size(), f) != raw.size()) {
        return false;
    }
    out.resize(n);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), raw.data(), raw.size());
    } else if (type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), out.data(), (int64_t) n);
    } else {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw.data(), out.data(), (int64_t) n);
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s MODEL.gguf [MAX_ELEMENTS]\n", argv[0]);
        return 2;
    }
    const int64_t max_elements = argc > 2 ? strtoll(argv[2], nullptr, 0) : INT64_MAX;

    gguf_init_params gp  = { true, nullptr };
    gguf_context *   ctx = gguf_init_from_file(argv[1], gp);
    if (!ctx) {
        fprintf(stderr, "hexhost_repack_bytes: cannot read the GGUF file %s\n", argv[1]);
        return 1;
    }
    FILE * f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "hexhost_repack_bytes: cannot open %s\n", argv[1]);
        gguf_free(ctx);
        return 1;
    }

    fakedsp::config cfg;
    fakedsp::configure(cfg);
    hexhost::options o;
    hexhost::set_options(o);
    hexhost::device * dev = hexhost::device_new();
    if (!hexhost::device_open(dev)) {
        fprintf(stderr, "hexhost_repack_bytes: the session of the fake DSP did not open\n");
        fclose(f);
        gguf_free(ctx);
        return 1;
    }

    // The shapes come from a context with the tensor metadata of the file
    ggml_context *   meta = nullptr;
    gguf_init_params gm   = { true, &meta };
    gguf_context *   ctx2 = gguf_init_from_file(argv[1], gm);

    uint64_t   total = 1469598103934665603ull;
    size_t     n_done = 0;
    const size_t data0 = gguf_get_data_offset(ctx);
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); i++) {
        const char *        name = gguf_get_tensor_name(ctx, i);
        const ggml_tensor * src  = ggml_get_tensor(meta, name);
        const ggml_type     st   = gguf_get_tensor_type(ctx, i);
        const bool float_type    = st == GGML_TYPE_F16 || st == GGML_TYPE_BF16 || st == GGML_TYPE_F32;
        if (!src || !float_type || ggml_n_dims(src) != 2 || src->ne[0] % 256 != 0 || ggml_nelements(src) > max_elements) {
            continue;
        }
        std::vector<float> values;
        if (!read_floats(f, data0 + gguf_get_tensor_offset(ctx, i), st, (size_t) ggml_nelements(src), values)) {
            fprintf(stderr, "hexhost_repack_bytes: cannot read the values of %s\n", name);
            continue;
        }
        for (ggml_type qt : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
            ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, true };
            ggml_context *   c = ggml_init(p);
            ggml_tensor *    w = ggml_new_tensor_2d(c, qt, src->ne[0], src->ne[1]);
            std::vector<uint8_t> q(ggml_nbytes(w));
            ggml_quantize_chunk(qt, values.data(), q.data(), 0, src->ne[1], src->ne[0], nullptr);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(c, hexhost::device_buft(dev));
            if (!buf) {
                fprintf(stderr, "hexhost_repack_bytes: no buffer for %s\n", name);
                ggml_free(c);
                continue;
            }
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_set(w, q.data(), 0, q.size());
            const size_t tile_bytes = ggml_backend_buft_get_alloc_size(hexhost::device_buft(dev), w);
            const uint64_t h_tile   = fnv1a((const uint8_t *) w->data, tile_bytes);
            std::vector<uint8_t> back(q.size());
            ggml_backend_tensor_get(w, back.data(), 0, back.size());
            const uint64_t h_back = fnv1a(back.data(), back.size());
            printf("%s %s [%lld,%lld] tile %016" PRIx64 " back %016" PRIx64 "\n", name, ggml_type_name(qt),
                   (long long) src->ne[0], (long long) src->ne[1], h_tile, h_back);
            total = fnv1a((const uint8_t *) &h_tile, 8, total);
            total = fnv1a((const uint8_t *) &h_back, 8, total);
            n_done++;
            ggml_backend_buffer_free(buf);
            ggml_free(c);
        }
    }
    printf("total %zu tensor forms, hash %016" PRIx64 "\n", n_done, total);

    hexhost::device_free(dev);
    gguf_free(ctx2);
    ggml_free(meta);
    fclose(f);
    gguf_free(ctx);
    return 0;
}
