// fuzz_repack: the repack of the quantized weights into the tiled layout of the
// DSP (set_tensor, set_tensor_2d and the shadow buffer), the read back
// (get_tensor, get_tensor_2d), and the allocation size of a repacked tensor.
//
// A tensor of a repack type (Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4, Q4_K, Q6_K) with a
// random shape goes into a weight buffer of the device. The harness writes
// random blocks in one piece, in random chunks, or with set_tensor_2d, then
// reads the tensor back. The invariants: no access outside the buffer
// (AddressSanitizer checks the tile writes against get_alloc_size), and the
// read back of a lossless type (Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4) gives the
// bytes that the harness wrote.

#include "fake_dsp.h"
#include "fuzz_death.h"
#include "harness.h"
#include "hexhost.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <fuzzer/FuzzedDataProvider.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerInitialize(int * argc, char *** argv) {
    (void) argc;
    (void) argv;
    harness::init();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
    FuzzedDataProvider fdp(data, size);

    static const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
                                       GGML_TYPE_MXFP4, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
    const ggml_type t    = types[fdp.ConsumeIntegralInRange<size_t>(0, 6)];
    const bool      kq   = t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K;
    const int64_t   blk  = kq ? 256 : 32;
    const int64_t   ne0  = blk * fdp.ConsumeIntegralInRange<int64_t>(1, kq ? 4 : 24);
    const int64_t   ne1  = fdp.ConsumeIntegralInRange<int64_t>(1, 200);
    const int64_t   ne2  = fdp.ConsumeIntegralInRange<int64_t>(1, 3);
    const int64_t   ne3  = fdp.ConsumeIntegralInRange<int64_t>(1, 2);
    const int       mode = fdp.ConsumeIntegralInRange<int>(0, 2);

    fakedsp::config cfg;
    fakedsp::configure(cfg);
    hexhost::options o;
    o.opbatch = 16;
    o.opqueue = 1;
    hexhost::set_options(o);

    hexhost::device * dev = hexhost::device_new();
    if (!hexhost::device_open(dev)) {
        hexhost::device_free(dev);
        return 0;
    }
    {
        ggml_init_params p   = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context *   ctx = ggml_init(p);
        ggml_tensor *    w   = ggml_new_tensor_4d(ctx, t, ne0, ne1, ne2, ne3);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, hexhost::device_buft(dev));
        if (buf) {
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            const size_t         n = ggml_nbytes(w);
            std::vector<uint8_t> in(n);
            for (size_t i = 0; i < n; i++) {
                in[i] = fdp.ConsumeIntegral<uint8_t>();
            }
            const size_t row   = ggml_row_size(t, ne0);
            const size_t rows  = (size_t) ne1 * ne2 * ne3;
            if (mode == 0 || kq) {
                ggml_backend_tensor_set(w, in.data(), 0, n);
            } else if (mode == 1) {
                // consecutive chunks of whole rows in a random order
                std::vector<std::pair<size_t, size_t>> chunks;
                size_t r = 0;
                while (r < rows) {
                    const size_t c = std::min(rows - r, (size_t) fdp.ConsumeIntegralInRange<int>(1, 64));
                    chunks.emplace_back(r * row, c * row);
                    r += c;
                }
                for (size_t i = chunks.size(); i > 1; i--) {
                    std::swap(chunks[i - 1], chunks[fdp.ConsumeIntegralInRange<size_t>(0, i - 1)]);
                }
                for (auto & c : chunks) {
                    ggml_backend_tensor_set(w, in.data() + c.first, c.first, c.second);
                }
            } else {
                // set_tensor_2d: whole rows, n_copies rows at a time
                const size_t per = (size_t) fdp.ConsumeIntegralInRange<int>(1, 8);
                for (size_t r = 0; r < rows; r += per) {
                    const size_t c = std::min(per, rows - r);
                    ggml_backend_tensor_set_2d(w, in.data() + r * row, r * row, row, c, row, row);
                }
            }

            std::vector<uint8_t> out(n, 0xcd);
            ggml_backend_tensor_get(w, out.data(), 0, n);
            const bool lossless = !kq;
            // The tile of a K-quant holds FP16 products of the block scales. A product above 65504
            // becomes Inf in the tile: the DSP multiplies with Inf, and the read back converts NaN
            // to an integer (UB, ggml-hexagon.cpp:1675 and 1850). Only UBSan sees the UB, thus the
            // harness reports the Inf scale of the read back in each configuration.
            if (kq) {
                const size_t bs = ggml_type_size(t);
                for (size_t b = 0; b + bs <= n; b += bs) {
                    const size_t off[2] = { t == GGML_TYPE_Q4_K ? (size_t) 0 : bs - 2, t == GGML_TYPE_Q4_K ? (size_t) 2 : bs - 2 };
                    for (size_t o : off) {
                        uint16_t h;
                        memcpy(&h, &out[b + o], 2);
                        if (((h >> 10) & 0x1f) == 0x1f) {
                            fakedsp::violation("repack-nonfinite-scale", "type %s block %zu: the read back scale at byte %zu is "
                                               "0x%04x (Inf or NaN): the tile holds a scale above the FP16 range",
                                               ggml_type_name(t), b / bs, o, h);
                        }
                    }
                }
            }
            if (lossless && memcmp(in.data(), out.data(), n) != 0) {
                size_t i = 0;
                while (i < n && in[i] == out[i]) {
                    i++;
                }
                fakedsp::violation("repack-roundtrip", "type %s %lldx%lldx%lldx%lld mode %d: byte %zu of %zu reads 0x%02x, the write was 0x%02x",
                                   ggml_type_name(t), (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3, mode, i, n,
                                   out[i], in[i]);
            }

            // get_tensor_2d over the first slice, 32 rows at a time from an offset of whole tiles
            if (!kq && ne1 >= 32 && fdp.ConsumeBool()) {
                const size_t tiles = (size_t) ne1 / 32;
                const size_t first = fdp.ConsumeIntegralInRange<size_t>(0, tiles - 1) * 32;
                std::vector<uint8_t> o2(row * 32, 0xcd);
                ggml_backend_tensor_get_2d(w, o2.data(), first * row, row, 32, row, row);
                if (memcmp(o2.data(), in.data() + first * row, row * 32) != 0) {
                    fakedsp::violation("repack-get-2d", "type %s rows %zu..%zu: get_tensor_2d differs from the written rows",
                                       ggml_type_name(t), first, first + 31);
                }
            }
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
    }
    hexhost::device_free(dev);
    return 0;
}
