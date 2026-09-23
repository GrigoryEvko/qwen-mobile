// fuzz_gguf: the GGUF reader and writer of ggml (ggml/src/gguf.cpp) on any byte string.
//
// One input goes through four readers, and the readers must agree:
//   1. gguf_init_from_buffer, metadata only.
//   2. gguf_init_from_callback, with a chunk size of 1 to 64 bytes from the last input byte.
//   3. gguf_init_from_file_ptr on an fmemopen() stream.
//   4. gguf_init_from_buffer with a ggml context and no_alloc = false, which also reads the tensor data.
// Readers 1, 2 and 3 must all accept or all refuse the input. Reader 4 can
// refuse an input that reader 1 accepts only when the data section does not
// fit in the input.
//
// For an accepted input, the harness reads every key with the getter of its
// type and every tensor info, and it does two round trips:
//   - metadata: write the metadata, read it again, and compare the keys, the
//     values and the tensor infos. A second write must give the same bytes.
//   - data: copy the keys and the tensors of reader 4 into a new context (as
//     gguf-split does), write the full file, read it again, and compare the
//     tensor bytes.
//
// Property "no allocation before the length check": when the data section
// does not fit in the input, reader 4 must refuse the input before it
// allocates the declared data size. A malloc hook of the sanitizer runtime
// measures the largest single allocation, thus this check runs in the asan,
// tsan and msan builds only.
//
// Property "no huge alignment": the reader refuses an alignment above 1 MiB.
// The writer pads the metadata to the alignment one byte at a time.
//
// The harness skips reader 4 when the declared data section is larger than
// FUZZ_GGUF_MAX_DATA bytes (the default is 256 MiB), to keep the RSS below the
// limit of libFuzzer.

#include "fuzz_common.h"

#include "ggml-impl.h"  // gguf_write_to_buf
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <string>
#include <vector>

// The allocator hooks of the sanitizer runtimes (ASan, TSan, MSan, HWASan). The none and ubsan
// builds have no such runtime, thus the declaration is weak, and there the check of
// "no allocation before the length check" is off.
extern "C" __attribute__((weak)) int __sanitizer_install_malloc_and_free_hooks(
    void (*malloc_hook)(const volatile void *, size_t), void (*free_hook)(const volatile void *));

namespace {

std::atomic<bool>   g_track{false};
std::atomic<size_t> g_max_alloc{0};
bool                g_have_hooks = false;

/** Record the size of the largest allocation while g_track is true. */
void malloc_hook(const volatile void * /*ptr*/, size_t size) {
    if (!g_track.load(std::memory_order_relaxed)) {
        return;
    }
    size_t cur = g_max_alloc.load(std::memory_order_relaxed);
    while (size > cur && !g_max_alloc.compare_exchange_weak(cur, size, std::memory_order_relaxed)) {
    }
}

/** The free hook. The harness does not need it, but the interface takes a pair. */
void free_hook(const volatile void * /*ptr*/) {}

/** The reader state of gguf_init_from_callback: the input bytes. */
struct buf_reader {
    const uint8_t * data;
    size_t          size;
};

/** The callback reader: copy up to len bytes at offset, and return the count. */
size_t read_cb(void * userdata, void * output, uint64_t offset, size_t len) {
    const buf_reader & r = *static_cast<const buf_reader *>(userdata);
    if (offset >= r.size) {
        return 0;
    }
    const size_t n = std::min<uint64_t>(len, r.size - offset);
    memcpy(output, r.data + offset, n);
    return n;
}

/** Read every key and every tensor info of ctx with the correct getter. Stop on an inconsistency. */
void walk(const gguf_context * ctx) {
    const int64_t n_kv = gguf_get_n_kv(ctx);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(ctx, i);
        if (gguf_find_key(ctx, key) != i) {
            fuzz::fail("gguf_find_key(\"%s\") does not return its index %" PRId64, key, i);
        }
        const gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            const gguf_type at = gguf_get_arr_type(ctx, i);
            const size_t    n  = gguf_get_arr_n(ctx, i);
            if (at == GGUF_TYPE_STRING) {
                for (size_t j = 0; j < n; ++j) {
                    const char * s = gguf_get_arr_str(ctx, i, j);
                    (void) strlen(s);
                }
            } else if (n > 0) {
                const uint8_t * p = (const uint8_t *) gguf_get_arr_data(ctx, i);
                volatile uint8_t sink = p[0] ^ p[n * gguf_type_size(at) - 1];
                (void) sink;
            }
            continue;
        }
        switch (type) {
            case GGUF_TYPE_UINT8:   (void) gguf_get_val_u8(ctx, i);   break;
            case GGUF_TYPE_INT8:    (void) gguf_get_val_i8(ctx, i);   break;
            case GGUF_TYPE_UINT16:  (void) gguf_get_val_u16(ctx, i);  break;
            case GGUF_TYPE_INT16:   (void) gguf_get_val_i16(ctx, i);  break;
            case GGUF_TYPE_UINT32:  (void) gguf_get_val_u32(ctx, i);  break;
            case GGUF_TYPE_INT32:   (void) gguf_get_val_i32(ctx, i);  break;
            case GGUF_TYPE_FLOAT32: (void) gguf_get_val_f32(ctx, i);  break;
            case GGUF_TYPE_UINT64:  (void) gguf_get_val_u64(ctx, i);  break;
            case GGUF_TYPE_INT64:   (void) gguf_get_val_i64(ctx, i);  break;
            case GGUF_TYPE_FLOAT64: (void) gguf_get_val_f64(ctx, i);  break;
            case GGUF_TYPE_BOOL:    (void) gguf_get_val_bool(ctx, i); break;
            case GGUF_TYPE_STRING:  (void) strlen(gguf_get_val_str(ctx, i)); break;
            default: fuzz::fail("key %" PRId64 " has the type %d, which the reader must refuse", i, (int) type);
        }
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        if (gguf_find_tensor(ctx, name) != i) {
            fuzz::fail("gguf_find_tensor(\"%s\") does not return its index %" PRId64, name, i);
        }
        const ggml_type t = gguf_get_tensor_type(ctx, i);
        if ((int) t < 0 || t >= GGML_TYPE_COUNT) {
            fuzz::fail("tensor %" PRId64 " has the type %d", i, (int) t);
        }
        (void) gguf_get_tensor_offset(ctx, i);
        (void) gguf_get_tensor_size(ctx, i);
        (void) gguf_get_tensor_ne(ctx, i);
    }
}

/** Compare two KV pairs, which must have the same key, type and value. */
void compare_kv(const gguf_context * a, const gguf_context * b, int64_t i) {
    if (strcmp(gguf_get_key(a, i), gguf_get_key(b, i)) != 0 || gguf_get_kv_type(a, i) != gguf_get_kv_type(b, i)) {
        fuzz::fail("round trip: key %" PRId64 " changed its name or type", i);
    }
    const gguf_type type = gguf_get_kv_type(a, i);
    if (type == GGUF_TYPE_ARRAY) {
        const gguf_type at = gguf_get_arr_type(a, i);
        const size_t    n  = gguf_get_arr_n(a, i);
        if (at != gguf_get_arr_type(b, i) || n != gguf_get_arr_n(b, i)) {
            fuzz::fail("round trip: array %" PRId64 " changed its type or length", i);
        }
        if (at == GGUF_TYPE_STRING) {
            for (size_t j = 0; j < n; ++j) {
                if (strcmp(gguf_get_arr_str(a, i, j), gguf_get_arr_str(b, i, j)) != 0) {
                    // an embedded NUL shortens the C string on the two sides equally, thus strcmp is sufficient here
                    fuzz::fail("round trip: string %zu of array %" PRId64 " changed", j, i);
                }
            }
        } else if (n > 0 && memcmp(gguf_get_arr_data(a, i), gguf_get_arr_data(b, i), n * gguf_type_size(at)) != 0) {
            fuzz::fail("round trip: the data of array %" PRId64 " changed", i);
        }
    } else if (type == GGUF_TYPE_STRING) {
        if (strcmp(gguf_get_val_str(a, i), gguf_get_val_str(b, i)) != 0) {
            fuzz::fail("round trip: the string value of key %" PRId64 " changed", i);
        }
    } else if (memcmp(gguf_get_val_data(a, i), gguf_get_val_data(b, i), gguf_type_size(type)) != 0) {
        fuzz::fail("round trip: the value of key %" PRId64 " changed", i);
    }
}

/** Compare the keys and the tensor infos of two contexts. */
void compare_ctx(const gguf_context * a, const gguf_context * b) {
    if (gguf_get_n_kv(a) != gguf_get_n_kv(b) || gguf_get_n_tensors(a) != gguf_get_n_tensors(b)) {
        fuzz::fail("round trip: the count of keys or tensors changed");
    }
    if (gguf_get_alignment(a) != gguf_get_alignment(b) || gguf_get_version(a) != gguf_get_version(b)) {
        fuzz::fail("round trip: the alignment or the version changed");
    }
    for (int64_t i = 0; i < gguf_get_n_kv(a); ++i) {
        compare_kv(a, b, i);
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(a); ++i) {
        if (strcmp(gguf_get_tensor_name(a, i), gguf_get_tensor_name(b, i)) != 0 ||
            gguf_get_tensor_type(a, i) != gguf_get_tensor_type(b, i) ||
            gguf_get_tensor_offset(a, i) != gguf_get_tensor_offset(b, i) ||
            memcmp(gguf_get_tensor_ne(a, i), gguf_get_tensor_ne(b, i), GGML_MAX_DIMS * sizeof(int64_t)) != 0) {
            fuzz::fail("round trip: the info of tensor %" PRId64 " changed", i);
        }
    }
}

/** Write the metadata of ctx, read it again, compare, and write it again. The two writes must be equal. */
void roundtrip_meta(const gguf_context * ctx) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ true);
    if (buf.size() != gguf_get_meta_size(ctx)) {
        fuzz::fail("gguf_get_meta_size gives %zu, the writer wrote %zu bytes", gguf_get_meta_size(ctx), buf.size());
    }
    gguf_context * again = gguf_init_from_buffer(buf.data(), buf.size(), { /*no_alloc =*/ true, /*ctx =*/ nullptr });
    if (again == nullptr) {
        fuzz::fail("the reader refuses the metadata that the writer wrote (%zu bytes)", buf.size());
    }
    compare_ctx(ctx, again);
    std::vector<int8_t> buf2;
    gguf_write_to_buf(again, buf2, true);
    if (buf2 != buf) {
        fuzz::fail("the second write of the metadata differs from the first");
    }
    gguf_free(again);
}

/** Copy ctx and the tensors of ctx_data into a new context, write the full file, read it, and compare the tensor bytes. */
void roundtrip_data(const gguf_context * ctx, ggml_context * ctx_data) {
    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, ctx);
    std::vector<const ggml_tensor *> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_data); t != nullptr; t = ggml_get_next_tensor(ctx_data, t)) {
        if (strcmp(t->name, "GGUF tensor data binary blob") == 0) {
            continue;
        }
        gguf_add_tensor(out, t);
        tensors.push_back(t);
    }
    std::vector<int8_t> buf;
    gguf_write_to_buf(out, buf, /*only_meta =*/ false);

    ggml_context * ctx_data2 = nullptr;
    gguf_context * again = gguf_init_from_buffer(buf.data(), buf.size(), { /*no_alloc =*/ false, &ctx_data2 });
    if (again == nullptr) {
        fuzz::fail("the reader refuses the full file that the writer wrote (%zu bytes)", buf.size());
    }
    compare_ctx(out, again);
    for (const ggml_tensor * t : tensors) {
        const ggml_tensor * t2 = ggml_get_tensor(ctx_data2, t->name);
        // a tensor of 0 bytes can have no data pointer, and memcmp must not get a null pointer
        const size_t nbytes = ggml_nbytes(t);
        if (t2 == nullptr || ggml_nbytes(t2) != nbytes || (nbytes > 0 && memcmp(t2->data, t->data, nbytes) != 0)) {
            fuzz::fail("round trip: the data of tensor \"%s\" changed", t->name);
        }
    }
    gguf_free(again);
    ggml_free(ctx_data2);
    gguf_free(out);
}

/** The declared end of the data section of ctx: the data offset plus the padded sizes of all tensors. */
uint64_t declared_data_end(const gguf_context * ctx, uint64_t * data_size) {
    const uint64_t align = gguf_get_alignment(ctx);
    uint64_t size = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        size += GGML_PAD(gguf_get_tensor_size(ctx, i), align);
    }
    *data_size = size;
    return gguf_get_data_offset(ctx) + size;
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int * /*argc*/, char *** /*argv*/) {
    fuzz::quiet_logs();
    if (__sanitizer_install_malloc_and_free_hooks != nullptr) {
        g_have_hooks = __sanitizer_install_malloc_and_free_hooks(malloc_hook, free_hook) != 0;
    }
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz::note_input(data, size);
    if (size == 0) {
        return 0;
    }
    static const uint64_t max_data = (uint64_t) fuzz::env_long("FUZZ_GGUF_MAX_DATA", 256l << 20);

    // 1. buffer, metadata only
    gguf_context * ctx = gguf_init_from_buffer(data, size, { /*no_alloc =*/ true, /*ctx =*/ nullptr });

    // 2. callback with small chunks
    {
        buf_reader r = { data, size };
        const size_t chunk = 1 + data[size - 1] % 64;
        gguf_context * c2 = gguf_init_from_callback(read_cb, &r, chunk, size, { true, nullptr });
        if ((c2 != nullptr) != (ctx != nullptr)) {
            fuzz::fail("the buffer reader %s the input, the callback reader (chunk %zu) %s it",
                       ctx ? "accepts" : "refuses", chunk, c2 ? "accepts" : "refuses");
        }
        if (c2 != nullptr) {
            compare_ctx(ctx, c2);
            gguf_free(c2);
        }
    }

    // 3. FILE stream
    {
        FILE * f = fmemopen((void *) data, size, "rb");
        if (f != nullptr) {
            gguf_context * c3 = gguf_init_from_file_ptr(f, { true, nullptr });
            fclose(f);
            if ((c3 != nullptr) != (ctx != nullptr)) {
                fuzz::fail("the buffer reader %s the input, the FILE reader %s it",
                           ctx ? "accepts" : "refuses", c3 ? "accepts" : "refuses");
            }
            if (c3 != nullptr) {
                compare_ctx(ctx, c3);
                gguf_free(c3);
            }
        }
    }

    if (ctx == nullptr) {
        return 0;
    }

    walk(ctx);

    // The writer pads the metadata to the alignment one byte at a time, thus an alignment of 1 GiB
    // costs 1 GiB of memory and 40 s. The reader must refuse an alignment above 1 MiB.
    if (gguf_get_alignment(ctx) > (1u << 20)) {
        fuzz::fail("the reader accepts the alignment %zu, and the writer then pads up to that many bytes one at a time",
                   gguf_get_alignment(ctx));
    }
    roundtrip_meta(ctx);

    // 4. buffer with the tensor data
    uint64_t data_size = 0;
    const uint64_t data_end = declared_data_end(ctx, &data_size);
    const bool fits = data_end <= size;
    if (data_size <= max_data) {
        ggml_context * ctx_data = nullptr;
        g_max_alloc.store(0, std::memory_order_relaxed);
        g_track.store(true, std::memory_order_relaxed);
        gguf_context * c4 = gguf_init_from_buffer(data, size, { /*no_alloc =*/ false, &ctx_data });
        g_track.store(false, std::memory_order_relaxed);
        const size_t max_alloc = g_max_alloc.load(std::memory_order_relaxed);

        if (fits && c4 == nullptr) {
            fuzz::fail("the data section fits (end %" PRIu64 " <= %zu), but the reader with data refuses the input", data_end, size);
        }
        if (!fits && c4 != nullptr) {
            fuzz::fail("the data section does not fit (end %" PRIu64 " > %zu), but the reader with data accepts the input", data_end, size);
        }
        if (g_have_hooks && !fits && data_size > size + (1u << 20) && max_alloc >= data_size) {
            fuzz::fail("the reader allocates %zu bytes for a data section of %" PRIu64 " bytes before it finds that the input holds only %zu bytes",
                       max_alloc, data_size, size);
        }
        if (c4 != nullptr) {
            compare_ctx(ctx, c4);
            roundtrip_data(c4, ctx_data);
            gguf_free(c4);
            ggml_free(ctx_data);
        }
    }

    gguf_free(ctx);
    return 0;
}
