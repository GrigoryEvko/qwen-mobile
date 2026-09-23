// The loader check of the quant fuzz area: read a GGUF file with the gguf loader of ggml,
// then decode each tensor to float32 with the row function of its type.
//
// Usage:
//   qfz-gguf-check <file.gguf> [dump-dir]
//   qfz-gguf-check --meta <file.gguf>
//
// The first form reads the tensor data into memory (no_alloc false). It prints one JSON line
// per tensor: the name, the type, the four dimensions, the count of values that are not finite,
// and a hash of the float32 bytes. With dump-dir, it also writes the float32 values of tensor i
// to <dump-dir>/<i>.f32, thus a test can compare them with the values of gguf-py.
//
// The second form reads only the metadata and the tensor infos (no_alloc true), as the model
// loader of llama.cpp does before its mmap, and it refuses a tensor whose data is not inside the
// file. The fuzzer of corrupted files uses this form, because the first form allocates the
// claimed data size before it compares that size with the file.
//
// Exit status: 0 when the file loads (and each tensor decodes), 3 when the loader refuses the
// file, 4 for a tensor type with no row function, 2 for a usage error. A sanitizer report
// uses the exit codes of ASAN_OPTIONS and UBSAN_OPTIONS.
//
// The build (run.sh) links the static ggml-base library of the ASan and UBSan build of
// llama.cpp, thus the loader and the row functions run under the two sanitizers.

#include "ggml.h"
#include "gguf.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The row decoders of ggml-quants.h, with their own block types. A direct call with the exact
// prototype is not a call through the generic function pointer of the type traits, which
// -fsanitize=function reports for each row. The blocks stay incomplete types here.
struct block_q4_0;
struct block_q8_0;
struct block_iq4_nl;
void dequantize_row_q4_0(const struct block_q4_0 * x, float * y, int64_t k);
void dequantize_row_q8_0(const struct block_q8_0 * x, float * y, int64_t k);
void dequantize_row_iq4_nl(const struct block_iq4_nl * x, float * y, int64_t k);

// Give the FNV-1a hash of n bytes.
static uint64_t fnv1a(const void * data, size_t n, uint64_t h) {
    const unsigned char * p = (const unsigned char *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Decode one row of a type to float32 with a direct call. Give 0 on success, 4 for a type that
// the pipeline does not write (F32, F16, Q4_0, Q8_0 and IQ4_NL are the types of the export).
static int decode_row(enum ggml_type type, const void * src, float * row, int64_t ne0) {
    switch (type) {
        case GGML_TYPE_F32:    memcpy(row, src, (size_t) ne0 * sizeof(float)); return 0;
        case GGML_TYPE_F16:    ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, row, ne0); return 0;
        case GGML_TYPE_Q4_0:   dequantize_row_q4_0((const struct block_q4_0 *) src, row, ne0); return 0;
        case GGML_TYPE_Q8_0:   dequantize_row_q8_0((const struct block_q8_0 *) src, row, ne0); return 0;
        case GGML_TYPE_IQ4_NL: dequantize_row_iq4_nl((const struct block_iq4_nl *) src, row, ne0); return 0;
        default:               return 4;
    }
}

// Decode every row of one tensor to float32. Write the rows to out when out is not NULL.
// Give 0 on success, 4 when the type has no row decoder here. The count of non-finite values
// goes to *nonfinite and the hash to *hash. Complexity is O(elements).
static int decode_tensor(const struct ggml_tensor * t, FILE * out, int64_t * nonfinite, uint64_t * hash) {
    const int64_t ne0 = t->ne[0];
    float * row = (float *) malloc((size_t) (ne0 > 0 ? ne0 : 1) * sizeof(float));
    if (row == NULL) {
        fprintf(stderr, "qfz-gguf-check: no memory for a row of %" PRId64 " values\n", ne0);
        exit(5);
    }
    *nonfinite = 0;
    *hash = 1469598103934665603ULL;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                const char * src = (const char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                if (decode_row(t->type, src, row, ne0) != 0) {
                    free(row);
                    return 4;
                }
                for (int64_t j = 0; j < ne0; ++j) {
                    *nonfinite += !isfinite(row[j]);
                }
                *hash = fnv1a(row, (size_t) ne0 * sizeof(float), *hash);
                if (out != NULL && fwrite(row, sizeof(float), (size_t) ne0, out) != (size_t) ne0) {
                    fprintf(stderr, "qfz-gguf-check: the dump write failed\n");
                    exit(5);
                }
            }
        }
    }
    free(row);
    return 0;
}

// Read the metadata and the tensor infos only, then refuse a tensor whose data is not inside the
// file. Give the exit status of main.
static int check_meta(const char * path) {
    FILE * f = fopen(path, "rb");
    if (f == NULL || fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "qfz-gguf-check: cannot open %s\n", path);
        return 3;
    }
    const long file_size = ftell(f);
    fclose(f);
    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ NULL };
    struct gguf_context * g = gguf_init_from_file(path, params);
    if (g == NULL) {
        printf("{\"status\": \"refused\"}\n");
        return 3;
    }
    int status = 0;
    const size_t base = gguf_get_data_offset(g);
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const size_t offset = gguf_get_tensor_offset(g, i);
        const size_t size = gguf_get_tensor_size(g, i);
        if (base > (size_t) file_size || offset > (size_t) file_size - base || size > (size_t) file_size - base - offset) {
            printf("{\"status\": \"out-of-file\", \"name\": \"%s\"}\n", gguf_get_tensor_name(g, i));
            status = 3;
            break;
        }
    }
    gguf_free(g);
    return status;
}

int main(int argc, char ** argv) {
    if (argc == 3 && strcmp(argv[1], "--meta") == 0) {
        return check_meta(argv[2]);
    }
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <file.gguf> [dump-dir]\n       %s --meta <file.gguf>\n", argv[0], argv[0]);
        return 2;
    }
    struct ggml_context * ctx = NULL;
    struct gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx };
    struct gguf_context * g = gguf_init_from_file(argv[1], params);
    if (g == NULL) {
        printf("{\"status\": \"refused\"}\n");
        return 3;
    }
    int status = 0;
    const int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n && status == 0; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx, name);
        FILE * out = NULL;
        if (argc == 3) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/%" PRId64 ".f32", argv[2], i);
            out = fopen(path, "wb");
            if (out == NULL) {
                fprintf(stderr, "qfz-gguf-check: cannot open %s for the dump\n", path);
                return 5;
            }
        }
        int64_t nonfinite = 0;
        uint64_t hash = 0;
        status = decode_tensor(t, out, &nonfinite, &hash);
        if (out != NULL) {
            fclose(out);
        }
        printf("{\"index\": %" PRId64 ", \"name\": \"%s\", \"type\": \"%s\", \"ne\": [%" PRId64 ", %" PRId64 ", %" PRId64
               ", %" PRId64 "], \"nonfinite\": %" PRId64 ", \"hash\": \"%016" PRIx64 "\", \"status\": %d}\n",
               i, name, ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3], nonfinite, hash, status);
    }
    gguf_free(g);
    ggml_free(ctx);
    return status;
}
