/*
 * The reproducers of the entries of tests/sanitizers/ubsan.supp (rule R13).
 *
 * Each mode runs one small ggml computation on the CPU backend. A mode of an
 * entry gives the UBSan report that the entry suppresses. A clean mode must
 * give no report: it runs a path where the sanitizer reported a defect of
 * ggml that the patch series corrects, thus it is a regression check. The
 * control modes give a report of a check in a function of this file, which no
 * entry names. With the full suppression file, a control must still stop the
 * run, thus the entries suppress their reports and nothing else.
 *
 * tests/sanitizers/supp-repro.sh builds this file with the flags of the
 * ubsan build of one profile, links it with the ggml libraries of that build,
 * and runs each mode with and without each entry.
 *
 * Usage: ubsan_ggml <mode>
 *
 * The map of the modes. supp-repro.sh and check-rules.sh read these lines.
 * An entry can have more than one mode (REPRO-ENTRY: <entry> MODE: <mode>).
 * It passes if one of its modes stops with a report of its check when the
 * entry is removed.
 *   REPRO-CLEAN: pointer-overflow MODE: graph_nbytes
 *   REPRO-CLEAN: function MODE: mul_mat_f32
 *   REPRO-CLEAN: function MODE: mul_mat_f16
 *   REPRO-CLEAN: function MODE: mul_mat_id
 *   REPRO-CLEAN: function MODE: set_rows
 *   REPRO-CLEAN: function MODE: flash_attn_ext
 *   REPRO-CLEAN: function MODE: dup_from_q
 *   REPRO-CLEAN: function MODE: add_q
 *   REPRO-CLEAN: function MODE: add1_q
 *   REPRO-CLEAN: function MODE: out_prod_q
 *   REPRO-CLEAN: function MODE: get_rows_q
 *   REPRO-CLEAN: function MODE: lightning_indexer
 *   REPRO-CLEAN: function MODE: dup_to_q
 *   REPRO-CLEAN: function MODE: repack_mul_mat
 *   REPRO-CLEAN: function MODE: repack_mul_mat_id
 *   REPRO-CONTROL: pointer-overflow MODE: control_pointer_overflow
 *   REPRO-CONTROL: function MODE: control_function
 *   REPRO-CONTROL: integer-divide-by-zero MODE: control_divide
 * The clean modes:
 *   graph_nbytes: ggml_graph_nbytes must compute the size of a graph with no
 *     arithmetic on a null pointer (patches/fuzz-ops/0001). The mode reaches
 *     it through ggml_graph_overhead, and each other mode reaches it through
 *     ggml_new_graph.
 *   The "function" modes: the CPU backend must call each type trait function
 *     (to_float, from_float, vec_dot) through a pointer of its own type
 *     (patches/fuzz-ops/0002). Each mode runs one op that makes such a call.
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Make a CPU context of 64 MB with the tensor data inside it. */
static struct ggml_context * make_ctx(void) {
    struct ggml_init_params params = { 64u * 1024u * 1024u, NULL, false };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        fprintf(stderr, "ubsan_ggml: ggml_init failed\n");
        exit(2);
    }
    return ctx;
}

static int compute(struct ggml_context * ctx, struct ggml_tensor * out);

/* Fill the bytes of Q8_0 or Q4_0 blocks: a finite scale of 0.01 and small
 * quants. A Q8_0 block is {fp16 d, int8 qs[32]}, a Q4_0 block is
 * {fp16 d, uint8 qs[16]}. */
static void fill_quant(enum ggml_type type, uint8_t * data, size_t nbytes) {
    const size_t bs = ggml_type_size(type);
    const ggml_fp16_t d = ggml_fp32_to_fp16(0.01f);
    for (size_t off = 0; off + bs <= nbytes; off += bs) {
        memcpy(data + off, &d, sizeof(d));
        for (size_t j = sizeof(d); j < bs; ++j) {
            data[off + j] = (uint8_t) ((off / bs * 7 + j * 13) % 64);
        }
    }
}

/* Fill a F32, F16, Q8_0 or Q4_0 tensor with small values that depend on the index. */
static void fill(struct ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_0) {
        fill_quant(t->type, (uint8_t *) t->data, ggml_nbytes(t));
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        const float v = (float) ((i * 37) % 101) / 101.0f - 0.5f;
        if (t->type == GGML_TYPE_F32) {
            ((float *) t->data)[i] = v;
        } else if (t->type == GGML_TYPE_F16) {
            ((ggml_fp16_t *) t->data)[i] = ggml_fp32_to_fp16(v);
        }
    }
}

/* A copy F32 to Q8_0 (to_q) or Q8_0 to F32 (from_q), 4 rows of 64. */
static int mode_dup(int to_q) {
    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, to_q ? GGML_TYPE_F32 : GGML_TYPE_Q8_0, 64, 4);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, to_q ? GGML_TYPE_Q8_0 : GGML_TYPE_F32, 64, 4);
    fill(a);
    fill(b);
    return compute(ctx, ggml_cpy(ctx, a, b));
}

/* An addition with a Q8_0 first operand: ADD (add1 = 0) or ADD1 (add1 = 1). */
static int mode_add_q(int add1) {
    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 4);
    struct ggml_tensor * b = add1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
                                  : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
    fill(a);
    fill(b);
    /* ggml_add1 is deprecated. The CPU backend has its op, thus the mode calls it. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    return compute(ctx, add1 ? ggml_add1(ctx, a, b) : ggml_add(ctx, a, b));
#pragma clang diagnostic pop
}

/* An outer product with a Q8_0 first operand: [64, 4] and F32 [8, 4]. */
static int mode_out_prod_q(void) {
    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 4);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    fill(a);
    fill(b);
    return compute(ctx, ggml_out_prod(ctx, a, b));
}

/* A row lookup in a Q8_0 matrix (the token embedding of a Q8_0 model). */
static int mode_get_rows_q(void) {
    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 8);
    struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
    fill(a);
    for (int i = 0; i < 3; ++i) {
        ((int32_t *) idx->data)[i] = 2 * i + 1;
    }
    return compute(ctx, ggml_get_rows(ctx, a, idx));
}

/* The lightning indexer of DeepSeek V4 with a F16 K: q [64, 2 heads, 2 tokens],
 * k [64, 1, 16], weights [2, 2], mask F16 [16, 2]. */
static int mode_lightning_indexer(void) {
    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 2, 2);
    struct ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 1, 16);
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    struct ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 16, 2);
    fill(q);
    fill(k);
    fill(w);
    for (int64_t i = 0; i < ggml_nelements(m); ++i) {
        ((ggml_fp16_t *) m->data)[i] = ggml_fp32_to_fp16(0.0f);
    }
    return compute(ctx, ggml_lightning_indexer(ctx, q, k, w, m));
}

/* A matrix product with Q4_0 weights in the CPU repack buffer type, thus the
 * repack path (ggml::cpu::repack::tensor_traits::forward_mul_mat and
 * forward_mul_mat_id) computes it. Status 4: the build has no repack buffer
 * type for Q4_0 (for example without AVX2), and the mode cannot run. */
static int mode_repack(int with_id) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    ggml_backend_dev_get_extra_bufts_t get_extra =
        (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t buft = NULL;
    if (get_extra != NULL) {
        for (ggml_backend_buffer_type_t * p = get_extra(dev); p != NULL && *p != NULL; ++p) {
            if (strstr(ggml_backend_buft_name(*p), "REPACK") != NULL) {
                buft = *p;
            }
        }
    }
    if (buft == NULL) {
        fprintf(stderr, "ubsan_ggml: no CPU_REPACK buffer type\n");
        ggml_backend_free(cpu);
        return 4;
    }

    struct ggml_init_params wp = { 8 * ggml_tensor_overhead(), NULL, true };
    struct ggml_context * cw = ggml_init(wp);
    struct ggml_tensor * w = with_id ? ggml_new_tensor_3d(cw, GGML_TYPE_Q4_0, 256, 16, 4)
                                     : ggml_new_tensor_2d(cw, GGML_TYPE_Q4_0, 256, 16);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(cw, buft);
    const size_t nb = ggml_nbytes(w);
    uint8_t * host = (uint8_t *) malloc(nb);
    fill_quant(GGML_TYPE_Q4_0, host, nb);
    ggml_backend_tensor_set(w, host, 0, nb);
    free(host);

    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * out;
    if (with_id) {
        struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, 8);
        struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 8);
        fill(x);
        for (int i = 0; i < 16; ++i) {
            ((int32_t *) ids->data)[i] = i % 4;
        }
        out = ggml_mul_mat_id(ctx, w, x, ids);
    } else {
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
        fill(x);
        out = ggml_mul_mat(ctx, w, x);
    }
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const enum ggml_status st = ggml_backend_graph_compute(cpu, gf);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(cw);
    ggml_backend_free(cpu);
    return st == GGML_STATUS_SUCCESS ? 0 : 3;
}

/* Build the graph of one output tensor and compute it with one thread. */
static int compute(struct ggml_context * ctx, struct ggml_tensor * out) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
    ggml_free(ctx);
    return st == GGML_STATUS_SUCCESS ? 0 : 3;
}

/* A matrix product: src0 of the given type [64, 8], src1 F32 [64, 4]. */
static int mode_mul_mat(enum ggml_type type) {
    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, type, 64, 8);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
    fill(a);
    fill(b);
    return compute(ctx, ggml_mul_mat(ctx, a, b));
}

/* An indirect matrix product: 4 F16 experts, 2 experts for each of 3 tokens. */
static int mode_mul_mat_id(void) {
    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * as = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 8, 4);
    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 2, 3);
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 3);
    fill(as);
    fill(b);
    for (int i = 0; i < 6; ++i) {
        ((int32_t *) ids->data)[i] = i % 4;
    }
    return compute(ctx, ggml_mul_mat_id(ctx, as, b, ids));
}

/* A row store: 3 F32 rows into a F16 matrix of 10 rows. */
static int mode_set_rows(void) {
    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 10);
    struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 3);
    struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 3);
    fill(dst);
    fill(src);
    for (int i = 0; i < 3; ++i) {
        ((int64_t *) idx->data)[i] = 2 * i + 1;
    }
    return compute(ctx, ggml_set_rows(ctx, dst, src, idx));
}

/* A flash attention: F32 Q [64, 2 tokens, 2 heads], F16 K and V [64, 16, 2]. */
static int mode_flash_attn_ext(void) {
    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 2, 2);
    struct ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 16, 2);
    struct ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 16, 2);
    fill(q);
    fill(k);
    fill(v);
    return compute(ctx, ggml_flash_attn_ext(ctx, q, k, v, NULL, 0.125f, 0.0f, 0.0f));
}

/* The control of pointer-overflow: an offset from a null pointer in a
 * function that no entry names. */
__attribute__((noinline)) static char * repro_control_null_offset(size_t n) {
    char * volatile base = NULL;
    return base + n;
}

typedef int (*repro_int_fn)(int);

/* A function of a different type than repro_int_fn. */
__attribute__((noinline)) static int repro_control_callee(const float * x) {
    return x == NULL ? 7 : 8;
}

/* The control of integer-divide-by-zero: a division by zero in a function
 * that no entry names. The report stops the run before the division, thus
 * x86 does not trap. */
__attribute__((noinline)) static unsigned repro_control_divide(unsigned d) {
    volatile unsigned n = 7;
    return n / d;
}

/* The control of function: a call through a pointer of the wrong function
 * type, in a function that no entry names. */
__attribute__((noinline)) static int repro_control_bad_call(void) {
    repro_int_fn volatile fn = (repro_int_fn) (void (*)(void)) repro_control_callee;
    return fn(0);
}

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "";
    int rc;
    if (strcmp(mode, "graph_nbytes") == 0) {
        rc = ggml_graph_overhead() > 0 ? 0 : 3;
    } else if (strcmp(mode, "mul_mat_f32") == 0) {
        rc = mode_mul_mat(GGML_TYPE_F32);
    } else if (strcmp(mode, "mul_mat_f16") == 0) {
        rc = mode_mul_mat(GGML_TYPE_F16);
    } else if (strcmp(mode, "mul_mat_id") == 0) {
        rc = mode_mul_mat_id();
    } else if (strcmp(mode, "set_rows") == 0) {
        rc = mode_set_rows();
    } else if (strcmp(mode, "flash_attn_ext") == 0) {
        rc = mode_flash_attn_ext();
    } else if (strcmp(mode, "dup_to_q") == 0) {
        rc = mode_dup(1);
    } else if (strcmp(mode, "dup_from_q") == 0) {
        rc = mode_dup(0);
    } else if (strcmp(mode, "add_q") == 0) {
        rc = mode_add_q(0);
    } else if (strcmp(mode, "add1_q") == 0) {
        rc = mode_add_q(1);
    } else if (strcmp(mode, "out_prod_q") == 0) {
        rc = mode_out_prod_q();
    } else if (strcmp(mode, "get_rows_q") == 0) {
        rc = mode_get_rows_q();
    } else if (strcmp(mode, "lightning_indexer") == 0) {
        rc = mode_lightning_indexer();
    } else if (strcmp(mode, "repack_mul_mat") == 0) {
        rc = mode_repack(0);
    } else if (strcmp(mode, "repack_mul_mat_id") == 0) {
        rc = mode_repack(1);
    } else if (strcmp(mode, "control_pointer_overflow") == 0) {
        rc = repro_control_null_offset(96) != NULL ? 0 : 3;
    } else if (strcmp(mode, "control_divide") == 0) {
        volatile unsigned zero = 0;
        rc = repro_control_divide(zero) == 0 ? 0 : 3;
    } else if (strcmp(mode, "control_function") == 0) {
        rc = repro_control_bad_call() > 0 ? 0 : 3;
    } else {
        fprintf(stderr, "ubsan_ggml: the mode '%s' is not known\n", mode);
        return 2;
    }
    printf("ubsan_ggml: %s done, status %d\n", mode, rc);
    return rc;
}
