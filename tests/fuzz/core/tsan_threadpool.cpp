// tsan_threadpool: the thread pool of the ggml CPU backend (ggml-cpu.c) under TSan.
//
// The TSan build has GGML_OPENMP=OFF, thus the graph compute uses the ggml
// thread pool with its barriers and its work counters, as on the phone.
//
// One input gives a seed, a list of 1 to 6 operations and 2 to 4 thread
// counts. The harness builds one graph with the operations on random data:
//   mul_mat      a weight of type F32, F16, Q8_0 or Q4_0 times an F32 input
//   add, mul     with a broadcast of the second operand
//   rms_norm, silu, soft_max
//   gated_delta_net  with K = 1 to 5 rollback slots (the op of our patches)
//   flash_attn_ext   F16 K and V, one to four heads
// Then it computes the graph one time with 1 thread (the reference), and one
// time for each thread count, with one of three mechanisms:
//   - ggml_backend_cpu with ggml_backend_cpu_set_n_threads (the path of llama.cpp)
//   - a user thread pool (ggml_threadpool_new), with a pause and a resume
//   - ggml_graph_plan and ggml_graph_compute with that pool
// An abort callback from the input stops some computes after a number of nodes.
//
// Properties:
//   P1  TSan: no data race, no lock order inversion, no thread leak.
//   P2  Each output of a compute with N threads equals the output with 1
//       thread, bit for bit. Each op gives each output row to one thread,
//       thus the order of the sums does not depend on the thread count.
//       FUZZ_THREADPOOL_TOL gives a relative tolerance in place of 0.

#include "fuzz_common.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <atomic>
#include <random>
#include <vector>

namespace {

/** The state of the abort callback: stop after n_left calls. A negative count never stops. */
struct AbortState {
    std::atomic<int> n_left{-1};
};

/** The abort callback of ggml: true stops the compute. Each worker thread can call it. */
bool abort_cb(void * data) {
    auto * st = static_cast<AbortState *>(data);
    if (st->n_left.load(std::memory_order_relaxed) < 0) {
        return false;
    }
    return st->n_left.fetch_sub(1, std::memory_order_relaxed) <= 0;
}

/** Fill a tensor with N(0, 1) values of its type from rng. */
void fill(ggml_tensor * t, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f(ggml_nelements(t));
    for (auto & v : f) {
        v = nd(rng);
    }
    if (t->type == GGML_TYPE_F32) {
        memcpy(t->data, f.data(), ggml_nbytes(t));
    } else {
        // ggml_quantize_chunk converts the values to each type of the public API that a graph here uses
        const int64_t n_per_row = t->ne[0];
        ggml_quantize_chunk(t->type, f.data(), t->data, 0, ggml_nelements(t) / n_per_row, n_per_row, nullptr);
    }
}

/** One graph with its outputs. */
struct Graph {
    ggml_context *             ctx = nullptr;
    ggml_cgraph *              gf  = nullptr;
    std::vector<ggml_tensor *> outs;
};

/** Build the graph of the input. The context holds the data (no_alloc = false). */
Graph build(FuzzedDataProvider & fdp, std::mt19937 & rng) {
    Graph g;
    ggml_init_params ip = { /*mem_size =*/ 64u << 20, /*mem_buffer =*/ nullptr, /*no_alloc =*/ false };
    g.ctx = ggml_init(ip);
    g.gf  = ggml_new_graph(g.ctx);

    const int n_ops = fdp.ConsumeIntegralInRange<int>(1, 6);
    for (int i = 0; i < n_ops; ++i) {
        ggml_tensor * out = nullptr;
        switch (fdp.ConsumeIntegralInRange<int>(0, 5)) {
            case 0: {
                static const ggml_type kTypes[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 };
                const ggml_type t = kTypes[fdp.ConsumeIntegralInRange<int>(0, 3)];
                const int64_t k = 32 * fdp.ConsumeIntegralInRange<int>(1, 8);
                const int64_t n = fdp.ConsumeIntegralInRange<int>(1, 96);
                const int64_t m = fdp.ConsumeIntegralInRange<int>(1, 17);
                ggml_tensor * w = ggml_new_tensor_2d(g.ctx, t, k, n);
                ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, k, m);
                fill(w, rng);
                fill(x, rng);
                out = ggml_mul_mat(g.ctx, w, x);
                break;
            }
            case 1: {
                const int64_t n = fdp.ConsumeIntegralInRange<int>(1, 300);
                const int64_t m = fdp.ConsumeIntegralInRange<int>(1, 9);
                ggml_tensor * a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n, m);
                ggml_tensor * b = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n, fdp.ConsumeBool() ? 1 : m);
                fill(a, rng);
                fill(b, rng);
                out = fdp.ConsumeBool() ? ggml_add(g.ctx, a, b) : ggml_mul(g.ctx, a, b);
                break;
            }
            case 2: {
                const int64_t n = fdp.ConsumeIntegralInRange<int>(1, 512);
                const int64_t m = fdp.ConsumeIntegralInRange<int>(1, 9);
                ggml_tensor * a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n, m);
                fill(a, rng);
                switch (fdp.ConsumeIntegralInRange<int>(0, 2)) {
                    case 0:  out = ggml_rms_norm(g.ctx, a, 1e-6f); break;
                    case 1:  out = ggml_silu(g.ctx, a); break;
                    default: out = ggml_soft_max(g.ctx, a); break;
                }
                break;
            }
            case 3: {
                // gated delta net: q, k [S, H_k, T, N], v [S, H_v, T, N], g and beta [1, H_v, T, N], state [S, S, H_v, N]
                static const int64_t kS[] = { 16, 32, 64, 128 };
                const int64_t S   = kS[fdp.ConsumeIntegralInRange<int>(0, 3)];
                const int64_t H_k = fdp.ConsumeIntegralInRange<int>(1, 2);
                const int64_t H_v = H_k * fdp.ConsumeIntegralInRange<int>(1, 2);
                const int64_t T   = fdp.ConsumeIntegralInRange<int>(1, 8);
                const int64_t N   = fdp.ConsumeIntegralInRange<int>(1, 3);
                const int64_t K   = fdp.ConsumeIntegralInRange<int>(1, 5);
                ggml_tensor * q  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, S, H_k, T, N);
                ggml_tensor * kk = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, S, H_k, T, N);
                ggml_tensor * v  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, S, H_v, T, N);
                ggml_tensor * gg = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, 1, H_v, T, N);
                ggml_tensor * be = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, 1, H_v, T, N);
                ggml_tensor * st = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, S, S, H_v, N);
                for (ggml_tensor * t : { q, kk, v, st }) {
                    fill(t, rng);
                }
                fill(gg, rng);
                fill(be, rng);
                // a gate below 0 and a beta in (0, 1), as the model gives them
                out = ggml_gated_delta_net(g.ctx, ggml_l2_norm(g.ctx, q, 1e-6f), ggml_l2_norm(g.ctx, kk, 1e-6f), v,
                                           ggml_neg(g.ctx, ggml_abs(g.ctx, gg)), ggml_sigmoid(g.ctx, be), st, K);
                break;
            }
            case 4: {
                static const int64_t kD[] = { 64, 128 };
                const int64_t D    = kD[fdp.ConsumeIntegralInRange<int>(0, 1)];
                const int64_t H    = fdp.ConsumeIntegralInRange<int>(1, 4);
                const int64_t n_q  = fdp.ConsumeIntegralInRange<int>(1, 8);
                const int64_t n_kv = 32 * fdp.ConsumeIntegralInRange<int>(1, 4);
                ggml_tensor * q = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, D, n_q, H, 1);
                ggml_tensor * k = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, D, n_kv, H, 1);
                ggml_tensor * v = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, D, n_kv, H, 1);
                fill(q, rng);
                fill(k, rng);
                fill(v, rng);
                out = ggml_flash_attn_ext(g.ctx, q, k, v, nullptr, 1.0f / std::sqrt((float) D), 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
                break;
            }
            default: {
                // get_rows from a quantized table, as the token embedding lookup
                const int64_t n = 32 * fdp.ConsumeIntegralInRange<int>(1, 8);
                const int64_t rows = fdp.ConsumeIntegralInRange<int>(1, 64);
                ggml_tensor * tab = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q8_0, n, rows);
                fill(tab, rng);
                const int64_t n_ids = fdp.ConsumeIntegralInRange<int>(1, 16);
                ggml_tensor * ids = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_ids);
                for (int64_t j = 0; j < n_ids; ++j) {
                    ((int32_t *) ids->data)[j] = (int32_t) (rng() % rows);
                }
                out = ggml_get_rows(g.ctx, tab, ids);
                break;
            }
        }
        ggml_build_forward_expand(g.gf, out);
        g.outs.push_back(out);
    }
    return g;
}

/**
 * Fill each output with the byte 0xA5 before a compute. An op can leave a part of its output
 * unwritten by design (the gated delta net writes min(T, K) snapshot slots of K), and each compute
 * must start from the same bytes: then a part that one thread count writes and a different one does
 * not write shows as a difference, and no compare reads memory that nothing wrote. O(output bytes).
 */
void poison(const Graph & g) {
    for (const ggml_tensor * t : g.outs) {
        memset(t->data, 0xA5, ggml_nbytes(t));
    }
}

/** A copy of the bytes of each output. */
std::vector<std::vector<uint8_t>> snapshot(const Graph & g) {
    std::vector<std::vector<uint8_t>> s;
    for (const ggml_tensor * t : g.outs) {
        s.emplace_back((const uint8_t *) t->data, (const uint8_t *) t->data + ggml_nbytes(t));
    }
    return s;
}

/** P2: compare the outputs with the reference. */
void compare(const Graph & g, const std::vector<std::vector<uint8_t>> & ref, int n_threads, int mech, double tol) {
    for (size_t i = 0; i < g.outs.size(); ++i) {
        const ggml_tensor * t = g.outs[i];
        if (tol == 0.0) {
            if (memcmp(t->data, ref[i].data(), ref[i].size()) != 0) {
                fuzz::fail("P2: output %zu (%s) with %d threads (mechanism %d) differs from the output with 1 thread",
                           i, ggml_op_desc(t), n_threads, mech);
            }
            continue;
        }
        const float * a = (const float *) t->data;
        const float * b = (const float *) ref[i].data();
        for (int64_t j = 0; j < ggml_nelements(t); ++j) {
            if (!(std::fabs(a[j] - b[j]) <= tol * (1.0 + std::fabs(b[j])))) {
                fuzz::fail("P2: output %zu (%s) element %lld is %g with %d threads, %g with 1 thread",
                           i, ggml_op_desc(t), (long long) j, a[j], n_threads, b[j]);
            }
        }
    }
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int * /*argc*/, char *** /*argv*/) {
    fuzz::quiet_logs();
    ggml_cpu_init();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz::note_input(data, size);
    FuzzedDataProvider fdp(data, size);
    static const double tol = [] {
        const char * s = getenv("FUZZ_THREADPOOL_TOL");
        return s ? atof(s) : 0.0;
    }();
    std::mt19937 rng(fdp.ConsumeIntegral<uint32_t>());
    Graph g = build(fdp, rng);

    // the reference: one thread, no pool
    poison(g);
    if (ggml_graph_compute_with_ctx(g.ctx, g.gf, 1) != GGML_STATUS_SUCCESS) {
        fuzz::fail("the compute with 1 thread fails");
    }
    const auto ref = snapshot(g);

    ggml_backend_t backend = ggml_backend_cpu_init();
    AbortState abort_state;
    const int n_runs = fdp.ConsumeIntegralInRange<int>(2, 4);
    for (int r = 0; r < n_runs; ++r) {
        const int n_threads = fdp.ConsumeIntegralInRange<int>(2, 8);
        const int mech = fdp.ConsumeIntegralInRange<int>(0, 2);
        const bool use_abort = fdp.ConsumeIntegralInRange<int>(0, 5) == 0;
        abort_state.n_left.store(use_abort ? fdp.ConsumeIntegralInRange<int>(0, 8) : -1);

        ggml_status st = GGML_STATUS_SUCCESS;
        poison(g);
        ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
        ggml_threadpool * pool = nullptr;
        if (mech == 0) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
            ggml_backend_cpu_set_threadpool(backend, nullptr);
            ggml_backend_cpu_set_abort_callback(backend, abort_cb, &abort_state);
            st = ggml_backend_graph_compute(backend, g.gf);
        } else {
            tpp.paused = fdp.ConsumeBool();
            pool = ggml_threadpool_new(&tpp);
            if (tpp.paused) {
                ggml_threadpool_resume(pool);
            }
            if (mech == 1) {
                ggml_backend_cpu_set_n_threads(backend, n_threads);
                ggml_backend_cpu_set_threadpool(backend, pool);
                ggml_backend_cpu_set_abort_callback(backend, abort_cb, &abort_state);
                st = ggml_backend_graph_compute(backend, g.gf);
                if (fdp.ConsumeBool()) {
                    ggml_threadpool_pause(pool);
                    ggml_threadpool_resume(pool);
                    st = ggml_backend_graph_compute(backend, g.gf);
                }
                ggml_backend_cpu_set_threadpool(backend, nullptr);
            } else {
                ggml_cplan plan = ggml_graph_plan(g.gf, n_threads, pool);
                std::vector<uint8_t> work(plan.work_size);
                plan.work_data = work.data();
                plan.abort_callback = abort_cb;
                plan.abort_callback_data = &abort_state;
                st = ggml_graph_compute(g.gf, &plan);
            }
        }
        if (st == GGML_STATUS_SUCCESS) {
            compare(g, ref, n_threads, mech, tol);
        } else if (st != GGML_STATUS_ABORTED || !use_abort) {
            fuzz::fail("the compute with %d threads (mechanism %d) gives the status %d", n_threads, mech, (int) st);
        }
        if (pool != nullptr) {
            ggml_threadpool_free(pool);
        }
    }
    ggml_backend_free(backend);
    ggml_free(g.ctx);
    return 0;
}
