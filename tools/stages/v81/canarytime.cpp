// canarytime: the load cost of the start-up self-test (canary) of the Hexagon backend.
//
//   canarytime [--device HTP0] [--rounds N]
//
// The program measures two steps and prints one line for each:
//   register  ggml_backend_dev_count(), the first call into the backend registry. The Hexagon backend
//             registers its devices there, and with GGML_HEXAGON_CANARY=1 it opens the DSP session of
//             the first device and runs the canary before it registers them.
//   open      ggml_backend_dev_init() of the device, which opens its DSP session when the session is not
//             open.
// A run with GGML_HEXAGON_CANARY=0 and a run with GGML_HEXAGON_CANARY=1 thus give the cost of the canary at
// load: the difference of the sums of the two steps. The canary itself logs its result, the open time
// and the op time (the line "ggml-hex: canary pass" or "canary FAIL" on stderr).
//
// With --rounds N (the preset value 3) the program then runs N MUL_MAT graphs of 256 x 256 on the device
// (the "work" lines), to show that the device computes after the canary. The work checks the output
// against the host (the tolerance 1e-3 of the sum of the magnitudes of the terms).
//
// Exit status: 0 when the device exists and each work graph passes, 3 when the backend registered no
// device of that name (the expected result with GGML_HEXAGON_CANARY=2), 1 on an other failure, 2 for
// a usage error.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

// The milliseconds since t0
double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

// One MUL_MAT of an F32 weight of n x k with an F32 activation of k x m on the backend, checked against the
// host. Returns true when every output value is within 1e-3 of the sum of the magnitudes of its terms.
bool work_graph(ggml_backend_t be, int64_t k, int64_t n, int64_t m, uint32_t seed, double * ms) {
    const ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *         ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (!ggml_backend_supports_op(be, y)) {
        printf("work: the device does not support the MUL_MAT of F32 weights\n");
        ggml_free(ctx);
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    std::vector<float> wv((size_t) (k * n)), xv((size_t) (k * m)), yv((size_t) (n * m));
    uint32_t           s = seed;
    for (auto * v : { &wv, &xv }) {
        for (float & f : *v) {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            f = (float) (s >> 8) / 16777216.0f * 2.0f - 1.0f;
        }
    }
    ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * sizeof(float));
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
    const auto t0 = clock_type::now();
    const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    *ms           = ms_since(t0);
    ggml_backend_tensor_get(y, yv.data(), 0, yv.size() * sizeof(float));
    bool pass = ok;
    for (int64_t r = 0; r < m && pass; r++) {
        for (int64_t c = 0; c < n && pass; c++) {
            double ref = 0.0, mag = 0.0;
            for (int64_t i = 0; i < k; i++) {
                const double t = (double) wv[(size_t) (c * k + i)] * xv[(size_t) (r * k + i)];
                ref += t;
                mag += std::fabs(t);
            }
            const float v = yv[(size_t) (r * n + c)];
            pass          = std::isfinite(v) && std::fabs((double) v - ref) <= 1e-3 * mag;
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return pass;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string device = "HTP0";
    int         rounds = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = argv[++i];
        } else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) {
            rounds = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: canarytime [--device NAME] [--rounds N]\n");
            return 2;
        }
    }
    const char * canary = getenv("GGML_HEXAGON_CANARY");
    printf("canarytime: GGML_HEXAGON_CANARY=%s\n", canary ? canary : "(unset, 1)");

    const auto   t0      = clock_type::now();
    const size_t n_dev   = ggml_backend_dev_count();
    const double reg_ms  = ms_since(t0);
    printf("register: %.2f ms, %zu devices\n", reg_ms, n_dev);
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < n_dev; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        printf("device %zu: %s (%s)\n", i, ggml_backend_dev_name(d), ggml_backend_dev_description(d));
        if (device == ggml_backend_dev_name(d)) {
            dev = d;
        }
    }
    if (!dev) {
        printf("result: no device %s\n", device.c_str());
        return 3;
    }

    const auto     t1      = clock_type::now();
    ggml_backend_t be      = ggml_backend_dev_init(dev, nullptr);
    const double   open_ms = ms_since(t1);
    if (!be) {
        printf("result: the device %s gives no backend\n", device.c_str());
        return 1;
    }
    printf("open: %.2f ms\n", open_ms);
    printf("load: %.2f ms (register + open)\n", reg_ms + open_ms);

    int fails = 0;
    for (int r = 0; r < rounds; r++) {
        double     ms   = 0.0;
        const bool pass = work_graph(be, 256, 256, 8, 0x1234567u + (uint32_t) r, &ms);
        printf("work %d: MUL_MAT 256 x 256 x 8 %s, %.2f ms\n", r, pass ? "pass" : "FAIL", ms);
        fails += pass ? 0 : 1;
    }
    ggml_backend_free(be);
    printf("result: %s\n", fails ? "FAIL" : "pass");
    return fails ? 1 : 0;
}
