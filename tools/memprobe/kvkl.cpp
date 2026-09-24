/**
 * kvkl: the KL divergence of the logits that a KV cache type (and a device) gives, as a
 * function of the depth in the context.
 *
 * The tool reads the first n_ctx tokens of a text. It decodes the positions [0, n_ctx - tail)
 * in batches of n_batch with logits at every stride-th position (the prefill path), then the
 * last tail positions one token at a time with logits at each (the decode path, at the depth
 * n_ctx - tail). Three modes:
 *
 *   pair:  two contexts in one process, the reference with an F16 K and V cache and the test
 *          with -ctk and -ctv. Only the KV cache type differs.
 *   --save-base FILE: one context with -ctk and -ctv. The log-probabilities of each sampled
 *          position go to FILE (16 bits for each token, as llama-perplexity stores them).
 *   --base FILE: one context with -ctk and -ctv, compared with FILE of an earlier run (the
 *          naive x86 oracle with F16 KV, for example). The text, n_ctx, stride and tail must
 *          be the same.
 *
 * For each sampled position the tool computes KL(reference || test) over the vocabulary and
 * whether the two top-1 tokens agree, and prints the statistics for each depth bin and for
 * the decode tail. --worst N also prints the N positions with the largest KL. --end P stops the
 * prefill at the position P and skips the decode tail (a shorter run for the rows before P).
 *
 *   kvkl -m MODEL -f TEXT [-c N_CTX] [-b N_BATCH] [-ctk TYPE] [-ctv TYPE] [-dev NAME|none]
 *        [-t THREADS] [--stride S] [--tail N] [--save-base FILE | --base FILE] [--worst N] [--end P]
 *
 * Memory: one or two KV caches of n_ctx cells, compute buffers with n_batch / S output rows,
 * and in --base mode the base file in RAM (n_positions x (n_vocab x 2 + 12) bytes).
 * Time: O(n_ctx) decodes plus O(n_positions x n_vocab) for the KL.
 * It is a measurement tool, not part of the app.
 */

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr char     kMagic[8]  = {'K', 'V', 'K', 'L', 'B', 'A', 'S', 'E'};
constexpr uint32_t kVersion   = 1;

ggml_type parse_type(const std::string & s) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * name = ggml_type_name((ggml_type) i);
        if (name != nullptr && s == name) return (ggml_type) i;
    }
    fprintf(stderr, "unknown type %s\n", s.c_str());
    std::exit(2);
}

/** The log-softmax of n logits into out, in double. O(n). */
void log_softmax(const float * logits, int n, std::vector<double> & out) {
    out.resize(n);
    double mx = logits[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, (double) logits[i]);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += std::exp((double) logits[i] - mx);
    const double lse = mx + std::log(sum);
    for (int i = 0; i < n; ++i) out[i] = (double) logits[i] - lse;
}

/**
 * One stored row: the log-probabilities as lp = lo + scale * q with 16-bit q, the scheme of
 * llama-perplexity (its two identical runs give a maximum KL of about 7e-5).
 */
struct BaseRow {
    int32_t               pos = 0;
    float                 lo = 0.0f;
    float                 scale = 0.0f;
    std::vector<uint16_t> q;

    void encode(int32_t p, const std::vector<double> & lp) {
        pos = p;
        double mn = lp[0], mx = lp[0];
        for (double v : lp) { mn = std::min(mn, v); mx = std::max(mx, v); }
        lo    = (float) mn;
        scale = (float) ((mx - mn) / 65535.0);
        q.resize(lp.size());
        for (size_t i = 0; i < lp.size(); ++i) {
            const double x = scale > 0 ? (lp[i] - lo) / scale : 0.0;
            q[i] = (uint16_t) std::min(65535.0, std::max(0.0, std::floor(x + 0.5)));
        }
    }

    void decode(std::vector<double> & lp) const {
        lp.resize(q.size());
        for (size_t i = 0; i < q.size(); ++i) lp[i] = (double) lo + (double) scale * q[i];
        // The rounding moves the sum of the probabilities off 1: normalize again.
        double mx = lp[0];
        for (double v : lp) mx = std::max(mx, v);
        double sum = 0.0;
        for (double v : lp) sum += std::exp(v - mx);
        const double lse = mx + std::log(sum);
        for (double & v : lp) v -= lse;
    }
};

struct Sample {
    int    pos;
    bool   tail;
    double kl;
    bool   same_top1;
    int    top_ref;   // The top-1 token of the reference.
    int    top_test;  // The top-1 token of the test.
    double lp[4];     // The log-probabilities ref(top_ref), test(top_ref), ref(top_test), test(top_test).
};

/** KL(ref || test) and the top-1 tokens of two log-probability rows. O(n). */
Sample compare(int pos, bool tail, const std::vector<double> & ref, const std::vector<double> & test) {
    double kl = 0.0;
    size_t a_ref = 0, a_test = 0;
    for (size_t v = 0; v < ref.size(); ++v) {
        kl += std::exp(ref[v]) * (ref[v] - test[v]);
        if (ref[v] > ref[a_ref]) a_ref = v;
        if (test[v] > test[a_test]) a_test = v;
    }
    return {pos, tail, std::max(0.0, kl), a_ref == a_test, (int) a_ref, (int) a_test,
            {ref[a_ref], test[a_ref], ref[a_test], test[a_test]}};
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, text_path, dev_name = "none", save_base, base_path;
    int n_ctx = 16384, n_batch = 1024, n_threads = 8, stride = 16, tail = 0, n_worst = 0, n_end = 0;
    ggml_type tk = GGML_TYPE_Q8_0, tv = GGML_TYPE_Q8_0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value of %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m") model_path = next();
        else if (a == "-f") text_path = next();
        else if (a == "-c") n_ctx = std::atoi(next().c_str());
        else if (a == "-b") n_batch = std::atoi(next().c_str());
        else if (a == "-ctk") tk = parse_type(next());
        else if (a == "-ctv") tv = parse_type(next());
        else if (a == "-dev") dev_name = next();
        else if (a == "-t") n_threads = std::atoi(next().c_str());
        else if (a == "--stride") stride = std::atoi(next().c_str());
        else if (a == "--tail") tail = std::atoi(next().c_str());
        else if (a == "--save-base") save_base = next();
        else if (a == "--base") base_path = next();
        else if (a == "--worst") n_worst = std::atoi(next().c_str());
        else if (a == "--end") n_end = std::atoi(next().c_str());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || text_path.empty() || stride < 1 || n_batch % stride != 0 || tail < 0 ||
        tail >= n_ctx || n_worst < 0 || n_end < 0 || (!save_base.empty() && !base_path.empty()) ||
        (n_end > 0 && !save_base.empty())) {
        fprintf(stderr, "usage: kvkl -m MODEL -f TEXT [-c N] [-b N] [-ctk T] [-ctv T] [-dev NAME|none] [-t N] "
                        "[--stride S] [--tail N] [--save-base FILE | --base FILE] [--worst N] [--end P], with "
                        "N_BATCH a multiple of S, and no --end with --save-base\n");
        return 2;
    }
    const bool pair = save_base.empty() && base_path.empty();
    const auto t_start = std::chrono::steady_clock::now();

    ggml_backend_load_all();
    llama_backend_init();

    std::vector<ggml_backend_dev_t> devices;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = dev_name == "none" ? 0 : 999;
    if (dev_name != "none") {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
        if (dev == nullptr) { fprintf(stderr, "no device %s\n", dev_name.c_str()); return 1; }
        devices = {dev, nullptr};
        mp.devices = devices.data();
    }
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (model == nullptr) { fprintf(stderr, "the model did not load\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::ifstream f(text_path);
    std::stringstream ss;
    ss << f.rdbuf();
    // A prefix of the text is enough: about 4 characters for each token.
    std::vector<llama_token> tokens = common_tokenize(vocab, ss.str().substr(0, (size_t) n_ctx * 8), true, false);
    if ((int) tokens.size() < n_ctx) {
        fprintf(stderr, "the text gives %zu tokens, less than n_ctx %d\n", tokens.size(), n_ctx);
        return 1;
    }
    tokens.resize(n_ctx);
    uint64_t tok_hash = 1469598103934665603ull;
    for (llama_token t : tokens) { tok_hash ^= (uint64_t) (uint32_t) t; tok_hash *= 1099511628211ull; }

    // The base of --base mode, read before the model runs.
    std::vector<BaseRow> base;
    if (!base_path.empty()) {
        std::ifstream in(base_path, std::ios::binary);
        char magic[8];
        uint32_t version = 0, b_ctx = 0, b_vocab = 0, b_stride = 0, b_tail = 0, b_n = 0;
        uint64_t b_hash = 0;
        in.read(magic, 8);
        in.read((char *) &version, 4); in.read((char *) &b_ctx, 4); in.read((char *) &b_vocab, 4);
        in.read((char *) &b_stride, 4); in.read((char *) &b_tail, 4); in.read((char *) &b_n, 4);
        in.read((char *) &b_hash, 8);
        if (!in || memcmp(magic, kMagic, 8) != 0 || version != kVersion || (int) b_ctx != n_ctx ||
            (int) b_vocab != n_vocab || (int) b_stride != stride || (int) b_tail != tail || b_hash != tok_hash) {
            fprintf(stderr, "the base %s does not match this run (n_ctx %u, vocab %u, stride %u, tail %u)\n",
                    base_path.c_str(), b_ctx, b_vocab, b_stride, b_tail);
            return 1;
        }
        base.resize(b_n);
        for (BaseRow & r : base) {
            r.q.resize(n_vocab);
            in.read((char *) &r.pos, 4); in.read((char *) &r.lo, 4); in.read((char *) &r.scale, 4);
            in.read((char *) r.q.data(), (std::streamsize) n_vocab * 2);
        }
        if (!in) { fprintf(stderr, "the base %s is short\n", base_path.c_str()); return 1; }
    }

    auto make_ctx = [&](ggml_type k, ggml_type v) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = (uint32_t) n_ctx;
        cp.n_batch         = (uint32_t) n_batch;
        cp.n_ubatch        = (uint32_t) n_batch;
        cp.n_seq_max       = 1;
        cp.kv_unified      = true;
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
        cp.type_k          = k;
        cp.type_v          = v;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        cp.n_outputs_max   = (uint32_t) (n_batch / stride);
        return llama_init_from_model(model, cp);
    };
    llama_context * test = make_ctx(tk, tv);
    llama_context * ref  = pair ? make_ctx(GGML_TYPE_F16, GGML_TYPE_F16) : nullptr;
    if (test == nullptr || (pair && ref == nullptr)) { fprintf(stderr, "a context did not initialize\n"); return 1; }

    std::vector<BaseRow> saved;
    std::vector<Sample> samples;
    std::vector<double> lp_ref, lp_test;
    size_t base_next = 0;
    // One row of logits of each context at batch index ib and position pos: compare or save it.
    auto take_row = [&](int ib, int pos, bool is_tail) -> bool {
        log_softmax(llama_get_logits_ith(test, ib), n_vocab, lp_test);
        if (!save_base.empty()) {
            saved.emplace_back();
            saved.back().encode(pos, lp_test);
            return true;
        }
        if (pair) {
            log_softmax(llama_get_logits_ith(ref, ib), n_vocab, lp_ref);
        } else {
            if (base_next >= base.size() || base[base_next].pos != pos) {
                fprintf(stderr, "the base has no row for position %d\n", pos);
                return false;
            }
            base[base_next++].decode(lp_ref);
        }
        samples.push_back(compare(pos, is_tail, lp_ref, lp_test));
        return true;
    };

    // --end P stops the prefill at the position P and skips the decode tail. The batches before P
    // are the batches of a full run, thus a P at a batch boundary gives the rows of a full run.
    const bool full      = n_end == 0 || n_end >= n_ctx - tail;
    const int  n_prefill = full ? n_ctx - tail : n_end;
    const int  n_last    = full ? n_ctx : n_end;
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int i0 = 0; i0 < n_prefill; i0 += n_batch) {
        const int count = std::min(n_batch, n_prefill - i0);
        std::vector<int> out_ib;
        for (int j = 0; j < count; ++j) {
            batch.token[j]     = tokens[i0 + j];
            batch.pos[j]       = i0 + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = ((i0 + j) % stride) == stride - 1;
            if (batch.logits[j]) out_ib.push_back(j);
        }
        batch.n_tokens = count;
        if (llama_decode(test, batch) != 0 || (pair && llama_decode(ref, batch) != 0)) {
            fprintf(stderr, "decode failed at %d\n", i0);
            return 1;
        }
        // llama_get_logits_ith takes the index of the token in the batch.
        for (int ib : out_ib) {
            if (!take_row(ib, i0 + ib, false)) return 1;
        }
        fprintf(stderr, "kvkl: %d / %d\n", i0 + count, n_ctx);
    }
    for (int p = n_prefill; p < n_last; ++p) {
        batch.n_tokens     = 1;
        batch.token[0]     = tokens[p];
        batch.pos[0]       = p;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        if (llama_decode(test, batch) != 0 || (pair && llama_decode(ref, batch) != 0)) {
            fprintf(stderr, "decode failed at %d\n", p);
            return 1;
        }
        if (!take_row(0, p, true)) return 1;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    if (!save_base.empty()) {
        std::ofstream out(save_base, std::ios::binary);
        const uint32_t hdr[6] = {kVersion, (uint32_t) n_ctx, (uint32_t) n_vocab, (uint32_t) stride, (uint32_t) tail,
                                 (uint32_t) saved.size()};
        out.write(kMagic, 8);
        out.write((const char *) hdr, sizeof(hdr));
        out.write((const char *) &tok_hash, 8);
        for (const BaseRow & r : saved) {
            out.write((const char *) &r.pos, 4); out.write((const char *) &r.lo, 4); out.write((const char *) &r.scale, 4);
            out.write((const char *) r.q.data(), (std::streamsize) n_vocab * 2);
        }
        printf("kvkl base %s: %zu rows, n_ctx=%d stride=%d tail=%d type_k=%s type_v=%s dev=%s, %.1f s\n",
               save_base.c_str(), saved.size(), n_ctx, stride, tail, ggml_type_name(tk), ggml_type_name(tv),
               dev_name.c_str(), seconds);
        return out ? 0 : 1;
    }

    printf("kvkl model=%s n_ctx=%d type_k=%s type_v=%s dev=%s stride=%d tail=%d end=%d reference=%s samples=%zu "
           "time=%.1fs\n",
           model_path.c_str(), n_ctx, ggml_type_name(tk), ggml_type_name(tv), dev_name.c_str(), stride, tail, n_last,
           pair ? "f16-kv-same-process" : base_path.c_str(), samples.size(), seconds);
    printf("%-18s %6s %12s %12s %12s %8s\n", "depth", "n", "mean_kl", "p99_kl", "max_kl", "top1_%");
    auto report = [&](int lo, int hi, int want_tail, const char * label) {
        std::vector<double> kls;
        int same = 0;
        for (const Sample & s : samples) {
            if (s.pos < lo || s.pos >= hi || (want_tail >= 0 && (int) s.tail != want_tail)) continue;
            kls.push_back(s.kl);
            same += s.same_top1;
        }
        if (kls.empty()) return;
        std::sort(kls.begin(), kls.end());
        double sum = 0.0;
        for (double k : kls) sum += k;
        const double p99 = kls[std::min(kls.size() - 1, (size_t) std::floor(0.99 * (kls.size() - 1) + 0.5))];
        printf("%-18s %6zu %12.6f %12.6f %12.6f %8.2f\n", label, kls.size(), sum / kls.size(), p99, kls.back(),
               100.0 * same / kls.size());
    };
    // The depth bins of the prefill path, then the decode tail, then all.
    const int edges[] = {0, 1024, 4096, 8192, 16384, 32768, 65536, 1 << 30};
    for (size_t b = 0; b + 1 < sizeof(edges) / sizeof(edges[0]); ++b) {
        if (edges[b] >= n_prefill) break;
        char label[40];
        snprintf(label, sizeof(label), "[%d,%d)", edges[b], std::min(edges[b + 1], n_prefill));
        report(edges[b], edges[b + 1], 0, label);
    }
    if (tail > 0 && full) {
        char label[40];
        snprintf(label, sizeof(label), "decode[%d,%d)", n_prefill, n_ctx);
        report(n_prefill, n_ctx, 1, label);
    }
    report(0, 1 << 30, -1, "all");

    // The n_worst rows with the largest KL, and the top-1 token of each side with its
    // probability in the reference and in the test. O(n_samples log n_samples).
    if (n_worst > 0) {
        std::vector<Sample> worst = samples;
        std::sort(worst.begin(), worst.end(), [](const Sample & a, const Sample & b) { return a.kl > b.kl; });
        worst.resize(std::min(worst.size(), (size_t) n_worst));
        printf("%-6s %6s %12s %8s %10s %10s %8s %10s %10s\n", "worst", "pos", "kl", "tok_ref", "p_ref", "p_test",
               "tok_test", "p_ref", "p_test");
        for (const Sample & s : worst) {
            printf("%-6s %6d %12.6f %8d %10.6f %10.6f %8d %10.6f %10.6f\n", s.tail ? "decode" : "batch", s.pos, s.kl,
                   s.top_ref, std::exp(s.lp[0]), std::exp(s.lp[1]), s.top_test, std::exp(s.lp[2]), std::exp(s.lp[3]));
        }
    }

    llama_batch_free(batch);
    if (ref) llama_free(ref);
    llama_free(test);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
