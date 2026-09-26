// vitprobe: run the vision encoder of mtmd on one image, time each encode, and write the image embeddings.
//
// Usage:
//   vitprobe -m MODEL --mmproj MMPROJ (--image FILE | --rgb FILE --size WxH) [--image-tokens N]
//            [--dev NAME|none] [-t N] [--reps N] [--rgb-out FILE] [--embd-out FILE]
//            [--dump DIR --dump-re REGEX] [--fa auto|on|off] [--log-ts]
//
// The tool loads only the vocabulary of the text model MODEL, because mtmd reads the special tokens from it. The
// length of an embedding comes from the projector file. The projector MMPROJ loads on the device --dev (none: the CPU),
// with the parameters of the app (llama_jni.cpp): no warmup, the flash attention AUTO, --image-tokens as
// image_max_tokens.
//
// The input image has two forms:
//   --image FILE  a JPEG or PNG file. stb_image decodes it, and the tool resizes the RGB bytes to the target size
//                 of the preprocessor with an exact integer box filter. The result does not depend on the CPU or
//                 on the compiler flags, thus the phone and the x86 oracle encode the same bytes.
//   --rgb FILE    the RGB bytes of an image of --size WxH, for example the --rgb-out file of an earlier run.
// mtmd resizes an image of the target size with a plain copy (img_tool::resize), thus the encoder gets the bytes
// of the file.
//
// Each of the --reps encodes calls mtmd_encode_chunk on the image chunk. The first one also reserves the graph
// and probes the flash attention (the first image of the app). The tool prints one TIME line per encode, and one
// REPDIFF line per encode after the first: the largest difference of its embeddings to those of the first encode.
//
// --embd-out writes the embeddings of the last encode as float32, row after row (n_tokens rows of n_embd). The
// EMBD line gives the shape, a hash of the bytes, and the count of values that are not finite.
// --dump DIR writes each graph node whose name matches --dump-re (std::regex_match, for example "layer_out-.*")
// to DIR/<name>.f32 as float32 during the last encode, and a line "<name> <type> ne0 ne1 ne2 ne3" to
// DIR/index.txt. With --dump - the tool writes no file: it prints one line "DUMP <name> nonfinite=N absmax=A
// rms=R hash=H" for each such node, in graph order. The callback splits the graph at each such node, thus a dump
// run is not a timing run.
// --log-ts gives each log line the time since the start, and STAMP lines around each encode.
//
// Time: the load of the projector plus reps times one encode. Memory: the projector, the compute buffers of the
// encoder, and one copy of the embeddings.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include "../common/fnv.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The side of one output token in pixels: the patch size 16 times the spatial merge 2 of the Qwen3-VL encoder.
constexpr int kAlign = 32;

struct options {
    std::string model;
    std::string mmproj;
    std::string image;
    std::string rgb;
    int         rgb_w        = 0;
    int         rgb_h        = 0;
    int         image_tokens = 0;
    std::string dev          = "HTP0";
    int         threads      = 4;
    int         reps         = 3;
    std::string rgb_out;
    std::string embd_out;
    std::string dump_dir;
    std::string dump_re;
    std::string fa = "auto";
    bool        log_ts = false;
};

std::mutex g_log_mutex;
int64_t    g_t0_us         = 0;
bool       g_log_ts        = false;
bool       g_log_line_open = false;

int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Write a log text to stderr. With --log-ts each line starts with the time since the start, "M.SS.mmm.uuu ".
// The caller holds g_log_mutex.
void log_write_locked(const char * text) {
    const char * p = text;
    while (*p != '\0') {
        const char * nl = std::strchr(p, '\n');
        const size_t n  = nl != nullptr ? (size_t) (nl - p) + 1 : std::strlen(p);
        if (g_log_ts && !g_log_line_open) {
            const int64_t t = now_us() - g_t0_us;
            std::fprintf(stderr, "%d.%02d.%03d.%03d ", (int) (t / 60000000), (int) (t / 1000000 % 60),
                         (int) (t / 1000 % 1000), (int) (t % 1000));
        }
        std::fwrite(p, 1, n, stderr);
        g_log_line_open = nl == nullptr;
        p += n;
    }
}

// The log callback of llama.cpp, ggml and mtmd: every level goes to stderr, thus the profile lines of the
// Hexagon backend (level DEBUG) are in the log.
void log_callback(ggml_log_level /*level*/, const char * text, void * /*user_data*/) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    log_write_locked(text);
}

// Write "vitprobe: STAMP <text>" to the log, on a line of its own.
void stamp(const char * fmt, ...) {
    char    buf[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log_line_open) {
        log_write_locked("\n");
    }
    log_write_locked("vitprobe: STAMP ");
    log_write_locked(buf);
    log_write_locked("\n");
}

// The hash over the bytes. The tool uses it to show that two runs hold the same bytes. The basis is
// the short one, thus a value of this tool is comparable with a value of this tool only, and the
// recorded embedding hashes of the encoder work hold that basis. O(n).
uint64_t fnv1a(const void * data, size_t n) {
    return fnv::hash64(data, n, fnv::kBasisShort);
}

// The target size of the preprocessor of Qwen3-VL for an image of w x h and a maximum of max_tokens tokens: the
// "smart resize" of calc_size_preserved_ratio (mtmd-image.cpp) with the minimum-pixel branch left out, because
// the tool checks the token count that mtmd gives against the result.
void target_size(int w, int h, int max_tokens, int & tw, int & th) {
    auto round_by = [](float x) { return static_cast<int>(std::round(x / static_cast<float>(kAlign))) * kAlign; };
    auto floor_by = [](float x) { return static_cast<int>(std::floor(x / static_cast<float>(kAlign))) * kAlign; };
    tw                   = std::max(kAlign, round_by((float) w));
    th                   = std::max(kAlign, round_by((float) h));
    const int max_pixels = max_tokens * kAlign * kAlign;
    if (max_tokens > 0 && tw * th > max_pixels) {
        const float beta = std::sqrt(static_cast<float>(h) * w / max_pixels);
        th               = std::max(kAlign, floor_by(h / beta));
        tw               = std::max(kAlign, floor_by(w / beta));
    }
}

// Resize RGB bytes of sw x sh to dw x dh with a box filter in exact integer arithmetic. Source pixel i spans
// [i * dw, (i + 1) * dw) and destination pixel o spans [o * sw, (o + 1) * sw) on a common axis, and the weight
// of i in o is the length of the overlap. Each output value is the weighted sum divided by sw * sh with the
// rounding to nearest (ties up). The result is the same on each CPU. O(sw * sh + dw * dh * the covered pixels).
std::vector<uint8_t> box_resize(const uint8_t * src, int sw, int sh, int dw, int dh) {
    // For each destination index: the first source index and the weights of the covered source indices.
    struct span {
        int                   first;
        std::vector<uint32_t> w;
    };
    auto spans = [](int s, int d) {
        std::vector<span> out(d);
        for (int o = 0; o < d; o++) {
            const int64_t lo = (int64_t) o * s, hi = (int64_t) (o + 1) * s;
            const int     i0 = (int) (lo / d), i1 = (int) ((hi - 1) / d);
            out[o].first     = i0;
            for (int i = i0; i <= i1; i++) {
                const int64_t a = std::max(lo, (int64_t) i * d), b = std::min(hi, (int64_t) (i + 1) * d);
                out[o].w.push_back((uint32_t) (b - a));
            }
        }
        return out;
    };
    const std::vector<span> hx = spans(sw, dw), vy = spans(sh, dh);
    // Horizontal pass: sums with the weight total sw, kept as integers.
    std::vector<uint64_t>   tmp((size_t) sh * dw * 3);
    for (int y = 0; y < sh; y++) {
        for (int o = 0; o < dw; o++) {
            for (int c = 0; c < 3; c++) {
                uint64_t acc = 0;
                for (size_t k = 0; k < hx[o].w.size(); k++) {
                    acc += (uint64_t) hx[o].w[k] * src[((size_t) y * sw + hx[o].first + k) * 3 + c];
                }
                tmp[((size_t) y * dw + o) * 3 + c] = acc;
            }
        }
    }
    // Vertical pass: the weight total of a value is sw * sh.
    const uint64_t       den = (uint64_t) sw * (uint64_t) sh;
    std::vector<uint8_t> dst((size_t) dw * dh * 3);
    for (int o = 0; o < dh; o++) {
        for (int x = 0; x < dw; x++) {
            for (int c = 0; c < 3; c++) {
                uint64_t acc = 0;
                for (size_t k = 0; k < vy[o].w.size(); k++) {
                    acc += (uint64_t) vy[o].w[k] * tmp[((size_t) (vy[o].first + k) * dw + x) * 3 + c];
                }
                dst[((size_t) o * dw + x) * 3 + c] = (uint8_t) std::min<uint64_t>(255, (2 * acc + den) / (2 * den));
            }
        }
    }
    return dst;
}

bool write_file(const std::string & path, const void * data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char *>(data), (std::streamsize) n);
    return (bool) f;
}

std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("vitprobe: cannot open " + path);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// The length of one output embedding of the projector: its key clip.vision.projection_dim. A text model that loads
// with only its vocabulary has no hyperparameters, thus the tool reads the projector file.
int64_t projection_dim(const std::string & path) {
    gguf_init_params p{ /* no_alloc */ true, /* ctx */ nullptr };
    gguf_context *   g = gguf_init_from_file(path.c_str(), p);
    if (g == nullptr) {
        throw std::runtime_error("vitprobe: " + path + " is not a GGUF file");
    }
    const int64_t key = gguf_find_key(g, "clip.vision.projection_dim");
    const int64_t dim = key >= 0 ? (int64_t) gguf_get_val_u32(g, key) : 0;
    gguf_free(g);
    if (dim <= 0) {
        throw std::runtime_error("vitprobe: " + path + " has no clip.vision.projection_dim");
    }
    return dim;
}

// The state of the tensor dump of --dump.
struct dump_state {
    std::string dir;
    std::regex  re;
    bool        armed = false;  // only the last encode writes files
    FILE *      index = nullptr;
};

// The eval callback of the scheduler. With ask set it tells whether the node must be seen; else it writes the
// node as float32. Only contiguous F32 and F16 nodes are written.
bool dump_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto & d = *static_cast<dump_state *>(user_data);
    if (!d.armed || !std::regex_match(t->name, d.re)) {
        return !ask;
    }
    if (ask) {
        return true;
    }
    if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16)) {
        std::fprintf(stderr, "vitprobe: dump skips %s (type %s, contiguous %d)\n", t->name, ggml_type_name(t->type),
                     (int) ggml_is_contiguous(t));
        return true;
    }
    const int64_t        n = ggml_nelements(t);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::vector<float> f((size_t) n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(f.data(), raw.data(), raw.size());
    } else {
        const auto * h = reinterpret_cast<const ggml_fp16_t *>(raw.data());
        for (int64_t i = 0; i < n; i++) {
            f[i] = ggml_fp16_to_fp32(h[i]);
        }
    }
    if (d.dir == "-") {
        size_t bad    = 0;
        double sumsq  = 0.0;
        float  absmax = 0.0f;
        for (float v : f) {
            if (!std::isfinite(v)) {
                bad++;
                continue;
            }
            sumsq += (double) v * v;
            absmax = std::max(absmax, std::fabs(v));
        }
        std::printf("DUMP %s nonfinite=%zu absmax=%.6g rms=%.6g hash=%016" PRIx64 "\n", t->name, bad, absmax,
                    std::sqrt(sumsq / (double) std::max<int64_t>(n, 1)), fnv1a(f.data(), f.size() * sizeof(float)));
        std::fflush(stdout);
        return true;
    }
    const std::string path = d.dir + "/" + t->name + ".f32";
    if (!write_file(path, f.data(), f.size() * sizeof(float))) {
        std::fprintf(stderr, "vitprobe: cannot write %s\n", path.c_str());
        return false;
    }
    std::fprintf(d.index, "%s %s %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "\n", t->name, ggml_type_name(t->type),
                 t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    return true;
}

[[noreturn]] void usage(const char * why) {
    std::fprintf(stderr,
                 "vitprobe: %s\n"
                 "usage: vitprobe -m MODEL --mmproj MMPROJ (--image FILE | --rgb FILE --size WxH) [--image-tokens N]\n"
                 "                [--dev NAME|none] [-t N] [--reps N] [--rgb-out FILE] [--embd-out FILE]\n"
                 "                [--dump DIR --dump-re REGEX] [--fa auto|on|off] [--log-ts]\n",
                 why);
    std::exit(2);
}

options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; i++) {
        const std::string a    = argv[i];
        auto              next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage(("the option " + a + " needs a value").c_str());
            }
            return argv[++i];
        };
        if (a == "-m") {
            o.model = next();
        } else if (a == "--mmproj") {
            o.mmproj = next();
        } else if (a == "--image") {
            o.image = next();
        } else if (a == "--rgb") {
            o.rgb = next();
        } else if (a == "--size") {
            const std::string s = next();
            if (std::sscanf(s.c_str(), "%dx%d", &o.rgb_w, &o.rgb_h) != 2 || o.rgb_w <= 0 || o.rgb_h <= 0) {
                usage("--size needs WxH");
            }
        } else if (a == "--image-tokens") {
            o.image_tokens = std::atoi(next().c_str());
        } else if (a == "--dev") {
            o.dev = next();
        } else if (a == "-t") {
            o.threads = std::atoi(next().c_str());
        } else if (a == "--reps") {
            o.reps = std::atoi(next().c_str());
        } else if (a == "--rgb-out") {
            o.rgb_out = next();
        } else if (a == "--embd-out") {
            o.embd_out = next();
        } else if (a == "--dump") {
            o.dump_dir = next();
        } else if (a == "--dump-re") {
            o.dump_re = next();
        } else if (a == "--fa") {
            o.fa = next();
        } else if (a == "--log-ts") {
            o.log_ts = true;
        } else {
            usage(("unknown option " + a).c_str());
        }
    }
    if (o.model.empty() || o.mmproj.empty()) {
        usage("-m and --mmproj are necessary");
    }
    if (o.image.empty() == o.rgb.empty()) {
        usage("give one of --image and --rgb");
    }
    if (!o.rgb.empty() && o.rgb_w == 0) {
        usage("--rgb needs --size");
    }
    if (o.reps < 1 || o.threads < 1) {
        usage("--reps and -t must be at least 1");
    }
    if (o.dump_dir.empty() != o.dump_re.empty()) {
        usage("--dump and --dump-re go together");
    }
    if (o.fa != "auto" && o.fa != "on" && o.fa != "off") {
        usage("--fa takes auto, on or off");
    }
    return o;
}

int run(const options & o) {
    g_t0_us  = now_us();
    g_log_ts = o.log_ts;
    llama_log_set(log_callback, nullptr);
    mtmd_helper_log_set(log_callback, nullptr);
    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only         = true;
    llama_model * model   = llama_model_load_from_file(o.model.c_str(), mp);
    if (model == nullptr) {
        std::fprintf(stderr, "vitprobe: the text model %s did not load\n", o.model.c_str());
        return 1;
    }

    ggml_backend_dev_t dev = nullptr;
    if (o.dev != "none") {
        dev = ggml_backend_dev_by_name(o.dev.c_str());
        if (dev == nullptr) {
            std::fprintf(stderr, "vitprobe: the device %s is not available\n", o.dev.c_str());
            return 1;
        }
    }
    dump_state dump;
    if (!o.dump_dir.empty()) {
        dump.dir   = o.dump_dir;
        dump.re    = std::regex(o.dump_re);
        dump.index = o.dump_dir == "-" ? nullptr : std::fopen((o.dump_dir + "/index.txt").c_str(), "w");
        if (dump.index == nullptr && o.dump_dir != "-") {
            std::fprintf(stderr, "vitprobe: cannot write %s/index.txt\n", o.dump_dir.c_str());
            return 1;
        }
    }
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu             = dev != nullptr;
    cp.device              = dev;
    cp.n_threads           = o.threads;
    cp.print_timings       = false;
    cp.warmup              = false;
    cp.image_max_tokens    = o.image_tokens > 0 ? o.image_tokens : -1;
    cp.flash_attn_type     = o.fa == "on"  ? LLAMA_FLASH_ATTN_TYPE_ENABLED :
                             o.fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED :
                                             LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (!o.dump_dir.empty()) {
        cp.cb_eval           = dump_callback;
        cp.cb_eval_user_data = &dump;
    }
    const int64_t  t_load = now_us();
    mtmd_context * ctx    = mtmd_init_from_file(o.mmproj.c_str(), model, cp);
    if (ctx == nullptr) {
        std::fprintf(stderr, "vitprobe: the projector %s did not load\n", o.mmproj.c_str());
        return 1;
    }
    std::printf("LOAD mmproj_ms=%.1f dev=%s\n", (now_us() - t_load) / 1000.0, o.dev.c_str());

    // The RGB bytes of the target size.
    std::vector<uint8_t> rgb;
    int                  w = 0, h = 0;
    if (!o.image.empty()) {
        mtmd_helper_init_opt       hopt = mtmd_helper_init_opt_default();
        mtmd_helper_bitmap_wrapper bw   = mtmd_helper_bitmap_init_from_file(ctx, o.image.c_str(), false, hopt);
        if (bw.bitmap == nullptr || mtmd_bitmap_is_audio(bw.bitmap)) {
            std::fprintf(stderr, "vitprobe: %s did not decode as an image\n", o.image.c_str());
            return 1;
        }
        const int sw = (int) mtmd_bitmap_get_nx(bw.bitmap), sh = (int) mtmd_bitmap_get_ny(bw.bitmap);
        target_size(sw, sh, o.image_tokens, w, h);
        rgb = box_resize(mtmd_bitmap_get_data(bw.bitmap), sw, sh, w, h);
        std::printf("IMAGE file=%s decoded=%dx%d decoded_hash=%016" PRIx64 " target=%dx%d\n", o.image.c_str(), sw, sh,
                    fnv1a(mtmd_bitmap_get_data(bw.bitmap), mtmd_bitmap_get_n_bytes(bw.bitmap)), w, h);
        mtmd_bitmap_free(bw.bitmap);
    } else {
        rgb = read_file(o.rgb);
        w   = o.rgb_w;
        h   = o.rgb_h;
        if (rgb.size() != (size_t) w * h * 3) {
            std::fprintf(stderr, "vitprobe: %s has %zu bytes, not %d x %d x 3\n", o.rgb.c_str(), rgb.size(), w, h);
            return 1;
        }
    }
    std::printf("RGB size=%dx%d hash=%016" PRIx64 "\n", w, h, fnv1a(rgb.data(), rgb.size()));
    if (!o.rgb_out.empty() && !write_file(o.rgb_out, rgb.data(), rgb.size())) {
        std::fprintf(stderr, "vitprobe: cannot write %s\n", o.rgb_out.c_str());
        return 1;
    }

    mtmd_bitmap * bitmap = mtmd_bitmap_init((uint32_t) w, (uint32_t) h, rgb.data());
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const std::string   prompt = mtmd_default_marker();
    mtmd_input_text     text{ prompt.c_str(), prompt.size(), false, true };
    const mtmd_bitmap * bitmaps[1] = { bitmap };
    if (mtmd_tokenize(ctx, chunks, &text, bitmaps, 1) != 0) {
        std::fprintf(stderr, "vitprobe: mtmd_tokenize failed\n");
        return 1;
    }
    const mtmd_input_chunk * chunk = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            chunk = c;
        }
    }
    if (chunk == nullptr) {
        std::fprintf(stderr, "vitprobe: the prompt has no image chunk\n");
        return 1;
    }
    const size_t  n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    const int64_t n_embd   = projection_dim(o.mmproj);
    std::printf("CHUNK tokens=%zu grid=%dx%d patches=%d n_embd=%" PRId64 "\n", n_tokens, w / kAlign, h / kAlign,
                (w / 16) * (h / 16), n_embd);
    if (n_tokens != (size_t) (w / kAlign) * (h / kAlign)) {
        std::fprintf(stderr, "vitprobe: mtmd gives %zu tokens, not the %d of the grid; the target size differs\n",
                     n_tokens, (w / kAlign) * (h / kAlign));
        return 1;
    }

    std::vector<float> first, last;
    const size_t       n_vals = n_tokens * (size_t) n_embd;
    for (int r = 1; r <= o.reps; r++) {
        dump.armed = r == o.reps;
        stamp("encode-begin rep=%d", r);
        const int64_t t0 = now_us();
        const int32_t rc = mtmd_encode_chunk(ctx, chunk);
        const int64_t t1 = now_us();
        stamp("encode-end rep=%d", r);
        if (rc != 0) {
            std::fprintf(stderr, "vitprobe: mtmd_encode_chunk gave %d\n", rc);
            return 1;
        }
        const float * embd = mtmd_get_output_embd(ctx);
        last.assign(embd, embd + n_vals);
        double maxdiff = 0.0;
        if (r == 1) {
            first = last;
        } else {
            for (size_t i = 0; i < n_vals; i++) {
                maxdiff = std::max(maxdiff, (double) std::fabs(last[i] - first[i]));
            }
        }
        std::printf("TIME encode rep=%d ms=%.2f\n", r, (t1 - t0) / 1000.0);
        if (r > 1) {
            std::printf("REPDIFF rep=%d maxabs=%.9g\n", r, maxdiff);
        }
        std::fflush(stdout);
    }

    size_t bad    = 0;
    double sumsq  = 0.0;
    float  absmax = 0.0f;
    for (float v : last) {
        if (!std::isfinite(v)) {
            bad++;
            continue;
        }
        sumsq += (double) v * v;
        absmax = std::max(absmax, std::fabs(v));
    }
    std::printf("EMBD rows=%zu cols=%" PRId64 " hash=%016" PRIx64 " nonfinite=%zu rms=%.6g absmax=%.6g\n", n_tokens,
                n_embd, fnv1a(last.data(), last.size() * sizeof(float)), bad, std::sqrt(sumsq / (double) n_vals),
                absmax);
    if (!o.embd_out.empty() && !write_file(o.embd_out, last.data(), last.size() * sizeof(float))) {
        std::fprintf(stderr, "vitprobe: cannot write %s\n", o.embd_out.c_str());
        return 1;
    }
    if (dump.index != nullptr) {
        std::fclose(dump.index);
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);
    mtmd_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return bad == 0 ? 0 : 3;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse(argc, argv));
    } catch (const std::exception & e) {
        std::fprintf(stderr, "vitprobe: %s\n", e.what());
        return 1;
    }
}
