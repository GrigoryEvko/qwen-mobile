// fuzz_model_load: the model loader of llama.cpp with vocab_only = true on any byte string.
//
// The loader reads the GGUF metadata, the hyperparameters of the architecture
// and the full tokenizer (tokens, scores, types, merges, the special ids, the
// chat template), but no tensor data. This is the part of a model file that
// the app parses before it maps the weights.
//
// The input goes into an anonymous memory file (memfd). The last input byte
// selects the path: bit 0 = 0 loads with llama_model_load_from_file through
// /proc/self/fd with mmap, bit 0 = 1 loads with llama_model_load_from_file_ptr.
//
// For a loaded vocabulary, the harness checks these properties:
//   - each special token id is LLAMA_TOKEN_NULL or less than n_vocab
//   - token_to_piece, get_text, get_score and get_attr accept each token id
//     below min(n_vocab, 4096), and the pieces fit their reported size
//   - tokenize gives token ids in [0, n_vocab) for a fixed set of texts, and
//     detokenize accepts those ids
//   - the chat template of the model is a C string
//
// The switches of the known findings: FUZZ_GGUF_KNOWN_ENUM_LOAD (gguf-enum-load),
// FUZZ_MODEL_LOAD_KNOWN_DUP_TOKENS, _BYTE_TYPE, _FTYPE, _META_LEAK (known_bad_file),
// _NONE_VOCAB and _SPM_BYTES (check_vocab).

#include "fuzz_common.h"

#include "gguf.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace {

/** The texts that the harness tokenizes with each loaded vocabulary. */
const char * const kTexts[] = {
    "Hello world",
    "  multiple   spaces\n\nand lines\t",
    "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xe4\xbd\xa0\xe5\xa5\xbd \xf0\x9f\x99\x82",
    "<|im_start|>user\nhi<|im_end|>",
    "\xff\xfe\xc3\x28 invalid",
    "",
};

/** The switches of the known findings that a file can show during the load. */
struct KnownSwitches {
    bool dup;        // FUZZ_MODEL_LOAD_KNOWN_DUP_TOKENS: vocab-dup-tokens
    bool byte;       // FUZZ_MODEL_LOAD_KNOWN_BYTE_TYPE: vocab-byte-type
    bool ftype;      // FUZZ_MODEL_LOAD_KNOWN_FTYPE: loader-ftype-enum
    bool meta_leak;  // FUZZ_MODEL_LOAD_KNOWN_META_LEAK: loader-meta-leak
};

/**
 * True when the GGUF metadata in data shows a known finding of a switch that is on:
 *   - dup:       a token text two times in tokenizer.ggml.tokens (vocab-dup-tokens)
 *   - byte:      a token of type BYTE (6) in a WPM vocabulary, or a BYTE token with a text of
 *                fewer than 5 bytes (vocab-byte-type: token_to_byte aborts or throws during the load)
 *   - ftype:     general.file_type is an integer above LLAMA_FTYPE_GUESSED (loader-ftype-enum: the
 *                loader converts it to llama_ftype, a UBSan report for a value outside the enum range)
 *   - meta_leak: general.architecture is not a string (loader-meta-leak: get_key throws before the
 *                loader owns the ggml context of the metadata, and the context leaks)
 * The caller must first make sure that the GGUF reader can read data without a known finding
 * (fuzz::gguf_bad_enum). O(n log n) in the count of tokens.
 */
bool known_bad_file(const uint8_t * data, size_t size, const KnownSwitches & sw) {
    gguf_context * ctx = gguf_init_from_buffer(data, size, { /*no_alloc =*/ true, /*ctx =*/ nullptr });
    if (ctx == nullptr) {
        return false;
    }
    const bool dup = sw.dup, byte = sw.byte;
    bool bad = false;
    const int64_t fkey = gguf_find_key(ctx, "general.file_type");
    if (sw.ftype && fkey >= 0) {
        switch (gguf_get_kv_type(ctx, fkey)) {
            case GGUF_TYPE_UINT32: bad = gguf_get_val_u32(ctx, fkey) > (uint32_t) LLAMA_FTYPE_GUESSED; break;
            case GGUF_TYPE_INT32:  bad = gguf_get_val_i32(ctx, fkey) < 0 || gguf_get_val_i32(ctx, fkey) > LLAMA_FTYPE_GUESSED; break;
            case GGUF_TYPE_UINT64: bad = gguf_get_val_u64(ctx, fkey) > (uint64_t) LLAMA_FTYPE_GUESSED; break;
            case GGUF_TYPE_INT64:  bad = gguf_get_val_i64(ctx, fkey) < 0 || gguf_get_val_i64(ctx, fkey) > LLAMA_FTYPE_GUESSED; break;
            default: break;
        }
    }
    const int64_t akey = gguf_find_key(ctx, "general.architecture");
    if (sw.meta_leak && akey >= 0 && gguf_get_kv_type(ctx, akey) != GGUF_TYPE_STRING) {
        bad = true;
    }
    const int64_t key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    const bool tokens_ok = key >= 0 && gguf_get_kv_type(ctx, key) == GGUF_TYPE_ARRAY && gguf_get_arr_type(ctx, key) == GGUF_TYPE_STRING;
    if (dup && tokens_ok) {
        std::set<std::string> seen;
        const size_t n = gguf_get_arr_n(ctx, key);
        for (size_t i = 0; i < n && !bad; ++i) {
            bad = !seen.insert(gguf_get_arr_str(ctx, key, i)).second;
        }
    }
    const int64_t tkey = gguf_find_key(ctx, "tokenizer.ggml.token_type");
    const int64_t mkey = gguf_find_key(ctx, "tokenizer.ggml.model");
    if (byte && !bad && tokens_ok && tkey >= 0 && gguf_get_kv_type(ctx, tkey) == GGUF_TYPE_ARRAY &&
        gguf_get_arr_type(ctx, tkey) == GGUF_TYPE_INT32) {
        const bool wpm = mkey >= 0 && gguf_get_kv_type(ctx, mkey) == GGUF_TYPE_STRING &&
                         strcmp(gguf_get_val_str(ctx, mkey), "bert") == 0;
        const int32_t * types = (const int32_t *) gguf_get_arr_data(ctx, tkey);
        const size_t n = std::min(gguf_get_arr_n(ctx, tkey), gguf_get_arr_n(ctx, key));
        for (size_t i = 0; i < n && !bad; ++i) {
            bad = types[i] == 6 && (wpm || strlen(gguf_get_arr_str(ctx, key, i)) < 5);
        }
    }
    gguf_free(ctx);
    return bad;
}

/** Check the vocabulary of a loaded model. Stop with fuzz::fail() on a broken property. */
void check_vocab(const llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab < 0) {
        fuzz::fail("n_vocab is negative: %d", n_vocab);
    }
    // A vocabulary of type NONE ("no_vocab") has token ids, but llama_vocab_get_text and the
    // other token functions stop on GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE) (finding
    // vocab-none-assert). FUZZ_MODEL_LOAD_KNOWN_NONE_VOCAB=1 skips such a model.
    static const bool known_none = fuzz::env_long("FUZZ_MODEL_LOAD_KNOWN_NONE_VOCAB", 0) != 0;
    if (known_none && llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        return;
    }

    const llama_token specials[] = {
        llama_vocab_bos(vocab), llama_vocab_eos(vocab), llama_vocab_eot(vocab), llama_vocab_sep(vocab),
        llama_vocab_nl(vocab),  llama_vocab_pad(vocab), llama_vocab_mask(vocab),
        llama_vocab_fim_pre(vocab), llama_vocab_fim_suf(vocab), llama_vocab_fim_mid(vocab),
        llama_vocab_fim_pad(vocab), llama_vocab_fim_rep(vocab), llama_vocab_fim_sep(vocab),
    };
    for (const llama_token id : specials) {
        if (id != LLAMA_TOKEN_NULL && (id < 0 || id >= n_vocab)) {
            fuzz::fail("a special token id is %d, outside [0, %d)", id, n_vocab);
        }
        if (id != LLAMA_TOKEN_NULL) {
            char buf[256];
            (void) llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
            (void) llama_vocab_is_eog(vocab, id);
            (void) llama_vocab_is_control(vocab, id);
        }
    }

    const int32_t n_check = n_vocab < 4096 ? n_vocab : 4096;
    std::vector<char> piece(64);
    for (llama_token id = 0; id < n_check; ++id) {
        (void) llama_vocab_get_text(vocab, id);
        (void) llama_vocab_get_score(vocab, id);
        (void) llama_vocab_get_attr(vocab, id);
        for (const bool special : { false, true }) {
            int32_t n = llama_token_to_piece(vocab, id, piece.data(), (int32_t) piece.size(), 0, special);
            if (n < 0) {
                piece.resize((size_t) -n);
                const int32_t n2 = llama_token_to_piece(vocab, id, piece.data(), (int32_t) piece.size(), 0, special);
                if (n2 != -n) {
                    fuzz::fail("token_to_piece(%d) reports %d bytes, then writes %d bytes", id, -n, n2);
                }
            }
        }
    }

    // An SPM vocabulary without the 256 byte tokens <0x00>..<0xFF> loads, and then
    // llama_tokenize throws std::out_of_range from byte_to_token (finding spm-byte-tokens).
    // FUZZ_MODEL_LOAD_KNOWN_SPM_BYTES=1 skips the tokenizer calls for such a vocabulary.
    static const bool known_spm = fuzz::env_long("FUZZ_MODEL_LOAD_KNOWN_SPM_BYTES", 0) != 0;
    if (known_spm && llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM) {
        std::set<std::string> texts;
        for (llama_token id = 0; id < n_vocab; ++id) {
            texts.insert(llama_vocab_get_text(vocab, id));
        }
        for (int b = 0; b < 256; ++b) {
            char name[8];
            snprintf(name, sizeof(name), "<0x%02X>", b);
            if (texts.count(name) == 0) {
                return;
            }
        }
    }

    std::vector<llama_token> tokens(512);
    for (const char * text : kTexts) {
        for (const bool parse_special : { false, true }) {
            const int32_t len = (int32_t) strlen(text);
            int32_t n = llama_tokenize(vocab, text, len, tokens.data(), (int32_t) tokens.size(), true, parse_special);
            if (n < 0 && n != INT32_MIN) {
                tokens.resize((size_t) -n);
                n = llama_tokenize(vocab, text, len, tokens.data(), (int32_t) tokens.size(), true, parse_special);
            }
            if (n < 0) {
                continue;
            }
            for (int32_t i = 0; i < n; ++i) {
                if (tokens[i] < 0 || tokens[i] >= n_vocab) {
                    fuzz::fail("tokenize gives the id %d, outside [0, %d)", tokens[i], n_vocab);
                }
            }
            char out[4096];
            (void) llama_detokenize(vocab, tokens.data(), n, out, sizeof(out), false, true);
        }
    }

    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl != nullptr) {
        (void) strlen(tmpl);
    }
    char desc[256];
    (void) llama_model_desc(model, desc, sizeof(desc));
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int * /*argc*/, char *** /*argv*/) {
    fuzz::quiet_logs();
    llama_backend_init();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz::note_input(data, size);
    if (size < 2) {
        return 0;
    }
    const uint8_t mode = data[size - 1];
    size -= 1;

    // The loader reads the file with the GGUF reader of ggml. FUZZ_GGUF_KNOWN_ENUM_LOAD=1 and
    // FUZZ_GGUF_KNOWN_NELEMENTS=1 skip a file that shows the finding gguf-enum-load or
    // gguf-nelements-overflow of that reader.
    static const bool known_enum = fuzz::env_long("FUZZ_GGUF_KNOWN_ENUM_LOAD", 0) != 0;
    static const bool known_nel  = fuzz::env_long("FUZZ_GGUF_KNOWN_NELEMENTS", 0) != 0;
    if (known_enum || known_nel) {
        const fuzz::GgufScan scan = fuzz::gguf_scan(data, size);
        if ((known_enum && scan.bad_enum) || (known_nel && scan.nelements_overflow)) {
            return 0;
        }
    }
    // Two equal texts in tokenizer.ggml.tokens stop the load with
    // GGML_ASSERT(id_to_token.size() == token_to_id.size()) (finding vocab-dup-tokens).
    // FUZZ_MODEL_LOAD_KNOWN_DUP_TOKENS=1 skips such a file. known_bad_file() gives the other switches.
    static const KnownSwitches sw = {
        fuzz::env_long("FUZZ_MODEL_LOAD_KNOWN_DUP_TOKENS", 0) != 0,
        fuzz::env_long("FUZZ_MODEL_LOAD_KNOWN_BYTE_TYPE", 0) != 0,
        fuzz::env_long("FUZZ_MODEL_LOAD_KNOWN_FTYPE", 0) != 0,
        fuzz::env_long("FUZZ_MODEL_LOAD_KNOWN_META_LEAK", 0) != 0,
    };
    // the pre-parse uses the GGUF reader too, thus it comes after the enum check
    if ((sw.dup || sw.byte || sw.ftype || sw.meta_leak) && known_bad_file(data, size, sw)) {
        return 0;
    }

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.load_mode  = LLAMA_LOAD_MODE_MMAP;
    mp.check_tensors = true;

    llama_model * model = nullptr;
    if ((mode & 1) == 0) {
        const int fd = memfd_create("fuzz_model_load", MFD_CLOEXEC);
        if (fd < 0) {
            fuzz::fail("memfd_create failed: %s", strerror(errno));
        }
        size_t off = 0;
        while (off < size) {
            const ssize_t w = write(fd, data + off, size - off);
            if (w <= 0) {
                fuzz::fail("write to the memfd failed: %s", strerror(errno));
            }
            off += (size_t) w;
        }
        const std::string path = "/proc/self/fd/" + std::to_string(fd);
        model = llama_model_load_from_file(path.c_str(), mp);
        close(fd);
    } else {
        FILE * f = fmemopen((void *) data, size, "rb");
        if (f == nullptr) {
            return 0;
        }
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        model = llama_model_load_from_file_ptr(f, mp);
        fclose(f);
    }

    if (model != nullptr) {
        check_vocab(model);
        llama_model_free(model);
    }
    return 0;
}
