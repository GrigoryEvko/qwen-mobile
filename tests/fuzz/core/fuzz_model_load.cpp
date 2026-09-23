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
// The switch of a known finding: FUZZ_GGUF_KNOWN_ENUM_LOAD (gguf-enum-load).

#include "fuzz_common.h"

#include "gguf.h"

#include <sys/mman.h>
#include <unistd.h>

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

/**
 * Check the vocabulary of a loaded model. Stop with fuzz::fail() on a broken property. A vocabulary of
 * type NONE ("no_vocab") has token ids but no token data: the token functions give neutral values
 * for it.
 */
void check_vocab(const llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab < 0) {
        fuzz::fail("n_vocab is negative: %d", n_vocab);
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

    // llama_tokenize gives INT32_MIN when the tokenizer fails (for example an SPM vocabulary without
    // the byte tokens <0x00>..<0xFF>), and a negative count when the buffer is too small.
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
