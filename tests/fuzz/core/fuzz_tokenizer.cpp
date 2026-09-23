// fuzz_tokenizer: the Qwen3.5 tokenizer of llama.cpp on any byte string.
//
// The harness loads data/qwen35-vocab.gguf (make_data.py writes it from the
// 2B Q8_0 model) with vocab_only = true, one time. FUZZ_VOCAB gives a
// different vocabulary file.
//
// The last input byte holds the flags, and the other bytes are the text:
//   bit 0  add_special       bit 2  remove_special
//   bit 1  parse_special     bit 3  unparse_special
//
// Properties:
//   P1  No crash and no sanitizer report on any text, valid UTF-8 or not.
//   P2  Each token id is in [0, n_vocab).
//   P3  A buffer that is too small gives the negative of the count that the
//       full buffer then gives, for tokenize and for detokenize.
//   P4  The concatenation of token_to_piece(id, special = unparse_special)
//       equals detokenize(ids, remove_special = false, unparse_special).
//   P5  Round trip: for valid UTF-8 text without add_special,
//       detokenize(tokenize(text, parse_special), unparse_special = parse_special) == text.
//       The byte-level BPE of Qwen has no normalizer and no space cleanup,
//       thus the tokenizer promises this. FUZZ_TOKENIZER_NO_ROUNDTRIP=1 turns
//       P5 off for a vocabulary that does not promise it.

#include "fuzz_common.h"

#include <string>
#include <vector>

namespace {

const llama_vocab * g_vocab   = nullptr;
int32_t             g_n_vocab = 0;
bool                g_check_roundtrip = true;

/** Write a byte string as hex into a std::string, for a failure message. Only the first 256 bytes. */
std::string hex(const std::string & s) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    for (size_t i = 0; i < s.size() && i < 256; ++i) {
        out += digits[(unsigned char) s[i] >> 4];
        out += digits[(unsigned char) s[i] & 15];
    }
    if (s.size() > 256) {
        out += "...";
    }
    return out;
}

/** Tokenize text. Check P3 when the first buffer is too small. */
std::vector<llama_token> tokenize(const std::string & text, bool add_special, bool parse_special) {
    std::vector<llama_token> tokens(text.size() / 4 + 2);
    int32_t n = llama_tokenize(g_vocab, text.data(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(),
                               add_special, parse_special);
    if (n == INT32_MIN) {
        fuzz::fail("tokenize fails (an overflow or an error of the tokenizer) for a text of %zu bytes", text.size());
    }
    if (n < 0) {
        tokens.resize((size_t) -n);
        const int32_t n2 = llama_tokenize(g_vocab, text.data(), (int32_t) text.size(), tokens.data(),
                                          (int32_t) tokens.size(), add_special, parse_special);
        if (n2 != -n) {
            fuzz::fail("tokenize reports %d tokens for a small buffer, then gives %d tokens", -n, n2);
        }
        n = n2;
    }
    tokens.resize((size_t) n);
    return tokens;
}

/** Detokenize tokens. Check P3 with a buffer of half the size first. */
std::string detokenize(const std::vector<llama_token> & tokens, bool remove_special, bool unparse_special) {
    std::string out(tokens.size(), '\0');
    int32_t n = llama_detokenize(g_vocab, tokens.data(), (int32_t) tokens.size(), out.data(), (int32_t) out.size(),
                                 remove_special, unparse_special);
    if (n < 0) {
        out.resize((size_t) -n);
        const int32_t n2 = llama_detokenize(g_vocab, tokens.data(), (int32_t) tokens.size(), out.data(),
                                            (int32_t) out.size(), remove_special, unparse_special);
        if (n2 != -n) {
            fuzz::fail("detokenize reports %d bytes for a small buffer, then writes %d bytes", -n, n2);
        }
        n = n2;
    }
    out.resize((size_t) n);
    return out;
}

/** The concatenation of the pieces of the tokens. */
std::string pieces(const std::vector<llama_token> & tokens, bool special) {
    std::string out;
    std::vector<char> buf(32);
    for (const llama_token id : tokens) {
        int32_t n = llama_token_to_piece(g_vocab, id, buf.data(), (int32_t) buf.size(), 0, special);
        if (n < 0) {
            buf.resize((size_t) -n);
            n = llama_token_to_piece(g_vocab, id, buf.data(), (int32_t) buf.size(), 0, special);
        }
        if (n < 0) {
            fuzz::fail("token_to_piece(%d) fails with a buffer of the size that it reported", id);
        }
        out.append(buf.data(), (size_t) n);
    }
    return out;
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int * /*argc*/, char *** /*argv*/) {
    fuzz::quiet_logs();
    llama_backend_init();
    const char * env = getenv("FUZZ_VOCAB");
    const std::string path = env != nullptr ? std::string(env) : fuzz::data_file("qwen35-vocab.gguf");
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (model == nullptr) {
        fuzz::fail("cannot load the vocabulary %s", path.c_str());
    }
    g_vocab   = llama_model_get_vocab(model);
    g_n_vocab = llama_vocab_n_tokens(g_vocab);
    g_check_roundtrip = fuzz::env_long("FUZZ_TOKENIZER_NO_ROUNDTRIP", 0) == 0;
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz::note_input(data, size);
    if (size == 0) {
        return 0;
    }
    const uint8_t flags = data[size - 1];
    const bool add_special     = flags & 1;
    const bool parse_special   = flags & 2;
    const bool remove_special  = flags & 4;
    const bool unparse_special = flags & 8;
    const std::string text((const char *) data, size - 1);

    const std::vector<llama_token> tokens = tokenize(text, add_special, parse_special);
    for (const llama_token id : tokens) {
        if (id < 0 || id >= g_n_vocab) {
            fuzz::fail("tokenize gives the id %d, outside [0, %d)", id, g_n_vocab);
        }
    }

    // P4
    const std::string detok = detokenize(tokens, false, unparse_special);
    const std::string cat   = pieces(tokens, unparse_special);
    if (detok != cat) {
        fuzz::fail("P4: detokenize and the concatenated pieces differ (unparse_special=%d)\n  detok  %s\n  pieces %s",
                   unparse_special, hex(detok).c_str(), hex(cat).c_str());
    }

    // P5
    if (g_check_roundtrip && !add_special && fuzz::is_valid_utf8(text)) {
        // Without parse_special, a special token text is plain text, thus no special id must appear,
        // and the detokenizer that renders no special token must give the text back.
        const std::string back = detokenize(tokens, false, parse_special);
        if (back != text) {
            fuzz::fail("P5: the round trip changes the text (parse_special=%d)\n  text %s\n  back %s",
                       parse_special, hex(text).c_str(), hex(back).c_str());
        }
    }

    (void) detokenize(tokens, remove_special, unparse_special);
    return 0;
}
