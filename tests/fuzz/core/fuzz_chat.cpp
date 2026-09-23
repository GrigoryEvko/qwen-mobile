// fuzz_chat: the chat templates of common/chat and the jinja engine of common/jinja.
//
// The harness loads data/qwen35-vocab.gguf one time. The app calls
// common_chat_templates_init(model, "") with that model, thus the harness
// does the same and gets the Qwen3.5 template and its special tokens.
//
// The last input byte selects the mode:
//   even  Message mode. The input gives a random conversation: roles (system,
//         user, assistant, tool, or a random role), contents with and without
//         the media marker, reasoning content, tool calls with valid or
//         invalid JSON arguments, tools, and the switches of
//         common_chat_templates_inputs. The harness applies the Qwen3.5
//         template as the app does, tokenizes the prompt as the app does
//         (add_special and parse_special), and parses a random model output
//         with the parser that the template gives, full and partial.
//   odd   Jinja mode. The other bytes are the source of a template. The
//         harness renders it with a fixed conversation.
//
// A std::exception is a correct refusal of an input. The properties:
//   P1  No crash, no sanitizer report, no exception of a different type.
//   P2  Message mode: when each input string is valid UTF-8, the prompt is valid UTF-8.
//
// The switch FUZZ_CHAT_KNOWN_JINJA=1 keeps two known findings off:
//   jinja-parser-assert  In a build with live asserts, jinja::chk_type fails an
//                        assert on a syntax tree node of an unexpected type.
//   jinja-float-cast     In a UBSan build, a number of a template or of a JSON
//                        text outside the int64_t range makes a float-cast-overflow
//                        report in common/jinja/value.h.
// With the switch, the jinja mode first renders the template in a child
// process, and skips the input when the child stops with one of the two
// signatures. The message mode skips an input when a JSON text holds a number
// of 19 or more digits or an exponent of two or more digits. In a release
// build without UBSan, the two findings cannot occur, and the switch does nothing.

#include "fuzz_common.h"

#include "chat.h"
#include "common.h"

#include <cctype>
#include <cerrno>
#include <csignal>
#include <string>
#include <vector>

#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

// These two symbols exist only in a build with a sanitizer runtime (weak: a null address otherwise).
extern "C" __attribute__((weak)) void __ubsan_handle_float_cast_overflow_abort();
extern "C" __attribute__((weak)) void __sanitizer_set_death_callback(void (*callback)(void));

namespace {

llama_model *             g_model = nullptr;
common_chat_templates_ptr g_tmpls;

/** True when the build can show one of the two known jinja findings (see the header). */
bool jinja_findings_possible() {
#ifndef NDEBUG
    return true;
#else
    return __ubsan_handle_float_cast_overflow_abort != nullptr;
#endif
}

/** True when FUZZ_CHAT_KNOWN_JINJA=1 and the build can show a known jinja finding. */
bool known_jinja() {
    static const bool on = fuzz::env_long("FUZZ_CHAT_KNOWN_JINJA", 0) != 0 && jinja_findings_possible();
    return on;
}

/** True when a JSON text can hold a number outside the int64_t range: 19 or more digits, or a 2-digit exponent. */
bool json_has_huge_number(const std::string & s) {
    size_t run = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        run = (c >= '0' && c <= '9') ? run + 1 : 0;
        if (run >= 19) {
            return true;
        }
        if ((c == 'e' || c == 'E') && i + 1 < s.size()) {
            size_t j = i + 1;
            if (s[j] == '+' || s[j] == '-') {
                ++j;
            }
            if (j + 1 < s.size() && isdigit((unsigned char) s[j]) && isdigit((unsigned char) s[j + 1])) {
                return true;
            }
        }
    }
    return false;
}

/**
 * Run fn in a child process. True when the child stops with a known jinja signature on its stderr.
 * The child uses the default signal actions and no death callback, thus libFuzzer writes no crash
 * file for it. The child stops after 10 s, and it stops when the parent stops.
 */
template <typename F>
bool child_shows_known_jinja(F && fn) {
    int fds[2];
    if (pipe(fds) != 0) {
        return false;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return false;
    }
    if (pid == 0) {
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        for (const int sig : { SIGABRT, SIGALRM, SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTERM, SIGINT, SIGUSR1, SIGUSR2 }) {
            signal(sig, SIG_DFL);
        }
        if (__sanitizer_set_death_callback != nullptr) {
            __sanitizer_set_death_callback(nullptr);
        }
        dup2(fds[1], STDERR_FILENO);
        close(fds[0]);
        close(fds[1]);
        alarm(10);
        fn();
        _exit(0);
    }
    close(fds[1]);
    std::string err;
    char buf[4096];
    ssize_t n;
    while ((n = read(fds[0], buf, sizeof(buf))) > 0 || (n < 0 && errno == EINTR)) {
        if (n > 0 && err.size() < (1u << 16)) {
            err.append(buf, (size_t) n);
        }
    }
    close(fds[0]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    const bool failed = WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) != 0);
    const bool parser_assert = err.find("jinja::chk_type") != std::string::npos;
    const bool float_cast = err.find("jinja/value.h") != std::string::npos &&
                            err.find("is outside the range of representable values") != std::string::npos;
    return failed && (parser_assert || float_cast);
}

/** A string from the input that is often one of the words that the Qwen3.5 template reacts to. */
std::string word(FuzzedDataProvider & fdp, size_t max_len) {
    static const char * const kWords[] = {
        "<__media__>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
        "</tool_response>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>",
        "\n", "\n\n", " ", "{", "}", "\"", "{{", "{%", "user", "assistant", "system", "tool",
        "<function=", "</function>", "<parameter=", "</parameter>",
    };
    if (fdp.ConsumeBool()) {
        return kWords[fdp.ConsumeIntegralInRange<size_t>(0, sizeof(kWords) / sizeof(kWords[0]) - 1)];
    }
    return fdp.ConsumeRandomLengthString(max_len);
}

/** A text that joins 0 to 5 words from the input. */
std::string text(FuzzedDataProvider & fdp) {
    std::string out;
    const int n = fdp.ConsumeIntegralInRange<int>(0, 5);
    for (int i = 0; i < n; ++i) {
        out += word(fdp, 64);
    }
    return out;
}

/** A JSON text for tool arguments or tool parameters: often valid, sometimes any string. */
std::string json_text(FuzzedDataProvider & fdp) {
    switch (fdp.ConsumeIntegralInRange<int>(0, 4)) {
        case 0:  return "{}";
        case 1:  return "{\"type\": \"object\", \"properties\": {\"q\": {\"type\": \"string\"}}, \"required\": [\"q\"]}";
        case 2:  return "{\"q\": " + std::string(fdp.ConsumeBool() ? "\"x\"" : "[1, 2, {\"a\": null}]") + "}";
        case 3:  return "[]";
        default: return fdp.ConsumeRandomLengthString(64);
    }
}

/** True when each string of the inputs is valid UTF-8. */
bool inputs_are_utf8(const common_chat_templates_inputs & in) {
    for (const auto & m : in.messages) {
        if (!fuzz::is_valid_utf8(m.role) || !fuzz::is_valid_utf8(m.content) || !fuzz::is_valid_utf8(m.reasoning_content) ||
            !fuzz::is_valid_utf8(m.tool_name) || !fuzz::is_valid_utf8(m.tool_call_id)) {
            return false;
        }
        for (const auto & tc : m.tool_calls) {
            if (!fuzz::is_valid_utf8(tc.name) || !fuzz::is_valid_utf8(tc.arguments) || !fuzz::is_valid_utf8(tc.id)) {
                return false;
            }
        }
        for (const auto & part : m.content_parts) {
            if (!fuzz::is_valid_utf8(part.type) || !fuzz::is_valid_utf8(part.text)) {
                return false;
            }
        }
    }
    for (const auto & t : in.tools) {
        if (!fuzz::is_valid_utf8(t.name) || !fuzz::is_valid_utf8(t.description) || !fuzz::is_valid_utf8(t.parameters)) {
            return false;
        }
    }
    return true;
}

/** The message mode: a random conversation through the Qwen3.5 template, the tokenizer and the output parser. */
void run_messages(FuzzedDataProvider & fdp) {
    static const char * const kRoles[] = { "system", "user", "assistant", "tool", "user", "assistant" };

    common_chat_templates_inputs in;
    const int n_msgs = fdp.ConsumeIntegralInRange<int>(0, 10);
    for (int i = 0; i < n_msgs; ++i) {
        common_chat_msg m;
        m.role = fdp.ConsumeIntegralInRange<int>(0, 7) < 7 ? kRoles[fdp.ConsumeIntegralInRange<int>(0, 5)] : word(fdp, 16);
        if (fdp.ConsumeIntegralInRange<int>(0, 3) == 0) {
            m.content = "<__media__>\n";  // the app puts the marker first for a message with an image
        }
        m.content += text(fdp);
        if (m.role == "assistant") {
            if (fdp.ConsumeBool()) {
                m.reasoning_content = text(fdp);
            }
            const int n_calls = fdp.ConsumeIntegralInRange<int>(0, 2);
            for (int c = 0; c < n_calls; ++c) {
                common_chat_tool_call tc;
                tc.name      = word(fdp, 16);
                tc.arguments = json_text(fdp);
                tc.id        = fdp.ConsumeBool() ? fdp.ConsumeRandomLengthString(8) : "";
                m.tool_calls.push_back(tc);
            }
        }
        if (m.role == "tool") {
            m.tool_name    = word(fdp, 16);
            m.tool_call_id = fdp.ConsumeRandomLengthString(8);
        }
        if (fdp.ConsumeIntegralInRange<int>(0, 7) == 0) {
            common_chat_msg_content_part part;
            part.type = fdp.ConsumeBool() ? "text" : "media_marker";
            part.text = text(fdp);
            m.content_parts.push_back(part);
            m.content.clear();  // content and content_parts are exclusive in the OpenAI form
        }
        in.messages.push_back(std::move(m));
    }
    const int n_tools = fdp.ConsumeIntegralInRange<int>(0, 2);
    for (int t = 0; t < n_tools; ++t) {
        in.tools.push_back({ word(fdp, 16), text(fdp), json_text(fdp) });
    }
    in.add_generation_prompt = fdp.ConsumeBool();
    in.use_jinja             = true;
    in.enable_thinking       = fdp.ConsumeBool();
    in.parallel_tool_calls   = fdp.ConsumeBool();
    in.tool_choice           = (common_chat_tool_choice) fdp.ConsumeIntegralInRange<int>(0, 2);
    in.reasoning_format      = (common_reasoning_format) fdp.ConsumeIntegralInRange<int>(0, 3);
    in.continue_final_message = (common_chat_continuation) fdp.ConsumeIntegralInRange<int>(0, 3);

    if (known_jinja()) {
        for (const auto & m : in.messages) {
            for (const auto & tc : m.tool_calls) {
                if (json_has_huge_number(tc.arguments)) {
                    return;
                }
            }
        }
        for (const auto & t : in.tools) {
            if (json_has_huge_number(t.parameters)) {
                return;
            }
        }
    }

    common_chat_params params;
    try {
        params = common_chat_templates_apply(g_tmpls.get(), in);
    } catch (const std::exception &) {
        return;
    }

    if (inputs_are_utf8(in) && !fuzz::is_valid_utf8(params.prompt)) {
        fuzz::fail("P2: the template gives a prompt that is not valid UTF-8 from inputs that are valid UTF-8 (%zu bytes)",
                   params.prompt.size());
    }

    // The app tokenizes the prompt with add_special and parse_special.
    const std::vector<llama_token> ids = common_tokenize(llama_model_get_vocab(g_model), params.prompt, true, true);
    (void) ids;

    // The output parser of the template, on a random model output, full and partial.
    const std::string output = text(fdp) + fdp.ConsumeRemainingBytesAsString();
    try {
        common_chat_parser_params pp(params);
        pp.reasoning_format = in.reasoning_format;
        pp.parse_tool_calls = !in.tools.empty();
        if (!params.parser.empty()) {
            pp.parser.load(params.parser);
        }
        (void) common_chat_parse(output, false, pp);
        (void) common_chat_parse(output.substr(0, output.size() / 2), true, pp);
    } catch (const std::exception &) {
    }
}

/** The jinja mode: the input is the template source, rendered with a fixed conversation. */
void run_jinja(const std::string & src) {
    common_chat_templates_inputs in;
    in.messages = {
        { "system", "You are a test.", {}, {}, "", "", "" },
        { "user", "Hello <__media__>", {}, {}, "", "", "" },
        { "assistant", "Hi", {}, { { "f", "{\"q\": 1}", "id0" } }, "thought", "", "" },
        { "tool", "result", {}, {}, "", "f", "id0" },
        { "user", "again", {}, {}, "", "", "" },
    };
    in.tools = { { "f", "a tool", "{\"type\": \"object\", \"properties\": {}}" } };
    in.add_generation_prompt = true;
    in.use_jinja = true;
    auto render = [&]() {
        try {
            common_chat_templates_ptr tmpls = common_chat_templates_init(nullptr, src, "<s>", "</s>");
            (void) common_chat_templates_apply(tmpls.get(), in);
        } catch (const std::exception &) {
        }
    };
    if (known_jinja() && child_shows_known_jinja(render)) {
        return;
    }
    render();
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int * /*argc*/, char *** /*argv*/) {
    fuzz::quiet_logs();
    llama_backend_init();
    const std::string path = fuzz::data_file("qwen35-vocab.gguf");
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    g_model = llama_model_load_from_file(path.c_str(), mp);
    if (g_model == nullptr) {
        fuzz::fail("cannot load the vocabulary %s", path.c_str());
    }
    g_tmpls = common_chat_templates_init(g_model, "");
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz::note_input(data, size);
    if (size == 0) {
        return 0;
    }
    if (data[size - 1] & 1) {
        run_jinja(std::string((const char *) data, size - 1));
        return 0;
    }
    FuzzedDataProvider fdp(data, size - 1);
    run_messages(fdp);
    return 0;
}
