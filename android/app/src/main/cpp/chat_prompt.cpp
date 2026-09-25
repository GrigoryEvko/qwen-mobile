#include "chat_prompt.h"

#include "common.h"

#include <utility>

namespace {

/** The text without the ASCII white space at its two ends, as the trim filter of the templates gives it. O(length). */
std::string trim_ascii(const std::string & s) {
    const char * ws = " \t\n\r\f\v";
    const size_t a  = s.find_first_not_of(ws);
    if (a == std::string::npos) {
        return {};
    }
    const size_t b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

/** True when s starts with prefix. */
bool starts_with(const std::string & s, const std::string & prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

/** True when s ends with suffix. */
bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

bool parse_history_form(const std::string & name, HistoryForm & out) {
    if (name == "template") {
        out = HistoryForm::kTemplate;
        return true;
    }
    if (name == "live") {
        out = HistoryForm::kLive;
        return true;
    }
    return false;
}

ChatPrompt::ChatPrompt(const common_chat_templates * tmpls, const llama_vocab * vocab, HistoryForm form)
    : tmpls_(tmpls), vocab_(vocab), form_(form) {
    if (form_ != HistoryForm::kLive) {
        return;
    }
    bool ok = false;
    try {
        ok = probe();
    } catch (const std::exception & ex) {
        refusal_ = std::string("the chat template failed on a probe: ") + ex.what();
    }
    if (!ok) {
        form_      = HistoryForm::kTemplate;
        eog_token_ = LLAMA_TOKEN_NULL;
        joint_.clear();
        gen_tail_.clear();
    }
}

std::string ChatPrompt::apply(const std::vector<PromptMessage> & msgs, bool add_generation_prompt, bool thinking) const {
    common_chat_templates_inputs inputs;
    inputs.messages.reserve(msgs.size());
    for (const PromptMessage & m : msgs) {
        common_chat_msg msg;
        msg.role    = m.role;
        msg.content = m.content;
        inputs.messages.push_back(std::move(msg));
    }
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.use_jinja             = true;
    inputs.enable_thinking       = thinking;
    return common_chat_templates_apply(tmpls_, inputs).prompt;
}

bool ChatPrompt::probe() {
    const PromptMessage sys{"system", "S", ""};
    const PromptMessage u{"user", "U", ""};
    const PromptMessage a{"assistant", "A", ""};
    const PromptMessage v{"user", "V", ""};

    // The last user message with the generation prompt, and the same user
    // message with its answer: the answer must continue the generation prompt.
    const std::string gen = apply({u}, true, false);
    const std::string seg = apply({u, a}, false, false);
    if (!starts_with(seg, gen) || seg.compare(gen.size(), a.content.size(), a.content) != 0) {
        refusal_ = "the render of an answer does not continue the generation prompt of its user message";
        return false;
    }
    // After the answer text: the end token of the template, then the joint to the next message.
    const std::string rest = seg.substr(gen.size() + a.content.size());
    const std::vector<llama_token> toks = common_tokenize(vocab_, rest, false, true);
    if (toks.empty() || !llama_vocab_is_eog(vocab_, toks[0])) {
        refusal_ = "the render of an answer does not end with an end token";
        return false;
    }
    const std::string eog_piece = common_token_to_piece(vocab_, toks[0], true);
    if (!starts_with(rest, eog_piece) || common_tokenize(vocab_, eog_piece, false, true).size() != 1) {
        refusal_ = "the end token of an answer does not have a text of its own";
        return false;
    }
    eog_token_ = toks[0];
    joint_     = rest.substr(eog_piece.size());

    // A user message renders the same text in any place of the conversation:
    // no render adds a message of its own (for example a default system message).
    const std::string first = apply({u}, false, false);
    const std::string two   = apply({u, a, v}, true, false);
    if (!starts_with(gen, first) || !starts_with(two, first) || !ends_with(two, apply({v}, true, false))) {
        refusal_ = "the template does not render each user message by itself";
        return false;
    }
    gen_tail_ = gen.substr(first.size());
    // The system message is a prefix of the render.
    const std::string with_sys = apply({sys, u}, true, false);
    if (!ends_with(with_sys, gen) || !ends_with(apply({sys, u}, false, false), first)) {
        refusal_ = "the template does not render the system message as a prefix";
        return false;
    }
    return true;
}

bool ChatPrompt::render_live(const std::vector<PromptMessage> & msgs, std::string & out) const {
    out.clear();
    const size_t n = msgs.size();
    size_t       i = 0;
    std::string  text;
    if (n > 0 && msgs[0].role == "system") {
        // The system message is the part of its render before the first user message.
        const PromptMessage probe_user{"user", "U", ""};
        const std::string   both  = apply({msgs[0], probe_user}, false, false);
        const std::string   alone = apply({probe_user}, false, false);
        if (!ends_with(both, alone)) {
            return false;
        }
        text = both.substr(0, both.size() - alone.size());
        i    = 1;
    }
    while (i < n) {
        if (msgs[i].role != "user") {
            return false;
        }
        if (i + 1 == n) {
            text += apply({msgs[i]}, true, false);
            out = std::move(text);
            return true;
        }
        if (msgs[i + 1].role != "assistant") {
            return false;
        }
        // The template renders an answer after the last user message with its think block.
        text += apply({msgs[i], msgs[i + 1]}, false, false);
        i += 2;
    }
    return false;
}

void ChatPrompt::render_template(const std::vector<PromptMessage> & msgs, bool thinking, std::string & prompt,
                                 std::string & tail) const {
    common_chat_templates_inputs inputs;
    for (const PromptMessage & m : msgs) {
        common_chat_msg msg;
        msg.role    = m.role;
        msg.content = m.content;
        inputs.messages.push_back(std::move(msg));
    }
    inputs.add_generation_prompt = true;
    inputs.use_jinja             = true;
    inputs.enable_thinking       = thinking;
    // The prompt, and the generation prompt at its end. The template gives the
    // tail directly, or the render without it gives the common prefix.
    common_chat_params params = common_chat_templates_apply(tmpls_, inputs);
    prompt = std::move(params.prompt);
    tail   = std::move(params.generation_prompt);
    if (!ends_with(prompt, tail) || tail.empty()) {
        inputs.add_generation_prompt = false;
        const std::string base = common_chat_templates_apply(tmpls_, inputs).prompt;
        size_t k = 0;
        while (k < base.size() && k < prompt.size() && base[k] == prompt[k]) {
            ++k;
        }
        tail = prompt.substr(k);
    }
}

bool ChatPrompt::continues(const std::vector<PromptMessage> & msgs, size_t mem_items, llama_token mem_last) const {
    if (held_.empty() || msgs.size() != held_.size() + 1 || msgs.back().role != "user") {
        return false;
    }
    if (mem_items != held_items_ || mem_last != held_last_ || held_last_ != eog_token_) {
        return false;
    }
    const size_t last = held_.size() - 1;
    for (size_t i = 0; i < last; ++i) {
        if (msgs[i].role != held_[i].role || msgs[i].content != held_[i].content || msgs[i].image_id != held_[i].image_id) {
            return false;
        }
    }
    // The app removes the white space at the start of an answer, and the template trims each message.
    return msgs[last].role == held_[last].role && msgs[last].image_id == held_[last].image_id &&
           trim_ascii(msgs[last].content) == trim_ascii(held_[last].content);
}

TurnPrompt ChatPrompt::make(const std::vector<PromptMessage> & msgs, bool thinking, size_t mem_items,
                            llama_token mem_last) {
    TurnPrompt p;
    const bool live = form_ == HistoryForm::kLive && !thinking;
    const bool cont = live && continues(msgs, mem_items, mem_last);
    // The record describes the memory before this turn. The turn changes the
    // memory, thus the record goes, and answer_ended() makes the next one.
    held_.clear();
    turn_.clear();
    turn_live_ = false;
    try {
        if (cont) {
            p.text          = joint_ + apply({msgs.back()}, true, false);
            p.tail          = gen_tail_;
            p.continuation  = true;
            p.base_snapshot = false;
            p.live          = true;
            p.first_message = msgs.size() - 1;
        } else if (live && render_live(msgs, p.text)) {
            p.tail          = gen_tail_;
            p.base_snapshot = false;
            p.live          = true;
        }
    } catch (const std::exception &) {
        // The template refused a part by itself (for example a user message
        // that holds only a tool response). The render of the whole request
        // decides, and its exception goes to the caller.
        p = TurnPrompt{};
    }
    if (!p.live) {
        render_template(msgs, thinking, p.text, p.tail);
    }
    if (p.live) {
        turn_      = msgs;
        turn_live_ = true;
    }
    return p;
}

bool ChatPrompt::answer_ended(const std::string & answer_text, bool clean, size_t mem_items, llama_token mem_last) {
    held_.clear();
    if (clean && turn_live_ && eog_token_ != LLAMA_TOKEN_NULL && mem_last == eog_token_) {
        held_ = std::move(turn_);
        held_.push_back(PromptMessage{"assistant", answer_text, ""});
        held_items_ = mem_items;
        held_last_  = mem_last;
    }
    turn_.clear();
    turn_live_ = false;
    return !held_.empty();
}

void ChatPrompt::forget() {
    held_.clear();
    turn_.clear();
    turn_live_ = false;
}
