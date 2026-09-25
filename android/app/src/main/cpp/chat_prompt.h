/**
 * The prompt text of a chat turn, and the record of the conversation that
 * the model memory holds.
 *
 * Two forms of the conversation history:
 *
 * - The template form. The chat template renders the whole conversation at
 *   each turn. The Qwen3.5 template renders an earlier answer without the
 *   think block that its generation prompt put in the memory, thus the
 *   memory is never a prefix of the next prompt. The engine keeps a snapshot
 *   of the state before each generation prompt and restores it.
 *
 * - The live form. An earlier answer keeps the empty think block of its
 *   generation prompt, thus the history holds the same text as the model
 *   memory after the answer. A turn that follows a complete answer renders
 *   and tokenizes only the new message and the generation prompt, and the
 *   memory continues without a restore and without a snapshot before the
 *   generation prompt. The live form applies to the answers without
 *   thinking. With thinking on, a turn takes the template form: the template
 *   removes the reasoning of the earlier answers, which is the rule of the
 *   model.
 *
 * The full render of the live form is the chat template applied to each
 * part by itself: the system message, then each pair of a user message and
 * its answer, then the last user message with the generation prompt. The
 * template renders an answer that follows the last user message with its
 * think block, thus each answer gets the form that the memory holds. The
 * constructor tests the template on probe messages. When a probe does not
 * give the pieces that the live form needs (for example, a template that
 * adds a default system message to each render), the engine uses the
 * template form.
 *
 * All calls come from one thread. The chat template can throw
 * std::exception, and a call that renders does not catch it.
 */
#pragma once

#include "chat.h"
#include "llama.h"

#include <cstddef>
#include <string>
#include <vector>

/** The form of the earlier answers in the prompt of a turn. */
enum class HistoryForm {
    /** The chat template renders the whole conversation. */
    kTemplate,
    /** An earlier answer keeps the empty think block of its generation prompt. */
    kLive,
};

/** The history form of a name: "template" or "live". Returns false for another name. */
bool parse_history_form(const std::string & name, HistoryForm & out);

/** One message of a request as the app gives it. */
struct PromptMessage {
    std::string role;
    /** The text. A message with an image starts with the media marker of mtmd and a line break. */
    std::string content;
    /** The SHA-256 of the image file, or empty for a message without an image. */
    std::string image_id;
};

/** The prompt of one turn. */
struct TurnPrompt {
    /** The text to tokenize: the whole prompt, or only the part after the memory when continuation is true. */
    std::string text;
    /** The generation prompt at the end of text. */
    std::string tail;
    /** True when text follows the model memory: the memory holds every message before the last one. */
    bool continuation = false;
    /** True when the turn keeps the state before its generation prompt in the snapshot store (the template form). */
    bool base_snapshot = true;
    /** True when the answer of this turn can become the record of the memory (the live form). */
    bool live = false;
    /** The first message of the request that text renders: 0, or the last message when continuation is true. */
    size_t first_message = 0;
};

class ChatPrompt {
public:
    /**
     * @param tmpls  The chat templates of the model. The object keeps the pointer.
     * @param vocab  The vocabulary of the model. The object keeps the pointer.
     * @param form   The form that the engine asks for. The live form becomes the
     *               template form when the template fails its probes.
     */
    ChatPrompt(const common_chat_templates * tmpls, const llama_vocab * vocab, HistoryForm form);

    /** The form that the engine uses after the probes. */
    HistoryForm form() const { return form_; }

    /** The reason when the probes refused the live form, else empty. */
    const std::string & refusal() const { return refusal_; }

    /**
     * The prompt of a turn. With the live form and thinking off, a request that
     * holds the recorded conversation plus one user message continues the
     * memory. mem_items is the number of items that the memory holds and
     * mem_last the last token of the memory: the record is valid only while
     * they have their recorded values. The call keeps the request for
     * answer_ended() and forgets the record. O(length of the request) when the
     * memory continues, one template render of each pair for a full render.
     */
    TurnPrompt make(const std::vector<PromptMessage> & msgs, bool thinking, size_t mem_items, llama_token mem_last);

    /**
     * The end of the answer of the last make(). With clean true, the memory
     * holds the prompt of that turn, the tokens of the answer and the end token
     * of the template, and nothing more: the request of that turn and the answer
     * become the record. Else the record stays empty.
     *
     * @param answer_text The UTF-8 bytes of the answer that the app received
     * @param clean       True when the answer ended with the end token of the
     *                    template, the app received every token, and no token
     *                    was a thinking tag
     * @param mem_items   The number of items of the memory at the end of the answer
     * @param mem_last    The last token of the memory
     * @return True when the call made a record
     */
    bool answer_ended(const std::string & answer_text, bool clean, size_t mem_items, llama_token mem_last);

    /** Forget the record: the memory changed outside an answer (a reset, a benchmark, a failure). */
    void forget();

    /** True when a record of the memory exists. */
    bool has_record() const { return !held_.empty(); }

    /** The end token of an answer in the template, or LLAMA_TOKEN_NULL when the live form is off. */
    llama_token eog_token() const { return eog_token_; }

    /**
     * The full render of the live form: the system message, each pair of a
     * user message and its answer, and the last user message with the
     * generation prompt. Returns false when the request does not have that
     * shape (a tool message, two user messages one after the other, a system
     * message after the first one), and then out is empty.
     */
    bool render_live(const std::vector<PromptMessage> & msgs, std::string & out) const;

    /** The render of the chat template for the whole request, and its generation prompt. */
    void render_template(const std::vector<PromptMessage> & msgs, bool thinking, std::string & prompt,
                         std::string & tail) const;

private:
    /** The chat template applied to the messages. */
    std::string apply(const std::vector<PromptMessage> & msgs, bool add_generation_prompt, bool thinking) const;

    /** Run the probes of the live form. Returns false with refusal_ set. */
    bool probe();

    /** True when the request is the record plus one user message. O(length of the request). */
    bool continues(const std::vector<PromptMessage> & msgs, size_t mem_items, llama_token mem_last) const;

    const common_chat_templates * tmpls_;
    const llama_vocab *           vocab_;
    HistoryForm                   form_;
    std::string                   refusal_;

    /** The text between the end token of an answer and the next message. */
    std::string joint_;
    /** The generation prompt of the live form (thinking off). */
    std::string gen_tail_;
    /** The end token of an answer, which the model generates and the memory holds. */
    llama_token eog_token_ = LLAMA_TOKEN_NULL;

    /** The request of the last make() when it can become the record. */
    std::vector<PromptMessage> turn_;
    bool                       turn_live_ = false;

    /** The record: the messages that the memory holds as their render, the last one the answer. */
    std::vector<PromptMessage> held_;
    size_t                     held_items_ = 0;
    llama_token                held_last_  = LLAMA_TOKEN_NULL;
};
