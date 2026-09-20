package ai.airi.qwenmobile

import android.view.View
import android.widget.TextView

/**
 * The phase machine of one streamed answer. The pieces of the token stream
 * move the message between the thinking and the answer. The end of the
 * stream closes the message: a stop or an error inside the thinking leaves
 * an interrupted message, an error before any answer text leaves a failed
 * one. The clock gives milliseconds, and only differences of it are used.
 */
class AnswerStream(val answer: ChatMessage, private val clock: () -> Long) {
    private var thinkingStart = clock()

    /** Start the thinking time at this moment, after the model load. */
    fun markStart() {
        thinkingStart = clock()
    }

    /** Apply one piece of the stream to the message. */
    fun accept(piece: Piece) {
        when (piece) {
            is Piece.Text -> if (answer.phase == ChatMessage.Phase.THINKING) {
                answer.appendThinking(piece.text)
            } else {
                // The answer starts without the blank lines of the template.
                answer.appendContent(if (answer.hasContent) piece.text else piece.text.trimStart())
            }
            Piece.ThinkOpen -> if (!answer.hasContent) {
                // A tag that the model emits after the answer text has begun is
                // part of that text, not a new thinking block: a move back into
                // THINKING would end the turn as interrupted and drop the text.
                answer.phase = ChatMessage.Phase.THINKING
                answer.thinkingExpanded = null
                thinkingStart = clock()
            } else {
                answer.appendContent("<think>")
            }
            Piece.ThinkClose -> {
                answer.thinkingMs = clock() - thinkingStart
                answer.phase = ChatMessage.Phase.ANSWERING
                answer.thinkingExpanded = null
            }
        }
    }

    /**
     * Close the message at the end of the stream. [error] is the text of a
     * failure, or null. [cancelled] tells that the user stopped the answer.
     */
    fun finish(error: String?, cancelled: Boolean) {
        if (answer.phase == ChatMessage.Phase.THINKING) {
            answer.thinkingMs = clock() - thinkingStart
        }
        // The answer text decides first: a turn that produced text is done,
        // whatever phase it stopped in. Then the error, thus a failure inside
        // an open thinking still shows its message instead of the word
        // Interrupted. A thinking that produced nothing else is interrupted.
        answer.phase = when {
            answer.hasContent -> ChatMessage.Phase.DONE
            error != null -> {
                answer.content = error
                ChatMessage.Phase.FAILED
            }
            cancelled || answer.phase == ChatMessage.Phase.THINKING -> ChatMessage.Phase.INTERRUPTED
            else -> ChatMessage.Phase.DONE
        }
        answer.thinkingExpanded = null
    }
}

/**
 * The thinking field of an answer.
 *
 * The phase of the message drives the field: the body is open while the
 * model thinks, and closed when the thinking closes or the answer stops. A
 * tap on the header overrides that until the phase changes again.
 */
object Thinking {
    /** The text of the header of the thinking field. */
    enum class Label { IN_PROGRESS, INTERRUPTED, DONE, DONE_WITH_SECONDS }

    /** Show the thinking field of [message] in the header row and the body. */
    fun bind(header: View, body: TextView, message: ChatMessage) {
        header.visibility = View.VISIBLE
        apply(header, body, message)
        header.setOnClickListener {
            message.thinkingExpanded = !isExpanded(message)
            apply(header, body, message)
        }
    }

    /** Open while the model thinks, closed after, unless the user tapped the header since the last phase change. */
    fun isExpanded(message: ChatMessage): Boolean =
        message.thinkingExpanded ?: (message.phase == ChatMessage.Phase.THINKING)

    /** The header text of the thinking field of [message]. */
    fun labelOf(message: ChatMessage): Label = when {
        message.phase == ChatMessage.Phase.THINKING -> Label.IN_PROGRESS
        message.phase == ChatMessage.Phase.INTERRUPTED -> Label.INTERRUPTED
        message.thinkingMs > 0 -> Label.DONE_WITH_SECONDS
        else -> Label.DONE
    }

    /** The duration of the thinking in whole seconds, one at the minimum. */
    fun seconds(message: ChatMessage): Long = (message.thinkingMs / 1000L).coerceAtLeast(1L)

    private fun apply(header: View, body: TextView, message: ChatMessage) {
        val context = header.context
        val label = header.findViewById<TextView>(R.id.thinkingLabel)
        val chevron = header.findViewById<TextView>(R.id.thinkingChevron)
        label.text = when (labelOf(message)) {
            Label.IN_PROGRESS -> context.getString(R.string.thinking_in_progress)
            Label.INTERRUPTED -> context.getString(R.string.thinking_interrupted)
            Label.DONE_WITH_SECONDS -> context.getString(R.string.thinking_done_seconds, seconds(message))
            Label.DONE -> context.getString(R.string.thinking_done)
        }
        val expanded = isExpanded(message)
        chevron.rotation = if (expanded) 90f else 0f
        body.visibility = if (expanded) View.VISIBLE else View.GONE
        if (expanded) {
            Markdown.render(body, message.thinking, streaming = message.phase == ChatMessage.Phase.THINKING)
        } else {
            Markdown.clear(body)
        }
    }
}
