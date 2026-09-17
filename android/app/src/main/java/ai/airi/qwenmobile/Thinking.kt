package ai.airi.qwenmobile

import android.view.View
import android.widget.TextView

/**
 * The thinking field of an answer.
 *
 * The phase of the message drives the field: the body is open while the
 * model thinks, and closed when the thinking closes or the answer stops. A
 * tap on the header overrides that until the phase changes again.
 */
object Thinking {
    /** Show the thinking field of [message] in the header row and the body. */
    fun bind(header: View, body: TextView, message: ChatMessage) {
        header.visibility = View.VISIBLE
        apply(header, body, message)
        header.setOnClickListener {
            message.thinkingExpanded = !(message.thinkingExpanded ?: defaultExpanded(message))
            apply(header, body, message)
        }
    }

    /** Open while the model thinks, closed after. */
    private fun defaultExpanded(message: ChatMessage): Boolean = message.phase == ChatMessage.Phase.THINKING

    private fun apply(header: View, body: TextView, message: ChatMessage) {
        val context = header.context
        val label = header.findViewById<TextView>(R.id.thinkingLabel)
        val chevron = header.findViewById<TextView>(R.id.thinkingChevron)
        label.text = when (message.phase) {
            ChatMessage.Phase.THINKING -> context.getString(R.string.thinking_in_progress)
            ChatMessage.Phase.INTERRUPTED -> context.getString(R.string.thinking_interrupted)
            else -> if (message.thinkingMs > 0) {
                context.getString(R.string.thinking_done_seconds, (message.thinkingMs / 1000L).coerceAtLeast(1L))
            } else {
                context.getString(R.string.thinking_done)
            }
        }
        val expanded = message.thinkingExpanded ?: defaultExpanded(message)
        chevron.rotation = if (expanded) 90f else 0f
        body.visibility = if (expanded) View.VISIBLE else View.GONE
        if (expanded) {
            Markdown.render(body, message.thinking)
        }
    }
}
