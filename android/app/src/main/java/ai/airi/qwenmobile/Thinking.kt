package ai.airi.qwenmobile

import android.os.SystemClock
import android.view.View
import android.widget.TextView
import java.util.WeakHashMap

/**
 * The thinking part of an answer.
 *
 * Qwen3.5 in thinking mode writes `<think>...</think>` before the answer.
 * While the closing tag is missing, the thinking is in progress. In the
 * other mode the model writes an empty block, which counts as no thinking.
 */
object Thinking {
    private const val OPEN = "<think>"
    private const val CLOSE = "</think>"

    /** The split of a raw answer. [thinking] is null when the answer has no thinking. */
    data class Parts(val thinking: String?, val answer: String, val thinkingDone: Boolean)

    /** The times and the choice of the user for one header view. */
    private class State {
        var lastThinking = ""
        var startedAt = 0L
        var doneAt = 0L
        var expanded: Boolean? = null
    }

    private val states = WeakHashMap<View, State>()

    /** Split the raw answer into the thinking and the visible answer. */
    fun split(raw: String): Parts {
        val start = raw.indexOf(OPEN)
        if (start < 0 || raw.substring(0, start).isNotBlank()) {
            return Parts(null, raw, true)
        }
        val from = start + OPEN.length
        val end = raw.indexOf(CLOSE, from)
        if (end < 0) {
            return Parts(raw.substring(from).trim(), "", false)
        }
        val thinking = raw.substring(from, end).trim()
        return Parts(thinking.ifEmpty { null }, raw.substring(end + CLOSE.length).trimStart(), true)
    }

    /**
     * Show the thinking of [parts] in the header row and the body.
     *
     * The header shows the state and toggles the body. The body is hidden
     * until the user taps the header, in progress or done. Both views hide
     * when there is no thinking.
     */
    fun bind(header: View, body: TextView, parts: Parts) {
        val thinking = parts.thinking
        if (thinking == null) {
            header.visibility = View.GONE
            body.visibility = View.GONE
            return
        }
        var state = states.getOrPut(header) { State() }
        val sameAnswer = thinking.startsWith(state.lastThinking) && (state.doneAt == 0L || parts.thinkingDone)
        if (!sameAnswer) {
            state = State()
            states[header] = state
        }
        state.lastThinking = thinking
        val now = SystemClock.uptimeMillis()
        if (!parts.thinkingDone && state.startedAt == 0L) {
            state.startedAt = now
        }
        if (parts.thinkingDone && state.startedAt != 0L && state.doneAt == 0L) {
            state.doneAt = now
        }
        header.visibility = View.VISIBLE
        apply(header, body, state, parts.thinkingDone)
        header.setOnClickListener {
            state.expanded = !(state.expanded ?: false)
            apply(header, body, state, parts.thinkingDone)
        }
    }

    private fun apply(header: View, body: TextView, state: State, done: Boolean) {
        val context = header.context
        val label = header.findViewById<TextView>(R.id.thinkingLabel)
        val chevron = header.findViewById<TextView>(R.id.thinkingChevron)
        label.text = when {
            !done -> context.getString(R.string.thinking_in_progress)
            state.startedAt != 0L -> {
                val seconds = ((state.doneAt - state.startedAt) / 1000L).coerceAtLeast(1L)
                context.getString(R.string.thinking_done_seconds, seconds)
            }
            else -> context.getString(R.string.thinking_done)
        }
        val expanded = state.expanded ?: false
        chevron.rotation = if (expanded) 90f else 0f
        body.visibility = if (expanded) View.VISIBLE else View.GONE
        if (expanded) {
            Markdown.render(body, state.lastThinking)
        }
    }
}
