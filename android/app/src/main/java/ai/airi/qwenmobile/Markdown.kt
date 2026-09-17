package ai.airi.qwenmobile

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.widget.TextView
import com.google.android.material.color.MaterialColors
import io.noties.markwon.AbstractMarkwonPlugin
import io.noties.markwon.Markwon
import io.noties.markwon.core.MarkwonTheme
import io.noties.markwon.ext.strikethrough.StrikethroughPlugin
import io.noties.markwon.ext.tables.TablePlugin
import io.noties.markwon.html.HtmlPlugin
import io.noties.markwon.linkify.LinkifyPlugin
import java.util.WeakHashMap

/**
 * Renders Markdown into a TextView with Markwon.
 *
 * The answer text changes many times per second while it streams. One view
 * parses at most one time per [THROTTLE_MS]. A change inside that window
 * waits for the next slot, and the last text always renders. A streamed
 * text is parsed in two parts: the head up to the last stable paragraph
 * break comes from the previous render, only the tail is parsed again.
 * The final render of a text is kept while the text is alive, thus a row
 * that binds again on a scroll does not parse.
 */
object Markdown {
    /** The minimum interval between two parses of one view, in milliseconds. */
    private const val THROTTLE_MS = 80L

    /** The heading sizes relative to the body text, h1 to h6. */
    private val HEADING_SIZES = floatArrayOf(1.35f, 1.2f, 1.1f, 1.0f, 1.0f, 1.0f)

    /** The render state of one view. */
    private class State {
        var pending: String? = null
        var streaming = false
        var lastRender = 0L
        var scheduled = false

        /** The source and the render of the head of the last streamed text. */
        var headSource = ""
        var headSpanned: Spanned? = null
    }

    private val states = WeakHashMap<TextView, State>()

    /** The final render of each text, by the text. The entry goes when no message holds the text. */
    private val rendered = WeakHashMap<String, Spanned>()

    /** The main thread. A view cannot post while it is detached, thus the render does not go through it. */
    private val handler by lazy { Handler(Looper.getMainLooper()) }

    private var markwon: Markwon? = null

    /**
     * Render [markdown] into [view]. Repeated calls while the text streams
     * are throttled. With [streaming], the head of the text comes from the
     * last render and only the tail after the last stable paragraph break
     * is parsed. A call without [streaming] parses the full text.
     */
    fun render(view: TextView, markdown: String, streaming: Boolean = false) {
        val state = states.getOrPut(view) { State() }
        state.pending = markdown
        state.streaming = streaming
        val wait = THROTTLE_MS - (SystemClock.uptimeMillis() - state.lastRender)
        if (wait <= 0) {
            flush(view, state)
        } else if (!state.scheduled) {
            state.scheduled = true
            handler.postDelayed({
                state.scheduled = false
                flush(view, state)
            }, wait)
        }
    }

    /** Remove the render that waits for [view], because the view shows a different text at this time. */
    fun clear(view: TextView) {
        states[view]?.pending = null
    }

    /**
     * The length of the head of a streamed text that later tokens do not
     * change: the text up to the last blank line that is outside a fenced
     * code block and that a line without indentation follows. Zero when
     * there is no such line. Complexity is O(length).
     */
    fun stableSplit(text: String): Int {
        var split = 0
        var fenceChar = NO_FENCE
        var fenceLength = 0
        var blankBefore = false
        var lineStart = 0
        while (lineStart < text.length) {
            var lineEnd = text.indexOf('\n', lineStart)
            if (lineEnd < 0) {
                lineEnd = text.length
            }
            val blank = isBlank(text, lineStart, lineEnd)
            val at = afterIndent(text, lineStart, lineEnd)
            val marker = if (at >= 0 && (text[at] == '`' || text[at] == '~')) text[at] else NO_FENCE
            val run = if (marker != NO_FENCE) runLength(text, at, lineEnd, marker) else 0
            if (fenceChar == NO_FENCE) {
                if (blankBefore && !blank && text[lineStart] != ' ' && text[lineStart] != '\t') {
                    split = lineStart
                }
                if (run >= 3) {
                    fenceChar = marker
                    fenceLength = run
                }
            } else if (marker == fenceChar && run >= fenceLength && isBlank(text, at + run, lineEnd)) {
                fenceChar = NO_FENCE
            }
            blankBefore = blank
            lineStart = lineEnd + 1
        }
        return split
    }

    private const val NO_FENCE = Char.MIN_VALUE

    private fun isBlank(text: String, start: Int, end: Int): Boolean {
        for (i in start until end) {
            if (text[i] != ' ' && text[i] != '\t') {
                return false
            }
        }
        return true
    }

    /** The index of the first character after a maximum of three spaces, or -1 for a blank or a deeper indentation. */
    private fun afterIndent(text: String, start: Int, end: Int): Int {
        var i = start
        while (i < end && i - start < 3 && text[i] == ' ') {
            i++
        }
        return if (i < end && text[i] != ' ') i else -1
    }

    private fun runLength(text: String, at: Int, end: Int, marker: Char): Int {
        var i = at
        while (i < end && text[i] == marker) {
            i++
        }
        return i - at
    }

    private fun flush(view: TextView, state: State) {
        val text = state.pending ?: return
        state.pending = null
        state.lastRender = SystemClock.uptimeMillis()
        val markwon = of(view.context)
        val spanned = if (state.streaming) {
            streamed(markwon, state, text)
        } else {
            state.headSource = ""
            state.headSpanned = null
            rendered.getOrPut(text) { markwon.toMarkdown(text) }
        }
        markwon.setParsedMarkdown(view, spanned)
    }

    /** The render of a text that still grows: the cached head plus a fresh parse of the tail. */
    private fun streamed(markwon: Markwon, state: State, text: String): Spanned {
        val split = stableSplit(text)
        if (split == 0) {
            return markwon.toMarkdown(text)
        }
        val cached = state.headSpanned
        val head: Spanned = if (cached != null && state.headSource.length == split && text.regionMatches(0, state.headSource, 0, split)) {
            cached
        } else {
            state.headSource = text.substring(0, split)
            markwon.toMarkdown(state.headSource).also { state.headSpanned = it }
        }
        val tail = markwon.toMarkdown(text.substring(split))
        // Markwon puts one empty line between two blocks: a line break when the head has none, then one more.
        return SpannableStringBuilder(head).apply {
            if (isNotEmpty() && this[length - 1] != '\n') {
                append('\n')
            }
            append('\n')
            append(tail)
        }
    }

    /** The Markwon instance, made one time from the theme of the first view. */
    private fun of(context: Context): Markwon {
        markwon?.let { return it }
        val density = context.resources.displayMetrics.density
        val scaled = context.resources.displayMetrics.scaledDensity
        val codeBackground = MaterialColors.getColor(
            context, com.google.android.material.R.attr.colorSurfaceContainerHighest, 0xFFF1F3F5.toInt(),
        )
        val linkColor = MaterialColors.getColor(context, androidx.appcompat.R.attr.colorPrimary, 0xFF1A5FB4.toInt())
        return Markwon.builder(context.applicationContext)
            .usePlugin(StrikethroughPlugin.create())
            .usePlugin(TablePlugin.create(context.applicationContext))
            .usePlugin(HtmlPlugin.create())
            .usePlugin(LinkifyPlugin.create())
            .usePlugin(object : AbstractMarkwonPlugin() {
                override fun configureTheme(builder: MarkwonTheme.Builder) {
                    builder
                        .headingTextSizeMultipliers(HEADING_SIZES)
                        .headingBreakHeight(0)
                        .blockMargin((16 * density).toInt())
                        .bulletWidth((6 * density).toInt())
                        .codeBackgroundColor(codeBackground)
                        .codeBlockBackgroundColor(codeBackground)
                        .codeBlockMargin((12 * density).toInt())
                        .codeTextSize((13 * scaled).toInt())
                        .codeBlockTextSize((13 * scaled).toInt())
                        .linkColor(linkColor)
                }
            })
            .build()
            .also { markwon = it }
    }
}
