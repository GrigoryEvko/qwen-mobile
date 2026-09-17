package ai.airi.qwenmobile

import android.content.Context
import android.os.SystemClock
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
 * waits for the next slot, and the last text always renders.
 */
object Markdown {
    /** The minimum interval between two parses of one view, in milliseconds. */
    private const val THROTTLE_MS = 80L

    /** The heading sizes relative to the body text, h1 to h6. */
    private val HEADING_SIZES = floatArrayOf(1.35f, 1.2f, 1.1f, 1.0f, 1.0f, 1.0f)

    /** The text and the time of the last render of one view. */
    private class State {
        var pending: String? = null
        var lastRender = 0L
        var scheduled = false
    }

    private val states = WeakHashMap<TextView, State>()

    private var markwon: Markwon? = null

    /** Render [markdown] into [view]. Repeated calls while the text streams are throttled. */
    fun render(view: TextView, markdown: String) {
        val state = states.getOrPut(view) { State() }
        state.pending = markdown
        val wait = THROTTLE_MS - (SystemClock.uptimeMillis() - state.lastRender)
        if (wait <= 0) {
            flush(view, state)
        } else if (!state.scheduled) {
            state.scheduled = true
            view.postDelayed({
                state.scheduled = false
                flush(view, state)
            }, wait)
        }
    }

    private fun flush(view: TextView, state: State) {
        val text = state.pending ?: return
        state.pending = null
        state.lastRender = SystemClock.uptimeMillis()
        of(view.context).setMarkdown(view, text)
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
