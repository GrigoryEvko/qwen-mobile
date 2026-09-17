package ai.airi.qwenmobile

import android.content.Context
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import androidx.recyclerview.widget.RecyclerView

/**
 * The message list. Only the user and the follow logic of the chat scroll
 * it. A selectable text view asks its parent to show its cursor after every
 * text change, which moved the list to the streaming row on each token.
 */
class MessageList @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : RecyclerView(context, attrs, defStyleAttr) {

    /** Refuse the request: the rows do not move the list. */
    override fun requestChildRectangleOnScreen(child: View, rect: Rect, immediate: Boolean): Boolean = false
}
