package ai.airi.qwenmobile

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import ai.airi.qwenmobile.databinding.ItemMessageBinding

/**
 * The message list. Every message takes the full width, with the avatar and
 * the role above the body. The assistant body is Markdown with the thinking
 * block above it. The user body sits in a rounded container.
 *
 * A streamed answer updates through [PAYLOAD_TEXT], which touches only the
 * text views of the bound row.
 */
class MessageAdapter(
    private val messages: List<ChatMessage>,
    private val meta: (ChatMessage) -> String?,
) : RecyclerView.Adapter<MessageAdapter.Holder>() {

    class Holder(val binding: ItemMessageBinding) : RecyclerView.ViewHolder(binding.root) {
        val thinkingHeader: View? = binding.root.findViewById(R.id.thinkingHeader)
        val thinkingBody: TextView? = binding.root.findViewById(R.id.thinkingBody)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder =
        Holder(ItemMessageBinding.inflate(LayoutInflater.from(parent.context), parent, false))

    override fun getItemCount(): Int = messages.size

    override fun onBindViewHolder(holder: Holder, position: Int) {
        val message = messages[position]
        val isUser = message.role == "user"
        val b = holder.binding
        val res = b.root.resources
        b.avatar.text = if (isUser) "U" else "Q"
        b.avatar.setBackgroundResource(if (isUser) R.drawable.bg_avatar_user else R.drawable.bg_avatar_assistant)
        b.roleText.text = res.getString(if (isUser) R.string.role_user else R.string.role_assistant)
        val bitmap = message.bitmap
        b.imageView.setImageBitmap(bitmap)
        b.imageView.visibility = if (bitmap == null) View.GONE else View.VISIBLE
        if (isUser) {
            b.body.setBackgroundResource(R.drawable.bg_user_message)
            val pad = (12 * res.displayMetrics.density).toInt()
            b.body.setPadding(pad, pad, pad, pad)
        } else {
            b.body.background = null
            b.body.setPadding(0, 0, 0, 0)
        }
        bindText(holder, message)
    }

    override fun onBindViewHolder(holder: Holder, position: Int, payloads: MutableList<Any>) {
        if (payloads.contains(PAYLOAD_TEXT)) {
            bindText(holder, messages[position])
        } else {
            super.onBindViewHolder(holder, position, payloads)
        }
    }

    /** The text of the body: plain for the user, thinking plus Markdown for the assistant. */
    private fun bindText(holder: Holder, message: ChatMessage) {
        val b = holder.binding
        if (message.role == "user") {
            b.thinkingContainer.visibility = View.GONE
            b.contentText.text = message.content
        } else {
            val parts = Thinking.split(message.content)
            val header = holder.thinkingHeader
            val body = holder.thinkingBody
            if (parts.thinking != null && header != null && body != null) {
                b.thinkingContainer.visibility = View.VISIBLE
                Thinking.bind(header, body, parts)
            } else {
                b.thinkingContainer.visibility = View.GONE
            }
            if (parts.answer.isBlank() && parts.thinking == null) {
                b.contentText.text = "…"
            } else {
                Markdown.render(b.contentText, parts.answer)
            }
        }
        val info = meta(message)
        b.metaText.text = info
        b.metaText.visibility = if (info == null) View.GONE else View.VISIBLE
    }

    companion object {
        /** The payload of a text-only update while an answer streams. */
        const val PAYLOAD_TEXT = "text"
    }
}
