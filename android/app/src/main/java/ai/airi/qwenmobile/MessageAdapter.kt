package ai.airi.qwenmobile

import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import androidx.recyclerview.widget.RecyclerView
import ai.airi.qwenmobile.databinding.ItemMessageBinding
import com.google.android.material.color.MaterialColors

/** The message list. The user is on the right, the assistant on the left. */
class MessageAdapter(private val messages: List<ChatMessage>) :
    RecyclerView.Adapter<MessageAdapter.Holder>() {

    class Holder(val binding: ItemMessageBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder =
        Holder(ItemMessageBinding.inflate(LayoutInflater.from(parent.context), parent, false))

    override fun getItemCount(): Int = messages.size

    override fun onBindViewHolder(holder: Holder, position: Int) {
        val message = messages[position]
        val isUser = message.role == "user"
        val b = holder.binding
        b.roleText.text = message.role
        b.contentText.text = message.content
        val bitmap = message.bitmap
        b.imageView.setImageBitmap(bitmap)
        b.imageView.visibility = if (bitmap == null) View.GONE else View.VISIBLE
        b.row.gravity = if (isUser) Gravity.END else Gravity.START
        val params = b.card.layoutParams as LinearLayout.LayoutParams
        params.gravity = if (isUser) Gravity.END else Gravity.START
        b.card.layoutParams = params
        val attr = if (isUser) {
            com.google.android.material.R.attr.colorPrimaryContainer
        } else {
            com.google.android.material.R.attr.colorSurfaceVariant
        }
        b.card.setCardBackgroundColor(MaterialColors.getColor(b.card, attr))
    }
}
