package ai.airi.qwenmobile

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Trace
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.PopupMenu
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import ai.airi.qwenmobile.databinding.FragmentChatBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * The conversation screen: a view over [ChatSession] with the message list,
 * the composer and the attachment. The session owns the messages and the
 * running answer, thus the screen can go and come back at any time.
 */
class ChatFragment : Fragment() {
    private var binding: FragmentChatBinding? = null
    private val session = ChatSession
    private val adapter = MessageAdapter(session.messages) { session.metaByMessage[it] }

    /** The encoded image that goes with the next message, and its thumbnail for the chip. */
    private var pendingImage: ByteArray? = null
    private var pendingThumbnail: Bitmap? = null

    /** An image shared to the app before the view exists. */
    private var sharedUri: Uri? = null

    /** The file the camera writes to. */
    private var captureUri: Uri? = null

    /** True while a redraw of the last row waits. */
    private var renderScheduled = false

    /**
     * True when the list shows its end. A streamed answer follows the list
     * only in that state, thus a user who scrolls up to read keeps the
     * position. Only a real scroll (dy ≠ 0) changes it, not a relayout.
     */
    private var atBottom = true

    /** The distance from the end of the list that still counts as the end, in pixels. */
    private var endSlackPx = 0

    private val settings by lazy { SettingsStore.of(requireContext()) }

    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            attachImage(uri)
        }
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        val uri = captureUri
        if (ok && uri != null) {
            attachImage(uri)
        }
    }

    private val askNotifications = registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        session.init(requireContext())
        val b = FragmentChatBinding.inflate(inflater, container, false)
        binding = b
        endSlackPx = (END_SLACK_DP * resources.displayMetrics.density).toInt()
        b.messages.layoutManager = LinearLayoutManager(requireContext())
        b.messages.adapter = adapter
        b.messages.itemAnimator = null
        b.messages.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                if (dy != 0) {
                    atBottom = isAtEnd(recyclerView)
                    updateJumpButton()
                }
            }
        })

        b.sendButton.setOnClickListener { onSend() }
        b.stopButton.setOnClickListener { session.stop() }
        b.jumpButton.setOnClickListener { scrollToEnd() }
        b.attachButton.setOnClickListener { showAttachMenu(it) }
        b.attachmentChip.setOnCloseIconClickListener { setPendingImage(null, null) }
        b.grantButton.setOnClickListener { startActivity(ModelFiles.allFilesAccessIntent(requireContext())) }
        showAttachment()

        val scope = viewLifecycleOwner.lifecycleScope
        scope.launch {
            session.structure.collect {
                adapter.notifyDataSetChanged()
                updateEmptyState()
                scrollToEnd()
            }
        }
        scope.launch { session.revision.collect { scheduleRender() } }
        scope.launch {
            session.generating.collect { active ->
                b.sendButton.visibility = if (active) View.GONE else View.VISIBLE
                b.stopButton.visibility = if (active) View.VISIBLE else View.GONE
                updateJumpButton()
                publishStatus()
            }
        }
        scope.launch { session.loading.collect { publishStatus() } }
        scope.launch { session.error.collect { publishStatus() } }
        scope.launch { session.problems.collect { toast(it) } }
        scope.launch { LlamaEngine.state.collect { publishStatus() } }
        scope.launch { settings.state.collect { publishStatus() } }
        return b.root
    }

    override fun onResume() {
        super.onResume()
        binding?.grantButton?.visibility = if (ModelFiles.hasAllFilesAccess()) View.GONE else View.VISIBLE
        sharedUri?.let { uri ->
            sharedUri = null
            attachImage(uri)
        }
        session.autoLoad()
    }

    override fun onDestroyView() {
        // The list holds an observer of the adapter: without this line the old list stays reachable.
        binding?.messages?.adapter = null
        renderScheduled = false
        binding = null
        super.onDestroyView()
    }

    // --- the model, from the app bar menu ---

    fun loadModel() {
        session.requestLoad()
    }

    fun unloadModel() {
        session.unload()
    }

    fun newChat() {
        session.clear()
    }

    /** The status chip and the title of the app bar. */
    private fun publishStatus() {
        val host = activity as? MainActivity ?: return
        val loaded = LlamaEngine.state.value
        val error = session.error.value
        when {
            session.loading.value -> host.setStatus(getString(R.string.status_loading), MainActivity.Tone.BUSY)
            loaded == null && error != null -> host.setStatus(getString(R.string.status_error), MainActivity.Tone.ERROR)
            loaded == null -> host.setStatus(getString(R.string.status_idle), MainActivity.Tone.IDLE)
            else -> {
                val id = if (session.generating.value) R.string.status_generating else R.string.status_ready
                host.setStatus(getString(id, loaded.config.backend.label), MainActivity.Tone.READY)
            }
        }
        val path = loaded?.config?.path ?: settings.state.value.modelPath
        host.setChatTitle(path?.let { ModelFiles.displayName(File(it)) } ?: getString(R.string.app_name))
    }

    // --- the attachment ---

    private fun showAttachMenu(anchor: View) {
        val menu = PopupMenu(requireContext(), anchor)
        menu.menu.add(0, 1, 0, R.string.attach_photo_library).setIcon(R.drawable.ic_image)
        menu.menu.add(0, 2, 1, R.string.attach_take_photo).setIcon(R.drawable.ic_camera)
        menu.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                1 -> pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                2 -> capturePhoto()
            }
            true
        }
        menu.show()
    }

    private fun capturePhoto() {
        val dir = File(requireContext().cacheDir, "images").apply { mkdirs() }
        val uri = FileProvider.getUriForFile(
            requireContext(),
            "${requireContext().packageName}.fileprovider",
            File(dir, "capture.jpg"),
        )
        captureUri = uri
        takePicture.launch(uri)
    }

    /** An image shared to the app: attach it now, or when the view exists. */
    fun attachShared(uri: Uri) {
        if (view == null) {
            sharedUri = uri
        } else {
            attachImage(uri)
        }
    }

    /** Read the image behind the URI off the main thread and hold it for the next message. */
    private fun attachImage(uri: Uri) {
        val resolver = requireContext().contentResolver
        val side = (THUMBNAIL_DP * resources.displayMetrics.density).toInt()
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val (bytes, thumbnail) = withContext(Dispatchers.IO) {
                    val bytes = ImageBytes.load(resolver, uri)
                    // The thumbnail is small, thus the decode samples the stored image: the header gives the size.
                    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
                    val options = BitmapFactory.Options().apply {
                        inSampleSize = ImageGeometry.sampleSize(bounds.outWidth, bounds.outHeight, ImageGeometry.Size(side, side))
                    }
                    val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
                    bytes to Bitmap.createScaledBitmap(bitmap, side, side, true)
                }
                setPendingImage(bytes, thumbnail)
            } catch (e: Exception) {
                toast(getString(R.string.image_failed, e.message ?: e.javaClass.simpleName))
            }
        }
    }

    private fun setPendingImage(bytes: ByteArray?, thumbnail: Bitmap?) {
        pendingImage = bytes
        pendingThumbnail = thumbnail
        showAttachment()
    }

    /** The chip of the pending image, or no chip. It survives a new view. */
    private fun showAttachment() {
        val b = binding ?: return
        val thumbnail = pendingThumbnail
        if (thumbnail == null) {
            b.attachmentRow.visibility = View.GONE
            b.attachmentChip.chipIcon = null
            return
        }
        b.attachmentChip.chipIcon = BitmapDrawable(resources, thumbnail)
        b.attachmentRow.visibility = View.VISIBLE
    }

    // --- the turn ---

    private fun onSend() {
        val b = binding ?: return
        val typed = b.input.text.toString().trim()
        val image = pendingImage
        if ((typed.isEmpty() && image == null) || session.generating.value) {
            return
        }
        ensureNotificationPermission()
        val text = typed.ifEmpty { getString(R.string.describe_image) }
        b.input.text?.clear()
        setPendingImage(null, null)
        session.send(text, image)
    }

    /** The foreground service shows a notification, thus the app asks for the permission one time. */
    private fun ensureNotificationPermission() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            return
        }
        val granted = ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.POST_NOTIFICATIONS)
        if (granted != PackageManager.PERMISSION_GRANTED) {
            askNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    /** Redraw the last row at most every 80 ms while the answer streams. */
    private fun scheduleRender() {
        val list = binding?.messages ?: return
        if (renderScheduled) {
            return
        }
        renderScheduled = true
        list.postDelayed({ renderLast() }, RENDER_INTERVAL_MS)
    }

    private fun renderLast() {
        renderScheduled = false
        if (binding == null || session.messages.isEmpty()) {
            return
        }
        // The decision comes before the update: the relayout of a grown row must not change it.
        val follow = atBottom
        Trace.beginSection("piece-render")
        try {
            adapter.notifyItemChanged(session.messages.size - 1, MessageAdapter.PAYLOAD_TEXT)
            if (follow) {
                scrollToEnd()
            }
        } finally {
            Trace.endSection()
        }
    }

    /** Show the end of the list after the next layout, and follow the stream again. */
    private fun scrollToEnd() {
        val b = binding ?: return
        b.messages.post {
            binding?.messages?.scrollBy(0, Int.MAX_VALUE / 2)
            atBottom = true
            updateJumpButton()
        }
    }

    /** True when the end of the list is in view, with a small margin for a fling that stops near it. */
    private fun isAtEnd(list: RecyclerView): Boolean {
        val below = list.computeVerticalScrollRange() - list.computeVerticalScrollExtent() - list.computeVerticalScrollOffset()
        return below <= endSlackPx
    }

    /** The round arrow above the composer: only while an answer streams and the list is scrolled up. */
    private fun updateJumpButton() {
        val b = binding ?: return
        b.jumpButton.visibility = if (session.generating.value && !atBottom) View.VISIBLE else View.GONE
    }

    private fun updateEmptyState() {
        binding?.emptyText?.visibility = if (session.messages.isEmpty()) View.VISIBLE else View.GONE
    }

    private fun toast(message: String) {
        val context = context ?: return
        Toast.makeText(context, message, Toast.LENGTH_LONG).show()
    }

    private companion object {
        const val RENDER_INTERVAL_MS = 80L
        const val THUMBNAIL_DP = 28
        const val END_SLACK_DP = 24
    }
}
