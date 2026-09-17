package ai.airi.qwenmobile

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
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

    /** The encoded image that goes with the next message. */
    private var pendingImage: ByteArray? = null

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
        b.messages.layoutManager = LinearLayoutManager(requireContext())
        b.messages.adapter = adapter
        b.messages.itemAnimator = null
        b.messages.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                if (dy != 0) {
                    atBottom = !recyclerView.canScrollVertically(1)
                    updateJumpButton()
                }
            }
        })

        b.sendButton.setOnClickListener { onSend() }
        b.stopButton.setOnClickListener { session.stop() }
        b.jumpButton.setOnClickListener { scrollToEnd() }
        b.attachButton.setOnClickListener { showAttachMenu(it) }
        b.attachmentChip.setOnCloseIconClickListener { setPendingImage(null) }
        b.grantButton.setOnClickListener { startActivity(ModelFiles.allFilesAccessIntent(requireContext())) }

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
        viewLifecycleOwner.lifecycleScope.launch {
            session.autoLoad()?.let { toast(it) }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        binding = null
    }

    // --- the model, from the app bar menu ---

    fun loadModel() {
        viewLifecycleOwner.lifecycleScope.launch {
            session.loadFromSettings()?.let { toast(it) }
        }
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
        when {
            session.loading.value -> host.setStatus(getString(R.string.status_loading), MainActivity.Tone.BUSY)
            loaded == null -> host.setStatus(getString(R.string.status_idle), MainActivity.Tone.IDLE)
            else -> {
                val id = if (session.generating.value) R.string.status_generating else R.string.status_ready
                host.setStatus(getString(id, loaded.config.backend.label), MainActivity.Tone.READY)
            }
        }
        val path = loaded?.config?.path ?: settings.state.value.modelPath
        host.setTitle(path?.let { File(it).nameWithoutExtension } ?: getString(R.string.app_name))
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

    /** Read the image behind the URI and hold it for the next message. */
    private fun attachImage(uri: Uri) {
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val resolver = requireContext().contentResolver
                setPendingImage(withContext(Dispatchers.IO) { ImageBytes.load(resolver, uri) })
            } catch (e: Exception) {
                toast(getString(R.string.image_failed, e.message ?: "?"))
            }
        }
    }

    private fun setPendingImage(bytes: ByteArray?) {
        val b = binding ?: return
        pendingImage = bytes
        if (bytes == null) {
            b.attachmentRow.visibility = View.GONE
            b.attachmentChip.chipIcon = null
            return
        }
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val side = (28 * resources.displayMetrics.density).toInt()
        b.attachmentChip.chipIcon = BitmapDrawable(resources, Bitmap.createScaledBitmap(bitmap, side, side, true))
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
        setPendingImage(null)
        session.send(text, image) { problem -> toast(problem) }
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
        if (renderScheduled) {
            return
        }
        renderScheduled = true
        binding?.messages?.postDelayed({ renderLast() }, 80)
    }

    private fun renderLast() {
        renderScheduled = false
        if (binding == null || session.messages.isEmpty()) {
            return
        }
        // The decision comes before the update: the relayout of a grown row must not change it.
        val follow = atBottom
        adapter.notifyItemChanged(session.messages.size - 1, MessageAdapter.PAYLOAD_TEXT)
        if (follow) {
            scrollToEnd()
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

    /** The round arrow above the composer: only while an answer streams and the list is scrolled up. */
    private fun updateJumpButton() {
        val b = binding ?: return
        b.jumpButton.visibility = if (session.generating.value && !atBottom) View.VISIBLE else View.GONE
    }

    private fun updateEmptyState() {
        binding?.emptyText?.visibility = if (session.messages.isEmpty()) View.VISIBLE else View.GONE
    }

    private fun toast(message: String) {
        Toast.makeText(requireContext(), message, Toast.LENGTH_LONG).show()
    }
}
