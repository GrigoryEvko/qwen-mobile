package ai.airi.qwenmobile

import android.app.GameManager
import android.app.GameState
import android.content.Intent
import android.content.res.ColorStateList
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.IntentCompat
import androidx.fragment.app.Fragment
import ai.airi.qwenmobile.databinding.ActivityMainBinding

/**
 * The app bar, the three destinations of the bottom navigation, and the
 * status chip. The fragments stay alive across the tabs. After a process
 * restart the fragment manager restores them, thus the activity finds them
 * by tag before it makes new ones.
 */
class MainActivity : AppCompatActivity() {
    /** The tone of the status chip. */
    enum class Tone { IDLE, BUSY, READY, ERROR }

    private lateinit var binding: ActivityMainBinding
    private lateinit var chat: ChatFragment
    private lateinit var settings: Fragment
    private lateinit var debug: Fragment

    /** The title of the chat tab: the model name, or the app name. */
    private var chatTitle: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val fm = supportFragmentManager
        chat = fm.findFragmentByTag(TAG_CHAT) as? ChatFragment ?: ChatFragment()
        settings = fm.findFragmentByTag(TAG_SETTINGS) ?: SettingsFragment()
        debug = fm.findFragmentByTag(TAG_DEBUG) ?: DebugFragment()
        val tx = fm.beginTransaction()
        for ((fragment, tag) in listOf(chat to TAG_CHAT, settings to TAG_SETTINGS, debug to TAG_DEBUG)) {
            if (!fragment.isAdded) {
                tx.add(binding.container.id, fragment, tag)
            }
        }
        tx.commitNow()

        binding.bottomNav.setOnItemSelectedListener { item ->
            show(
                when (item.itemId) {
                    R.id.nav_settings -> settings
                    R.id.nav_debug -> debug
                    else -> chat
                },
            )
            true
        }
        show(
            when (binding.bottomNav.selectedItemId) {
                R.id.nav_settings -> settings
                R.id.nav_debug -> debug
                else -> chat
            },
        )
        binding.toolbar.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_new_chat -> chat.newChat()
                R.id.action_load -> chat.loadModel()
                R.id.action_unload -> chat.unloadModel()
            }
            true
        }
        takeSharedImage(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        takeSharedImage(intent)
    }

    override fun onResume() {
        super.onResume()
        // The game state tells the OEM policy that gameplay is in progress.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            getSystemService(GameManager::class.java)
                .setGameState(GameState(false, GameState.MODE_GAMEPLAY_UNINTERRUPTIBLE))
        }
    }

    /** The title of the app bar, the model name on the chat tab. */
    fun setTitle(title: String) {
        chatTitle = title
        if (binding.bottomNav.selectedItemId == R.id.nav_chat) {
            binding.toolbar.title = title
        }
    }

    /** The status chip: the text and the color of its tone. */
    fun setStatus(text: String, tone: Tone) {
        val chip = binding.statusChip
        chip.text = text
        val (background, foreground) = when (tone) {
            Tone.IDLE, Tone.BUSY -> R.color.surface_variant to R.color.muted
            Tone.READY -> R.color.ready_green_bg to R.color.ready_green
            Tone.ERROR -> R.color.error_red_bg to R.color.error_red
        }
        chip.chipBackgroundColor = ColorStateList.valueOf(ContextCompat.getColor(this, background))
        chip.setTextColor(ContextCompat.getColor(this, foreground))
    }

    /** An image shared to the app goes to the chat as the attachment of the next message. */
    private fun takeSharedImage(intent: Intent?) {
        if (intent?.action != Intent.ACTION_SEND || intent.type?.startsWith("image/") != true) {
            return
        }
        val uri = IntentCompat.getParcelableExtra(intent, Intent.EXTRA_STREAM, Uri::class.java) ?: return
        chat.attachShared(uri)
        binding.bottomNav.selectedItemId = R.id.nav_chat
    }

    private fun show(target: Fragment) {
        val tx = supportFragmentManager.beginTransaction()
        for (fragment in listOf(chat, settings, debug)) {
            if (fragment === target) tx.show(fragment) else tx.hide(fragment)
        }
        tx.commit()
        val onChat = target === chat
        binding.toolbar.menu.setGroupVisible(0, onChat)
        binding.statusChip.visibility = if (onChat) android.view.View.VISIBLE else android.view.View.GONE
        binding.toolbar.title = when (target) {
            settings -> getString(R.string.tab_settings)
            debug -> getString(R.string.tab_debug)
            else -> chatTitle.ifEmpty { getString(R.string.app_name) }
        }
    }

    private companion object {
        const val TAG_CHAT = "chat"
        const val TAG_SETTINGS = "settings"
        const val TAG_DEBUG = "debug"
    }
}
