package ai.airi.qwenmobile

import android.app.GameManager
import android.app.GameState
import android.os.Build
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import ai.airi.qwenmobile.databinding.ActivityMainBinding
import com.google.android.material.tabs.TabLayout

/** Two tabs: the chat and the debug/benchmark screen. The fragments stay alive. */
class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val chat = ChatFragment()
    private val debug = DebugFragment()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        LlamaNative.setWorkingDirectory(filesDir.absolutePath)

        if (savedInstanceState == null) {
            supportFragmentManager.beginTransaction()
                .add(binding.container.id, chat, "chat")
                .add(binding.container.id, debug, "debug")
                .hide(debug)
                .commit()
        }
        binding.tabs.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) {
                show(if (tab.position == 0) chat else debug)
            }

            override fun onTabUnselected(tab: TabLayout.Tab) = Unit
            override fun onTabReselected(tab: TabLayout.Tab) = Unit
        })
    }

    override fun onResume() {
        super.onResume()
        // The game state tells the OEM policy that gameplay is in progress.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            getSystemService(GameManager::class.java)
                .setGameState(GameState(false, GameState.MODE_GAMEPLAY_UNINTERRUPTIBLE))
        }
    }

    private fun show(target: Fragment) {
        val other = if (target === chat) debug else chat
        supportFragmentManager.beginTransaction().hide(other).show(target).commit()
    }
}
