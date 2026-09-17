package ai.airi.qwenmobile

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat

/**
 * A foreground service that holds the process alive while an answer streams.
 * The generation itself runs in [ChatSession]. The service only shows the
 * notification "Qwen is answering" and stops when the session tells it to.
 */
class GenerationService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        return START_NOT_STICKY
    }

    private fun buildNotification(): Notification {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL, getString(R.string.notification_channel), NotificationManager.IMPORTANCE_LOW),
        )
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_chat)
            .setContentTitle(getString(R.string.notification_answering))
            .setContentIntent(open)
            .setOngoing(true)
            .setSilent(true)
            .build()
    }

    companion object {
        private const val CHANNEL = "generation"
        private const val NOTIFICATION_ID = 1
        private const val ACTION_STOP = "ai.airi.qwenmobile.STOP"

        /** Show the notification and keep the process in the foreground. */
        fun start(context: Context) {
            ContextCompat.startForegroundService(context, Intent(context, GenerationService::class.java))
        }

        /** Remove the notification and stop the service. */
        fun stop(context: Context) {
            context.startService(Intent(context, GenerationService::class.java).setAction(ACTION_STOP))
        }
    }
}
