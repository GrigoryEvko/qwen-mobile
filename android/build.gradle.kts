// AGP 9 has built-in Kotlin. Do not apply org.jetbrains.kotlin.android here.
plugins {
    id("com.android.application") version "9.4.0" apply false
}

// buildscript-gradle.lockfile in this directory pins the plugin classpath. After
// a change of the plugin version, write it again with:
//   ./gradlew buildEnvironment --write-locks -Pprebuilt=true
buildscript {
    configurations.classpath {
        resolutionStrategy.activateDependencyLocking()
    }
}
