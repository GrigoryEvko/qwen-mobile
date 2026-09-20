import java.util.Properties

plugins {
    id("com.android.application")
}

// The llama.cpp checkout is the submodule third_party/llama.cpp, which holds the patch series.
// llama.dir in local.properties, or LLAMA_CPP_DIR, selects a different checkout.
val localProperties = Properties().apply {
    val file = rootProject.file("local.properties")
    if (file.exists()) {
        file.inputStream().use { load(it) }
    }
}
val llamaDir: String = localProperties.getProperty("llama.dir")
    ?: System.getenv("LLAMA_CPP_DIR")
    ?: rootProject.file("../third_party/llama.cpp").canonicalPath

// ./gradlew assembleRelease -Pprebuilt=true packages the libraries of the Snapdragon
// container build (CPU, OpenCL and Hexagon backends) from snapdragon/jniLibs, made by
// scripts/build-native.sh, and does not run the CMake build of llama.cpp.
val prebuilt: Boolean = project.findProperty("prebuilt") == "true"

// ./gradlew assembleRelease -Punsigned=true writes app-release-unsigned.apk, the
// reproducible artifact. Without it, the debug key of this machine signs the release.
val unsigned: Boolean = project.findProperty("unsigned") == "true"

// The version of the app. android/version.properties holds versionName, and the
// release workflow tags v<versionName> when that tag does not exist yet.
// versionCode is the number of commits of the branch, which scripts/build-apk.sh
// passes as -PversionCode. A build without that property takes the fallback of
// the file, thus the version of the app is readable from the source alone.
val versionProperties = Properties().apply {
    rootProject.file("version.properties").inputStream().use { load(it) }
}
val appVersionName: String = versionProperties.getProperty("versionName")
    ?: error("Set versionName in android/version.properties")
val appVersionCode: Int = (project.findProperty("versionCode") as String?)?.toInt()
    ?: versionProperties.getProperty("versionCodeFallback").toInt()

android {
    namespace = "ai.airi.qwenmobile"
    compileSdk = 36
    // The pins of the SDK components. ci/Containerfile installs these versions.
    buildToolsVersion = "36.0.0"
    ndkVersion = "30.0.16248370"

    defaultConfig {
        applicationId = "ai.airi.qwenmobile"
        minSdk = 28
        targetSdk = 36
        versionCode = appVersionCode
        versionName = appVersionName

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
        if (!prebuilt) {
            logger.lifecycle("llama.cpp: $llamaDir")
            externalNativeBuild {
                cmake {
                    // The native code is always an optimized build. A -O0 ggml is not measurable.
                    arguments += listOf(
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DLLAMA_CPP_DIR=$llamaDir",
                    )
                    // ./gradlew assembleRelease -Pprofiling=true records each OpenCL kernel
                    // and writes cl_profiling.csv into the app files directory on unload.
                    if (project.findProperty("profiling") == "true") {
                        arguments += "-DGGML_OPENCL_PROFILING=ON"
                    }
                }
            }
        }
    }

    if (!prebuilt) {
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/CMakeLists.txt")
                version = "4.1.2"
            }
        }
    }

    sourceSets {
        getByName("main") {
            if (prebuilt) {
                jniLibs.srcDirs("../snapdragon/jniLibs")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Debuggable, thus simpleperf and run-as reach the process. The native code stays -O3.
            isDebuggable = true
            // The debug key lets adb install the release build without a keystore.
            signingConfig = if (unsigned) null else signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        viewBinding = true
    }

    testOptions {
        // The unit tests run on the JVM: a stub of the Android SDK returns a default value, it does not throw.
        unitTests.isReturnDefaultValues = true
    }

    packaging {
        jniLibs {
            // The Hexagon backend hands the DSP library to FastRPC by file path, thus the
            // prebuilt package extracts its libraries. The DSP library is not an arm64
            // ELF, thus the packager must not strip it.
            useLegacyPackaging = prebuilt
            keepDebugSymbols += "**/libggml-htp-*.so"
        }
    }
}

// app/gradle.lockfile pins the resolved graph of every configuration. After a
// change of a dependency, write it again with:
//   ./gradlew :app:dependencies --write-locks -Pprebuilt=true
dependencyLocking {
    lockAllConfigurations()
}

dependencies {
    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("androidx.activity:activity-ktx:1.13.0")
    implementation("androidx.fragment:fragment-ktx:1.8.9")
    implementation("androidx.recyclerview:recyclerview:1.4.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.11.0")
    implementation("com.google.android.material:material:1.13.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    // Markdown rendering of the answers.
    implementation("io.noties.markwon:core:4.6.2")
    implementation("io.noties.markwon:ext-strikethrough:4.6.2")
    implementation("io.noties.markwon:ext-tables:4.6.2")
    implementation("io.noties.markwon:html:4.6.2")
    implementation("io.noties.markwon:linkify:4.6.2")
    testImplementation("junit:junit:4.13.2")
    // The conversation store writes JSON. The Android SDK stub of org.json has no implementation.
    testImplementation("org.json:json:20250107")
}
