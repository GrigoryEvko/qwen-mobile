cmake_minimum_required(VERSION 3.22)
project(qwenmobile-app-fuzz LANGUAGES C CXX)

# The shared flag functions of the areas (the tables of the sanitizers and of the profiles).
include("${CMAKE_CURRENT_LIST_DIR}/../../sanitizers/fuzz-common.cmake")

# The fuzzers of the native layer of the app: the JNI entry points of
# android/app/src/main/cpp/llama_jni.cpp with a fake Java VM, the two engine
# caches, and the draft length policy. tests/fuzz/app/run.sh configures this
# project. Refer to the usage text of run.sh.
#
#   FUZZ_PROFILE     debug or release. One profile and one sanitizer for each
#                    build directory build/fuzz/app-<profile>-<sanitizer>.
#                    debug: -O1 -g -fno-omit-frame-pointer, no NDEBUG, no LTO.
#                    release: the flags that ship (refer to the release block below).
#   FUZZ_SANITIZER   none, asan, ubsan, tsan, msan, or hwasan (Android only)
#   FUZZ_MSAN_PREFIX The install directory of an MSan-instrumented libc++, for msan
#   FUZZ_LIBFUZZER   ON: the fuzzers link libFuzzer. The driver never does.
#   LLAMA_CPP_DIR    A private snapshot of the llama.cpp tree with the patch series
#   FUZZ_APP_MODEL_DIR  The default directory of the tiny models
#   FUZZ_JNI_INCLUDE A directory that holds jni.h, for a host build
#   FUZZ_APP_CPP_DIR The native sources of the app (the default is android/app/src/main/cpp)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FUZZ_PROFILE "debug" CACHE STRING "debug or release")
set(FUZZ_SANITIZER "asan" CACHE STRING "none, asan, ubsan, tsan, msan, or hwasan (Android only)")
if(NOT FUZZ_PROFILE MATCHES "^(debug|release)$")
    message(FATAL_ERROR "FUZZ_PROFILE must be debug or release, not '${FUZZ_PROFILE}'")
endif()
option(FUZZ_LIBFUZZER "Link the fuzzers with libFuzzer" ON)
set(LLAMA_CPP_DIR "" CACHE PATH "The llama.cpp tree")
set(FUZZ_APP_MODEL_DIR "" CACHE PATH "The default directory of the tiny models")
set(FUZZ_JNI_INCLUDE "" CACHE PATH "A directory with jni.h for a host build")

if(NOT LLAMA_CPP_DIR OR NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "LLAMA_CPP_DIR is not a llama.cpp tree: '${LLAMA_CPP_DIR}'. Use a private snapshot, not the submodule.")
endif()

# The native sources of the app. Another directory holds another version of them, for example a copy with a patch.
set(FUZZ_APP_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../android/app/src/main/cpp" CACHE PATH "The native sources of the app")
get_filename_component(APP_CPP_DIR "${FUZZ_APP_CPP_DIR}" ABSOLUTE)

# One sanitizer for each build. It applies to every target, llama.cpp
# included: ThreadSanitizer must see each access, AddressSanitizer each
# allocation, and MemorySanitizer each store.
set(SAN_COMPILE "")
set(SAN_LINK "")
set(SAN_CXX "")
if(CMAKE_CXX_FLAGS MATCHES "-fsanitize=")
    # The shared initial cache (cmake -C tests/sanitizers/<value>.cmake,
    # which run.sh gives on the host) holds the flags of the configuration.
    message(STATUS "app fuzz: the sanitizer flags come from the initial cache: ${CMAKE_CXX_FLAGS}")
else()
    # The builds of the phone get no initial cache: the shared function of
    # tests/sanitizers/fuzz-common.cmake gives the flags of the configuration.
    if(NOT ANDROID AND FUZZ_SANITIZER STREQUAL "hwasan")
        message(FATAL_ERROR "hwasan is for the Android builds only")
    endif()
    fuzz_sanitizer_flags("${FUZZ_SANITIZER}" san_compile san_link san_cxx)
    separate_arguments(SAN_COMPILE UNIX_COMMAND "${san_compile}")
    separate_arguments(SAN_LINK UNIX_COMMAND "${san_link}")
    separate_arguments(SAN_CXX UNIX_COMMAND "${san_cxx}")
endif()
# The coverage of libFuzzer goes to the app, llama, common and mtmd. The
# kernels of ggml get none: their loops add cost and no useful coverage.
set(COV_FLAGS "")
if(FUZZ_LIBFUZZER)
    set(COV_FLAGS -fsanitize=fuzzer-no-link)
    if(ANDROID)
        # libFuzzer of the NDK (r29) defines no __sancov_lowest_stack, thus the link of the
        # stack-depth coverage fails with "undefined symbol: __sancov_lowest_stack".
        list(APPEND COV_FLAGS -fno-sanitize-coverage=stack-depth)
    endif()
endif()
add_compile_options(${SAN_COMPILE} "$<$<COMPILE_LANGUAGE:CXX>:${SAN_CXX}>")
add_link_options(${SAN_LINK})

# The flags of the llama.cpp libraries that the app ships
# (android/snapdragon/CMakeUserPresets.json, preset arm64-android-snapdragon):
#   -march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE
# with CMAKE_BUILD_TYPE=Release (-O3 -DNDEBUG). The x86 builds use GGML_NATIVE=ON
# in place of the -march of arm64 (msan keeps the SIMD off: MSan cannot follow the SIMD intrinsics).
# libqwenmobile.so (android/snapdragon/CMakeLists.txt, scripts/build-native.sh)
# builds the app sources with CMAKE_BUILD_TYPE=Release and no more flags than the
# NDK gives: -O3 -DNDEBUG, no -flto, no floating-point flags. The fuzz builds
# apply the flags of the profile to every target, as the shared initial caches
# on x86 do, thus the app sources here get them too.
if(DEFINED SANMATRIX_PROFILE)
    # The x86 builds of run.sh take the shared initial caches
    # (tests/sanitizers/profile-<profile>.cmake, then <sanitizer>.cmake). They
    # give the build type, the optimizer, the floating-point flags and LTO to
    # every target, and tests/sanitizers/check-rules.sh compares them with the
    # flags that ship. Thus this file adds no profile flag of its own.
    if(NOT SANMATRIX_PROFILE STREQUAL FUZZ_PROFILE)
        message(FATAL_ERROR "the initial cache is for the profile ${SANMATRIX_PROFILE}, FUZZ_PROFILE is ${FUZZ_PROFILE}")
    endif()
    message(STATUS "app fuzz: profile ${FUZZ_PROFILE} from the initial cache: ${CMAKE_CXX_FLAGS}")
else()
    # The builds of the phone get no initial cache: the shared functions of
    # tests/sanitizers/fuzz-common.cmake give the flags of the profile, and the
    # -march of the shipped build goes with them.
    fuzz_profile_build_type("${FUZZ_PROFILE}")
    fuzz_profile_flags("${FUZZ_PROFILE}" profile_flags profile_link)
    fuzz_profile_cxx_flags("${FUZZ_PROFILE}" "${FUZZ_SANITIZER}" profile_cxx)
    separate_arguments(PROFILE_FLAGS UNIX_COMMAND "${profile_flags}")
    separate_arguments(PROFILE_LINK UNIX_COMMAND "${profile_link}")
    separate_arguments(PROFILE_CXX UNIX_COMMAND "${profile_cxx}")
    if(ANDROID)
        list(APPEND PROFILE_FLAGS -march=armv8.7a+fp16+dotprod+i8mm)
    endif()
    add_compile_options(${PROFILE_FLAGS} "$<$<COMPILE_LANGUAGE:CXX>:${PROFILE_CXX}>")
    add_link_options(${PROFILE_LINK})
endif()

# llama.cpp with the CPU backend always, and the Hexagon backend on Android.
# OpenMP is off: ThreadSanitizer does not know its runtime.
# The host builds take BUILD_SHARED_LIBS from the profile (ON). The app ships
# llama, ggml, mtmd and llama-common as shared libraries, thus no inlining
# crosses the boundary of a library, and the full LTO of the release profile
# runs one time for each library, not one time for each executable. The phone
# builds have no initial cache and link the libraries static.
if(NOT DEFINED SANMATRIX_PROFILE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()
set(GGML_OPENMP           OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE        OFF CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI     OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD      ON  CACHE BOOL "" FORCE)
set(MTMD_VIDEO            OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL         OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS    OFF CACHE BOOL "" FORCE)
if(ANDROID)
    set(GGML_NATIVE  OFF CACHE BOOL "" FORCE)
    set(GGML_OPENCL  OFF CACHE BOOL "" FORCE)
    set(GGML_HEXAGON ON  CACHE BOOL "" FORCE)
endif()
if(FUZZ_SANITIZER STREQUAL "msan")
    # MemorySanitizer does not know the x86 intrinsics of the kernels: the scalar paths only.
    fuzz_scalar_x86()
endif()
set(FUZZ_LLAMA_PREBUILT_DIR "" CACHE PATH
    "The llama.cpp libraries that ship (android/snapdragon/jniLibs/arm64-v8a), for the release none build of the phone")
if(FUZZ_LLAMA_PREBUILT_DIR)
    # The release "none" run of the phone links the shipped libraries, thus it
    # tests the code of the APK. run.sh checks their hashes first.
    if(NOT (FUZZ_PROFILE STREQUAL "release" AND FUZZ_SANITIZER STREQUAL "none" AND ANDROID))
        message(FATAL_ERROR "FUZZ_LLAMA_PREBUILT_DIR is for the release none build of the phone only")
    endif()
    foreach(lib llama ggml ggml-base ggml-cpu mtmd llama-common)
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES
            IMPORTED_LOCATION "${FUZZ_LLAMA_PREBUILT_DIR}/lib${lib}.so"
            INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_CPP_DIR}/include;${LLAMA_CPP_DIR}/ggml/include")
    endforeach()
else()
    add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)
    # In the release profile the libraries are LTO bitcode. With the coverage
    # of libFuzzer on static libraries as well, the link of clang 22 fails
    # ("ld.lld: error: undefined symbol: __sancov_gen_.473", from a template
    # of stl_tree.h that the LTO objects and the other objects both hold).
    # Thus with static libraries in that profile, the coverage goes to the
    # app sources only. Each shared library links its own LTO objects.
    if(NOT FUZZ_PROFILE STREQUAL "release" OR BUILD_SHARED_LIBS)
        foreach(target llama llama-common mtmd)
            target_compile_options(${target} PRIVATE ${COV_FLAGS})
        endforeach()
    endif()
endif()

find_package(Threads REQUIRED)

# jni.h: the sysroot of the NDK on Android. On the host, a copy of the jni.h
# of the NDK (the same header as on the phone), else the one of a JDK.
set(JNI_INCLUDE "")
if(NOT ANDROID)
    if(FUZZ_JNI_INCLUDE)
        set(JNI_INCLUDE "${FUZZ_JNI_INCLUDE}")
    else()
        file(GLOB NDK_JNI "$ENV{ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/*/sysroot/usr/include/jni.h"
                          "$ENV{HOME}/Android/Sdk/ndk/*/toolchains/llvm/prebuilt/*/sysroot/usr/include/jni.h")
        if(NDK_JNI)
            list(GET NDK_JNI 0 NDK_JNI_H)
            configure_file("${NDK_JNI_H}" "${CMAKE_BINARY_DIR}/jni-include/jni.h" COPYONLY)
            set(JNI_INCLUDE "${CMAKE_BINARY_DIR}/jni-include")
        else()
            find_package(JNI REQUIRED)
            set(JNI_INCLUDE ${JNI_INCLUDE_DIRS})
        endif()
    endif()
endif()

set(APP_CACHE_SOURCES
    "${APP_CPP_DIR}/cache_io.cpp"
    "${APP_CPP_DIR}/image_cache.cpp"
    "${APP_CPP_DIR}/state_cache.cpp")

# The harness: the fake VM, the interpreter with llama_jni.cpp inside, and
# the other native files of the app as they are.
add_library(app_harness STATIC
    harness/app_harness.cpp
    harness/fake_jni.cpp
    ${APP_CACHE_SOURCES}
    "${APP_CPP_DIR}/chat_prompt.cpp"
    "${APP_CPP_DIR}/engine_tasks.cpp"
    "${APP_CPP_DIR}/perf_hint.cpp"
    "${APP_CPP_DIR}/spec_policy.cpp"
    "${APP_CPP_DIR}/token_rows.cpp"
    "${APP_CPP_DIR}/trace.cpp")
# FUZZ_SHARED_DIR (tests/sanitizers) holds fuzz_death.h, the shared TSan death callback of each
# libFuzzer target.
target_include_directories(app_harness PUBLIC
    harness
    "${FUZZ_SHARED_DIR}"
    "${APP_CPP_DIR}"
    "${LLAMA_CPP_DIR}/common"
    "${LLAMA_CPP_DIR}/tools/mtmd"
    ${JNI_INCLUDE})
if(NOT ANDROID)
    target_include_directories(app_harness PUBLIC harness/host)
endif()
target_compile_definitions(app_harness PUBLIC
    QWEN_NO_OPENCL_INFO
    FUZZ_APP_DEFAULT_MODEL_DIR="${FUZZ_APP_MODEL_DIR}")
target_compile_options(app_harness PUBLIC ${COV_FLAGS})
target_link_libraries(app_harness PUBLIC llama llama-common mtmd Threads::Threads ${CMAKE_DL_LIBS})
if(FUZZ_LLAMA_PREBUILT_DIR)
    # The imported shared libraries carry no link interface: the harness names ggml itself.
    target_link_libraries(app_harness PUBLIC ggml ggml-base ggml-cpu)
endif()
if(ANDROID)
    target_link_libraries(app_harness PUBLIC android log)
endif()

set(FUZZ_LINK "")
if(FUZZ_LIBFUZZER)
    set(FUZZ_LINK -fsanitize=fuzzer)
endif()

add_executable(fuzz_jni_api fuzz_jni_api.cpp)
target_link_libraries(fuzz_jni_api PRIVATE app_harness)
target_link_options(fuzz_jni_api PRIVATE ${FUZZ_LINK})

add_executable(fuzz_jni_threads fuzz_jni_threads.cpp)
target_link_libraries(fuzz_jni_threads PRIVATE app_harness)
target_link_options(fuzz_jni_threads PRIVATE ${FUZZ_LINK})

add_executable(app_fuzz_driver driver_main.cpp)
target_link_libraries(app_fuzz_driver PRIVATE app_harness)
if(FUZZ_LIBFUZZER)
    # The driver has its own main, but its objects call the coverage callbacks.
    # A sanitizer runtime defines them, and in the "none" configuration only the
    # libFuzzer runtime without main does.
    fuzz_fuzzer_no_main(FUZZER_NO_MAIN)
    target_link_libraries(app_fuzz_driver PRIVATE "${FUZZER_NO_MAIN}")
endif()

add_executable(fuzz_caches fuzz_caches.cpp ${APP_CACHE_SOURCES})
target_include_directories(fuzz_caches PRIVATE "${APP_CPP_DIR}" "${FUZZ_SHARED_DIR}")
target_compile_options(fuzz_caches PRIVATE ${COV_FLAGS})
target_link_libraries(fuzz_caches PRIVATE Threads::Threads)
target_link_options(fuzz_caches PRIVATE ${FUZZ_LINK})

add_executable(fuzz_spec_policy fuzz_spec_policy.cpp "${APP_CPP_DIR}/spec_policy.cpp")
target_include_directories(fuzz_spec_policy PRIVATE "${APP_CPP_DIR}" "${FUZZ_SHARED_DIR}")
target_compile_options(fuzz_spec_policy PRIVATE ${COV_FLAGS})
target_link_options(fuzz_spec_policy PRIVATE ${FUZZ_LINK})

if(NOT FUZZ_LIBFUZZER)
    # Without libFuzzer the two small fuzzers need a main: the standalone runner of libFuzzer inputs.
    target_sources(fuzz_caches PRIVATE standalone_main.cpp)
    target_sources(fuzz_spec_policy PRIVATE standalone_main.cpp)
    target_sources(fuzz_jni_api PRIVATE standalone_main.cpp)
    target_sources(fuzz_jni_threads PRIVATE standalone_main.cpp)
endif()
