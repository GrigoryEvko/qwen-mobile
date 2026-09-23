cmake_minimum_required(VERSION 3.22)
project(qwenmobile-app-fuzz LANGUAGES C CXX)

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
set(FUZZ_MSAN_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/fuzz/msan-libcxx/install" CACHE PATH
    "The install directory of an MSan-instrumented libc++")
option(FUZZ_LIBFUZZER "Link the fuzzers with libFuzzer" ON)
set(LLAMA_CPP_DIR "" CACHE PATH "The llama.cpp tree")
set(FUZZ_APP_MODEL_DIR "" CACHE PATH "The default directory of the tiny models")
set(FUZZ_JNI_INCLUDE "" CACHE PATH "A directory with jni.h for a host build")

if(NOT LLAMA_CPP_DIR OR NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "LLAMA_CPP_DIR is not a llama.cpp tree: '${LLAMA_CPP_DIR}'. Use a private snapshot, not the submodule.")
endif()

# The native sources of the app. A different directory (a copy with a fix patch applied) tests the fix.
set(FUZZ_APP_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../android/app/src/main/cpp" CACHE PATH "The native sources of the app")
get_filename_component(APP_CPP_DIR "${FUZZ_APP_CPP_DIR}" ABSOLUTE)

# One sanitizer for each build. It applies to every target, llama.cpp
# included: ThreadSanitizer must see each access, AddressSanitizer each
# allocation, and MemorySanitizer each store.
set(SAN_COMPILE "")
set(SAN_LINK "")
if(CMAKE_CXX_FLAGS MATCHES "-fsanitize=")
    # The initial cache of the campaign (cmake -C tests/sanitizers/<value>.cmake,
    # which run.sh gives on the host) holds the flags of the configuration.
    message(STATUS "app fuzz: the sanitizer flags come from the initial cache: ${CMAKE_CXX_FLAGS}")
elseif(FUZZ_SANITIZER STREQUAL "asan")
    set(SAN_COMPILE -fsanitize=address)
elseif(FUZZ_SANITIZER STREQUAL "ubsan")
    # The flags of tests/sanitizers/ubsan.cmake, for the builds without that
    # initial cache (the phone): every report stops the run.
    set(SAN_COMPILE -fsanitize=undefined -fno-sanitize-recover=undefined)
elseif(FUZZ_SANITIZER STREQUAL "tsan")
    set(SAN_COMPILE -fsanitize=thread)
elseif(FUZZ_SANITIZER STREQUAL "msan")
    if(NOT EXISTS "${FUZZ_MSAN_PREFIX}/include/c++/v1/vector")
        message(FATAL_ERROR "msan needs an MSan-instrumented libc++ in FUZZ_MSAN_PREFIX='${FUZZ_MSAN_PREFIX}' "
                            "(include/c++/v1 and lib). The sanitizer-matrix agent builds it into "
                            "build/fuzz/msan-libcxx/install. That directory does not exist yet.")
    endif()
    set(SAN_COMPILE -fsanitize=memory -fsanitize-memory-track-origins=2 -stdlib=libc++ -nostdinc++
                    -isystem ${FUZZ_MSAN_PREFIX}/include/c++/v1)
    set(SAN_LINK -stdlib=libc++ -L${FUZZ_MSAN_PREFIX}/lib -Wl,-rpath,${FUZZ_MSAN_PREFIX}/lib -lc++ -lc++abi)
elseif(FUZZ_SANITIZER STREQUAL "hwasan")
    if(NOT ANDROID)
        message(FATAL_ERROR "hwasan is for the Android builds only")
    endif()
    set(SAN_COMPILE -fsanitize=hwaddress)
elseif(NOT FUZZ_SANITIZER STREQUAL "none")
    message(FATAL_ERROR "FUZZ_SANITIZER must be none, asan, ubsan, tsan, msan or hwasan, not '${FUZZ_SANITIZER}'")
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
add_compile_options(${SAN_COMPILE})
add_link_options(${SAN_COMPILE} ${SAN_LINK})

# The flags of the llama.cpp libraries that the app ships
# (android/snapdragon/CMakeUserPresets.json, preset arm64-android-snapdragon):
#   -march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE
# with CMAKE_BUILD_TYPE=Release (-O3 -DNDEBUG). The x86 builds use GGML_NATIVE=ON
# in place of the -march of arm64 (msan keeps the SIMD off under rule R7).
# libqwenmobile.so (android/snapdragon/CMakeLists.txt, scripts/build-native.sh)
# builds the app sources with CMAKE_BUILD_TYPE=Release and no more flags than the
# NDK gives: -O3 -DNDEBUG, no -flto, no floating-point flags. The campaign
# applies the flags of the profile to every target (rule R12, the same as the
# shared initial caches on x86), thus the app sources here get them too.
set(LLAMA_FP_FLAGS -fvectorize -ffp-model=fast -fno-finite-math-only -D_GNU_SOURCE)
if(ANDROID)
    list(APPEND LLAMA_FP_FLAGS -march=armv8.7a+fp16+dotprod+i8mm)
endif()
if(DEFINED SANMATRIX_PROFILE)
    # The x86 builds of run.sh take the shared initial caches of the campaign
    # (tests/sanitizers/profile-<profile>.cmake, then <sanitizer>.cmake). They
    # give the build type, the optimizer, the floating-point flags and LTO to
    # every target, and tests/sanitizers/check-rules.sh compares them with the
    # flags that ship. Thus this file adds no profile flag of its own.
    if(NOT SANMATRIX_PROFILE STREQUAL FUZZ_PROFILE)
        message(FATAL_ERROR "the initial cache is for the profile ${SANMATRIX_PROFILE}, FUZZ_PROFILE is ${FUZZ_PROFILE}")
    endif()
    set(LLAMA_FP_FLAGS "")
    message(STATUS "app fuzz: profile ${FUZZ_PROFILE} from the initial cache: ${CMAKE_CXX_FLAGS}")
elseif(FUZZ_PROFILE STREQUAL "release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
    add_compile_options(-g)
    list(APPEND LLAMA_FP_FLAGS -flto)
    add_compile_options(${LLAMA_FP_FLAGS})
    # clang LTO needs a linker that reads its bitcode.
    add_link_options(-flto -fuse-ld=lld)
else()
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O1 -g" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O1 -g" CACHE STRING "" FORCE)
    add_compile_options(-g -O1 -fno-omit-frame-pointer ${LLAMA_FP_FLAGS})
endif()

# llama.cpp as static libraries, CPU backend always, Hexagon backend on
# Android. OpenMP is off: ThreadSanitizer does not know its runtime.
set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
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
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(opt SSE42 AVX AVX_VNNI AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 FMA F16C AMX_TILE AMX_INT8 AMX_BF16)
        set(GGML_${opt} OFF CACHE BOOL "" FORCE)
    endforeach()
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
    # of libFuzzer on them as well, the link of clang 22 fails ("ld.lld: error:
    # undefined symbol: __sancov_gen_.473", from a template of stl_tree.h that
    # the LTO objects and the other objects both hold). Thus in that profile
    # the coverage goes to the app sources only.
    if(NOT FUZZ_PROFILE STREQUAL "release")
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
    "${APP_CPP_DIR}/perf_hint.cpp"
    "${APP_CPP_DIR}/spec_policy.cpp"
    "${APP_CPP_DIR}/trace.cpp")
# tests/sanitizers holds fuzz_death.h, the shared TSan death callback of each libFuzzer target.
set(FUZZ_SHARED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../sanitizers")
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
    # libFuzzer runtime without main does. The name of the file depends on the
    # layout of the runtime directory of clang.
    if(ANDROID)
        set(_rt_arch aarch64-android)
    else()
        set(_rt_arch ${CMAKE_SYSTEM_PROCESSOR})
    endif()
    set(FUZZER_NO_MAIN "")
    set(_target_flag "")
    if(CMAKE_CXX_COMPILER_TARGET)
        set(_target_flag --target=${CMAKE_CXX_COMPILER_TARGET})
    endif()
    foreach(name libclang_rt.fuzzer_no_main.a libclang_rt.fuzzer_no_main-${_rt_arch}.a)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${_target_flag} --print-file-name=${name}
                        OUTPUT_VARIABLE path OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT FUZZER_NO_MAIN AND IS_ABSOLUTE "${path}" AND EXISTS "${path}")
            set(FUZZER_NO_MAIN "${path}")
        endif()
    endforeach()
    if(NOT FUZZER_NO_MAIN)
        message(FATAL_ERROR "no libclang_rt.fuzzer_no_main of ${CMAKE_CXX_COMPILER}: the driver cannot link")
    endif()
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
