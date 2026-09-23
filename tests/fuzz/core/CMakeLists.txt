# The fuzz harnesses of the llama.cpp core and of our patches to it.
#
# This project builds llama.cpp from FUZZ_LLAMA_DIR out of tree, with one
# profile and one sanitizer on every object, and links each harness with
# libFuzzer. tests/fuzz/core/run.sh configures it: one build directory for
# each pair, build/fuzz/core-<profile>-<sanitizer> on the host and
# build/fuzz/core-android-<profile>-<sanitizer> for the phone. A build never
# has two sanitizers (rule R1). Do not add this directory to another CMake
# project.
#
#   FUZZ_LLAMA_DIR   The llama.cpp source tree. The default is the submodule.
#                    The Android build gives a private copy of the submodule.
#   FUZZ_PROFILE     debug | release (rule R11).
#                      release  the flags of the shipped build (preset
#                               arm64-android-snapdragon of
#                               android/snapdragon/CMakeUserPresets.json):
#                               -O3 -DNDEBUG -flto -fvectorize -ffp-model=fast
#                               -fno-finite-math-only -D_GNU_SOURCE, plus -g
#                      debug    the same flags with -O1, no LTO, live asserts
#   FUZZ_SANITIZER   none | asan | ubsan | tsan | msan on the host, and none |
#                    asan | hwasan | ubsan | tsan on Android. The old values
#                    "address" and "thread" are aliases of asan and tsan, and
#                    give a deprecation message.
#   FUZZ_HEXAGON     ON to build the Hexagon backend (Android only).
#   FUZZ_TARGETS     The harnesses to build, separated by semicolons.
#
# Where the flags come from:
#   - A host build gets the profile and the sanitizer from the shared initial
#     cache files: cmake -C tests/sanitizers/profile-<profile>.cmake
#     -C tests/sanitizers/<sanitizer>.cmake (run.sh gives them). They set
#     CMAKE_<LANG>_FLAGS, the build type, FUZZ_PROFILE, FUZZ_SANITIZER,
#     GGML_OPENMP=OFF, GGML_NATIVE, and for msan the MSan libc++ of
#     FUZZ_MSAN_PREFIX and the x86 SIMD options OFF (rule R7).
#   - An Android build (the NDK toolchain) and a host build without those
#     files get the same profile and sanitizer flags from this file, with the
#     -march of the shipped build from the caller (run.sh).
# This file adds only the coverage of libFuzzer (-fsanitize=fuzzer-no-link on
# every object, -fsanitize=fuzzer on the harness executables), which goes with
# the one sanitizer of the build (rule R1).

cmake_minimum_required(VERSION 3.22)
project(qwen_mobile_fuzz_core C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FUZZ_LLAMA_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../third_party/llama.cpp" CACHE PATH "The llama.cpp source tree")
set(FUZZ_PROFILE "debug" CACHE STRING "debug or release")
set(FUZZ_SANITIZER "asan" CACHE STRING "none, asan, ubsan, tsan, msan (host), or hwasan (Android)")
option(FUZZ_HEXAGON "Build the Hexagon backend (Android only)" OFF)
set(FUZZ_TARGETS
    "fuzz_gguf;fuzz_model_load;fuzz_tokenizer;fuzz_chat;fuzz_sampler;fuzz_recurrent;fuzz_npu_decode;tsan_threadpool;tsan_decode;tsan_topset;make_tiny_model"
    CACHE STRING "The harnesses to build")

if (FUZZ_SANITIZER STREQUAL "address")
    message(DEPRECATION "FUZZ_SANITIZER=address is an old name. It gives asan (ASan only). Use FUZZ_SANITIZER=asan.")
    set(FUZZ_SANITIZER "asan" CACHE STRING "none, asan, ubsan, tsan, msan (host), or hwasan (Android)" FORCE)
elseif (FUZZ_SANITIZER STREQUAL "thread")
    message(DEPRECATION "FUZZ_SANITIZER=thread is an old name. It gives tsan. Use FUZZ_SANITIZER=tsan.")
    set(FUZZ_SANITIZER "tsan" CACHE STRING "none, asan, ubsan, tsan, msan (host), or hwasan (Android)" FORCE)
endif()
if (NOT FUZZ_PROFILE MATCHES "^(debug|release)$")
    message(FATAL_ERROR "FUZZ_PROFILE=${FUZZ_PROFILE} is not known. Use debug or release.")
endif()

get_filename_component(FUZZ_LLAMA_DIR "${FUZZ_LLAMA_DIR}" ABSOLUTE)
if (NOT EXISTS "${FUZZ_LLAMA_DIR}/include/llama.h")
    message(FATAL_ERROR "FUZZ_LLAMA_DIR=${FUZZ_LLAMA_DIR} holds no include/llama.h. Give the path of a llama.cpp tree.")
endif()

if (DEFINED SANMATRIX_PROFILE)
    # A host build with the shared initial cache files: they gave every flag.
    if (NOT FUZZ_SANITIZER MATCHES "^(none|asan|ubsan|tsan|msan)$")
        message(FATAL_ERROR "FUZZ_SANITIZER=${FUZZ_SANITIZER} is not a host configuration. Use none, asan, ubsan, tsan or msan.")
    endif()
    if (NOT FUZZ_PROFILE STREQUAL SANMATRIX_PROFILE)
        message(FATAL_ERROR "FUZZ_PROFILE=${FUZZ_PROFILE} differs from the profile file (${SANMATRIX_PROFILE}).")
    endif()
else()
    # An Android build, or a host build without the shared files: the same rules here.
    set(fuzz_prof "-g -fvectorize -ffp-model=fast -fno-finite-math-only -D_GNU_SOURCE")
    set(fuzz_link "")
    if (FUZZ_PROFILE STREQUAL "release")
        # the flags of the preset arm64-android-snapdragon with CMAKE_BUILD_TYPE=Release
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
        set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
        string(APPEND fuzz_prof " -flto")
        set(fuzz_link "-flto")
        if (NOT ANDROID)
            string(APPEND fuzz_link " -fuse-ld=lld")  # the x86 substitution for the NDK linker, necessary for -flto
        endif()
    else()
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
        set(CMAKE_C_FLAGS_DEBUG "-O1 -g" CACHE STRING "" FORCE)
        set(CMAKE_CXX_FLAGS_DEBUG "-O1 -g" CACHE STRING "" FORCE)
        string(APPEND fuzz_prof " -fno-omit-frame-pointer")
    endif()

    set(fuzz_san "")
    set(fuzz_san_link "")
    if (FUZZ_SANITIZER STREQUAL "none")
    elseif (FUZZ_SANITIZER STREQUAL "asan")
        set(fuzz_san "-fsanitize=address -fno-optimize-sibling-calls")
        set(fuzz_san_link "-fsanitize=address")
    elseif (FUZZ_SANITIZER STREQUAL "ubsan")
        # Rule R5: the first report stops the run (the flags of tests/sanitizers/ubsan.cmake).
        set(fuzz_san "-fsanitize=undefined -fno-sanitize-recover=undefined")
        set(fuzz_san_link "-fsanitize=undefined")
    elseif (FUZZ_SANITIZER STREQUAL "tsan")
        set(fuzz_san "-fsanitize=thread")
        set(fuzz_san_link "-fsanitize=thread")
    elseif (FUZZ_SANITIZER STREQUAL "hwasan" AND ANDROID)
        set(fuzz_san "-fsanitize=hwaddress")
        set(fuzz_san_link "-fsanitize=hwaddress")
    else()
        message(FATAL_ERROR "FUZZ_SANITIZER=${FUZZ_SANITIZER} is not known for this build. Android: none, asan, hwasan, "
                            "ubsan, tsan. Host: give -C tests/sanitizers/profile-<profile>.cmake -C tests/sanitizers/<sanitizer>.cmake "
                            "(msan needs the MSan libc++ of those files).")
    endif()
    string(APPEND CMAKE_C_FLAGS   " ${fuzz_prof} ${fuzz_san}")
    string(APPEND CMAKE_CXX_FLAGS " ${fuzz_prof} ${fuzz_san}")
    if (FUZZ_PROFILE STREQUAL "debug")
        # the library asserts of the debug profile: libc++ (NDK) or libstdc++ (host)
        if (ANDROID)
            string(APPEND CMAKE_CXX_FLAGS " -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE")
        else()
            string(APPEND CMAKE_CXX_FLAGS " -D_GLIBCXX_ASSERTIONS")
        endif()
    endif()
    string(APPEND CMAKE_EXE_LINKER_FLAGS " ${fuzz_link} ${fuzz_san_link}")
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
endif()

# The coverage of libFuzzer. -fsanitize=fuzzer-no-link adds stack-depth, and the
# Android runtimes define its symbol __sancov_lowest_stack only in libFuzzer,
# thus a plain program (a CMake check) does not link. The Android build names
# the coverage kinds without stack-depth.
if (ANDROID)
    set(FUZZ_COV_FLAGS "-fsanitize-coverage=inline-8bit-counters,pc-table,trace-cmp,indirect-calls")
else()
    set(FUZZ_COV_FLAGS "-fsanitize=fuzzer-no-link")
endif()
string(APPEND CMAKE_C_FLAGS   " ${FUZZ_COV_FLAGS}")
string(APPEND CMAKE_CXX_FLAGS " ${FUZZ_COV_FLAGS}")
message(STATUS "fuzz core: profile ${FUZZ_PROFILE}, sanitizer ${FUZZ_SANITIZER}, type ${CMAKE_BUILD_TYPE}")

# llama.cpp as a static library set, with the CPU backend. The OpenMP runtime
# is not instrumented, thus TSan and MSan cannot see its barriers. GGML_OPENMP=OFF
# gives the ggml thread pool, which is the pool of the app (the Android preset
# sets the same value).
set(BUILD_SHARED_LIBS       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON      ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP         OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL           OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS        OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_IS_DEV      OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP             OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE          OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE             ON  CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL         OFF CACHE BOOL "" FORCE)
if (ANDROID)
    set(GGML_NATIVE         OFF CACHE BOOL "" FORCE)
    set(GGML_OPENCL         OFF CACHE BOOL "" FORCE)
    set(GGML_HEXAGON        ${FUZZ_HEXAGON} CACHE BOOL "" FORCE)
endif()

add_subdirectory("${FUZZ_LLAMA_DIR}" llama EXCLUDE_FROM_ALL)

# Add one harness. A name that starts with "fuzz_" or "tsan_" links libFuzzer.
# The other names are plain tools. All the harnesses can read the internal
# headers of llama.cpp: the tiny model generator needs llama-model-saver.h.
function(fuzz_add name)
    cmake_parse_arguments(FA "" "" "LIBS" ${ARGN})
    if (NOT name IN_LIST FUZZ_TARGETS)
        return()
    endif()
    add_executable(${name} ${name}.cpp fuzz_common.h)
    target_include_directories(${name} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}/../../sanitizers"
        "${FUZZ_LLAMA_DIR}/include"
        "${FUZZ_LLAMA_DIR}/src"
        "${FUZZ_LLAMA_DIR}/common"
        "${FUZZ_LLAMA_DIR}/ggml/src")
    target_link_libraries(${name} PRIVATE ${FA_LIBS})
    if (name MATCHES "^(fuzz|tsan)_")
        target_link_options(${name} PRIVATE -fsanitize=fuzzer)
    endif()
endfunction()

# fuzz_common.h calls llama_log_set, thus each harness links llama.
fuzz_add(fuzz_gguf        LIBS llama ggml)
fuzz_add(fuzz_model_load  LIBS llama)
fuzz_add(fuzz_tokenizer   LIBS llama)
fuzz_add(fuzz_chat        LIBS llama-common llama)
fuzz_add(fuzz_sampler     LIBS llama)
fuzz_add(fuzz_recurrent   LIBS llama)
fuzz_add(fuzz_npu_decode  LIBS llama)
fuzz_add(tsan_threadpool  LIBS llama ggml)
fuzz_add(tsan_decode      LIBS llama)
fuzz_add(tsan_topset      LIBS llama-common llama)
fuzz_add(make_tiny_model  LIBS llama)
