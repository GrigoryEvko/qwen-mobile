# The op fuzzer of the Qwen3.5 path: a standalone project with three variants, and one profile and
# one sanitizer for each build directory.
#
#   FUZZ_OPS_VARIANT=fuzz     the libFuzzer harness fuzz_ops and the replay driver ops_replay, with the
#                             CPU backend of ggml for this host. Build directory:
#                             build/fuzz/ops-<FUZZ_PROFILE>-<FUZZ_SANITIZER>. The flags come from the
#                             shared initial caches, given with two -C options (run.sh does it):
#                             tests/sanitizers/profile-<FUZZ_PROFILE>.cmake, then
#                             tests/sanitizers/<FUZZ_SANITIZER>.cmake. This file adds only the
#                             libFuzzer instrumentation.
#   FUZZ_OPS_VARIANT=oracle   the reference tool ops_oracle: the scalar CPU backend with strict IEEE
#                             arithmetic, as build/oracle-x86 (GGML_NATIVE and every AVX, FMA and F16C
#                             option off, -ffp-contract=off -fno-fast-math, -O3 -DNDEBUG). Use gcc like it.
#                             It is the reference of each profile and each configuration, and it is
#                             not a configuration: build directory build/fuzz/ops/oracle.
#   FUZZ_OPS_VARIANT=android  the replay driver ops_replay for arm64 Android with the CPU and the
#                             Hexagon backends. Build it in the Snapdragon container (run.sh
#                             phone-build). Build directory: build/fuzz/ops-<profile>-<sanitizer>/android.
#                             The flags are below, because the shared files are for x86_64 Linux.
#
# FUZZ_PROFILE (a build directory has one profile and one sanitizer):
#   release  the shipped flags of android/snapdragon/CMakeUserPresets.json, preset
#            arm64-android-snapdragon, CMAKE_BUILD_TYPE=Release: -O3 -DNDEBUG
#            -march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto
#            -D_GNU_SOURCE, GGML_OPENMP=OFF, plus -g. With FUZZ_OPS_GGML_PREBUILT the replay driver
#            links the shipped libraries themselves (the release "none" phone run).
#   debug    the same flags with -O1 in place of -O3, no -flto, and no -DNDEBUG.
#
# FUZZ_SANITIZER selects one sanitizer (never two in one build):
#   none    no sanitizer
#   asan    -fsanitize=address
#   ubsan   -fsanitize=undefined -fno-sanitize-recover=undefined; -fsanitize-recover only for a check
#           with a known defect and a function-level entry in tests/sanitizers/ubsan.supp
#   tsan    -fsanitize=thread, with GGML_OPENMP=OFF (host only)
#   msan    -fsanitize=memory -fsanitize-memory-track-origins=2, with GGML_OPENMP=OFF, GGML_NATIVE=OFF,
#           the x86 SIMD options off and the MSan libc++ of FUZZ_MSAN_PREFIX (host only)
#   hwasan  -fsanitize=hwaddress (Android only; MSan does not exist on Android)
#
# FUZZ_OPS_GGML_SRC is a ggml source tree. run.sh gives a private snapshot of
# third_party/llama.cpp/ggml, because another session edits that tree.

cmake_minimum_required(VERSION 3.20)
project(fuzz_ops C CXX)

set(FUZZ_OPS_GGML_SRC "" CACHE PATH "The ggml source tree (a snapshot of third_party/llama.cpp/ggml)")
set(FUZZ_OPS_VARIANT "fuzz" CACHE STRING "fuzz, oracle or android")
set(FUZZ_SANITIZER "none" CACHE STRING "none, asan, ubsan, tsan, msan (host) or hwasan (Android)")
set_property(CACHE FUZZ_SANITIZER PROPERTY STRINGS none asan ubsan tsan msan hwasan)
set(FUZZ_PROFILE "debug" CACHE STRING "debug or release")
set_property(CACHE FUZZ_PROFILE PROPERTY STRINGS debug release)
set(FUZZ_OPS_GGML_PREBUILT "" CACHE PATH
    "Android only: a directory with the shipped libggml*.so. The replay driver links them in place of a ggml build")

if (NOT IS_DIRECTORY "${FUZZ_OPS_GGML_SRC}/src")
    message(FATAL_ERROR "fuzz_ops: set FUZZ_OPS_GGML_SRC to a ggml source tree (the directory with include/ and src/)")
endif()
if (NOT FUZZ_PROFILE MATCHES "^(debug|release)$")
    message(FATAL_ERROR "fuzz_ops: FUZZ_PROFILE must be debug or release, not '${FUZZ_PROFILE}'")
endif()

# The allowed sanitizers of each variant.
if (FUZZ_OPS_VARIANT STREQUAL "oracle")
    set(allowed none)
elseif (FUZZ_OPS_VARIANT STREQUAL "fuzz")
    set(allowed none asan ubsan tsan msan)
elseif (FUZZ_OPS_VARIANT STREQUAL "android")
    set(allowed none asan hwasan ubsan)
else()
    message(FATAL_ERROR "fuzz_ops: FUZZ_OPS_VARIANT must be fuzz, oracle or android, not '${FUZZ_OPS_VARIANT}'")
endif()
if (NOT FUZZ_SANITIZER IN_LIST allowed)
    message(FATAL_ERROR "fuzz_ops: FUZZ_SANITIZER=${FUZZ_SANITIZER} is not valid for the ${FUZZ_OPS_VARIANT} variant "
                        "(valid: ${allowed})")
endif()
if (FUZZ_OPS_GGML_PREBUILT AND NOT (FUZZ_OPS_VARIANT STREQUAL "android" AND FUZZ_PROFILE STREQUAL "release"
                                    AND FUZZ_SANITIZER STREQUAL "none"))
    message(FATAL_ERROR "fuzz_ops: FUZZ_OPS_GGML_PREBUILT is for the release none build of the android variant only: "
                        "the shipped libraries have no sanitizer")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL     OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE      OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# The harness makes the input values with float arithmetic, and the phone and the host must make
# the same bits. Thus its own sources never contract a multiply and an add, and never use fast math,
# also in a profile with -ffp-model=fast: the options below come after the profile flags.
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(FUZZ_OPS_HARNESS_FP -ffp-model=precise -ffp-contract=off -fno-fast-math)
else()
    set(FUZZ_OPS_HARNESS_FP -ffp-contract=off -fno-fast-math)
endif()

# Turn the x86 SIMD options of ggml off (the oracle and MSan).
macro(fuzz_ops_scalar_x86)
    foreach (opt GGML_NATIVE GGML_AVX GGML_AVX2 GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16
                 GGML_AVX_VNNI GGML_FMA GGML_F16C GGML_SSE42 GGML_BMI2 GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
        set(${opt} OFF CACHE BOOL "" FORCE)
    endforeach()
endmacro()

if (FUZZ_OPS_VARIANT STREQUAL "oracle")
    fuzz_ops_scalar_x86()
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -ffp-contract=off -fno-fast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffp-contract=off -fno-fast-math")
elseif (FUZZ_OPS_VARIANT STREQUAL "fuzz")
    # The profile and the sanitizer come from the shared initial caches.
    if (NOT DEFINED SANMATRIX_PROFILE)
        message(FATAL_ERROR "fuzz_ops: configure the host build with -C tests/sanitizers/profile-<debug|release>.cmake "
                            "-C tests/sanitizers/<config>.cmake")
    endif()
    set(expect_flag "")
    if (FUZZ_SANITIZER STREQUAL "asan")
        set(expect_flag "-fsanitize=address")
    elseif (FUZZ_SANITIZER STREQUAL "ubsan")
        set(expect_flag "-fsanitize=undefined")
    elseif (FUZZ_SANITIZER STREQUAL "tsan")
        set(expect_flag "-fsanitize=thread")
    elseif (FUZZ_SANITIZER STREQUAL "msan")
        set(expect_flag "-fsanitize=memory")
    endif()
    if (expect_flag AND NOT CMAKE_CXX_FLAGS MATCHES "${expect_flag}")
        message(FATAL_ERROR "fuzz_ops: CMAKE_CXX_FLAGS has no ${expect_flag}: give -C tests/sanitizers/${FUZZ_SANITIZER}.cmake")
    endif()
    # libFuzzer instrumentation (not a sanitizer). The comparison hooks (trace-cmp) in the
    # loops of the kernels and of the bound rules make each execution much slower, thus they are
    # off. The edge counters stay on.
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -fsanitize=fuzzer-no-link -fno-sanitize-coverage=trace-cmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer-no-link -fno-sanitize-coverage=trace-cmp")
else()
    # The sanitizer of the Android builds. No UBSan check recovers: each report stops the run.
    # The ubsan build links the UBSan runtime statically (-static-libsan). With the shared runtime,
    # two modules hold a copy of the libc++abi type-info classes: ops_replay (the static NDK libc++)
    # and libclang_rt.ubsan_standalone-aarch64-android.so (a private copy). The vptr check walks the
    # class hierarchy with a dynamic_cast to the copy of the runtime, which fails for the type_info
    # objects of ops_replay, thus each check of a derived class gives a false report (for example
    # 'ggml::cpu::repack::extra_buffer_type' for 'ggml::cpu::extra_buffer_type'). With the static
    # runtime, the references resolve to one copy, and vptr stays on without a recovery.
    set(android_san "")
    set(android_link "")
    if (FUZZ_SANITIZER STREQUAL "asan")
        set(android_san "-fsanitize=address")
    elseif (FUZZ_SANITIZER STREQUAL "ubsan")
        # No check recovers: refer to the text above and to tests/sanitizers/ubsan.cmake.
        set(android_san "-fsanitize=undefined -fno-sanitize-recover=undefined")
        set(android_link "-static-libsan")
    elseif (FUZZ_SANITIZER STREQUAL "hwasan")
        set(android_san "-fsanitize=hwaddress")
    endif()
    # The shipped flags of the preset arm64-android-snapdragon. Keep them equal.
    set(shipped "-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE")
    if (FUZZ_PROFILE STREQUAL "release")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
        set(CMAKE_C_FLAGS_RELEASE   "-O3 -DNDEBUG" CACHE STRING "" FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
        set(pflags "${shipped} -g")
        set(plink "-flto")
    else()
        # the release flags less -flto, with -O1 and without -DNDEBUG
        string(REPLACE " -flto" "" pflags "${shipped}")
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
        set(CMAKE_C_FLAGS_DEBUG   "-O1 -g" CACHE STRING "" FORCE)
        set(CMAKE_CXX_FLAGS_DEBUG "-O1 -g" CACHE STRING "" FORCE)
        set(plink "")
    endif()
    set(GGML_NATIVE  OFF CACHE BOOL "" FORCE)
    set(GGML_HEXAGON ON  CACHE BOOL "" FORCE)
    set(GGML_OPENCL  OFF CACHE BOOL "" FORCE)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${pflags} ${android_san}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${pflags} ${android_san}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${plink} ${android_san} ${android_link}")
endif()

if (FUZZ_OPS_GGML_PREBUILT)
    # The shipped libraries: the same source tree (the snapshot gives the headers), thus the C API
    # and the backend structures of ggml-backend-impl.h agree. The harness uses only the C API.
    foreach (lib ggml ggml-base ggml-cpu)
        if (NOT EXISTS "${FUZZ_OPS_GGML_PREBUILT}/lib${lib}.so")
            message(FATAL_ERROR "fuzz_ops: ${FUZZ_OPS_GGML_PREBUILT}/lib${lib}.so is missing")
        endif()
        add_library(${lib} SHARED IMPORTED)
        set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${FUZZ_OPS_GGML_PREBUILT}/lib${lib}.so")
    endforeach()
else()
    add_subdirectory(${FUZZ_OPS_GGML_SRC} ggml)
endif()

add_library(fuzz_ops_core STATIC
    src/common.cpp
    src/case.cpp
    src/kinds.cpp
    src/exec.cpp
    src/compare.cpp
    src/wire.cpp)
target_include_directories(fuzz_ops_core PUBLIC ${FUZZ_OPS_GGML_SRC}/include ${FUZZ_OPS_GGML_SRC}/src)
target_link_libraries(fuzz_ops_core PUBLIC ggml ggml-base ggml-cpu)
target_compile_options(fuzz_ops_core PRIVATE ${FUZZ_OPS_HARNESS_FP})
# The CPU backend under test uses fast math in both profiles (-ffp-model=fast), thus its bound rule
# of the comparison has the slack of a fast-math build (compare.h).
if (CMAKE_CXX_FLAGS MATCHES "ffp-model=fast" OR SANMATRIX_PROFILE_FLAGS MATCHES "ffp-model=fast")
    target_compile_definitions(fuzz_ops_core PUBLIC FUZZ_OPS_BACKEND_FAST=1)
endif()

# The replay driver: the phone program, and a host program for tests of the phone flow.
function(fuzz_ops_replay)
    add_executable(ops_replay src/replay_main.cpp src/crashdiag.cpp)
    target_link_libraries(ops_replay PRIVATE fuzz_ops_core)
    target_compile_options(ops_replay PRIVATE ${FUZZ_OPS_HARNESS_FP})
endfunction()

if (FUZZ_OPS_VARIANT STREQUAL "fuzz")
    add_executable(fuzz_ops src/fuzz_main.cpp)
    target_link_libraries(fuzz_ops PRIVATE fuzz_ops_core)
    target_compile_options(fuzz_ops PRIVATE ${FUZZ_OPS_HARNESS_FP})
    target_link_options(fuzz_ops PRIVATE -fsanitize=fuzzer)
    fuzz_ops_replay()
elseif (FUZZ_OPS_VARIANT STREQUAL "oracle")
    add_executable(ops_oracle src/oracle_main.cpp)
    target_link_libraries(ops_oracle PRIVATE fuzz_ops_core)
    target_compile_options(ops_oracle PRIVATE ${FUZZ_OPS_HARNESS_FP})
else()
    fuzz_ops_replay()
endif()
