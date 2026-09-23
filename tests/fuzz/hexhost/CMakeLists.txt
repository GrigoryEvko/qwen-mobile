# The x86 fuzz harness of the host part of the Hexagon backend (hexhost).
#
# This project builds the ggml base library, the host source of the Hexagon
# backend (ggml-hexagon.cpp) against the stub SDK headers of stubs/, the fake
# DSP of common/, and one libFuzzer target for each fuzz_*.cpp file.
#
#   cmake -S tests/fuzz/hexhost -B build/fuzz/hexhost-debug-asan -G Ninja \
#         -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=None \
#         -DFUZZ_PROFILE=debug -DFUZZ_SANITIZER=asan -DHEXHOST_LLAMA_DIR=<llama.cpp tree>
#
# One build directory for each pair: build/fuzz/hexhost-<profile>-<sanitizer>.
#
# FUZZ_PROFILE:
#   debug    -O1 -g -fno-omit-frame-pointer, assert() active, no LTO, the fp flags of the app
#   release  the flags of the shipped library (android/snapdragon/CMakeUserPresets.json,
#            preset arm64-android-snapdragon): -O3 -DNDEBUG -flto -fvectorize -ffp-model=fast
#            -fno-finite-math-only -D_GNU_SOURCE, plus -g. The x86 substitution: -march=native
#            in place of -march=armv8.7a+fp16+dotprod+i8mm (not for msan, rule R7).
#
# FUZZ_SANITIZER (the flags are the flags of tests/sanitizers/<sanitizer>.cmake):
#   none   no sanitizer (the speed and coverage baseline)
#   asan   AddressSanitizer only
#   ubsan  UndefinedBehaviorSanitizer only. The first report stops the run.
#   tsan   ThreadSanitizer only (each dspqueue has a DSP thread)
#   msan   MemorySanitizer only, with the MSan libc++ of FUZZ_MSAN_PREFIX
#
# The debug profile adds -D_GLIBCXX_ASSERTIONS (libstdc++) or the extensive hardening of
# libc++ (msan), as tests/sanitizers/<sanitizer>.cmake does. The release profile does not,
# because the shipped library does not.
#
# Use tests/fuzz/hexhost/run.sh to configure, build and run.

cmake_minimum_required(VERSION 3.20)
project(hexhost_fuzz C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEXHOST_LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp" CACHE PATH
    "The llama.cpp tree with the patch series applied")
set(FUZZ_SANITIZER "asan" CACHE STRING "none, asan, ubsan, tsan or msan")
set_property(CACHE FUZZ_SANITIZER PROPERTY STRINGS none asan ubsan tsan msan)
set(FUZZ_PROFILE "debug" CACHE STRING "debug or release")
set_property(CACHE FUZZ_PROFILE PROPERTY STRINGS debug release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(FUZZ_MSAN_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/fuzz/msan-libcxx/install" CACHE PATH
    "The install prefix of the MSan-instrumented libc++")

# HEXHOST_SANITIZER is the old name of FUZZ_SANITIZER
if (DEFINED HEXHOST_SANITIZER)
    message(DEPRECATION "hexhost: HEXHOST_SANITIZER is deprecated, use FUZZ_SANITIZER. "
                        "FUZZ_SANITIZER is set to ${HEXHOST_SANITIZER}.")
    set(FUZZ_SANITIZER "${HEXHOST_SANITIZER}" CACHE STRING "none, asan, ubsan, tsan or msan" FORCE)
    unset(HEXHOST_SANITIZER CACHE)
endif()

if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "hexhost: libFuzzer needs clang, the compiler is ${CMAKE_CXX_COMPILER_ID}")
endif()

set(G   "${HEXHOST_LLAMA_DIR}/ggml")
set(HEX "${G}/src/ggml-hexagon")
if (NOT EXISTS "${HEX}/ggml-hexagon.cpp")
    message(FATAL_ERROR "hexhost: ${HEX}/ggml-hexagon.cpp does not exist; set HEXHOST_LLAMA_DIR to a llama.cpp tree")
endif()

# This project builds no ggml-cpu and no target uses OpenMP (GGML_OPENMP=OFF in effect, rule
# R7). GGML_NATIVE has its effect through -march=native of the release profile.
set(SAN_LINK_FLAGS "")
set(SAN_CXX_FLAGS -D_GLIBCXX_ASSERTIONS)
if (FUZZ_SANITIZER STREQUAL "asan")
    set(SAN_FLAGS -fsanitize=address -fno-optimize-sibling-calls)
    add_compile_definitions(HEXHOST_ASYNC_DSP=0)
elseif (FUZZ_SANITIZER STREQUAL "ubsan")
    # No check recovers, thus each UBSan report stops the run. harness.cpp gives the shared file
    # tests/sanitizers/ubsan.supp to the runtime: an entry there has an effect only for a check
    # that tests/sanitizers/ubsan.cmake makes recover.
    set(SAN_FLAGS -fsanitize=undefined -fno-sanitize-recover=undefined)
    set(HEXHOST_UBSAN_SUPP "${CMAKE_CURRENT_SOURCE_DIR}/../../sanitizers/ubsan.supp")
    get_filename_component(HEXHOST_UBSAN_SUPP "${HEXHOST_UBSAN_SUPP}" ABSOLUTE)
    add_compile_definitions(HEXHOST_ASYNC_DSP=0 HEXHOST_UBSAN_SUPP="${HEXHOST_UBSAN_SUPP}")
    if (NOT EXISTS "${HEXHOST_UBSAN_SUPP}")
        message(WARNING "hexhost: ${HEXHOST_UBSAN_SUPP} does not exist. The UBSan runtime then gets no "
                        "suppression file.")
    endif()
elseif (FUZZ_SANITIZER STREQUAL "tsan")
    set(SAN_FLAGS -fsanitize=thread)
    add_compile_definitions(HEXHOST_ASYNC_DSP=1)
elseif (FUZZ_SANITIZER STREQUAL "msan")
    if (NOT EXISTS "${FUZZ_MSAN_PREFIX}/include/c++/v1/vector" OR NOT EXISTS "${FUZZ_MSAN_PREFIX}/lib/libc++.so")
        message(FATAL_ERROR "hexhost: msan needs an MSan-instrumented libc++ in FUZZ_MSAN_PREFIX "
                            "(${FUZZ_MSAN_PREFIX}). tests/sanitizers/build-msan-libcxx.sh builds it into "
                            "build/fuzz/msan-libcxx/install. Build it first, or set FUZZ_MSAN_PREFIX.")
    endif()
    set(SAN_FLAGS -fsanitize=memory -fsanitize-memory-track-origins=2)
    set(SAN_CXX_FLAGS -stdlib=libc++ -nostdinc++ "SHELL:-isystem ${FUZZ_MSAN_PREFIX}/include/c++/v1"
                      -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE)
    add_compile_definitions(HEXHOST_ASYNC_DSP=0)
    set(SAN_LINK_FLAGS -stdlib=libc++ -L${FUZZ_MSAN_PREFIX}/lib -Wl,-rpath,${FUZZ_MSAN_PREFIX}/lib -lc++ -lc++abi)
elseif (FUZZ_SANITIZER STREQUAL "none")
    set(SAN_FLAGS "")
    add_compile_definitions(HEXHOST_ASYNC_DSP=0)
else()
    message(FATAL_ERROR "hexhost: FUZZ_SANITIZER must be none, asan, ubsan, tsan or msan, it is ${FUZZ_SANITIZER}")
endif()

# The fp flags of the app (preset arm64-android-snapdragon) in the two profiles
set(APP_FP_FLAGS -fvectorize -ffp-model=fast -fno-finite-math-only -D_GNU_SOURCE)
if (FUZZ_PROFILE STREQUAL "debug")
    set(PROFILE_FLAGS -O1 -g -fno-omit-frame-pointer ${APP_FP_FLAGS})
    set(PROFILE_LINK_FLAGS "")
elseif (FUZZ_PROFILE STREQUAL "release")
    # -fno-omit-frame-pointer: the x86 substitution of the arm64 frame pointer (profile-release.cmake)
    set(PROFILE_FLAGS -O3 -DNDEBUG -flto ${APP_FP_FLAGS} -g -fno-omit-frame-pointer)
    if (NOT FUZZ_SANITIZER STREQUAL "msan")
        list(APPEND PROFILE_FLAGS -march=native)
    endif()
    # The LTO objects need the LLVM archiver and linker
    set(PROFILE_LINK_FLAGS -flto -fuse-ld=lld)
    find_program(LLVM_AR NAMES llvm-ar REQUIRED)
    find_program(LLVM_RANLIB NAMES llvm-ranlib REQUIRED)
    set(CMAKE_AR "${LLVM_AR}")
    set(CMAKE_RANLIB "${LLVM_RANLIB}")
    # the release profile has no libstdc++ assertions, as the shipped library
    if (NOT FUZZ_SANITIZER STREQUAL "msan")
        set(SAN_CXX_FLAGS "")
    else()
        list(REMOVE_ITEM SAN_CXX_FLAGS -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE)
    endif()
else()
    message(FATAL_ERROR "hexhost: FUZZ_PROFILE must be debug or release, it is ${FUZZ_PROFILE}")
endif()

add_compile_options(${PROFILE_FLAGS} ${SAN_FLAGS} -fsanitize=fuzzer-no-link
                    "$<$<COMPILE_LANGUAGE:CXX>:${SAN_CXX_FLAGS}>")
add_link_options(${PROFILE_LINK_FLAGS} ${SAN_FLAGS} ${SAN_LINK_FLAGS})
message(STATUS "hexhost: profile ${FUZZ_PROFILE}, sanitizer ${FUZZ_SANITIZER}: "
               "${PROFILE_FLAGS} ${SAN_FLAGS} | C++ ${SAN_CXX_FLAGS} | link ${PROFILE_LINK_FLAGS} ${SAN_FLAGS} ${SAN_LINK_FLAGS}")

# The version header of ggml.c
set(GGML_VERSION "0.0.0")
set(GGML_BUILD_COMMIT "hexhost")
configure_file("${G}/src/ggml-version.h.in" "${CMAKE_CURRENT_BINARY_DIR}/ggml-version.h" @ONLY)

# The ggml base library
add_library(hexhost_ggml STATIC
    ${G}/src/ggml.c
    ${G}/src/ggml.cpp
    ${G}/src/ggml-alloc.c
    ${G}/src/ggml-backend.cpp
    ${G}/src/ggml-backend-meta.cpp
    ${G}/src/ggml-opt.cpp
    ${G}/src/ggml-threading.cpp
    ${G}/src/ggml-quants.c
    ${G}/src/gguf.cpp)
target_include_directories(hexhost_ggml PUBLIC ${G}/include ${G}/src ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(hexhost_ggml PUBLIC GGML_SCHED_MAX_COPIES=4 _GNU_SOURCE)
target_compile_options(hexhost_ggml PRIVATE -w)
find_package(Threads REQUIRED)
target_link_libraries(hexhost_ggml PUBLIC Threads::Threads m)

# The host part of the Hexagon backend, the fake DSP and the model of the DSP checks
add_library(hexhost_core STATIC
    common/hexhost_tu.cpp
    common/fake_dsp.cpp
    common/dsp_model.cpp
    common/harness.cpp
    common/symexec.cpp
    common/graphgen.cpp)
target_include_directories(hexhost_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/stubs
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${HEX}
    ${HEX}/htp)
target_link_libraries(hexhost_core PUBLIC hexhost_ggml)
target_compile_options(hexhost_core PRIVATE -Wall -Wno-unused-function -Wno-unused-variable -Wno-missing-field-initializers)

# One libFuzzer target for each fuzz_*.cpp. tests/sanitizers/fuzz_death.h gives the death
# callback that a TSan report cannot deadlock, and the crash file in FUZZ_ARTIFACT_DIR.
foreach(name kparams graph repack envparse dirty)
    add_executable(fuzz_${name} fuzz_${name}.cpp)
    target_link_libraries(fuzz_${name} PRIVATE hexhost_core)
    target_include_directories(fuzz_${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../sanitizers")
    target_link_options(fuzz_${name} PRIVATE -fsanitize=fuzzer)
endforeach()

# The repacked bytes of the K-quant forms of the tensors of a model (tools/repack_bytes.cpp): two
# builds against two llama.cpp trees compare a change of the repack on real weights.
add_executable(hexhost_repack_bytes tools/repack_bytes.cpp)
target_link_libraries(hexhost_repack_bytes PRIVATE hexhost_core)

# The dirty range tracker of the DSP (htp/htp-tensor.c) runs on x86 with the
# stubs of stubs/dsp. A copy of the source goes into the build directory, thus
# its quoted includes find the stubs before the real headers of htp/.
configure_file("${HEX}/htp/htp-tensor.c" "${CMAKE_CURRENT_BINARY_DIR}/dsp/htp-tensor.c" COPYONLY)
target_sources(fuzz_dirty PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/dsp/htp-tensor.c")
target_include_directories(fuzz_dirty BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stubs/dsp)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/dsp/htp-tensor.c" PROPERTIES COMPILE_OPTIONS "-w")
