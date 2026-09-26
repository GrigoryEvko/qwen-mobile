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

# The shared flag functions of the areas (the tables of the sanitizers and of the profiles).
include("${CMAKE_CURRENT_LIST_DIR}/../../sanitizers/fuzz-common.cmake")

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
#
# The flags of the sanitizer come from tests/sanitizers/fuzz-common.cmake, which holds the table of
# each area. The thread of the fake DSP runs only in the tsan configuration (HEXHOST_ASYNC_DSP): the
# other configurations answer each queue call in the thread of the caller.
if (NOT FUZZ_SANITIZER MATCHES "^(none|asan|ubsan|tsan|msan)$")
    message(FATAL_ERROR "hexhost: FUZZ_SANITIZER must be none, asan, ubsan, tsan or msan, it is ${FUZZ_SANITIZER}")
endif()
fuzz_sanitizer_flags("${FUZZ_SANITIZER}" san_flags san_link san_cxx)
separate_arguments(SAN_FLAGS UNIX_COMMAND "${san_flags}")
separate_arguments(SAN_LINK_FLAGS UNIX_COMMAND "${san_link}")
separate_arguments(SAN_CXX_FLAGS UNIX_COMMAND "${san_cxx}")
if (FUZZ_SANITIZER STREQUAL "tsan")
    add_compile_definitions(HEXHOST_ASYNC_DSP=1)
else()
    add_compile_definitions(HEXHOST_ASYNC_DSP=0)
endif()
if (FUZZ_SANITIZER STREQUAL "ubsan")
    # harness.cpp gives the shared file tests/sanitizers/ubsan.supp to the runtime: an entry there
    # has an effect only for a check that tests/sanitizers/ubsan.cmake makes recover, and no check
    # recovers.
    set(HEXHOST_UBSAN_SUPP "${FUZZ_SHARED_DIR}/ubsan.supp")
    add_compile_definitions(HEXHOST_UBSAN_SUPP="${HEXHOST_UBSAN_SUPP}")
    if (NOT EXISTS "${HEXHOST_UBSAN_SUPP}")
        message(WARNING "hexhost: ${HEXHOST_UBSAN_SUPP} does not exist. The UBSan runtime then gets no "
                        "suppression file.")
    endif()
endif()

# The flags of the two profiles come from tests/sanitizers/fuzz-common.cmake, which holds the table
# of each area: the optimizer flags, the floating-point flags of the app, the frame pointer and the
# library asserts of the debug profile. The build type of this project is None, thus the optimizer
# flags go into the compile options.
# The two profiles build shared libraries (hexhost_ggml, hexhost_core), as the app ships the
# libraries and as tests/sanitizers/profile-<profile>.cmake does. Thus the LTO of the release profile
# runs one time for each library, not again for each fuzz target.
set(BUILD_SHARED_LIBS ON)
fuzz_profile_opt_flags("${FUZZ_PROFILE}" opt_flags)
fuzz_profile_flags("${FUZZ_PROFILE}" prof_flags prof_link)
fuzz_profile_cxx_flags("${FUZZ_PROFILE}" "${FUZZ_SANITIZER}" prof_cxx)
separate_arguments(PROFILE_FLAGS UNIX_COMMAND "${opt_flags} ${prof_flags}")
# The optimizer flags and the profile flags both hold -g. Each flag of the list is a switch that one
# occurrence sets, thus one occurrence is enough.
list(REMOVE_DUPLICATES PROFILE_FLAGS)
separate_arguments(PROFILE_LINK_FLAGS UNIX_COMMAND "${prof_link}")
separate_arguments(PROFILE_CXX_FLAGS UNIX_COMMAND "${prof_cxx}")
set(CXX_ONLY_FLAGS ${SAN_CXX_FLAGS} ${PROFILE_CXX_FLAGS})
if (FUZZ_PROFILE STREQUAL "release")
    if (NOT FUZZ_SANITIZER STREQUAL "msan")
        # the x86 substitution of the -march of the shipped build. MemorySanitizer does not follow
        # the SIMD intrinsics, thus its build stays scalar (rule R7).
        list(APPEND PROFILE_FLAGS -march=native)
    endif()
    # The LTO objects need the LLVM archiver
    find_program(LLVM_AR NAMES llvm-ar REQUIRED)
    find_program(LLVM_RANLIB NAMES llvm-ranlib REQUIRED)
    set(CMAKE_AR "${LLVM_AR}")
    set(CMAKE_RANLIB "${LLVM_RANLIB}")
endif()

add_compile_options(${PROFILE_FLAGS} ${SAN_FLAGS} -fsanitize=fuzzer-no-link
                    "$<$<COMPILE_LANGUAGE:CXX>:${CXX_ONLY_FLAGS}>")
add_link_options(${PROFILE_LINK_FLAGS} ${SAN_LINK_FLAGS})
message(STATUS "hexhost: profile ${FUZZ_PROFILE}, sanitizer ${FUZZ_SANITIZER}: "
               "${PROFILE_FLAGS} ${SAN_FLAGS} | C++ ${CXX_ONLY_FLAGS} | link ${PROFILE_LINK_FLAGS} ${SAN_LINK_FLAGS}")

# The version header of ggml.c
set(GGML_VERSION "0.0.0")
set(GGML_BUILD_COMMIT "hexhost")
configure_file("${G}/src/ggml-version.h.in" "${CMAKE_CURRENT_BINARY_DIR}/ggml-version.h" @ONLY)

# The ggml base library
add_library(hexhost_ggml
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
add_library(hexhost_core
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
foreach(name kparams graph repack envparse dirty mmap)
    add_executable(fuzz_${name} fuzz_${name}.cpp)
    target_link_libraries(fuzz_${name} PRIVATE hexhost_core)
    target_include_directories(fuzz_${name} PRIVATE "${FUZZ_SHARED_DIR}")
    target_link_options(fuzz_${name} PRIVATE -fsanitize=fuzzer)
endforeach()

# The repacked bytes of the K-quant forms of the tensors of a model (tools/repack_bytes.cpp): two
# builds against two llama.cpp trees compare a change of the repack on real weights. The objects
# have the coverage of -fsanitize=fuzzer-no-link, and the tool has its own main, thus it links the
# coverage callbacks of libFuzzer without the main of libFuzzer. The none config has no sanitizer
# runtime that gives these callbacks. The shared libraries find the callbacks in the tool, thus the
# tool exports its symbols.
fuzz_fuzzer_no_main(FUZZER_NO_MAIN)
add_executable(hexhost_repack_bytes tools/repack_bytes.cpp)
target_link_libraries(hexhost_repack_bytes PRIVATE hexhost_core "${FUZZER_NO_MAIN}")
set_target_properties(hexhost_repack_bytes PROPERTIES ENABLE_EXPORTS ON)

# The dirty range tracker of the DSP (htp/htp-tensor.c) runs on x86 with the
# stubs of stubs/dsp. A copy of the source goes into the build directory, thus
# its quoted includes find the stubs before the real headers of htp/.
configure_file("${HEX}/htp/htp-tensor.c" "${CMAKE_CURRENT_BINARY_DIR}/dsp/htp-tensor.c" COPYONLY)
target_sources(fuzz_dirty PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/dsp/htp-tensor.c")
target_include_directories(fuzz_dirty BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stubs/dsp)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/dsp/htp-tensor.c" PROPERTIES COMPILE_OPTIONS "-w")

# The mapping table of the DSP (prep_op_bufs and its helpers of htp/main.c) runs on x86. main.c needs the
# headers of the Hexagon SDK and the generated htp_iface.h, thus the configure step writes a copy of the
# table code into dsp/htp-mmap.c: stubs/mmap/htp-mmap-host.h, HTP_MAX_MMAPS and struct htp_mmap of
# htp/htp-ctx.h, a context with the table and max_vmem, the text of main.c from reuse_buf to the line before
# prep_tensor, and stubs/mmap/htp-mmap-api.inc. fuzz_mmap.cpp gives htp_mmap and htp_munmap. When main.c or
# htp-ctx.h no longer has these names, the configure step stops with an error. A change of main.c runs the
# configure step again.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${HEX}/htp/main.c" "${HEX}/htp/htp-ctx.h")
file(READ "${HEX}/htp/main.c" HTP_MAIN_C)
file(READ "${HEX}/htp/htp-ctx.h" HTP_CTX_H)
string(FIND "${HTP_MAIN_C}" "static inline bool reuse_buf(" MMAP_BEGIN)
string(FIND "${HTP_MAIN_C}" "static void prep_tensor(" MMAP_END)
string(REGEX MATCH "#define HTP_MAX_MMAPS[ \t]+[0-9]+" MMAP_SLOTS "${HTP_CTX_H}")
string(REGEX MATCH "struct htp_mmap {[^}]*};" MMAP_STRUCT "${HTP_CTX_H}")
if (MMAP_BEGIN LESS 0 OR MMAP_END LESS_EQUAL MMAP_BEGIN OR NOT MMAP_SLOTS OR NOT MMAP_STRUCT)
    message(FATAL_ERROR "hexhost: fuzz_mmap does not find reuse_buf and prep_tensor in ${HEX}/htp/main.c, or "
                        "HTP_MAX_MMAPS and struct htp_mmap in ${HEX}/htp/htp-ctx.h. Update the copy step of "
                        "tests/fuzz/hexhost/CMakeLists.txt.")
endif()
math(EXPR MMAP_LEN "${MMAP_END} - ${MMAP_BEGIN}")
string(SUBSTRING "${HTP_MAIN_C}" ${MMAP_BEGIN} ${MMAP_LEN} MMAP_CODE)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dsp/htp-mmap.c.new"
     "// A copy of the mapping table of htp/main.c for fuzz_mmap (tests/fuzz/hexhost/CMakeLists.txt)\n"
     "#include \"htp-mmap-host.h\"\n\n${MMAP_SLOTS}\n\n${MMAP_STRUCT}\n\n"
     "struct htp_context {\n    struct htp_mmap mmap[HTP_MAX_MMAPS];\n    uint64_t        max_vmem;\n};\n\n"
     "${MMAP_CODE}\n#include \"htp-mmap-api.inc\"\n")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/dsp/htp-mmap.c.new" "${CMAKE_CURRENT_BINARY_DIR}/dsp/htp-mmap.c" COPYONLY)
target_sources(fuzz_mmap PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/dsp/htp-mmap.c")
target_include_directories(fuzz_mmap BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stubs/mmap
                           ${CMAKE_CURRENT_SOURCE_DIR}/stubs/dsp)
# main.c assigns NULL to the uint64_t base of a buffer descriptor, which the C of clang refuses by default
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/dsp/htp-mmap.c" PROPERTIES COMPILE_OPTIONS
                            "-Wno-int-conversion;-Wno-unused-function")
