# Initial cache for the "msan" configuration: MemorySanitizer only.
#
# MSan must see each instruction that writes memory. Thus:
#   - The C++ code uses the MSan libc++ of FUZZ_MSAN_PREFIX, not libstdc++.
#     tests/sanitizers/build-msan-libcxx.sh builds it. It is the same for
#     the two profiles.
#   - Rule R7: GGML_OPENMP=OFF (common.cmake), because libgomp has no MSan
#     instrumentation.
#   - Rule R7: GGML_NATIVE=OFF and the x86 SIMD options OFF, with FORCE,
#     thus the profile cannot turn them on. The scalar code of ggml is the
#     code that MSan handles best. It is also the configuration of the naive
#     CPU oracle build (build/oracle-x86). This is the one difference of the
#     release profile of msan from the shipped flags (-march).
#
# FUZZ_MSAN_PREFIX: this file uses, in this sequence, a -DFUZZ_MSAN_PREFIX
# that comes before -C on the command line, the environment variable
# FUZZ_MSAN_PREFIX, or build/fuzz/msan-libcxx/install of this repository.
#
# Usage:
#   cmake -S <project> -B <build> -C tests/sanitizers/profile-<debug|release>.cmake \
#       -C tests/sanitizers/msan.cmake
include("${CMAKE_CURRENT_LIST_DIR}/common.cmake")

if (DEFINED FUZZ_MSAN_PREFIX)
    set(_msan_prefix "${FUZZ_MSAN_PREFIX}")
elseif (DEFINED ENV{FUZZ_MSAN_PREFIX})
    set(_msan_prefix "$ENV{FUZZ_MSAN_PREFIX}")
else()
    get_filename_component(_msan_prefix
        "${CMAKE_CURRENT_LIST_DIR}/../../build/fuzz/msan-libcxx/install" ABSOLUTE)
endif()
if (NOT EXISTS "${_msan_prefix}/include/c++/v1/vector" OR NOT EXISTS "${_msan_prefix}/lib/libc++.so")
    message(FATAL_ERROR
        "msan: the MSan libc++ is not in '${_msan_prefix}'. "
        "Run tests/sanitizers/build-msan-libcxx.sh to build it, or give its install "
        "directory with -DFUZZ_MSAN_PREFIX=<dir> before -C, or in the environment "
        "variable FUZZ_MSAN_PREFIX.")
endif()
set(FUZZ_MSAN_PREFIX "${_msan_prefix}" CACHE PATH "The install directory of the MSan libc++")

sanitizer_matrix_apply(msan
    "-fsanitize=memory -fsanitize-memory-track-origins=2"
    "-fsanitize=memory -stdlib=libc++ -L${_msan_prefix}/lib -Wl,-rpath,${_msan_prefix}/lib -lc++ -lc++abi"
    "-stdlib=libc++ -nostdinc++ -isystem ${_msan_prefix}/include/c++/v1")

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
foreach (_opt AVX AVX2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AVX_VNNI FMA F16C BMI2 SSE42 AMX_TILE AMX_INT8 AMX_BF16)
    set(GGML_${_opt} OFF CACHE BOOL "" FORCE)
endforeach()
