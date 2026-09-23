# Initial cache for the "debug" profile.
#
# Give this file first, then one sanitizer file:
#   cmake -S <project> -B <build> -C tests/sanitizers/profile-debug.cmake \
#       -C tests/sanitizers/<none|asan|ubsan|tsan|msan>.cmake
#
# The debug profile has the flags of the release profile
# (profile-release.cmake), with three differences:
#   - The optimizer: -O1 in place of -O3.
#   - No LTO: no -flto.
#   - Live asserts: no -DNDEBUG, thus assert() runs. The C++ objects also get
#     -D_GLIBCXX_ASSERTIONS (the libstdc++ bounds checks), or the libc++
#     hardening mode "extensive" in the msan configuration.
# The floating-point flags (-ffp-model=fast -fno-finite-math-only),
# GGML_OPENMP=OFF, GGML_NATIVE=ON (OFF for msan) and -g are the same.

set(SANMATRIX_PROFILE debug CACHE INTERNAL "")
set(FUZZ_PROFILE debug CACHE STRING "debug or release")

set(SANMATRIX_PROFILE_FLAGS "-fvectorize -ffp-model=fast -fno-finite-math-only -D_GNU_SOURCE" CACHE INTERNAL "")
set(SANMATRIX_PROFILE_LINK_FLAGS "" CACHE INTERNAL "")
set(SANMATRIX_PROFILE_ASSERTS ON CACHE INTERNAL "")

set(CMAKE_BUILD_TYPE Debug CACHE STRING "")
set(CMAKE_C_FLAGS_DEBUG "-O1 -g" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O1 -g" CACHE STRING "")
set(GGML_NATIVE ON CACHE BOOL "")
