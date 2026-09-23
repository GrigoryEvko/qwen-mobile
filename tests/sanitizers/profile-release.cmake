# Initial cache for the "release" profile: the flags of the shipped build.
#
# Give this file first, then one sanitizer file:
#   cmake -S <project> -B <build> -C tests/sanitizers/profile-release.cmake \
#       -C tests/sanitizers/<none|asan|ubsan|tsan|msan>.cmake
#
# The source of the flags (rule R12): android/snapdragon/CMakeUserPresets.json,
# preset arm64-android-snapdragon, as scripts/build-native.sh uses it with
# CMAKE_BUILD_TYPE=Release. build/native/llama/compile_commands.json shows
# them. tests/sanitizers/check-rules.sh compares the SANMATRIX_SHIPPED_*
# lines of this file with the preset and with that file, and the
# SANMATRIX_PROFILE_* lines with the shipped lines and the substitutions.
#
# The x86 substitutions (each one is in SANMATRIX_X86_SUBSTITUTIONS):
#   -march=armv8.7a+fp16+dotprod+i8mm   GGML_NATIVE=ON gives -march=native
#                                        (msan.cmake sets it OFF, rule R7)
#   the NDK linker lld                  -fuse-ld=lld, necessary for -flto
#   the arm64 frame pointer             -fno-omit-frame-pointer (common.cmake)
# The NDK toolchain file also adds -fstack-protector-strong,
# -D_FORTIFY_SOURCE=2, -ffunction-sections, -fdata-sections and
# -funwind-tables. They are not in the preset, and this profile does not add
# them: _FORTIFY_SOURCE replaces memcpy and the other functions with the
# __*_chk versions, which the MSan and ASan interceptors do not see in full.
# Plus -g (debug information does not change the code, and a report needs
# lines).

set(SANMATRIX_PROFILE release CACHE INTERNAL "")
set(FUZZ_PROFILE release CACHE STRING "debug or release")

# The shipped flags, copied from the preset. Do not edit these two lines
# without the same edit in the preset.
set(SANMATRIX_SHIPPED_FLAGS "-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE" CACHE INTERNAL "")
set(SANMATRIX_SHIPPED_RELEASE_FLAGS "-O3 -DNDEBUG" CACHE INTERNAL "")
set(SANMATRIX_X86_SUBSTITUTIONS "-march=armv8.7a+fp16+dotprod+i8mm=GGML_NATIVE;lld=-fuse-ld=lld;frame-pointer=-fno-omit-frame-pointer" CACHE INTERNAL "")

# The flags of this profile: the shipped flags less -march.
set(SANMATRIX_PROFILE_FLAGS "-fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE" CACHE INTERNAL "")
set(SANMATRIX_PROFILE_LINK_FLAGS "-flto -fuse-ld=lld" CACHE INTERNAL "")
set(SANMATRIX_PROFILE_ASSERTS OFF CACHE INTERNAL "")

set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "")
set(GGML_NATIVE ON CACHE BOOL "")
