# The shared part of the sanitizer and profile initial-cache files.
#
# Do not give this file to "cmake -C". Give two files: first one profile
# file, then one sanitizer file (rule R1: one sanitizer; rule R11: one
# profile):
#
#   cmake -G Ninja -S <project> -B build/fuzz/<area>-<profile>-<config> \
#       -C tests/sanitizers/profile-<debug|release>.cmake \
#       -C tests/sanitizers/<none|asan|ubsan|tsan|msan>.cmake
#
# The profiles (profile-debug.cmake, profile-release.cmake):
#   release  The flags of the shipped build (android/snapdragon/
#            CMakeUserPresets.json, preset arm64-android-snapdragon, with
#            CMAKE_BUILD_TYPE=Release) and the documented x86 substitutions.
#   debug    The same flags with -O1, no LTO and live asserts.
#
# The configurations (none.cmake ... msan.cmake): no sanitizer, or exactly
# one sanitizer.
#
# The files are for x86_64 Linux and clang. An Android build uses the NDK
# toolchain and the flags of its area, not these files.
#
# The cache entries that the files set. A -D option that comes before the
# first -C on the command line has priority, because the files do not use
# FORCE, except for the MSan rules of R7 in msan.cmake:
#   CMAKE_C_COMPILER, CMAKE_CXX_COMPILER   clang, clang++
#   CMAKE_BUILD_TYPE and CMAKE_<LANG>_FLAGS_<TYPE>   from the profile
#   CMAKE_C_FLAGS, CMAKE_CXX_FLAGS         -g -fno-omit-frame-pointer, the
#                                          flags of the profile and the flags
#                                          of the sanitizer
#   CMAKE_EXE_LINKER_FLAGS, CMAKE_SHARED_LINKER_FLAGS, CMAKE_MODULE_LINKER_FLAGS
#   FUZZ_SANITIZER, FUZZ_PROFILE           the names of the configuration
#   GGML_OPENMP                            OFF (rule R7, and the app also
#                                          builds with GGML_OPENMP=OFF)
#   GGML_NATIVE                            from the profile (ON), OFF for msan
#
# -fno-omit-frame-pointer in each profile is an x86 substitution: the
# shipped arm64 Android code keeps the frame pointer (clang
# --target=aarch64-linux-android34 -O3 emits "mov x29, sp"), and x86-64
# omits it from -O1. The sanitizers use the frame pointer for the stacks of
# malloc and free.

# Set the cache entries of one configuration in the loaded profile.
# Arguments:
#   value          none, asan, ubsan, tsan or msan
#   compile_flags  the flags of the sanitizer for C and C++ objects
#   link_flags     the flags of the sanitizer for each link step
#   cxx_flags      more flags of the sanitizer for C++ objects only
# The profile file must come first. It sets the SANMATRIX_PROFILE_* entries.
function(sanitizer_matrix_apply value compile_flags link_flags cxx_flags)
    if (NOT DEFINED SANMATRIX_PROFILE)
        message(FATAL_ERROR
            "sanitizer matrix: no profile. Give -C tests/sanitizers/profile-debug.cmake or "
            "-C tests/sanitizers/profile-release.cmake before -C tests/sanitizers/${value}.cmake.")
    endif()
    set(asserts "")
    if (SANMATRIX_PROFILE_ASSERTS)
        # The C++ library asserts of the debug profile. The msan build uses
        # the MSan libc++, thus it uses the libc++ equivalent.
        if (value STREQUAL "msan")
            set(asserts "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE")
        else()
            set(asserts "-D_GLIBCXX_ASSERTIONS")
        endif()
    endif()
    set(common "-g -fno-omit-frame-pointer ${SANMATRIX_PROFILE_FLAGS}")
    set(c_all "${common} ${compile_flags}")
    set(cxx_all "${common} ${compile_flags} ${cxx_flags} ${asserts}")
    set(link_all "${SANMATRIX_PROFILE_LINK_FLAGS} ${link_flags}")
    foreach (var c_all cxx_all link_all)
        string(REGEX REPLACE "  +" " " ${var} "${${var}}")
        string(STRIP "${${var}}" ${var})
    endforeach()

    if (NOT DEFINED CMAKE_C_COMPILER)
        set(CMAKE_C_COMPILER clang CACHE STRING "The sanitizer matrix needs clang")
    endif()
    if (NOT DEFINED CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER clang++ CACHE STRING "The sanitizer matrix needs clang++")
    endif()
    set(CMAKE_C_FLAGS "${c_all}" CACHE STRING "")
    set(CMAKE_CXX_FLAGS "${cxx_all}" CACHE STRING "")
    set(CMAKE_EXE_LINKER_FLAGS "${link_all}" CACHE STRING "")
    set(CMAKE_SHARED_LINKER_FLAGS "${link_all}" CACHE STRING "")
    set(CMAKE_MODULE_LINKER_FLAGS "${link_all}" CACHE STRING "")
    set(FUZZ_SANITIZER "${value}" CACHE STRING "none, asan, ubsan, tsan or msan")
    set(GGML_OPENMP OFF CACHE BOOL "")
    message(STATUS "sanitizer matrix: ${SANMATRIX_PROFILE}-${value}: "
                   "C '${c_all}', C++ '${cxx_all}', link '${link_all}', "
                   "type ${CMAKE_BUILD_TYPE}")
endfunction()
