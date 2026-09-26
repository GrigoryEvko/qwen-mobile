# The shared CMake functions of the fuzz areas (tests/fuzz/<area>/CMakeLists.txt).
#
# Include this file at the start of the project:
#   include("${CMAKE_CURRENT_LIST_DIR}/../../sanitizers/fuzz-common.cmake")
#
# A host build takes its flags from the two initial cache files
# (cmake -C tests/sanitizers/profile-<profile>.cmake -C tests/sanitizers/<config>.cmake), and it
# needs the flag functions of this file only for a target of its own. A build with the NDK toolchain
# gets no initial cache, because those files are for x86_64 Linux and clang. Thus each area wrote the
# same tables again, and this file holds them one time:
#   fuzz_sanitizer_flags     the compile, link and C++ flags of one sanitizer
#   fuzz_profile_flags       the compile flags and the link flags of one profile
#   fuzz_profile_opt_flags   the optimizer flags of one profile
#   fuzz_profile_build_type  the build type and the optimizer flags in the cache
#   fuzz_profile_cxx_flags   the library asserts of the debug profile
#   fuzz_scalar_x86          each x86 SIMD option of ggml OFF
#   fuzz_fuzzer_no_main      the libFuzzer runtime without its main
#
# The flags are the flags of tests/sanitizers/common.cmake, profile-<profile>.cmake and
# <config>.cmake. tests/sanitizers/check-rules.sh (rule R12) compares the flags of each build
# directory with the shipped flags of the preset, thus a drift of this file fails the suite.
#
# FUZZ_MSAN_PREFIX is the install directory of the MSan libc++
# (tests/sanitizers/build-msan-libcxx.sh). The default is
# build/fuzz/msan-libcxx/install of this repository.

# The directory of the shared files, for a target that includes fuzz_death.h.
get_filename_component(FUZZ_SHARED_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

if (NOT DEFINED FUZZ_MSAN_PREFIX OR FUZZ_MSAN_PREFIX STREQUAL "")
    get_filename_component(FUZZ_MSAN_PREFIX "${FUZZ_SHARED_DIR}/../../build/fuzz/msan-libcxx/install" ABSOLUTE)
endif()

# The floating-point flags and the feature define of the shipped build
# (android/snapdragon/CMakeUserPresets.json, preset arm64-android-snapdragon).
set(FUZZ_FP_FLAGS "-fvectorize -ffp-model=fast -fno-finite-math-only -D_GNU_SOURCE")

# Give the flags of one sanitizer in the three variables that the caller names.
#
# Arguments:
#   config       none, asan, ubsan, tsan, msan or hwasan
#   out_compile  the flags of each C and C++ object
#   out_link     the flags of each link step
#   out_cxx      more flags of each C++ object (the MSan libc++)
#
# The msan configuration needs the MSan libc++: MSan must see each store, and libstdc++ has no
# instrumentation. The function stops the configure step when that library is missing.
#
# An Android build of the none or the ubsan configuration links the UBSan runtime statically. The
# libFuzzer of the NDK links that runtime also with no sanitizer. The shared runtime of the NDK
# holds a private copy of the libc++abi type-info classes, and each executable holds the copy of the
# static NDK libc++. The vptr check walks the class hierarchy with a dynamic_cast to the copy of the
# runtime, which fails for the type_info objects of the executable. Thus, with the shared runtime,
# each check of a derived class gives a false report. The static runtime uses the one copy of the
# executable, and the vptr check stays on without a recovery.
function(fuzz_sanitizer_flags config out_compile out_link out_cxx)
    set(compile "")
    set(link "")
    set(cxx "")
    if (config STREQUAL "none")
    elseif (config STREQUAL "asan")
        # -fno-optimize-sibling-calls keeps each caller in the stack of a report.
        set(compile "-fsanitize=address -fno-optimize-sibling-calls")
        set(link "-fsanitize=address")
    elseif (config STREQUAL "ubsan")
        # Rule R5: no check recovers, thus the first report stops the run.
        set(compile "-fsanitize=undefined -fno-sanitize-recover=undefined")
        set(link "-fsanitize=undefined")
    elseif (config STREQUAL "tsan")
        set(compile "-fsanitize=thread")
        set(link "-fsanitize=thread")
    elseif (config STREQUAL "hwasan")
        set(compile "-fsanitize=hwaddress")
        set(link "-fsanitize=hwaddress")
    elseif (config STREQUAL "msan")
        if (NOT EXISTS "${FUZZ_MSAN_PREFIX}/include/c++/v1/vector" OR NOT EXISTS "${FUZZ_MSAN_PREFIX}/lib/libc++.so")
            message(FATAL_ERROR
                "msan: the MSan libc++ is not in '${FUZZ_MSAN_PREFIX}'. "
                "tests/sanitizers/build-msan-libcxx.sh builds it into build/fuzz/msan-libcxx/install. "
                "Build it first, or give its install directory with -DFUZZ_MSAN_PREFIX=<dir>.")
        endif()
        set(compile "-fsanitize=memory -fsanitize-memory-track-origins=2")
        set(link "-fsanitize=memory -stdlib=libc++ -L${FUZZ_MSAN_PREFIX}/lib -Wl,-rpath,${FUZZ_MSAN_PREFIX}/lib -lc++ -lc++abi")
        set(cxx "-stdlib=libc++ -nostdinc++ -isystem ${FUZZ_MSAN_PREFIX}/include/c++/v1")
    else()
        message(FATAL_ERROR "fuzz: the configuration '${config}' is not known. Use none, asan, ubsan, "
                            "tsan, msan (host) or hwasan (Android).")
    endif()
    if (ANDROID AND (config STREQUAL "ubsan" OR config STREQUAL "none"))
        string(APPEND link " -static-libsan")
    endif()
    string(STRIP "${compile}" compile)
    string(STRIP "${link}" link)
    string(STRIP "${cxx}" cxx)
    set(${out_compile} "${compile}" PARENT_SCOPE)
    set(${out_link} "${link}" PARENT_SCOPE)
    set(${out_cxx} "${cxx}" PARENT_SCOPE)
endfunction()

# Give the compile flags and the link flags of one profile in the two variables that the caller
# names. The flags are the flags of tests/sanitizers/common.cmake and profile-<profile>.cmake:
#   release  the flags of the shipped build: -flto and the floating-point flags, plus -g
#   debug    the same flags without -flto
# -fno-omit-frame-pointer is the x86 substitution of the frame pointer that the shipped arm64 code
# keeps. The sanitizers use it for the stacks of malloc and free.
# The linker of a host build is lld, which reads the bitcode of -flto. An Android build takes the
# linker of the NDK, which is lld.
#
# Arguments: the profile, the variable of the compile flags, the variable of the link flags.
function(fuzz_profile_flags profile out_flags out_link)
    if (profile STREQUAL "release")
        set(flags "-g -fno-omit-frame-pointer ${FUZZ_FP_FLAGS} -flto")
        set(link "-flto")
    elseif (profile STREQUAL "debug")
        set(flags "-g -fno-omit-frame-pointer ${FUZZ_FP_FLAGS}")
        set(link "")
    else()
        message(FATAL_ERROR "fuzz: the profile '${profile}' is not known. Use debug or release.")
    endif()
    if (NOT ANDROID)
        string(APPEND link " -fuse-ld=lld")
    endif()
    string(STRIP "${flags}" flags)
    string(STRIP "${link}" link)
    set(${out_flags} "${flags}" PARENT_SCOPE)
    set(${out_link} "${link}" PARENT_SCOPE)
endfunction()

# Give the optimizer flags of one profile in the variable that the caller names, as
# CMAKE_<LANG>_FLAGS_<TYPE> of the profile file gives them.
# Arguments: the profile, the variable of the flags.
function(fuzz_profile_opt_flags profile out_flags)
    if (profile STREQUAL "release")
        set(${out_flags} "-O3 -DNDEBUG" PARENT_SCOPE)
    elseif (profile STREQUAL "debug")
        set(${out_flags} "-O1 -g" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "fuzz: the profile '${profile}' is not known. Use debug or release.")
    endif()
endfunction()

# Set the build type and the optimizer flags of one profile in the cache. A build that gives the
# initial cache files does not call this function: those files set the same entries.
# Argument: the profile.
macro(fuzz_profile_build_type profile)
    fuzz_profile_opt_flags("${profile}" _fuzz_opt_flags)
    if ("${profile}" STREQUAL "release")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
        set(CMAKE_C_FLAGS_RELEASE "${_fuzz_opt_flags}" CACHE STRING "" FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "${_fuzz_opt_flags}" CACHE STRING "" FORCE)
    else()
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
        set(CMAKE_C_FLAGS_DEBUG "${_fuzz_opt_flags}" CACHE STRING "" FORCE)
        set(CMAKE_CXX_FLAGS_DEBUG "${_fuzz_opt_flags}" CACHE STRING "" FORCE)
    endif()
endmacro()

# Give the flags that the debug profile adds to each C++ object in the variable that the caller
# names: the library asserts. The release profile adds none, because the shipped build has none.
# The Android builds and the msan configuration use libc++, which has a hardening mode. The host
# builds use libstdc++.
# Arguments: the profile, the configuration, the variable of the flags.
function(fuzz_profile_cxx_flags profile config out_flags)
    set(flags "")
    if (profile STREQUAL "debug")
        if (ANDROID OR config STREQUAL "msan")
            set(flags "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE")
        else()
            set(flags "-D_GLIBCXX_ASSERTIONS")
        endif()
    endif()
    set(${out_flags} "${flags}" PARENT_SCOPE)
endfunction()

# Set GGML_NATIVE and each x86 SIMD option of ggml to OFF in the cache, with FORCE. The scalar code
# of ggml is the code that MemorySanitizer handles, and it is also the code of the naive oracle
# build (rule R7).
macro(fuzz_scalar_x86)
    foreach (_fuzz_opt NATIVE AVX AVX2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AVX_VNNI FMA F16C
                       SSE42 BMI2 AMX_TILE AMX_INT8 AMX_BF16)
        set(GGML_${_fuzz_opt} OFF CACHE BOOL "" FORCE)
    endforeach()
endmacro()

# Give the path of the libFuzzer runtime without its main in the variable that the caller names.
# A program with its own main (a driver, a tool) has objects with the coverage callbacks of
# libFuzzer. A sanitizer runtime holds those callbacks, and in the none configuration only this
# archive does. The file name depends on the layout of the runtime directory of clang.
# Argument: the variable of the path.
function(fuzz_fuzzer_no_main out_path)
    set(arch "${CMAKE_SYSTEM_PROCESSOR}")
    if (ANDROID)
        set(arch aarch64-android)
    endif()
    set(target_flag "")
    if (CMAKE_CXX_COMPILER_TARGET)
        set(target_flag "--target=${CMAKE_CXX_COMPILER_TARGET}")
    endif()
    set(found "")
    foreach (name libclang_rt.fuzzer_no_main.a libclang_rt.fuzzer_no_main-${arch}.a)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${target_flag} --print-file-name=${name}
                        OUTPUT_VARIABLE path OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (NOT found AND IS_ABSOLUTE "${path}" AND EXISTS "${path}")
            set(found "${path}")
        endif()
    endforeach()
    if (NOT found)
        message(FATAL_ERROR "fuzz: ${CMAKE_CXX_COMPILER} has no libclang_rt.fuzzer_no_main, thus a "
                            "program with its own main cannot link the coverage callbacks.")
    endif()
    set(${out_path} "${found}" PARENT_SCOPE)
endfunction()
