# The phone driver of the hexhost fuzz targets (hexhost_phone) and the ggml libraries that it loads,
# for arm64 Android. tests/fuzz/hexhost/run.sh phone-build runs this project in the Snapdragon
# container with the NDK toolchain file. One build directory for each pair:
# build/fuzz/hexhost-android-<profile>-<sanitizer>.
#
# FUZZ_PROFILE (the flags of the preset arm64-android-snapdragon of android/snapdragon/CMakeUserPresets.json):
#   release  -march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only
#            -flto -D_GNU_SOURCE -O3 -DNDEBUG -g (the flags of the shipped libraries)
#   debug    the same -march and fp flags, -O1 -g, no LTO, no NDEBUG, the libc++ asserts
# FUZZ_SANITIZER: none, asan, hwasan or ubsan (one each). The NDK has no TSan and no MSan.
# The flags of the profile and of the sanitizer come from tests/sanitizers/fuzz-common.cmake, which
# reads them from the initial cache files of the host builds.
#
# The flags go to every target, thus the ggml libraries have the same instrumentation as the
# driver. Only the target hexhost_phone and its dependencies build: the DSP libraries (the
# ExternalProject targets htp-v73 to htp-v81) do not. run.sh phone-build makes the DSP library of
# the same tree with the release preset of the app (the release none run stages the shipped one).
#
# HEXHOST_PHONE_X86=ON builds the driver for the host with the CPU backend only, as a check of
# the driver before a phone run (hexhost_phone --device CPU).

cmake_minimum_required(VERSION 3.20)
project(hexhost_phone C CXX)

# The shared flag functions of the areas (the tables of the sanitizers and of the profiles).
include("${CMAKE_CURRENT_LIST_DIR}/../../../sanitizers/fuzz-common.cmake")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEXHOST_LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/llama.cpp" CACHE PATH
    "The llama.cpp tree with the patch series applied")
set(FUZZ_SANITIZER "none" CACHE STRING "none, asan, hwasan or ubsan")
set(FUZZ_PROFILE "debug" CACHE STRING "debug or release")
option(HEXHOST_PHONE_X86 "Build for the host with the CPU backend only (a check of the driver)" OFF)

if (NOT FUZZ_SANITIZER MATCHES "^(none|asan|hwasan|ubsan)$")
    message(FATAL_ERROR "hexhost_phone: FUZZ_SANITIZER must be none, asan, hwasan or ubsan, it is ${FUZZ_SANITIZER}")
endif()
if (HEXHOST_PHONE_X86 AND FUZZ_SANITIZER STREQUAL "hwasan")
    message(FATAL_ERROR "hexhost_phone: hwasan is for arm64 only")
endif()

# The build type of this project is None, thus the optimizer flags go into the compile options.
# The optimizer flags and the profile flags both hold -g: one occurrence is enough.
fuzz_profile_opt_flags("${FUZZ_PROFILE}" opt_flags)
fuzz_profile_flags("${FUZZ_PROFILE}" prof_flags prof_link)
fuzz_profile_cxx_flags("${FUZZ_PROFILE}" "${FUZZ_SANITIZER}" prof_cxx)
fuzz_sanitizer_flags("${FUZZ_SANITIZER}" san_flags san_link san_cxx)
separate_arguments(PROFILE_FLAGS UNIX_COMMAND "${opt_flags} ${prof_flags}")
list(REMOVE_DUPLICATES PROFILE_FLAGS)
if (NOT HEXHOST_PHONE_X86)
    list(APPEND PROFILE_FLAGS -march=armv8.7a+fp16+dotprod+i8mm)
endif()
separate_arguments(PROFILE_LINK_FLAGS UNIX_COMMAND "${prof_link}")
separate_arguments(SAN_FLAGS UNIX_COMMAND "${san_flags}")
# The ubsan build links the UBSan runtime statically (fuzz_sanitizer_flags), and it exports its
# handlers to the ggml libraries. This driver links no libFuzzer, thus the none build has no UBSan
# runtime to link.
if (FUZZ_SANITIZER STREQUAL "none")
    set(san_link "")
endif()
separate_arguments(SAN_LINK_FLAGS UNIX_COMMAND "${san_link}")
separate_arguments(CXX_ONLY_FLAGS UNIX_COMMAND "${san_cxx} ${prof_cxx}")

add_compile_options(${PROFILE_FLAGS} ${SAN_FLAGS} "$<$<COMPILE_LANGUAGE:CXX>:${CXX_ONLY_FLAGS}>")
add_link_options(${PROFILE_LINK_FLAGS} ${SAN_LINK_FLAGS})
message(STATUS "hexhost_phone: profile ${FUZZ_PROFILE}, sanitizer ${FUZZ_SANITIZER}: ${PROFILE_FLAGS} ${SAN_FLAGS} "
               "| C++ ${CXX_ONLY_FLAGS} | link ${PROFILE_LINK_FLAGS} ${SAN_LINK_FLAGS}")

# ggml as the app builds it (the preset arm64-android-snapdragon), without OpenCL. The ubsan
# build for the phone links ggml statically: the NDK links a shared library with --no-undefined,
# and the static UBSan runtime of the executable cannot give its handlers to such a library.
if (ANDROID AND FUZZ_SANITIZER STREQUAL "ubsan")
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
else()
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
if (HEXHOST_PHONE_X86)
    set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)
else()
    set(GGML_HEXAGON ON CACHE BOOL "" FORCE)
endif()
add_subdirectory("${HEXHOST_LLAMA_DIR}/ggml" ggml)

set(HEXHOST "${CMAKE_CURRENT_SOURCE_DIR}/..")
add_executable(hexhost_phone driver.cpp "${HEXHOST}/common/graphgen.cpp")
target_include_directories(hexhost_phone PRIVATE "${HEXHOST}/common" "${HEXHOST_LLAMA_DIR}/ggml/src")
if (DEFINED HEXHOST_FUZZER_INCLUDE)
    # the directory that holds fuzzer/FuzzedDataProvider.h when the compiler has no copy
    target_include_directories(hexhost_phone PRIVATE "${HEXHOST_FUZZER_INCLUDE}")
endif()
target_link_libraries(hexhost_phone PRIVATE ggml ggml-base)
