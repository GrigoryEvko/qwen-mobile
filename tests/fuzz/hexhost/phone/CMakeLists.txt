# The phone driver of the hexhost fuzz targets (hexhost_phone) and the ggml libraries that it loads,
# for arm64 Android. tests/fuzz/hexhost/run.sh phone-build runs this project in the Snapdragon
# container with the NDK toolchain file. One build directory for each pair:
# build/fuzz/hexhost-android-<profile>-<sanitizer>.
#
# FUZZ_PROFILE (the flags of the preset arm64-android-snapdragon of android/snapdragon/CMakeUserPresets.json):
#   release  -march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only
#            -flto -D_GNU_SOURCE -O3 -DNDEBUG -g (the flags of the shipped libraries)
#   debug    the same -march and fp flags, -O1 -g -fno-omit-frame-pointer, no LTO, no NDEBUG
# FUZZ_SANITIZER: none, asan, hwasan or ubsan (one each). The NDK has no TSan and no MSan.
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

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEXHOST_LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/llama.cpp" CACHE PATH
    "The llama.cpp tree with the patch series applied")
set(FUZZ_SANITIZER "none" CACHE STRING "none, asan, hwasan or ubsan")
set(FUZZ_PROFILE "debug" CACHE STRING "debug or release")
option(HEXHOST_PHONE_X86 "Build for the host with the CPU backend only (a check of the driver)" OFF)

if (HEXHOST_PHONE_X86)
    set(ARCH_FLAGS "")
else()
    set(ARCH_FLAGS -march=armv8.7a+fp16+dotprod+i8mm)
endif()
set(APP_FP_FLAGS -fvectorize -ffp-model=fast -fno-finite-math-only -D_GNU_SOURCE)

if (FUZZ_PROFILE STREQUAL "release")
    set(PROFILE_FLAGS ${ARCH_FLAGS} ${APP_FP_FLAGS} -flto -O3 -DNDEBUG -g)
    set(PROFILE_LINK_FLAGS -flto)
elseif (FUZZ_PROFILE STREQUAL "debug")
    set(PROFILE_FLAGS ${ARCH_FLAGS} ${APP_FP_FLAGS} -O1 -g -fno-omit-frame-pointer)
    set(PROFILE_LINK_FLAGS "")
else()
    message(FATAL_ERROR "hexhost_phone: FUZZ_PROFILE must be debug or release, it is ${FUZZ_PROFILE}")
endif()

if (FUZZ_SANITIZER STREQUAL "asan")
    set(SAN_FLAGS -fsanitize=address -fno-optimize-sibling-calls)
elseif (FUZZ_SANITIZER STREQUAL "hwasan")
    if (HEXHOST_PHONE_X86)
        message(FATAL_ERROR "hexhost_phone: hwasan is for arm64 only")
    endif()
    set(SAN_FLAGS -fsanitize=hwaddress)
elseif (FUZZ_SANITIZER STREQUAL "ubsan")
    # No check recovers, thus each UBSan report stops the run.
    set(SAN_FLAGS -fsanitize=undefined -fno-sanitize-recover=undefined)
    if (ANDROID)
        # The shared UBSan runtime of the NDK has its own copy of the libc++abi type_info
        # classes, and the executable has the copy of the static NDK libc++. The vptr check
        # does a dynamic_cast to the copy of the runtime, which fails for the type_info
        # objects of the executable, thus it gave a false report at the start
        # ("dynamic-type-mismatch" in sstream). The static runtime uses the one copy of the
        # executable, and it exports its handlers to the ggml libraries.
        set(SAN_LINK_FLAGS -static-libsan)
    endif()
elseif (FUZZ_SANITIZER STREQUAL "none")
    set(SAN_FLAGS "")
else()
    message(FATAL_ERROR "hexhost_phone: FUZZ_SANITIZER must be none, asan, hwasan or ubsan, it is ${FUZZ_SANITIZER}")
endif()

add_compile_options(${PROFILE_FLAGS} ${SAN_FLAGS})
add_link_options(${PROFILE_LINK_FLAGS} ${SAN_FLAGS} ${SAN_LINK_FLAGS})
message(STATUS "hexhost_phone: profile ${FUZZ_PROFILE}, sanitizer ${FUZZ_SANITIZER}: ${PROFILE_FLAGS} ${SAN_FLAGS}")

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
