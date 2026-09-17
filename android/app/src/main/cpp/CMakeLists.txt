cmake_minimum_required(VERSION 3.22)
project(qwenmobile LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_CPP_DIR "" CACHE PATH "The llama.cpp checkout")
if(NOT LLAMA_CPP_DIR OR NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "LLAMA_CPP_DIR is not a llama.cpp checkout: '${LLAMA_CPP_DIR}'")
endif()

# ggml and llama become static archives inside libqwenmobile.so.
# The OpenCL backend targets the Adreno GPU. The CPU backend keeps the Oryon features.
set(BUILD_SHARED_LIBS               OFF CACHE BOOL   "" FORCE)
set(GGML_OPENCL                     ON  CACHE BOOL   "" FORCE)
set(GGML_OPENCL_EMBED_KERNELS       ON  CACHE BOOL   "" FORCE)
set(GGML_OPENCL_USE_ADRENO_KERNELS  ON  CACHE BOOL   "" FORCE)
set(GGML_NATIVE                     OFF CACHE BOOL   "" FORCE)
set(GGML_CPU_ARM_ARCH "armv8.6-a+fp16+dotprod+i8mm+bf16" CACHE STRING "" FORCE)
set(GGML_CPU_KLEIDIAI               OFF CACHE BOOL   "" FORCE)
set(GGML_OPENMP                     OFF CACHE BOOL   "" FORCE)
set(GGML_LLAMAFILE                  OFF CACHE BOOL   "" FORCE)
set(LLAMA_BUILD_COMMON              ON  CACHE BOOL   "" FORCE)
set(LLAMA_BUILD_TESTS               OFF CACHE BOOL   "" FORCE)
set(LLAMA_BUILD_TOOLS               OFF CACHE BOOL   "" FORCE)
set(LLAMA_BUILD_EXAMPLES            OFF CACHE BOOL   "" FORCE)
set(LLAMA_BUILD_SERVER              OFF CACHE BOOL   "" FORCE)
set(LLAMA_BUILD_APP                 OFF CACHE BOOL   "" FORCE)
set(LLAMA_OPENSSL                   OFF CACHE BOOL   "" FORCE)
set(LLAMA_ALL_WARNINGS              OFF CACHE BOOL   "" FORCE)
# The vision projector runs through libmtmd. Video needs ffmpeg, thus it is off.
set(LLAMA_BUILD_MTMD                ON  CACHE BOOL   "" FORCE)
set(MTMD_VIDEO                      OFF CACHE BOOL   "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)

# The JNI layer reads the OpenCL device properties itself.
find_package(OpenCL REQUIRED)

add_library(qwenmobile SHARED
    cache_io.cpp
    image_cache.cpp
    llama_jni.cpp
    perf_hint.cpp
    state_cache.cpp)

target_include_directories(qwenmobile PRIVATE
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/tools/mtmd)

target_link_libraries(qwenmobile PRIVATE
    llama
    llama-common
    mtmd
    OpenCL::OpenCL
    android
    log)
