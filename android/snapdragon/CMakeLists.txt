cmake_minimum_required(VERSION 3.22)
project(qwenmobile-snapdragon LANGUAGES C CXX)

# The JNI library against the shared llama.cpp build of the Snapdragon
# container (CPU, OpenCL and Hexagon backends, mtmd). Build with build.sh.

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_CPP_DIR "" CACHE PATH "The llama.cpp checkout")
set(LLAMA_BUILD_DIR "" CACHE PATH "The Snapdragon build directory of llama.cpp")
if(NOT EXISTS "${LLAMA_BUILD_DIR}/bin/libllama.so")
    message(FATAL_ERROR "LLAMA_BUILD_DIR has no bin/libllama.so: '${LLAMA_BUILD_DIR}'")
endif()

# The thread pool functions live in the CPU backend library.
foreach(lib llama ggml ggml-base ggml-cpu mtmd llama-common)
    add_library(${lib} SHARED IMPORTED)
    set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${LLAMA_BUILD_DIR}/bin/lib${lib}.so")
endforeach()

set(JNI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp")
add_library(qwenmobile SHARED
    "${JNI_DIR}/cache_io.cpp"
    "${JNI_DIR}/image_cache.cpp"
    "${JNI_DIR}/llama_jni.cpp"
    "${JNI_DIR}/perf_hint.cpp"
    "${JNI_DIR}/spec_policy.cpp"
    "${JNI_DIR}/state_cache.cpp"
    "${JNI_DIR}/trace.cpp")

target_include_directories(qwenmobile PRIVATE
    "${JNI_DIR}"
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/common"
    "${LLAMA_CPP_DIR}/tools/mtmd")

# The device list comes from the ggml backends here, not from the OpenCL loader.
target_compile_definitions(qwenmobile PRIVATE QWEN_NO_OPENCL_INFO)

target_link_libraries(qwenmobile PRIVATE
    llama
    llama-common
    mtmd
    ggml
    ggml-base
    ggml-cpu
    android
    log)
