# hexhost_graphs: the llama.cpp graphs of a real model on the fake DSP of the hexhost harness
# (refer to graphs.cpp). It gives the supports_op decisions of HTP0 and the fused op counts of a
# decode and a prefill graph, thus a fix of the host can show that it keeps both.
#
#   cmake -S tests/fuzz/hexhost/graphs -B build/fuzz/hexhost/graphs-<name> -G Ninja \
#         -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release \
#         -DHEXHOST_LLAMA_DIR=<llama.cpp tree>
#
# No sanitizer: the program loads a full model, and it measures, it does not fuzz.

cmake_minimum_required(VERSION 3.20)
project(hexhost_graphs C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEXHOST_LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/llama.cpp" CACHE PATH
    "The llama.cpp tree with the patch series applied")

# llama.cpp with the CPU backend only; the Hexagon host code comes from the harness below
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
add_subdirectory("${HEXHOST_LLAMA_DIR}" llama)

set(H   "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(HEX "${HEXHOST_LLAMA_DIR}/ggml/src/ggml-hexagon")
add_executable(hexhost_graphs graphs.cpp "${H}/common/hexhost_tu.cpp" "${H}/common/fake_dsp.cpp" "${H}/common/dsp_model.cpp")
target_include_directories(hexhost_graphs PRIVATE "${H}/stubs" "${H}/common" "${HEX}" "${HEX}/htp"
                           "${HEXHOST_LLAMA_DIR}/ggml/src" "${HEXHOST_LLAMA_DIR}/ggml/include")
target_compile_definitions(hexhost_graphs PRIVATE HEXHOST_ASYNC_DSP=0 _GNU_SOURCE)
target_compile_options(hexhost_graphs PRIVATE -w)
target_link_libraries(hexhost_graphs PRIVATE llama ggml ggml-base ggml-cpu)
