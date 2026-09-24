# The DSP library of the DDR read probe, libddrbw_skel.so, for one Hexagon version.
#
# The parent project (../CMakeLists.txt) adds one ExternalProject of this directory for each
# version. The toolchain file is htp/cmake-toolchain.cmake of the llama.cpp backend, thus the
# library gets the compiler flags of libggml-htp-<version>.so. The library also compiles
# htp/dma-queue.c of the backend: the DMA descriptors of the probe are the descriptors of the GEMV
# kernels.
cmake_minimum_required(VERSION 3.22.2)
project(ddrbw-dsp C)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

set(HTP_DIR "" CACHE PATH "The kernel directory of the llama.cpp backend (ggml/src/ggml-hexagon/htp)")
set(DDRBW_OUT "" CACHE PATH "The install directory of the library")
if (NOT EXISTS "${HTP_DIR}/dma-queue.c")
    message(FATAL_ERROR "ddrbw: ${HTP_DIR}/dma-queue.c does not exist. Set HTP_DIR to the htp directory of the backend.")
endif()

set(PROBE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_library(ddrbw_skel SHARED
    ddrbw_dsp.c
    ${HTP_DIR}/dma-queue.c
    ${CMAKE_CURRENT_BINARY_DIR}/ddrbw_skel.c)

target_include_directories(ddrbw_skel PRIVATE
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${PROBE_DIR}
    ${HTP_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(ddrbw_skel PRIVATE NDEBUG=1)

build_idl(${PROBE_DIR}/ddrbw.idl ddrbw_skel)

install(TARGETS ddrbw_skel LIBRARY DESTINATION ${DDRBW_OUT})
