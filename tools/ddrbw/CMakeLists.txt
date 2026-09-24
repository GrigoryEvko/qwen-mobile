# The DDR read probe: the Android host program ddrbw and one DSP library libddrbw_skel.so for each
# Hexagon version. build.sh runs this project in the Snapdragon container. The structure follows
# ggml/src/ggml-hexagon/CMakeLists.txt of llama.cpp: the host target uses the qaic stub, and each
# DSP version is an ExternalProject with the Hexagon toolchain of the backend.
#
# Cache variables:
#   DDRBW_DSPS  The Hexagon versions, for example "v79"
#   DDRBW_OUT   The output directory: bin/ddrbw and <version>/libddrbw_skel.so
#   LLAMA_DIR   The llama.cpp tree (read only): the toolchain file and htp/dma-queue.c
cmake_minimum_required(VERSION 3.22)
project(ddrbw C)

set(DDRBW_DSPS "v79" CACHE STRING "The Hexagon versions of the DSP libraries")
set(DDRBW_OUT "" CACHE PATH "The output directory")
set(LLAMA_DIR "" CACHE PATH "The llama.cpp tree")
if (NOT DDRBW_OUT OR NOT LLAMA_DIR)
    message(FATAL_ERROR "Set DDRBW_OUT and LLAMA_DIR. Run tools/ddrbw/build.sh.")
endif()
if (NOT IS_DIRECTORY "${HEXAGON_SDK_ROOT}" OR NOT IS_DIRECTORY "${HEXAGON_TOOLS_ROOT}")
    message(FATAL_ERROR "Set HEXAGON_SDK_ROOT and HEXAGON_TOOLS_ROOT to the Hexagon SDK. The Snapdragon image sets them.")
endif()

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
include(ExternalProject)

set(HTP_DIR "${LLAMA_DIR}/ggml/src/ggml-hexagon/htp")

# The host program. -O3 for the CPU read loops, as the release build of the app.
add_executable(ddrbw
    host/ddrbw.c
    host/rpc.c
    ${CMAKE_CURRENT_BINARY_DIR}/ddrbw_stub.c)
target_include_directories(ddrbw PRIVATE
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/host
    ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(ddrbw PRIVATE _GNU_SOURCE)
target_compile_options(ddrbw PRIVATE -O3 -Wall -Wextra)
set_source_files_properties(host/ddrbw.c host/rpc.c PROPERTIES COMPILE_OPTIONS "-Werror")
target_link_libraries(ddrbw PRIVATE dl log m)
find_package(Threads REQUIRED)
target_link_libraries(ddrbw PRIVATE Threads::Threads)
build_idl(ddrbw.idl ddrbw)
install(TARGETS ddrbw RUNTIME DESTINATION ${DDRBW_OUT}/bin)

foreach(V ${DDRBW_DSPS})
    ExternalProject_Add(ddrbw-skel-${V}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dsp
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dsp-${V}
        BUILD_ALWAYS ON
        BUILD_BYPRODUCTS ${DDRBW_OUT}/${V}/libddrbw_skel.so
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_TOOLCHAIN_FILE=${HTP_DIR}/cmake-toolchain.cmake
            -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}
            -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_ROOT}
            -DDSP_VERSION=${V}
            -DPREBUILT_LIB_DIR=toolv19_${V}
            -DHTP_DIR=${HTP_DIR}
            -DDDRBW_OUT=${DDRBW_OUT}/${V})
endforeach()
