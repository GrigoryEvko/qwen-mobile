# The ISA silicon probe: the Android host program isaprobe and one DSP library
# libisaprobe_skel.so for each Hexagon version. build.sh runs this project in the Snapdragon
# container. The structure follows ggml/src/ggml-hexagon/CMakeLists.txt of llama.cpp: the host
# target uses the qaic stub, and each DSP version is an ExternalProject with the Hexagon toolchain.
#
# Cache variables:
#   ISAPROBE_DSPS   The Hexagon versions, for example "v73;v75;v79;v81"
#   ISAPROBE_OUT    The output directory: bin/isaprobe and <version>/libisaprobe_skel.so
#   LLAMA_DIR       The llama.cpp tree (read only): the toolchain file and the kernel headers
#   ISA_DIR         The census kernels. Default: tools/htp-lab/isa. Without isa_kernels.c there the
#                   DSP library takes isa_kernels_stub.c, and without isa_kernels.h the host
#                   program has the "info" mode only.
cmake_minimum_required(VERSION 3.22)
project(isaprobe C)

set(ISAPROBE_DSPS "v73;v75;v79;v81" CACHE STRING "The Hexagon versions of the DSP libraries")
set(ISAPROBE_OUT "" CACHE PATH "The output directory")
set(LLAMA_DIR "" CACHE PATH "The llama.cpp tree")
if (NOT ISAPROBE_OUT OR NOT LLAMA_DIR)
    message(FATAL_ERROR "Set ISAPROBE_OUT and LLAMA_DIR. Run tools/htp-lab/probe/build.sh.")
endif()
if (NOT IS_DIRECTORY "${HEXAGON_SDK_ROOT}" OR NOT IS_DIRECTORY "${HEXAGON_TOOLS_ROOT}")
    message(FATAL_ERROR "Set HEXAGON_SDK_ROOT and HEXAGON_TOOLS_ROOT to the Hexagon SDK. The Snapdragon image sets them.")
endif()

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
include(ExternalProject)

set(HTP_DIR "${LLAMA_DIR}/ggml/src/ggml-hexagon/htp")
set(LAB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lab")
set(ISA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../isa" CACHE PATH "The census kernels (isa_kernels.c, isa_kernels.h)")

# The host program. The CPU oracles (q8_0_ref.c and host/oracle.c) are the naive IEEE execution of
# scalar C: no fused multiply and add (-ffp-contract=off) and no fast math.
set(ORACLE_FLAGS "-Werror;-ffp-contract=off;-fno-fast-math")
add_executable(isaprobe
    host/isaprobe.c
    host/cdsprpc.c
    q8_0_ref.c
    ${CMAKE_CURRENT_BINARY_DIR}/isaprobe_stub.c)
target_include_directories(isaprobe PRIVATE
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/host
    ${CMAKE_CURRENT_SOURCE_DIR}/dsp
    ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(isaprobe PRIVATE _GNU_SOURCE)
# The census mode needs the file format and the hash of isa_kernels.h. Without the header the
# program has the "info" and "kernel" modes only.
if (EXISTS "${ISA_DIR}/isa_kernels.h")
    target_sources(isaprobe PRIVATE host/census_io.c host/oracle.c)
    set_source_files_properties(host/census_io.c PROPERTIES COMPILE_OPTIONS "-Werror")
    set_source_files_properties(host/oracle.c PROPERTIES COMPILE_OPTIONS "${ORACLE_FLAGS}")
    target_include_directories(isaprobe PRIVATE ${ISA_DIR})
    target_compile_definitions(isaprobe PRIVATE ISAPROBE_HAVE_ISA_KERNELS_H=1)
else()
    message(WARNING "isaprobe: ${ISA_DIR}/isa_kernels.h does not exist. The host program has no census mode.")
endif()
target_compile_options(isaprobe PRIVATE -O2 -Wall -Wextra)
set_source_files_properties(host/isaprobe.c host/cdsprpc.c PROPERTIES COMPILE_OPTIONS "-Werror")
set_source_files_properties(q8_0_ref.c PROPERTIES COMPILE_OPTIONS "${ORACLE_FLAGS}")
target_link_libraries(isaprobe PRIVATE dl log m)
build_idl(isaprobe.idl isaprobe)
install(TARGETS isaprobe RUNTIME DESTINATION ${ISAPROBE_OUT}/bin)

# One DSP library for each version. BUILD_ALWAYS makes the ExternalProject see an edit of a source.
foreach(V ${ISAPROBE_DSPS})
    ExternalProject_Add(isaprobe-skel-${V}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dsp
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dsp-${V}
        BUILD_ALWAYS ON
        BUILD_BYPRODUCTS ${ISAPROBE_OUT}/${V}/libisaprobe_skel.so
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_TOOLCHAIN_FILE=${HTP_DIR}/cmake-toolchain.cmake
            -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}
            -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_ROOT}
            -DDSP_VERSION=${V}
            -DPREBUILT_LIB_DIR=toolv19_${V}
            -DISA_DIR=${ISA_DIR}
            -DLAB_DIR=${LAB_DIR}
            -DHTP_DIR=${HTP_DIR}
            -DISAPROBE_OUT=${ISAPROBE_OUT}/${V})
endforeach()
