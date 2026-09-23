# The DSP library of the ISA silicon probe, libisaprobe_skel.so, for one Hexagon version.
#
# The parent project (../CMakeLists.txt) adds one ExternalProject of this directory for each
# version, as the llama.cpp backend does for libggml-htp-v<N>.so. The toolchain file is the one of
# the llama.cpp backend (htp/cmake-toolchain.cmake). This project reads it and changes nothing in
# the llama.cpp tree.
#
# The census kernels: when ISA_DIR/isa_kernels.c exists, the library uses it and its header
# isa_kernels.h, else the stub ../isa_kernels_stub.c. With the real kernels, isaprobe_dsp.c
# includes the header: it gives the op table to the host, and the compiler compares the
# declaration of isa_run with the one of the probe. The stub has no op table.
#
# The compiler flags: the silicon must run the instruction sequences of the simulator census.
# The toolchain file of the backend gives -flto, and for v79 and v81 also
# "-mllvm -enable-xqf-gen=true". The lab builds without the two, and each one changes the
# instructions. Thus this project removes the flags of the toolchain and uses LAB_C_FLAGS of
# tools/htp-lab/CMakeLists.txt, which it reads at configure time, plus -fpic for the library.
# A change of LAB_C_FLAGS thus goes into the next build of the probe.
cmake_minimum_required(VERSION 3.22.2)
project(isaprobe-dsp C)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

set(PROBE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(ISA_DIR "" CACHE PATH "The directory of the census kernels (tools/htp-lab/isa)")
set(LAB_DIR "" CACHE PATH "The directory of the lab runtime (tools/htp-lab/lab)")
set(HTP_DIR "" CACHE PATH "The kernel directory of the llama.cpp backend (ggml/src/ggml-hexagon/htp)")
set(ISAPROBE_OUT "" CACHE PATH "The install directory of the library")

# LAB_C_FLAGS of the lab, with the version of this build. The regular expression reads the
# first "set(LAB_C_FLAGS ...)" of the file.
set(LAB_CMAKE "${LAB_DIR}/../CMakeLists.txt")
if (NOT EXISTS "${LAB_CMAKE}")
    message(FATAL_ERROR "isaprobe: ${LAB_CMAKE} does not exist. Set LAB_DIR to tools/htp-lab/lab.")
endif()
file(READ "${LAB_CMAKE}" lab_cmake_text)
string(REGEX MATCH "set\\(LAB_C_FLAGS([^)]*)\\)" lab_match "${lab_cmake_text}")
if (NOT lab_match)
    message(FATAL_ERROR "isaprobe: ${LAB_CMAKE} has no set(LAB_C_FLAGS ...). Update the regular expression of "
                        "tools/htp-lab/probe/dsp/CMakeLists.txt for the new form of that file.")
endif()
set(kernel_flags "${CMAKE_MATCH_1}")
string(REPLACE "\${HEXAGON_ARCH}" "${HEXAGON_ARCH}" kernel_flags "${kernel_flags}")
separate_arguments(kernel_flags UNIX_COMMAND "${kernel_flags}")
list(APPEND kernel_flags -fpic)
set(CMAKE_C_FLAGS_RELEASE "")

if (ISA_DIR AND EXISTS "${ISA_DIR}/isa_kernels.c")
    set(ISA_SOURCE "${ISA_DIR}/isa_kernels.c")
else()
    set(ISA_SOURCE "${PROBE_DIR}/isa_kernels_stub.c")
endif()
message(STATUS "isaprobe ${DSP_VERSION}: census kernels ${ISA_SOURCE}")
message(STATUS "isaprobe ${DSP_VERSION}: compiler flags ${kernel_flags}")

add_library(isaprobe_skel SHARED
    isaprobe_dsp.c
    candidates.c
    ${PROBE_DIR}/q8_0_ref.c
    ${CMAKE_CURRENT_BINARY_DIR}/isaprobe_skel.c
    ${ISA_SOURCE})

target_include_directories(isaprobe_skel PRIVATE
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${PROBE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})
if (ISA_DIR AND EXISTS "${ISA_DIR}")
    target_include_directories(isaprobe_skel PRIVATE ${ISA_DIR})
endif()
if (ISA_SOURCE STREQUAL "${ISA_DIR}/isa_kernels.c")
    if (NOT EXISTS "${ISA_DIR}/isa_kernels.h")
        message(FATAL_ERROR "isaprobe: ${ISA_DIR} has isa_kernels.c but no isa_kernels.h. Run gen_census.py generate.")
    endif()
    set_source_files_properties(isaprobe_dsp.c PROPERTIES COMPILE_DEFINITIONS ISAPROBE_HAVE_ISA_KERNELS_H=1)
endif()
# The lab header and the kernel headers of the backend come after the SDK headers, thus the
# stubs of the lab (lab/stubs) never shadow the DSP runtime here.
foreach(dir ${LAB_DIR} ${HTP_DIR} ${HTP_DIR}/../.. ${HTP_DIR}/../../../include)
    if (dir AND EXISTS "${dir}")
        target_include_directories(isaprobe_skel PRIVATE ${dir})
    endif()
endforeach()

target_compile_options(isaprobe_skel PRIVATE ${kernel_flags})
set_source_files_properties(isaprobe_dsp.c candidates.c ${CMAKE_CURRENT_BINARY_DIR}/isaprobe_skel.c PROPERTIES
    COMPILE_OPTIONS "-Werror;-Wformat")
# The ggml reference (candidate kernel 0) runs as naive IEEE code: no fused multiply and add.
# -fno-builtin-roundf keeps roundf a call of the weak symbol (q8_0_ref.c), not a libcall of the
# backend, which is always a strong reference.
set_source_files_properties(${PROBE_DIR}/q8_0_ref.c PROPERTIES COMPILE_OPTIONS
    "-Werror;-ffp-contract=off;-fno-fast-math;-fno-builtin-roundf")

# The compiler calls helper functions of libgcc for some conversions, for example
# __qf_convert_hf_to_h_rne on v79. The lab program links them from the libgcc.a of this
# toolchain. The library links the same archive (the PIC form), thus the phone runs the helper
# code of the simulator run, and the DSP loader needs no helper from the libgcc.so of the phone.
set(ISAPROBE_LIBGCC "${HEXAGON_TOOLS_ROOT}/Tools/target/hexagon/lib/${HEXAGON_ARCH}/G0/pic/libgcc.a")
if (NOT EXISTS "${ISAPROBE_LIBGCC}")
    message(FATAL_ERROR "isaprobe: ${ISAPROBE_LIBGCC} does not exist. Check HEXAGON_TOOLS_ROOT.")
endif()
target_link_libraries(isaprobe_skel PRIVATE ${ISAPROBE_LIBGCC})

build_idl(${PROBE_DIR}/isaprobe.idl isaprobe_skel)

install(TARGETS isaprobe_skel LIBRARY DESTINATION ${ISAPROBE_OUT})

# The census kernels as the lab compiles them: LAB_C_FLAGS without -fpic. build.sh compares the
# census functions of this object with those of the library (check_disasm.py).
if (ISA_SOURCE STREQUAL "${ISA_DIR}/isa_kernels.c")
    set(lab_flags ${kernel_flags})
    list(REMOVE_ITEM lab_flags -fpic)
    add_library(isa_lab_ref OBJECT ${ISA_SOURCE})
    set_target_properties(isa_lab_ref PROPERTIES POSITION_INDEPENDENT_CODE OFF)
    target_include_directories(isa_lab_ref PRIVATE $<TARGET_PROPERTY:isaprobe_skel,INCLUDE_DIRECTORIES>)
    target_compile_options(isa_lab_ref PRIVATE ${lab_flags})
    install(FILES $<TARGET_OBJECTS:isa_lab_ref> DESTINATION ${ISAPROBE_OUT} RENAME lab_ref.o)
endif()
