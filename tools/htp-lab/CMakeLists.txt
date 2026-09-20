# The kernel lab for the Hexagon HTP kernels of llama.cpp. Refer to run.sh for the usage.
#
# Each target compiles one kernel of the llama.cpp Hexagon backend verbatim (the kernel file or the
# kernel header is included into the lab program) together with the lab runtime and the stubs of
# the DSP runtime. The "*_after" targets compile the same program against a copy of the kernel
# directory with the proposal patches applied (HTP_PROPOSED_DIR, prepared by run.sh).
cmake_minimum_required(VERSION 3.22)
project(htp-lab C)

set(LLAMA_DIR "/llama" CACHE PATH "The llama.cpp checkout (read only)")
set(HTP_DIR "${LLAMA_DIR}/ggml/src/ggml-hexagon/htp" CACHE PATH "The kernel directory")
set(HTP_PROPOSED_DIR "" CACHE PATH "The patched copy of the kernel directory, empty for none")
set(HEXAGON_ARCH "v79" CACHE STRING "The Hexagon architecture of the phone")

set(LAB_C_FLAGS
    -m${HEXAGON_ARCH} -mhvx=${HEXAGON_ARCH} -mhvx-length=128B -mhmx
    -O2 -g -G0 -fvectorize -fno-zero-initialized-in-bss
    -DNDEBUG=1 -DHTP_MAX_NTHREADS=10
    -Wall -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable
    -Wno-gnu-zero-variadic-macro-arguments -Wno-format)

function(lab_add_target name source htp_include)
    add_executable(${name} ${source} ${CMAKE_CURRENT_SOURCE_DIR}/lab/lab.c ${CMAKE_CURRENT_SOURCE_DIR}/lab/stubs.c)
    target_compile_options(${name} PRIVATE ${LAB_C_FLAGS})
    # The stub directory comes first: it shadows the DSP runtime headers of the SDK.
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/lab/stubs
        ${CMAKE_CURRENT_SOURCE_DIR}/lab
        ${htp_include}
        ${LLAMA_DIR}/ggml/src
        ${LLAMA_DIR}/ggml/include)
    target_link_options(${name} PRIVATE -m${HEXAGON_ARCH} -mhvx=${HEXAGON_ARCH} -mhmx -G0)
endfunction()

# The targets are the files lab/target_<name>.c. A new target needs no edit of this file, thus
# several agents can add targets at the same time without a conflict. CONFIGURE_DEPENDS makes
# the build system run CMake again when a file appears or goes away.
# LAB_TARGETS selects the targets. When it is empty every lab/target_*.c is a target, thus a
# new target needs no edit of this file and several agents can add targets at the same time.
# Naming the targets is faster and it keeps a broken target of another agent out of the build.
set(LAB_TARGETS "" CACHE STRING "The targets to build, empty for every lab/target_*.c")
if (NOT LAB_TARGETS)
    file(GLOB LAB_TARGET_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/lab/target_*.c)
    foreach(src ${LAB_TARGET_SOURCES})
        get_filename_component(base ${src} NAME_WE)
        string(REPLACE "target_" "" name ${base})
        list(APPEND LAB_TARGETS ${name})
    endforeach()
endif()
separate_arguments(LAB_TARGETS)
list(SORT LAB_TARGETS)
message(STATUS "lab targets: ${LAB_TARGETS}")
foreach(t ${LAB_TARGETS})
    lab_add_target(lab_${t} ${CMAKE_CURRENT_SOURCE_DIR}/lab/target_${t}.c ${HTP_DIR})
    if (HTP_PROPOSED_DIR)
        lab_add_target(lab_${t}_after ${CMAKE_CURRENT_SOURCE_DIR}/lab/target_${t}.c ${HTP_PROPOSED_DIR})
        target_compile_definitions(lab_${t}_after PRIVATE LAB_PROPOSED=1)
    endif()
endforeach()
