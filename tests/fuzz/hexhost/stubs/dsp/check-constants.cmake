# The stubs of stubs/dsp give the constants that the dirty range tracker (htp/htp-tensor.c) reads. They
# must be those of the tree under test, else fuzz_dirty and the tracker replay of hexhost_graphs run a
# different tracker. hexhost_check_dsp_constants(HEX) stops the configure when a constant of the stubs is
# not the same text as in HEX/htp (HEX is the ggml-hexagon directory of the tree).

set(HEXHOST_DSP_STUBS_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(hexhost_check_dsp_constants hex)
    foreach(entry htp-ctx.h:HTP_MAX_DIRTY_RANGES hex-utils.h:HEX_L2_LINE_SIZE hex-utils.h:HEX_L2_BLOCK_SIZE
                  hex-utils.h:HEX_L2_FLUSH_WQ_THRESHOLD hex-utils.h:HEX_L2_FLUSH_ALL_THRESHOLD)
        string(REPLACE ":" ";" parts "${entry}")
        list(GET parts 0 header)
        list(GET parts 1 name)
        set(values "")
        foreach(path "${hex}/htp/${header}" "${HEXHOST_DSP_STUBS_DIR}/${header}")
            file(STRINGS "${path}" line REGEX "^#define[ \t]+${name}[ \t]")
            string(REGEX REPLACE "^#define[ \t]+${name}[ \t]+([^/]*).*$" "\\1" value "${line}")
            string(STRIP "${value}" value)
            list(APPEND values "${value}")
        endforeach()
        list(GET values 0 real)
        list(GET values 1 stub)
        if (real STREQUAL "" OR NOT real STREQUAL stub)
            message(FATAL_ERROR "hexhost: ${name} is '${real}' in ${hex}/htp/${header} and '${stub}' in "
                                "${HEXHOST_DSP_STUBS_DIR}/${header}. Give the stub the value of the tree.")
        endif()
    endforeach()
endfunction()
