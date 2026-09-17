# CMake toolchain for the Hexagon standalone runtime (hexagon-clang, no QuRT).
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR Hexagon)

if (NOT HEXAGON_TOOLS_DIR)
    set(HEXAGON_TOOLS_DIR "$ENV{HEXAGON_TOOLS_DIR}")
endif()
if (NOT HEXAGON_TOOLS_DIR)
    set(HEXAGON_TOOLS_DIR "/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools")
endif()

set(CMAKE_C_COMPILER "${HEXAGON_TOOLS_DIR}/bin/hexagon-clang")
set(CMAKE_ASM_COMPILER "${HEXAGON_TOOLS_DIR}/bin/hexagon-clang")
set(CMAKE_AR "${HEXAGON_TOOLS_DIR}/bin/hexagon-ar")
set(CMAKE_RANLIB "${HEXAGON_TOOLS_DIR}/bin/hexagon-ranlib")

# The test program of CMake cannot run on the host. A static library is sufficient for the check.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
