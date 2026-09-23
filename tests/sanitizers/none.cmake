# Initial cache for the "none" configuration: no sanitizer.
# This build is the baseline for the speed, the coverage and the correct
# result. It uses the flags of the profile file that comes before it.
#
# Usage:
#   cmake -S <project> -B <build> -C tests/sanitizers/profile-<debug|release>.cmake \
#       -C tests/sanitizers/none.cmake
include("${CMAKE_CURRENT_LIST_DIR}/common.cmake")
sanitizer_matrix_apply(none "" "" "")
