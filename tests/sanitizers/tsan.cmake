# Initial cache for the "tsan" configuration: ThreadSanitizer only.
#
# Rule R7: GGML_OPENMP=OFF (common.cmake). libgomp has no TSan
# instrumentation, thus TSan cannot see its barriers and reports false races.
#
# Usage:
#   cmake -S <project> -B <build> -C tests/sanitizers/profile-<debug|release>.cmake \
#       -C tests/sanitizers/tsan.cmake
include("${CMAKE_CURRENT_LIST_DIR}/common.cmake")
sanitizer_matrix_apply(tsan
    "-fsanitize=thread"
    "-fsanitize=thread"
    "")
