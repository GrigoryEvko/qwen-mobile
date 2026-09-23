# Initial cache for the "asan" configuration: AddressSanitizer only.
# AddressSanitizer includes LeakSanitizer on x86_64 Linux. The runtime
# options (ASAN_OPTIONS, LSAN_OPTIONS) come from tests/sanitizers/env.sh.
# -fno-optimize-sibling-calls keeps each caller in the stack of a report.
#
# Usage:
#   cmake -S <project> -B <build> -C tests/sanitizers/profile-<debug|release>.cmake \
#       -C tests/sanitizers/asan.cmake
include("${CMAKE_CURRENT_LIST_DIR}/common.cmake")
sanitizer_matrix_apply(asan
    "-fsanitize=address -fno-optimize-sibling-calls"
    "-fsanitize=address"
    "")
