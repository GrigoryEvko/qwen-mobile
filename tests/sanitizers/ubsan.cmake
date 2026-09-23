# Initial cache for the "ubsan" configuration: UndefinedBehaviorSanitizer only.
#
# Rule R5: -fno-sanitize-recover=undefined, thus the first report stops the
# run. Only a check with a known defect that has an entry in
# tests/sanitizers/ubsan.supp can recover (-fsanitize-recover=<check>, with a
# comment that names the defect). The runtime option halt_on_error=1
# (tests/sanitizers/env.sh) then stops the run at each report of that check
# that the file does not suppress. No check recovers in this file.
#
# Usage:
#   cmake -S <project> -B <build> -C tests/sanitizers/profile-<debug|release>.cmake \
#       -C tests/sanitizers/ubsan.cmake
include("${CMAKE_CURRENT_LIST_DIR}/common.cmake")
sanitizer_matrix_apply(ubsan
    "-fsanitize=undefined -fno-sanitize-recover=undefined"
    "-fsanitize=undefined"
    "")
