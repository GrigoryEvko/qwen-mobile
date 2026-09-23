# Initial cache for the "ubsan" configuration: UndefinedBehaviorSanitizer only.
#
# Rule R5: -fno-sanitize-recover=undefined, thus the first report stops the
# run. Only a check with a known, tracked finding can recover. The runtime
# option halt_on_error=1 (tests/sanitizers/env.sh) stops the run at each
# report of that check that tests/sanitizers/ubsan.supp does not suppress.
#
# Usage:
#   cmake -S <project> -B <build> -C tests/sanitizers/profile-<debug|release>.cmake \
#       -C tests/sanitizers/ubsan.cmake
include("${CMAKE_CURRENT_LIST_DIR}/common.cmake")
# pointer-overflow does not recover: its finding (task #125, ggml_graph_nbytes)
# has its fix in patches/fuzz-ops/0001 (commit bac1229), thus each
# pointer-overflow report stops the run.
#
# function does not recover: its finding (task #127, the casts of the type
# traits) has its fix in patches/fuzz-ops/0002, thus each function report
# stops the run.
#
# Reason for -fsanitize-recover=integer-divide-by-zero: the tracked finding
# of task #156 (2026-09-23). The supports_op of MUL_MAT_ID in
# ggml-hexagon.cpp calls init_fastdiv_values(ne12 / ne02), which divides by
# zero when there are fewer tokens than experts. The entry
# "integer-divide-by-zero:^init_fastdiv_values(" of ubsan.supp suppresses it.
# On x86 the division then traps with SIGFPE, thus only arm64 (the phone) runs
# past it. Each other division report stops the run. Remove
# "integer-divide-by-zero" when task #156 lands.
sanitizer_matrix_apply(ubsan
    "-fsanitize=undefined -fno-sanitize-recover=undefined -fsanitize-recover=integer-divide-by-zero"
    "-fsanitize=undefined"
    "")
