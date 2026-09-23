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
# integer-divide-by-zero does not recover: its finding (task #156,
# init_fastdiv_values in the matmul params of the Hexagon host) has its fix
# in patches/fuzz-hexhost/0004, thus each division report stops the run.
sanitizer_matrix_apply(ubsan
    "-fsanitize=undefined -fno-sanitize-recover=undefined"
    "-fsanitize=undefined"
    "")
