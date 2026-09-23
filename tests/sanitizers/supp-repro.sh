#!/usr/bin/env bash
# Validate each entry of tests/sanitizers/ubsan.supp in one profile (rule R13).
#
# Usage:
#   tests/sanitizers/supp-repro.sh --profile <debug|release> [--jobs N]
#
# Inlining decides which function a UBSan report names, and -O3 with -flto
# inlines more than -O1. Thus an entry that matches in one profile can miss in
# the other. The way of the link also changes the inlining: the app and the
# llama.cpp tests link the shared ggml libraries, and the fuzz targets link
# static ggml libraries into one LTO program. For the profile and for each
# link (shared, static), the script:
#   1. Builds the ggml libraries of the live submodule (third_party/llama.cpp,
#      read only) with tests/sanitizers/profile-<profile>.cmake and
#      ubsan.cmake, in build/fuzz/matrix-supp-repro-<profile>/ggml-<link>.
#   2. Builds tests/sanitizers/repro/ubsan_ggml.c with the same flags and
#      links it with those libraries.
#   3. Runs each mode of the reproducer with the full ubsan.supp. The run
#      must give no report. A report means that an entry is missing (for
#      example, LTO put the defect into a caller that no entry names), and the
#      record gives the frames of the report.
#   4. For each entry, runs each mode of the entry with ubsan.supp less that
#      entry. A report of the check of the entry means that the entry is
#      necessary and matches (status pass). No report in each mode means that
#      the entry does not match (status no-match). An entry with the marker
#      "R13: <other profile> only" in its comment accepts that.
#   5. Runs the control modes with the full ubsan.supp. Each control must
#      give its report, thus the entries do not hide a report of the same
#      check in a function that no entry names.
# The records go to build/fuzz/matrix-supp-repro-<profile>/results.jsonl
# (area supp-repro, sanitizer ubsan, target "<link>/<entry>"). The script
# tests/sanitizers/check-rules.sh reads them.
#
# Requirements: cmake, ninja, clang, ld.lld, jq, rg. No container. Time:
# approximately 1 minute (debug) or 3 minutes (release, LTO). RAM: 2 GB.
#
# Exit status: 0 if each entry and each control agree with the rules, 1 if
# not, 2 if the step cannot run.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../suite" && pwd)/lib.sh"

readonly SUPP="$SUITE_REPO_ROOT/tests/sanitizers/ubsan.supp"
readonly REPRO_SRC="$SUITE_REPO_ROOT/tests/sanitizers/repro/ubsan_ggml.c"
readonly EXTERNAL="$SUITE_REPO_ROOT/tests/sanitizers/repro/external.txt"
readonly LLAMA_SRC="$SUITE_REPO_ROOT/third_party/llama.cpp"
readonly LINKS="shared static"
PROFILE=""
JOBS=8

# Read the command-line options into the global variables.
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) PROFILE="$2"; shift 2 ;;
            --jobs) JOBS="$2"; shift 2 ;;
            -h|--help) head -42 "${BASH_SOURCE[0]}" | tail -41; exit 0 ;;
            *) suite_die "The option '$1' is not known." ;;
        esac
    done
    suite_is_profile "$PROFILE" || suite_die "--profile must be debug or release, not '$PROFILE'."
}

# Configure and build the three ggml libraries with the ubsan flags.
# The ggml directory alone does not configure (ggml.pc.in is not in the
# tree), thus the whole project configures and only the libraries build.
# Argument: the link, shared or static.
build_ggml() {
    local link="$1" dir="$BUILD/ggml-$1" fp stamp init_args shared=ON
    [[ "$link" == static ]] && shared=OFF
    fp="$(suite_fingerprint "$PROFILE" ubsan "ggml-only-$link")"
    stamp="$dir/.matrix-fingerprint"
    if [[ ! -f "$stamp" || "$(cat "$stamp")" != "$fp" ]]; then
        rm -rf "$dir"
        mkdir -p "$dir"
        mapfile -t init_args < <(suite_cmake_init_args "$PROFILE" ubsan)
        cmake -G Ninja -S "$LLAMA_SRC" -B "$dir" "${init_args[@]}" \
            -DGGML_CCACHE=OFF -DGGML_BACKEND_DL=OFF -DGGML_LLAMAFILE=OFF -DBUILD_SHARED_LIBS="$shared" \
            -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            > "$dir/configure.log" 2>&1 \
            || { tail -20 "$dir/configure.log" >&2; suite_die "$PROFILE-$link: the configure step of ggml failed."; }
        echo "$fp" > "$stamp"
    fi
    nice -n 10 cmake --build "$dir" -j "$JOBS" --target ggml ggml-base ggml-cpu > "$dir/build.log" 2>&1 \
        || { tail -30 "$dir/build.log" >&2; suite_die "$PROFILE-$link: the build of ggml failed."; }
}

# Print the value of one entry of a CMake cache.
# Arguments: the build directory, the name.
cache_value() {
    rg -o -r '$1' "^$2:[A-Z]+=(.*)$" "$1/CMakeCache.txt" || true
}

# Compile the reproducer with the C flags and the link flags of one build.
# The C++ driver links, because ggml-cpu has C++ objects.
# Argument: the link, shared or static.
build_repro() {
    local link="$1" dir="$BUILD/ggml-$1" type_upper cflags ldflags libs=()
    type_upper="$(cache_value "$dir" CMAKE_BUILD_TYPE | tr '[:lower:]' '[:upper:]')"
    cflags="$(cache_value "$dir" CMAKE_C_FLAGS) $(cache_value "$dir" "CMAKE_C_FLAGS_$type_upper")"
    ldflags="$(cache_value "$dir" CMAKE_EXE_LINKER_FLAGS)"
    if [[ "$link" == shared ]]; then
        libs=(-L"$dir/bin" -Wl,-rpath,"$dir/bin" -lggml -lggml-cpu -lggml-base)
    else
        libs=("$dir/ggml/src/libggml.a" "$dir/ggml/src/libggml-cpu.a" "$dir/ggml/src/libggml-base.a" -lpthread)
    fi
    # shellcheck disable=SC2086
    { clang $cflags -I"$LLAMA_SRC/ggml/include" -c "$REPRO_SRC" -o "$dir/ubsan_ggml.o" \
        && clang++ $cflags $ldflags "$dir/ubsan_ggml.o" -o "$dir/ubsan_ggml" "${libs[@]}" -lm; } \
        > "$dir/repro-build.log" 2>&1 \
        || { cat "$dir/repro-build.log" >&2; suite_die "$PROFILE-$link: the reproducer does not build."; }
    echo "$cflags | $ldflags" > "$dir/repro-flags.txt"
}

# Run one mode of the reproducer with one suppression file.
# Arguments: the program, the mode, the suppression file, the log file.
# Print "<exit status> <number of reports>".
run_mode() {
    local prog="$1" mode="$2" supp="$3" log="$4" rc=0 n
    UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1:report_error_type=1:suppressions=$supp" \
        timeout -s KILL 120 "$prog" "$mode" > "$log" 2>&1 || rc=$?
    n="$(rg -c -e 'runtime error:' "$log" || true)"
    echo "$rc ${n:-0}"
}

# Print the functions of the first three frames of the first report of a
# log, as "#0 f0 | #1 f1 | #2 f2". A C++ name keeps its parameters.
report_frames() {
    local s
    s="$( { rg -o -r '#$1 $2' '^\s+#([0-9]+) 0x[0-9a-f]+ in (.+) (/\S+:[0-9]+(:[0-9]+)?|\(.+\))$' "$1" 2> /dev/null || true; } \
        | head -3 | tr '\n' '|')"
    s="${s%|}"
    echo "${s//|/ | }"
}

# Print the check name of an entry (the text before the colon).
check_of() {
    echo "${1%%:*}"
}

# Return 0 if the log has a report of the check.
log_has_check() {
    local log="$1" check="$2"
    case "$check" in
        pointer-overflow) rg -q -e 'runtime error: (applying (non-)?zero offset|.*pointer.*overflow|addition of unsigned offset)' "$log" ;;
        function) rg -q -e 'runtime error: call to function .* through pointer to incorrect function type' "$log" ;;
        *) rg -q -e 'runtime error:' "$log" ;;
    esac
}

# Run steps 3, 4 and 5 for one link. Append the records.
# Argument: the link. Return status: 1 if one record does not pass.
validate_link() {
    local link="$1" prog="$BUILD/ggml-$1/ubsan_ggml" logs="$BUILD/logs/$1" failed=0
    local entries=() entry mode line modes rc n st reason log safe hits tried marker check
    mkdir -p "$logs"
    mapfile -t entries < <(rg -v -e '^\s*#' -e '^\s*$' "$SUPP")
    declare -A mode_of=()
    while IFS= read -r line; do
        entry="$(rg -o -r '$1' 'REPRO-ENTRY: (\S+) MODE:' <<< "$line")"
        mode="$(rg -o -r '$1' 'MODE: (\S+)' <<< "$line")"
        mode_of["$entry"]="${mode_of[$entry]:-}${mode_of[$entry]:+ }$mode"
    done < <(rg -e 'REPRO-ENTRY: ' "$REPRO_SRC")

    # Step 3: each mode with the full file.
    mapfile -t modes < <(printf '%s\n' ${mode_of[@]} | sort -u)
    for mode in "${modes[@]}"; do
        read -r rc n < <(run_mode "$prog" "$mode" "$SUPP" "$logs/$mode.full.log")
        if [[ "$rc" -eq 0 && "$n" -eq 0 ]]; then
            st=pass; reason="no report with the full ubsan.supp"
        else
            st=fail; failed=1
            reason="with the full ubsan.supp: exit $rc, $n report(s). An entry is missing. Frames: $(report_frames "$logs/$mode.full.log")"
        fi
        suite_record "$RESULTS" supp-repro "$link/mode:$mode" "$PROFILE" ubsan test 0 1 "$n" "$st" "$reason" "$logs/$mode.full.log"
    done

    # Step 4: each entry, left out one at a time, in each of its modes.
    local tmp_supp="$BUILD/leave-one-out-$link.supp"
    for entry in "${entries[@]}"; do
        # An entry of repro/external.txt has the reproducer of its area.
        if rg -q -F -e "REPRO-ENTRY: $entry AREA:" "$EXTERNAL"; then
            continue
        fi
        if [[ -z "${mode_of[$entry]:-}" ]]; then
            suite_record "$RESULTS" supp-repro "$link/$entry" "$PROFILE" ubsan test 0 0 0 fail \
                "no REPRO-ENTRY line in $REPRO_SRC for this entry" ""
            failed=1
            continue
        fi
        rg -v -x -F -e "$entry" "$SUPP" > "$tmp_supp" || true
        safe="${entry//[^A-Za-z0-9_-]/_}"
        hits=""
        tried=""
        log=""
        for mode in ${mode_of[$entry]}; do
            read -r rc n < <(run_mode "$prog" "$mode" "$tmp_supp" "$logs/$mode.without.$safe.log")
            tried+=" $mode (exit $rc, $n report(s))"
            if [[ "$rc" -ne 0 ]] && log_has_check "$logs/$mode.without.$safe.log" "$(check_of "$entry")"; then
                hits+=" $mode"
                log="$logs/$mode.without.$safe.log"
            fi
        done
        if [[ -n "$hits" ]]; then
            st=pass
            reason="necessary: without it, these modes stop with a report:$hits. Frames: $(report_frames "$log")"
        else
            marker="$( { rg -B12 -x -F -e "$entry" "$SUPP" || true; } | { rg -o -e 'R13: (debug|release) only' || true; } | tail -1)"
            if [[ -n "$marker" && "$marker" != "R13: $PROFILE only" ]]; then
                st=pass
                reason="no match in $PROFILE, as its marker '$marker' says:$tried"
            else
                st=no-match; failed=1
                reason="without it, no mode stops:$tried. The entry does not match in the $PROFILE profile"
            fi
        fi
        suite_record "$RESULTS" supp-repro "$link/$entry" "$PROFILE" ubsan test 0 1 0 "$st" "$reason" "${log:-}"
    done
    rm -f "$tmp_supp"

    # Step 5: the controls must still stop the run.
    while IFS= read -r line; do
        check="$(rg -o -r '$1' 'REPRO-CONTROL: (\S+) MODE:' <<< "$line")"
        mode="$(rg -o -r '$1' 'MODE: (\S+)' <<< "$line")"
        read -r rc n < <(run_mode "$prog" "$mode" "$SUPP" "$logs/$mode.log")
        if [[ "$rc" -ne 0 ]] && log_has_check "$logs/$mode.log" "$check"; then
            st=pass; reason="the full ubsan.supp does not hide this $check report. Frames: $(report_frames "$logs/$mode.log")"
        else
            st=fail; failed=1
            reason="the full ubsan.supp hides a $check report in a function that no entry names (exit $rc, $n report(s))"
        fi
        suite_record "$RESULTS" supp-repro "$link/control:$check" "$PROFILE" ubsan test 0 1 "$n" "$st" "$reason" "$logs/$mode.log"
    done < <(rg -e 'REPRO-CONTROL: ' "$REPRO_SRC")
    return $failed
}

# Build the target of one fuzz area with the ubsan flags of ubsan.cmake, in
# build/fuzz/matrix-supp-repro-<profile>/<area>. The area CMake gets
# FUZZ_SANITIZER=none, thus the flags come only from this script, and
# FUZZ_PROFILE gives its profile flags. Only hexhost is known today.
# Arguments: the area, the target. Print the path of the program.
build_external() {
    local area="$1" target="$2" dir="$BUILD/$1" san fp stamp
    [[ "$area" == hexhost ]] || suite_die "repro/external.txt names the area '$area'. Only hexhost is known."
    san="$(rg -o -r '$1' '"(-fsanitize=undefined -fno-sanitize-recover=undefined[^"]*)"' "$SUITE_REPO_ROOT/tests/sanitizers/ubsan.cmake")"
    [[ -n "$san" ]] || suite_die "No ubsan compile flags found in tests/sanitizers/ubsan.cmake."
    fp="$(printf '%s\n' "$san" "$PROFILE" | sha256sum | cut -d' ' -f1)"
    stamp="$dir/.matrix-fingerprint"
    if [[ ! -f "$stamp" || "$(cat "$stamp")" != "$fp" ]]; then
        rm -rf "$dir"
        mkdir -p "$dir"
        CC=clang CXX=clang++ cmake -S "$SUITE_REPO_ROOT/tests/fuzz/hexhost" -B "$dir" -G Ninja \
            -DCMAKE_BUILD_TYPE=None -DFUZZ_PROFILE="$PROFILE" -DFUZZ_SANITIZER=none \
            -DHEXHOST_LLAMA_DIR="$LLAMA_SRC" -DCMAKE_C_FLAGS="$san" -DCMAKE_CXX_FLAGS="$san" \
            -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=undefined > "$dir/configure.log" 2>&1 \
            || { tail -20 "$dir/configure.log" >&2; suite_die "$PROFILE: the configure step of $area failed."; }
        echo "$fp" > "$stamp"
    fi
    nice -n 10 cmake --build "$dir" -j "$JOBS" --target "fuzz_$target" > "$dir/build.log" 2>&1 \
        || { tail -30 "$dir/build.log" >&2; suite_die "$PROFILE: the build of $area fuzz_$target failed."; }
    echo "$dir/fuzz_$target"
}

# Run the reproducer of one external entry with one suppression file.
# Arguments: the program, the input, the suppression file, the log.
# Print "<exit status> <number of reports>".
run_external() {
    local prog="$1" input="$2" supp="$3" log="$4" rc=0 n
    UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1:report_error_type=1:suppressions=$supp" \
        timeout -s KILL 120 "$prog" -runs=1 -artifact_prefix="$BUILD/artifacts/" "$SUITE_REPO_ROOT/$input" \
        > "$log" 2>&1 || rc=$?
    n="$(rg -c -e 'runtime error:' "$log" || true)"
    echo "$rc ${n:-0}"
}

# Validate the entries of repro/external.txt. The match criterion is the
# report line: without the entry the log has a report of its check, with
# the full file it has none. The exit status is not the criterion, because
# x86 traps with SIGFPE at a division by zero after the suppressed report.
# Return status: 1 if one entry does not match.
validate_external() {
    local line entry area target input prog rc n st reason failed=0 logs="$BUILD/logs/external"
    local tmp_supp="$BUILD/leave-one-out-external.supp" safe
    mkdir -p "$logs" "$BUILD/artifacts"
    while IFS= read -r line; do
        entry="$(rg -o -r '$1' 'REPRO-ENTRY: (\S+) AREA:' <<< "$line")"
        area="$(rg -o -r '$1' 'AREA: (\S+)' <<< "$line")"
        target="$(rg -o -r '$1' 'TARGET: (\S+)' <<< "$line")"
        input="$(rg -o -r '$1' 'INPUT: (\S+)' <<< "$line")"
        if ! rg -q -x -F -e "$entry" "$SUPP"; then
            continue
        fi
        prog="$(build_external "$area" "$target")"
        safe="${entry//[^A-Za-z0-9_-]/_}"
        read -r rc n < <(run_external "$prog" "$input" "$SUPP" "$logs/$safe.full.log")
        if [[ "$n" -ne 0 ]]; then
            st=fail; failed=1
            reason="with the full ubsan.supp the replay of $input has $n report(s). Frames: $(report_frames "$logs/$safe.full.log")"
            suite_record "$RESULTS" supp-repro "$area/$entry" "$PROFILE" ubsan test 0 1 "$n" "$st" "$reason" "$logs/$safe.full.log"
            continue
        fi
        rg -v -x -F -e "$entry" "$SUPP" > "$tmp_supp" || true
        read -r rc n < <(run_external "$prog" "$input" "$tmp_supp" "$logs/$safe.without.log")
        if log_has_check "$logs/$safe.without.log" "$(check_of "$entry")"; then
            st=pass
            reason="necessary: without it, the replay of $input reports. Frames: $(report_frames "$logs/$safe.without.log"). With it: no report (exit $rc without it)"
        else
            st=no-match; failed=1
            reason="without it, the replay of $input gives exit $rc and $n report(s) of the check: the entry does not match in the $PROFILE profile"
        fi
        suite_record "$RESULTS" supp-repro "$area/$entry" "$PROFILE" ubsan test 0 1 0 "$st" "$reason" "$logs/$safe.without.log"
    done < <(rg -e '^REPRO-ENTRY: ' "$EXTERNAL")
    rm -f "$tmp_supp"
    return $failed
}

main() {
    parse_args "$@"
    suite_require cmake ninja clang clang++ ld.lld jq rg
    export CCACHE_DISABLE=1
    BUILD="$SUITE_REPO_ROOT/build/fuzz/matrix-supp-repro-$PROFILE"
    RESULTS="$BUILD/results.jsonl"
    local link failed=0
    mkdir -p "$BUILD"
    rm -rf "$BUILD/logs"
    rm -f "$RESULTS"
    for link in $LINKS; do
        build_ggml "$link"
        build_repro "$link"
        validate_link "$link" || failed=1
    done
    validate_external || failed=1
    jq -r '[.status, .target, .reason] | @tsv' "$RESULTS" | while IFS=$'\t' read -r st t r; do
        printf '[supp-repro %s] %-9s %s: %s\n' "$PROFILE" "$st" "$t" "$r"
    done
    return $failed
}

main "$@"
