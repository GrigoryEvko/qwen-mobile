#!/usr/bin/env bash
# Check the sanitizer rules (R1 to R13) that a machine can check.
#
# Usage:
#   tests/sanitizers/check-rules.sh [--areas core,ops,hexhost,app,quant] [--no-builds]
#                                   [--write-requests] [--json FILE]
#   tests/sanitizers/check-rules.sh --copies-only [--areas LIST] [--build NAME]...
#                                   [--accept-stamp COMMIT:PATCHES]...
#   tests/sanitizers/check-rules.sh --commit-msg FILE
#   tests/sanitizers/check-rules.sh --noid-self-test
#
#   --areas LIST       The fuzz areas to check. The preset value is each
#                      directory of tests/fuzz. "none" checks only the
#                      shared files of tests/sanitizers (and the builds).
#   --no-builds        Do not check build/fuzz (for a clean CI checkout,
#                      before the builds).
#   --copies-only      Check only the rule LLAMA-COPY. The other rules do
#                      not include LLAMA-COPY.
#   --build NAME       With --copies-only: check only the build directory
#                      build/fuzz/NAME (for example core-debug-none), not
#                      each matrix build of the areas. The option can repeat.
#   --accept-stamp C:P Also accept the stamp of the llama.cpp commit C and
#                      the patches tree P (the value of HEAD when an area
#                      step started). The stamp of HEAD is always accepted.
#   --commit-msg FILE  Check only the commit message in FILE (NOID-MSG) and
#                      the staged diff (NOID-STAGED). This is the check of
#                      the commit-msg hook tests/sanitizers/git-hooks/commit-msg,
#                      and each landing runs it with its message file.
#   --noid-self-test   Test the pattern of NOID on its positive and negative
#                      cases, on the pushed history and on the tree.
#   --write-requests   Write the violations of each area to
#                      build/fuzz/matrix/requests/<area>-rules.txt.
#   --json FILE        Also write each violation as one JSON line to FILE.
#
# The checks:
#   R1   Each -fsanitize= list in tests/fuzz/<area> (CMakeLists.txt, *.cmake,
#        *.sh), in tests/sanitizers/*.cmake, and in each build/fuzz/*/
#        CMakeCache.txt and compile_commands.json has at most one sanitizer
#        family (fuzzer and fuzzer-no-link are not sanitizers). In a build
#        directory with a configuration in its name (<area>-<profile>-<config>),
#        the family is that configuration, and "none" has no family.
#   R3   Each tests/fuzz/<area>/run.sh exists, and its --help lists test,
#        fuzz, the five configurations, --budget-seconds, --jobs and --profile.
#   R3   Each line of each build/fuzz/*/results.jsonl has the schema, and its
#        sanitizer and profile agree with the directory name.
#   R5   A UBSan suppression file exists only as tests/sanitizers/ubsan.supp.
#        Each entry names a function (no file, no src:, no wildcard, and
#        anchored), and its comment names the defect in words (at least 8
#        words) and has a date. Each check of a -fsanitize-recover= flag has an
#        entry of that check.
#   R6   An ASan, LSan, TSan or MSan suppression file exists only as
#        tests/sanitizers/<config>.supp, with the same entry rules.
#   R5/R6 Each -fno-sanitize= and -fsanitize-recover= flag has a comment with
#        the word "reason", "evidence" or "defect" in the 15 lines before it.
#   R7   A tsan or msan build directory with GGML_OPENMP in its cache has
#        GGML_OPENMP=OFF. An msan build directory has FUZZ_MSAN_PREFIX, and
#        GGML_NATIVE=OFF and the x86 SIMD options OFF if it builds ggml.
#   R11  The build directories have a profile in the name, and the results
#        have a "profile" field.
#   R12  The shipped flags of tests/sanitizers/profile-release.cmake agree
#        with the preset (android/snapdragon/CMakeUserPresets.json) and with
#        build/native/llama/compile_commands.json. Each release build uses
#        those flags (x86: less -march) with -O3 -DNDEBUG, and each debug build
#        uses -O1, no -flto and no -DNDEBUG. The lab release build of the DSP
#        code uses each code generation flag of the shipped DSP library.
#        The linker (lld, mold or the default) and BUILD_SHARED_LIBS are not
#        part of the shipped flags, thus R12 does not check them.
#   LTO-PART No file of an area or of tests/sanitizers and no build cache has
#        --lto-partitions or -flto-partitions: the parallel code generation
#        of full LTO in lld and clang 22.1.8 can drop the dynamic initializer
#        of a C++17 inline variable, and the shipped build has one
#        partition. tests/sanitizers/repro/lto-partitions/run.sh shows the
#        defect, and the rule does not check that directory.
#   LLAMA-COPY (only with --copies-only) A matrix build of an area
#        (build/fuzz/<area>[-android]-<profile>-<config>) that takes
#        llama.cpp or ggml from a copy under build/, not from the submodule,
#        has the stamp .llama-copy-stamp of tests/sanitizers/llama-copy.sh in
#        the root of the copy (the directory that holds ggml/). The stamp has
#        the llama.cpp commit and the tree id of patches/ of HEAD. A copy with
#        an older stamp or with no stamp can hold old llama.cpp code.
#        tests/run-suite.sh runs this rule after each area step. The other
#        rules do not include it: each landing makes all copies old at once,
#        thus it would stop each next landing until each area refreshes.
#   R13  Each entry of tests/sanitizers/ubsan.supp has a reproducer in
#        tests/sanitizers/repro/. With the build directories (not
#        --no-builds): the last run of tests/sanitizers/supp-repro.sh passed
#        for the entry in the two profiles.
#   ASAN-RT No area takes the Android ASan runtime from an NDK, and each copy
#        of that runtime under build/fuzz has the sha256 of
#        tests/sanitizers/build-asan-android-runtime.sh.
#   NOID A tracked file in tests/, patches/, android/app/src/, quant/,
#        scripts/ or tools/ has no task number and no finding ID (a comment
#        names the defect in words). NOID_ALLOW below lists the lines of
#        real code that match the pattern. NOID_PATTERN gives the forms.
#   NOID-STAGED Each line that the staged diff adds, in each directory, has
#        no task number and no finding ID.
#   NOID-MSG (only with --commit-msg) The commit message has no task number
#        and no finding ID.
#   L9   Each libFuzzer command of the scripts of an area has -artifact_prefix,
#        each fuzz run has an outer "timeout -s KILL", and the root of the
#        repository has no stray crash-*, leak-*, timeout-* or oom-* file.
#   TSan Each libFuzzer target of an area includes tests/sanitizers/fuzz_death.h
#        and calls fuzz_death_note_input() (the death callback without the
#        deadlock of TSan and libFuzzer).
#
# Output: one line for each violation, "<rule> <area> <place>: <message>".
# Requirements: bash, jq, rg. No build and no container.
# Exit status: 0 if there is no violation, 1 if there is one or more.

set -euo pipefail

readonly REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly FUZZ_DIR="$REPO/tests/fuzz"
readonly SAN_DIR="$REPO/tests/sanitizers"
readonly BUILD_FUZZ="$REPO/build/fuzz"
readonly PRESET_FILE="$REPO/android/snapdragon/CMakeUserPresets.json"
readonly NATIVE_CC="$REPO/build/native/llama/compile_commands.json"
readonly CONFIG_TOKENS="none asan ubsan tsan msan hwasan"

AREAS=""
CHECK_BUILDS=1
COPIES_ONLY=0
COMMIT_MSG=""
COPY_BUILDS=""
NOID_SELF_TEST=0
ACCEPT_STAMPS=""
WRITE_REQUESTS=0
JSON_OUT=""
VIOLATIONS=()

# Write the usage text, from the header comment of this file.
print_usage() {
    local line
    while IFS= read -r line; do
        [[ "$line" == "#!"* ]] && continue
        [[ "$line" != "#"* ]] && break
        line="${line#\#}"
        echo "${line# }"
    done < "${BASH_SOURCE[0]}"
}

# Record one violation.
# Arguments: the rule, the area, the place (file:line or directory), the message.
violation() {
    VIOLATIONS+=("$1"$'\t'"$2"$'\t'"${3#"$REPO/"}"$'\t'"$4")
}

# Print the sanitizer family of one -fsanitize= item, or nothing for an item
# that is not a sanitizer (fuzzer, fuzzer-no-link).
family_of() {
    case "$1" in
        fuzzer|fuzzer-no-link) ;;
        address|kernel-address|pointer-compare|pointer-subtract) echo asan ;;
        hwaddress|kernel-hwaddress) echo hwasan ;;
        thread) echo tsan ;;
        memory|kernel-memory) echo msan ;;
        leak) echo lsan ;;
        undefined|alignment|bool|builtin|bounds|array-bounds|local-bounds|enum|float-cast-overflow| \
        float-divide-by-zero|function|integer|integer-divide-by-zero|nonnull-attribute|null| \
        nullability*|object-size|pointer-overflow|return|returns-nonnull-attribute|shift|shift-*| \
        signed-integer-overflow|unsigned-integer-overflow|unsigned-shift-base|unreachable|vla-bound| \
        vptr|implicit-*) echo ubsan ;;
        *) echo "other:$1" ;;
    esac
}

# Print the sorted, unique families of one comma list of -fsanitize= items.
families_of_list() {
    local item
    tr ',' '\n' <<< "$1" | while IFS= read -r item; do
        if [[ -n "$item" ]]; then family_of "$item"; fi
    done | sort -u
}

# Print the area of a path: tests/fuzz/<area>/..., build/fuzz/<area>-...,
# or "sanitizers" and "matrix" for the files of this agent.
area_of() {
    local rel="${1#"$REPO/"}"
    case "$rel" in
        tests/fuzz/*) rel="${rel#tests/fuzz/}"; echo "${rel%%/*}" ;;
        patches/fuzz-*) rel="${rel#patches/fuzz-}"; echo "${rel%%/*}" ;;
        quant/*) echo quant ;;
        android/*) echo app ;;
        tests/sanitizers/*|tests/suite/*|tests/run-suite.sh) echo sanitizers ;;
        build/fuzz/matrix*|build/fuzz/msan-libcxx*) echo matrix ;;
        build/fuzz/*) rel="${rel#build/fuzz/}"; rel="${rel%%/*}"; echo "${rel%%-*}" ;;
        *) echo other ;;
    esac
}

# Print the configuration token of a build directory name, or nothing.
config_of_dir() {
    local part found=""
    for part in ${1//-/ }; do
        [[ " $CONFIG_TOKENS " == *" $part "* ]] && found="$part"
    done
    echo "$found"
}

# Print the profile token of a build directory name, or nothing.
profile_of_dir() {
    local part found=""
    for part in ${1//-/ }; do
        [[ "$part" == debug || "$part" == release ]] && found="$part"
    done
    echo "$found"
}

# R1 in source files: each -fsanitize= list has one family at most.
check_sources_r1() {
    local file line num text list fams
    for file in "$@"; do
        [[ -f "$file" ]] || continue
        while IFS=: read -r num text; do
            while IFS= read -r list; do
                list="${list#-fsanitize=}"
                # A variable or a generator expression is checked in the build.
                [[ "$list" == *'$'* || "$list" == *'<'* ]] && continue
                fams="$(families_of_list "$list" | tr '\n' ' ')"
                if [[ "$(wc -w <<< "$fams")" -gt 1 ]]; then
                    violation R1 "$(area_of "$file")" "$file:$num" "-fsanitize=$list has more than one sanitizer family ($fams)"
                fi
            done < <(rg -o -e '-fsanitize=[A-Za-z0-9_,${}<>:.-]+' <<< "$text" || true)
        done < <(rg -n -e '-fsanitize=' "$file" || true)
    done
}

# R5 and R6: each -fno-sanitize= and -fsanitize-recover= flag has a reason
# in a comment of the 15 lines before it, or on its line.
check_reasons() {
    local file num start text
    for file in "$@"; do
        [[ -f "$file" ]] || continue
        while IFS=: read -r num text; do
            # A comment that names a flag is not a flag.
            [[ "$text" =~ ^[[:space:]]*(#|//) ]] && continue
            start=$(( num > 15 ? num - 15 : 1 ))
            if ! line_range "$file" "$start" "$num" | rg -q -i -e '(#|//).*(reason|evidence|defect)'; then
                violation R5 "$(area_of "$file")" "$file:$num" "-fno-sanitize= or -fsanitize-recover= has no comment with the word reason, evidence or defect in the 15 lines before it"
            fi
        done < <(rg -n -e '-fno-sanitize=' -e '-fsanitize-recover=' "$file" || true)
    done
}

# R5: each check in a -fsanitize-recover= list has at least one entry of that
# check in the shared file tests/sanitizers/ubsan.supp. A recover flag
# without an entry lets the reports of that check continue with no reason
# (for example after a fix removed the entry).
# The check reads each text file of each area and of tests/sanitizers (CMake,
# shell, Python, Gradle, JSON presets and the others), not only the build
# files, thus a recover flag in any place of an area is found. "all" and a
# sanitizer name (address, memory) are checks with no entry, thus flagged.
# A list with a variable ($, <) is checked in the build directories by R1.
check_recover_entries() {
    local file num text list check types area
    types="$( { rg -v -e '^\s*#' -e '^\s*$' "$SAN_DIR/ubsan.supp" || true; } | { rg -o -r '$1' '^([a-z-]+):' || true; } | sort -u | tr '\n' ' ')"
    local dirs=("$SAN_DIR")
    for area in $AREAS; do
        dirs+=("$FUZZ_DIR/$area")
    done
    while IFS=: read -r file num text; do
        [[ "$text" =~ ^[[:space:]]*(#|//|\*) ]] && continue
        while IFS= read -r list; do
            list="${list#-fsanitize-recover=}"
            [[ "$list" == *'$'* || "$list" == *'<'* ]] && continue
            for check in ${list//,/ }; do
                [[ " $types " == *" $check "* ]] \
                    || violation R5 "$(area_of "$file")" "$file:$num" "-fsanitize-recover=$check has no entry of that check in tests/sanitizers/ubsan.supp: remove the check from the recover list"
            done
        done < <(rg -o -e '-fsanitize-recover=[A-Za-z0-9_,$<>{}-]+' <<< "$text" || true)
    done < <(rg -n --no-heading -e '-fsanitize-recover=' "${dirs[@]}" \
                -g '!*.supp' -g '!corpus/**' -g '!regress/**' -g '!regressions/**' -g '!seeds/**' \
                -g '!check-rules.sh' 2> /dev/null || true)
}

# Print the lines FIRST thru LAST of a file.
line_range() {
    local file="$1" first="$2" last="$3" n=0 line
    while IFS= read -r line || [[ -n "$line" ]]; do
        n=$((n + 1))
        (( n < first )) && continue
        (( n > last )) && break
        printf '%s\n' "$line"
    done < "$file"
}

# R5 and R6: the suppression files and their entries.
check_suppressions() {
    local file base area
    while IFS= read -r file; do
        base="$(basename "$file")"
        area="$(area_of "$file")"
        case "$file" in
            "$SAN_DIR"/ubsan.supp|"$SAN_DIR"/asan.supp|"$SAN_DIR"/lsan.supp|"$SAN_DIR"/tsan.supp|"$SAN_DIR"/msan.supp) ;;
            *) violation R5 "$area" "$file" "a suppression file outside tests/sanitizers/<config>.supp: move each entry to the shared file of its sanitizer" ;;
        esac
        check_supp_entries "$file" "$area"
    done < <(find "$FUZZ_DIR" "$SAN_DIR" -name '*.supp' -type f 2> /dev/null | sort)
}

# Check each entry of one suppression file. An entry is "<type>:<pattern>".
# Its comment is the block of comment lines directly before it (other entries
# in the block are permitted).
check_supp_entries() {
    local file="$1" area="$2" n=0 line comment="" type pattern
    while IFS= read -r line || [[ -n "$line" ]]; do
        n=$((n + 1))
        if [[ -z "${line// /}" ]]; then
            comment=""
            continue
        fi
        if [[ "$line" == \#* ]]; then
            comment+="$line"$'\n'
            continue
        fi
        type="${line%%:*}"
        pattern="${line#*:}"
        if [[ "$type" == "src" || "$pattern" == */* || "$pattern" == *'*'* || "$pattern" =~ \.(c|cc|cpp|h|hpp)$ ]]; then
            violation R5 "$area" "$file:$n" "the entry '$line' is not at the function level (a file, src: or a wildcard)"
        elif [[ ! ( ( "$pattern" == ^* && ( "$pattern" == *'$' || "$pattern" == *'(' ) ) || "$pattern" == *'<' ) ]]; then
            # The runtime matches a pattern as a substring, thus an entry
            # without anchors also matches each longer name.
            violation R5 "$area" "$file:$n" "the entry '$line' is not anchored: use ^name\$ (C), ^name( (C++) or name< (C++ template), because the runtime matches a substring"
        fi
        # The comment names the defect in words: at least 8 words, the date
        # and the "#" signs not counted.
        if [[ "$(tr -d '#' <<< "$comment" | wc -w)" -lt 9 ]]; then
            violation R5 "$area" "$file:$n" "the entry '$line' has no comment that names the defect in words (the function, the input that causes it, the evidence)"
        fi
        if ! rg -q -e '20[0-9]{2}-[0-9]{2}-[0-9]{2}' <<< "$comment"; then
            violation R5 "$area" "$file:$n" "the entry '$line' has no date (YYYY-MM-DD) in its comment"
        fi
    done < "$file"
}

# R3: the interface of each run.sh.
check_run_sh() {
    local area script help word
    for area in $AREAS; do
        script="$FUZZ_DIR/$area/run.sh"
        if [[ ! -x "$script" ]]; then
            violation R3 "$area" "tests/fuzz/$area/run.sh" "run.sh is missing or not executable"
            continue
        fi
        if ! help="$(timeout 30 "$script" --help 2>&1)"; then
            violation R3 "$area" "tests/fuzz/$area/run.sh" "'run.sh --help' does not exit with status 0"
        fi
        for word in test fuzz none asan ubsan tsan msan --budget-seconds --jobs --profile; do
            rg -q -F -e "$word" <<< "$help" \
                || violation R3 "$area" "tests/fuzz/$area/run.sh" "--help does not list '$word'"
        done
    done
}

# R3 and R11: the schema of each results.jsonl.
check_results() {
    local file dir config profile bad
    for file in "$BUILD_FUZZ"/*/results.jsonl; do
        [[ -f "$file" ]] || continue
        dir="$(basename "$(dirname "$file")")"
        config="$(config_of_dir "$dir")"
        profile="$(profile_of_dir "$dir")"
        bad="$(jq -c --arg c "$config" --arg p "$profile" '
            select(
                (.area | type) != "string" or (.target | type) != "string" or
                ((.sanitizer | type) != "string") or
                (.mode != "test" and .mode != "fuzz") or
                (.seconds | type) != "number" or (.executions | type) != "number" or
                (.findings | type) != "number" or (.crash_files | type) != "array" or
                (.profile != "debug" and .profile != "release") or
                ($c != "" and .sanitizer != $c) or ($p != "" and .profile != $p)
            ) | {target, sanitizer, profile, mode}' "$file" 2>&1 | head -3 || true)"
        if ! jq -e . "$file" > /dev/null 2>&1; then
            violation R3 "$(area_of "$file")" "$file" "a line is not valid JSON"
        elif [[ -n "$bad" ]]; then
            violation R3 "$(area_of "$file")" "$file" "lines without the schema {area, target, profile (debug|release), sanitizer (= $config), mode, seconds, executions, findings, crash_files}, for example: ${bad//$'\n'/ }"
        fi
    done
}

# R1, R7, R11 and R12 in one build directory.
check_build_dir() {
    local dir="$1" name config profile cache cc fams flags area android=0
    name="$(basename "$dir")"
    area="$(area_of "$dir")"
    config="$(config_of_dir "$name")"
    profile="$(profile_of_dir "$name")"
    cache="$dir/CMakeCache.txt"
    cc="$dir/compile_commands.json"
    [[ -f "$cache" || -f "$cc" ]] || return 0
    [[ -f "$cache" ]] && rg -q '^CMAKE_SYSTEM_NAME:[A-Z]*=Android' "$cache" && android=1

    # R1: the families of all commands and of the cache flags.
    flags=""
    [[ -f "$cc" ]] && flags+="$(jq -r '.[] | (.command // (.arguments | join(" ")))' "$cc" 2> /dev/null || true)"$'\n'
    [[ -f "$cache" ]] && flags+="$(rg '^CMAKE_(C|CXX|EXE_LINKER|SHARED_LINKER|MODULE_LINKER)_FLAGS[A-Z_]*:STRING=' "$cache" || true)"
    fams="$({ rg -o -e '-fsanitize=[A-Za-z0-9_,-]+' <<< "$flags" || true; } \
        | while IFS= read -r l; do families_of_list "${l#-fsanitize=}"; done | sort -u | tr '\n' ' ')"
    fams="${fams% }"
    if [[ "$(wc -w <<< "$fams")" -gt 1 ]]; then
        violation R1 "$area" "$dir" "the build has more than one sanitizer family: $fams"
    elif [[ -n "$config" ]]; then
        if [[ "$config" == none && -n "$fams" ]]; then
            violation R1 "$area" "$dir" "the configuration none has the sanitizer $fams"
        elif [[ "$config" != none && "$fams" != "$config" ]]; then
            violation R1 "$area" "$dir" "the configuration $config has the sanitizer family '${fams:-none}'"
        fi
    fi
    # R11: the profile in the name.
    [[ -z "$profile" && -n "$config" ]] \
        && violation R11 "$area" "$dir" "the directory name has no profile (build/fuzz/<area>-<debug|release>-<config>)"

    # R7: the ggml options of tsan and msan.
    if [[ -f "$cache" && ( "$config" == tsan || "$config" == msan ) ]]; then
        if rg -q '^GGML_OPENMP:BOOL=' "$cache" && ! rg -q '^GGML_OPENMP:BOOL=OFF' "$cache"; then
            violation R7 "$area" "$dir" "GGML_OPENMP is not OFF"
        fi
    fi
    if [[ -f "$cache" && "$config" == msan ]]; then
        rg -q '^FUZZ_MSAN_PREFIX:[A-Z]*=.+' "$cache" || violation R7 "$area" "$dir" "FUZZ_MSAN_PREFIX is not set"
        if rg -q '^GGML_NATIVE:BOOL=' "$cache"; then
            local opt
            for opt in NATIVE AVX AVX2 AVX512 AVX_VNNI FMA F16C BMI2 SSE42; do
                if rg -q "^GGML_$opt:BOOL=" "$cache" && ! rg -q "^GGML_$opt:BOOL=OFF" "$cache"; then
                    violation R7 "$area" "$dir" "GGML_$opt is not OFF"
                fi
            done
        fi
    fi

    # R12: the flags of the profile, in the C and C++ compile commands.
    [[ -n "$profile" && -f "$cc" ]] || return 0
    check_profile_flags "$dir" "$area" "$profile" "$config" "$android" "$cc"
}

# R12 for one build directory. It reads the compile commands of the .c and
# .cpp files, and checks the last -O, NDEBUG, -flto and the profile flags.
check_profile_flags() {
    local dir="$1" area="$2" profile="$3" config="$4" android="$5" cc="$6"
    local shipped=() want=() tok cmds last_o bad_tok="" want_o=" -O1"
    read -r -a shipped <<< "$(shipped_flags)"
    for tok in "${shipped[@]}"; do
        [[ $android -eq 0 && "$tok" == -march=* ]] && continue
        [[ "$profile" == debug && ( "$tok" == -flto || "$tok" == -O3 || "$tok" == -DNDEBUG ) ]] && continue
        want+=("$tok")
    done
    cmds="$(jq -r '.[] | select(.file | test("\\.(c|cc|cpp)$")) | (.command // (.arguments | join(" ")))' "$cc" 2> /dev/null || true)"
    [[ -n "$cmds" ]] || return 0
    for tok in "${want[@]}"; do
        if rg -v -q -F -e " $tok" <<< "$cmds"; then
            bad_tok+=" $tok"
        fi
    done
    [[ -n "$bad_tok" ]] && violation R12 "$area" "$dir" "some compile commands do not have the $profile flags:$bad_tok"
    # The last -O of each command decides the optimization level.
    [[ "$profile" == release ]] && want_o=" -O3"
    last_o="$(jq -r '.[] | select(.file | test("\\.(c|cc|cpp)$")) | (.command // (.arguments | join(" ")))
                     | ([scan(" -O[0-3sz](?= |$)")] | last // "none")' "$cc" | sort | uniq -c | tr -s ' ' | tr '\n' ',')"
    if [[ "$last_o" != *"$want_o," || "$(tr ',' '\n' <<< "$last_o" | rg -c -v -e '^$')" -ne 1 ]]; then
        violation R12 "$area" "$dir" "the $profile profile needs$want_o as the last -O of each command. The counts are: ${last_o%,}"
    fi
    if [[ "$profile" == debug ]]; then
        rg -q -e ' -flto' <<< "$cmds" && violation R12 "$area" "$dir" "the debug profile has -flto"
        rg -q -e ' -DNDEBUG' <<< "$cmds" && violation R12 "$area" "$dir" "the debug profile has -DNDEBUG, thus assert() does not run"
    fi
    return 0
}

# Print the shipped flags of tests/sanitizers/profile-release.cmake, with
# the release flags (-O3 -DNDEBUG).
shipped_flags() {
    local a b
    a="$(rg -o -r '$1' 'SANMATRIX_SHIPPED_FLAGS "([^"]*)"' "$SAN_DIR/profile-release.cmake")"
    b="$(rg -o -r '$1' 'SANMATRIX_SHIPPED_RELEASE_FLAGS "([^"]*)"' "$SAN_DIR/profile-release.cmake")"
    echo "$a $b"
}

# R12: the shipped flags of profile-release.cmake against the preset and
# against the compile commands of the shipped build.
check_parity_sources() {
    local ours preset preset_rel profile_flags tok native_cmd lab_cc v shipped_cmd missing
    ours="$(rg -o -r '$1' 'SANMATRIX_SHIPPED_FLAGS "([^"]*)"' "$SAN_DIR/profile-release.cmake")"
    if [[ -f "$PRESET_FILE" ]]; then
        preset="$(jq -r '.configurePresets[] | select(.name == "arm64-android-snapdragon") | .cacheVariables.CMAKE_C_FLAGS' "$PRESET_FILE")"
        preset_rel="$(jq -r '.configurePresets[] | select(.name == "arm64-android-snapdragon") | .cacheVariables.CMAKE_C_FLAGS_RELEASE' "$PRESET_FILE")"
        [[ "$preset" == "$ours" ]] \
            || violation R12 sanitizers "tests/sanitizers/profile-release.cmake" "SANMATRIX_SHIPPED_FLAGS '$ours' differs from the preset CMAKE_C_FLAGS '$preset'"
        [[ "$(jq -r '.configurePresets[] | select(.name == "arm64-android-snapdragon") | .cacheVariables.CMAKE_CXX_FLAGS' "$PRESET_FILE")" == "$preset" ]] \
            || violation R12 sanitizers "$PRESET_FILE" "the preset CMAKE_CXX_FLAGS differs from CMAKE_C_FLAGS"
        [[ "$preset_rel" == "$(rg -o -r '$1' 'SANMATRIX_SHIPPED_RELEASE_FLAGS "([^"]*)"' "$SAN_DIR/profile-release.cmake")" ]] \
            || violation R12 sanitizers "tests/sanitizers/profile-release.cmake" "SANMATRIX_SHIPPED_RELEASE_FLAGS differs from the preset CMAKE_C_FLAGS_RELEASE '$preset_rel'"
    else
        violation R12 sanitizers "$PRESET_FILE" "the preset file is missing"
    fi
    # The profile flags are the shipped flags less -march.
    profile_flags="$(rg -o -r '$1' 'SANMATRIX_PROFILE_FLAGS "([^"]*)"' "$SAN_DIR/profile-release.cmake")"
    [[ "$profile_flags" == "$(tr ' ' '\n' <<< "$ours" | rg -v '^-march=' | tr '\n' ' ' | xargs)" ]] \
        || violation R12 sanitizers "tests/sanitizers/profile-release.cmake" "SANMATRIX_PROFILE_FLAGS '$profile_flags' is not the shipped flags less -march"
    # The shipped build itself.
    if [[ -f "$NATIVE_CC" ]]; then
        native_cmd="$(jq -r '.[] | select(.file | test("ggml-cpu/ggml-cpu\\.c$")) | .command' "$NATIVE_CC" | head -1)"
        missing=""
        for tok in $ours -O3 -DNDEBUG; do
            [[ " $native_cmd " == *" $tok "* ]] || missing+=" $tok"
        done
        [[ -z "$missing" ]] || violation R12 sanitizers "build/native/llama/compile_commands.json" "the shipped ggml-cpu.c command does not have:$missing (the preset and the shipped build differ)"
    fi
    # The DSP profile of the lab against the shipped DSP libraries.
    for v in v73 v75 v79 v81; do
        local shipped_cc="$REPO/build/native/llama/ggml/src/ggml-hexagon/htp-$v-prefix/src/htp-$v-build/compile_commands.json"
        lab_cc="$BUILD_FUZZ/matrix-lab-release/build-$v-release/compile_commands.json"
        [[ -f "$shipped_cc" && -f "$lab_cc" ]] || continue
        shipped_cmd="$(jq -r '.[] | select(.file | test("htp/matmul-ops\\.c$")) | .command' "$shipped_cc" | head -1)"
        local lab_cmd
        lab_cmd="$(jq -r '.[0].command' "$lab_cc")"
        missing=""
        # The code generation flags: each option that is not an include, a
        # define other than NDEBUG, a warning or a path map.
        for tok in $(tr ' ' '\n' <<< "$shipped_cmd" | rg -e '^-(m|f|O|G|DNDEBUG|enable-)' | rg -v -e '^-f(file|debug)-prefix-map' -e '^-W'); do
            [[ " $lab_cmd " == *" $tok "* ]] || missing+=" $tok"
        done
        [[ -z "$missing" ]] || violation R12 sanitizers "build/fuzz/matrix-lab-release/build-$v-release" "the lab release build of the DSP code does not have the shipped flags:$missing"
    done
}

# R13: each ubsan.supp entry has a reproducer, and the last runs passed.
check_repro() {
    local entry func profile res
    while IFS= read -r entry; do
        func="${entry#*:}"
        if ! rg -q -F -e "REPRO-ENTRY: $entry" "$SAN_DIR/repro/" 2> /dev/null; then
            violation R13 sanitizers "tests/sanitizers/ubsan.supp" "the entry '$entry' has no reproducer (a file in tests/sanitizers/repro/ with the line 'REPRO-ENTRY: $entry')"
        fi
        # The results come from the suite (supp-repro.sh). --no-builds (the
        # check before the suite, and a clean CI checkout) has none yet.
        [[ $CHECK_BUILDS -eq 1 ]] || continue
        for profile in debug release; do
            res="$BUILD_FUZZ/matrix-supp-repro-$profile/results.jsonl"
            if [[ ! -f "$res" ]]; then
                violation R13 sanitizers "$res" "no run of tests/sanitizers/supp-repro.sh --profile $profile"
                continue
            fi
            local link links="shared static" ext_area
            # An entry of repro/external.txt has the reproducer of its area,
            # and its record is "<area>/<entry>".
            if rg -q -F -e "REPRO-ENTRY: $entry AREA: " "$SAN_DIR/repro/external.txt" 2> /dev/null; then
                ext_area="$(rg -F -e "REPRO-ENTRY: $entry AREA: " "$SAN_DIR/repro/external.txt" | rg -o -r '$1' 'AREA: (\S+)' | head -1)"
                links="$ext_area"
            fi
            for link in $links; do
                jq -e --arg e "$link/$entry" 'select(.target == $e and .status == "pass")' "$res" > /dev/null 2>&1 \
                    || violation R13 sanitizers "$res" "the entry '$entry' did not pass its reproducer in the $profile profile ($link)"
            done
        done
    done < <(rg -v -e '^\s*#' -e '^\s*$' "$SAN_DIR/ubsan.supp" || true)
}

# Write the violations of each area to its request file.
write_requests() {
    local dir="$BUILD_FUZZ/matrix/requests" area v file
    mkdir -p "$dir"
    for area in $AREAS; do
        file="$dir/$area-rules.txt"
        {
            echo "$(date -u +%Y-%m-%d), from sanitizer-matrix: tests/sanitizers/check-rules.sh found these violations in the area $area."
            echo "Format: <rule> <place>: <message>. The rules are in the message of the coordinator (R1 to R13)."
            echo
            for v in "${VIOLATIONS[@]}"; do
                IFS=$'\t' read -r r a p m <<< "$v"
                if [[ "$a" == "$area" ]]; then echo "$r $p: $m"; fi
            done
        } > "$file"
        if [[ "$(wc -l < "$file")" -le 3 ]]; then
            rm -f "$file"
        fi
    done
}

# L9 and the TSan hang: each libFuzzer command in the scripts of an area has
# -artifact_prefix, and each fuzz run (-max_total_time=) has an outer
# "timeout -s KILL". A command can go on more than one line with "\".
check_libfuzzer_commands() {
    local file area n start line logical
    for area in $AREAS; do
        while IFS= read -r file; do
            n=0
            logical=""
            start=0
            while IFS= read -r line || [[ -n "$line" ]]; do
                n=$((n + 1))
                [[ -z "$logical" ]] && start=$n
                if [[ "$line" == *'\' ]]; then
                    logical+="${line%\\} "
                    continue
                fi
                logical+="$line"
                if [[ ! "$logical" =~ ^[[:space:]]*# ]] \
                    && [[ "$logical" == *-max_total_time=* || "$logical" == *-runs=* || "$logical" == *-rss_limit_mb=* ]]; then
                    [[ "$logical" == *-artifact_prefix* ]] \
                        || violation L9 "$area" "$file:$start" "a libFuzzer command without -artifact_prefix (crash files go to the working directory)"
                    if [[ "$logical" == *-max_total_time=* && "$logical" != *"timeout -s KILL"* \
                          && "$logical" != *"timeout --signal=KILL"* && "$logical" != *"timeout -s 9"* ]]; then
                        violation L9 "$area" "$file:$start" "a fuzz run without an outer 'timeout -s KILL' (a TSan hang then blocks the job)"
                    fi
                fi
                logical=""
            done < "$file"
        done < <(find "$FUZZ_DIR/$area" -name '*.sh' -type f 2> /dev/null | sort)
    done
    # Stray libFuzzer files in the root of the repository.
    local stray
    while IFS= read -r stray; do
        violation L9 other "$stray" "a stray libFuzzer file in the root of the repository: move it to the regress/ directory of its area"
    done < <(find "$REPO" -maxdepth 1 -type f \( -name 'crash-*' -o -name 'leak-*' -o -name 'timeout-*' -o -name 'oom-*' -o -name 'slow-unit-*' \) 2> /dev/null | sort)
    # Any untracked file in the root (git status entries with no slash), for
    # example the dump_state.bin of a test that ran with the root as its
    # working directory.
    if command -v git > /dev/null && git -C "$REPO" rev-parse --git-dir > /dev/null 2>&1; then
        while IFS= read -r stray; do
            violation L9 other "$stray" "an untracked file in the root of the repository: a test or a fuzzer wrote it there. Remove it, and give the program a working directory under build/"
        done < <(git -C "$REPO" status --porcelain --untracked-files=normal 2> /dev/null \
                 | { rg -o -r '$1' '^\?\? ([^/]+)$' || true; })
    fi
}

# The TSan death callback: each libFuzzer target of an area includes
# tests/sanitizers/fuzz_death.h (directly or through a header of its area)
# and calls fuzz_death_note_input (directly or through a wrapper of that
# header). Each area builds tsan, thus the rule is for each target.
check_death_callback() {
    local area file text inc
    for area in $AREAS; do
        while IFS= read -r file; do
            text="$(cat "$file")"
            # The local headers that the file includes, one level deep.
            while IFS= read -r inc; do
                [[ -f "$(dirname "$file")/$inc" ]] && text+=$'\n'"$(cat "$(dirname "$file")/$inc")"
                [[ -f "$FUZZ_DIR/$area/$inc" ]] && text+=$'\n'"$(cat "$FUZZ_DIR/$area/$inc")"
            done < <(rg -o -r '$1' '^\s*#\s*include\s+"([^"]+)"' "$file" || true)
            if ! rg -q -e '#\s*include\s+[<"]([^">]*/)?fuzz_death\.h[">]' <<< "$text"; then
                violation R1 "$area" "$file" "a libFuzzer target without tests/sanitizers/fuzz_death.h (the TSan death callback without a deadlock)"
            elif ! rg -q -e 'fuzz_death_note_input\(' <<< "$text"; then
                violation R1 "$area" "$file" "a libFuzzer target that does not call fuzz_death_note_input()"
            fi
        done < <(rg -l -e 'LLVMFuzzerTestOneInput' "$FUZZ_DIR/$area" -g '*.c' -g '*.cc' -g '*.cpp' 2> /dev/null | sort)
    done
}

# ASAN-RT: each Android ASan run uses the runtime of
# tests/sanitizers/build-asan-android-runtime.sh (compiler-rt 22.1.8), not
# the runtime of the NDK r29, whose prctl interceptor signs its return address
# and traps on a core with FEAT_FPAC when bionic resets the PAC key.
#   - No script or CMake file of an area copies the ASan runtime of an NDK.
#   - With the build directories: each copy of
#     libclang_rt.asan-aarch64-android.so in an Android ASan build or phone
#     directory has the sha256 of
#     build/fuzz/asan-android-runtime/libclang_rt.asan-aarch64-android.so.sha256.
# The order of LD_LIBRARY_PATH on the phone is not a static fact of a file,
# thus the phone commands of each area give it, and this script does not
# check it.
check_android_asan_runtime() {
    local area file num text rt="libclang_rt.asan-aarch64-android.so"
    local ref_file="$BUILD_FUZZ/asan-android-runtime/$rt.sha256" ref copy sum
    for area in $AREAS; do
        while IFS= read -r file; do
            while IFS=: read -r num text; do
                [[ "$text" =~ ^[[:space:]]*(#|//) ]] && continue
                if [[ "$text" == *android-ndk* || "$text" == *toolchains/llvm/prebuilt* || "$text" == *'$ANDROID_NDK'* || "$text" == *'${ANDROID_NDK'* ]]; then
                    violation ASAN-RT "$area" "$file:$num" "this line takes $rt from an NDK: use build/fuzz/asan-android-runtime/$rt of tests/sanitizers/build-asan-android-runtime.sh"
                fi
            done < <(rg -n -F -e "$rt" "$file" || true)
        done < <(find "$FUZZ_DIR/$area" \( -name '*.sh' -o -name CMakeLists.txt -o -name '*.cmake' \) -type f 2> /dev/null | sort)
    done
    [[ $CHECK_BUILDS -eq 1 && -f "$ref_file" ]] || return 0
    ref="$(cut -d' ' -f1 "$ref_file")"
    while IFS= read -r copy; do
        # Only the copies in the build or phone directory of an Android ASan
        # configuration are the runtimes of runs. A tool or source tree of
        # an area is not.
        [[ "${copy#"$BUILD_FUZZ"/}" =~ (^|/)([a-z]+-android-(debug|release)-asan|[a-z]+-(debug|release)-asan|phone-(debug|release)-asan)(/|$) ]] || continue
        sum="$(sha256sum "$copy" | cut -d' ' -f1)"
        [[ "$sum" == "$ref" ]] \
            || violation ASAN-RT "$(area_of "$copy")" "$copy" "this ASan runtime has sha256 ${sum:0:16}, not ${ref:0:16} of tests/sanitizers/build-asan-android-runtime.sh"
    done < <(find "$BUILD_FUZZ" -name "$rt" -type f \
                -not -path "$BUILD_FUZZ/asan-android-runtime/*" 2> /dev/null | sort)
}

# LTO-PART: no LTO partition option in the files of the areas, in
# tests/sanitizers and in the CMake caches of build/fuzz.
check_lto_partitions() {
    local dirs=("$SAN_DIR") area file num text
    for area in $AREAS; do
        dirs+=("$FUZZ_DIR/$area")
    done
    while IFS=: read -r file num text; do
        [[ "$file" == */check-rules.sh ]] && continue
        [[ "$text" =~ ^[[:space:]]*(#|//) ]] && continue
        violation LTO-PART "$(area_of "$file")" "$file:$num" "an LTO partition option: full LTO with more than one partition can drop the dynamic initializer of an inline variable; use one partition, as the shipped build"
    done < <(rg -n --no-heading -e '-lto-partitions' "${dirs[@]}" -g '!corpus/**' -g '!regress/**' -g '!seeds/**' \
                -g '!**/tests/sanitizers/repro/lto-partitions/**' 2> /dev/null || true)
    [[ $CHECK_BUILDS -eq 1 && -d "$BUILD_FUZZ" ]] || return 0
    while IFS=: read -r file num text; do
        violation LTO-PART "$(area_of "$file")" "$file:$num" "a build with an LTO partition option: configure it again with one partition"
    done < <(find "$BUILD_FUZZ" -maxdepth 3 -name CMakeCache.txt -print0 2> /dev/null \
                | xargs -0 -r rg -n --no-heading -e '^CMAKE_[A-Z_]*LINKER_FLAGS[A-Z_]*:STRING=.*-lto-partitions' || true)
}

# LLAMA-COPY: each copy of llama.cpp or ggml under build/ that a matrix build
# of an area names in its CMake cache has the stamp of HEAD (the llama.cpp
# commit and the tree id of patches/), or a stamp of ACCEPT_STAMPS. A build
# that names the submodule is not checked. Each copy gets one report, with
# the first build that names it.
# Complexity: one pass over the cache of each matrix build directory.
check_llama_copies() {
    [[ -d "$BUILD_FUZZ" ]] || return 0
    command -v git > /dev/null && git -C "$REPO" rev-parse --git-dir > /dev/null 2>&1 || return 0
    local head_stamp dir name area cache value root stamp got build_real
    local -A seen=()
    # The caches hold physical paths, and the path of the repository can
    # hold a symbolic link.
    build_real="$(cd "$REPO/build" && pwd -P)"
    head_stamp="$(git -C "$REPO" rev-parse HEAD:third_party/llama.cpp):$(git -C "$REPO" rev-parse HEAD:patches)"
    for dir in "$BUILD_FUZZ"/*/; do
        dir="${dir%/}"
        name="$(basename "$dir")"
        area="$(area_of "$dir")"
        [[ " $AREAS " == *" $area "* ]] || continue
        # Only the matrix builds. A name with a tag after the configuration is
        # the private build of one check, and its tree can differ on purpose.
        [[ "$name" =~ ^[a-z]+(-android)?-(debug|release)-(none|asan|ubsan|tsan|msan|hwasan)$ ]] || continue
        # --build: only the build directories of one area step.
        [[ -z "$COPY_BUILDS" || " $COPY_BUILDS " == *" $name "* ]] || continue
        for cache in "$dir/CMakeCache.txt" "$dir/build/CMakeCache.txt"; do
            [[ -f "$cache" ]] || continue
            while IFS= read -r value; do
                # A container build names the repository /workspace.
                [[ "$value" == /workspace/build/* ]] && value="$build_real/${value#/workspace/build/}"
                [[ "$value" == "$build_real/"* || "$value" == "$REPO/build/"* ]] || continue
                value="${value%/}"
                # The root of the copy is the directory that holds ggml/.
                if [[ -f "$value/ggml/src/ggml.c" ]]; then
                    root="$value"
                elif [[ -f "$value/src/ggml.c" && "$(basename "$value")" == ggml ]]; then
                    root="$(dirname "$value")"
                else
                    continue
                fi
                [[ -z "${seen[$root]:-}" ]] || continue
                seen[$root]=1
                stamp="$root/.llama-copy-stamp"
                if [[ ! -f "$stamp" ]]; then
                    violation LLAMA-COPY "$area" "$root" "the build $name takes llama.cpp from this copy, and the copy has no .llama-copy-stamp: make the copy with tests/sanitizers/llama-copy.sh"
                    continue
                fi
                got="$(rg -o -r '$1' '^commit ([0-9a-f]{40})$' "$stamp" || true):$(rg -o -r '$1' '^patches ([0-9a-f]{40})$' "$stamp" || true)"
                if [[ "$got" != "$head_stamp" && " $ACCEPT_STAMPS " != *" $got "* ]]; then
                    violation LLAMA-COPY "$area" "$root" "the build $name takes llama.cpp from this copy, which has the stamp ${got:0:12}:${got:41:12}, but HEAD has ${head_stamp:0:12}:${head_stamp:41:12} (llama.cpp commit:patches tree): make the copy again with tests/sanitizers/llama-copy.sh, then build again"
                fi
            done < <(rg -o -r '$1' '^[A-Za-z0-9_.+-]+:[A-Z]+=(/[^;]*)$' "$cache" || true)
        done
    done
}

# NOID: the forms of a task number and of a finding ID. A comment, a commit
# message or a name describes the defect in words (the function, the input,
# the evidence), never with an ID. The forms:
#   - "#" and 2 or 3 digits, not after a letter, a digit, "_", "/", ".",
#     "&", "#" or "-" (thus not a URL fragment and not an HTML entity)
#   - "task" or "tasks" and a number, with or without "#"
#   - QF, QG, QR or QT and a number (the IDs of the quant findings)
#   - F-<WORD>-<number>, for example F-UB-4 (the IDs of the fuzz findings)
#   - "finding" or "findings" and an uppercase ID or a hyphenated name
#   - "item" or "items" and an uppercase letter with a number, for example
#     item D9
# #include, #define, #pragma, a hex colour with a letter or with more than 3
# digits, and an upstream pull request number with 4 or more digits do not
# match. The rules R1 to R13 and L1 to L9, F16, L2 and HTP0 do not match.
readonly NOID_PATTERN='((^|[^A-Za-z0-9_/.&#-])#[0-9]{2,3}\b|\b[Tt]asks? #?[0-9]{2,4}\b|\bQ[FGRT][0-9]{1,2}\b|\bF-[A-Z0-9]+(-[A-Z0-9]+)*-[0-9]+\b|\b[Ff]indings? [A-Z]+[0-9-]*[0-9]\b|\b[Ff]indings? [a-z][a-z0-9]*(-[a-z0-9]+)+\b|\bitems? [A-Z][0-9]{1,2}\b)'
# The pushed history when the forms above were written, and the count of its
# commit message lines that have an ID. --noid-self-test uses them as the
# positive case. A change of NOID_PATTERN that changes the count is
# deliberate, thus it changes this count too.
readonly NOID_HISTORY_REF="4c4f59ae2d4333371647a79ab00b8df1c75af0d3"
readonly NOID_HISTORY_LINES=112
# The lines of real code that match NOID_PATTERN. Format: <path>:<fixed text
# of the line>. Each entry has the reason as a comment.
readonly NOID_ALLOW=(
    # An assembly immediate of Hexagon ("r7 = #64").
    'tools/htp-lab/lab/target_hvxcost.c:#64'
    # A log line of the patch: "op #17 of 96".
    'patches/hexagon-fusion/0004-hexagon-htp-gdn-slot-index-from-host-and-error-report.patch:op #'
    # An assembly immediate of Hexagon: "pause(#255)".
    'patches/hexagon-fusion/0005-hexagon-htp-gdn-chunked-hmx.patch:pause(#255)'
    # A log line of the patch: "decode #12 tokens".
    'patches/hexagon-host/0002:decode #'
)
readonly NOID_DIRS="tests patches android/app/src quant scripts tools"

# Return 0 if one line of NOID_ALLOW permits this match.
# Arguments: the path (relative to the repository), the text of the line.
noid_allowed() {
    local entry path text
    for entry in "${NOID_ALLOW[@]}"; do
        path="${entry%%:*}"
        text="${entry#*:}"
        [[ "$1" == "$path"* && "$2" == *"$text"* ]] && return 0
    done
    return 1
}

# NOID: each tracked file of NOID_DIRS has no task number and no finding ID.
# Complexity: one rg pass over the tracked files of the six directories.
check_noid() {
    command -v git > /dev/null && git -C "$REPO" rev-parse --git-dir > /dev/null 2>&1 || return 0
    local file num text
    while IFS=: read -r file num text; do
        # This file holds the pattern and the allowlist themselves.
        [[ "$file" == tests/sanitizers/check-rules.sh ]] && continue
        noid_allowed "$file" "$text" && continue
        violation NOID "$(area_of "$REPO/$file")" "$file:$num" "a task number or a finding ID: describe the defect in words ('${text:0:90}')"
    done < <(cd "$REPO" && git ls-files -z -- $NOID_DIRS \
                | xargs -0 -r rg -n --no-heading -e "$NOID_PATTERN" -- 2> /dev/null || true)
}

# NOID-STAGED: each line that the staged diff (the index against HEAD) adds
# has no task number and no finding ID, in each directory. In a hook of
# "git commit", the index is the content of the commit, also for
# "git commit -- <paths>". The added lines go to one rg call as
# "<text> US <path> US <line>" (US is the byte 0x1f), thus the text keeps its
# start of line for the pattern.
# Complexity: one pass over the staged diff.
check_noid_staged() {
    command -v git > /dev/null && git -C "$REPO" rev-parse --verify -q HEAD > /dev/null 2>&1 || return 0
    local us=$'\x1f' line file="" num=0 added="" text path header=0
    while IFS= read -r line; do
        # A "+++" line is a file header only between "diff --git" and the
        # first hunk, not an added line that starts with "++".
        case "$header:$line" in
            *:'diff --git '*) header=1; file="" ;;
            '1:+++ b/'*) file="${line#+++ b/}" ;;
            '1:+++ '*) file="" ;;
            *:'@@ '*) header=0; num="${line#*+}"; num="${num%%[, ]*}" ;;
            '0:+'*)
                # This file holds the pattern and its examples.
                [[ -n "$file" && "$file" != tests/sanitizers/check-rules.sh ]] \
                    && added+="${line#+}$us$file$us$num"$'\n'
                num=$((num + 1)) ;;
        esac
    done < <(git -C "$REPO" diff --cached --no-color --no-ext-diff --no-renames -U0 2> /dev/null || true)
    [[ -n "$added" ]] || return 0
    while IFS="$us" read -r text path num; do
        noid_allowed "$path" "$text" && continue
        violation NOID-STAGED "$(area_of "$REPO/$path")" "$path:$num" "the staged diff adds a task number or a finding ID: describe the defect in words ('${text:0:90}')"
    done < <(rg --no-heading -e "$NOID_PATTERN" <<< "$added" || true)
}

# NOID-MSG: the commit message in the file $1 has no task number and no
# finding ID. The comment lines of git ("# Please enter ...") do not match.
check_noid_msg() {
    local msg="$1" num text
    [[ -f "$msg" ]] || { echo "check-rules: the commit message file '$msg' does not exist." >&2; exit 2; }
    while IFS=: read -r num text; do
        violation NOID-MSG commit "commit message:$num" "a task number or a finding ID: describe the defect in words ('${text:0:90}')"
    done < <(rg -n --no-heading -e "$NOID_PATTERN" -- "$msg" || true)
}

# The self-test of NOID_PATTERN (--noid-self-test). The positive cases: each
# form, and the pushed history of NOID_HISTORY_REF (if the clone has it).
# The negative cases: the lines that look like an ID but are not, and the
# tracked files of the tree (rule NOID). Output: one line for each case that
# fails. Return status: 0 if all cases pass.
noid_self_test() {
    local failed=0 s n
    local -a positive=(
        "the fix of task #168" "(#173)" "#125, #126 and #127" "Tasks 167 and 168"
        "finding QF6" "QR3 in gguf-py" "QG1" "QT1 in llama-perplexity" "F-UB-4 of this area"
        "F-REPACK-1" "Finding sampler-nan" "findings conversation-all-or-nothing"
        "Task 94, item D9" "fixes finding mm-id-div-zero (task"
    )
    local -a negative=(
        "#include <stdio.h>" "#define N 12" "#pragma once" "#if 0" "color: #fff;" "color: #a1b2c3;"
        "color: #123456;" "https://github.com/ggml-org/llama.cpp/blob/master/README.md#L120"
        "https://example.org/page#section-12" "https://example.org/a/#12" "&#123;"
        "ci: switch fast jobs back to github (#28959)" "the L2 line" "R13 and L9" "Q8_0 and Q4_K"
        "F16 and FP16" "HTP0" "{D0, D2}" "QK1_0" "a finding of UBSan" "the findings of the run"
        "r7 = #64" "task-parallel" "F-16"
    )
    for s in "${positive[@]}"; do
        rg -q -e "$NOID_PATTERN" <<< "$s" || { echo "NOID self-test: no match on the positive case '$s'"; failed=1; }
    done
    for s in "${negative[@]}"; do
        # r7 = #64 is the assembly immediate of NOID_ALLOW: the pattern
        # matches it, and the allowlist keeps it.
        [[ "$s" == "r7 = #64" ]] && { rg -q -e "$NOID_PATTERN" <<< "$s" || { echo "NOID self-test: the allowlist case '$s' does not match"; failed=1; }; continue; }
        rg -q -e "$NOID_PATTERN" <<< "$s" && { echo "NOID self-test: a match on the negative case '$s'"; failed=1; }
    done
    if git -C "$REPO" cat-file -e "$NOID_HISTORY_REF^{commit}" 2> /dev/null; then
        n="$(git -C "$REPO" log "$NOID_HISTORY_REF" --format=%B | rg -c -e "$NOID_PATTERN" || true)"
        [[ "${n:-0}" -eq $NOID_HISTORY_LINES ]] \
            || { echo "NOID self-test: the history of ${NOID_HISTORY_REF:0:12} has ${n:-0} lines with an ID, not $NOID_HISTORY_LINES"; failed=1; }
    else
        echo "NOID self-test: the clone has no commit ${NOID_HISTORY_REF:0:12}, thus the history case does not run"
    fi
    VIOLATIONS=()
    check_noid
    [[ ${#VIOLATIONS[@]} -eq 0 ]] || { echo "NOID self-test: the tree has ${#VIOLATIONS[@]} NOID violation(s)"; printf '  %s\n' "${VIOLATIONS[@]}"; failed=1; }
    [[ $failed -eq 0 ]] && echo "NOID self-test: ${#positive[@]} positive and ${#negative[@]} negative cases pass, the history has $NOID_HISTORY_LINES ID lines, the tree has none."
    return $failed
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --areas) AREAS="${2//,/ }"; shift 2 ;;
            --no-builds) CHECK_BUILDS=0; shift ;;
            --copies-only) COPIES_ONLY=1; shift ;;
            --commit-msg) COMMIT_MSG="${2:-}"; [[ -n "$COMMIT_MSG" ]] || { echo "check-rules: --commit-msg needs a file." >&2; exit 2; }; shift 2 ;;
            --noid-self-test) NOID_SELF_TEST=1; shift ;;
            --build)
                [[ "${2:-}" =~ ^[a-z]+(-android)?-(debug|release)-(none|asan|ubsan|tsan|msan|hwasan)$ ]] \
                    || { echo "check-rules: --build needs the name of a matrix build directory, for example core-debug-none." >&2; exit 2; }
                COPY_BUILDS+=" $2"; shift 2 ;;
            --accept-stamp)
                [[ "${2:-}" =~ ^[0-9a-f]{40}:[0-9a-f]{40}$ ]] \
                    || { echo "check-rules: --accept-stamp needs COMMIT:PATCHES (two 40-digit ids)." >&2; exit 2; }
                ACCEPT_STAMPS+=" $2"; shift 2 ;;
            --write-requests) WRITE_REQUESTS=1; shift ;;
            --json) JSON_OUT="$2"; shift 2 ;;
            -h|--help) print_usage; exit 0 ;;
            *) echo "check-rules: the option '$1' is not known. Use --help." >&2; exit 2 ;;
        esac
    done
    command -v jq > /dev/null || { echo "check-rules: jq is necessary." >&2; exit 2; }
    command -v rg > /dev/null || { echo "check-rules: rg (ripgrep) is necessary." >&2; exit 2; }
    if [[ $NOID_SELF_TEST -eq 1 ]]; then
        noid_self_test
        exit $?
    fi
    local full=1 sources=() area dir
    if [[ -n "$COMMIT_MSG" ]]; then
        # The commit-msg hook: only the message and the staged diff, thus it
        # takes less than one second.
        full=0
        AREAS=""
        check_noid_msg "$COMMIT_MSG"
        check_noid_staged
    else
        if [[ -z "$AREAS" ]]; then
            AREAS="$(find "$FUZZ_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2> /dev/null | sort | tr '\n' ' ')"
        elif [[ "$AREAS" == "none" ]]; then
            # --areas none: only the shared files (tests/sanitizers) and the
            # build directories.
            AREAS=""
        fi
        for area in $AREAS; do
            while IFS= read -r f; do sources+=("$f"); done < <(find "$FUZZ_DIR/$area" \
                \( -name CMakeLists.txt -o -name '*.cmake' -o -name '*.sh' \) -type f 2> /dev/null | sort)
        done
        while IFS= read -r f; do sources+=("$f"); done < <(find "$SAN_DIR" -maxdepth 1 -name '*.cmake' -type f | sort)
    fi
    if [[ -n "$COMMIT_MSG" ]]; then
        :
    elif [[ $COPIES_ONLY -eq 1 ]]; then
        full=0
        check_llama_copies
    else
        check_sources_r1 "${sources[@]}"
        check_reasons "${sources[@]}"
        check_recover_entries
        check_suppressions
        check_run_sh
        check_libfuzzer_commands
        check_death_callback
        check_noid
        check_noid_staged
        check_lto_partitions
        check_android_asan_runtime
        check_parity_sources
        check_repro
    fi
    if [[ $full -eq 1 && $CHECK_BUILDS -eq 1 && -d "$BUILD_FUZZ" ]]; then
        for dir in "$BUILD_FUZZ"/*/; do
            dir="${dir%/}"
            case "$(basename "$dir")" in
                matrix|matrix-upstream|msan-libcxx|*-src) continue ;;
            esac
            check_build_dir "$dir"
        done
        check_results
    fi

    local v r a p m
    for v in "${VIOLATIONS[@]}"; do
        IFS=$'\t' read -r r a p m <<< "$v"
        echo "$r $a $p: $m"
        if [[ -n "$JSON_OUT" ]]; then
            jq -nc --arg rule "$r" --arg area "$a" --arg place "$p" --arg message "$m" \
                '{rule: $rule, area: $area, place: $place, message: $message}' >> "$JSON_OUT"
        fi
    done
    [[ $WRITE_REQUESTS -eq 1 ]] && write_requests
    if [[ -n "$COMMIT_MSG" ]]; then
        echo "check-rules: ${#VIOLATIONS[@]} violation(s) in the commit message and the staged diff." >&2
    else
        echo "check-rules: ${#VIOLATIONS[@]} violation(s) in the areas [$AREAS] and tests/sanitizers." >&2
    fi
    [[ ${#VIOLATIONS[@]} -eq 0 ]]
}

main "$@"
