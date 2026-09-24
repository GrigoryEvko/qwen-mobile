#!/usr/bin/env bash
# Land one llama.cpp patch in patches/ under the landing lock, and commit it.
#
#   scripts/land-patch.sh [--after-group] [--areas LIST] [--check] PATCH DEST MSG_FILE [PATH...]
#
#   PATCH          The patch file to land.
#   DEST           The name of the patch below patches/, for example
#                  memory/0007-hexagon-a-switch-for-the-fwht-form-of-the-kv-cache-rotation.patch.
#                  The directory of DEST is the group of the patch.
#   MSG_FILE       The commit message.
#   PATH           More paths for the same commit, relative to the root of the
#                  repository, for example a regression input of a fuzz area.
#   --after-group  Put the series line after the last line of the group of DEST.
#                  Without it, the line goes to the end of the series.
#   --areas LIST   Give "--areas LIST --no-builds" to tests/sanitizers/check-rules.sh.
#                  Without it, check-rules.sh checks each area and the builds.
#   --check        Do the steps 1 to 3 and stop before a change.
#
# The steps:
#   1. Take the lock build/.land.lock. Do not run the script under a second
#      flock of that file, because the two locks then wait for each other.
#   2. Stop when patches/DEST exists, or when scripts/check-patches.sh fails
#      before the landing.
#   3. Apply the committed series (patches/series of HEAD) with the new line in
#      its place to a scratch tree of the pinned commit. Each patch must apply.
#      Thus the new patch applies to the committed series as it is, and each
#      patch after it applies too.
#   4. Copy PATCH to patches/DEST, apply it to the submodule, and put the line
#      into the series of the work tree. A line of a different agent that is
#      not committed stays in the series of the work tree.
#   5. Run scripts/check-patches.sh and tests/sanitizers/check-rules.sh.
#   6. Commit patches/DEST, patches/series and each PATH. The commit takes the
#      committed series plus the new line, and not the lines that are not
#      committed. The commit-msg hook checks the message.
# When a step after step 3 fails, the script removes the patch from the
# submodule and from patches/, and puts back the series of the work tree.
#
# This script replaces the landing helpers that each fuzz area kept below
# build/. It does not build a tree or a library. For a private tree of HEAD
# plus a patch, use tests/sanitizers/llama-copy.sh and then git apply.
#
# Exit status: 0 when the patch is committed (or passes the checks with
# --check), 1 on an error, 2 for a usage error.
# Time: about 30 s (the scratch tree, check-patches.sh and check-rules.sh).
# The scratch tree applies each patch of the series one time: O(patches).

set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"

usage() {
    echo "usage: scripts/land-patch.sh [--after-group] [--areas LIST] [--check] PATCH DEST MSG_FILE [PATH...]" >&2
    exit 2
}

# Print the lines of the series file $1 with the line $2 in its place: after
# the last line of the group $3, or at the end when $3 is empty or has no line.
# Each line gets a newline, also a last line that has none in the file. O(lines).
insert_line() {
    local series=$1 line=$2 group=$3 last=0 n=0 i text
    local -a lines=()
    while IFS= read -r text || [[ -n $text ]]; do
        lines+=("$text")
        n=$((n + 1))
        [[ -n $group && $text == "$group"/* ]] && last=$n
    done < "$series"
    [[ $last -eq 0 ]] && last=$n
    [[ $last -eq 0 ]] && printf '%s\n' "$line"
    for ((i = 0; i < n; i++)); do
        printf '%s\n' "${lines[i]}"
        [[ $((i + 1)) -eq $last ]] && printf '%s\n' "$line"
    done
    return 0
}

main() {
    local after_group=0 areas="" check_only=0
    while [[ $# -gt 0 ]]; do
        case $1 in
            --after-group) after_group=1; shift ;;
            --areas) [[ $# -ge 2 ]] || usage; areas=$2; shift 2 ;;
            --check) check_only=1; shift ;;
            -h|--help) sed -n '2,/^$/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; exit 0 ;;
            -*) echo "land-patch: the option $1 is not known" >&2; usage ;;
            *) break ;;
        esac
    done
    [[ $# -ge 3 ]] || usage
    local patch msg dest=$2
    patch=$(realpath -e "$1") || die "the patch $1 does not exist"
    msg=$(realpath -e "$3") || die "the message file $3 does not exist"
    shift 3
    local -a extra=("$@")
    [[ $dest =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.patch$ ]] \
        || die "DEST must be <group>/<name>.patch below patches/, not $dest"
    local group=""
    [[ $after_group -eq 1 ]] && group=${dest%%/*}

    cd "$REPO_ROOT"
    mkdir -p build
    exec 9> build/.land.lock
    echo "land-patch: wait for the lock build/.land.lock"
    flock 9

    # 2. The state before the landing.
    [[ ! -e patches/$dest ]] || die "patches/$dest exists"
    scripts/check-patches.sh > /dev/null || die "scripts/check-patches.sh fails before the landing"

    local work
    work=$(mktemp -d "$REPO_ROOT/build/.land-patch.XXXXXX")
    # shellcheck disable=SC2064
    trap "rm -rf '$work'" EXIT
    cp patches/series "$work/series.before"
    git show HEAD:patches/series > "$work/series.head"
    insert_line "$work/series.before" "$dest" "$group" > "$work/series.work"
    insert_line "$work/series.head" "$dest" "$group" > "$work/series.commit"

    # 3. The committed series with the new line, on a scratch tree of the pin.
    mkdir -p "$work/src"
    git -C "$LLAMA_SUBMODULE" archive --format=tar "$LLAMA_COMMIT" | tar -x -C "$work/src"
    git -C "$work/src" init -q
    local line n=0 file
    while IFS= read -r line; do
        [[ -z $line || $line == \#* ]] && continue
        if [[ $line == "$dest" ]]; then
            file=$patch
        else
            file=$work/committed.patch
            git show "HEAD:patches/$line" > "$file" || die "HEAD has no patches/$line"
        fi
        git -C "$work/src" apply --whitespace=nowarn "$file" \
            || die "patches/$line does not apply in the committed series with the new patch"
        n=$((n + 1))
    done < "$work/series.commit"
    rm -rf "$work/src"
    echo "land-patch: the committed series with patches/$dest applies ($n patches)"
    if [[ $check_only -eq 1 ]]; then
        diff "$work/series.head" "$work/series.commit" || true
        echo "land-patch: --check, no change"
        return 0
    fi

    # 4. The patch, the submodule and the series of the work tree.
    local applied=0
    undo() {
        echo "land-patch: $1. Remove patches/$dest again." >&2
        [[ $applied -eq 1 ]] && git -C "$LLAMA_SUBMODULE" apply --reverse "$REPO_ROOT/patches/$dest"
        cp "$work/series.before" patches/series
        rm -f "patches/$dest"
        exit 1
    }
    mkdir -p "patches/$(dirname "$dest")"
    cp "$patch" "patches/$dest"
    git -C "$LLAMA_SUBMODULE" apply --check "$REPO_ROOT/patches/$dest" || undo "the patch does not apply to the submodule"
    git -C "$LLAMA_SUBMODULE" apply "$REPO_ROOT/patches/$dest" || undo "git apply failed in the submodule"
    applied=1
    cp "$work/series.work" patches/series

    # 5. The checks.
    scripts/check-patches.sh || undo "scripts/check-patches.sh fails"
    if [[ -n $areas ]]; then
        tests/sanitizers/check-rules.sh --areas "$areas" --no-builds || undo "tests/sanitizers/check-rules.sh fails"
    else
        tests/sanitizers/check-rules.sh || undo "tests/sanitizers/check-rules.sh fails"
    fi

    # 6. The commit of the committed series plus the line, then the series of the work tree again.
    cp "$work/series.commit" patches/series
    git add -- "patches/$dest" patches/series "${extra[@]}" || undo "git add failed"
    if ! git commit -q -F "$msg" -- "patches/$dest" patches/series "${extra[@]}"; then
        git restore --staged -- "patches/$dest" patches/series "${extra[@]}"
        undo "git commit failed"
    fi
    cp "$work/series.work" patches/series
    git log --oneline -1
}

main "$@"
