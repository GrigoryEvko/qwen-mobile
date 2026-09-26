#!/usr/bin/env bash
# Rebuild one patch of the series from the live submodule.
#
#   scripts/regen-patch.sh patches/<group>/<name>.patch [file ...]
#
# A patch of the series applies on top of the patches before it, thus a plain
# `git diff` of the submodule gives the whole stack and not one patch. This
# script builds the correct base: a scratch copy of the pinned commit with the
# patches before the target applied to it. Then it copies the files of the
# target from the live submodule and takes the difference.
#
# For a patch that patches/series already names, the script keeps the header
# of the old file, which is everything before the first line of the diff body,
# and it takes the file list from the old body. For a new patch, name the files
# on the command line, and add the patch to patches/series yourself. A new
# patch goes on top of the whole series.
#
# THE TARGET MUST BE THE LAST PATCH THAT TOUCHES ITS FILES. The script takes
# the files from the live submodule, which holds the whole series. Thus a target
# with a later patch on one of its files absorbs the change of that later patch,
# and the rebuilt patch is wrong. The script finds each such patch and stops.
# ggml/src/ggml-hexagon/ggml-hexagon.cpp has 39 patches and
# tests/test-backend-ops.cpp has 17, thus this condition is the usual one.
#
# Environment:
#   REGEN_ALLOW_LATER=1   Rebuild the patch although a later patch of the series
#                         touches one of its files. Use it only when you give the
#                         file list on the command line and no later patch
#                         touches those files. The applied tree of the series is
#                         the proof: run scripts/check-patches.sh after the
#                         rebuild.
#
# Three traps that this script avoids:
#   - `git apply --include` reports success for a patch that touches none of
#     the selected files, which would give a base with patches missing. Each
#     patch applies in full and the script stops when one fails.
#   - A diff written inside the scratch repository becomes part of the next
#     `git add`. The script writes to a temporary file outside the scratch.
#   - A patch body does not always start with `diff --git`. 20 patches of the
#     series start with `--- a/<file>`, thus a cut at `diff --git` keeps the
#     whole body as the header. The header stops at the first line of the body.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Print the header of a patch: each line before the first line of the diff body.
patch_header() {
    local line
    while IFS= read -r line; do
        case "$line" in
            'diff --git '*|'diff -urN '*|'--- '*) return 0 ;;
        esac
        printf '%s\n' "$line"
    done < "$1"
}

# Print the files that a patch touches, one for each line, without the leading
# path component that `git apply -p1` removes. A file header is a line that
# starts with "--- " and a line that starts with "+++ " after it. A content line
# of the body can start with the same characters, thus the pair is the test.
patch_file_list() {
    local -a lines
    local i src dst
    mapfile -t lines < "$1"
    for ((i = 0; i < ${#lines[@]} - 1; i++)); do
        [[ "${lines[i]}" == '--- '* && "${lines[i + 1]}" == '+++ '* ]] || continue
        src="${lines[i]#--- }"
        src="${src%%$'\t'*}"
        dst="${lines[i + 1]#+++ }"
        dst="${dst%%$'\t'*}"
        [[ "$dst" == /dev/null ]] && dst="$src"
        [[ "$dst" == /dev/null ]] && continue
        printf '%s\n' "${dst#*/}"
    done | sort -u
}

target="${1:-}"
[[ -n "$target" ]] || die "usage: scripts/regen-patch.sh patches/<group>/<name>.patch [file ...]"
shift || true
files=("$@")

cd "$REPO_ROOT"
[[ "$target" == patches/* ]] || die "the target must be below patches/: $target"
rel="${target#patches/}"

# The patches that come before the target, and the patches after it. A target
# that the series does not name yet goes on top of all of them.
before=()
after=()
found=0
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ "$line" == "$rel" ]]; then found=1; continue; fi
    if [[ $found -eq 0 ]]; then before+=("$line"); else after+=("$line"); fi
done < patches/series

header=""
if [[ -f "$target" ]]; then
    header=$(patch_header "$target")
    if [[ ${#files[@]} -eq 0 ]]; then
        mapfile -t files < <(patch_file_list "$target")
    fi
fi
[[ ${#files[@]} -gt 0 ]] || die "no file list: give the files on the command line"
[[ $found -eq 1 || ! -f "$target" ]] || die "patches/series does not name $rel"

# The guard: a later patch of the series must not touch a file of the target.
conflicts=""
for line in "${after[@]}"; do
    shared=""
    while IFS= read -r f; do
        for want in "${files[@]}"; do
            [[ "$f" == "$want" ]] && shared+="               $f"$'\n'
        done
    done < <(patch_file_list "$REPO_ROOT/patches/$line")
    [[ -n "$shared" ]] && conflicts+="           patches/$line"$'\n'"$shared"
done
if [[ -n "$conflicts" ]]; then
    if [[ "${REGEN_ALLOW_LATER:-0}" == 1 ]]; then
        echo "regen: REGEN_ALLOW_LATER=1, and these later patches touch the files of the target:" >&2
        printf '%s' "$conflicts" >&2
    else
        echo "error: $rel is not the last patch that touches its files." >&2
        echo "       The script takes the files from the live submodule, which holds the whole" >&2
        echo "       series. Thus the rebuilt patch would absorb the change of each patch below." >&2
        printf '%s' "$conflicts" >&2
        echo "       Give the file list on the command line and set REGEN_ALLOW_LATER=1 when no" >&2
        echo "       later patch touches the files that you name." >&2
        exit 1
    fi
fi

scratch=$(mktemp -d "${TMPDIR:-/tmp}/regen-patch.XXXXXX")
out=$(mktemp "${TMPDIR:-/tmp}/regen-patch-body.XXXXXX")
trap 'rm -rf "$scratch" "$out"' EXIT

git -C "$LLAMA_SUBMODULE" archive --format=tar HEAD | tar -x -C "$scratch"
git -C "$scratch" init -q
git -C "$scratch" add -A
git -C "$scratch" -c user.email=regen@local -c user.name=regen commit -qm base

for line in "${before[@]}"; do
    git -C "$scratch" apply "$REPO_ROOT/patches/$line" \
        || die "the base is not buildable: patches/$line does not apply"
done
git -C "$scratch" add -A
git -C "$scratch" -c user.email=regen@local -c user.name=regen commit -qm "series before $rel"
echo "regen: base = the pinned commit plus ${#before[@]} patches"

for f in "${files[@]}"; do
    if [[ -e "$LLAMA_SUBMODULE/$f" ]]; then
        mkdir -p "$scratch/$(dirname "$f")"
        cp "$LLAMA_SUBMODULE/$f" "$scratch/$f"
    else
        rm -f "$scratch/$f"
    fi
done

git -C "$scratch" add -A -- "${files[@]}"
git -C "$scratch" diff --cached --binary -- "${files[@]}" > "$out" || true
[[ -s "$out" ]] || die "the difference is empty: the submodule matches the base for those files"

old_lines=0
if [[ -f "$target" ]]; then
    old_lines=$(( $(grep -c '' < "$target") - $(printf '%s\n' "$header" | grep -c '') ))
fi
new_lines=$(grep -c '' < "$out")

mkdir -p "$(dirname "$target")"
{ [[ -n "$header" ]] && printf '%s\n' "$header"; cat "$out"; } > "$target"
echo "regen: $target  body $old_lines -> $new_lines lines, ${#files[@]} files"

if ! scripts/check-patches.sh > /dev/null 2>&1; then
    scripts/check-patches.sh || true
    die "scripts/check-patches.sh does not pass after the rebuild"
fi
echo "regen: scripts/check-patches.sh passes"
