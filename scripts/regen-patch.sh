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
# of the old file, which is everything before the first `diff --git`, and it
# takes the file list from the old body. For a new patch, name the files on
# the command line, and add the patch to patches/series yourself. A new patch
# goes on top of the whole series.
#
# Two traps that this script avoids:
#   - `git apply --include` reports success for a patch that touches none of
#     the selected files, which would give a base with patches missing. Each
#     patch applies in full and the script stops when one fails.
#   - A diff written inside the scratch repository becomes part of the next
#     `git add`. The script writes to a temporary file outside the scratch.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

target="${1:-}"
[[ -n "$target" ]] || die "usage: scripts/regen-patch.sh patches/<group>/<name>.patch [file ...]"
shift || true
files=("$@")

cd "$REPO_ROOT"
[[ "$target" == patches/* ]] || die "the target must be below patches/: $target"
rel="${target#patches/}"

# The patches that come before the target. A target that the series does not
# name yet goes on top of all of them.
before=()
found=0
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ "$line" == "$rel" ]]; then found=1; break; fi
    before+=("$line")
done < patches/series

header=""
if [[ -f "$target" ]]; then
    header=$(sed '/^diff --git /,$d' "$target")
    if [[ ${#files[@]} -eq 0 ]]; then
        mapfile -t files < <(grep '^diff --git ' "$target" | sed 's|^diff --git a/||; s| b/.*$||')
    fi
fi
[[ ${#files[@]} -gt 0 ]] || die "no file list: give the files on the command line"
[[ $found -eq 1 || ! -f "$target" ]] || die "patches/series does not name $rel"

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
[[ -f "$target" ]] && old_lines=$(grep -c '' < <(sed -n '/^diff --git /,$p' "$target") || true)
new_lines=$(grep -c '' < "$out")

mkdir -p "$(dirname "$target")"
{ [[ -n "$header" ]] && printf '%s\n' "$header"; cat "$out"; } > "$target"
echo "regen: $target  body $old_lines -> $new_lines lines, ${#files[@]} files"

if ! scripts/check-patches.sh > /dev/null 2>&1; then
    scripts/check-patches.sh || true
    die "scripts/check-patches.sh does not pass after the rebuild"
fi
echo "regen: scripts/check-patches.sh passes"
