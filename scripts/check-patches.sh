#!/usr/bin/env bash
# Make sure that the llama.cpp submodule is the pinned commit plus the patch
# series, and nothing else. The script stops with the code 1 if one condition
# is not satisfied.
#
#   scripts/check-patches.sh
#
# The checks are:
#
#   1. The submodule HEAD is the commit that the superproject records, and it
#      is the pin in scripts/lib.sh.
#   2. The last patch of the series accepts `git apply --reverse --check`: the
#      top of the stack is applied.
#   3. A scratch copy of the commit, with the series applied to it, has the
#      same content as the submodule, file by file. The comparison covers the
#      tracked files of the commit and the files that the series touches.
#   4. The submodule has no other new file. CMakeUserPresets.json is the one
#      permitted extra file, and build directories are ignored by llama.cpp.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SERIES="$REPO_ROOT/patches/series"
status=0

# 1. The commit.
recorded=$(git -C "$REPO_ROOT" ls-tree HEAD third_party/llama.cpp | cut -d' ' -f3 | cut -f1)
head=$(git -C "$LLAMA_SUBMODULE" rev-parse HEAD)
if [[ "$head" != "$recorded" ]]; then
    echo "commit   FAIL the submodule is at $head, the superproject records $recorded" >&2
    status=1
elif [[ "$head" != "$LLAMA_COMMIT" ]]; then
    echo "commit   FAIL the submodule is at $head, the pin in scripts/lib.sh is $LLAMA_COMMIT" >&2
    status=1
else
    echo "commit   ok   $head"
fi

patches=()
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ ! -f "$REPO_ROOT/patches/$line" ]]; then
        echo "series   FAIL missing file patches/$line" >&2
        status=1
        continue
    fi
    patches+=("$line")
done < "$SERIES"
[[ ${#patches[@]} -gt 0 ]] || die "patches/series is empty"

# 2. The top of the stack.
top="${patches[${#patches[@]} - 1]}"
if git -C "$LLAMA_SUBMODULE" apply --reverse --check "$REPO_ROOT/patches/$top" 2> /dev/null; then
    echo "top      ok   patches/$top is applied"
else
    echo "top      FAIL patches/$top is not applied: scripts/apply-patches.sh" >&2
    status=1
fi

# 3. The reconstruction. The scratch directory is its own git repository, thus
# `git apply` resolves the paths from its root.
scratch=$(mktemp -d "${TMPDIR:-/tmp}/check-patches.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
git -C "$LLAMA_SUBMODULE" archive --format=tar HEAD | tar -x -C "$scratch"
git -C "$scratch" init -q
touched=()
for line in "${patches[@]}"; do
    patch="$REPO_ROOT/patches/$line"
    while IFS= read -r file; do
        touched+=("$file")
    done < <(git -C "$scratch" apply --numstat "$patch" | cut -f3)
    git -C "$scratch" apply "$patch" || die "patches/$line does not apply to the commit"
done

different=0
while IFS= read -r file; do
    if [[ -f "$scratch/$file" && -f "$LLAMA_SUBMODULE/$file" ]]; then
        cmp -s "$scratch/$file" "$LLAMA_SUBMODULE/$file" || { echo "content  FAIL $file differs" >&2; different=$((different + 1)); }
    elif [[ -f "$scratch/$file" ]]; then
        echo "content  FAIL $file is missing in the submodule" >&2
        different=$((different + 1))
    elif [[ -f "$LLAMA_SUBMODULE/$file" ]]; then
        echo "content  FAIL $file must not exist in the submodule" >&2
        different=$((different + 1))
    fi
done < <( { git -C "$LLAMA_SUBMODULE" ls-tree -r --name-only HEAD; printf '%s\n' "${touched[@]}"; } | sort -u)
if [[ $different -eq 0 ]]; then
    echo "content  ok   the submodule is the commit plus ${#patches[@]} patches"
else
    status=1
fi

# 4. Other new files.
extra=$(git -C "$LLAMA_SUBMODULE" status --porcelain --untracked-files=all \
    | grep '^??' | cut -c4- | grep -v '^CMakeUserPresets.json$' \
    | grep -v -F -x -f <(printf '%s\n' "${touched[@]}") || true)
if [[ -n "$extra" ]]; then
    echo "files    FAIL the submodule has new files outside the series:" >&2
    while IFS= read -r file; do
        printf '           %s\n' "$file" >&2
    done <<< "$extra"
    status=1
else
    echo "files    ok   no new file outside the series"
fi

exit $status
