#!/usr/bin/env bash
# Apply the patch series to the llama.cpp submodule. The script is idempotent.
#
#   scripts/apply-patches.sh
#
# patches/series holds one patch path per line, relative to patches/, in the
# order of application. A patch can change the files of an earlier patch, thus
# the applied state of the stack is only readable from its top. The script does
# these steps:
#
#   1. From the last patch down to the first, find the first patch that
#      `git apply --reverse --check` accepts. That patch and all patches before
#      it are applied. No patch accepts the reverse check on a clean tree.
#   2. For each patch after that one, in order: if `git apply --check` passes,
#      apply the patch. If not, the patch is in conflict: the script stops.
#
# The submodule must be at the pinned commit. Get it with:
#
#   git submodule update --init --depth 1 third_party/llama.cpp

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SERIES="$REPO_ROOT/patches/series"

[[ -f "$LLAMA_SUBMODULE/CMakeLists.txt" ]] \
    || die "the submodule is empty: git submodule update --init --depth 1 third_party/llama.cpp"

head=$(git -C "$LLAMA_SUBMODULE" rev-parse HEAD)
[[ "$head" == "$LLAMA_COMMIT" ]] \
    || die "the submodule is at $head, the pin is $LLAMA_COMMIT (scripts/lib.sh)"

patches=()
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ -f "$REPO_ROOT/patches/$line" ]] || die "the series names a missing file: patches/$line"
    patches+=("$line")
done < "$SERIES"

# 1. The applied prefix of the stack.
applied=0
for ((i = ${#patches[@]} - 1; i >= 0; i--)); do
    if git -C "$LLAMA_SUBMODULE" apply --reverse --check "$REPO_ROOT/patches/${patches[i]}" 2> /dev/null; then
        applied=$((i + 1))
        break
    fi
done
for ((i = 0; i < applied; i++)); do
    echo "applied  patches/${patches[i]}"
done

# 2. The patches above the prefix.
new=0
for ((i = applied; i < ${#patches[@]}; i++)); do
    patch="$REPO_ROOT/patches/${patches[i]}"
    if git -C "$LLAMA_SUBMODULE" apply --check "$patch" 2> /dev/null; then
        git -C "$LLAMA_SUBMODULE" apply "$patch"
        echo "apply    patches/${patches[i]}"
        new=$((new + 1))
    else
        echo "conflict patches/${patches[i]}" >&2
        git -C "$LLAMA_SUBMODULE" apply --check "$patch" >&2 || true
        die "the patch does not apply: patches/${patches[i]}"
    fi
done

echo "patches: $applied applied before, $new applied at this time"
