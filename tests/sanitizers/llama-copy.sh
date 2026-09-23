#!/usr/bin/env bash
# Make or refresh a private copy of the patched llama.cpp submodule, and
# write the stamp that the rule LLAMA-COPY of check-rules.sh reads.
#
# Usage:
#   tests/sanitizers/llama-copy.sh [--ggml] DEST
#
#   DEST     The root of the copy, for example build/fuzz/<area>/llama-src.
#            A build names DEST (a full copy) or DEST/ggml (--ggml) as its
#            source tree.
#   --ggml   Copy only third_party/llama.cpp/ggml, into DEST/ggml.
#
# The copy:
#   - It takes the landing lock build/.land.lock as a shared lock. Thus no
#     landing changes the submodule or patches/series during the copy.
#   - It stops if the submodule is not at the commit that HEAD records, or if
#     patches/series has a change that is not in HEAD.
#   - rsync compares the contents (-c) and does not keep the times of the
#     submodule (no -t). A file with new contents gets the time of the copy,
#     thus ninja compiles each object of that file again. A copy that keeps
#     the times (cp -a, rsync -a) can give a changed header a time before its
#     objects, and ninja then keeps the objects of the old header.
#   - --delete removes the files that the submodule does not have.
#   - The stamp DEST/.llama-copy-stamp has two lines:
#       commit <the llama.cpp commit of HEAD:third_party/llama.cpp>
#       series <the git object id of HEAD:patches/series>
#     The script removes the stamp before the copy and writes it after the
#     copy. Thus a copy that stops before the end has no stamp.
#
# Exit status: 0 on success, 1 on an error, 2 for a usage error.
# Complexity: one checksum pass over the submodule (about 400 MB).

set -euo pipefail

readonly REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly SUBMODULE="$REPO/third_party/llama.cpp"
readonly LOCK="$REPO/build/.land.lock"
readonly STAMP_NAME=".llama-copy-stamp"

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

# Write a message to stderr and stop with the code 1.
die() {
    echo "llama-copy: $*" >&2
    exit 1
}

# Copy the tree and write the stamp. The caller holds the shared lock.
# Arguments: the source directory, the destination directory, the stamp file.
copy_tree() {
    local src="$1" dst="$2" stamp="$3" commit series head
    commit="$(git -C "$REPO" rev-parse HEAD:third_party/llama.cpp)"
    series="$(git -C "$REPO" rev-parse HEAD:patches/series)"
    head="$(git -C "$SUBMODULE" rev-parse HEAD)"
    [[ "$head" == "$commit" ]] \
        || die "the submodule is at $head, but HEAD records $commit. Update the submodule and run scripts/apply-patches.sh, then try again."
    [[ "$(git -C "$REPO" hash-object patches/series)" == "$series" ]] \
        || die "patches/series has a change that is not in HEAD. Commit it (or wait for the landing), then try again."
    rm -f "$stamp"
    mkdir -p "$dst"
    rsync -rlc --delete --exclude .git --exclude '/build*/' --exclude __pycache__ \
        --exclude "/$STAMP_NAME" "$src/" "$dst/"
    printf 'commit %s\nseries %s\n' "$commit" "$series" > "$stamp"
}

main() {
    local ggml=0 dest="" sub_real
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --ggml) ggml=1; shift ;;
            -h|--help) print_usage; exit 0 ;;
            -*) echo "llama-copy: the option '$1' is not known. Use --help." >&2; exit 2 ;;
            *) [[ -z "$dest" ]] || { echo "llama-copy: give one DEST only." >&2; exit 2; }
               dest="$1"; shift ;;
        esac
    done
    [[ -n "$dest" ]] || { echo "llama-copy: DEST is necessary. Use --help." >&2; exit 2; }
    command -v rsync > /dev/null || die "rsync is necessary."
    command -v flock > /dev/null || die "flock is necessary."
    # realpath -m: the physical path, also for a DEST that does not exist yet.
    dest="$(realpath -m "$dest")"
    sub_real="$(realpath "$SUBMODULE")"
    [[ "$dest/" != "$sub_real"/* ]] || die "DEST must not be in the submodule."
    mkdir -p "$dest" "$(dirname "$LOCK")"

    local src="$SUBMODULE" dst="$dest"
    if [[ $ggml -eq 1 ]]; then
        src="$SUBMODULE/ggml"
        dst="$dest/ggml"
    fi
    # The child shell of flock gets the functions and the constants with
    # declare, because it does not inherit them.
    flock -s "$LOCK" bash -c "set -euo pipefail
        $(declare -p REPO SUBMODULE STAMP_NAME)
        $(declare -f die copy_tree)
        copy_tree \"\$1\" \"\$2\" \"\$3\"" _ "$src" "$dst" "$dest/$STAMP_NAME"
    echo "llama-copy: $dst is a copy of $src, with the stamp $dest/$STAMP_NAME."
}

main "$@"
