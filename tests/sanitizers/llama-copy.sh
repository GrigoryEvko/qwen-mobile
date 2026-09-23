#!/usr/bin/env bash
# Make or refresh a private copy of the patched llama.cpp tree of HEAD, and
# write the stamp that the rule LLAMA-COPY of check-rules.sh reads.
#
# Usage:
#   tests/sanitizers/llama-copy.sh [--ggml] DEST
#
#   DEST     The root of the copy, for example build/fuzz/<area>/llama-src.
#            A build names DEST (a full copy) or DEST/ggml (--ggml) as its
#            source tree.
#   --ggml   Copy only the ggml directory, into DEST/ggml.
#
# The copy:
#   - The source is the git objects of one commit of the repository (HEAD
#     when the script starts): the llama.cpp commit that it records, plus
#     each patch of its patches/series. The work tree of the submodule is not
#     read, thus an uncommitted edit there, or a landing during the copy,
#     does not go into the copy.
#   - rsync compares the contents (-c) and does not keep the times (no -t).
#     A file with new contents gets the time of the copy, thus ninja
#     compiles each object of that file again, and a file with the same
#     contents keeps its time. A copy that keeps the times (cp -a, rsync -a)
#     can give a changed header a time before its objects, and ninja then
#     keeps the objects of the old header.
#   - --delete removes the files that the patched tree does not have.
#   - The stamp DEST/.llama-copy-stamp has two lines:
#       commit <the llama.cpp commit that the repository commit records>
#       patches <the git tree id of patches/ in the repository commit>
#     The tree id changes when a patch file or patches/series changes. The
#     script removes the stamp before the copy and writes it after the copy.
#     Thus a copy that stops before the end has no stamp.
#
# Exit status: 0 on success, 1 on an error, 2 for a usage error.
# Time: about 10 s on a loaded build server (the series has about 180
# patches). Disk: one temporary tree of about 400 MB under build/.

set -euo pipefail

readonly REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly SUBMODULE="$REPO/third_party/llama.cpp"
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

main() {
    local ggml=0 dest="" sub_real head commit patches scratch line src dst
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
    # realpath -m: the physical path, also for a DEST that does not exist yet.
    dest="$(realpath -m "$dest")"
    sub_real="$(realpath "$SUBMODULE")"
    [[ "$dest/" != "$sub_real"/* ]] || die "DEST must not be in the submodule."

    head="$(git -C "$REPO" rev-parse HEAD)"
    commit="$(git -C "$REPO" rev-parse "$head:third_party/llama.cpp")"
    patches="$(git -C "$REPO" rev-parse "$head:patches")"
    git -C "$SUBMODULE" cat-file -e "$commit^{commit}" 2> /dev/null \
        || die "the submodule has no object of the llama.cpp commit $commit. Fetch it in third_party/llama.cpp, then try again."

    mkdir -p "$REPO/build" "$dest"
    scratch="$(mktemp -d "$REPO/build/.llama-copy.XXXXXX")"
    # shellcheck disable=SC2064
    trap "rm -rf '$scratch'" EXIT
    mkdir -p "$scratch/src" "$scratch/patches"
    git -C "$SUBMODULE" archive --format=tar "$commit" | tar -x -C "$scratch/src"
    git -C "$REPO" archive --format=tar "$patches" | tar -x -C "$scratch/patches"
    # The scratch tree is its own git repository, thus git apply resolves the
    # paths from its root and not from the repository around build/.
    git -C "$scratch/src" init -q
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        [[ -f "$scratch/patches/$line" ]] || die "patches/series of $head names patches/$line, which does not exist."
        git -C "$scratch/src" apply --whitespace=nowarn "$scratch/patches/$line" \
            || die "patches/$line of $head does not apply to llama.cpp $commit."
    done < "$scratch/patches/series"
    rm -rf "$scratch/src/.git"

    src="$scratch/src"
    dst="$dest"
    if [[ $ggml -eq 1 ]]; then
        src="$scratch/src/ggml"
        dst="$dest/ggml"
    fi
    rm -f "$dest/$STAMP_NAME"
    mkdir -p "$dst"
    rsync -rlc --delete --exclude '/build*/' --exclude "/$STAMP_NAME" "$src/" "$dst/"
    printf 'commit %s\npatches %s\n' "$commit" "$patches" > "$dest/$STAMP_NAME"
    echo "llama-copy: $dst is the patched llama.cpp tree of $head (llama.cpp ${commit:0:12}, patches ${patches:0:12})."
}

main "$@"
