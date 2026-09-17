# Shared constants and functions of the build scripts. Source this file, do not
# execute it. Each constant here is a pin: change it only together with the
# item that it pins.
#
# The llama.cpp pin is the commit of the submodule third_party/llama.cpp. The
# build number is the upstream count of that commit: the release tag b11007
# is its parent, and b11008 was never tagged. A shallow clone counts 1, thus
# the build passes the number and the short hash to CMake.
#
# The Snapdragon image is pinned by the digest of its amd64 manifest. The
# multi-arch index of the tag v0.7 has the digest 91714433... and contains it.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly REPO_ROOT

readonly LLAMA_SUBMODULE="$REPO_ROOT/third_party/llama.cpp"
readonly LLAMA_COMMIT=c6824a9e42ceeda5d58089fa274ddd816e59e68e
readonly LLAMA_BUILD_NUMBER=11008
readonly SNAPDRAGON_IMAGE=ghcr.io/snapdragon-toolchain/arm64-android@sha256:c012b8174f4154088ee027077e9cb80e68cc9a494d46f63454a050fa4789897b

# The libraries that the app ships, in the order of android/snapdragon/build.sh.
readonly LLAMA_LIBS="llama ggml ggml-base ggml-cpu ggml-opencl ggml-hexagon mtmd llama-common"
readonly HTP_DSP=v79

# Write a message to stderr and stop with the code 1.
die() {
    echo "error: $*" >&2
    exit 1
}

# Print the name of the container engine: podman, or docker when podman is absent.
container_engine() {
    if command -v podman > /dev/null 2>&1; then
        echo podman
    elif command -v docker > /dev/null 2>&1; then
        echo docker
    else
        die "no container engine: install podman or docker"
    fi
}

# Start a container with the repository at /workspace. The arguments are engine
# options, then the image, then the command. The container gets the uid of the
# caller, a writable HOME below build/cache, a UTC clock, and a C locale. Podman
# on a SELinux host needs label=disable, or the mount is empty.
container_run() {
    local engine
    engine=$(container_engine)
    mkdir -p "$REPO_ROOT/build/cache/home"
    local -a opts=(
        --rm
        -v "$REPO_ROOT:/workspace"
        -w /workspace
        -e HOME=/workspace/build/cache/home
        -e TZ=UTC
        -e LC_ALL=C.UTF-8
    )
    case $engine in
        podman) opts+=(--userns=keep-id --security-opt label=disable) ;;
        docker) opts+=(--user "$(id -u):$(id -g)") ;;
    esac
    "$engine" run "${opts[@]}" "$@"
}

# Print the tag of the APK container image, and build the image when it does
# not exist. The tag holds a hash of ci/Containerfile, thus a change of that
# file gives a new image. The build output goes to stderr, thus a caller can
# read the tag from stdout.
ensure_apk_image() {
    local engine image
    engine=$(container_engine)
    image="localhost/qwen-mobile-apk:$(sha256sum "$REPO_ROOT/ci/Containerfile" | cut -c1-12)"
    if ! "$engine" image inspect "$image" > /dev/null 2>&1; then
        echo "apk: build the image $image" >&2
        "$engine" build -t "$image" -f "$REPO_ROOT/ci/Containerfile" "$REPO_ROOT/ci" >&2
    fi
    echo "$image"
}

# Print the number of commits of the branch, the versionCode of the APK. A tree
# without a git directory gives the fallback of android/version.properties.
version_code() {
    local count
    if count=$(git -C "$REPO_ROOT" rev-list --count HEAD 2> /dev/null) && [[ -n $count ]]; then
        echo "$count"
    else
        sed -n 's/^versionCodeFallback=//p' "$REPO_ROOT/android/version.properties" | tr -d ' \r'
    fi
}

# Print the versionName of android/version.properties.
version_name() {
    local name
    name=$(sed -n 's/^versionName=//p' "$REPO_ROOT/android/version.properties" | tr -d ' \r')
    [[ -n $name ]] || die "no versionName in android/version.properties"
    echo "$name"
}

# Print the SOURCE_DATE_EPOCH of the repository: the date of the last commit.
source_date_epoch() {
    git -C "$REPO_ROOT" log -1 --format=%ct
}

# Write "<sha256>  <name>" for each file argument, sorted by name, to stdout.
# The name is the base name of the file.
sha256_table() {
    local file
    for file in "$@"; do
        printf '%s  %s\n' "$(sha256sum "$file" | cut -d' ' -f1)" "$(basename "$file")"
    done | sort -k2
}
