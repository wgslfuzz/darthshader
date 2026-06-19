#!/bin/bash

set -e

# --- Globals and Defaults ---
DAWN_REVISION="3de0f00"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DEFAULT_DAWN_DIR="$SCRIPT_DIR/dawn"
PATCH_FILE="$SCRIPT_DIR/patches/dawn_${DAWN_REVISION}.diff"
DOCKER_IMAGE="aflplusplus/aflplusplus:v4.10c"

# Variables to be set by arguments or detection
DAWN_DIR=""
BUILD_DIR=""
AFL_PATH=""
USE_DOCKER=false
USE_SUDO=false
CC=""
CXX=""
LLVM_AR=""
LLVM_RANLIB=""
DOCKER_CMD="${DOCKER_CMD:-}"

# --- Functions ---

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -s <dir>   Dawn source directory (default: $DEFAULT_DAWN_DIR)"
    echo "  -b <dir>   Dawn build directory (default: <source_dir>/out/build)"
    echo "  -a <path>  Path to AFL++ bin directory (ignored if -d is used)"
    echo "  -d         Use Docker ($DOCKER_IMAGE) for compilation"
    echo "  -u         Use sudo for Docker commands"
    echo "  -h         Show this help message"
    exit 1
}

parse_args() {
    OPTIND=1
    while getopts "s:b:a:duh" opt; do
        case ${opt} in
            s )
                DAWN_DIR=$OPTARG
                ;;
            b )
                BUILD_DIR=$OPTARG
                ;;
            a )
                AFL_PATH=$OPTARG
                ;;
            d )
                USE_DOCKER=true
                ;;
            u )
                USE_SUDO=true
                ;;
            h )
                usage
                ;;
            \? )
                usage
                ;;
        esac
    done
    shift $((OPTIND -1))

    DAWN_DIR="${DAWN_DIR:-$DEFAULT_DAWN_DIR}"
    BUILD_DIR="${BUILD_DIR:-$DAWN_DIR/out/build}"

    echo "Dawn source directory: $DAWN_DIR"
    echo "Dawn build directory:  $BUILD_DIR"
    echo "Use Docker:            $USE_DOCKER"
    echo "Use sudo:              $USE_SUDO"
}

check_dependencies() {
    # Check for gclient (depot_tools)
    if ! command -v gclient &> /dev/null; then
        echo "Error: gclient (depot_tools) not found in PATH."
        echo "Please install depot_tools and add it to your PATH."
        echo "See: https://commondatastorage.googleapis.com/chrome-infra-docs//html/user/depot_tools_tutorial.html#_setting_up"
        exit 1
    fi

    # Check for cargo
    if ! command -v cargo &> /dev/null; then
        echo "Error: cargo not found in PATH. Rust toolchain is required."
        exit 1
    fi

    if $USE_DOCKER; then
        if ! command -v docker &> /dev/null; then
            echo "Error: docker not found in PATH."
            exit 1
        fi

        if $USE_SUDO; then
            DOCKER_CMD="sudo docker"
        else
            DOCKER_CMD="docker"
        fi
    else
        # Find AFL compilers locally
        if [ -n "$AFL_PATH" ]; then
            CC="$AFL_PATH/afl-clang-fast"
            CXX="$AFL_PATH/afl-clang-fast++"
        else
            if command -v afl-clang-fast &> /dev/null && command -v afl-clang-fast++ &> /dev/null; then
                CC="afl-clang-fast"
                CXX="afl-clang-fast++"
            else
                echo "Error: afl-clang-fast or afl-clang-fast++ not found."
                echo "Please set AFL_PATH environment variable or use -a option to point to AFL++ bin directory,"
                echo "or ensure they are in your PATH."
                echo "Alternatively, use -d to build using Docker."
                exit 1
            fi
        fi

        # Verify AFL compilers exist and are executable
        if [ ! -x "$CC" ] || [ ! -x "$CXX" ]; then
            echo "Error: AFL compilers not found or not executable at:"
            echo "  CC:  $CC"
            echo "  CXX: $CXX"
            exit 1
        fi

        # Find LLVM tools
        LLVM_AR=$(command -v llvm-ar-18 || command -v llvm-ar || true)
        LLVM_RANLIB=$(command -v llvm-ranlib-18 || command -v llvm-ranlib || true)

        if [ -z "$LLVM_AR" ]; then
            echo "Error: llvm-ar or llvm-ar-18 not found. AFL++ build requires LLVM tools."
            exit 1
        fi

        if [ -z "$LLVM_RANLIB" ]; then
            echo "Error: llvm-ranlib or llvm-ranlib-18 not found. AFL++ build requires LLVM tools."
            exit 1
        fi
    fi
}

setup_dawn_source() {
    if [ ! -f "$PATCH_FILE" ]; then
        echo "Error: Patch file not found at $PATCH_FILE"
        exit 1
    fi

    # Clone Dawn if it doesn't exist
    if [ ! -d "$DAWN_DIR" ]; then
        echo "Cloning Dawn..."
        git clone https://dawn.googlesource.com/dawn "$DAWN_DIR"
    fi

    cd "$DAWN_DIR"

    # Ensure we are on the correct revision
    echo "Checking out revision $DAWN_REVISION..."
    git checkout "$DAWN_REVISION"

    echo "Setting up .gclient..."
    cp scripts/standalone.gclient .gclient

    echo "Running gclient sync..."
    # gclient sync might return non-zero but still work, as per README.
    gclient sync || echo "gclient sync completed with some errors (this is often fine)"

    echo "Applying patch..."
    # Try to apply patch, if it fails, check if it was already applied
    if git apply --check "$PATCH_FILE" &> /dev/null; then
        git apply "$PATCH_FILE"
        echo "Patch applied successfully."
    elif git apply -R --check "$PATCH_FILE" &> /dev/null; then
        echo "Patch seems to be already applied."
    else
        echo "Error: Could not apply patch. It might be dirty or already modified."
        echo "Dry-run of git apply to show errors:"
        git apply --check "$PATCH_FILE"
        exit 1
    fi
}

configure_build_local() {
    local cmake_args=(
        "-GNinja"
        "-DTINT_BUILD_AFL_FUZZER=ON"
        "-DDAWN_ENABLE_ASAN=ON"
        "-DTINT_BUILD_MSL_WRITER=ON"
        "-DTINT_BUILD_SPV_WRITER=ON"
        "-DTINT_BUILD_HLSL_WRITER=ON"
        "-DDAWN_USE_GLFW=OFF"
        "-DDAWN_BUILD_SAMPLES=OFF"
        "-DTINT_BUILD_CMD_TOOLS=OFF"
        "-DDAWN_USE_X11=OFF"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_CXX_FLAGS=-fuse-ld=lld"
        "-DCMAKE_C_FLAGS=-fuse-ld=lld"
        "-DCMAKE_CXX_COMPILER_AR=$LLVM_AR"
        "-DCMAKE_C_COMPILER_AR=$LLVM_AR"
        "-DCMAKE_CXX_COMPILER_RANLIB=$LLVM_RANLIB"
        "-DCMAKE_C_COMPILER_RANLIB=$LLVM_RANLIB"
    )

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    echo "Running CMake locally..."
    CC="$CC" CXX="$CXX" cmake "${cmake_args[@]}" "$DAWN_DIR"
}

configure_build_docker() {
    mkdir -p "$BUILD_DIR"
    local abs_dawn_dir=$(cd "$DAWN_DIR" && pwd)
    local abs_build_dir=$(cd "$BUILD_DIR" && pwd)

    echo "Running CMake in Docker..."
    $DOCKER_CMD run --rm \
        --user "$(id -u):$(id -g)" \
        -v "$abs_dawn_dir:/dawn" \
        -v "$abs_build_dir:/build" \
        -w /build \
        "$DOCKER_IMAGE" \
        bash -c "CC=afl-clang-fast CXX=afl-clang-fast++ cmake \
            -GNinja \
            -DTINT_BUILD_AFL_FUZZER=ON \
            -DDAWN_ENABLE_ASAN=ON \
            -DTINT_BUILD_MSL_WRITER=ON \
            -DTINT_BUILD_SPV_WRITER=ON \
            -DTINT_BUILD_HLSL_WRITER=ON \
            -DDAWN_USE_GLFW=OFF \
            -DDAWN_BUILD_SAMPLES=OFF \
            -DTINT_BUILD_CMD_TOOLS=OFF \
            -DDAWN_USE_X11=OFF \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_FLAGS=-fuse-ld=lld \
            -DCMAKE_C_FLAGS=-fuse-ld=lld \
            /dawn"
}

configure_build() {
    if $USE_DOCKER; then
        configure_build_docker
    else
        configure_build_local
    fi
}

build_target_local() {
    cd "$BUILD_DIR"
    echo "Building tint_afl_all_fuzzer locally..."
    ninja tint_afl_all_fuzzer
}

build_target_docker() {
    local abs_dawn_dir=$(cd "$DAWN_DIR" && pwd)
    local abs_build_dir=$(cd "$BUILD_DIR" && pwd)

    echo "Building tint_afl_all_fuzzer in Docker..."
    $DOCKER_CMD run --rm \
        --user "$(id -u):$(id -g)" \
        -v "$abs_dawn_dir:/dawn" \
        -v "$abs_build_dir:/build" \
        -w /build \
        "$DOCKER_IMAGE" \
        ninja tint_afl_all_fuzzer
}

build_target() {
    if $USE_DOCKER; then
        build_target_docker
    else
        build_target_local
    fi
}

build_darthshader_afl() {
    echo "Building darthshader-afl..."
    (
        cd "$SCRIPT_DIR/../.."
        cargo +nightly build
    )
}

main() {
    parse_args "$@"
    check_dependencies
    setup_dawn_source
    configure_build
    build_target
    build_darthshader_afl
    echo "Build complete. Fuzzer binary should be at $BUILD_DIR/tint_afl_all_fuzzer"
}

# --- Execute Main ---
main "$@"
