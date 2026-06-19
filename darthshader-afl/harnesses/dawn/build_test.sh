#!/bin/bash
# Test script for build.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_DIR="$SCRIPT_DIR/test_out"
DAWN_DIR="$SCRIPT_DIR/test_dawn"

KEEP_BUILD=false
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -k|--keep)
            KEEP_BUILD=true
            shift
            ;;
        -b)
            BUILD_DIR="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        -s)
            DAWN_DIR="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Test Options:"
            echo "  -k, --keep    Keep build directory (do not delete test_out)"
            echo "  -h, --help    Show this help message"
            echo ""
            echo "Build Options (forwarded to build.sh):"
            "$SCRIPT_DIR/build.sh" -h
            exit 0
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

# Clean up previous build directory to ensure a clean build (if not keeping)
if [ "$KEEP_BUILD" = false ]; then
    echo "Cleaning up previous build directory..."
    rm -rf "$BUILD_DIR"
fi

# Run build.sh
echo "Running build.sh..."
"$SCRIPT_DIR/build.sh" -s "$DAWN_DIR" -b "$BUILD_DIR" "${FORWARD_ARGS[@]}"

BINARY="$BUILD_DIR/tint_afl_all_fuzzer"
RUST_BINARY="$SCRIPT_DIR/../../../target/debug/darthshader-afl"

echo "Checking if Dawn harness binary exists..."
if [ ! -f "$BINARY" ]; then
    echo "FAIL: Dawn harness binary not found at $BINARY"
    exit 1
fi

if [ ! -x "$BINARY" ]; then
    echo "FAIL: Dawn harness binary is not executable"
    exit 1
fi

echo "Checking if darthshader-afl binary exists..."
if [ ! -f "$RUST_BINARY" ]; then
    echo "FAIL: Rust binary not found at $RUST_BINARY"
    exit 1
fi

if [ ! -x "$RUST_BINARY" ]; then
    echo "FAIL: Rust binary is not executable"
    exit 1
fi

echo "Running fuzzer via darthshader-afl..."
TEST_CORPUS="$BUILD_DIR/test_corpus"
set +e
# Run with 1 iteration to verify it starts up and runs the loop once
OUTPUT=$( "$RUST_BINARY" -d "$BINARY" -o "$TEST_CORPUS" --iterations 1 2>&1 )
EXIT_CODE=$?
set -e

echo "Fuzzer exited with code: $EXIT_CODE"
echo "Fuzzer output:"
echo "$OUTPUT"

if [ $EXIT_CODE -ne 0 ]; then
    echo "FAIL: Fuzzer failed to run successfully."
    if [[ "$OUTPUT" == *"error while loading shared libraries"* ]]; then
        echo "FAIL: Missing shared libraries."
    fi
    exit 1
fi

echo "PASS: Fuzzer ran successfully."

# Clean up build dir on success, keep dawn source to speed up next run
if [ "$KEEP_BUILD" = false ]; then
    echo "Cleaning up build directory..."
    rm -rf "$BUILD_DIR"
fi
echo "Test passed successfully."
