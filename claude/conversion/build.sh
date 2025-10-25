#!/bin/bash

# Build script for MLIR Dialect Conversion Tutorial
# This script builds the tutorial examples using your existing MLIR build.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}MLIR Dialect Conversion Tutorial - Build Script${NC}"
echo "================================================"

# Find the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Tutorial directory: $SCRIPT_DIR"

# Detect MLIR build directory
# Try a few common locations relative to llvm-project
LLVM_PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
MLIR_BUILD_DIR=""

if [ -d "$LLVM_PROJECT_DIR/build" ]; then
    MLIR_BUILD_DIR="$LLVM_PROJECT_DIR/build"
elif [ -d "$LLVM_PROJECT_DIR/cmake-build-debug" ]; then
    MLIR_BUILD_DIR="$LLVM_PROJECT_DIR/cmake-build-debug"
elif [ -d "$LLVM_PROJECT_DIR/cmake-build-release" ]; then
    MLIR_BUILD_DIR="$LLVM_PROJECT_DIR/cmake-build-release"
fi

# Allow override via environment variable
if [ -n "$LLVM_BUILD_DIR" ]; then
    MLIR_BUILD_DIR="$LLVM_BUILD_DIR"
fi

if [ -z "$MLIR_BUILD_DIR" ]; then
    echo -e "${RED}Error: Could not find MLIR build directory${NC}"
    echo "Please set LLVM_BUILD_DIR environment variable:"
    echo "  export LLVM_BUILD_DIR=/path/to/llvm-project/build"
    exit 1
fi

echo "Using MLIR build: $MLIR_BUILD_DIR"

# Check if build directory exists
if [ ! -d "$MLIR_BUILD_DIR" ]; then
    echo -e "${RED}Error: MLIR build directory does not exist: $MLIR_BUILD_DIR${NC}"
    exit 1
fi

# Check for required CMake files
if [ ! -f "$MLIR_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo -e "${RED}Error: MLIRConfig.cmake not found in $MLIR_BUILD_DIR${NC}"
    echo "Please build MLIR first"
    exit 1
fi

# Create build directory
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

echo ""
echo "Configuring CMake..."
cd "$BUILD_DIR"

cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DMLIR_DIR="$MLIR_BUILD_DIR/lib/cmake/mlir" \
    -DLLVM_DIR="$MLIR_BUILD_DIR/lib/cmake/llvm" \
    -DLLVM_EXTERNAL_LIT="$MLIR_BUILD_DIR/bin/llvm-lit"

echo ""
echo -e "${GREEN}Building...${NC}"
ninja

echo ""
echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "Executables are in: $BUILD_DIR/bin/"
echo "  - simple-opt  : Example 1 (Simple arithmetic conversion)"
echo "  - typed-opt   : Example 2 (Type conversion)"
echo "  - partial-opt : Example 3 (Partial conversion)"
echo ""
echo "To test the examples, run:"
echo "  ./test.sh"
