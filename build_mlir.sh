#!/bin/bash

set -euo pipefail

# Configuration
BUILD_PATH="build"
INSTALL_PATH="install"
REPO_ROOT="$(git rev-parse --show-toplevel)"
# CUDA_HOME="/data/jeromeku/cuda-toolkit-13"

# Build flags
CONFIGURE="0"
BUILD="0"

# Log files
CONFIG_LOG="_cmake_config.log"
BUILD_LOG="_cmake_build.log"

# Function to configure the build
configure_build() {
    echo "Configuring LLVM/MLIR build..."
    # rm -rf "${BUILD_PATH}"
    
    cmake -G Ninja -S llvm -B "${BUILD_PATH}" \
        -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
        -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DMLIR_ENABLE_CUDA_RUNNER=ON \
        -DPython3_EXECUTABLE="$(which python)" \
        -DCMAKE_VERBOSE_MAKEFILE=1 \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DLLVM_CCACHE_BUILD=OFF \
        -DCMAKE_C_COMPILER=clang-22 \
        -DCMAKE_CXX_COMPILER=clang++-22 \
        -DLLVM_ENABLE_LLD=ON \
        -DLLVM_OPTIMIZED_TABLEGEN=ON \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
        -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PATH}" \
        2>&1 | tee "${CONFIG_LOG}"
}

# Function to build the project
build_project() {
    echo "Building LLVM/MLIR..."
    cmake --build "${BUILD_PATH}" -j8 2>&1 | tee "${BUILD_LOG}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [configure|build]"
    echo "  configure  - Configure the CMake build"
    echo "  build      - Build the project"
    echo ""
    echo "Set CONFIGURE=1 or BUILD=1 environment variables, or modify the script directly"
    exit 1
}

# Main execution logic
case "${1:-}" in
    configure)
        configure_build
        ;;
    build)
        build_project
        ;;
    *)
        show_usage
        ;;
esac

# TODO: Install Python dependencies
# uv pip install -r mlir/python/requirements.txt