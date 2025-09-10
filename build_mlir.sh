#!/bin/bash

set -euo pipefail
BUILD_PATH=build
INSTALL_PATH=install
REPO_ROOT="$(git rev-parse --show-toplevel)"
rm -rf ${BUILD_PATH}
CUDA_HOME=/home/jeromeku/cuda-toolkit

CMD="cmake -G Ninja -S llvm -B${BUILD_PATH} \
    -DLLVM_ENABLE_PROJECTS=\"mlir;llvm;lld\" \
    -DLLVM_TARGETS_TO_BUILD=\"Native;NVPTX\" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DPython3_EXECUTABLE=$(which python) \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DLLVM_CCACHE_BUILD=OFF \
    -DCMAKE_C_COMPILER=clang-21 \
    -DCMAKE_CXX_COMPILER=clang++-21 \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
    -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}"

echo ${CMD}
eval ${CMD}

#uv pip install -r mlir/python/requirements.txt