#!/bin/bash

set -euo pipefail

EXAMPLE=${1:-Ch1.py}
BASE_NAME=$(basename "$EXAMPLE")

LLVM_HOME="/home/jeromeku/llvm-project"
export SUPPORT_LIB="${LLVM_HOME}/build/lib/libmlir_cuda_runtime.so"
export PYTHONPATH="${LLVM_HOME}/build/tools/mlir/python_packages/mlir_core"
export CUDA_ROOT="/home/jeromeku/cuda-toolkit"
export MLIR_CUDA_DEBUG=1
export PATH="/home/jeromeku/cuda-toolkit/bin:$PATH"
export NVDSL_COMPILE_ONLY=0
export NVDSL_TRACE_DIR="${BASE_NAME}.trace.mlir"

# export CUDA_VISIBLE_DEVICES=0

if [[ ! -d ${LLVM_HOME} ]]; then
    echo "LLVM_HOME ${LLVM_HOME} not found"
    exit 1
fi

if [[ ! -f ${SUPPORT_LIB} ]]; then
    echo "SUPPORT_LIB ${SUPPORT_LIB} not found"
    exit 1
fi

if [[ ! -d ${PYTHONPATH} ]]; then
    echo "PYTHONPATH ${PYTHONPATH} not found"
    exit 1
fi

if [[ ! -f ${EXAMPLE} ]]; then
    echo "Example ${EXAMPLE} not found"
    exit 1
fi

CMD="python ${EXAMPLE}"
eval ${CMD}