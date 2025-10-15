#!/bin/bash

set -euo pipefail

TEST=$1

LLVM_ROOT=/home/jeromeku/llvm-project
MLIR_PYTHON_PACKAGE=${LLVM_ROOT}/build/tools/mlir/python_packages/mlir_core
MLIR_LIB_DIR=${LLVM_ROOT}/build/lib
MLIR_BIN_DIR=${LLVM_ROOT}/build/bin

if [[ ! -d ${MLIR_PYTHON_PACKAGE} ]]; then
    echo "MLIR_PYTHON_PACKAGE not set"
    exit 1
fi

export PATH=${MLIR_BIN_DIR}:${PATH}
export PYTHONPATH=${MLIR_PYTHON_PACKAGE}

python ${TEST}