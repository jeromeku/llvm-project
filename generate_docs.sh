#!/bin/bash

set -euo pipefail

MLIR_INCLUDES=mlir/include
TYPE="dialect"
PATH=build/bin:$PATH
ODS_FILE="mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td"
BASENAME=$(basename ${ODS_FILE} .td)

echo ${BASENAME}

CMD="mlir-tblgen -I${MLIR_INCLUDES} --gen-${TYPE}-doc ${ODS_FILE} -o ${BASENAME}.md"
echo ${CMD}
eval ${CMD}