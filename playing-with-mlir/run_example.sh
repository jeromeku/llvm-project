#!/bin/bash

set -euo pipefail

# LLVM_PREFIX=/home/jeromeku/mlir/llvm-project
# FILENAME=$1
# EXECUTABLE="playing-with-mlir"
# CXXFLAGS="$(llvm-config --cxxflags)"    # includes -I and -f* you need
# LDFLAGS="$(llvm-config --ldflags)"
# LLVMLIBS="$(llvm-config --libs core support) $(llvm-config --system-libs)"

# clang++ 1-ir-traversal/1-ir-traversal.cpp -g -O0 -fno-exceptions -fno-rtti \
#   -I /home/jeromeku/mlir/llvm-project/mlir/include \
#   -I /home/jeromeku/mlir/llvm-project/build/tools/mlir/include \
#   $CXXFLAGS \
#   -L /home/jeromeku/mlir/llvm-project/build/lib \
#   -Wl,--start-group \
#     -lMLIRIR -lMLIRParser -lMLIRSupport -lMLIRPass -lMLIRTransforms \
#     -lMLIRDialectUtils -lMLIRFuncDialect -lMLIRArithDialect \
#     $LLVMLIBS \
#   -Wl,--end-group \
#   -Wl,-rpath=/home/jeromeku/mlir/llvm-project/build/lib \
#   -o playing-with-mlir

# CMD="clang++ ${FILENAME} -g -O0 -fno-exceptions -fno-rtti \
#   -o ${EXECUTABLE} \
#   -I $LLVM_PREFIX/mlir/include \
#   -I $LLVM_PREFIX/build/tools/mlir/include \
#   -I $LLVM_PREFIX/llvm/include \
#   -I $LLVM_PREFIX/build/include \
#   -L$LLVM_PREFIX/build/lib \
#   -lMLIR -lLLVM -Wl,-rpath=$LLVM_PREFIX/build/lib"
# echo ${CMD}
# eval ${CMD}

# ./${EXECUTABLE} $input $quoted_args"

SOURCE=${1:-"1-ir-traversal/1-ir-traversal-solution.cpp"}
INPUT_MLIR=${2:-"1-ir-traversal/1-input1.mlir"}
quoted_args=""
if [ "$#" -gt 0 ]; then
    quoted_args=$(printf '%q ' "$@")
    quoted_args=${quoted_args//\\|/|}
    quoted_args=${quoted_args% }   # remove trailing space
fi

COMPILE_CMD="clang++ ${SOURCE} \
-fno-exceptions -fno-rtti -o /tmp/playing-with-mlir \
-I /home/mlir/llvm-project/install/include -L/home/mlir/llvm-project/install/lib \
-lMLIR -lLLVM -Wl,-rpath=/home/mlir/llvm-project/install/lib"
RUN_CMD="/tmp/playing-with-mlir ${INPUT_MLIR} ${quoted_args}"

sudo docker run --rm -it -v /home/jeromeku/mlir/llvm-project/playing-with-mlir:/tmp/tutorial \
-w /tmp/tutorial jokereph/mlir-tutorial:debug \
bash -c "${COMPILE_CMD} && ${RUN_CMD}" 