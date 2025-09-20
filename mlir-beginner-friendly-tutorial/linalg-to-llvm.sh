CF_MLIR_PATH="demo3.cf.mlir"
LLVM_MLIR_PATH="demo3.llvm.mlir"

mlir-opt demo3-lowering-mlir/demo3-0-linalg-on-tensor.mlir \
--pass-pipeline="builtin.module(one-shot-bufferize,convert-linalg-to-loops,convert-scf-to-cf)" \
--mlir-print-ir-after-all \
--mlir-print-ir-tree-dir="linalg-to-cf" \
-o ${CF_MLIR_PATH}

mlir-opt ${CF_MLIR_PATH} \
    -convert-func-to-llvm \
    -convert-cf-to-llvm \
    -finalize-memref-to-llvm \
    -convert-arith-to-llvm \
    -reconcile-unrealized-casts \
    -canonicalize \
    --mlir-print-ir-after-all \
    --mlir-print-ir-tree-dir="cf-to-llvm" \
    -o ${LLVM_MLIR_PATH}

