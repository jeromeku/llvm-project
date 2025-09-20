mlir-opt demo3-lowering-mlir/demo3-0-linalg-on-tensor.mlir \
--pass-pipeline="builtin.module(one-shot-bufferize,convert-linalg-to-loops,convert-scf-to-cf)" \
--mlir-print-ir-after-all \
--mlir-print-ir-tree-dir="linalg_to_cf" \
-o demo3.cf.mlir