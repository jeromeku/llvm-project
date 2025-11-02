from mlir.ir import _GlobalDebug
from mlir import ir
from mlir.passmanager import PassManager

MLIR = """
func.func @main() -> (i1, i1) {
  // CHECK-LABEL: func @main
  // CHECK-NEXT: arith.constant true
  // CHECK-NEXT: return
  %true = arith.constant true
  %true1 = arith.constant true
  return %true, %true1 : i1, i1
}
"""
with ir.Context():
    module = ir.Module.parse(MLIR)
    print("Module before pass:")
    module.operation.print()
    # Enable LLVM debug output
    _GlobalDebug.flag = True
    # _GlobalDebug.set_types(["cse", "canonicalize"])
    pm_mgr = PassManager.parse("builtin.module(cse,canonicalize)")
    pm_mgr.run(module.operation)
    print("Module after pass:")
    module.operation.print() 
    # Disable when done
    _GlobalDebug.flag = False