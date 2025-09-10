import sys
import os
sys.path.insert(0, "./build/tools/mlir/python_packages/mlir_core")
os.environ["CUDA_ROOT"] = "/home/jeromeku/cuda-toolkit"

from mlir import ir
from mlir.passmanager import PassManager
from mlir.dialects import gpu, nvvm
with ir.Context() as ctx, ir.Location.unknown():
    ctx.emit_error_diagnostics = True
    src = r'''
    module attributes {gpu.container_module} {
      // chip attribute drives the target; sm_70 is broadly supported
      gpu.module @km [#nvvm.target<chip = "sm_70">] {
        llvm.func @k(%x: i32) attributes {gpu.kernel} { llvm.return }
      }
    }'''
    m = ir.Module.parse(src)
    pm = PassManager.parse("builtin.module(gpu-module-to-binary{format=isa})")  # emits PTX
    pm.run(m.operation)
    # A gpu.binary object attribute with PTX bytes is attached to the module
    print(m)
