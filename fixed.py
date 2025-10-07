# REQUIRES: host-supports-nvptx
# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.passmanager import *
import mlir.dialects.gpu as gpu
import mlir.dialects.gpu.passes  # registers gpu passes
import mlir.dialects.nvvm as nvvm  # ensure #nvvm.* parses
import mlir.dialects.llvm as llvm  # ensure llvm.func parses

def run(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown(ctx):
        # Optional: make failures throw instead of tripping the context dtor.
        # ctx.allow_unregistered_dialects = False
        f(ctx)
    return f

# CHECK-LABEL: testGPUToLLVMBin
@run
def testGPUToLLVMBin(ctx):
    module = Module.parse(
        r"""
module attributes {gpu.container_module} {
  gpu.module @kernel_module1 [#nvvm.target<chip = "sm_70">] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
""",
        ctx,
    )

    # Build the pass manager in the SAME context and run.
    pm = PassManager(context=ctx)                # or PassManager(context=ctx, nest=True)
    pm.add("gpu-module-to-binary{format=llvm}")  # relies on mlir.dialects.gpu.passes
    pm.run(module.operation)

    # CHECK-LABEL: gpu.binary @kernel_module1
    print(module)

    # Grab the produced gpu.binary op and its first object attr.
    bin_op = module.body.operations[0]     # gpu.binary
    obj_attr = bin_op.objects[0]           # ArrayAttr element (a #gpu.object)
    o = gpu.ObjectAttr(obj_attr)

    # CHECK: #gpu.object<#nvvm.target<chip = "sm_70">, offload = "{{.*}}">
    print(o)
    # CHECK: #nvvm.target<chip = "sm_70">
    print(o.target)
    # CHECK: offload
    print(gpu.CompilationTarget(o.format))
    # CHECK: b'{{.*}}'
    print(o.object)
    # CHECK: None
    print(o.properties)
