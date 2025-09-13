# Ch0.py (trace-only variant): build IR, do not run, and dump IR per pass.

import os
from mlir import ir
from mlir.passmanager import PassManager
from mlir.dialects import gpu
from mlir.passmanager import PassManager
from tools.nvdsl import NVDSL
from tools.nvgpucompiler import NvgpuCompiler
from contextlib import contextmanager

TRACE_DIR = os.environ.get("NVDSL_TRACE_DIR", "./mlir-nvdsl-trace")
TARGET_CHIP = os.environ.get("NVDSL_CHIP", "sm_90a")
TARGET_PTX  = os.environ.get("NVDSL_PTX", "+ptx87")  # e.g., +ptx80, +ptx86, +ptx87
COMPILE_ONLY = os.environ.get("NVDSL_COMPILE_ONLY", "0") == "1"

@NVDSL.mlir_func(save_ir=True, compile_only=COMPILE_ONLY)
def main(alpha):
    @NVDSL.mlir_gpu_launch(grid=(1, 1, 1), block=(4, 1, 1))
    def kernel():
        tidx = gpu.thread_id(gpu.Dimension.x)
        myValue = alpha + tidx
        gpu.printf("GPU thread %llu has %llu\n", [tidx, myValue])
    kernel()

@contextmanager
def cuda_context(device=0):
    # import ctypes
    # cuda = ctypes.CDLL("libcuda.so")
    # cuda.cuInit(0)
    from cuda.core.experimental import Device
    dev0 = Device(device)
    dev0.set_current()
    yield dev0

if __name__ == "__main__":
    # import ctypes, os
    # lib = ctypes.CDLL(os.environ["SUPPORT_LIB"])
    # lib.mgpuSetDefaultDevice.argtypes = [ctypes.c_int]
    # lib.mgpuSetDefaultDevice(0)
    alpha = 100

    results = main(alpha)

    if COMPILE_ONLY:
        module, compiler = results
        module: ir.Module
        compiler: NvgpuCompiler

        with module.context as ctx, ir.Location.unknown():
            print(f"Compiler pipeline: {compiler.pipeline}")
            pm = PassManager.parse(compiler.pipeline)
            ctx.enable_multithreading(False)
            ctx.emit_error_diagnostics = True
            # Print before/after every pass, include locations, and dump each pass’s IR to a directory.
            pm.enable_ir_printing(
                print_before_all=True,
                print_after_all=True,
                print_module_scope=True,
                print_after_change=False,
                print_after_failure=True,
                enable_debug_info=True,
                tree_printing_dir_path=os.environ.get("NVDSL_TRACE_DIR")
            )
            pm.run(module.operation)
