# Ch0.py (trace-only variant): build IR, do not run, and dump IR per pass.

import os
from mlir import ir
from mlir.passmanager import PassManager
from mlir.dialects import gpu
from tools.nvdsl import NVDSL

TRACE_DIR = os.environ.get("NVDSL_TRACE_DIR", "./mlir-nvdsl-trace")
TARGET_CHIP = os.environ.get("NVDSL_CHIP", "sm_90a")
TARGET_PTX  = os.environ.get("NVDSL_PTX", "+ptx87")  # e.g., +ptx80, +ptx86, +ptx87

@NVDSL.mlir_func
def main(alpha):
    @NVDSL.mlir_gpu_launch(grid=(1, 1, 1), block=(4, 1, 1))
    def kernel():
        tidx = gpu.thread_id(gpu.Dimension.x)
        myValue = alpha + tidx
        gpu.printf("GPU thread %llu has %llu\n", [tidx, myValue])
    kernel()

if __name__ == "__main__":
    # IMPORTANT: make sure the decorator returns (module, engine) and does not invoke.
    os.environ["NVDSL_COMPILE_ONLY"] = "1"

    alpha = 100
    mod, engine = main(alpha)   # <-- only builds & JIT-compiles; does not run
    
    if False:
        # Build a pass pipeline and enable verbose IR printing.
        #
        # gpu-lower-to-nvvm-pipeline handles the typical path:
        #   {arith,memref,scf,vector,gpu,nvgpu} -> NVVM + host LLVM dialect
        # Add gpu-module-to-binary{format=isa} to attach PTX text into the IR
        # (as a gpu.binary attribute) without ever launching.
        with ir.Context() as ctx, ir.Location.unknown():

            pipeline = (
                "builtin.module("
                    f"gpu-lower-to-nvvm-pipeline{{cubin-chip={TARGET_CHIP} cubin-features={TARGET_PTX} opt-level=3}} , "
                    f"gpu.module(gpu-module-to-binary{{format=isa cubin-chip={TARGET_CHIP} cubin-features={TARGET_PTX} opt-level=3}})"
                ")"
            )

            pm = PassManager.parse(pipeline)

            # Turn on per-pass IR dumps (before & after), include debug locs, and write a tree of .mlir files.
            pm.enable_ir_printing(
                print_before_all=True,
                print_after_all=True,
                print_after_change=False,
                print_after_failure=True,
                print_module_scope=True,
                enable_debug_info=True,
                large_elements_limit=64,
                tree_printing_dir_path=TRACE_DIR,
            )
            pm.enable_timing()

            # Run the pipeline over the module we just built.
            pm.run(mod.operation)

            # Optional: also print the final IR to stdout (comment out if too chatty).
            print("\n// === Final IR after pipeline ===")
            print(mod)
