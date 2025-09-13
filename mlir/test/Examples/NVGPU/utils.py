from mlir.ir import Context, Location
from mlir.passmanager import PassManager
import os

def run_pipeline(module, pipeline):
    with module.context as ctx, Location.unknown():
        # print(f"Compiler pipeline: {pipeline}")
        pm = PassManager.parse(pipeline)
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
