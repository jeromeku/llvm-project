from mlir import ir
from mlir.passmanager import PassManager

def log(diag):
    # Prints severity, location, message; notes follow.
    print(f"DIAGNOSTIC_ERROR: {diag.message}")
    breakpoint()
    notes = "\n".join(str(n) for n in diag.notes)
    print(f"DAIGNOSTIC_NOTES: {notes}")

with ir.Context() as ctx, ir.Location.unknown():
    # Strict + serialized while debugging
    # ctx.allow_unregistered_dialects = False
    ctx.enable_multithreading(False)
    ctx.emit_error_diagnostics = True

    # Keep the handler alive across *everything* that can emit diagnostics
    with ctx.attach_diagnostic_handler(log):
        # Register dialects *before* parse (import registers into the current ctx)
        # import mlir.dialects.gpu as gpu
        # import mlir.dialects.nvvm as nvvm
        # import mlir.dialects.llvm as llvm
        # import mlir.dialects.func as func

        # Always pass ctx explicitly
        mod = ir.Module.parse(r"""
          module attributes {gpu.container_module} {
            gpu.module @m [#nvvm.target<chip = "sm_70">] {
              llvm.func @k() attributes {gpu.kernel} { llvm.return }
            }
          }
        """, ctx)

        # Build/run pipeline inside the same handler scope
        pm = PassManager.parse("any(gpu-module-to-binary{format=llvm})", context=ctx)
        # pm.enable_ir_printing(print_module_scope=True, print_before=True, print_after=True)
        # pm.enable_timing()

        try:
            pm.run(mod.operation)
        except ir.MLIRError as e:
            print("Pass pipeline failed:", e)
            print(f"e.error_diagnostics: {[str(diag) for diag in e.error_diagnostics]}")

        # If you want CI to fail loudly when any error happened:
        # h = ctx.attach_diagnostic_handler(lambda d: None)  # no-op just to get handle
        # if h.had_error: raise RuntimeError("MLIR reported errors")
