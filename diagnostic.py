from mlir.ir import Context, DiagnosticHandler, Location
def log(diag):
    # Minimal, robust printing; 'diag' pretty-prints severity/location/message.
    print(diag)
    breakpoint()
    for n in diag.notes:
        print("  note:", n)

def run():
    with Context() as ctx, Location.unknown(ctx):
        ctx.enable_multithreading(False)
        ctx.emit_error_diagnostics = True
        ctx.attach_diagnostic_handler(log)
    
run()