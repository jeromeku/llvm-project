import logging
import sys

# [%(module)s.%(funcName)s]
LOG_FORMAT = "%(levelname)s::%(name)s [%(pathname)s:%(lineno)d] - %(message)s"


# def configure_mlir_logging(
#     fmt: str = LOG_FORMAT, level: int = logging.DEBUG, stream=None
# ) -> None:
#     """Configure formatting for all loggers under the `mlir` namespace.

#     Does not modify MLIR sources. Attaches a single StreamHandler to the
#     top-level `mlir` logger and disables propagation so messages from any
#     `mlir.*` logger use this formatter and do not get duplicated by root.
#     """
#     mlir_logger = logging.getLogger("mlir")
#     mlir_logger.setLevel(level)

#     # Avoid adding multiple identical handlers if called repeatedly.
#     for h in list(mlir_logger.handlers):
#         if getattr(h, "_mlir_custom", False):
#             # Update formatter/level in case caller wants to change them.
#             h.setFormatter(logging.Formatter(fmt))
#             h.setLevel(level)
#             break
#     else:
#         handler = logging.StreamHandler(stream or sys.stderr)
#         handler.setLevel(level)
#         handler.setFormatter(logging.Formatter(fmt))
#         handler._mlir_custom = True  # sentinel to detect our handler later
#         mlir_logger.addHandler(handler)

#     # Prevent messages from bubbling up to root and being reformatted/duplicated.
#     mlir_logger.propagate = False


# # Also set a sensible root configuration if nothing has configured logging yet.
# # This avoids "No handlers could be found for logger..." warnings when code
# # outside of `mlir` logs.
# if not logging.getLogger().handlers:
#     logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)

# # Apply MLIR-specific formatting by default when this module is imported.
# configure_mlir_logging()

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

from mlir.ir import Diagnostic, Context, DiagnosticHandler, Operation, Location
from mlir._mlir_libs import get_dialect_registry
from mlir._mlir_libs._mlir.ir import MLIRError, _GlobalDebug
#print(help(_GlobalDebug.set_types))
_GlobalDebug.flag = True

print(f"DIALECT_REGISTRY: {get_dialect_registry()}")
registry = get_dialect_registry()

def get_diagnostic_handler(ctx: Context) -> DiagnosticHandler:
    ctx.emit_error_diagnostics = True
    
    def callback(dx: Diagnostic):
        notes = "\n".join(list(map(str, dx.notes)))

        print(f"DIAGNOSTIC:")
        print(f"  message={dx.message}")
        print(f"  notes={notes}")
        return True
    
    handler = ctx.attach_diagnostic_handler(callback)

    return handler

def testDiagnosticNonEmptyNotes():
    ctx = Context()
    # handler = get_diagnostic_handler(ctx)

    loc = Location.unknown(ctx)
    try:
        Operation.create("arith.addi", loc=loc).verify()
    except MLIRError as e:
        print(f"{e}")
    # finally:
    #     print(f"{handler.had_error=}")

if __name__ == "__main__":
    testDiagnosticNonEmptyNotes()
