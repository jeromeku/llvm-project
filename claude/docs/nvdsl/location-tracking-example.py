#!/usr/bin/env python3
"""
Example: Adding automatic location tracking to MLIR Python code.

This shows how to wrap your IR generation with automatic location tracking
so that enable_debug_info=True actually shows useful source locations.
"""

import inspect
from contextlib import contextmanager
from mlir import ir
from mlir.dialects import arith, func


# ============================================================================
# Helper: Automatic Location Tracker
# ============================================================================

class AutoLocationTracker:
    """
    Helper class for automatic Python source location tracking.

    Usage:
        tracker = AutoLocationTracker(ctx)

        with tracker.here():  # Captures calling line
            op = create_operation()

        with tracker.named("important_section"):  # Named location
            op = create_operation()
    """

    def __init__(self, ctx=None, enabled=True):
        self.ctx = ctx
        self.enabled = enabled

    @contextmanager
    def here(self):
        """Auto-capture the calling Python source location."""
        if not self.enabled:
            with ir.Location.unknown(context=self.ctx):
                yield
            return

        # Walk up 2 frames: here() -> contextmanager wrapper -> actual caller
        frame = inspect.currentframe().f_back.f_back
        filename = frame.f_code.co_filename
        line = frame.f_lineno

        loc = ir.Location.file(filename, line, 0, context=self.ctx)
        with loc:
            yield loc

    @contextmanager
    def named(self, name):
        """Named semantic location."""
        if not self.enabled:
            with ir.Location.unknown(context=self.ctx):
                yield
            return

        loc = ir.Location.name(name, context=self.ctx)
        with loc:
            yield loc


# ============================================================================
# Example 1: Manual locations with context managers
# ============================================================================

def example_manual_locations():
    """Example showing manual location setting."""
    print("\n=== Example 1: Manual Locations ===\n")

    with ir.Context() as ctx:
        module = ir.Module.create()
        i32 = ir.IntegerType.get_signless(32)

        with ir.InsertionPoint(module.body):
            # Each operation gets explicit location
            with ir.Location.file("example.py", 10, 0):
                c1 = arith.ConstantOp(i32, 1)

            with ir.Location.file("example.py", 11, 0):
                c2 = arith.ConstantOp(i32, 2)

            with ir.Location.file("example.py", 12, 0):
                result = arith.AddIOp(c1, c2)

        print("Module with manual locations:")
        module.operation.print(ir.OpPrintingFlags().enable_debug_info())


# ============================================================================
# Example 2: Automatic location tracking
# ============================================================================

def example_auto_locations():
    """Example showing automatic location tracking."""
    print("\n=== Example 2: Automatic Locations ===\n")

    with ir.Context() as ctx:
        tracker = AutoLocationTracker(ctx)
        module = ir.Module.create()
        i32 = ir.IntegerType.get_signless(32)

        with ir.InsertionPoint(module.body):
            # Each operation automatically gets the line number it's created on
            with tracker.here():  # Line 95 - location captured!
                c1 = arith.ConstantOp(i32, 1)

            with tracker.here():  # Line 98 - different location!
                c2 = arith.ConstantOp(i32, 2)

            with tracker.here():  # Line 101 - another location!
                result = arith.AddIOp(c1, c2)

        print("Module with automatic locations:")
        module.operation.print(ir.OpPrintingFlags().enable_debug_info())


# ============================================================================
# Example 3: Named semantic locations
# ============================================================================

def example_named_locations():
    """Example showing named semantic locations."""
    print("\n=== Example 3: Named Locations ===\n")

    with ir.Context() as ctx:
        tracker = AutoLocationTracker(ctx)
        module = ir.Module.create()
        f32 = ir.F32Type.get()

        with ir.InsertionPoint(module.body):
            with tracker.named("load_constant_1"):
                c1 = arith.ConstantOp(f32, 1.0)

            with tracker.named("load_constant_2"):
                c2 = arith.ConstantOp(f32, 2.0)

            with tracker.named("compute_sum"):
                result = arith.AddFOp(c1, c2)

        print("Module with named locations:")
        module.operation.print(ir.OpPrintingFlags().enable_debug_info())


# ============================================================================
# Example 4: Building a function with locations
# ============================================================================

def example_function_with_locations():
    """Example showing a complete function with location tracking."""
    print("\n=== Example 4: Function with Locations ===\n")

    with ir.Context() as ctx:
        tracker = AutoLocationTracker(ctx)
        module = ir.Module.create()

        i32 = ir.IntegerType.get_signless(32)
        func_type = ir.FunctionType.get([i32, i32], [i32])

        with ir.InsertionPoint(module.body):
            with tracker.named("func_declaration"):
                func_op = func.FuncOp("add_example", func_type)
                func_op.attributes["sym_visibility"] = ir.StringAttr.get("private")

            entry_block = func_op.add_entry_block()

            with ir.InsertionPoint(entry_block):
                arg0, arg1 = entry_block.arguments

                with tracker.named("load_arg0"):
                    val0 = arg0

                with tracker.named("load_arg1"):
                    val1 = arg1

                with tracker.named("compute_result"):
                    result = arith.AddIOp(val0, val1)

                with tracker.named("return_result"):
                    func.ReturnOp([result])

        print("Function with semantic locations:")
        module.operation.print(ir.OpPrintingFlags().enable_debug_info())


# ============================================================================
# Example 5: Integration with PassManager
# ============================================================================

def example_passmanager_with_locations():
    """Example showing PassManager with location tracking."""
    print("\n=== Example 5: PassManager with Locations ===\n")

    with ir.Context() as ctx:
        tracker = AutoLocationTracker(ctx)

        # Create module with locations
        module = ir.Module.create()
        i32 = ir.IntegerType.get_signless(32)

        with ir.InsertionPoint(module.body):
            with tracker.named("constant_1"):
                c1 = arith.ConstantOp(i32, 5)

            with tracker.named("constant_2"):
                c2 = arith.ConstantOp(i32, 5)

            with tracker.named("redundant_add"):
                # This will be canonicalized away
                result = arith.AddIOp(c1, c2)

        print("Original IR:")
        module.operation.print(ir.OpPrintingFlags().enable_debug_info())

        # Run PassManager with debug info enabled
        from mlir.passmanager import PassManager
        import sys
        from io import StringIO

        # Capture pass manager output
        old_stderr = sys.stderr
        sys.stderr = StringIO()

        with module.context:
            pm = PassManager.parse("builtin.module(canonicalize)")
            pm.enable_ir_printing(
                print_before_all=False,
                print_after_all=True,
                enable_debug_info=True,  # Show locations!
            )
            pm.run(module.operation)

        output = sys.stderr.getvalue()
        sys.stderr = old_stderr

        print("\n" + "=" * 70)
        print("PassManager output (with locations):")
        print("=" * 70)
        print(output)


# ============================================================================
# Example 6: Practical NVDSL-style pattern
# ============================================================================

def example_nvdsl_style():
    """Example mimicking NVDSL usage pattern."""
    print("\n=== Example 6: NVDSL-Style Pattern ===\n")

    with ir.Context() as ctx:
        tracker = AutoLocationTracker(ctx)
        module = ir.Module.create()

        i32 = ir.IntegerType.get_signless(32)
        func_type = ir.FunctionType.get([i32], [])

        with ir.InsertionPoint(module.body):
            # Kernel function
            with tracker.named("gpu_kernel_declaration"):
                kernel = func.FuncOp("kernel", func_type)

            entry = kernel.add_entry_block()
            with ir.InsertionPoint(entry):
                tid = entry.arguments[0]

                # Simulate GPU operations with locations
                with tracker.named("get_thread_id"):
                    # In real code: gpu.thread_id()
                    thread_val = tid

                with tracker.named("compute_offset"):
                    offset = arith.ConstantOp(i32, 100)

                with tracker.named("add_offset_to_tid"):
                    my_value = arith.AddIOp(thread_val, offset)

                with tracker.named("kernel_return"):
                    func.ReturnOp([])

        print("NVDSL-style kernel with locations:")
        module.operation.print(ir.OpPrintingFlags().enable_debug_info())


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MLIR Python Location Tracking Examples")
    print("=" * 70)

    example_manual_locations()
    example_auto_locations()
    example_named_locations()
    example_function_with_locations()
    example_passmanager_with_locations()
    example_nvdsl_style()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("1. Use 'with ir.Location.file(...)' for manual locations")
    print("2. Use AutoLocationTracker.here() for automatic Python line tracking")
    print("3. Use tracker.named('...') for semantic location names")
    print("4. Enable debug info in PassManager: enable_debug_info=True")
    print("5. Locations must be SET during IR creation, not just enabled at print time")
    print("=" * 70)
