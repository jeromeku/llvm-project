# MLIR Python Location Tracking Guide

## Problem: All Locations Show as `loc(unknown)`

When you enable `enable_debug_info=True` in PassManager, but all your IR dumps show `loc(unknown)`, it's because **MLIR doesn't automatically track source locations**. You must explicitly set them during IR construction.

## Solution Overview

There are three approaches:

1. **Manual Location Setting** - Explicitly set locations using context managers
2. **Automatic Location Tracking** - Use Python's `inspect` module to capture source locations
3. **Named Locations** - Use semantic names instead of file/line information

---

## Approach 1: Manual Location Setting

### Basic Usage with Context Manager

The `Location` class is a context manager. Any operations created within the context automatically get that location:

```python
from mlir import ir
from mlir.dialects import arith

with ir.Context() as ctx:
    # All operations in this block will have file/line location
    with ir.Location.file("my_script.py", line=10, col=5):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            # This operation gets loc("my_script.py":10:5)
            i32 = ir.IntegerType.get_signless(32)
            c5 = arith.ConstantOp(i32, 5)
            print(c5.operation.location)  # loc("my_script.py":10:5)
```

### Multiple Locations in Same Code

You can nest location context managers to set different locations for different operations:

```python
with ir.Context() as ctx, ir.Location.unknown():
    module = ir.Module.create()

    with ir.InsertionPoint(module.body):
        i32 = ir.IntegerType.get_signless(32)

        # First operation at line 10
        with ir.Location.file("script.py", 10, 0):
            c1 = arith.ConstantOp(i32, 1)

        # Second operation at line 11
        with ir.Location.file("script.py", 11, 0):
            c2 = arith.ConstantOp(i32, 2)

        # Third operation at line 12
        with ir.Location.file("script.py", 12, 0):
            sum = arith.AddIOp(c1, c2)

# When printed with enable_debug_info=True:
# %c1 = arith.constant 1 : i32 loc("script.py":10:0)
# %c2 = arith.constant 2 : i32 loc("script.py":11:0)
# %0 = arith.addi %c1, %c2 : i32 loc("script.py":12:0)
```

---

## Approach 2: Automatic Location Tracking

### Using Python's `inspect` Module

You can create a helper that automatically captures the calling location:

```python
import inspect
from mlir import ir

def auto_location(ctx=None):
    """Automatically capture the calling location from Python source."""
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    line = frame.f_lineno
    # Column information not available from inspect, use 0
    return ir.Location.file(filename, line, 0, context=ctx)

# Usage:
with ir.Context() as ctx:
    module = ir.Module.create()

    with ir.InsertionPoint(module.body):
        i32 = ir.IntegerType.get_signless(32)

        with auto_location(ctx):  # Captures this line number
            c1 = arith.ConstantOp(i32, 1)

        with auto_location(ctx):  # Captures this different line number
            c2 = arith.ConstantOp(i32, 2)
```

### Location Tracking Decorator

For functions that generate IR, you can create a decorator:

```python
import functools
import inspect
from mlir import ir

def with_location(func):
    """Decorator that wraps function execution in auto-location context."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        line = frame.f_lineno

        # Get context from first argument if it's a Context
        ctx = None
        if args and isinstance(args[0], ir.Context):
            ctx = args[0]

        loc = ir.Location.file(filename, line, 0, context=ctx)
        with loc:
            return func(*args, **kwargs)
    return wrapper

# Usage:
@with_location
def create_constant(value):
    i32 = ir.IntegerType.get_signless(32)
    return arith.ConstantOp(i32, value)

with ir.Context() as ctx:
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        c1 = create_constant(1)  # Gets location of this call site
        c2 = create_constant(2)  # Gets location of this call site
```

---

## Approach 3: Named Locations

Instead of file/line locations, you can use semantic names:

```python
with ir.Context() as ctx:
    module = ir.Module.create()

    with ir.InsertionPoint(module.body):
        i32 = ir.IntegerType.get_signless(32)

        with ir.Location.name("load_input_A"):
            load_a = memref.LoadOp(...)

        with ir.Location.name("load_input_B"):
            load_b = memref.LoadOp(...)

        with ir.Location.name("compute_result"):
            result = arith.AddFOp(load_a, load_b)

# When printed:
# %0 = memref.load ... loc("load_input_A")
# %1 = memref.load ... loc("load_input_B")
# %2 = arith.addf %0, %1 : f32 loc("compute_result")
```

### Nested Named Locations

You can nest named locations to create hierarchical information:

```python
with ir.Location.name("gemm_kernel"):
    with ir.Location.name("load_inputs"):
        # All operations here get loc("load_inputs"("gemm_kernel"))
        pass

    with ir.Location.name("compute"):
        # All operations here get loc("compute"("gemm_kernel"))
        pass
```

---

## Integrating with NVDSL

### Modifying NVDSL to Track Locations

You can modify the NVDSL builder to accept and use locations:

```python
class NVDSL:
    # ... existing code ...

    @staticmethod
    def mlir_func(save_ir=False, compile_only=False):
        def decorator(py_func):
            @functools.wraps(py_func)
            def wrapper(*args):
                # Capture the location where the decorated function is called
                frame = inspect.currentframe().f_back
                filename = frame.f_code.co_filename
                line = frame.f_lineno

                # Generate MLIR Context and start generating IR
                with ir.Context() as ctx:
                    # Use the captured location for the module
                    with ir.Location.file(filename, line, 0):
                        types = []
                        for arg in args:
                            types.append(get_mlir_ty(arg))

                        # Rest of the module generation...
                        module = ir.Module.create()
                        # ...
            return wrapper
        return decorator
```

### Example: Adding Locations to GPU Launch

```python
@NVDSL.mlir_func(save_ir=True, compile_only=True)
def main(alpha):
    # Capture location for the kernel launch
    frame = inspect.currentframe()
    launch_loc = ir.Location.file(__file__, frame.f_lineno, 0)

    @NVDSL.mlir_gpu_launch(grid=(1, 1, 1), block=(4, 1, 1), location=launch_loc)
    def kernel():
        # Each operation can have its own location
        with ir.Location.name("get_thread_id"):
            tidx = gpu.thread_id(gpu.Dimension.x)

        with ir.Location.name("compute_value"):
            myValue = alpha + tidx

        with ir.Location.name("print_output"):
            gpu.printf("GPU thread %llu has %llu\n", [tidx, myValue])

    kernel()
```

---

## Complete Example: Auto-Tracking Wrapper

Here's a complete helper class for automatic location tracking:

```python
import inspect
from contextlib import contextmanager
from mlir import ir

class LocationTracker:
    """Helper class for automatic location tracking in MLIR Python."""

    def __init__(self, ctx=None):
        self.ctx = ctx
        self.enabled = True

    @contextmanager
    def auto(self):
        """Context manager that automatically captures calling location."""
        if not self.enabled:
            yield
            return

        frame = inspect.currentframe().f_back.f_back
        filename = frame.f_code.co_filename
        line = frame.f_lineno

        loc = ir.Location.file(filename, line, 0, context=self.ctx)
        with loc:
            yield loc

    @contextmanager
    def named(self, name):
        """Context manager with a semantic name."""
        if not self.enabled:
            yield
            return

        loc = ir.Location.name(name, context=self.ctx)
        with loc:
            yield loc

    def disable(self):
        """Disable location tracking (use unknown locations)."""
        self.enabled = False

    def enable(self):
        """Enable location tracking."""
        self.enabled = True

# Usage:
with ir.Context() as ctx:
    tracker = LocationTracker(ctx)
    module = ir.Module.create()

    with ir.InsertionPoint(module.body):
        i32 = ir.IntegerType.get_signless(32)

        with tracker.auto():  # Captures this line
            c1 = arith.ConstantOp(i32, 1)

        with tracker.named("important_constant"):  # Named location
            c2 = arith.ConstantOp(i32, 2)

        with tracker.auto():  # Captures this line
            result = arith.AddIOp(c1, c2)

# Result with enable_debug_info=True:
# %c1 = arith.constant 1 : i32 loc("/path/to/script.py":15:0)
# %c2 = arith.constant 2 : i32 loc("important_constant")
# %0 = arith.addi %c1, %c2 : i32 loc("/path/to/script.py":18:0)
```

---

## PassManager Integration

Once you've added locations to your IR, enable debug info in PassManager:

```python
from mlir.passmanager import PassManager

with module.context as ctx, ir.Location.unknown():
    pm = PassManager.parse("builtin.module(canonicalize)")

    # Enable debug info to see locations in IR dumps
    pm.enable_ir_printing(
        print_before_all=True,
        print_after_all=True,
        print_module_scope=True,
        enable_debug_info=True,  # <-- Show locations
        tree_printing_dir_path="./ir_dumps"
    )

    pm.run(module.operation)
```

---

## Location Types Reference

### Available Location Types

**File Location**:
```python
# Basic file location
loc = ir.Location.file("script.py", line=10, col=5)

# File location with range
loc = ir.Location.file("script.py", line=10, col=5, end_line=10, end_col=20)
```

**Named Location**:
```python
# Simple name
loc = ir.Location.name("my_operation")

# Named with child location
child_loc = ir.Location.file("script.py", 10, 5)
loc = ir.Location.name("my_operation", child_loc)
# Prints as: loc("my_operation"("script.py":10:5))
```

**CallSite Location** (for representing call stacks):
```python
callee = ir.Location.file("util.py", 50, 10)
frame1 = ir.Location.file("helper.py", 100, 5)
frame2 = ir.Location.file("main.py", 20, 3)

loc = ir.Location.callsite(callee, [frame1, frame2])
# Prints as: loc(callsite("util.py":50:10 at callsite("helper.py":100:5 at "main.py":20:3)))
```

**Unknown Location**:
```python
# Used when no location information is available
loc = ir.Location.unknown()
```

---

## Best Practices

1. **Use named locations for semantic meaning**: When the Python source location isn't meaningful (e.g., inside deep library code), use named locations

2. **Create location helpers early**: Set up location tracking utilities at the start of your project

3. **Balance granularity vs overhead**: Not every operation needs a unique location - group related operations

4. **Use hierarchical locations**: Nest locations to create logical groupings (e.g., "kernel" → "load_inputs" → "load_tile_A")

5. **Disable in production**: Location tracking adds overhead - consider making it optional:
   ```python
   DEBUG = os.environ.get("MLIR_DEBUG", "0") == "1"

   if DEBUG:
       with tracker.auto():
           op = create_operation()
   else:
       op = create_operation()
   ```

---

## Common Pitfalls

### Pitfall 1: Using `Location.unknown()` Everywhere

```python
# DON'T DO THIS - locations will all be unknown
with ir.Location.unknown():
    # All operations created here have no location info
    create_lots_of_operations()
```

### Pitfall 2: Forgetting Context Manager

```python
# DON'T DO THIS - location not used as context manager
loc = ir.Location.file("script.py", 10, 0)
op = arith.ConstantOp(...)  # Still has unknown location!

# DO THIS - use as context manager
with ir.Location.file("script.py", 10, 0):
    op = arith.ConstantOp(...)  # Now has proper location
```

### Pitfall 3: Not Passing Context

```python
# Can fail if no implicit context is active
loc = ir.Location.file("script.py", 10, 0)  # May fail!

# Better - explicitly pass context
with ir.Context() as ctx:
    loc = ir.Location.file("script.py", 10, 0, context=ctx)
```

---

## Summary

**The key insight**: MLIR doesn't automatically track locations. You must explicitly set them using:

1. **Context managers**: `with ir.Location.file(...):` or `with ir.Location.name(...):`
2. **Python's inspect module**: Auto-capture calling location with `inspect.currentframe()`
3. **Helper classes**: Create utilities to make location tracking easier

**Remember**: Setting `enable_debug_info=True` in PassManager only **displays** locations that already exist in your IR. It doesn't create them!
