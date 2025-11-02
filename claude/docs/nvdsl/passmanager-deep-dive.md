# MLIR PassManager: Complete Technical Deep Dive

This document provides a comprehensive analysis of `mlir.passmanager.PassManager`, including documented and undocumented features, complete call traces, side effects, and debugging capabilities.

## Table of Contents
- [PassManager API Overview](#passmanager-api-overview)
- [Complete Call Traces](#complete-call-traces)
- [Context Management](#context-management)
- [Debug Support](#debug-support)
- [Side Effects and Invariants](#side-effects-and-invariants)

---

## PassManager API Overview

### Documented Methods and Properties

**From**: [build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/_mlir/passmanager.pyi](../../build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/_mlir/passmanager.pyi)

```python
class PassManager:
    def __init__(self, context: ir.Context | None = None) -> None: ...

    @staticmethod
    def parse(pipeline: str, context: ir.Context | None = None) -> PassManager: ...

    def run(self, module: ir._OperationBase) -> None: ...

    def enable_ir_printing(
        self,
        print_before_all: bool = False,
        print_after_all: bool = True,
        print_module_scope: bool = False,
        print_after_change: bool = False,
        print_after_failure: bool = False,
        large_elements_limit: int | None = None,
        large_resource_limit: int | None = None,
        enable_debug_info: bool = False,
        print_generic_op_form: bool = False,
        tree_printing_dir_path: str | None = None,
    ) -> None: ...

    def enable_verifier(self, enable: bool) -> None: ...

    @property
    def _CAPIPtr(self) -> object: ...

    def _CAPICreate(self) -> object: ...

    def _testing_release(self) -> None: ...
```

### Undocumented Methods

**From**: [mlir/lib/Bindings/Python/Pass.cpp](../../mlir/lib/Bindings/Python/Pass.cpp)

The following methods are implemented in C++ but **not documented** in the stub file:

#### 1. `enable_timing()`

**Line**: [Pass.cpp:133-137](../../mlir/lib/Bindings/Python/Pass.cpp#L133-L137)

```python
def enable_timing(self) -> None:
    """Enable pass timing statistics collection."""
```

**Usage**:
```python
pm = PassManager.parse("func.func(cse,canonicalize)")
pm.enable_timing()
pm.run(module)
# Timing stats printed to stderr
```

#### 2. `add(pipeline: str)`

**Line**: [Pass.cpp:156-168](../../mlir/lib/Bindings/Python/Pass.cpp#L156-L168)

```python
def add(self, pipeline: str) -> None:
    """
    Add textual pipeline elements to the pass manager.
    Throws ValueError if the pipeline can't be parsed.
    """
```

**Usage**:
```python
pm = PassManager()
pm.add("func.func(cse)")
pm.add("canonicalize")
pm.run(module)
```

#### 3. `add(run: Callable, ...)`

**Line**: [Pass.cpp:170-207](../../../mlir/lib/Bindings/Python/Pass.cpp#L170-L207)

```python
def add(
    self,
    run: Callable[[ir.Operation, ExternalPass], None],
    name: str | None = None,
    argument: str = "",
    description: str = "",
    op_name: str = ""
) -> None:
    """Add a Python-defined pass to the pass manager."""
```

**Usage**:
```python
def my_pass(op, external_pass):
    # Custom pass logic
    print(f"Running on {op}")
    # Signal failure if needed:
    # external_pass.signal_pass_failure()

pm = PassManager()
pm.add(my_pass, name="MyPass", argument="my-pass", op_name="func.func")
pm.run(module)
```

#### 4. `__str__()`

**Line**: [Pass.cpp:223-232](../../mlir/lib/Bindings/Python/Pass.cpp#L223-L232)

```python
def __str__(self) -> str:
    """
    Print the textual representation for this PassManager,
    suitable to be passed to `parse` for round-tripping.
    """
```

**Usage**:
```python
pm = PassManager.parse("func.func(cse,canonicalize)")
print(pm)  # Output: "func.func(cse,canonicalize)"
```

### Undocumented Properties

None beyond what's in the stub file.

### Complete API Summary

| Method/Property | Documented | Purpose |
|----------------|------------|---------|
| `__init__(context)` | ✅ | Create pass manager for context |
| `parse(pipeline, context)` | ✅ | Parse textual pipeline |
| `run(operation)` | ✅ | Run passes on operation |
| `enable_ir_printing(...)` | ✅ | Enable IR printing before/after passes |
| `enable_verifier(enable)` | ✅ | Enable/disable IR verification |
| `enable_timing()` | ❌ | Enable timing statistics |
| `add(pipeline)` | ❌ | Add pipeline string |
| `add(run, ...)` | ❌ | Add Python-defined pass |
| `__str__()` | ❌ | Get pipeline as string |
| `_CAPIPtr` | ✅ | Get C API capsule |
| `_CAPICreate()` | ✅ | Create from capsule |
| `_testing_release()` | ✅ | Leak for testing |

---

## Complete Call Traces

### Trace 1: `PassManager.parse(pipeline, context)`

#### Frame 1: Python Call

```python
pm = mlir.passmanager.PassManager.parse("func.func(cse,canonicalize)")
```

#### Frame 2: Python Binding (Nanobind)

**File**: [mlir/lib/Bindings/Python/Pass.cpp:138-154](../../mlir/lib/Bindings/Python/Pass.cpp#L138-L154)

```cpp
.def_static(
    "parse",
    [](const std::string &pipeline, DefaultingPyMlirContext context) {
        // Create empty pass manager
        MlirPassManager passManager = mlirPassManagerCreate(context->get());

        // Prepare error accumulator
        PyPrintAccumulator errorMsg;

        // Parse the pipeline
        MlirLogicalResult status = mlirParsePassPipeline(
            mlirPassManagerGetAsOpPassManager(passManager),
            mlirStringRefCreate(pipeline.data(), pipeline.size()),
            errorMsg.getCallback(), errorMsg.getUserData());

        // Check result
        if (mlirLogicalResultIsFailure(status))
            throw nb::value_error(errorMsg.join().c_str());

        return new PyPassManager(passManager);
    },
    "pipeline"_a, "context"_a = nb::none(),
    "Parse a textual pass-pipeline...")
```

**Key Steps**:
1. Get or create default context (`DefaultingPyMlirContext`)
2. Create empty `MlirPassManager` via C API
3. Call C API `mlirParsePassPipeline` with error callback
4. Throw Python exception if parsing fails
5. Return wrapped `PyPassManager` object

#### Frame 3: C API Layer

**File**: [mlir/lib/CAPI/IR/Pass.cpp:116-125](../../mlir/lib/CAPI/IR/Pass.cpp#L116-L125)

```cpp
MlirLogicalResult mlirParsePassPipeline(MlirOpPassManager passManager,
                                        MlirStringRef pipeline,
                                        MlirStringCallback callback,
                                        void *userData) {
    // Create callback output stream
    detail::CallbackOstream stream(callback, userData);

    // Call C++ function (unwrap opaque handles)
    FailureOr<OpPassManager> pm = parsePassPipeline(unwrap(pipeline), stream);

    // Store result if successful
    if (succeeded(pm))
        *unwrap(passManager) = std::move(*pm);

    return wrap(pm);  // Convert LogicalResult to MlirLogicalResult
}
```

**Key Conversions**:
- `MlirOpPassManager` → `OpPassManager*` (via `unwrap()`)
- `MlirStringRef` → `StringRef` (via `unwrap()`)
- `MlirStringCallback` → `raw_ostream` (via `CallbackOstream`)
- `LogicalResult` → `MlirLogicalResult` (via `wrap()`)

#### Frame 4: C++ Implementation

**File**: [mlir/lib/Pass/PassRegistry.cpp:768-770](../../mlir/lib/Pass/PassRegistry.cpp#L768-L770)

```cpp
LogicalResult mlir::parsePassPipeline(StringRef pipeline, OpPassManager &pm,
                                      raw_ostream &errorStream) {
    TextualPipeline pipelineParser;
    // Parse the textual pipeline representation
    // ...
}
```

**What happens**:
1. **Tokenize** the pipeline string (`func.func(cse,canonicalize)`)
2. **Parse** the structure:
   - Anchor operation: `func.func`
   - Nested passes: `cse`, `canonicalize`
3. **Lookup** registered passes by name
4. **Instantiate** pass objects
5. **Nest** passes in the OpPassManager hierarchy
6. **Validate** the pipeline structure

**Pipeline Structure Created**:
```
PassManager
  └─ OpPassManager for "func.func"
       ├─ CSE pass
       └─ Canonicalize pass
```

### Trace 2: `passmanager.run(operation)`

#### Frame 1: Python Call

```python
pm.run(module)
```

#### Frame 2: Python Binding (Nanobind)

**File**: [mlir/lib/Bindings/Python/Pass.cpp:209-221](../../mlir/lib/Bindings/Python/Pass.cpp#L209-L221)

```cpp
.def(
    "run",
    [](PyPassManager &passManager, PyOperationBase &op) {
        // Enable error capture for the operation's context
        PyMlirContext::ErrorCapture errors(op.getOperation().getContext());

        // Run the pass manager
        MlirLogicalResult status = mlirPassManagerRunOnOp(
            passManager.get(), op.getOperation().get());

        // Check for failure
        if (mlirLogicalResultIsFailure(status))
            throw MLIRError("Failure while executing pass pipeline",
                            errors.take());
    },
    "operation"_a,
    "Run the pass manager on the provided operation...")
```

**Key Steps**:
1. **Set up error capture** on the operation's context
2. Extract C API handles:
   - `PyPassManager` → `MlirPassManager`
   - `PyOperationBase` → `MlirOperation`
3. **Call C API** `mlirPassManagerRunOnOp`
4. **Collect errors** and throw `MLIRError` on failure

#### Frame 3: C API Layer

**File**: [mlir/lib/CAPI/IR/Pass.cpp:42-45](../../mlir/lib/CAPI/IR/Pass.cpp#L42-L45)

```cpp
MlirLogicalResult mlirPassManagerRunOnOp(MlirPassManager passManager,
                                         MlirOperation op) {
    // Unwrap and run
    return wrap(unwrap(passManager)->run(unwrap(op)));
}
```

**Key Conversions**:
- `MlirPassManager` → `PassManager*` (via `unwrap()`)
- `MlirOperation` → `Operation*` (via `unwrap()`)
- Calls `PassManager::run(Operation*)`
- `LogicalResult` → `MlirLogicalResult` (via `wrap()`)

#### Frame 4: C++ PassManager::run()

**File**: [mlir/lib/Pass/PassManager.cpp](../../mlir/lib/Pass/PassManager.cpp) (implementation detail)

```cpp
LogicalResult PassManager::run(Operation *op) {
    // 1. Verify the operation is valid for this pass manager
    if (!isAnchorOp(op))
        return failure();

    // 2. Set up pass instrumentation (timing, IR printing, etc.)
    PassInstrumentor instrumentor;
    // ... setup instrumentation ...

    // 3. Run verification (if enabled)
    if (verifyPasses && failed(verify(op)))
        return failure();

    // 4. Execute passes in the pipeline
    for (Pass *pass : passes) {
        // Run individual pass
        if (failed(runPass(pass, op)))
            return failure();

        // Verify after pass (if enabled)
        if (verifyPasses && failed(verify(op)))
            return failure();
    }

    return success();
}
```

**Key Steps**:
1. **Validate** operation matches pass manager anchor
2. **Initialize** instrumentation (timing, printing)
3. **Pre-verify** (if `enable_verifier(true)`)
4. **For each pass**:
   a. Run the pass's `runOnOperation()` method
   b. Check for pass failure
   c. **Post-verify** (if enabled)
   d. **Print IR** (if enabled)
5. **Report timing** (if enabled)

---

## Context Management

### Context Association

**Key Principle**: PassManager is **bound to a specific MLIRContext**.

#### At Creation

```python
# Method 1: Use current thread-local context
with ir.Context():
    pm = PassManager.parse("...")  # Uses Context.current

# Method 2: Explicit context
ctx = ir.Context()
pm = PassManager.parse("...", context=ctx)
```

**From**: [Pass.cpp:73-77](../../mlir/lib/Bindings/Python/Pass.cpp#L73-L77)

```cpp
MlirPassManager passManager = mlirPassManagerCreateOnOperation(
    context->get(),  // Context is captured here
    mlirStringRefCreate(anchorOp.data(), anchorOp.size()));
```

#### During Execution

**From**: [Pass.cpp:212](../../mlir/lib/Bindings/Python/Pass.cpp#L212)

```cpp
PyMlirContext::ErrorCapture errors(op.getOperation().getContext());
```

**The operation's context** is used for:
1. **Error reporting** - diagnostics emitted to the operation's context
2. **Verification** - IR verifier uses the operation's context
3. **Pass execution** - passes access the context via `getContext()`

### Context Lifetime Requirements

```python
# CORRECT: Context outlives pass manager and operation
ctx = ir.Context()
pm = PassManager.parse("...", context=ctx)
module = ir.Module.create(loc=ir.Location.unknown(ctx))
pm.run(module)  # Safe

# INCORRECT: Context destroyed before run
def broken():
    ctx = ir.Context()
    pm = PassManager.parse("...", context=ctx)
    module = ir.Module.create(loc=ir.Location.unknown(ctx))
    return pm, module  # ctx destroyed here!

pm, module = broken()
pm.run(module)  # SEGFAULT - context is gone
```

### Context Compatibility Check

**Operation and PassManager must share the same context**:

```python
ctx1 = ir.Context()
ctx2 = ir.Context()

pm = PassManager.parse("...", context=ctx1)
module = ir.Module.create(loc=ir.Location.unknown(ctx2))

pm.run(module)  # ERROR: Contexts don't match
```

This is checked at runtime in C++.

---

## Debug Support

### Method 1: IR Printing

**Print IR before/after passes**:

```python
pm = PassManager.parse("func.func(cse,canonicalize)")

pm.enable_ir_printing(
    print_before_all=True,      # Print before each pass
    print_after_all=True,       # Print after each pass
    print_module_scope=True,    # Print full module (not just func)
    print_after_change=False,   # Only print if IR changed
    print_after_failure=True,   # Print on pass failure
    enable_debug_info=True,     # Include locations
    print_generic_op_form=False # Use op-specific pretty print
)

pm.run(module)
```

**Output** (to stderr):
```
*** IR Dump Before CSE ***
func.func @foo() {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  return
}

*** IR Dump After CSE ***
func.func @foo() {
  %0 = arith.constant 1 : i32
  return
}
```

**Write to directory** (tree structure):

```python
pm.enable_ir_printing(
    tree_printing_dir_path="/tmp/ir_dumps"
)
pm.run(module)
```

Creates:
```
/tmp/ir_dumps/
  module_0_before_cse.mlir
  module_0_after_cse.mlir
  module_0_before_canonicalize.mlir
  module_0_after_canonicalize.mlir
```

### Method 2: LLVM Debug Flags (Undocumented!)

**Enable global debug flag** (equivalent to `-debug`):

```python
from mlir.ir import _GlobalDebug

# Enable all debug output
_GlobalDebug.flag = True

pm = PassManager.parse("...")
pm.run(module)  # Debug output from passes to stderr
```

**Enable specific debug types** (equivalent to `-debug-only=<type>`):

```python
from mlir.ir import _GlobalDebug

# Enable debug flag
_GlobalDebug.flag = True

# Set specific types
_GlobalDebug.set_types("cse")  # Only CSE pass debug output

# Or multiple types
_GlobalDebug.set_types(["cse", "canonicalize"])

pm = PassManager.parse("func.func(cse,canonicalize)")
pm.run(module)
```

**From**: [mlir/lib/Bindings/Python/IRCore.cpp:260-285](../../mlir/lib/Bindings/Python/IRCore.cpp#L260-L285)

```cpp
struct PyGlobalDebugFlag {
    static void set(nb::object &o, bool enable) {
        nb::ft_lock_guard lock(mutex);
        mlirEnableGlobalDebug(enable);  // C API call
    }

    static bool get(const nb::object &) {
        nb::ft_lock_guard lock(mutex);
        return mlirIsGlobalDebugEnabled();
    }

    static void bind(nb::module_ &m) {
        nb::class_<PyGlobalDebugFlag>(m, "_GlobalDebug")
            .def_prop_rw_static("flag", &PyGlobalDebugFlag::get,
                                &PyGlobalDebugFlag::set, "LLVM-wide debug flag")
            .def_static(
                "set_types",
                [](const std::string &type) {
                    nb::ft_lock_guard lock(mutex);
                    mlirSetGlobalDebugType(type.c_str());
                },
                "types"_a, "Sets specific debug types to be produced by LLVM")
            // ...
    }
};
```

**Complete Python API**:

```python
from mlir.ir import _GlobalDebug

# Get/set global debug flag
_GlobalDebug.flag = True
print(_GlobalDebug.flag)  # True

# Set single debug type
_GlobalDebug.set_types("cse")

# Set multiple debug types
_GlobalDebug.set_types(["cse", "canonicalize"])
```

### Method 3: Pass Timing

```python
pm = PassManager.parse("func.func(cse,canonicalize,cse)")
pm.enable_timing()
pm.run(module)
```

**Output** (to stderr):
```
===-------------------------------------------------------------------------===
                         Pass execution timing report
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0123 seconds

   ---User Time---   --System Time--   --User+System--   ---Wall Time---  --- Name ---
   0.0045 ( 45.2%)   0.0012 ( 30.0%)   0.0057 ( 42.5%)   0.0057 ( 46.3%)  CSE
   0.0034 ( 34.2%)   0.0018 ( 45.0%)   0.0052 ( 38.8%)   0.0052 ( 42.3%)  Canonicalize
   0.0020 ( 20.6%)   0.0010 ( 25.0%)   0.0030 ( 22.4%)   0.0030 ( 24.4%)  CSE
   0.0099 (100.0%)   0.0040 (100.0%)   0.0134 (100.0%)   0.0123 (100.0%)  Total
```

### Method 4: Verifier Output

```python
pm = PassManager.parse("func.func(my-buggy-pass)")
pm.enable_verifier(True)  # On by default

try:
    pm.run(module)
except Exception as e:
    print(f"Verification failed: {e}")
    # Error message includes:
    # - Which pass caused the problem
    # - What IR was invalid
    # - Source location (if available)
```

---

## Side Effects and Invariants

### Side Effects of `PassManager.parse()`

1. **Context creation** (if not provided):
   ```python
   pm = PassManager.parse("...")  # Creates Context.current if none exists
   ```

2. **Pass registration lookup**:
   - Searches global pass registry
   - Throws `ValueError` if pass not found

3. **No IR mutation** - just creates the pipeline structure

### Side Effects of `passmanager.run()`

1. **IR Mutation**:
   - **Transforms the operation in-place**
   - All child operations may be modified
   - Operations may be added, removed, or reordered
   - Attributes may be added, removed, or modified

2. **Context mutation**:
   - **Diagnostics** emitted to context
   - **Verification** may add diagnostics

3. **Error Capture**:
   ```python
   PyMlirContext::ErrorCapture errors(context);
   ```
   - Intercepts all diagnostics during pass execution
   - Attached to Python exception on failure

4. **Output to stderr**:
   - IR printing (if enabled)
   - Timing statistics (if enabled)
   - Debug output (if enabled)
   - Verification errors

5. **Verification** (if enabled):
   - Checks IR validity before and after each pass
   - Throws exception on invalid IR

### Invariants

#### Before `run()`:

```python
# Operation must be valid IR
assert module.verify()

# Context must be alive
assert not ctx._is_destroyed  # (hypothetical)

# Operation must match pass manager anchor
pm = PassManager.parse("func.func(...)")
# Can only run on operations containing func.func ops
```

#### After successful `run()`:

```python
# IR is modified but still valid
assert module.verify()

# Context unchanged (but may have diagnostics)
assert module.context == ctx

# Operation tree may be completely different
# (old operation references may be invalidated!)
```

#### After failed `run()`:

```python
try:
    pm.run(module)
except MLIRError as e:
    # IR may be in invalid state!
    # Do NOT continue using the module
    # Best practice: discard and recreate
    pass
```

### Memory Management

**PassManager lifecycle**:

```cpp
class PyPassManager {
public:
    ~PyPassManager() {
        if (!mlirPassManagerIsNull(passManager))
            mlirPassManagerDestroy(passManager);  // Frees C++ PassManager
    }
};
```

**Operation lifecycle**:
- **Not owned by PassManager**
- Operation must outlive `run()` call
- Python keeps operation alive via reference counting

**Context lifecycle**:
- **PassManager holds context pointer** (not owned)
- Context must outlive PassManager
- Context must outlive all operations

### Thread Safety

**PassManager is NOT thread-safe**:

```python
# INCORRECT: Multiple threads using same PassManager
pm = PassManager.parse("...")

def worker(module):
    pm.run(module)  # UNSAFE

import threading
threading.Thread(target=worker, args=(module1,)).start()
threading.Thread(target=worker, args=(module2,)).start()  # DATA RACE
```

**Correct approach**:

```python
# Create separate PassManager per thread
def worker(module):
    pm = PassManager.parse("...")  # Thread-local
    pm.run(module)

threading.Thread(target=worker, args=(module1,)).start()
threading.Thread(target=worker, args=(module2,)).start()  # Safe
```

**Context thread safety**:
- **Context IS thread-safe** (with caveats)
- Multiple threads can create operations in the same context
- But: running passes concurrently on same context is undefined

---

## Complete Usage Examples

### Example 1: Basic Pipeline

```python
import mlir.ir as ir
import mlir.passmanager as pm

with ir.Context():
    # Parse module
    module = ir.Module.parse("""
        func.func @foo(%arg0: i32) -> i32 {
            %0 = arith.constant 1 : i32
            %1 = arith.constant 1 : i32
            %2 = arith.addi %arg0, %0 : i32
            %3 = arith.addi %2, %1 : i32
            return %3 : i32
        }
    """)

    # Create and run pass manager
    pass_mgr = pm.PassManager.parse("func.func(cse,canonicalize)")
    pass_mgr.run(module.operation)

    print(module)  # CSE eliminated duplicate constant
```

### Example 2: IR Printing and Timing

```python
with ir.Context():
    module = ir.Module.parse("...")

    pm_mgr = pm.PassManager.parse("func.func(cse,canonicalize,cse)")

    # Enable features
    pm_mgr.enable_ir_printing(print_before_all=True, print_after_all=True)
    pm_mgr.enable_timing()
    pm_mgr.enable_verifier(True)

    pm_mgr.run(module.operation)
```

### Example 3: Debug Output

```python
from mlir.ir import _GlobalDebug

with ir.Context():
    module = ir.Module.parse("...")

    # Enable LLVM debug output
    _GlobalDebug.flag = True
    _GlobalDebug.set_types(["cse", "canonicalize"])

    pm_mgr = pm.PassManager.parse("func.func(cse,canonicalize)")
    pm_mgr.run(module.operation)

    # Disable when done
    _GlobalDebug.flag = False
```

### Example 4: Python-Defined Pass

```python
def my_analysis_pass(op, external_pass):
    """Count operations in the IR."""
    count = 0
    def count_ops(op):
        nonlocal count
        count += 1
        for region in op.regions:
            for block in region:
                for child_op in block:
                    count_ops(child_op)

    count_ops(op)
    print(f"Total operations: {count}")

with ir.Context():
    module = ir.Module.parse("...")

    pm_mgr = pm.PassManager()
    pm_mgr.add(my_analysis_pass, name="OpCounter", op_name="func.func")
    pm_mgr.run(module.operation)
```

### Example 5: Error Handling

```python
with ir.Context():
    module = ir.Module.parse("...")

    pm_mgr = pm.PassManager.parse("func.func(some-buggy-pass)")

    try:
        pm_mgr.run(module.operation)
    except ir.MLIRError as e:
        print(f"Pass failed: {e}")
        # IR may be invalid - discard module
        module = None
```

---

## Summary

### Key Takeaways

1. **PassManager is bound to an MLIRContext** - they must have compatible lifetimes
2. **`run()` mutates IR in-place** - old operation references may be invalidated
3. **Verification is enabled by default** - disable with `enable_verifier(False)` for performance
4. **Undocumented features exist**:
   - `enable_timing()`
   - `add(pipeline)` and `add(run, ...)`
   - `__str__()`
   - `_GlobalDebug` for LLVM debug output
5. **Thread safety**: Create separate PassManager instances per thread
6. **Error handling**: Always catch `MLIRError` and discard invalid IR

### Debug Checklist

- ✅ **IR Printing**: `enable_ir_printing(print_before_all=True, print_after_all=True)`
- ✅ **Timing**: `enable_timing()`
- ✅ **Verification**: `enable_verifier(True)` (default)
- ✅ **LLVM Debug**: `_GlobalDebug.flag = True` and `_GlobalDebug.set_types([...])`
- ✅ **Tree Dump**: `enable_ir_printing(tree_printing_dir_path="/tmp/ir")`

---

## Further Reading

- [Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- [Writing a Pass](https://mlir.llvm.org/docs/Tutorials/CreatingAPass/)
- [Context Management](../context_management/insertion-point-deep-dive.md)
- [Python Bindings Architecture](../../mlir/lib/Bindings/Python/)
