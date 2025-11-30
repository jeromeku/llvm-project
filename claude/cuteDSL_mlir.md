# CuTeDSL's Usage of MLIR Python Bindings

## Overview

CuTeDSL is built on top of MLIR's Python bindings and provides a Domain-Specific Language (DSL) for writing GPU kernels using the CUTLASS library. This document provides a comprehensive mapping of how CuTeDSL uses MLIR's core Python bindings.

## Architecture Overview

### Directory Structure

- **CuTeDSL MLIR bindings**: `/home/jeromeku/llvm-project/cutlass/python/CuTeDSL/cutlass/_mlir/`
- **Core MLIR bindings**: `/home/jeromeku/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/`

### Key Relationship

CuTeDSL wraps and extends MLIR's core bindings through:
1. Direct re-exports from `_cutlass_ir._mlir` (their custom built bindings)
2. Custom helper classes and utilities
3. Generated dialect operations (ODS-based)
4. Higher-level DSL abstractions

---

## 1. Core IR Infrastructure

### 1.1 Module and Context Management

**File**: [cutlass/_mlir/ir.py](cutlass/python/CuTeDSL/cutlass/_mlir/ir.py:5)

```python
# CuTeDSL re-exports all core IR classes
from ._mlir_libs._cutlass_ir._mlir.ir import *
```

**Core MLIR API**: [mlir/python/mlir/ir.py](mlir/python/mlir/ir.py:10)

```python
# Core MLIR exports from C++ bindings
from ._mlir_libs._mlir.ir import *
```

#### Context Creation and Initialization

**File**: [cutlass/_mlir/_mlir_libs/__init__.py](cutlass/python/CuTeDSL/cutlass/_mlir/_mlir_libs/__init__.py:107-143)

CuTeDSL extends `ir.Context` to customize dialect loading:

```python
class Context(ir._BaseContext):
    def __init__(self, load_on_create_dialects=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register custom dialects
        self.append_dialect_registry(get_dialect_registry())

        # Execute post-init hooks
        for hook in post_init_hooks:
            hook(self)

        # Enable multi-threading
        if not disable_multithreading:
            self.enable_multithreading(True)

        # Load dialects on demand or all at once
        if load_on_create_dialects is not None:
            for dialect in load_on_create_dialects:
                _ = self.dialects[dialect]
        else:
            self.load_all_available_dialects()
```

**Mapping to Core MLIR**:
- `ir._BaseContext`: The C++-bound context class from `mlir::python::PyMlirContext`
- `append_dialect_registry()`: Maps to `MLIRContext::appendDialectRegistry()` in C++
- `enable_multithreading()`: Maps to `MLIRContext::enableMultithreading()`
- `load_all_available_dialects()`: Loads all registered dialects

**C++ API Reference**:
- Header: `mlir/include/mlir/IR/MLIRContext.h`
- Class: `mlir::MLIRContext`

---

### 1.2 Location Management

**File**: [cutlass/base_dsl/_mlir_helpers/op.py](cutlass/python/CuTeDSL/cutlass/base_dsl/_mlir_helpers/op.py:22-63)

CuTeDSL provides automatic location tracking for operations:

```python
@dsl_user_op
def some_operation(*args, **kwargs):
    # Automatically captures source location
    pass

# Implementation in op.py:
def dsl_user_op(opFunc):
    @wraps(opFunc)
    def wrapper(*args, **kwargs):
        loc = kwargs.pop("loc", None)
        if loc is None:
            frame = inspect.currentframe().f_back
            frameInfo = inspect.getframeinfo(frame)

            # Create file location from Python stack frame
            file_loc = ir.Location.file(
                frameInfo.filename,
                frameInfo.lineno,
                frameInfo.col_offset,
            )

            # Create named location with source code context
            loc = ir.Location.name(
                frameInfo.code_context,
                childLoc=file_loc,
            )
        res_or_list = opFunc(*args, **kwargs, loc=loc)
        return res_or_list
    return wrapper
```

**Core MLIR API**: `ir.Location`

```python
# Location types in MLIR
ir.Location.file(filename, line, col)      # FileLineColLoc
ir.Location.name(name, childLoc)           # NameLoc
ir.Location.unknown()                       # UnknownLoc
ir.Location.fused([loc1, loc2, ...])      # FusedLoc
ir.Location.callsite(callee, caller)       # CallSiteLoc
```

**C++ API Reference**:
- Header: `mlir/include/mlir/IR/Location.h`
- Classes: `Location`, `FileLineColLoc`, `NameLoc`, `FusedLoc`

---

### 1.3 InsertionPoint Management

**File**: [cutlass/_mlir/dialects/func.py](cutlass/python/CuTeDSL/cutlass/_mlir/dialects/func.py:63-64)

```python
def __init__(self, name, type, *, visibility=None, body_builder=None, loc=None, ip=None):
    # ... initialization ...

    if body_builder:
        entry_block = self.add_entry_block()
        # Set insertion point to the entry block
        with InsertionPoint(entry_block):
            body_builder(self)
```

**Core MLIR API**: `ir.InsertionPoint`

Used as a context manager to control where operations are inserted:

```python
with ir.InsertionPoint(block):
    # Operations created here are inserted at block's end
    op = some_dialect.SomeOp(...)

with ir.InsertionPoint.at_block_begin(block):
    # Operations inserted at block's beginning
    op = some_dialect.SomeOp(...)

with ir.InsertionPoint.at_block_terminator(block):
    # Operations inserted before terminator
    op = some_dialect.SomeOp(...)
```

**C++ API Reference**:
- Header: `mlir/include/mlir/IR/Builders.h`
- Class: `mlir::OpBuilder` (InsertionPoint wraps this)

---

### 1.4 Module Management

**File**: [cutlass/base_dsl/compiler.py](cutlass/python/CuTeDSL/cutlass/base_dsl/compiler.py:136-161)

```python
def compile(self, module, pipeline: str, cuda_toolkit: str = "",
            arch: str = "", enable_verifier=False):
    """Compiles the module by invoking the pipeline."""
    try:
        # Parse and run pass pipeline on module
        pm = self.passmanager.PassManager.parse(pipeline)
        pm.enable_verifier(enable_verifier)
        pm.run(module.operation)  # Run on module's operation
    except Exception as e:
        # Error handling...
        raise e
```

**Core MLIR API**: `ir.Module`

```python
# Creating a module
with ir.Context() as ctx, ir.Location.unknown():
    module = ir.Module.create()

    # Module is an Operation with specific semantics
    module.operation  # Returns the underlying Operation

    # Getting module body
    module.body  # Returns the Region containing the module

    # Parsing a module from string
    module = ir.Module.parse("""
        module {
            func.func @foo() {
                return
            }
        }
    """)
```

**C++ API Reference**:
- Header: `mlir/include/mlir/IR/BuiltinOps.h`
- Class: `mlir::ModuleOp`

---

## 2. Type System

### 2.1 Type Helpers and Constructors

**File**: [cutlass/_mlir/extras/types.py](cutlass/python/CuTeDSL/cutlass/_mlir/extras/types.py:8-167)

CuTeDSL provides convenient type constructors that wrap core MLIR types:

```python
from ..ir import (
    IntegerType, IndexType, F16Type, F32Type, F64Type,
    VectorType, MemRefType, RankedTensorType, FunctionType,
    # ... more types
)

# Scalar types
index = lambda: IndexType.get()
i32 = lambda: IntegerType.get_signless(32)
f32 = lambda: F32Type.get()
f16 = lambda: F16Type.get()
bf16 = lambda: BF16Type.get()

# Exotic float types (for ML workloads)
f8E5M2 = lambda: Float8E5M2Type.get()
f8E4M3FN = lambda: Float8E4M3FNType.get()

# Vector types
def vector(*shape, element_type: Type = None,
           scalable: Optional[List[bool]] = None):
    return VectorType.get(shape, element_type,
                         scalable=scalable)

# Tensor types
def tensor(*shape, element_type: Type = None,
           encoding: Optional[str] = None):
    if encoding is not None:
        encoding = StringAttr.get(encoding)
    return RankedTensorType.get(shape, element_type,
                               encoding=encoding)

# MemRef types
def memref(*shape, element_type: Type = None,
           memory_space: Optional[int] = None,
           layout: Optional[StridedLayoutAttr] = None):
    return MemRefType.get(shape, element_type,
                         memory_space=memory_space,
                         layout=layout)
```

**Usage Example**:

```python
from cutlass._mlir.extras import types as T

# Creating types
i32_type = T.i32()              # 32-bit integer
f32_vec = T.vector(4, T.f32())  # Vector of 4 f32s
memref = T.memref(16, 16, T.f16(), memory_space=3)  # Shared memory
```

**Core MLIR API Mapping**:

| CuTeDSL Helper | Core MLIR Type | C++ Class |
|----------------|----------------|-----------|
| `T.i32()` | `IntegerType.get_signless(32)` | `mlir::IntegerType` |
| `T.f32()` | `F32Type.get()` | `mlir::Float32Type` |
| `T.vector()` | `VectorType.get()` | `mlir::VectorType` |
| `T.memref()` | `MemRefType.get()` | `mlir::MemRefType` |
| `T.tensor()` | `RankedTensorType.get()` | `mlir::RankedTensorType` |

**C++ Headers**:
- `mlir/include/mlir/IR/BuiltinTypes.h`
- `mlir/include/mlir/IR/BuiltinTypeInterfaces.h`

---

### 2.2 Type Manipulation

**File**: [cutlass/base_dsl/_mlir_helpers/arith.py](cutlass/python/CuTeDSL/cutlass/base_dsl/_mlir_helpers/arith.py:32-69)

```python
def recast_type(src_type, res_elem_type) -> ir.Type:
    """Changes element type while preserving shape"""
    if isinstance(src_type, ir.VectorType):
        if src_type.scalable:
            res_type = T.vector(
                *src_type.shape,
                res_elem_type,
                scalable=src_type.scalable,
                scalable_dims=src_type.scalable_dims,
            )
        else:
            res_type = T.vector(*src_type.shape, res_elem_type)
    elif isinstance(src_type, ir.RankedTensorType):
        res_type = T.RankedTensorType.get(
            element_type=res_elem_type,
            shape=src_type.shape
        )
    # ... more type cases
    return res_type

def element_type(ty) -> ir.Type:
    """Extract element type from shaped types"""
    if not is_scalar(ty):
        return ty.element_type
    else:
        return ty
```

**Core MLIR API**:

```python
# Type introspection
if isinstance(value.type, ir.VectorType):
    shape = value.type.shape         # Get shape tuple
    elem_type = value.type.element_type  # Get element type
    rank = value.type.rank           # Get rank

if isinstance(value.type, ir.MemRefType):
    memory_space = value.type.memory_space
    layout = value.type.layout
```

---

### 2.3 Type Casting and Conversion

**File**: [cutlass/_mlir/ir.py](cutlass/python/CuTeDSL/cutlass/_mlir/ir.py:7)

```python
from ._mlir_libs._cutlass_ir._mlir import register_type_caster, register_value_caster

# These allow automatic Python-to-MLIR type conversions
register_type_caster(kind, converter_func)
register_value_caster(kind, converter_func)
```

This enables CuTeDSL to automatically convert Python objects to MLIR types/values.

---

## 3. Attributes

### 3.1 Attribute Builders

**File**: [cutlass/_mlir/ir.py](cutlass/python/CuTeDSL/cutlass/_mlir/ir.py:14-263)

CuTeDSL registers custom attribute builders for automatic type coercion:

```python
@register_attribute_builder("I32Attr")
def _i32Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), x)

@register_attribute_builder("F32Attr")
def _f32Attr(x, context):
    return FloatAttr.get_f32(x, context=context)

@register_attribute_builder("ArrayAttr")
def _arrayAttr(x, context):
    return ArrayAttr.get(x, context=context)

@register_attribute_builder("DenseI64ArrayAttr")
def _denseI64ArrayAttr(x, context):
    return DenseI64ArrayAttr.get(x, context=context)

# NumPy-based dense attributes
@register_attribute_builder("I32ElementsAttr")
def _i32ElementsAttr(x, context):
    return DenseElementsAttr.get(
        np.array(x, dtype=np.int32),
        type=IntegerType.get_signless(32, context=context),
        context=context,
    )
```

**Usage in Operation Definitions**:

When MLIR operations are generated from TableGen (ODS), these builders are automatically invoked:

```python
# In generated code (_ops_gen.py files):
# Instead of manually creating IntegerAttr.get(...), you can pass Python int
some_op = SomeOp(
    some_integer=42,           # Automatically converted to I32Attr
    some_array=[1, 2, 3],      # Automatically converted to ArrayAttr
    some_float=3.14,           # Automatically converted to F32Attr
)
```

**Core MLIR API**:

```python
# Attribute types
ir.IntegerAttr.get(type, value)
ir.FloatAttr.get_f32(value, context)
ir.StringAttr.get(string, context)
ir.ArrayAttr.get([attr1, attr2, ...], context)
ir.DenseElementsAttr.get(numpy_array, type, context)
ir.DenseI64ArrayAttr.get([1, 2, 3], context)
ir.AffineMapAttr.get(affine_map)
ir.TypeAttr.get(type)
```

**C++ Headers**:
- `mlir/include/mlir/IR/BuiltinAttributes.h`
- `mlir/include/mlir/IR/BuiltinAttributeInterfaces.h`

---

### 3.2 Specialized Attributes

**File**: [cutlass/cutlass_dsl/cutlass.py](cutlass/python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:1601)

CuTeDSL defines custom attributes for loop unrolling:

```python
class LoopUnroll(ir.Attribute):
    """Custom attribute for loop unrolling directives"""

    def __init__(self, factor: int):
        # Wraps MLIR attribute creation
        pass
```

This extends `ir.Attribute` which maps to `mlir::Attribute` in C++.

---

## 4. Operations and Dialects

### 4.1 Operation Construction Patterns

**File**: [cutlass/_mlir/dialects/_ods_common.py](cutlass/python/CuTeDSL/cutlass/_mlir/dialects/_ods_common.py:86-138)

CuTeDSL uses several helper functions for operation construction:

```python
def get_op_result_or_value(arg) -> ir.Value:
    """
    Returns the given value or the single result of the given op.
    Enables passing operations as arguments instead of extracting results.
    """
    if isinstance(arg, ir.OpView):
        return arg.operation.result  # Single result
    elif isinstance(arg, ir.Operation):
        return arg.result
    elif isinstance(arg, ir.OpResultList):
        return arg[0]
    else:
        return arg  # Already a Value

def get_op_results_or_values(arg) -> Sequence[ir.Value]:
    """Returns sequence of values or all results of an operation"""
    if isinstance(arg, ir.OpView):
        return arg.operation.results
    elif isinstance(arg, ir.Operation):
        return arg.results
    else:
        return [get_op_result_or_value(element) for element in arg]
```

**Usage Pattern**:

```python
# Instead of:
add_op = arith.AddIOp(lhs, rhs)
result = add_op.result
next_op = arith.MulIOp(result, other)

# Can write:
add_op = arith.AddIOp(lhs, rhs)
next_op = arith.MulIOp(add_op, other)  # Automatically extracts result
```

**Core MLIR API**:

```python
# Operation hierarchy
operation = some_dialect.SomeOp(...)  # Returns OpView
operation.operation                    # Get underlying Operation
operation.result                       # Get single result (if exactly 1)
operation.results                      # Get all results (OpResultList)
operation.results[0]                  # Get specific result (OpResult)

# Value types
value: ir.Value                       # Base class for SSA values
op_result: ir.OpResult               # Result of an operation (is-a Value)
block_arg: ir.BlockArgument          # Block argument (is-a Value)
```

**C++ Classes**:
- `mlir::Operation`: Base operation class
- `mlir::OpResult`: SSA value produced by operation
- `mlir::Value`: Base class for SSA values
- `mlir::BlockArgument`: Function/block argument

---

### 4.2 Dialect Registration and Usage

**File**: [cutlass/_mlir/_mlir_libs/__init__.py](cutlass/python/CuTeDSL/cutlass/_mlir/_mlir_libs/__init__.py:30-51)

```python
def get_dialect_registry():
    """Returns global dialect registry"""
    global _dialect_registry
    if _dialect_registry is None:
        from ._cutlass_ir._mlir import ir
        _dialect_registry = ir.DialectRegistry()
    return _dialect_registry

def append_load_on_create_dialect(dialect: str):
    """Register dialect to be loaded on context creation"""
    global _load_on_create_dialects
    if _load_on_create_dialects is None:
        _load_on_create_dialects = [dialect]
    else:
        _load_on_create_dialects.append(dialect)
```

**Dialects Used by CuTeDSL**:

1. **Standard MLIR Dialects**:
   - `func`: Function definitions
   - `arith`: Arithmetic operations
   - `math`: Mathematical functions
   - `scf`: Structured control flow
   - `cf`: Control flow
   - `vector`: Vector operations
   - `gpu`: GPU abstractions
   - `nvgpu`: NVIDIA GPU-specific ops
   - `nvvm`: NVIDIA PTX/NVVM ops
   - `llvm`: LLVM dialect

2. **CuTeDSL Custom Dialects**:
   - `cute`: CuTe tensor algebra
   - `cute_nvgpu`: CuTe NVIDIA GPU operations
   - `cuda`: CUDA runtime operations

---

### 4.3 Func Dialect Usage

**File**: [cutlass/_mlir/dialects/func.py](cutlass/python/CuTeDSL/cutlass/_mlir/dialects/func.py:33-240)

```python
@_ods_cext.register_operation(_Dialect, replace=True)
class FuncOp(FuncOp):
    """Enhanced func.func operation"""

    def __init__(self, name, type, *, visibility=None, body_builder=None,
                 loc=None, ip=None):
        # Create function with name and type
        sym_name = StringAttr.get(str(name))
        type = TypeAttr.get(FunctionType.get(inputs=type[0], results=type[1]))

        super().__init__(sym_name, type, sym_visibility=sym_visibility,
                        loc=loc, ip=ip)

        # Build body if provided
        if body_builder:
            entry_block = self.add_entry_block()
            with InsertionPoint(entry_block):
                body_builder(self)

    def add_entry_block(self, arg_locs=None):
        """Add entry block with arguments from function signature"""
        if not self.is_external:
            raise IndexError("Function already has entry block")
        self.body.blocks.append(*self.type.inputs, arg_locs=arg_locs)
        return self.body.blocks[0]

    @property
    def entry_block(self):
        """Get the entry block"""
        return self.regions[0].blocks[0]

    @property
    def arguments(self):
        """Get function arguments"""
        return self.entry_block.arguments
```

**Usage Example**:

```python
# Creating a function
func_type = FunctionType.get(
    inputs=[T.i32(), T.i32()],
    results=[T.i32()]
)

def build_body(func_op):
    # Access function arguments
    a, b = func_op.arguments
    # Build operations
    result = arith.AddIOp(a, b)
    func.ReturnOp([result])

func_op = func.FuncOp("add", func_type, body_builder=build_body)
```

**Core MLIR Mapping**:
- Maps to `mlir::func::FuncOp` in C++
- Header: `mlir/include/mlir/Dialect/Func/IR/FuncOps.h`

---

### 4.4 SCF Dialect Usage

**File**: [cutlass/_mlir/dialects/scf.py](cutlass/python/CuTeDSL/cutlass/_mlir/dialects/scf.py:22-135)

```python
@_ods_cext.register_operation(_Dialect, replace=True)
class ForOp(ForOp):
    """SCF for loop operation"""

    def __init__(self, lower_bound, upper_bound, step,
                 iter_args=None, *, loc=None, ip=None):
        if iter_args is None:
            iter_args = []
        iter_args = _get_op_results_or_values(iter_args)

        # Results have same types as iter_args
        results = [arg.type for arg in iter_args]

        super().__init__(results, lower_bound, upper_bound, step,
                        iter_args, loc=loc, ip=ip)

        # Create body block with induction var + iter_args
        self.regions[0].blocks.append(self.operands[0].type, *results)

    @property
    def induction_variable(self):
        """Loop induction variable"""
        return self.body.arguments[0]

    @property
    def inner_iter_args(self):
        """Loop-carried arguments (inside loop)"""
        return self.body.arguments[1:]

# Convenient builder
def for_(start, stop=None, step=None, iter_args=None, *, loc=None, ip=None):
    """Python-style for loop builder"""
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0

    # Convert Python ints to MLIR constants
    params = [start, stop, step]
    for i, p in enumerate(params):
        if isinstance(p, int):
            p = constant(IndexType.get(), p)
        params[i] = p

    for_op = ForOp(*params, iter_args, loc=loc, ip=ip)
    with InsertionPoint(for_op.body):
        yield for_op.induction_variable, for_op.inner_iter_args
```

**Usage Example**:

```python
# Creating a for loop
with for_(0, 10, 1) as (i, _):
    # i is the induction variable
    # Operations go here
    pass

# With loop-carried values
init_value = arith.ConstantOp(T.i32(), 0)
with for_(0, 10, 1, iter_args=[init_value]) as (i, acc, results):
    # acc is the current accumulator value
    new_acc = arith.AddIOp(acc, i)
    scf.YieldOp([new_acc])  # Yield new accumulator value
# results[0] contains final accumulator value
```

**Core MLIR Mapping**:
- `scf.for`: Maps to `mlir::scf::ForOp`
- `scf.if`: Maps to `mlir::scf::IfOp`
- `scf.while`: Maps to `mlir::scf::WhileOp`
- `scf.yield`: Maps to `mlir::scf::YieldOp`
- Header: `mlir/include/mlir/Dialect/SCF/IR/SCF.h`

---

### 4.5 Arith Dialect Usage

**File**: [cutlass/base_dsl/_mlir_helpers/arith.py](cutlass/python/CuTeDSL/cutlass/base_dsl/_mlir_helpers/arith.py)

CuTeDSL provides helpers for arithmetic operations:

```python
from ..._mlir.dialects import arith

# Constant creation
def const(value, type=None, *, loc=None, ip=None):
    """Create constant operation"""
    if type is None:
        # Infer type from value
        if isinstance(value, int):
            type = T.i32()
        elif isinstance(value, float):
            type = T.f32()

    return arith.ConstantOp(type, value, loc=loc, ip=ip)
```

**Common Arith Operations**:

```python
# Integer arithmetic
result = arith.AddIOp(lhs, rhs)      # Integer addition
result = arith.SubIOp(lhs, rhs)      # Integer subtraction
result = arith.MulIOp(lhs, rhs)      # Integer multiplication
result = arith.DivSIOp(lhs, rhs)     # Signed integer division
result = arith.RemSIOp(lhs, rhs)     # Signed integer remainder

# Floating-point arithmetic
result = arith.AddFOp(lhs, rhs)      # Float addition
result = arith.MulFOp(lhs, rhs)      # Float multiplication

# Comparisons
result = arith.CmpIOp(predicate, lhs, rhs)  # Integer compare
result = arith.CmpFOp(predicate, lhs, rhs)  # Float compare

# Type conversions
result = arith.TruncIOp(dest_type, src)     # Truncate integer
result = arith.ExtSIOp(dest_type, src)      # Sign-extend integer
result = arith.FPToSIOp(dest_type, src)     # Float to signed int
result = arith.SIToFPOp(dest_type, src)     # Signed int to float
```

**Core MLIR Mapping**:
- Header: `mlir/include/mlir/Dialect/Arith/IR/Arith.h`
- Ops: `mlir::arith::AddIOp`, `mlir::arith::MulFOp`, etc.

---

## 5. Regions and Blocks

### 5.1 Region Management

**File**: [cutlass/_mlir/extras/meta.py](cutlass/python/CuTeDSL/cutlass/_mlir/extras/meta.py:11-81)

```python
def op_region_builder(op, op_region, terminator=None):
    """Build a region with automatic block creation"""

    def builder_wrapper(body_builder):
        # Create entry block if doesn't exist
        if len(op_region.blocks) == 0:
            # Get type annotations from function signature
            sig = inspect.signature(body_builder)
            types = [p.annotation for p in sig.parameters.values()]

            # Create block with typed arguments
            op_region.blocks.append(*types)

        # Build region body
        with InsertionPoint(op_region.blocks[0]):
            results = body_builder(*list(op_region.blocks[0].arguments))

        # Add terminator if specified
        with InsertionPoint(list(op_region.blocks)[-1]):
            if terminator is not None:
                if isinstance(results, (tuple, list)):
                    terminator(results)
                elif results is not None:
                    terminator([results])

        return get_op_result_or_op_results(op)

    return builder_wrapper
```

**Core MLIR API**:

```python
# Region and Block API
operation.regions              # Access operation's regions
region = operation.regions[0]  # Get first region
region.blocks                  # Access blocks in region
block = region.blocks[0]       # Get first block

# Creating blocks
block = region.blocks.append(type1, type2, ...)  # Create block with args
block.arguments               # Access block arguments

# Block operations
block.operations              # Iterate operations in block
for op in block:
    # Process operation
    pass
```

**C++ Classes**:
- `mlir::Region`: Container for blocks
- `mlir::Block`: Container for operations
- Header: `mlir/include/mlir/IR/Block.h`, `mlir/include/mlir/IR/Region.h`

---

### 5.2 Block Argument Handling

**File**: [cutlass/_mlir/dialects/func.py](cutlass/python/CuTeDSL/cutlass/_mlir/dialects/func.py:92-101)

```python
def add_entry_block(self, arg_locs: Optional[Sequence[Location]] = None):
    """Add entry block with arguments from function type"""
    if not self.is_external:
        raise IndexError("Function already has entry block")

    # Create block with arguments matching function inputs
    self.body.blocks.append(*self.type.inputs, arg_locs=arg_locs)
    return self.body.blocks[0]

@property
def arguments(self):
    """Get function arguments (block arguments of entry block)"""
    return self.entry_block.arguments
```

**Core MLIR API**:

```python
# Block arguments
block.arguments                    # Get all arguments
block.arguments[0]                # Get specific argument
arg = block.add_argument(type, loc)  # Add new argument
block.erase_argument(index)       # Remove argument

# Argument properties
arg.type                          # Get argument type
arg.owner                         # Get owning block
arg.arg_number                    # Get position in block
```

---

## 6. Pass Management

### 6.1 PassManager Usage

**File**: [cutlass/_mlir/passmanager.py](cutlass/python/CuTeDSL/cutlass/_mlir/passmanager.py:5)

```python
# Simple re-export
from ._mlir_libs._cutlass_ir._mlir.passmanager import *
```

**File**: [cutlass/base_dsl/compiler.py](cutlass/python/CuTeDSL/cutlass/base_dsl/compiler.py:136-161)

```python
def compile(self, module, pipeline: str, cuda_toolkit: str = "",
            arch: str = "", enable_verifier=False):
    """Compile module using pass pipeline"""
    try:
        # Parse pass pipeline from string
        pm = self.passmanager.PassManager.parse(pipeline)

        # Enable IR verification between passes
        pm.enable_verifier(enable_verifier)

        # Run pass pipeline on module
        pm.run(module.operation)

    except Exception as e:
        # Enhanced error reporting
        error_msg = str(e)
        nvvm_error, ir_msg = self._process_error(error_msg)
        if nvvm_error:
            raise CompilationError(...)
        raise e
```

**Common Pipeline Examples**:

```python
# GPU compilation pipeline
pipeline = """
    builtin.module(
        func.func(convert-linalg-to-loops),
        gpu-kernel-outlining,
        convert-scf-to-cf,
        convert-arith-to-llvm,
        convert-func-to-llvm,
        gpu.module(
            strip-debuginfo,
            convert-gpu-to-nvvm,
            gpu-to-cubin
        ),
        gpu-to-llvm
    )
"""

# Optimization pipeline
pipeline = "builtin.module(canonicalize, cse, symbol-dce)"

# Custom nested pipeline
pipeline = "builtin.module(func.func(my-custom-pass))"
```

**Core MLIR API**:

```python
from mlir import passmanager as pm

# Creating pass manager
pass_manager = pm.PassManager.parse(pipeline_str)

# Running passes
pass_manager.run(module.operation)

# Enabling options
pass_manager.enable_verifier(True)
pass_manager.enable_timing()
```

**C++ API Reference**:
- Header: `mlir/include/mlir/Pass/PassManager.h`
- Class: `mlir::PassManager`
- Pipeline syntax: Textual pass pipeline format

---

## 7. Execution Engine

### 7.1 JIT Compilation

**File**: [cutlass/_mlir/execution_engine.py](cutlass/python/CuTeDSL/cutlass/_mlir/execution_engine.py:5-41)

```python
from ._mlir_libs._cutlass_ir import _mlirExecutionEngine as _execution_engine
import ctypes

class ExecutionEngine(_execution_engine.ExecutionEngine):
    """Wrapper for MLIR JIT execution engine"""

    def lookup(self, name):
        """
        Lookup a function with llvm.emit_c_interface attribute.
        Returns a ctypes callable.
        """
        func = self.raw_lookup("_mlir_ciface_" + name)
        if not func:
            raise RuntimeError("Unknown function " + name)

        # Create function prototype
        prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        return prototype(func)

    def invoke(self, name, *ctypes_args):
        """Invoke a function with ctypes arguments"""
        func = self.lookup(name)

        # Pack arguments
        packed_args = (ctypes.c_void_p * len(ctypes_args))()
        for argNum in range(len(ctypes_args)):
            packed_args[argNum] = ctypes.cast(ctypes_args[argNum],
                                             ctypes.c_void_p)
        func(packed_args)

    def register_runtime(self, name, ctypes_callback):
        """
        Register a runtime function available to JITted code.
        The callback must outlive the execution engine.
        """
        callback = ctypes.cast(ctypes_callback, ctypes.c_void_p)
        self.raw_register_runtime("_mlir_ciface_" + name, callback)
```

**Usage Example**:

```python
# Compile module to LLVM
pipeline = "convert-to-llvm, reconcile-unrealized-casts"
pm = PassManager.parse(pipeline)
pm.run(module.operation)

# Create execution engine
engine = ExecutionEngine(
    module,
    opt_level=2,
    shared_libs=["libcudart.so", "libnvgpu.so"]
)

# Invoke function
import ctypes
result = ctypes.c_float(0.0)
engine.invoke("my_function", ctypes.byref(result))
```

**Core MLIR API**:

```python
from mlir.execution_engine import ExecutionEngine

# Create engine
engine = ExecutionEngine(
    module,
    opt_level=2,              # LLVM optimization level (0-3)
    shared_libs=[...]         # Additional shared libraries
)

# Lookup and invoke functions
func_ptr = engine.lookup("function_name")
func_ptr(arg1, arg2, ...)
```

**C++ API Reference**:
- Header: `mlir/include/mlir/ExecutionEngine/ExecutionEngine.h`
- Class: `mlir::ExecutionEngine`
- Uses LLVM ORC JIT under the hood

---

### 7.2 Runtime Integration

**File**: [cutlass/base_dsl/compiler.py](cutlass/python/CuTeDSL/cutlass/base_dsl/compiler.py:166-193)

```python
def jit(self, module, opt_level: int = 2, shared_libs: Sequence[str] = ()):
    """Create JIT execution engine for module"""

    # Check CUDA dependencies before JIT
    self._check_cuda_dependencies_once(shared_libs)

    # Create execution engine
    return self.execution_engine.ExecutionEngine(
        module,
        opt_level=opt_level,
        shared_libs=shared_libs
    )

def compile_and_jit(self, module, pipeline: str,
                    shared_libs: Sequence[str] = (),
                    opt_level: int = 2, cuda_toolkit: str = "",
                    arch: str = ""):
    """Complete compilation and JIT pipeline"""
    # Run pass pipeline
    self.compile(module, pipeline, cuda_toolkit, arch)

    # Create execution engine
    return self.jit(module, opt_level, shared_libs)
```

---

## 8. Advanced Features

### 8.1 Custom Value Wrappers

**File**: [cutlass/base_dsl/_mlir_helpers/arith.py](cutlass/python/CuTeDSL/cutlass/base_dsl/_mlir_helpers/arith.py:394)

```python
class ArithValue(ir.Value):
    """
    Enhanced Value wrapper with operator overloading.
    Enables natural Python arithmetic on MLIR values.
    """

    def __add__(self, other):
        if is_float_type(self.type):
            return arith.AddFOp(self, other)
        else:
            return arith.AddIOp(self, other)

    def __mul__(self, other):
        if is_float_type(self.type):
            return arith.MulFOp(self, other)
        else:
            return arith.MulIOp(self, other)

    # ... more operators
```

**Usage**:

```python
# Instead of:
result = arith.AddIOp(a, b)
result = arith.MulIOp(result, c)

# Can write:
result = (a + b) * c  # If wrapped in ArithValue
```

---

### 8.2 Custom Type Wrappers

**File**: [cutlass/cute/typing.py](cutlass/python/CuTeDSL/cutlass/cute/typing.py:94)

```python
class Layout(ir.Value):
    """Wraps MLIR value representing a CuTe layout"""

    def __init__(self, shape, stride):
        # Create underlying MLIR value
        self._value = _cute_ir.make_layout(shape, stride)

    @property
    def shape(self):
        return self._shape

    @property
    def stride(self):
        return self._stride
```

This pattern allows CuTeDSL to provide domain-specific abstractions while using MLIR values underneath.

---

### 8.3 Error Handling with Diagnostics

**File**: [cutlass/_mlir/_mlir_libs/__init__.py](cutlass/python/CuTeDSL/cutlass/_mlir/_mlir_libs/__init__.py:145-177)

```python
class MLIRError(Exception):
    """Exception with diagnostic information"""

    def __init__(self, message, error_diagnostics):
        self.message = message
        self.error_diagnostics = error_diagnostics
        super().__init__(message, error_diagnostics)

    def __str__(self):
        s = self.message
        if self.error_diagnostics:
            s += ":"

        # Format diagnostics
        for diag in self.error_diagnostics:
            s += (
                "\nerror: " + str(diag.location)[4:-1] +
                ": " + diag.message.replace("\n", "\n  ")
            )

            # Include notes
            for note in diag.notes:
                s += (
                    "\n note: " + str(note.location)[4:-1] +
                    ": " + note.message.replace("\n", "\n  ")
                )
        return s

ir.MLIRError = MLIRError
```

**Core MLIR API**:

```python
# Diagnostic handling
class DiagnosticHandler:
    def __call__(self, diagnostic):
        # diagnostic.location: Source location
        # diagnostic.message: Error message
        # diagnostic.severity: ERROR, WARNING, NOTE, REMARK
        # diagnostic.notes: List of attached notes
        pass

# Attach handler to context
with ir.Context() as ctx:
    ctx.attach_diagnostic_handler(DiagnosticHandler())
```

---

## 9. Core MLIR APIs Reference Table

| CuTeDSL Component | MLIR Python API | C++ Class | Header File |
|-------------------|-----------------|-----------|-------------|
| Context | `ir.Context` | `mlir::MLIRContext` | `mlir/IR/MLIRContext.h` |
| Module | `ir.Module` | `mlir::ModuleOp` | `mlir/IR/BuiltinOps.h` |
| Operation | `ir.Operation` | `mlir::Operation` | `mlir/IR/Operation.h` |
| Value | `ir.Value` | `mlir::Value` | `mlir/IR/Value.h` |
| Type | `ir.Type` | `mlir::Type` | `mlir/IR/Types.h` |
| Attribute | `ir.Attribute` | `mlir::Attribute` | `mlir/IR/Attributes.h` |
| Location | `ir.Location` | `mlir::Location` | `mlir/IR/Location.h` |
| Region | `ir.Region` | `mlir::Region` | `mlir/IR/Region.h` |
| Block | `ir.Block` | `mlir::Block` | `mlir/IR/Block.h` |
| InsertionPoint | `ir.InsertionPoint` | `mlir::OpBuilder` | `mlir/IR/Builders.h` |
| PassManager | `passmanager.PassManager` | `mlir::PassManager` | `mlir/Pass/PassManager.h` |
| ExecutionEngine | `execution_engine.ExecutionEngine` | `mlir::ExecutionEngine` | `mlir/ExecutionEngine/ExecutionEngine.h` |

---

## 10. Key Patterns and Best Practices

### 10.1 Context Management

```python
# Always use context manager
with ir.Context() as ctx:
    ctx.load_all_available_dialects()

    with ir.Location.unknown():
        module = ir.Module.create()
        # Build IR...
```

### 10.2 Operation Building

```python
# Use InsertionPoint context manager
with ir.InsertionPoint(block):
    # Operations are automatically inserted at the right place
    op1 = arith.AddIOp(a, b)
    op2 = arith.MulIOp(op1, c)
```

### 10.3 Function Construction

```python
# Use body_builder pattern for clean function construction
def build_func_body(func_op):
    args = func_op.arguments
    # Build operations using args
    result = arith.AddIOp(args[0], args[1])
    func.ReturnOp([result])

func_op = func.FuncOp(
    "my_func",
    ([T.i32(), T.i32()], [T.i32()]),
    body_builder=build_func_body
)
```

### 10.4 Type Safety

```python
# Use typed wrappers
from cutlass._mlir.extras import types as T

# Type-safe construction
memref_type = T.memref(16, 16, T.f16(), memory_space=3)
vector_type = T.vector(4, T.f32())
```

---

## 11. Common Usage Patterns in CuTeDSL

### 11.1 Creating a GPU Kernel

```python
from cutlass._mlir import ir
from cutlass._mlir.dialects import func, gpu, arith
from cutlass._mlir.extras import types as T

with ir.Context() as ctx, ir.Location.unknown():
    module = ir.Module.create()

    # Create kernel function
    with ir.InsertionPoint(module.body):
        @func.from_py_func(T.memref(T.f32()), T.memref(T.f32()))
        def kernel(input_memref, output_memref):
            # GPU grid/block configuration
            block_x = arith.ConstantOp(T.index(), 256)

            # Launch GPU kernel
            gpu.launch_func(
                grid_size=[block_x, 1, 1],
                block_size=[256, 1, 1],
                args=[input_memref, output_memref]
            )
            return []
```

### 11.2 Building Tensor Operations

```python
from cutlass.cute import Tensor, Layout, make_tensor

# High-level CuTe API (built on MLIR)
layout = Layout(shape=(16, 16), stride=(16, 1))
tensor = make_tensor(ptr, layout)

# Internally creates MLIR operations:
# cute.make_layout, cute.make_tensor, etc.
```

---

## 12. Debugging and Introspection

### 12.1 Printing IR

```python
# Print module
print(module)

# Print operation
print(operation)

# Print with debug info
module.operation.print(enable_debug_info=True, print_generic_op_form=True)
```

### 12.2 Walking Operations

```python
# Walk all operations
def walk_callback(op):
    print(f"Operation: {op.name}")
    return WalkResult.ADVANCE

module.operation.walk(walk_callback)
```

### 12.3 Verification

```python
# Verify module/operation
try:
    module.operation.verify()
except Exception as e:
    print(f"Verification failed: {e}")
```

---

## 13. Performance Considerations

### 13.1 Context Reuse

```python
# Reuse context for multiple modules
ctx = ir.Context()
ctx.load_all_available_dialects()

# Build multiple modules with same context
for _ in range(10):
    with ctx, ir.Location.unknown():
        module = ir.Module.create()
        # ...
```

### 13.2 Pass Pipeline Optimization

```python
# Use efficient pass pipelines
# Bad: Running many small pipelines
pm1 = PassManager.parse("canonicalize")
pm1.run(module.operation)
pm2 = PassManager.parse("cse")
pm2.run(module.operation)

# Good: Single pipeline
pm = PassManager.parse("canonicalize,cse")
pm.run(module.operation)
```

---

## 14. Resources and References

### Official MLIR Documentation
- **MLIR Website**: https://mlir.llvm.org/
- **Python Bindings**: https://mlir.llvm.org/docs/Bindings/Python/
- **Dialects**: https://mlir.llvm.org/docs/Dialects/

### C++ API Documentation
- **Doxygen**: https://mlir.llvm.org/doxygen/
- **Core IR**: https://mlir.llvm.org/docs/LangRef/

### Source Code Locations
- **MLIR Source**: `mlir/include/mlir/`
- **Python Bindings**: `mlir/lib/Bindings/Python/`
- **Python Bindings Source**: `mlir/python/mlir/`

### CuTeDSL Specific
- **CuTeDSL Docs**: Check `cutlass/python/CuTeDSL/docs/` (if available)
- **CuTe Dialect**: `cutlass/python/CuTeDSL/cutlass/_mlir/dialects/cute.py`

---

## Summary

CuTeDSL extensively uses MLIR's Python bindings at multiple levels:

1. **Core IR**: Full use of Context, Module, Operation, Value, Type, Attribute APIs
2. **Type System**: Wraps and extends MLIR types with convenient constructors
3. **Dialects**: Uses standard dialects (func, arith, scf, gpu) and adds custom ones (cute)
4. **Pass Management**: Leverages PassManager for optimization and lowering
5. **Execution**: Uses ExecutionEngine for JIT compilation
6. **Advanced**: Custom wrappers, operator overloading, enhanced error handling

The key pattern is that CuTeDSL provides high-level abstractions while directly exposing MLIR's power when needed, creating a smooth gradient from easy-to-use DSL to low-level IR control.
