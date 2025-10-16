# MLIR Python ODS Integration: Complete Guide

> **A comprehensive guide to understanding how Python integrates with MLIR's Operation Definition Specification (ODS) system**

This guide traces the entire Python-to-C++ stack for MLIR operations, from high-level Python APIs down to the underlying C/C++ implementation. It's designed for someone learning MLIR through its Python bindings with minimal background knowledge.

---

## Table of Contents

1. [What is ODS?](#what-is-ods)
2. [Architecture Overview](#architecture-overview)
3. [The Generation Pipeline](#the-generation-pipeline)
4. [Generated `_{DIALECT}_ops_gen.py` Files](#generated-_dialect_ops_genpy-files)
5. [The `_ods_common` Module](#the-_ods_common-module)
6. [The `_ods_ir` Module and `OpView`](#the-_ods_ir-module-and-opview)
7. [Full Stack Trace: Python to C++](#full-stack-trace-python-to-c)
8. [Common Usage Examples](#common-usage-examples)
9. [How to Extend Operations](#how-to-extend-operations)
10. [How to Bind New Dialects](#how-to-bind-new-dialects)

---

## What is ODS?

**ODS (Operation Definition Specification)** is MLIR's declarative system for defining operations, types, and attributes using TableGen (a domain-specific language). Instead of writing boilerplate C++ code for each operation, you write concise `.td` (TableGen) files that describe:

- Operation names and namespaces
- Input operands and output results
- Attributes (compile-time parameters)
- Regions (nested IR scopes)
- Traits (properties like commutativity, pure functions, etc.)
- Verification logic
- Assembly format

**Why ODS?** It provides:
- **Single source of truth**: One definition generates C++, Python, documentation, and more
- **Consistency**: Ensures operations follow MLIR conventions
- **Maintainability**: Changes propagate automatically to all language bindings

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ODS Specification (.td files)                │
│                  e.g., ArithOps.td, FuncOps.td                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ mlir-tblgen
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                 Generated Python Bindings                      │
│         _arith_ops_gen.py, _func_ops_gen.py, etc.              │
└────────────┬───────────────────────────────────────────────────┘
             │
             │ imports
             ▼
┌────────────────────────────────────────────────────────────────┐
│                  Hand-Written Python Wrappers                  │
│              arith.py, func.py (in dialects/)                  │
│         • Imports from _*_ops_gen.py                           │
│         • Adds convenience methods                             │
│         • Overrides with @register_operation(replace=True)     │
└────────────┬───────────────────────────────────────────────────┘
             │
             │ uses
             ▼
┌────────────────────────────────────────────────────────────────┐
│              _ods_common.py (Python utilities)                 │
│         segmented_accessor, equally_sized_accessor, etc.       │
└────────────┬───────────────────────────────────────────────────┘
             │
             │ calls into
             ▼
┌────────────────────────────────────────────────────────────────┐
│           _mlir.ir (C++ Extension via nanobind)                │
│         OpView, Operation, Context, Value, Type, etc.          │
└────────────┬───────────────────────────────────────────────────┘
             │
             │ wraps
             ▼
┌────────────────────────────────────────────────────────────────┐
│                  MLIR C API (mlir-c/)                          │
│         MlirOperation, MlirContext, MlirValue, etc.            │
└────────────┬───────────────────────────────────────────────────┘
             │
             │ wraps
             ▼
┌────────────────────────────────────────────────────────────────┐
│               MLIR C++ Core (mlir/IR/)                         │
│         mlir::Operation, mlir::MLIRContext, etc.               │
└────────────────────────────────────────────────────────────────┘
```

---

## The Generation Pipeline

### Step 1: Write ODS Definition

**File**: [`mlir/include/mlir/Dialect/Arith/IR/ArithOps.td`](mlir/include/mlir/Dialect/Arith/IR/ArithOps.td)

```tablegen
def AddFOp : Arith_Op<"addf", [Pure, SameOperandsAndResultType]> {
  let summary = "floating-point addition";
  let description = [{
    Performs floating-point addition of two operands.
  }];

  let arguments = (ins
    AnyFloat:$lhs,
    AnyFloat:$rhs,
    DefaultValuedAttr<Arith_FastMathAttr, "fastmath::FastMathFlags::none">:$fastmath
  );

  let results = (outs AnyFloat:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

### Step 2: Configure CMake Binding

**File**: [`mlir/python/CMakeLists.txt`](mlir/python/CMakeLists.txt)

```cmake
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/ArithOps.td
  SOURCES
    dialects/arith.py
  DIALECT_NAME arith
  GEN_ENUM_BINDINGS)
```

This invokes:
```bash
mlir-tblgen -gen-python-op-bindings \
            -bind-dialect=arith \
            ArithOps.td \
            -o _arith_ops_gen.py
```

**Generator Source**: [`mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp`](mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp#L1)

### Step 3: Generated Python Code

**File**: [`build/tools/mlir/python_packages/mlir_core/mlir/dialects/_arith_ops_gen.py`](build/tools/mlir/python_packages/mlir_core/mlir/dialects/_arith_ops_gen.py)

```python
# Autogenerated by mlir-tblgen; don't manually edit.

from ._ods_common import _cext as _ods_cext
from ._ods_common import (
    equally_sized_accessor as _ods_equally_sized_accessor,
    get_default_loc_context as _ods_get_default_loc_context,
    get_op_result_or_op_results as _get_op_result_or_op_results,
    get_op_results_or_values as _get_op_results_or_values,
    segmented_accessor as _ods_segmented_accessor,
)
_ods_ir = _ods_cext.ir
_ods_cext.globals.register_traceback_file_exclusion(__file__)

import builtins
from typing import Sequence as _Sequence, Union as _Union

@_ods_cext.register_dialect
class _Dialect(_ods_ir.Dialect):
  DIALECT_NAMESPACE = "arith"

@_ods_cext.register_operation(_Dialect)
class AddFOp(_ods_ir.OpView):
  OPERATION_NAME = "arith.addf"
  _ODS_REGIONS = (0, True)

  def __init__(self, lhs, rhs, *, fastmath=None, results=None, loc=None, ip=None):
    operands = []
    attributes = {}
    regions = None
    operands.append(lhs)
    operands.append(rhs)
    _ods_context = _ods_get_default_loc_context(loc)
    if fastmath is not None:
      attributes["fastmath"] = (fastmath if (
          isinstance(fastmath, _ods_ir.Attribute) or
          not _ods_ir.AttrBuilder.contains('Arith_FastMathAttr')) else
            _ods_ir.AttrBuilder.get('Arith_FastMathAttr')(fastmath, context=_ods_context))
    if results is None:
      results = [operands[0].type] * 1  # Type inference from SameOperandsAndResultType
    _ods_successors = None
    super().__init__(self.OPERATION_NAME, self._ODS_REGIONS, self._ODS_OPERAND_SEGMENTS,
                     self._ODS_RESULT_SEGMENTS, attributes=attributes, results=results,
                     operands=operands, successors=_ods_successors, regions=regions,
                     loc=loc, ip=ip)

  @builtins.property
  def lhs(self):
    return self.operation.operands[0]

  @builtins.property
  def rhs(self):
    return self.operation.operands[1]

  @builtins.property
  def fastmath(self):
    return self.operation.attributes["fastmath"]

  @fastmath.setter
  def fastmath(self, value):
    if value is None:
      raise ValueError("'None' not allowed as value for mandatory attributes")
    self.operation.attributes["fastmath"] = value

  @builtins.property
  def result(self):
    return self.operation.results[0]

def addf(lhs, rhs, *, fastmath=None, results=None, loc=None, ip=None) -> _ods_ir.Value:
  return AddFOp(lhs=lhs, rhs=rhs, fastmath=fastmath, results=results, loc=loc, ip=ip).result
```

---

## Generated `_{DIALECT}_ops_gen.py` Files

### Purpose

These auto-generated files provide:

1. **Dialect Registration**: `@register_dialect` decorator registers the dialect with the Python runtime
2. **Operation Classes**: Python classes inheriting from `OpView` for each operation
3. **Type-Safe Constructors**: `__init__` methods with proper argument handling
4. **Property Accessors**: Python properties for operands, results, attributes, and regions
5. **Convenience Functions**: Snake_case wrapper functions (e.g., `addf()` for `AddFOp`)

### File Structure

| Component | Purpose | Example |
|-----------|---------|---------|
| **Header** | Import dependencies from `_ods_common` | `from ._ods_common import _cext` |
| **Dialect Class** | Declares dialect namespace | `class _Dialect(_ods_ir.Dialect): DIALECT_NAMESPACE = "arith"` |
| **Operation Classes** | One per ODS operation definition | `class AddFOp(_ods_ir.OpView)` |
| **Operand Accessors** | Properties to access operation operands | `@property def lhs(self): return self.operation.operands[0]` |
| **Result Accessors** | Properties to access operation results | `@property def result(self): return self.operation.results[0]` |
| **Attribute Accessors** | Getter/setter/deleter for attributes | `@property def fastmath(self)` |
| **Region Accessors** | Access to nested regions | `@property def body(self): return self.regions[0]` |
| **Builder Functions** | Convenience constructors | `def addf(lhs, rhs, ...) -> Value` |

### Key Generated Patterns

#### 1. Single Operand/Result

```python
@builtins.property
def lhs(self):
  return self.operation.operands[0]
```

**Location**: [`mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp#L96-100`](mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp#L96-100)

#### 2. Variadic Operands/Results (Single Group)

```python
@builtins.property
def inputs(self):
  _ods_variadic_group_length = len(self.operation.operands) - 2 + 1
  return self.operation.operands[0:0 + _ods_variadic_group_length]
```

**Location**: [`mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp#L131-140`](mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp#L131-140)

#### 3. Attribute-Sized Segments (Multiple Variadic Groups)

```python
@builtins.property
def outputs(self):
  operand_range = _ods_segmented_accessor(
       self.operation.operands,
       self.operation.attributes["operandSegmentSizes"], 2)
  return operand_range
```

**Location**: [`mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp#L168-181`](mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp#L168-181)

Uses the `AttrSizedOperandSegments` trait, which requires an attribute storing segment sizes.

#### 4. Type Inference

```python
if results is None:
  results = [operands[0].type] * 1
```

For operations with `SameOperandsAndResultType` trait - result types are inferred from operand types.

---

## The `_ods_common` Module

**File**: [`mlir/python/mlir/dialects/_ods_common.py`](mlir/python/mlir/dialects/_ods_common.py)

This module provides runtime utilities for generated code to handle complex operand/result access patterns.

### Key Functions

#### `segmented_accessor(elements, raw_segments, idx)`

**Purpose**: Access a segment when using `AttrSizedOperandSegments` or `AttrSizedResultSegments` traits.

**Source**: [`_ods_common.py#L38-50`](mlir/python/mlir/dialects/_ods_common.py#L38-50)

```python
def segmented_accessor(elements, raw_segments, idx):
    """
    Returns a slice of elements corresponding to the idx-th segment.

    elements: a sliceable container (operands or results).
    raw_segments: an mlir.ir.Attribute, of DenseI32Array subclass containing
        sizes of the segments.
    idx: index of the segment.
    """
    segments = _cext.ir.DenseI32ArrayAttr(raw_segments)
    start = sum(segments[i] for i in range(idx))
    end = start + segments[idx]
    return elements[start:end]
```

**Example Usage**:
```python
# For an op with operands [a, b, c, d, e] and segment sizes [2, 3]
# Accessing segment 1 returns [c, d, e]
operand_range = segmented_accessor(
    self.operation.operands,
    self.operation.attributes["operandSegmentSizes"],
    1)
```

**Binding**: This is pure Python, no C++ binding needed.

---

#### `equally_sized_accessor(elements, n_simple, n_variadic, n_preceding_simple, n_preceding_variadic)`

**Purpose**: Access variadic groups when using `SameVariadicOperandSize` or `SameVariadicResultSize` traits.

**Source**: [`_ods_common.py#L53-76`](mlir/python/mlir/dialects/_ods_common.py#L53-76)

```python
def equally_sized_accessor(
    elements, n_simple, n_variadic, n_preceding_simple, n_preceding_variadic
):
    """
    Returns a starting position and a number of elements per variadic group
    assuming equally-sized groups.
    """
    total_variadic_length = len(elements) - n_simple
    assert total_variadic_length % n_variadic == 0

    elements_per_group = total_variadic_length // n_variadic
    start = n_preceding_simple + n_preceding_variadic * elements_per_group
    return start, elements_per_group
```

**Example Usage**:
```python
# For an op with 3 variadic groups of equal size
# operands = [a, b, c, d, e, f] (2 simple + 2*2 variadic)
start, per_group = equally_sized_accessor(self.operation.operands, 2, 2, 1, 0)
# Returns start=1, per_group=2 for the first variadic group after the first simple operand
```

**Binding**: Pure Python utility.

---

#### `get_default_loc_context(location=None)`

**Purpose**: Get the MLIR context from a location, or from the thread-local stack if location is None.

**Source**: [`_ods_common.py#L78-88`](mlir/python/mlir/dialects/_ods_common.py#L78-88)

```python
def get_default_loc_context(location=None):
    """
    Returns a context in which the defaulted location is created.
    """
    if location is None:
        if _cext.ir.Location.current:
            return _cext.ir.Location.current.context
        return None
    return location.context
```

**Binding**:
- `_cext.ir.Location.current` → C++ property [`PyLocation::current`](mlir/lib/Bindings/Python/IRModule.h#L130)
- `.context` → C++ property [`PyLocation::getContext()`](mlir/lib/Bindings/Python/IRModule.h#L276)

---

#### `get_op_result_or_value(arg)`

**Purpose**: Convert an `OpView` or `Operation` to a single `Value` (convenience for operation constructors).

**Source**: [`_ods_common.py#L90-110`](mlir/python/mlir/dialects/_ods_common.py#L90-110)

```python
def get_op_result_or_value(
    arg: _Union[_cext.ir.OpView, _cext.ir.Operation, _cext.ir.Value, _cext.ir.OpResultList]
) -> _cext.ir.Value:
    """Returns the given value or the single result of the given op."""
    if isinstance(arg, _cext.ir.OpView):
        return arg.operation.result
    elif isinstance(arg, _cext.ir.Operation):
        return arg.result
    elif isinstance(arg, _cext.ir.OpResultList):
        return arg[0]
    else:
        assert isinstance(arg, _cext.ir.Value)
        return arg
```

**Binding**:
- All types (`OpView`, `Operation`, `Value`) are C++ classes bound via nanobind
- `.operation` → [`PyOpView::getOperationObject()`](mlir/lib/Bindings/Python/IRCore.cpp#L3590)
- `.result` → [`PyOperation::result`](mlir/lib/Bindings/Python/IRModule.h#L138) property

---

### Type Aliases

**Source**: [`_ods_common.py#L151-167`](mlir/python/mlir/dialects/_ods_common.py#L151-167)

```python
ResultValueT = _Union[Operation, OpView, Value]
VariadicResultValueT = _Union[ResultValueT, _Sequence[ResultValueT]]
StaticIntLike = _Union[int, IntegerAttr]
ValueLike = _Union[Operation, OpView, Value]
MixedInt = _Union[StaticIntLike, ValueLike]
MixedValues = _Union[_Sequence[_Union[StaticIntLike, ValueLike]], ArrayAttr, ValueLike]
```

These type aliases help generated code handle the flexible input formats that MLIR Python bindings accept.

---

## The `_ods_ir` Module and `OpView`

### What is `_ods_ir`?

In generated code, you'll see:
```python
_ods_ir = _ods_cext.ir
```

**Breakdown**:
- `_ods_cext` → The `_mlir` C++ extension module
- `_cext` is defined in [`_ods_common.py#L14`](mlir/python/mlir/dialects/_ods_common.py#L14): `from .._mlir_libs import _mlir as _cext`
- `.ir` → The `ir` submodule containing core IR classes

**C++ Source**: [`mlir/lib/Bindings/Python/MainModule.cpp#L25-147`](mlir/lib/Bindings/Python/MainModule.cpp#L25-147)

```cpp
NB_MODULE(_mlir, m) {
  m.doc() = "MLIR Python Native Extension";

  // ... PyGlobals setup ...

  // Define and populate IR submodule.
  auto irModule = m.def_submodule("ir", "MLIR IR Bindings");
  populateIRCore(irModule);
  populateIRAffine(irModule);
  populateIRAttributes(irModule);
  populateIRInterfaces(irModule);
  populateIRTypes(irModule);
  // ...
}
```

### `OpView` Class

**Purpose**: Base class for all generated operation classes. Provides a Python-friendly wrapper around the lower-level `Operation` class.

**Python Interface**: [`mlir/python/mlir/ir.py`](mlir/python/mlir/ir.py) (imports from C++ module)

**C++ Binding**: [`mlir/lib/Bindings/Python/IRCore.cpp#L3560-3657`](mlir/lib/Bindings/Python/IRCore.cpp#L3560-3657)

```cpp
auto opViewClass =
    nb::class_<PyOpView, PyOperationBase>(m, "OpView")
        .def(nb::init<nb::object>(), nb::arg("operation"))
        .def(
            "__init__",
            [](PyOpView *self, std::string_view name,
               std::tuple<int, bool> opRegionSpec,
               nb::object operandSegmentSpecObj,
               nb::object resultSegmentSpecObj,
               std::optional<nb::list> resultTypeList, nb::list operandList,
               std::optional<nb::dict> attributes,
               std::optional<std::vector<PyBlock *>> successors,
               std::optional<int> regions,
               const std::optional<PyLocation> &location,
               const nb::object &maybeIp) {
              PyLocation pyLoc = maybeGetTracebackLocation(location);
              new (self) PyOpView(PyOpView::buildGeneric(
                  name, opRegionSpec, operandSegmentSpecObj,
                  resultSegmentSpecObj, resultTypeList, operandList,
                  attributes, successors, regions, pyLoc, maybeIp));
            },
            nb::arg("name"), nb::arg("opRegionSpec"),
            // ... more args ...
        )
        .def_prop_ro("operation", &PyOpView::getOperationObject)
        .def_prop_ro("opview", [](nb::object self) { return self; })
        // ... more methods ...
```

#### Key `OpView` Properties

| Property | Type | C++ Binding | Purpose |
|----------|------|-------------|---------|
| `operation` | `Operation` | [`PyOpView::getOperationObject()`](mlir/lib/Bindings/Python/IRCore.cpp#L3590) | Access underlying Operation |
| `opview` | `OpView` | Returns self | Identity accessor |
| `successors` | `OpSuccessors` | [`PyOpSuccessors`](mlir/lib/Bindings/Python/IRCore.cpp#L3597) | Block successors |

**C++ Class**: [`mlir/lib/Bindings/Python/IRModule.h#L1000-1100`](mlir/lib/Bindings/Python/IRModule.h#L1000) (approximate line)

---

## Full Stack Trace: Python to C++

Let's trace what happens when you create an operation in Python, all the way down to MLIR's C++ core.

### Example: Creating an `arith.addf` Operation

```python
from mlir import ir
from mlir.dialects import arith

with ir.Context() as ctx:
  with ir.Location.unknown(ctx):
    f32 = ir.F32Type.get()
    v1 = ...  # some Value
    v2 = ...  # some Value

    # This is what we're tracing:
    result = arith.addf(v1, v2)
```

---

### Trace Level 1: Python Generated Code

**File**: [`build/.../mlir/dialects/_arith_ops_gen.py#L66`](build/tools/mlir/python_packages/mlir_core/mlir/dialects/_arith_ops_gen.py#L66)

```python
def addf(lhs, rhs, *, fastmath=None, results=None, loc=None, ip=None) -> _ods_ir.Value:
  return AddFOp(lhs=lhs, rhs=rhs, fastmath=fastmath, results=results, loc=loc, ip=ip).result
```

**What happens**:
1. Calls `AddFOp.__init__()` (generated constructor)
2. Returns the `.result` property

---

### Trace Level 2: Generated `AddFOp.__init__`

**File**: [`build/.../mlir/dialects/_arith_ops_gen.py#L29`](build/tools/mlir/python_packages/mlir_core/mlir/dialects/_arith_ops_gen.py#L29)

```python
def __init__(self, lhs, rhs, *, fastmath=None, results=None, loc=None, ip=None):
  operands = []
  attributes = {}
  regions = None
  operands.append(lhs)
  operands.append(rhs)
  _ods_context = _ods_get_default_loc_context(loc)
  if fastmath is not None:
    attributes["fastmath"] = ...
  if results is None:
    results = [operands[0].type] * 1
  _ods_successors = None
  super().__init__(
      self.OPERATION_NAME,        # "arith.addf"
      self._ODS_REGIONS,          # (0, True)
      self._ODS_OPERAND_SEGMENTS, # None
      self._ODS_RESULT_SEGMENTS,  # None
      attributes=attributes,
      results=results,
      operands=operands,
      successors=_ods_successors,
      regions=regions,
      loc=loc,
      ip=ip)
```

**What happens**:
1. Collects operands into a list
2. Builds attributes dictionary
3. Infers result types (using `SameOperandsAndResultType` trait)
4. Calls `OpView.__init__()` (C++ binding)

---

### Trace Level 3: `OpView.__init__` (C++ Binding)

**File**: [`mlir/lib/Bindings/Python/IRCore.cpp#L3564-3588`](mlir/lib/Bindings/Python/IRCore.cpp#L3564-3588)

```cpp
.def(
    "__init__",
    [](PyOpView *self, std::string_view name,
       std::tuple<int, bool> opRegionSpec,
       nb::object operandSegmentSpecObj,
       nb::object resultSegmentSpecObj,
       std::optional<nb::list> resultTypeList, nb::list operandList,
       std::optional<nb::dict> attributes,
       std::optional<std::vector<PyBlock *>> successors,
       std::optional<int> regions,
       const std::optional<PyLocation> &location,
       const nb::object &maybeIp) {
      PyLocation pyLoc = maybeGetTracebackLocation(location);
      new (self) PyOpView(PyOpView::buildGeneric(
          name, opRegionSpec, operandSegmentSpecObj,
          resultSegmentSpecObj, resultTypeList, operandList,
          attributes, successors, regions, pyLoc, maybeIp));
    },
    nb::arg("name"), nb::arg("opRegionSpec"),
    // ... args ...
)
```

**What happens**:
1. Extract C++ types from Python objects via nanobind
2. Call `PyOpView::buildGeneric()`

---

### Trace Level 4: `PyOpView::buildGeneric`

**File**: [`mlir/lib/Bindings/Python/IRCore.cpp#L2100-2300`](mlir/lib/Bindings/Python/IRCore.cpp#L2100) (approximate)

This method:
1. **Resolves context**: Gets the current MLIR context from thread-local storage or arguments
2. **Processes operands**: Converts Python `Value` objects to `MlirValue` handles
3. **Processes attributes**: Converts Python dict to `MlirNamedAttribute` array
4. **Processes result types**: Converts Python `Type` objects to `MlirType` handles
5. **Creates regions**: If needed, creates `MlirRegion` structures
6. **Builds the operation**: Calls `mlirOperationCreate()` (C API)

```cpp
PyOpView PyOpView::buildGeneric(
    std::string_view name,
    std::tuple<int, bool> regionSpec,
    nb::object operandSegmentSpecObj,
    nb::object resultSegmentSpecObj,
    std::optional<nb::list> resultTypeList,
    nb::list operandList,
    std::optional<nb::dict> attributes,
    std::optional<std::vector<PyBlock *>> successors,
    std::optional<int> regions,
    PyLocation &pyLoc,
    const nb::object &maybeIp) {

  // ... validation and preparation ...

  // Convert operands
  SmallVector<MlirValue> mlirOperands;
  for (auto operand : operandList) {
    mlirOperands.push_back(nb::cast<PyValue *>(operand)->get());
  }

  // Convert result types
  SmallVector<MlirType> mlirResultTypes;
  for (auto type : *resultTypeList) {
    mlirResultTypes.push_back(nb::cast<PyType *>(type)->get());
  }

  // Create operation state
  MlirOperationState state = mlirOperationStateGet(
      toMlirStringRef(name),
      pyLoc);

  mlirOperationStateAddOperands(&state, mlirOperands.size(), mlirOperands.data());
  mlirOperationStateAddResults(&state, mlirResultTypes.size(), mlirResultTypes.data());
  // ... add attributes, regions, successors ...

  // Create the operation via C API
  MlirOperation operation = mlirOperationCreate(&state);

  // Wrap in Python object
  PyOperationRef pyOp = PyOperation::forOperation(
      pyContext, operation, insertionPoint);

  return PyOpView(pyOp.getObject());
}
```

**What happens**:
- Prepares `MlirOperationState` structure
- Calls into MLIR C API

---

### Trace Level 5: MLIR C API

**File**: [`mlir/lib/CAPI/IR/IR.cpp#L500-550`](mlir/lib/CAPI/IR/IR.cpp#L500) (approximate)

```cpp
MlirOperation mlirOperationCreate(MlirOperationState *state) {
  // Unwrap the state
  OperationState *cppState = unwrap(state);

  // Create the operation using C++ API
  Operation *op = Operation::create(*cppState);

  // Wrap and return
  return wrap(op);
}
```

**What happens**:
- Unwraps C handles to C++ pointers
- Calls C++ `Operation::create()`

---

### Trace Level 6: MLIR C++ Core

**File**: [`mlir/lib/IR/Operation.cpp#L200-300`](mlir/lib/IR/Operation.cpp#L200) (approximate)

```cpp
Operation *Operation::create(const OperationState &state) {
  // Lookup operation info in the dialect registry
  auto *opInfo = state.name.getAbstractOperation();

  // Allocate memory for the operation
  size_t totalSize = totalSizeToAlloc(state);
  void *rawMem = malloc(totalSize);

  // Construct the operation in-place
  Operation *op = ::new (rawMem) Operation(
      state.name,
      state.location,
      state.operands.size(),
      state.results.size(),
      state.attributes,
      state.successors.size(),
      state.regions.size(),
      opInfo);

  // Initialize operands
  op->setOperands(state.operands);

  // Initialize results
  for (unsigned i = 0; i < state.results.size(); ++i) {
    op->getResult(i).setType(state.results[i]);
  }

  // Initialize regions
  for (unsigned i = 0; i < state.regions.size(); ++i) {
    new (&op->getRegion(i)) Region(op);
  }

  // Run verifiers if registered
  if (opInfo && opInfo->verifyInvariants) {
    if (failed(opInfo->verifyInvariants(op))) {
      op->destroy();
      return nullptr;
    }
  }

  return op;
}
```

**What happens**:
- Allocates memory using trailing objects pattern
- Initializes operation data structure
- Runs verification hooks
- Returns pointer to the created operation

---

### Summary Diagram: Full Stack

```
Python:     arith.addf(v1, v2)
                    ↓
Generated:  AddFOp.__init__(v1, v2, ...)
                    ↓
Generated:  super().__init__("arith.addf", ...)
                    ↓
C++ Binding: PyOpView::__init__ [IRCore.cpp:3564]
                    ↓
C++ Binding: PyOpView::buildGeneric [IRCore.cpp:2100]
                    ↓
C API:      mlirOperationCreate(&state) [CAPI/IR/IR.cpp:500]
                    ↓
C++ Core:   Operation::create(state) [IR/Operation.cpp:200]
                    ↓
Memory:     Allocated mlir::Operation in C++ heap
```

---

## Common Usage Examples

### Example 1: Basic Operation Creation

```python
from mlir.dialects import arith
from mlir import ir

with ir.Context() as ctx, ir.Location.unknown():
  module = ir.Module.create()

  with ir.InsertionPoint(module.body):
    # Create a function
    f32 = ir.F32Type.get()
    func_type = ir.FunctionType.get([f32, f32], [f32])

    @func.FuncOp.from_py_func(f32, f32, name="add_floats")
    def add_floats(a, b):
      result = arith.addf(a, b)  # Uses generated convenience function
      return result

  print(module)
```

**Output**:
```mlir
module {
  func.func @add_floats(%arg0: f32, %arg1: f32) -> f32 {
    %0 = arith.addf %arg0, %arg1 : f32
    return %0 : f32
  }
}
```

**Trace**:
- [`arith.addf()`](build/tools/mlir/python_packages/mlir_core/mlir/dialects/_arith_ops_gen.py#L66) → calls `AddFOp(...)`
- [`AddFOp.__init__()`](build/tools/mlir/python_packages/mlir_core/mlir/dialects/_arith_ops_gen.py#L29) → builds operands, attributes
- `OpView.__init__()` → C++ creates operation

---

### Example 2: Operation with Attributes

```python
from mlir.dialects import arith
from mlir import ir

with ir.Context(), ir.Location.unknown():
  f32 = ir.F32Type.get()

  # Access the fast-math flags enum
  fastmath = arith.FastMathFlags.none

  # Create addition with fast-math attribute
  result = arith.addf(v1, v2, fastmath=fastmath)

  # Or use the class directly
  add_op = arith.AddFOp(v1, v2, fastmath=fastmath)

  # Access the attribute
  print(add_op.fastmath)

  # Modify the attribute
  add_op.fastmath = arith.FastMathFlags.fast
```

**Generated Attribute Accessor**: [`_arith_ops_gen.py#L53`](build/tools/mlir/python_packages/mlir_core/mlir/dialects/_arith_ops_gen.py#L53)

```python
@builtins.property
def fastmath(self):
  return self.operation.attributes["fastmath"]

@fastmath.setter
def fastmath(self, value):
  if value is None:
    raise ValueError("'None' not allowed as value for mandatory attributes")
  self.operation.attributes["fastmath"] = value
```

---

### Example 3: Variadic Operands

```python
from mlir.dialects import scf
from mlir import ir

with ir.Context(), ir.Location.unknown():
  # scf.while has variadic operands (loop-carried values)
  i32 = ir.IntegerType.get_signless(32)

  # Initial values for loop
  init_values = [val1, val2, val3]

  while_op = scf.WhileOp(
      [i32, i32, i32],  # result types
      init_values       # initial operands
  )

  # Access variadic operands
  before_block = while_op.before
  before_args = list(before_block.arguments)
```

**Generated Accessor** (for variadic):
```python
@builtins.property
def inits(self):
  _ods_variadic_group_length = len(self.operation.operands) - 0 + 1
  return self.operation.operands[0:0 + _ods_variadic_group_length]
```

---

### Example 4: Operations with Regions

```python
from mlir.dialects import scf, arith
from mlir import ir

with ir.Context(), ir.Location.unknown():
  i32 = ir.IntegerType.get_signless(32)

  # Create an scf.if operation (has two regions: then and else)
  condition = ...  # some i1 value

  if_op = scf.IfOp(condition, [i32], hasElse=True)

  # Access the regions
  then_block = if_op.then_block
  else_block = if_op.else_block

  # Build IR inside the then region
  with ir.InsertionPoint(then_block):
    one = arith.constant(i32, 1)
    scf.yield_([one])

  # Build IR inside the else region
  with ir.InsertionPoint(else_block):
    zero = arith.constant(i32, 0)
    scf.yield_([zero])
```

**Generated Region Accessor**: [`OpPythonBindingGen.cpp#L267-271`](mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp#L267-271)

```python
@builtins.property
def then_region(self):
  return self.regions[0]

@builtins.property
def else_region(self):
  return self.regions[1]
```

---

## How to Extend Operations

You can extend generated operation classes with custom methods by using the `@register_operation` decorator with `replace=True`.

### Pattern: Override Generated Class

**File**: [`mlir/python/mlir/dialects/arith.py`](mlir/python/mlir/dialects/arith.py)

```python
from ._arith_ops_gen import *
from ._arith_ops_gen import _Dialect
from ..ir import OpView

# Override the generated ConstantOp with custom logic
@register_operation(_Dialect, replace=True)
class ConstantOp(ConstantOp):
    """Specialization for the constant op class."""

    def __init__(self, result_type, value, *, loc=None, ip=None):
        # Custom constructor that's more ergonomic
        if isinstance(value, int) or isinstance(value, float):
            # Automatically create the appropriate attribute
            if isinstance(result_type, IntegerType):
                value = IntegerAttr.get(result_type, value)
            elif isinstance(result_type, FloatType):
                value = FloatAttr.get(result_type, value)

        # Call the generated constructor
        super().__init__(value, results=[result_type], loc=loc, ip=ip)

    @staticmethod
    def create_index(value, *, loc=None, ip=None):
        """Create an index-typed constant."""
        return ConstantOp(IndexType.get(), value, loc=loc, ip=ip)

    @property
    def literal_value(self):
        """Extract the Python literal value from the constant."""
        attr = self.value
        if isinstance(attr, IntegerAttr):
            return attr.value
        elif isinstance(attr, FloatAttr):
            return attr.value
        return None
```

**Key Points**:
1. Import the generated class: `from ._arith_ops_gen import ConstantOp`
2. Use `@register_operation(_Dialect, replace=True)` to override
3. Inherit from the generated class to keep generated methods
4. Add custom constructors, properties, or methods

### Example: Custom Builder Method

```python
@register_operation(_Dialect, replace=True)
class AddFOp(AddFOp):
    """Extended AddFOp with convenience methods."""

    @staticmethod
    def create_fast(lhs, rhs, *, loc=None, ip=None):
        """Create an addf with fast-math flags enabled."""
        from . import FastMathFlags
        return AddFOp(
            lhs, rhs,
            fastmath=FastMathFlags.fast,
            loc=loc, ip=ip
        )

    def is_fast_math_enabled(self):
        """Check if any fast-math flags are set."""
        from . import FastMathFlags
        return self.fastmath != FastMathFlags.none
```

**Usage**:
```python
# Using the custom builder
result = arith.AddFOp.create_fast(v1, v2)

# Using the custom property
if result.is_fast_math_enabled():
    print("Fast math is enabled!")
```

---

## How to Bind New Dialects

To create Python bindings for a new dialect, you need to:

1. Write the ODS definition (`.td` files)
2. Configure CMake to generate Python bindings
3. Optionally write hand-written Python extensions

### Step 1: Write ODS Definition

**File**: `mlir/include/mlir/Dialect/MyDialect/IR/MyDialectOps.td`

```tablegen
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MyDialect : Dialect {
  let name = "mydialect";
  let summary = "My custom dialect for ...";
  let cppNamespace = "::mlir::mydialect";
}

class MyDialect_Op<string mnemonic, list<Trait> traits = []> :
    Op<MyDialect, mnemonic, traits>;

def MyDialect_AddOp : MyDialect_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "Addition operation";
  let description = [{
    Performs addition of two integers.
  }];

  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def MyDialect_MulOp : MyDialect_Op<"mul", [Pure, Commutative]> {
  let summary = "Multiplication operation";

  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
}
```

### Step 2: Create Python TableGen File

**File**: `mlir/python/mlir/dialects/MyDialectOps.td`

```tablegen
include "mlir/Bindings/Python/Attributes.td"

def MyDialect_Python : PythonDialect {
  let dialectName = "mydialect";
  let dialectModule = "mlir.dialects.mydialect";
}
```

### Step 3: Configure CMake

**File**: `mlir/python/CMakeLists.txt` (add this section)

```cmake
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/MyDialectOps.td
  SOURCES
    dialects/mydialect.py
  DIALECT_NAME mydialect
)
```

**File**: `mlir/lib/Bindings/Python/CMakeLists.txt` (if you need special C++ bindings)

```cmake
declare_mlir_python_sources(MLIRPythonSources.Dialects.mydialect
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${MLIR_SOURCE_DIR}/python"
  SOURCES
    dialects/mydialect.py
)
```

### Step 4: Create Hand-Written Python Wrapper (Optional)

**File**: `mlir/python/mlir/dialects/mydialect.py`

```python
"""My custom dialect Python bindings."""

from ._mydialect_ops_gen import *
from ._mydialect_ops_gen import _Dialect
from ..ir import OpView, IntegerType, IntegerAttr

# Re-export convenience functions with better names
__all__ = [
    "AddOp",
    "MulOp",
    "add",
    "mul",
]

# Optional: Override generated classes
@register_operation(_Dialect, replace=True)
class AddOp(AddOp):
    """Enhanced addition operation."""

    @staticmethod
    def create_with_constant(value, operand, *, loc=None, ip=None):
        """Create addition with a constant."""
        const_type = operand.type
        const_val = ConstantOp(const_type, value, loc=loc, ip=ip).result
        return AddOp(const_val, operand, loc=loc, ip=ip)

# Optional: Add helper functions
def build_arithmetic_expr(op, *operands, loc=None, ip=None):
    """Build a chain of operations."""
    if len(operands) < 2:
        raise ValueError("Need at least 2 operands")

    result = op(operands[0], operands[1], loc=loc, ip=ip)
    for operand in operands[2:]:
        result = op(result.result, operand, loc=loc, ip=ip)

    return result
```

### Step 5: Build and Test

```bash
cd build
cmake --build . --target check-mlir-python
```

### Step 6: Use Your Dialect

```python
from mlir.ir import Context, Module, Location, InsertionPoint, IntegerType
from mlir.dialects import mydialect, func

with Context() as ctx:
  # Load your dialect
  ctx.load_dialect("mydialect")

  with Location.unknown():
    module = Module.create()

    with InsertionPoint(module.body):
      i32 = IntegerType.get_signless(32)

      @func.FuncOp.from_py_func(i32, i32)
      def test_func(a, b):
        # Use generated bindings
        sum = mydialect.add(a, b)
        product = mydialect.mul(sum, b)
        return product

    print(module)
```

**Output**:
```mlir
module {
  func.func @test_func(%arg0: i32, %arg1: i32) -> i32 {
    %0 = mydialect.add %arg0, %arg1 : i32
    %1 = mydialect.mul %0, %arg1 : i32
    return %1 : i32
  }
}
```

---

## Advanced Topics

### Custom Attribute Builders

**Purpose**: Automatically convert Python types to MLIR attributes in generated code.

**Register in**: [`mlir/python/mlir/ir.py`](mlir/python/mlir/ir.py#L49-54)

```python
@register_attribute_builder("MyCustomAttr")
def _myCustomAttr(x, context):
    """Convert Python object to MyCustomAttr."""
    if isinstance(x, int):
        return MyCustomAttr.get(IntegerAttr.get(IntegerType.get_signless(64, context), x))
    elif isinstance(x, str):
        return MyCustomAttr.get(StringAttr.get(x, context))
    return x
```

**Usage in Generated Code**: [`_arith_ops_gen.py#L36-39`](build/tools/mlir/python_packages/mlir_core/mlir/dialects/_arith_ops_gen.py#L36-39)

```python
if fastmath is not None:
  attributes["fastmath"] = (fastmath if (
      isinstance(fastmath, _ods_ir.Attribute) or
      not _ods_ir.AttrBuilder.contains('Arith_FastMathAttr')) else
        _ods_ir.AttrBuilder.get('Arith_FastMathAttr')(fastmath, context=_ods_context))
```

### Trait-Based Code Generation

MLIR operations can have **traits** that affect how Python bindings are generated:

| Trait | Effect on Python Binding |
|-------|--------------------------|
| `SameOperandsAndResultType` | Result types inferred from operand types |
| `FirstAttrDerivedResultType` | Result type extracted from first attribute |
| `AttrSizedOperandSegments` | Uses `segmented_accessor()` for variadic operands |
| `SameVariadicOperandSize` | Uses `equally_sized_accessor()` for variadic operands |
| `Pure` | No side effects (documented in Python) |
| `Commutative` | Operation is commutative (documented) |

**Example**: `SameOperandsAndResultType`

**ODS**:
```tablegen
def AddIOp : Arith_Op<"addi", [SameOperandsAndResultType]> { ... }
```

**Generated**:
```python
if results is None:
  results = [operands[0].type] * 1
```

**Generator Source**: [`OpPythonBindingGen.cpp#L550-600`](mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp#L550)

---

## Reference Tables

### Python to C++ Class Mapping

| Python Class | C++ Class | Source File |
|--------------|-----------|-------------|
| `ir.Context` | `PyMlirContext` | [`IRModule.h:186`](mlir/lib/Bindings/Python/IRModule.h#L186) |
| `ir.Module` | `PyModule` | [`IRModule.h:496`](mlir/lib/Bindings/Python/IRModule.h#L496) |
| `ir.Operation` | `PyOperation` | [`IRModule.h:800`](mlir/lib/Bindings/Python/IRModule.h#L800) |
| `ir.OpView` | `PyOpView` | [`IRModule.h:1000`](mlir/lib/Bindings/Python/IRModule.h#L1000) |
| `ir.Value` | `PyValue` | [`IRModule.h:1200`](mlir/lib/Bindings/Python/IRModule.h#L1200) |
| `ir.Type` | `PyType` | [`IRModule.h:1400`](mlir/lib/Bindings/Python/IRModule.h#L1400) |
| `ir.Attribute` | `PyAttribute` | [`IRModule.h:1600`](mlir/lib/Bindings/Python/IRModule.h#L1600) |
| `ir.Block` | `PyBlock` | [`IRModule.h:1800`](mlir/lib/Bindings/Python/IRModule.h#L1800) |
| `ir.Region` | `PyRegion` | [`IRModule.h:1900`](mlir/lib/Bindings/Python/IRModule.h#L1900) |
| `ir.Location` | `PyLocation` | [`IRModule.h:283`](mlir/lib/Bindings/Python/IRModule.h#L283) |
| `ir.InsertionPoint` | `PyInsertionPoint` | [`IRModule.h:2000`](mlir/lib/Bindings/Python/IRModule.h#L2000) |

### C++ to C API Mapping

| C++ Type | C API Type | Header |
|----------|------------|--------|
| `mlir::MLIRContext*` | `MlirContext` | [`mlir-c/IR.h`](mlir/include/mlir-c/IR.h) |
| `mlir::Operation*` | `MlirOperation` | [`mlir-c/IR.h`](mlir/include/mlir-c/IR.h) |
| `mlir::Value` | `MlirValue` | [`mlir-c/IR.h`](mlir/include/mlir-c/IR.h) |
| `mlir::Type` | `MlirType` | [`mlir-c/IR.h`](mlir/include/mlir-c/IR.h) |
| `mlir::Attribute` | `MlirAttribute` | [`mlir-c/IR.h`](mlir/include/mlir-c/IR.h) |
| `mlir::Block*` | `MlirBlock` | [`mlir-c/IR.h`](mlir/include/mlir-c/IR.h) |
| `mlir::Region*` | `MlirRegion` | [`mlir-c/IR.h`](mlir/include/mlir-c/IR.h) |
| `mlir::Location` | `MlirLocation` | [`mlir-c/IR.h`](mlir/include/mlir-c/IR.h) |

---

## Key Takeaways

1. **ODS is the source of truth**: All operation definitions start in `.td` files
2. **TableGen generates Python**: `mlir-tblgen` with `-gen-python-op-bindings` creates `_*_ops_gen.py` files
3. **`_ods_common` provides utilities**: Helper functions for accessing variadic operands/results
4. **`OpView` is the base class**: All generated operation classes inherit from `OpView`
5. **Nanobind connects Python to C++**: The `_mlir` extension module wraps MLIR's C API
6. **Registration happens at import time**: `@register_dialect` and `@register_operation` decorators
7. **You can extend generated classes**: Use `@register_operation(..., replace=True)` to override
8. **Multiple layers of indirection**: Python → Generated Python → C++ Bindings → C API → C++ Core

---

## Further Reading

- [MLIR Python Bindings Documentation](https://mlir.llvm.org/docs/Bindings/Python/)
- [ODS Framework Documentation](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [TableGen Language Reference](https://llvm.org/docs/TableGen/)
- [MLIR Operation Definition Specification](https://mlir.llvm.org/docs/OpDefinitions/)
- [Nanobind Documentation](https://nanobind.readthedocs.io/)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-16
**MLIR Version**: Latest from main branch
