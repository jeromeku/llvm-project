# ODS Metadata: `_ODS_OPERAND_SEGMENTS` and `_ODS_REGIONS`

This document explains the metadata structures used in MLIR's auto-generated Python operation classes, specifically `_ODS_OPERAND_SEGMENTS` and `_ODS_REGIONS`.

## Table of Contents
- [Overview](#overview)
- [_ODS_OPERAND_SEGMENTS](#_ods_operand_segments)
- [_ODS_REGIONS](#_ods_regions)
- [How They Work Together](#how-they-work-together)
- [Real-World Examples](#real-world-examples)
- [TableGen Origins](#tablegen-origins)

---

## Overview

When MLIR operations are defined in TableGen (`.td` files), the `mlir-tblgen` tool generates Python bindings. These generated classes include special metadata that describes:

1. **How operands are organized** (`_ODS_OPERAND_SEGMENTS`)
2. **How many regions the operation has** (`_ODS_REGIONS`)

### Example from Generated Code

**File**: `build/tools/mlir/python_packages/mlir_core/mlir/dialects/_nvgpu_ops_gen.py`

```python
@_ods_cext.register_operation(_Dialect)
class TmaAsyncLoadOp(_ods_ir.OpView):
    OPERATION_NAME = "nvgpu.tma.async.load"

    _ODS_OPERAND_SEGMENTS = [1, 1, 1, -1, 1, 0, 0]
    _ODS_REGIONS = (0, True)

    def __init__(self, dst, barriers, tensorMapDescriptor, coordinates, mbarId, *,
                 multicastMask=None, predicate=None, loc=None, ip=None):
        # ... implementation
```

These metadata structures enable:
- **Validation** of operands and regions at construction time
- **Automatic packing/unpacking** of variadic operands
- **Property accessors** that correctly slice the flat operand list
- **Runtime attribute generation** for segment sizes

---

## `_ODS_OPERAND_SEGMENTS`

### Purpose

Describes how **operands** are organized into logical groups (segments). Essential for operations with **variadic operands** (operands that can accept 0 or more values).

### Syntax

```python
_ODS_OPERAND_SEGMENTS = [seg1_size, seg2_size, ..., segN_size]
```

A list of integers where each element corresponds to one operand segment in the operation's signature.

### Value Encoding

| Value | Type | Meaning | Python Signature |
|-------|------|---------|------------------|
| **`1`** | Required single | Exactly 1 operand must be provided | `operand_name` (required positional) |
| **`0`** | Optional single | 0 or 1 operand may be provided | `operand_name=None` (keyword-only) |
| **`-1`** | Variadic | Any number of operands (0+), length determined at runtime | `operand_name` (accepts list) |
| **`N > 1`** | Fixed group | Exactly N operands required as a group | Rare, usually split into N separate parameters |

### Valid Values

- **`0`**: Optional operand (may be absent)
- **`1`**: Required single operand (must be present, exactly one value)
- **`-1`**: Variadic operand (0 or more values, length determined at runtime)
- **Any positive integer `N`**: Required N operands (fixed-size group)

### Example Breakdown

**Operation**: `nvgpu.TmaAsyncLoadOp`

```python
_ODS_OPERAND_SEGMENTS = [1, 1, 1, -1, 1, 0, 0]
```

Maps to Python signature:
```python
def __init__(self,
             dst,                   # Segment 0: [1] - required single
             barriers,              # Segment 1: [1] - required single
             tensorMapDescriptor,   # Segment 2: [1] - required single
             coordinates,           # Segment 3: [-1] - VARIADIC!
             mbarId,                # Segment 4: [1] - required single
             *,                     # Keyword-only marker
             multicastMask=None,    # Segment 5: [0] - optional
             predicate=None,        # Segment 6: [0] - optional
             loc=None, ip=None):
```

**Key Points**:
- Segments 0-4 are **required** (all have value `1` or `-1`)
- Segment 3 (`coordinates`) is **variadic** (`-1`)
- Segments 5-6 are **optional** (`0`)
- Optional segments appear after `*` in Python (keyword-only)

### How It's Used: Operand Building

**In generated `__init__`**:
```python
def __init__(self, dst, barriers, tensorMapDescriptor, coordinates, mbarId, *,
             multicastMask=None, predicate=None, loc=None, ip=None):
    operands = []

    # Build flat operands list according to segments
    operands.append(dst)                                      # Seg 0: size 1
    operands.append(barriers)                                 # Seg 1: size 1
    operands.append(tensorMapDescriptor)                      # Seg 2: size 1
    operands.append(_get_op_results_or_values(coordinates))  # Seg 3: size -1 (variadic!)
    operands.append(mbarId)                                   # Seg 4: size 1

    # Optional operands
    if multicastMask is not None:                             # Seg 5: size 0 or 1
        operands.append(multicastMask)
    if predicate is not None:                                 # Seg 6: size 0 or 1
        operands.append(predicate)

    # Pass metadata to base class
    super().__init__(
        self.OPERATION_NAME,
        self._ODS_REGIONS,
        self._ODS_OPERAND_SEGMENTS,  # ← Tells runtime about segments
        self._ODS_RESULT_SEGMENTS,
        operands=operands,
        # ...
    )
```

### How It's Used: Property Accessors

Generated property accessors use segment metadata to slice the flat operands list:

```python
@builtins.property
def coordinates(self):
    """Access the variadic coordinates operand segment"""
    operand_range = _ods_segmented_accessor(
        self.operation.operands,                        # All operands (flat list)
        self.operation.attributes["operandSegmentSizes"],  # Runtime sizes
        3  # Segment index (coordinates is segment 3)
    )
    return operand_range
```

**Helper function** ([mlir/dialects/_ods_common.py:38-50](../../mlir/dialects/_ods_common.py#L38-L50)):

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
    start = sum(segments[i] for i in range(idx))  # Sum all preceding segments
    end = start + segments[idx]                    # Add current segment size
    return elements[start:end]
```

### Runtime Attribute: `operandSegmentSizes`

When an operation has variable-sized operand segments, MLIR stores the actual sizes at runtime:

**Example**:
```python
op = nvgpu.TmaAsyncLoadOp(
    dst=memref_dst,
    barriers=barrier,
    tensorMapDescriptor=tma_desc,
    coordinates=[coord_x, coord_y, coord_z],  # 3 coordinates
    mbarId=mbar,
    multicastMask=mask  # Provided (not None)
    # predicate omitted (None)
)

# Runtime attribute created automatically:
# op.attributes["operandSegmentSizes"] = [1, 1, 1, 3, 1, 1, 0]
#                                                 ^        ^  ^
#                                                 |        |  |
#                                     3 coordinates        |  0 predicate
#                                                  1 multicastMask
```

This allows property accessors to work correctly even though the flat operands list varies in length.

---

## `_ODS_REGIONS`

### Purpose

Specifies how many **regions** the operation contains and whether it supports variadic regions.

**Regions** are nested IR structures within an operation (like function bodies, loop bodies, conditional branches).

### Syntax

```python
_ODS_REGIONS = (min_regions, no_variadic_regions)
```

A tuple with exactly **2 elements**:
1. **int**: Minimum number of regions required
2. **bool**: Whether the operation has NO variadic regions (i.e., fixed count)

### Value Encoding

| Position | Type | Meaning |
|----------|------|---------|
| **0** (first) | `int` | **Minimum regions**: Operation requires at least this many regions |
| **1** (second) | `bool` | **No variadic regions**: `True` = fixed count, `False` = can have more than minimum |

### Valid Values

**First element (min_regions)**:
- Any non-negative integer: `0`, `1`, `2`, `3`, ...
- Specifies the **minimum** number of regions required

**Second element (no_variadic_regions)**:
- `True`: Operation has a **fixed** number of regions (max = min)
- `False`: Operation can have **more** than min regions (variadic)

### Common Patterns

| `_ODS_REGIONS` | Interpretation | Example Operations |
|----------------|----------------|-------------------|
| `(0, True)` | **Exactly 0 regions** (operation contains no nested IR) | Most leaf operations (`arith.addi`, `nvgpu.tma.async.load`) |
| `(1, True)` | **Exactly 1 region** required | `scf.while` (loop body), `func.func` (function body) |
| `(2, True)` | **Exactly 2 regions** required | `scf.if` (then + else), `scf.index_switch` (cases) |
| `(0, False)` | **0 or more regions** (variadic) | Rare, used for custom dialects with variable structure |
| `(1, False)` | **1 or more regions** | Operations with at least one region but can have more |
| `(N, True)` | **Exactly N regions** | Fixed structure with N nested regions |

### Example

**Most NVGPU operations**:
```python
_ODS_REGIONS = (0, True)
```
- Minimum: 0 regions
- No variadic regions: True
- **Result**: Operation has exactly 0 regions (doesn't contain nested IR)

**Control flow operations**:
```python
# scf.for operation
_ODS_REGIONS = (1, True)  # Exactly 1 region (the loop body)

# scf.if operation
_ODS_REGIONS = (2, True)  # Exactly 2 regions (then branch + else branch)
```

### How It's Used: Validation

**From** [mlir/lib/Bindings/Python/IRCore.cpp:1840-1860](../../mlir/lib/Bindings/Python/IRCore.cpp#L1840-L1860):

```cpp
// Validate/determine region count.
int opMinRegionCount = std::get<0>(opRegionSpec);       // First element
bool opHasNoVariadicRegions = std::get<1>(opRegionSpec); // Second element

if (!regions) {
    regions = opMinRegionCount;  // Default to minimum
}

if (*regions < opMinRegionCount) {
    throw nb::value_error(
        "Operation requires a minimum of N regions but was built with regions=M");
}

if (opHasNoVariadicRegions && *regions > opMinRegionCount) {
    throw nb::value_error(
        "Operation requires a maximum of N regions but was built with regions=M");
}
```

**Validation logic**:
1. If regions not specified, default to `min_regions`
2. Check that actual regions ≥ `min_regions`
3. If `no_variadic_regions=True`, check that actual regions ≤ `min_regions`

---

## How They Work Together

### Full Flow: Construction to Access

#### 1. Operation Construction

```python
# User code
op = nvgpu.TmaAsyncLoadOp(
    dst=memref_dst,
    barriers=barrier,
    tensorMapDescriptor=tma_desc,
    coordinates=[coord_x, coord_y, coord_z],  # 3 values
    mbarId=mbar,
    multicastMask=mask
    # predicate=None (omitted)
)
```

#### 2. Generated `__init__` Processing

```python
def __init__(self, dst, barriers, tensorMapDescriptor, coordinates, mbarId, *,
             multicastMask=None, predicate=None, loc=None, ip=None):
    operands = []
    attributes = {}
    regions = None  # No regions for this op (_ODS_REGIONS = (0, True))

    # Build operands according to _ODS_OPERAND_SEGMENTS
    operands.append(dst)                                      # [1]
    operands.append(barriers)                                 # [1]
    operands.append(tensorMapDescriptor)                      # [1]
    operands.append(_get_op_results_or_values(coordinates))  # [-1] → [coord_x, coord_y, coord_z]
    operands.append(mbarId)                                   # [1]
    if multicastMask is not None:                             # [0] → mask provided
        operands.append(multicastMask)
    if predicate is not None:                                 # [0] → None, skip
        operands.append(predicate)

    # Flat operands list:
    # [memref_dst, barrier, tma_desc, coord_x, coord_y, coord_z, mbar, mask]
    # Total: 8 operands

    # Call base class with metadata
    super().__init__(
        self.OPERATION_NAME,              # "nvgpu.tma.async.load"
        self._ODS_REGIONS,                # (0, True)
        self._ODS_OPERAND_SEGMENTS,       # [1,1,1,-1,1,0,0]
        self._ODS_RESULT_SEGMENTS,        # Result segments (similar concept)
        attributes=attributes,
        results=results,
        operands=operands,                # Flat list of 8 operands
        successors=_ods_successors,
        regions=regions,                  # None (0 regions)
        loc=loc,
        ip=ip
    )
```

#### 3. Base Class Processing (C++)

**From** [mlir/lib/Bindings/Python/IRCore.cpp:1820-1899](../../mlir/lib/Bindings/Python/IRCore.cpp#L1820-L1899):

```cpp
nb::object PyOpView::buildGeneric(
    std::string_view name,
    std::tuple<int, bool> opRegionSpec,     // _ODS_REGIONS
    nb::object operandSegmentSpecObj,        // _ODS_OPERAND_SEGMENTS
    nb::object resultSegmentSpecObj,
    // ...
) {
    // Validate regions
    int opMinRegionCount = std::get<0>(opRegionSpec);  // 0
    bool opHasNoVariadicRegions = std::get<1>(opRegionSpec);  // True
    // ... validation logic ...

    // Process operand segments
    auto operandSegmentSpec = nb::cast<std::vector<int>>(operandSegmentSpecObj);
    // operandSegmentSpec = [1, 1, 1, -1, 1, 0, 0]

    std::vector<int32_t> operandSegmentLengths;
    for (segment_size, actual_operand : zip(operandSegmentSpec, operandList)) {
        if (segment_size == -1) {
            // Variadic: determine length from actual operand list
            operandSegmentLengths.push_back(len(actual_operand));
        } else if (segment_size == 0) {
            // Optional: 0 if None, 1 if provided
            operandSegmentLengths.push_back(actual_operand ? 1 : 0);
        } else {
            // Fixed size: use as-is
            operandSegmentLengths.push_back(segment_size);
        }
    }
    // operandSegmentLengths = [1, 1, 1, 3, 1, 1, 0]
    //                                    ^        ^
    //                                    |        |
    //                          3 coordinates      no predicate

    // Store as runtime attribute
    attributes["operandSegmentSizes"] = DenseI32ArrayAttr::get(operandSegmentLengths);

    // Create the operation
    Operation *op = Operation::create(state);
    return op;
}
```

#### 4. Property Access

```python
# Later, access specific operand segments
coords = op.coordinates  # Access segment 3 (variadic)

# Implementation:
@builtins.property
def coordinates(self):
    # Get runtime segment sizes
    segments = self.operation.attributes["operandSegmentSizes"]  # [1,1,1,3,1,1,0]

    # Calculate slice for segment 3
    start = sum(segments[0:3])  # 1+1+1 = 3
    end = start + segments[3]    # 3 + 3 = 6

    # Return slice of flat operands list
    return self.operation.operands[3:6]  # [coord_x, coord_y, coord_z]
```

---

## Real-World Examples

### Example 1: Simple Operation (No Variadic)

**Operation**: `arith.ConstantOp`

```python
class ConstantOp(_ods_ir.OpView):
    OPERATION_NAME = "arith.constant"

    _ODS_OPERAND_SEGMENTS = None  # No segments needed (no operands)
    _ODS_REGIONS = (0, True)       # No regions

    def __init__(self, value, *, results=None, loc=None, ip=None):
        # No operands, just an attribute
        attributes = {"value": value}
        regions = None
        operands = []

        super().__init__(
            self.OPERATION_NAME,
            self._ODS_REGIONS,
            self._ODS_OPERAND_SEGMENTS,
            self._ODS_RESULT_SEGMENTS,
            attributes=attributes,
            results=results,
            operands=operands,
            regions=regions,
            loc=loc,
            ip=ip
        )
```

**Key points**:
- No operands → `_ODS_OPERAND_SEGMENTS = None`
- No regions → `_ODS_REGIONS = (0, True)`

### Example 2: Variadic Operands

**Operation**: `nvgpu.DeviceAsyncCopyOp`

```python
class DeviceAsyncCopyOp(_ods_ir.OpView):
    OPERATION_NAME = "nvgpu.device_async_copy"

    _ODS_OPERAND_SEGMENTS = [1, -1, 1, -1, 0]
    #                           ^      ^    ^
    #                           |      |    |
    #                      dst_indices |  optional
    #                                  |
    #                           src_indices (variadic)
    _ODS_REGIONS = (0, True)

    def __init__(self, dst, dstIndices, src, srcIndices, dstElements, *,
                 srcElements=None, bypassL1=None, results=None, loc=None, ip=None):
        # dstIndices and srcIndices are VARIADIC
        # srcElements is OPTIONAL
```

**Usage**:
```python
op = nvgpu.DeviceAsyncCopyOp(
    dst=memref_dst,
    dstIndices=[idx0, idx1, idx2],  # Variadic: 3 indices
    src=memref_src,
    srcIndices=[src_idx0, src_idx1], # Variadic: 2 indices
    dstElements=size,
    srcElements=src_size  # Optional: provided
)

# Runtime segment sizes: [1, 3, 1, 2, 1]
#                            ^     ^  ^
#                            |     |  |
#                      3 dst_indices  2 src_indices
#                                     1 srcElements (provided)
```

### Example 3: Operations with Regions

**Operation**: `scf.IfOp`

```python
class IfOp(_ods_ir.OpView):
    OPERATION_NAME = "scf.if"

    _ODS_OPERAND_SEGMENTS = [1]  # condition operand
    _ODS_REGIONS = (2, True)      # Exactly 2 regions (then + else)

    def __init__(self, condition, *, results=None, loc=None, ip=None):
        operands = [condition]
        regions = None  # Created separately

        super().__init__(
            self.OPERATION_NAME,
            self._ODS_REGIONS,
            self._ODS_OPERAND_SEGMENTS,
            self._ODS_RESULT_SEGMENTS,
            attributes=attributes,
            results=results,
            operands=operands,
            regions=2,  # Must pass 2 regions!
            loc=loc,
            ip=ip
        )
```

**Usage**:
```python
with ir.InsertionPoint(block):
    if_op = scf.IfOp(condition)

    # Access the two regions
    then_block = if_op.then_block
    else_block = if_op.else_block

    # Build IR in each region
    with ir.InsertionPoint(then_block):
        # then branch operations
        scf.YieldOp([result1])

    with ir.InsertionPoint(else_block):
        # else branch operations
        scf.YieldOp([result2])
```

### Example 4: Multiple Variadic Segments

**Hypothetical operation**:

```python
class MultiVariadicOp(_ods_ir.OpView):
    _ODS_OPERAND_SEGMENTS = [1, -1, 1, -1, 0]
    #                           ^      ^    ^
    #                           |      |    |
    #                      inputs1  inputs2  optional

    def __init__(self, base, inputs1, middle, inputs2, *, optional=None):
        # Two variadic segments!
```

**Usage**:
```python
op = MultiVariadicOp(
    base=base_value,
    inputs1=[a, b, c],      # 3 values
    middle=middle_value,
    inputs2=[x, y],         # 2 values
    optional=opt            # 1 value
)

# Runtime: operandSegmentSizes = [1, 3, 1, 2, 1]
# Flat operands: [base_value, a, b, c, middle_value, x, y, opt]

# Property accessors:
op.inputs1  # Returns [a, b, c] (operands[1:4])
op.inputs2  # Returns [x, y] (operands[5:7])
```

---

## TableGen Origins

These metadata values are generated from **TableGen** operation definitions.

### TableGen Definition

**File**: `mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td`

```tablegen
def NVGPU_TmaAsyncLoadOp : NVGPU_Op<"tma.async.load"> {
  let summary = "TMA async load operation";

  let arguments = (ins
    // Required single operands (generates: 1)
    NVGPU_DeviceAsyncToken:$dst,
    NVGPU_MBarrierGroup:$barriers,
    NVGPU_TensorMapDescriptor:$tensorMapDescriptor,

    // Variadic operand (generates: -1)
    Variadic<Index>:$coordinates,

    // Required single operand (generates: 1)
    NVGPU_MBarrier:$mbarId,

    // Optional operands (generates: 0)
    Optional<I32>:$multicastMask,
    Optional<I1>:$predicate
  );

  let results = (outs);  // No results

  let regions = (region);  // No regions → (0, True)
}
```

### Code Generation

**`mlir-tblgen -gen-python-op-bindings`** analyzes the TableGen definition:

1. **Count operands and their types**:
   - `dst` → single required → `1`
   - `barriers` → single required → `1`
   - `tensorMapDescriptor` → single required → `1`
   - `coordinates` → `Variadic<>` → `-1`
   - `mbarId` → single required → `1`
   - `multicastMask` → `Optional<>` → `0`
   - `predicate` → `Optional<>` → `0`

   Result: `[1, 1, 1, -1, 1, 0, 0]`

2. **Count regions**:
   - `let regions = (region);` → no regions specified
   - Default: min=0, fixed (no variadic)

   Result: `(0, True)`

3. **Generate Python class** with metadata

### TableGen Operand Types

| TableGen Type | Python Metadata | Example |
|---------------|-----------------|---------|
| `Type:$name` | `1` | `I32:$index` |
| `Optional<Type>:$name` | `0` | `Optional<I32>:$mask` |
| `Variadic<Type>:$name` | `-1` | `Variadic<Index>:$coords` |
| `VariadicOfVariadic<Type>:$name` | Special handling | Rare |

---

## Debugging and Introspection

### Inspect Segment Metadata

```python
import mlir.dialects.nvgpu as nvgpu

# Check class metadata
print(nvgpu.TmaAsyncLoadOp._ODS_OPERAND_SEGMENTS)  # [1, 1, 1, -1, 1, 0, 0]
print(nvgpu.TmaAsyncLoadOp._ODS_REGIONS)           # (0, True)

# Check runtime segments after creating operation
op = nvgpu.TmaAsyncLoadOp(...)
if "operandSegmentSizes" in op.attributes:
    sizes = op.attributes["operandSegmentSizes"]
    print(f"Runtime segment sizes: {sizes}")
    # DenseI32ArrayAttr: [1, 1, 1, 3, 1, 1, 0]
```

### Verify Operand Slicing

```python
def debug_operand_segments(op):
    """Debug helper to show operand segmentation"""
    if "operandSegmentSizes" not in op.attributes:
        print("No operand segments (simple operation)")
        return

    segments = op.attributes["operandSegmentSizes"]
    operands = op.operands

    print(f"Total operands: {len(operands)}")
    print(f"Segment sizes: {segments}")

    offset = 0
    for i, size in enumerate(segments):
        segment_operands = operands[offset:offset+size]
        print(f"  Segment {i}: size={size}, operands={segment_operands}")
        offset += size
```

### Common Errors and Solutions

#### Error 1: Wrong Number of Arguments

```python
# Error: Missing required operand
op = nvgpu.TmaAsyncLoadOp(
    dst=dst,
    barriers=barriers,
    # Missing: tensorMapDescriptor!
    coordinates=[x, y],
    mbarId=mbar
)
# TypeError: __init__() missing 1 required positional argument: 'tensorMapDescriptor'
```

**Solution**: Provide all required operands (segments with value `1` or `-1`).

#### Error 2: Passing Wrong Type to Variadic

```python
# Error: Passing single value instead of list
op = nvgpu.TmaAsyncLoadOp(
    dst=dst,
    barriers=barriers,
    tensorMapDescriptor=desc,
    coordinates=coord_x,  # Should be list: [coord_x]
    mbarId=mbar
)
# Runtime error in _get_op_results_or_values
```

**Solution**: Variadic operands (segment `-1`) expect a list, even for single values:
```python
coordinates=[coord_x]  # Correct
```

#### Error 3: Region Count Mismatch

```python
# Error: Operation requires 0 regions
op = nvgpu.TmaAsyncLoadOp(
    dst=dst,
    # ... other operands ...
    regions=1  # Error! _ODS_REGIONS = (0, True)
)
# ValueError: Operation "nvgpu.tma.async.load" requires a maximum of 0 regions
```

**Solution**: Don't pass `regions` parameter for operations with `_ODS_REGIONS = (0, True)`.

---

## Summary

### `_ODS_OPERAND_SEGMENTS` Quick Reference

```python
_ODS_OPERAND_SEGMENTS = [1, 1, 1, -1, 1, 0, 0]
                         │  │  │   │  │  │  └─ Seg 6: Optional (0/1)
                         │  │  │   │  │  └──── Seg 5: Optional (0/1)
                         │  │  │   │  └─────── Seg 4: Required single
                         │  │  │   └────────── Seg 3: Variadic (0+)
                         │  │  └────────────── Seg 2: Required single
                         │  └───────────────── Seg 1: Required single
                         └──────────────────── Seg 0: Required single
```

| Value | Meaning | Python Arg Type |
|-------|---------|-----------------|
| `1` | Required single operand | `arg` (positional) |
| `0` | Optional single operand | `arg=None` (keyword) |
| `-1` | Variadic operands | `arg` (list/tuple) |

### `_ODS_REGIONS` Quick Reference

```python
_ODS_REGIONS = (min_count, no_variadic)
                    │           │
                    │           └─ True: fixed count, False: can have more
                    └───────────── Minimum regions required
```

| Tuple | Meaning |
|-------|---------|
| `(0, True)` | No regions (leaf operation) |
| `(1, True)` | Exactly 1 region |
| `(2, True)` | Exactly 2 regions |
| `(N, False)` | N or more regions |

### Key Takeaways

1. **Generated automatically** from TableGen definitions
2. **Enable dynamic operation construction** with variadic operands/regions
3. **Stored as runtime attributes** when needed (`operandSegmentSizes`)
4. **Used by property accessors** to slice flat operand lists
5. **Validated at construction time** to catch errors early

---

## Further Reading

- [ODS (Operation Definition Specification)](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [Python Bindings Architecture](../../mlir/lib/Bindings/Python/)
- [TableGen Backend: Python Bindings](../../mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp)
- [Context Management: InsertionPoint](../context_management/insertion-point-deep-dive.md)
