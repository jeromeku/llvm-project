# Layer 4: NVDSL Python DSL Implementation

## Introduction

This document provides a **complete architectural breakdown** of the NVDSL DSL implementation in [nvdsl.py](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py).

NVDSL is a **pure Python library** that wraps MLIR Python bindings to provide a high-level, Pythonic API for GPU programming.

---

## File Structure

**Location**: [mlir/test/Examples/NVGPU/tools/nvdsl.py](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py)
**Size**: 460 lines
**Language**: Pure Python (no C/C++)

### Module Organization

```
nvdsl.py
├── Imports & Constants (lines 1-11)
├── Utility Functions (lines 14-54)
│   ├── const() - Constant creation
│   ├── get_type_size() - Type size calculation
│   └── get_mlir_func_obj_ty() - Type conversion for execution
├── DSL Classes (lines 57-276)
│   ├── Mbarriers - Memory barriers
│   ├── TMA - Tensor Memory Accelerator
│   ├── Warpgroup - Warpgroup abstraction
│   └── WGMMAMatrix - Warpgroup matrix operations
├── Helper Functions (lines 259-304)
│   ├── get_dynamic_shared_memory()
│   └── get_mlir_ty()
└── NVDSL Class (lines 306-461)
    ├── @mlir_gpu_launch decorator
    └── @mlir_func decorator
```

---

## Section 1: Imports and Constants

**Lines 1-11**: [nvdsl.py:1-11](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L1-L11)

```python
from enum import Enum
import functools, sys, ctypes, os, errno
import numpy as np
from functools import partialmethod
from mlir import ir
from mlir.dialects import arith, func, gpu, memref, nvgpu, scf, nvvm
from mlir.extras import types as T
from mlir import runtime as rt
from tools import nvgpucompiler

MLIR_DYNAMIC = -9223372036854775808
```

| Import | Purpose |
|--------|---------|
| `mlir.ir` | Core IR: Context, Module, Location, InsertionPoint, Value, Type, Operation |
| `mlir.dialects.*` | GPU/NVGPU dialect operations |
| `mlir.runtime` | Memref descriptor utilities for NumPy interop |
| `nvgpucompiler` | Compilation pipeline wrapper |
| `ctypes` | FFI for calling compiled functions |

**MLIR_DYNAMIC constant**: Represents dynamic dimension in MLIR (`-9223372036854775808 = -(2^63)`)

---

## Section 2: Utility Functions

### 2.1 const() - Universal Constant Creator

**Lines 14-20**: [nvdsl.py:14-20](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L14-L20)

```python
def const(value: int, ty=None):
    ty = T.index() if ty is None else ty
    if isinstance(value, ir.Value) and (
        value.type.isinstance(value.type) or T.bool().isinstance(value.type)
    ):
        return value
    return arith.constant(ty, value)
```

**Purpose**: Creates MLIR constant operations, with smart pass-through if already an `ir.Value`.

**Usage Examples**:
```python
const(128)              # → arith.constant 128 : index
const(0, T.i32())       # → arith.constant 0 : i32
const(True, T.bool())   # → arith.constant true
```

**Call Path**:
```
const(128)
    ↓
arith.constant(T.index(), 128)
    ↓
Python binding: arith.ConstantOp [auto-generated from TableGen]
    ↓
_mlirDialectsArith.ConstantOp(...)
    ↓
C++: arith::ConstantOp::build(builder, result, IntegerAttr::get(type, value))
    ↓
Returns ir.Value (Python wrapper around MlirValue handle)
```

### 2.2 get_type_size() - Calculate Type Sizes

**Lines 23-33**: [nvdsl.py:23-33](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L23-L33)

```python
def get_type_size(ty):
    if ir.MemRefType.isinstance(ty):
        size = get_type_size(ty.element_type)
        for sz in ty.shape:
            size *= sz
        return size
    if ir.FloatType.isinstance(ty):
        return ir.FloatType(ty).width // 8
    if ir.IntegerType.isinstance(ty):
        return ir.IntegerType(ty).width // 8
    raise NotImplementedError(ty)
```

**Purpose**: Recursively calculates size in **bytes** for MLIR types.

**Examples**:
```python
get_type_size(T.f16())                    # → 2 bytes
get_type_size(T.f32())                    # → 4 bytes
get_type_size(memref<128x64xf16>)         # → 128 * 64 * 2 = 16384 bytes
get_type_size(memref<64x64xf16>)          # → 64 * 64 * 2 = 8192 bytes
```

**Type Hierarchy**:
```
ir.Type (base)
├── ir.MemRefType
│   ├── .shape → [int, ...]
│   └── .element_type → ir.Type
├── ir.FloatType
│   └── .width → int (bits)
└── ir.IntegerType
    └── .width → int (bits)
```

### 2.3 get_mlir_func_obj_ty() - Prepare Execution Arguments

**Lines 36-54**: [nvdsl.py:36-54](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L36-L54)

```python
def get_mlir_func_obj_ty(inputArgs):
    args = []
    c_int_p = ctypes.c_int * 1
    c_float_p = ctypes.c_float * 1
    c_bool_p = ctypes.c_bool * 1
    for arg in inputArgs:
        if isinstance(arg, bool):
            args.append(c_bool_p(arg))
        elif isinstance(arg, int):
            args.append(c_int_p(arg))
        elif isinstance(arg, float):
            args.append(c_float_p(arg))
        elif isinstance(arg, np.ndarray):
            args.append(
                ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(arg)))
            )
        else:
            raise NotImplementedError(arg)
    return args
```

**Purpose**: Converts Python objects → ctypes pointers for JIT engine invocation.

**NumPy Array Handling**:
```python
np.ndarray → rt.get_ranked_memref_descriptor(arg) → MemRefDescriptor
    ↓
MemRefDescriptor struct (C struct):
    void* allocated;     // Base pointer
    void* aligned;       // Aligned pointer
    intptr_t offset;     // Offset
    intptr_t sizes[N];   // Dimension sizes
    intptr_t strides[N]; // Strides
    ↓
ctypes.pointer(ctypes.pointer(descriptor))  // Double pointer!
```

**Why double pointer?**: MLIR calling convention passes memref by pointer-to-pointer for ABI compatibility.

---

## Section 3: DSL Classes

### 3.1 Mbarriers Class

**Lines 57-100**: [nvdsl.py:57-100](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L57-L100)

```python
class Mbarriers:
    def __init__(self, number_of_barriers=1):
        self.mbar_ty = ir.Type.parse(
            "!nvgpu.mbarrier.group<memorySpace=#gpu.address_space<workgroup>, num_barriers = "
            + str(number_of_barriers)
            + ">"
        )
        self.mbar_group_op = nvgpu.mbarrier_create(self.mbar_ty)
        self.number_of_barriers = number_of_barriers

    def __getitem__(self, key):
        self.id_op = const(key)
        return self

    def init(self, count: int, predicate=None):
        count_op = const(count)
        if predicate is None:
            nvgpu.mbarrier_init(self.mbar_group_op, count_op, self.id_op)
        else:
            nvgpu.mbarrier_init(
                self.mbar_group_op, count_op, self.id_op, predicate=predicate
            )

    def arrive(self, txcount: int = 0, predicate=None):
        if txcount != 0:
            txcount_op = const(txcount)
            nvgpu.mbarrier_arrive_expect_tx(
                self.mbar_group_op, txcount_op, self.id_op, predicate=predicate
            )
        else:
            nvgpu.mbarrier_arrive(
                ir.Type.parse("!nvgpu.mbarrier.token"), self.mbar_group_op, self.id_op
            )

    def try_wait(self, phase: bool = False, ticks: int = 10000000):
        ticks_op = const(ticks)
        phase_op = const(phase, T.bool())
        nvgpu.MBarrierTryWaitParityOp(
            self.mbar_group_op,
            phase_op,
            ticks_op,
            mbarId=self.id_op,
        )
```

**Purpose**: High-level Python API for NVIDIA mbarrier (memory barrier) operations.

#### Architecture

**Mbarrier Group Type**:
```
!nvgpu.mbarrier.group<
  memorySpace=#gpu.address_space<workgroup>,
  num_barriers = 7
>
```

**Usage Pattern**:
```python
# Create group of 7 barriers
mbar_group = Mbarriers(number_of_barriers=7)

# Access specific barrier
mbar_group[0].init(1, predicate=isThread0)
mbar_group[0].arrive(txcount=16384)
mbar_group[0].try_wait(phase=False)
```

#### Method Breakdown

**`__init__(number_of_barriers=1)`**:
```
1. Parse type string:
   "!nvgpu.mbarrier.group<...>"
       ↓
   ir.Type.parse(type_string)
       ↓
   C API: mlirTypeParseGet(ctx, type_string)
       ↓
   C++: parseType(type_string, context)

2. Create barrier group:
   nvgpu.mbarrier_create(mbar_ty)
       ↓
   Python binding: nvgpu.MBarrierCreateOp
       ↓
   C++: nvgpu::MBarrierCreateOp::build(...)
       ↓
   MLIR: %mbar_group = nvgpu.mbarrier.create : !nvgpu.mbarrier.group<...>
```

**`__getitem__(key)`** - Indexing operator:
```python
mbar_group[0]  # Calls __getitem__(0)
    ↓
self.id_op = const(0)  # Creates arith.constant 0
return self
```

**Design**: Returns `self` to enable method chaining:
```python
mbar_group[0].arrive(...)  # Works!
```

**`init(count, predicate)`**:
```
nvgpu.mbarrier_init(mbar_group_op, count_op, id_op, predicate=predicate)
    ↓
Python binding [auto-generated]
    ↓
C++: nvgpu::MBarrierInitOp::build(...)
    ↓
MLIR: nvgpu.mbarrier.init %mbar_group[%c0], %c1 {predicate = %isThread0}
```

**`arrive(txcount, predicate)`**:

Two modes:
1. **With transaction count** (TMA operations):
   ```mlir
   nvgpu.mbarrier.arrive.expect_tx %mbar_group[%c0], %c16384 {predicate = %p}
   ```

2. **Without transaction count** (regular arrive):
   ```mlir
   %token = nvgpu.mbarrier.arrive %mbar_group[%c0] : !nvgpu.mbarrier.token
   ```

**`try_wait(phase, ticks)`**:
```
nvgpu.MBarrierTryWaitParityOp(mbar_group_op, phase_op, ticks_op, mbarId=id_op)
    ↓
MLIR: nvgpu.mbarrier.try_wait.parity %mbar_group[%c0], %phase, %c10000000
```

**Phase parity**: Alternates between `false` and `true` for each stage in pipelined loops.

---

### 3.2 TMA Class

**Lines 102-163**: [nvdsl.py:102-163](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L102-L163)

```python
class TMA:
    """A class that builds a TMA descriptor."""

    def __init__(
        self,
        tma_box_shape,
        memref_ty,
        swizzle=nvgpu.TensorMapSwizzleKind.SWIZZLE_NONE,
        l2promo=nvgpu.TensorMapL2PromoKind.L2PROMO_NONE,
        oob=nvgpu.TensorMapOOBKind.OOB_ZERO,
        interleave=nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE,
    ):
        self.swizzle = swizzle
        self.l2promo = l2promo
        self.oob = oob
        self.interleave = interleave
        self.tma_box_shape = tma_box_shape
        self.memref_ty = memref_ty
        self.tma_memref = ir.MemRefType.get(tma_box_shape, memref_ty.element_type)

    @property
    def tensormap_descriptor_ty(self):
        """Returns a tensormap descriptor type."""
        tensorMemrefType = ir.MemRefType.get(
            self.tma_box_shape,
            self.memref_ty.element_type,
            memory_space=ir.Attribute.parse("3"),  # Shared memory
        )
        return nvgpu.TensorMapDescriptorType.get(
            tensorMemrefType,
            self.swizzle,
            self.l2promo,
            self.oob,
            self.interleave,
        )

    def create_descriptor(self, device_ptr):
        tma_descriptor_ty = self.tensormap_descriptor_ty
        device_unranked_memref = memref.CastOp(
            ir.UnrankedMemRefType.get(
                self.memref_ty.element_type, self.memref_ty.memory_space
            ),
            device_ptr,
        )
        self.tma_descriptor = nvgpu.TmaCreateDescriptorOp(
            tma_descriptor_ty, device_unranked_memref, map(const, self.tma_box_shape)
        )
        return self.tma_descriptor.result

    def prefetch(self, predicate=None):
        nvgpu.tma_prefetch_descriptor(self.tma_descriptor, predicate=predicate)

    def load(self, dest, mbarrier: Mbarriers, coords=[0], predicate=None):
        nvgpu.TmaAsyncLoadOp(
            dest,
            mbarrier.mbar_group_op,
            self.tma_descriptor,
            coordinates=map(const, coords),
            mbarId=mbarrier.id_op,
            predicate=predicate,
        )
```

**Purpose**: Manages NVIDIA TMA (Tensor Memory Accelerator) descriptors for efficient memory transfers.

#### TMA Architecture

**What is TMA?**
- Hardware unit on Hopper GPUs (SM90+)
- Accelerates multidimensional memory transfers
- Global memory ↔ Shared memory
- Coordinates specified as tile indices

**TMA Descriptor**: A hardware structure describing:
- Source/destination memory layout
- Transfer dimensions
- Swizzling pattern
- L2 cache promotion hints
- Out-of-bounds handling

#### Usage Pattern

```python
# 1. Create TMA object (metadata only)
a_tma = TMA([128, 64], memref_ty, swizzle=SWIZZLE_128B)

# 2. Create hardware descriptor
a_tma.create_descriptor(device_ptr)

# 3. Prefetch descriptor (optional optimization)
a_tma.prefetch(predicate=isThread0)

# 4. Load data
a_tma.load(shared_mem_view, mbar_group[0], coords=[0, 0], predicate=isThread0)
```

#### Method Details

**`__init__(...)`**:

```python
TMA([128, 64], memref<512x1024xf16>, swizzle=SWIZZLE_128B)
```

Creates **Python object only** (no MLIR operations yet):
- `tma_box_shape = [128, 64]` - Transfer tile size
- `memref_ty = memref<512x1024xf16>` - Full tensor type
- `tma_memref = memref<128x64xf16>` - Tile type

**`tensormap_descriptor_ty` property**:

Constructs MLIR type:
```
!nvgpu.tensormap.descriptor<
  tensor = memref<128x64xf16, 3>,  # 3 = shared memory
  swizzle = swizzle_128b,
  l2promo = l2promo_none,
  oob = oob_zero,
  interleave = interleave_none
>
```

**Call Path**:
```
nvgpu.TensorMapDescriptorType.get(...)
    ↓
Python binding: _mlirDialectsNVGPU.TensorMapDescriptorType.get(...)
    ↓
C API: mlirNVGPUTensorMapDescriptorTypeGet(ctx, tensor_type, swizzle, ...)
    ↓
C++: nvgpu::TensorMapDescriptorType::get(context, ...)
```

**`create_descriptor(device_ptr)`**:

**Step 1**: Cast to unranked memref
```python
device_unranked = memref.CastOp(
    ir.UnrankedMemRefType.get(f16, memory_space),
    device_ptr
)
```
```mlir
%unranked = memref.cast %device_ptr : memref<512x1024xf16> to memref<*xf16>
```

**Why?** TMA operates on runtime shapes, needs unranked memref.

**Step 2**: Create TMA descriptor
```python
self.tma_descriptor = nvgpu.TmaCreateDescriptorOp(
    tma_descriptor_ty,
    device_unranked_memref,
    map(const, self.tma_box_shape)  # [const(128), const(64)]
)
```

**Call Path**:
```
nvgpu.TmaCreateDescriptorOp(...)
    ↓
Python binding [auto-generated from TableGen]
    ↓
C++: nvgpu::TmaCreateDescriptorOp::build(...)
    ↓
MLIR IR:
%c128 = arith.constant 128 : index
%c64 = arith.constant 64 : index
%tma_desc = nvgpu.tma.create.descriptor %unranked box[%c128, %c64]
  : memref<*xf16> -> !nvgpu.tensormap.descriptor<...>
```

**Hardware Level**: This creates a `CUtensorMap` structure passed to `cuTensorMapEncodeTiled`.

**`prefetch(predicate)`**:

Hints to prefetch TMA descriptor into cache:
```mlir
nvgpu.tma.prefetch.descriptor %tma_desc {predicate = %isThread0}
```

**`load(dest, mbarrier, coords, predicate)`**:

Initiates asynchronous TMA load:
```python
a_tma.load(
    dest=shared_mem_view,         # Destination in shared memory
    mbarrier=mbar_group[0],       # Barrier to signal on completion
    coords=[tile_x, tile_y],      # Tile coordinates in global memory
    predicate=isThread0           # Only thread 0 executes
)
```

**Generated MLIR**:
```mlir
nvgpu.tma.async.load %dest[%coords...], %mbar_group[%id], %tma_desc
  {predicate = %isThread0}
  : memref<128x64xf16, 3>  # Shared memory (address space 3)
```

**Hardware**: Executes as `cp.async.bulk.tensor` PTX instruction.

---

### 3.3 Warpgroup Class

**Lines 165-191**: [nvdsl.py:168-191](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L168-L191)

```python
WARP_GROUP_SIZE = 128  # Number of threads in a warpgroup

class Warpgroup:
    def __init__(self, primary_thread, register_size):
        assert (primary_thread % WARP_GROUP_SIZE) == 0
        tidx = gpu.thread_id(gpu.Dimension.x)
        self.primary_thread = primary_thread
        self.register_size = register_size
        self.is_wg_primary = (tidx % WARP_GROUP_SIZE) == 0
        self.wg_id = tidx / WARP_GROUP_SIZE
        self.is_me = self.wg_id == (primary_thread // WARP_GROUP_SIZE)

    def __enter__(self):
        if_op = scf.IfOp(self.is_me)
        self.ipoint_op = ir.InsertionPoint(if_op.then_block)
        self.ipoint_op.__enter__()
        if self.register_size < 64:
            nvvm.setmaxregister(self.register_size, nvvm.SetMaxRegisterAction.decrease)
        else:
            nvvm.setmaxregister(self.register_size, nvvm.SetMaxRegisterAction.increase)

    def __exit__(self, exc_type, exc_value, traceback):
        scf.yield_([])
        self.ipoint_op.__exit__(exc_type, exc_value, traceback)
        return True
```

**Purpose**: Context manager for warpgroup-specific code with register allocation control.

**Warpgroup**: 128 threads (4 warps) that can execute specialized instructions (WGMMA).

#### Usage Pattern

```python
# Create two warpgroups in a 256-thread block
wg_producer = Warpgroup(primary_thread=128, register_size=40)
wg_consumer = Warpgroup(primary_thread=0, register_size=232)

# Producer code
with wg_producer:
    # Only threads 128-255 execute this
    tma_load(...)

# Consumer code
with wg_consumer:
    # Only threads 0-127 execute this
    matrix_multiply(...)
```

#### Implementation Details

**`__init__(...)`**:

```python
tidx = gpu.thread_id(gpu.Dimension.x)  # Current thread ID
self.is_wg_primary = (tidx % 128) == 0  # Is thread leader of warpgroup?
self.wg_id = tidx / 128                 # Which warpgroup? (0, 1, 2, ...)
self.is_me = self.wg_id == (primary_thread // 128)  # Is this my warpgroup?
```

**Calculation Example** (256 threads, 2 warpgroups):

| Thread ID | wg_id | WG0 (primary=0) | WG1 (primary=128) |
|-----------|-------|-----------------|-------------------|
| 0         | 0     | ✓ (is_me)       | ✗                 |
| 64        | 0     | ✓               | ✗                 |
| 127       | 0     | ✓               | ✗                 |
| 128       | 1     | ✗               | ✓ (is_me)         |
| 192       | 1     | ✗               | ✓                 |
| 255       | 1     | ✗               | ✓                 |

**`__enter__()` - Context manager entry**:

```python
if_op = scf.IfOp(self.is_me)
self.ipoint_op = ir.InsertionPoint(if_op.then_block)
self.ipoint_op.__enter__()
```

**Generated MLIR**:
```mlir
%tidx = gpu.thread_id x : index
%c128 = arith.constant 128 : index
%wg_id = arith.divui %tidx, %c128 : index
%target_wg = arith.constant 0 : index  // or 1 for WG1
%is_me = arith.cmpi eq, %wg_id, %target_wg : index

scf.if %is_me {
  // Warpgroup-specific code here
  nvvm.setmaxregister.dec 40
  // ...
  scf.yield
}
```

**Register allocation**:
```python
if self.register_size < 64:
    nvvm.setmaxregister(self.register_size, nvvm.SetMaxRegisterAction.decrease)
else:
    nvvm.setmaxregister(self.register_size, nvvm.SetMaxRegisterAction.increase)
```

**Purpose**: Control register allocation per warpgroup:
- **Producer** (40 registers): Uses fewer registers, leaves more for consumer
- **Consumer** (232 registers): Needs many registers for accumulator tiles

**PTX Generated**:
```ptx
.maxnreg 40   // For producer warpgroup
.maxnreg 232  // For consumer warpgroup
```

---

### 3.4 WGMMAMatrix Class

**Lines 193-257**: [nvdsl.py:198-257](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L198-L257)

```python
class WGMMAType(Enum):
    Accumulator = 1
    Descriptor = 2

class WGMMAMatrix:
    def __init__(
        self,
        matrix_type: WGMMAType,
        shape: list = None,
        desc: TMA = None,
        smem=None,
        ty=None,
        acc_op=None,
    ):
        if acc_op is None:
            self.M = shape[0]
            self.N = shape[1]
            self.ty = ty
            self.matrix_type = matrix_type
            self.desc = desc
            self.smem = smem
            if matrix_type is WGMMAType.Accumulator:
                self.acc_op = nvgpu.warpgroup_mma_init_accumulator(self.acc_ty)
        elif acc_op:
            self.acc_op = acc_op
            self.matrix_type = WGMMAType.Accumulator

    @property
    def acc_ty(self):
        parse_str = f"!nvgpu.warpgroup.accumulator<fragmented=vector<{self.M}x{self.N}x{self.ty}>>"
        return ir.Type.parse(parse_str)

    @property
    def wgmma_ty(self):
        parse_str = f"!nvgpu.warpgroup.descriptor<tensor=memref<{self.M}x{self.N}x{self.desc.memref_ty.element_type}, #gpu.address_space<workgroup>>>"
        return ir.Type.parse(parse_str)

    def store_accumulator(self, dest):
        assert self.matrix_type == WGMMAType.Accumulator
        nvgpu.warpgroup_mma_store(self.acc_op, dest)

    def update_smem(self, smem):
        self.smem = smem

    def update_accumulator(self, acc_op):
        self.acc_op = acc_op

    def __matmul__(self, rhs):
        lhs = nvgpu.warpgroup_generate_descriptor(
            self.wgmma_ty, self.smem, self.desc.tma_descriptor
        )
        rhs = nvgpu.warpgroup_generate_descriptor(
            rhs.wgmma_ty, rhs.smem, rhs.desc.tma_descriptor
        )
        return [lhs, rhs]

    def __iadd__(self, matmulResult):
        lhs = matmulResult[0]
        rhs = matmulResult[1]
        acc_op = nvgpu.WarpgroupMmaOp(
            self.acc_op.type, lhs, rhs, self.acc_op, transposeB=True
        )
        return WGMMAMatrix(WGMMAType.Accumulator, acc_op=acc_op)
```

**Purpose**: High-level API for WGMMA (Warpgroup Matrix Multiply-Accumulate) operations.

#### WGMMA Overview

**WGMMA**: Hopper Tensor Core instruction for matrix multiplication:
- Input: 128 threads (warpgroup) cooperatively multiply matrices
- Shapes: 64×N×K (flexible N and K)
- Data layout: Distributed across registers
- Result: Fragmented accumulator

#### Usage Pattern

```python
# Create input matrices (descriptors)
A = WGMMAMatrix(WGMMAType.Descriptor, [128, 64], desc=a_tma, smem=a_smem)
B = WGMMAMatrix(WGMMAType.Descriptor, [64, 128], desc=b_tma, smem=b_smem)

# Create accumulator
D = WGMMAMatrix(WGMMAType.Accumulator, shape=[128, 128], ty=T.f32())

# Matrix multiply with Pythonic syntax!
D += A @ B

# Store result
D.store_accumulator(d_smem)
```

#### Operator Overloading Magic

**`D += A @ B`** expands to:

**Step 1**: `A @ B` → calls `__matmul__`
```python
def __matmul__(self, rhs):
    lhs = nvgpu.warpgroup_generate_descriptor(self.wgmma_ty, self.smem, self.desc.tma_descriptor)
    rhs = nvgpu.warpgroup_generate_descriptor(rhs.wgmma_ty, rhs.smem, rhs.desc.tma_descriptor)
    return [lhs, rhs]
```

**Generated MLIR**:
```mlir
%desc_a = nvgpu.warpgroup.generate.descriptor %a_smem, %tma_desc_a
  : memref<128x64xf16, 3>, !nvgpu.tensormap.descriptor<...>
  -> !nvgpu.warpgroup.descriptor<tensor=memref<128x64xf16, 3>>

%desc_b = nvgpu.warpgroup.generate.descriptor %b_smem, %tma_desc_b
  : memref<64x128xf16, 3>, !nvgpu.tensormap.descriptor<...>
  -> !nvgpu.warpgroup.descriptor<tensor=memref<64x128xf16, 3>>
```

**Step 2**: `D += [desc_a, desc_b]` → calls `__iadd__`
```python
def __iadd__(self, matmulResult):
    lhs = matmulResult[0]  # desc_a
    rhs = matmulResult[1]  # desc_b
    acc_op = nvgpu.WarpgroupMmaOp(
        self.acc_op.type, lhs, rhs, self.acc_op, transposeB=True
    )
    return WGMMAMatrix(WGMMAType.Accumulator, acc_op=acc_op)
```

**Generated MLIR**:
```mlir
%d_new = nvgpu.warpgroup.mma %desc_a, %desc_b, %d_old {transposeB = true}
  : !nvgpu.warpgroup.descriptor<...>,
    !nvgpu.warpgroup.descriptor<...>,
    !nvgpu.warpgroup.accumulator<fragmented=vector<128x128xf32>>
  -> !nvgpu.warpgroup.accumulator<fragmented=vector<128x128xf32>>
```

**Hardware**: Compiles to `wgmma.mma_async.sync` PTX instruction.

---

## Section 4: NVDSL Decorators

### 4.1 @mlir_gpu_launch

**Lines 308-327**: [nvdsl.py:308-327](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L308-L327)

```python
@staticmethod
def mlir_gpu_launch(grid=(1, 1, 1), block=(1, 1, 1), smem=0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            launch_op = gpu.LaunchOp(
                None,
                [],
                *map(const, grid),
                *map(const, block),
                dynamicSharedMemorySize=arith.constant(T.i32(), smem),
            )
            launch_op.body.blocks.append(*([T.index()] * 12))
            with ir.InsertionPoint(launch_op.body.blocks[0]):
                result = func(*args, **kwargs)
                gpu.terminator()
                return result

        return wrapper

    return decorator
```

**Purpose**: Creates GPU launch operation with specified grid/block dimensions.

#### Usage

```python
@NVDSL.mlir_gpu_launch(grid=(4, 2, 1), block=(128, 1, 1), smem=32768)
def kernel():
    tidx = gpu.thread_id(gpu.Dimension.x)
    # Kernel code...
```

#### Execution Flow

```
1. Decorator call: mlir_gpu_launch(grid=(4,2,1), ...)
   Returns: decorator function

2. Function decoration: @decorator
   Returns: wrapper function

3. Kernel invocation: kernel()
   Executes: wrapper()

4. Inside wrapper:
   a. Create gpu.LaunchOp
   b. Add entry block with 12 arguments
   c. Set insertion point
   d. Execute kernel body (builds IR)
   e. Add gpu.terminator
```

#### GPU Launch Operation

**Call Path**:
```
gpu.LaunchOp(...)
    ↓
Python binding: _mlirDialectsGPU.LaunchOp
    ↓
C++: gpu::LaunchOp::build(...)
```

**Generated MLIR**:
```mlir
gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg3 = %c4, %arg4 = %c2, %arg5 = %c1)
           threads(%arg6, %arg7, %arg8) in (%arg9 = %c128, %arg10 = %c1, %arg11 = %c1)
           dynamic_shared_memory_size %c32768 {
  // Kernel body
  gpu.terminator
}
```

**Block arguments** (12 total):
- `%arg0, %arg1, %arg2`: `blockIdx.{x,y,z}`
- `%arg3, %arg4, %arg5`: `gridDim.{x,y,z}`
- `%arg6, %arg7, %arg8`: `threadIdx.{x,y,z}`
- `%arg9, %arg10, %arg11`: `blockDim.{x,y,z}`

---

### 4.2 @mlir_func - The Master Decorator

**Lines 329-460**: [nvdsl.py:329-460](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L329-L460)

**This is the most complex and important part of NVDSL!**

#### High-Level Flow

```
@NVDSL.mlir_func
def my_kernel(a, b, c):
    # Kernel code
    ...

# User calls:
my_kernel(np_array_a, np_array_b, np_array_c)
    ↓
Decorator intercepts call
    ↓
1. Create MLIR Context & Module
2. Build IR by executing function body
3. Verify IR
4. Compile IR → executable
5. Execute with NumPy arrays
6. Return results
```

#### Detailed Implementation

**Lines 331-332**: Wrapper setup
```python
@functools.wraps(funcBody)
def wrapper(*args, **kwargs):
    function_name = funcBody.__name__
```

**Lines 377-409**: Register value casters (operator overloading)
```python
@ir.register_value_caster(ir.IndexType.static_typeid)
@ir.register_value_caster(ir.F32Type.static_typeid)
@ir.register_value_caster(ir.F16Type.static_typeid)
@ir.register_value_caster(ir.F64Type.static_typeid)
@ir.register_value_caster(ir.IntegerType.static_typeid)
class ArithValue(ir.Value):
    __add__ = partialmethod(_binary_op, op="Add")
    __sub__ = partialmethod(_binary_op, op="Sub")
    __mul__ = partialmethod(_binary_op, op="Mul")
    __truediv__ = partialmethod(_binary_op, op="Div")
    __mod__ = partialmethod(_binary_op, op="Rem")
    __xor__ = partialmethod(_binary_op, op="XOr")
    __lt__ = partialmethod(_binary_op, op="Cmp", predAtt="ult")
    __eq__ = partialmethod(_binary_op, op="Cmp", predAtt="eq")
    __and__ = partialmethod(_binary_op, op="And")
    __or__ = partialmethod(_binary_op, op="Or")
```

**Purpose**: Makes MLIR values behave like Python numbers!

**Example**:
```python
tidx = gpu.thread_id(gpu.Dimension.x)  # Returns ir.Value
result = tidx + 10  # Calls ArithValue.__add__(10)
                    # Generates: arith.addi %tidx, %c10
```

**Lines 411-428**: Build MLIR module
```python
with ir.Context(), ir.Location.unknown():
    types = []
    for arg in args:
        types.append(get_mlir_ty(arg))

    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        fop = func.FuncOp(function_name, (types, []))
        fop.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        with ir.InsertionPoint(fop.add_entry_block()):
            fargs = []
            for i, a in enumerate(types):
                fargs.append(fop.arguments[i])

            # Execute user function body!
            result = funcBody(*fargs, **kwargs)
            func.ReturnOp([])
```

**Key insight**: `funcBody(*fargs, **kwargs)` **executes the user's Python function**, which builds IR!

**Lines 434**: Verify
```python
module.operation.verify()
```

**Lines 437-446**: Compile
```python
options = f"cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3"
support_lib = os.getenv("SUPPORT_LIB")
compiler = nvgpucompiler.NvgpuCompiler(
    options, opt_level=3, shared_libs=[support_lib]
)
engine = compiler.compile_and_jit(module)
engine.initialize()
```

**Lines 453-456**: Execute
```python
newArgs = get_mlir_func_obj_ty(args)
engine.invoke(function_name, *newArgs)
```

---

## Complete Example Trace

Let's trace `D += A @ B`:

```python
# User code
D = WGMMAMatrix(WGMMAType.Accumulator, shape=[128, 128], ty=T.f32())
A = WGMMAMatrix(WGMMAType.Descriptor, [128, 64], desc=a_tma, smem=a_smem)
B = WGMMAMatrix(WGMMAType.Descriptor, [64, 128], desc=b_tma, smem=b_smem)
D += A @ B
```

**Execution**:

```
1. D created:
   └─> nvgpu.warpgroup_mma_init_accumulator(acc_ty)
       └─> MLIR: %d_init = nvgpu.warpgroup.mma.init.accumulator
                    : !nvgpu.warpgroup.accumulator<fragmented=vector<128x128xf32>>

2. A @ B evaluated:
   └─> A.__matmul__(B)
       ├─> nvgpu.warpgroup_generate_descriptor(A.wgmma_ty, A.smem, A.desc.tma_descriptor)
       │   └─> MLIR: %desc_a = nvgpu.warpgroup.generate.descriptor %a_smem, %tma_desc_a ...
       └─> nvgpu.warpgroup_generate_descriptor(B.wgmma_ty, B.smem, B.desc.tma_descriptor)
           └─> MLIR: %desc_b = nvgpu.warpgroup.generate.descriptor %b_smem, %tma_desc_b ...
       └─> Returns [%desc_a, %desc_b]

3. D += [%desc_a, %desc_b]:
   └─> D.__iadd__([%desc_a, %desc_b])
       └─> nvgpu.WarpgroupMmaOp(D.acc_op.type, %desc_a, %desc_b, D.acc_op, transposeB=True)
           └─> MLIR: %d_new = nvgpu.warpgroup.mma %desc_a, %desc_b, %d_init {transposeB = true} ...
       └─> Returns WGMMAMatrix(WGMMAType.Accumulator, acc_op=%d_new)

4. D updated with new accumulator
```

---

## Summary

**NVDSL Layer Responsibilities**:

1. **High-level API**: Pythonic interface (TMA, Mbarriers, WGMMAMatrix)
2. **Operator overloading**: Make MLIR values behave like Python objects
3. **Context management**: `with` statements for structured code
4. **Decorator magic**: Transform function execution semantics
5. **Type conversion**: NumPy ↔ MLIR types
6. **Compilation orchestration**: IR building → compilation → execution

**Next**: [03_python_bindings.md](03_python_bindings.md) - MLIR Python bindings layer
