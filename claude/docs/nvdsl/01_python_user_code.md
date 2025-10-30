# Layer 5: User Python Code - Ch3.py Walkthrough

## Introduction

This document provides a **line-by-line** walkthrough of [Ch3.py](../../../mlir/test/Examples/NVGPU/Ch3.py), showing how a user writes GPU kernels in Python that compile to executable CUDA code.

## File: Ch3.py - GEMM 128×128×64 with Tensor Core

**Location**: [mlir/test/Examples/NVGPU/Ch3.py](../../../mlir/test/Examples/NVGPU/Ch3.py)

**Purpose**: Demonstrates a matrix multiplication (GEMM) operation using NVIDIA Tensor Cores with TMA (Tensor Memory Accelerator) for memory transfers.

### Operation: D = A × B
- A: 128×64 matrix (f16)
- B: 64×128 matrix (f16)
- D: 128×128 matrix (f32 accumulator)

---

## Code Structure Overview

```
Ch3.py
├── Imports (lines 18-22)
├── Helper Function: tma_load (lines 25-58)
├── Main Kernel: gemm_128_128_64 (lines 60-115)
│   ├── GPU Memory Allocation (lines 62-70)
│   ├── TMA Descriptor Creation (lines 71-78)
│   ├── GPU Kernel Launch (lines 80-111)
│   └── Result Copy Back (lines 113-114)
└── Python Execution & Verification (lines 118-130)
```

---

## Detailed Code Walkthrough

### Section 1: Imports

**Lines 18-22**: [Ch3.py:18-22](../../../mlir/test/Examples/NVGPU/Ch3.py#L18-L22)

```python
from mlir import ir
from mlir.dialects import nvgpu, scf, arith, memref, vector, gpu
from tools.nvdsl import *
from mlir.extras import types as T
import numpy as np
```

| Import | Purpose | Implementation Layer |
|--------|---------|---------------------|
| `mlir.ir` | Core IR building blocks (Context, Module, InsertionPoint) | Python bindings → C API |
| `mlir.dialects.*` | Dialect-specific operations (nvgpu.mbarrier_create, gpu.thread_id) | Python bindings → C API |
| `tools.nvdsl` | High-level DSL (TMA, Mbarriers, WGMMAMatrix, decorators) | Pure Python DSL |
| `mlir.extras.types` | Type helpers (T.f16(), T.f32()) | Python bindings |
| `numpy` | Input/output data arrays | Pure Python/C |

**What Happens**: Python imports load compiled shared libraries (`.so` files) that expose C API functions.

**Call Path**:
```
Python import statement
    ↓
Python module loader
    ↓
Load _mlirDialectsGPU.so, _mlirDialectsNVGPU.so, etc.
    ↓
Shared libraries expose C functions via nanobind
    ↓
Python objects wrapping C API handles
```

---

### Section 2: Helper Function - tma_load

**Lines 25-58**: [Ch3.py:25-58](../../../mlir/test/Examples/NVGPU/Ch3.py#L25-L58)

```python
def tma_load(
    mbar_group: Mbarriers,
    a_tma: TMA,
    b_tma: TMA,
    p,
):
```

**Purpose**: Loads two input matrices from global memory to shared memory using TMA.

**Parameters**:
- `mbar_group`: Mbarrier group for synchronization
- `a_tma`: TMA descriptor for matrix A
- `b_tma`: TMA descriptor for matrix B
- `p`: Predicate (which thread executes TMA)

#### Line-by-Line Analysis

**Lines 41-43**: Calculate TMA transfer sizes

```python
size_tma_a = get_type_size(a_tma.tma_memref)  # 128×64×2 bytes = 16384 bytes
size_tma_b = get_type_size(b_tma.tma_memref)  # 64×64×2 bytes = 8192 bytes
ta_count = size_tma_a + (size_tma_b * 2)       # Total: 32768 bytes
```

**What Happens**:
1. `a_tma.tma_memref` → accesses Python object attribute → returns MLIR `MemRefType`
2. `get_type_size(...)` → calls [nvdsl.py:23-33](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L23-L33) → calculates byte size
3. Result stored in Python integer

**Lines 45-51**: Calculate shared memory offsets and create views

```python
off_b = size_tma_a                  # Offset for B matrix (after A)
off_b2 = off_b + size_tma_b         # Offset for second B tile
a_elem_ty = a_tma.tma_memref.element_type    # f16
b_elem_ty = b_tma.tma_memref.element_type    # f16
a = get_dynamic_shared_memory(a_tma.tma_memref.shape, a_elem_ty)
b1 = get_dynamic_shared_memory(b_tma.tma_memref.shape, b_elem_ty, off_b)
b2 = get_dynamic_shared_memory(b_tma.tma_memref.shape, b_elem_ty, off_b2)
```

**What Happens** - `get_dynamic_shared_memory` call:

```
Python function call: get_dynamic_shared_memory([128, 64], f16, 0)
    ↓
NVDSL function (nvdsl.py:259-275)
    ↓
gpu.dynamic_shared_memory(...) — Python binding call
    ↓
_mlirDialectsGPU.so: gpu.DynamicSharedMemoryOp(...)
    ↓
C API: mlirGPUDynamicSharedMemoryOp(...)
    ↓
C++: gpu::DynamicSharedMemoryOp::build(...)
    ↓
MLIR Operation created in IR
    ↓
Returns ir.Value (Python wrapper around MlirValue handle)
```

**Line 53**: Mbarrier arrive with transaction count

```python
mbar_group[0].arrive(ta_count, predicate=p)
```

**Call Stack Trace**:

```
1. Python: mbar_group[0]
   └─> Calls Mbarriers.__getitem__(0)  [nvdsl.py:67-69]
       └─> Sets self.id_op = const(0)
       └─> Returns self

2. Python: .arrive(ta_count, predicate=p)
   └─> Calls Mbarriers.arrive(...)  [nvdsl.py:80-89]

3. Inside arrive():
   txcount_op = const(ta_count)  # Convert Python int → MLIR constant
   └─> arith.constant(T.index(), 32768)
       └─> Python binding: arith.ConstantOp
           └─> C API: mlirArithConstantOp
               └─> C++: arith::ConstantOp::build()

4. nvgpu.mbarrier_arrive_expect_tx(...)
   └─> Python binding call [generated from TableGen]
       └─> _mlirDialectsNVGPU.mbarrier_arrive_expect_tx
           └─> C API: mlirNVGPUMBarrierArriveExpectTx
               └─> C++: nvgpu::MBarrierArriveExpectTxOp::build()
```

**Generated MLIR IR** (conceptual):
```mlir
%c32768 = arith.constant 32768 : index
nvgpu.mbarrier.arrive.expect_tx %mbar_group[%c0], %c32768 predicate %p
```

**Lines 55-57**: TMA load operations

```python
a_tma.load(a, mbar_group[0], coords=[0, 0], predicate=p)
b_tma.load(b1, mbar_group[0], coords=[0, 0], predicate=p)
b_tma.load(b2, mbar_group[0], coords=[64, 0], predicate=p)
```

**Deep Dive**: `a_tma.load(...)` execution:

```
1. Python: a_tma.load(a, mbar_group[0], coords=[0, 0], predicate=p)
   └─> TMA.load method [nvdsl.py:154-162]

2. Inside TMA.load():
   nvgpu.TmaAsyncLoadOp(
       dest,                          # a (shared memory view)
       mbarrier.mbar_group_op,        # mbarrier group
       self.tma_descriptor,           # TMA descriptor (created earlier)
       coordinates=map(const, coords), # [const(0), const(0)]
       mbarId=mbarrier.id_op,         # const(0)
       predicate=predicate             # p
   )

3. Python binding: nvgpu.TmaAsyncLoadOp(...)
   └─> Calls __init__ in generated Python class
       └─> _mlirDialectsNVGPU.TmaAsyncLoadOp(...)

4. C++ Builder (simplified):
   └─> nvgpu::TmaAsyncLoadOp::build(
           builder, result,
           dest, mbar_group, descriptor,
           coordinates, mbarId, predicate
       )
```

**Generated MLIR IR**:
```mlir
%c0 = arith.constant 0 : index
%c64 = arith.constant 64 : index
nvgpu.tma.async.load %a[%c0, %c0], %mbar_group[%c0], %tma_desc_a
  : memref<128x64xf16, 3>, !nvgpu.mbarrier.group<...>, !nvgpu.tensormap.descriptor<...>
nvgpu.tma.async.load %b1[%c0, %c0], %mbar_group[%c0], %tma_desc_b
  : memref<64x64xf16, 3>, !nvgpu.mbarrier.group<...>, !nvgpu.tensormap.descriptor<...>
nvgpu.tma.async.load %b2[%c64, %c0], %mbar_group[%c0], %tma_desc_b
  : memref<64x64xf16, 3>, !nvgpu.mbarrier.group<...>, !nvgpu.tensormap.descriptor<...>
```

---

### Section 3: Main Kernel Function

**Lines 60-115**: [Ch3.py:60-115](../../../mlir/test/Examples/NVGPU/Ch3.py#L60-L115)

```python
@NVDSL.mlir_func
def gemm_128_128_64(a, b, d):
```

#### The @NVDSL.mlir_func Decorator

**Critical**: This decorator **completely changes** how the function executes!

**Location**: [nvdsl.py:329-460](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L329-L460)

**What It Does**:
1. **Intercepts** the function call
2. **Does NOT execute** the Python function body normally
3. **Instead**: Builds MLIR IR by interpreting the Python code
4. **Compiles** the IR to executable code
5. **Executes** the compiled code
6. **Returns** results

**Execution Flow**:

```
Python: gemm_128_128_64(a, b, d)  # User calls function
    ↓
@NVDSL.mlir_func wrapper intercepts
    ↓
wrapper(*args, **kwargs) executes [nvdsl.py:332]
    ↓
1. Create MLIR Context [line 411]
   with ir.Context(), ir.Location.unknown():
       ↓
       Creates new MLIRContext (C++ object)

2. Build IR Module [lines 417-428]
   module = ir.Module.create()
       ↓
       Python: ir.Module.create()
           ↓
           C API: mlirModuleCreateEmpty(mlirLocationUnknownGet(ctx))
               ↓
               C++: ModuleOp::create(location)

   with ir.InsertionPoint(module.body):
       fop = func.FuncOp(function_name, (types, []))
           ↓
           Creates function operation

       with ir.InsertionPoint(fop.add_entry_block()):
           result = funcBody(*fargs, **kwargs)  # ← EXECUTES USER FUNCTION
               ↓
               User's gemm_128_128_64 function body runs
               Building IR as it executes!

3. Verify Module [line 434]
   module.operation.verify()
       ↓
       C++: module.verify()

4. Compile [lines 437-446]
   compiler = nvgpucompiler.NvgpuCompiler(...)
   engine = compiler.compile_and_jit(module)
       ↓
       Runs pass pipeline: gpu-lower-to-nvvm-pipeline
       Converts to LLVM IR
       JIT compiles to machine code

5. Execute [line 456]
   engine.invoke(function_name, *newArgs)
       ↓
       Runs compiled GPU code!
```

#### GPU Memory Allocation (Lines 62-70)

```python
token_ty = gpu.AsyncTokenType.get()
t1 = gpu.wait(token_ty, [])
a_dev, t2 = gpu.alloc(a.type, token_ty, [t1], [], [])
b_dev, t3 = gpu.alloc(b.type, token_ty, [t2], [], [])
d_dev, t4 = gpu.alloc(d.type, token_ty, [t3], [], [])
t5 = gpu.memcpy(token_ty, [t4], a_dev, a)
t6 = gpu.memcpy(token_ty, [t5], b_dev, b)
t7 = gpu.wait(token_ty, [t6])
```

**Pattern**: Async GPU operations with dependency tokens

**Each operation builds MLIR IR**:

```python
token_ty = gpu.AsyncTokenType.get()
```
**Trace**:
```
Python: gpu.AsyncTokenType.get()
    ↓
_mlirDialectsGPU.AsyncTokenType.get()  [Python binding]
    ↓
C API: mlirGPUAsyncTokenTypeGet(ctx)
    ↓
C++: gpu::AsyncTokenType::get(context)
    ↓
Returns MlirType handle (wrapped as Python ir.Type)
```

```python
a_dev, t2 = gpu.alloc(a.type, token_ty, [t1], [], [])
```
**Trace**:
```
Python: gpu.alloc(memref<128x64xf16>, !gpu.async.token, [t1], [], [])
    ↓
_mlirDialectsGPU.alloc(...)  [Python binding - generated from TableGen]
    ↓
C++: gpu::AllocOp::build(
        builder, result,
        asyncDependencies=[t1],
        asyncToken=token_ty,
        memrefType=memref<128x64xf16>
     )
    ↓
Creates: %a_dev, %t2 = gpu.alloc async [%t1] () : memref<128x64xf16>
```

**Generated MLIR IR**:
```mlir
%token_ty = !gpu.async.token
%t1 = gpu.wait async : !gpu.async.token
%a_dev, %t2 = gpu.alloc async [%t1] () : memref<128x64xf16>
%b_dev, %t3 = gpu.alloc async [%t2] () : memref<64x128xf16>
%d_dev, %t4 = gpu.alloc async [%t3] () : memref<128x128xf32>
%t5 = gpu.memcpy async [%t4] %a_dev, %a : memref<128x64xf16>, memref<128x64xf16>
%t6 = gpu.memcpy async [%t5] %b_dev, %b : memref<64x128xf16>, memref<64x128xf16>
%t7 = gpu.wait async [%t6] : !gpu.async.token
```

#### TMA Descriptor Creation (Lines 71-78)

```python
sw = nvgpu.TensorMapSwizzleKind.SWIZZLE_128B
a_tma = TMA([128, 64], a.type, swizzle=sw)
b_tma = TMA([64, 64], b.type, swizzle=sw)
a_tma.create_descriptor(a_dev)
b_tma.create_descriptor(b_dev)
```

**Deep Dive**: `TMA([128, 64], a.type, swizzle=sw)`

**Trace**:
```
1. Python: TMA([128, 64], a.type, swizzle=sw)
   └─> Calls TMA.__init__ [nvdsl.py:105-120]

2. Inside __init__:
   self.tma_box_shape = [128, 64]
   self.memref_ty = a.type  # memref<128x64xf16>
   self.tma_memref = ir.MemRefType.get([128, 64], a.type.element_type)
       ↓
       Python: ir.MemRefType.get([128, 64], f16)
           ↓
           C API: mlirMemRefTypeGet(f16, rank=2, shape=[128,64], ...)
               ↓
               C++: MemRefType::get({128, 64}, f16Type, ...)

3. Returns TMA object (pure Python, no IR yet)
```

**Deep Dive**: `a_tma.create_descriptor(a_dev)`

**Trace**:
```
1. Python: a_tma.create_descriptor(a_dev)
   └─> Calls TMA.create_descriptor [nvdsl.py:138-149]

2. Inside create_descriptor:
   # Cast to unranked memref
   device_unranked_memref = memref.CastOp(
       ir.UnrankedMemRefType.get(f16, memory_space),
       device_ptr
   )
       ↓
       Python binding: memref.CastOp
           ↓
           C++: memref::CastOp::build(...)

   # Create TMA descriptor
   self.tma_descriptor = nvgpu.TmaCreateDescriptorOp(
       tma_descriptor_ty,
       device_unranked_memref,
       map(const, self.tma_box_shape)  # [const(128), const(64)]
   )
       ↓
       Python binding: nvgpu.TmaCreateDescriptorOp
           ↓
           C API: mlirNVGPUTmaCreateDescriptorOp(...)
               ↓
               C++: nvgpu::TmaCreateDescriptorOp::build(...)
```

**Generated MLIR IR**:
```mlir
%a_unranked = memref.cast %a_dev : memref<128x64xf16> to memref<*xf16>
%c128 = arith.constant 128 : index
%c64 = arith.constant 64 : index
%tma_desc_a = nvgpu.tma.create.descriptor %a_unranked box[%c128, %c64]
  : memref<*xf16> -> !nvgpu.tensormap.descriptor<...>
```

#### GPU Kernel Launch (Lines 80-111)

```python
@NVDSL.mlir_gpu_launch(grid=(1, 1, 1), block=(128, 1, 1), smem=smem_size_in_bytes)
def gemm_tma_kernel():
    # Kernel body
```

**The @mlir_gpu_launch Decorator**:

**Location**: [nvdsl.py:308-327](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L308-L327)

**Execution**:
```python
@NVDSL.mlir_gpu_launch(grid=(1,1,1), block=(128,1,1), smem=16384)
def gemm_tma_kernel():
    tidx = gpu.thread_id(gpu.Dimension.x)
    # ...
```

**What Happens**:
```
1. Decorator evaluates: NVDSL.mlir_gpu_launch(grid=(1,1,1), ...)
   └─> Returns decorator function [nvdsl.py:309]

2. Decorator wraps gemm_tma_kernel
   └─> Returns wrapper function [nvdsl.py:310-325]

3. When gemm_tma_kernel() is called (line 111):
   └─> Wrapper executes [nvdsl.py:311-323]

4. Inside wrapper:
   launch_op = gpu.LaunchOp(
       None, [],
       *map(const, grid),    # grid dimensions
       *map(const, block),   # block dimensions
       dynamicSharedMemorySize=arith.constant(T.i32(), smem)
   )
       ↓
       Python binding: gpu.LaunchOp
           ↓
           C++: gpu::LaunchOp::build(...)

5. Add entry block to launch region:
   launch_op.body.blocks.append(*([T.index()] * 12))
       ↓
       Creates block with 12 index arguments:
       blockIdx.{x,y,z}, blockDim.{x,y,z},
       threadIdx.{x,y,z}, threadDim.{x,y,z}

6. Set insertion point inside launch:
   with ir.InsertionPoint(launch_op.body.blocks[0]):
       result = func(*args, **kwargs)  # Execute kernel body!
       gpu.terminator()
```

**Generated MLIR IR**:
```mlir
gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
           threads(%tx, %ty, %tz) in (%block_x = %c128, %block_y = %c1, %block_z = %c1)
           dynamic_shared_memory_size %c16384 {
  // Kernel body inserted here
  gpu.terminator
}
```

#### Inside Kernel: Thread ID (Line 82)

```python
tidx = gpu.thread_id(gpu.Dimension.x)
```

**Trace**:
```
Python: gpu.thread_id(gpu.Dimension.x)
    ↓
_mlirDialectsGPU.thread_id(dimension=gpu.Dimension.x)
    ↓
C++: gpu::ThreadIdOp::build(builder, result, gpu::Dimension::x)
    ↓
Creates: %tidx = gpu.thread_id x : index
```

#### Mbarrier Initialization (Lines 84-87)

```python
mbar_group = Mbarriers(number_of_barriers=1)
isThread0 = tidx == 0
mbar_group[0].init(1, predicate=isThread0)
```

**Trace for `tidx == 0`**:
```
Python: tidx == 0
    ↓
ArithValue.__eq__(0)  [nvdsl.py:396] (operator overloading!)
    ↓
_binary_op(tidx, 0, op="Cmp", predAtt="eq")  [nvdsl.py:345-375]
    ↓
arith.CmpIOp(arith.CmpIPredicate.eq, tidx, const(0))
    ↓
C++: arith::CmpIOp::build(...)
    ↓
Creates: %isThread0 = arith.cmpi eq, %tidx, %c0 : index
```

**Operator Overloading Magic**:

The `ArithValue` class in nvdsl.py overloads Python operators:

```python
class ArithValue(ir.Value):
    __add__ = partialmethod(_binary_op, op="Add")
    __eq__ = partialmethod(_binary_op, op="Cmp", predAtt="eq")
    # etc.
```

So when you write `tidx == 0`, Python:
1. Calls `tidx.__eq__(0)`
2. Which calls `_binary_op(tidx, 0, op="Cmp", predAtt="eq")`
3. Which generates MLIR: `arith.cmpi eq, %tidx, %c0`

#### Matrix Multiply (Lines 101-106)

```python
A = WGMMAMatrix(WGMMAType.Descriptor, [M, K], desc=a_tma, smem=a_smem)
B = WGMMAMatrix(WGMMAType.Descriptor, [K, N], desc=b_tma, smem=b_smem)
D = WGMMAMatrix(WGMMAType.Accumulator, shape=[M, N], ty=T.f32())

# Matrix Multiply
D += A @ B
```

**Deep Dive**: `D += A @ B`

This Python expression triggers operator overloading:

**Step 1**: `A @ B` (matrix multiplication operator)
```
Python: A @ B
    ↓
WGMMAMatrix.__matmul__(B)  [nvdsl.py:241-248]
    ↓
lhs = nvgpu.warpgroup_generate_descriptor(A.wgmma_ty, A.smem, A.desc.tma_descriptor)
rhs = nvgpu.warpgroup_generate_descriptor(B.wgmma_ty, B.smem, B.desc.tma_descriptor)
return [lhs, rhs]
```

**Step 2**: `D += [lhs, rhs]`
```
Python: D += matmulResult
    ↓
WGMMAMatrix.__iadd__(matmulResult)  [nvdsl.py:250-256]
    ↓
acc_op = nvgpu.WarpgroupMmaOp(
    D.acc_op.type,
    lhs, rhs, D.acc_op,
    transposeB=True
)
    ↓
Python binding: nvgpu.WarpgroupMmaOp
    ↓
C++: nvgpu::WarpgroupMmaOp::build(...)
```

**Generated MLIR IR**:
```mlir
%desc_a = nvgpu.warpgroup.generate.descriptor %a_smem, %tma_desc_a
  : memref<128x64xf16, 3>, !nvgpu.tensormap.descriptor<...>
  -> !nvgpu.warpgroup.descriptor<...>

%desc_b = nvgpu.warpgroup.generate.descriptor %b_smem, %tma_desc_b
  : memref<64x128xf16, 3>, !nvgpu.tensormap.descriptor<...>
  -> !nvgpu.warpgroup.descriptor<...>

%d_acc = nvgpu.warpgroup.mma %desc_a, %desc_b, %d_init {transposeB = true}
  : !nvgpu.warpgroup.descriptor<...>, !nvgpu.warpgroup.descriptor<...>,
    !nvgpu.warpgroup.accumulator<...>
  -> !nvgpu.warpgroup.accumulator<...>
```

---

### Section 4: Python Execution

**Lines 118-130**: [Ch3.py:118-130](../../../mlir/test/Examples/NVGPU/Ch3.py#L118-L130)

```python
M = 128
N = 128
K = 64
a = np.random.randn(M, K).astype(np.float16)
b = np.random.randn(K, N).astype(np.float16)
d = np.zeros((M, N), np.float32)
gemm_128_128_64(a, b, d)  # ← THIS TRIGGERS EVERYTHING!

ref_d = a.astype(np.float16) @ b.astype(np.float16)
np.testing.assert_allclose(d, ref_d, rtol=5e-03, atol=1e-01)
print("PASS")
```

**When `gemm_128_128_64(a, b, d)` is called**:

```
1. @NVDSL.mlir_func wrapper intercepts call
2. Extracts types from NumPy arrays:
   a: np.ndarray[128, 64, f16] → memref<128x64xf16>
   b: np.ndarray[64, 128, f16] → memref<64x128xf16>
   d: np.ndarray[128, 128, f32] → memref<128x128xf32>
3. Creates MLIR module and builds IR
4. Compiles to executable
5. Creates memref descriptors from NumPy arrays
6. Invokes compiled function with memref descriptors
7. GPU code executes!
8. Results written back to NumPy array 'd'
9. Function returns
10. Verification runs: d == ref_d ✓
```

---

## Complete Call Graph Summary

```
User calls gemm_128_128_64(a, b, d)
    ↓
@NVDSL.mlir_func wrapper [nvdsl.py:329-460]
    ↓
Create MLIR Context & Module
    ↓
Execute function body (building IR):
    ├─> gpu.alloc(...) → builds IR operations
    ├─> TMA(...).create_descriptor(...) → builds IR
    ├─> @mlir_gpu_launch → gpu.LaunchOp
    │   ├─> gpu.thread_id → builds IR
    │   ├─> Mbarriers → nvgpu.mbarrier.* ops
    │   ├─> tma_load → nvgpu.tma.async.load ops
    │   ├─> WGMMAMatrix operations → nvgpu.warpgroup.* ops
    │   └─> gpu.terminator
    └─> gpu.memcpy(...) → builds IR
    ↓
Verify module
    ↓
Compile [nvgpucompiler.py]
    ├─> Run pass pipeline: gpu-lower-to-nvvm-pipeline
    ├─> Convert NVVM → LLVM IR
    ├─> JIT compile
    └─> Load GPU binary
    ↓
Execute compiled code with NumPy arrays
    ↓
Return results
```

---

## Data Flow: NumPy Array → GPU Memory

```
NumPy Array (Python)
  shape: (128, 64)
  dtype: float16
  data: contiguous memory buffer
    ↓
get_mlir_ty(arg) [nvdsl.py:278-303]
    ↓
rt.get_ranked_memref_descriptor(arg)
  Returns: MemRefDescriptor {
    allocated: pointer to data
    aligned: pointer to data
    offset: 0
    sizes: [128, 64]
    strides: [64, 1]
  }
    ↓
ir.MemRefType.get([128, 64], T.f16())
    ↓
MlirType handle (C API)
    ↓
gpu.alloc creates GPU memory
    ↓
gpu.memcpy copies data
    ↓
TMA loads data to shared memory
    ↓
WGMMA operates on shared memory
    ↓
Store results to GPU memory
    ↓
gpu.memcpy copies back
    ↓
Results written to NumPy array
```

---

## Key Insights

1. **No Python execution inside kernel**: The Python code builds MLIR IR, it doesn't execute Python operations
2. **Operator overloading is crucial**: `+`, `==`, `@` etc. are overloaded to build IR operations
3. **Decorators transform semantics**: `@NVDSL.mlir_func` completely changes function behavior
4. **Type conversion happens multiple times**: NumPy → MLIR → C API → C++ → LLVM → PTX
5. **IR is built incrementally**: Each Python statement adds operations to the IR

---

**Next**: [02_nvdsl_layer.md](02_nvdsl_layer.md) - Deep dive into the NVDSL DSL implementation
