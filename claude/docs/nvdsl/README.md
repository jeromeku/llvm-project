# NVDSL: Complete System Documentation

## Overview

This directory contains comprehensive documentation of the NVDSL (NVIDIA DSL) system - a Python-embedded domain-specific language for writing high-performance CUDA kernels that compile through MLIR.

**What is NVDSL?**
- Python DSL for GPU programming
- Uses MLIR's GPU/NVGPU/NVVM dialects
- Supports Hopper architecture features (TMA, WGMMA, mbarriers)
- JIT compiles Python code to executable CUDA binaries
- Provides Pythonic API with operator overloading

---

## Documentation Index

| Document | Description | Key Topics |
|----------|-------------|------------|
| **[00_overview.md](00_overview.md)** | System architecture overview | 5-layer architecture, compilation pipeline, key concepts |
| **[01_python_user_code.md](01_python_user_code.md)** | User-facing Python code (Ch3.py walkthrough) | Line-by-line trace, operator overloading, decorators, call paths |
| **[02_nvdsl_layer.md](02_nvdsl_layer.md)** | NVDSL DSL implementation | Classes (TMA, Mbarriers, WGMMAMatrix), decorators, utilities |
| **[06_compilation_pipeline.md](06_compilation_pipeline.md)** | IR → executable compilation | Pass pipeline, PTX generation, GPU runtime, optimization |
| **[07_ch5_complete_trace.md](07_ch5_complete_trace.md)** | Complete Python→C++ dispatch trace | Frame-by-frame execution, state changes, memory layout, full call stacks |
| **[08_mlir_uniquers.md](08_mlir_uniquers.md)** | MLIR uniquing system internals | TypeUniquer, AttributeUniquer, AffineUniquer, hash-consing, storage classes |

---

## Quick Start

### Example: Simple GEMM

```python
from mlir import ir
from mlir.dialects import nvgpu, gpu, memref
from tools.nvdsl import *
import numpy as np

@NVDSL.mlir_func
def gemm_kernel(a, b, d):
    # Allocate GPU memory
    token_ty = gpu.AsyncTokenType.get()
    t1 = gpu.wait(token_ty, [])
    a_dev, t2 = gpu.alloc(a.type, token_ty, [t1], [], [])
    b_dev, t3 = gpu.alloc(b.type, token_ty, [t2], [], [])
    d_dev, t4 = gpu.alloc(d.type, token_ty, [t3], [], [])

    # Copy to GPU
    t5 = gpu.memcpy(token_ty, [t4], a_dev, a)
    t6 = gpu.memcpy(token_ty, [t5], b_dev, b)

    # Create TMA descriptors
    a_tma = TMA([128, 64], a.type)
    b_tma = TMA([64, 128], b.type)
    a_tma.create_descriptor(a_dev)
    b_tma.create_descriptor(b_dev)

    # Launch GPU kernel
    @NVDSL.mlir_gpu_launch(grid=(1,1,1), block=(128,1,1))
    def kernel():
        # Get thread ID
        tidx = gpu.thread_id(gpu.Dimension.x)

        # Create barriers
        mbar = Mbarriers(number_of_barriers=1)
        mbar[0].init(1, predicate=(tidx == 0))

        # TMA load matrices
        a_smem = get_dynamic_shared_memory([128, 64], T.f16())
        b_smem = get_dynamic_shared_memory([64, 128], T.f16())

        mbar[0].arrive(...)
        a_tma.load(a_smem, mbar[0], coords=[0, 0])
        b_tma.load(b_smem, mbar[0], coords=[0, 0])

        # Wait for loads
        mbar[0].try_wait()

        # Matrix multiply with Tensor Cores
        A = WGMMAMatrix(WGMMAType.Descriptor, [128, 64], desc=a_tma, smem=a_smem)
        B = WGMMAMatrix(WGMMAType.Descriptor, [64, 128], desc=b_tma, smem=b_smem)
        D = WGMMAMatrix(WGMMAType.Accumulator, shape=[128, 128], ty=T.f32())

        # Pythonic matrix multiply!
        D += A @ B

        # Store results
        D.store_accumulator(d_dev)

    kernel()

    # Copy results back
    t7 = gpu.memcpy(token_ty, [t6], d, d_dev)
    gpu.wait(None, [t7])

# Execute with NumPy arrays
M, N, K = 128, 128, 64
a = np.random.randn(M, K).astype(np.float16)
b = np.random.randn(K, N).astype(np.float16)
d = np.zeros((M, N), np.float32)

gemm_kernel(a, b, d)  # Compiles and executes!
```

---

## System Architecture

### 5-Layer Architecture

```
┌─────────────────────────────────────┐
│ Layer 5: User Python Code          │  Ch3.py, Ch4.py, Ch5.py
│          NumPy arrays, decorators   │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│ Layer 4: NVDSL Python DSL           │  nvdsl.py
│          TMA, Mbarriers, WGMMAMatrix│
│          @mlir_func, @mlir_gpu_launch│
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│ Layer 3: MLIR Python Bindings       │  mlir.dialects.*, mlir.ir
│          ir.Module, gpu.*, nvgpu.*  │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│ Layer 2: MLIR C API                 │  mlir-c/*.h
│          MlirContext, MlirType      │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│ Layer 1: MLIR C++ Core              │  mlir/IR/*, mlir/Dialect/*
│          Operation definitions      │
└─────────────────────────────────────┘
```

### Compilation Pipeline

```
Python Code
    ↓
MLIR IR (GPU/NVGPU)
    ↓ [gpu-lower-to-nvvm-pipeline]
MLIR IR (NVVM)
    ↓ [LLVM]
LLVM IR
    ↓ [NVPTX backend]
PTX Assembly
    ↓ [ptxas]
CUDA Binary
    ↓ [GPU Runtime]
Executable
```

---

## Key Concepts

### Decorators Transform Execution

**@NVDSL.mlir_func**: Transforms function into MLIR IR builder
```python
@NVDSL.mlir_func
def kernel(a, b, c):
    # This code BUILDS IR, doesn't execute Python!
    x = a + b  # Generates: arith.addi %a, %b
```

**@NVDSL.mlir_gpu_launch**: Creates GPU kernel launch
```python
@NVDSL.mlir_gpu_launch(grid=(4,2,1), block=(128,1,1))
def gpu_kernel():
    tidx = gpu.thread_id(gpu.Dimension.x)
```

### Operator Overloading

Python operators build MLIR operations:

| Python | MLIR Operation | Example |
|--------|----------------|---------|
| `a + b` | `arith.addi` | `%sum = arith.addi %a, %b` |
| `a == b` | `arith.cmpi eq` | `%eq = arith.cmpi eq, %a, %b` |
| `A @ B` | `nvgpu.warpgroup.mma` | Matrix multiply |
| `D += matmul` | `nvgpu.warpgroup.mma` | Accumulate |

### High-Level Abstractions

**TMA (Tensor Memory Accelerator)**:
```python
tma = TMA([128, 64], memref_ty, swizzle=SWIZZLE_128B)
tma.create_descriptor(device_ptr)
tma.load(shared_mem, mbar[0], coords=[0, 0])
```

**Mbarriers (Memory Barriers)**:
```python
mbar = Mbarriers(number_of_barriers=7)
mbar[0].init(1)
mbar[0].arrive(txcount=16384)
mbar[0].try_wait(phase=False)
```

**WGMMAMatrix (Warpgroup Matrix)**:
```python
A = WGMMAMatrix(WGMMAType.Descriptor, [128, 64], ...)
B = WGMMAMatrix(WGMMAType.Descriptor, [64, 128], ...)
D = WGMMAMatrix(WGMMAType.Accumulator, [128, 128], ty=T.f32())
D += A @ B  # Tensor Core operation!
```

---

## Examples Analyzed

### Ch3.py - Basic GEMM

**File**: [mlir/test/Examples/NVGPU/Ch3.py](../../../mlir/test/Examples/NVGPU/Ch3.py)

**Features**:
- 128×128×64 GEMM with Tensor Cores
- TMA loads for input matrices
- Single-stage computation
- Warpgroup matrix multiply

**Key Operations**:
1. GPU memory allocation
2. TMA descriptor creation
3. Kernel launch with 128 threads
4. TMA async loads to shared memory
5. WGMMA Tensor Core operation
6. Result store and copy back

### Ch4.py - Multistage GEMM

**File**: [mlir/test/Examples/NVGPU/Ch4.py](../../../mlir/test/Examples/NVGPU/Ch4.py)

**Features**:
- Software pipelining with 7 stages
- Overlapped TMA loads and MMA operations
- Multiple memory barriers for staging
- Block-level parallelization

**Key Techniques**:
- Prologue: Fill pipeline stages
- Mainloop: Overlapped compute and memory
- Epilogue: Store results
- Phase parity for barrier reuse

### Ch5.py - Warp-Specialized GEMM

**File**: [mlir/test/Examples/NVGPU/Ch5.py](../../../mlir/test/Examples/NVGPU/Ch5.py)

**Features**:
- 256 threads: 2 warpgroups (producer + consumer)
- Producer warpgroup: TMA loads
- Consumer warpgroup: MMA operations
- Separate control flow per warpgroup
- Register allocation tuning

**Key Techniques**:
- Warpgroup context managers
- Producer-consumer synchronization
- Dual barrier sets (TMA + MMA)
- Asymmetric register allocation

---

## File Locations

### Source Files

| File | Location | Description |
|------|----------|-------------|
| **nvdsl.py** | [mlir/test/Examples/NVGPU/tools/nvdsl.py](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py) | NVDSL DSL implementation (460 lines) |
| **nvgpucompiler.py** | [mlir/test/Examples/NVGPU/tools/nvgpucompiler.py](../../../mlir/test/Examples/NVGPU/tools/nvgpucompiler.py) | Compilation pipeline wrapper (45 lines) |
| **Ch3.py** | [mlir/test/Examples/NVGPU/Ch3.py](../../../mlir/test/Examples/NVGPU/Ch3.py) | Basic GEMM example (130 lines) |
| **Ch4.py** | [mlir/test/Examples/NVGPU/Ch4.py](../../../mlir/test/Examples/NVGPU/Ch4.py) | Multistage GEMM (324 lines) |
| **Ch5.py** | [mlir/test/Examples/NVGPU/Ch5.py](../../../mlir/test/Examples/NVGPU/Ch5.py) | Warp-specialized GEMM (322 lines) |

### MLIR Infrastructure

| Component | Location | Description |
|-----------|----------|-------------|
| **Python Bindings** | [mlir/python/mlir/dialects/](../../../mlir/python/mlir/dialects/) | Python wrappers for dialects |
| **C API** | [mlir/include/mlir-c/](../../../mlir/include/mlir-c/) | C API headers |
| **C++ Dialects** | [mlir/lib/Dialect/](../../../mlir/lib/Dialect/) | Dialect implementations |
| **Conversions** | [mlir/lib/Conversion/](../../../mlir/lib/Conversion/) | Dialect lowering passes |
| **GPU Pipeline** | [mlir/lib/Dialect/GPU/Pipelines/](../../../mlir/lib/Dialect/GPU/Pipelines/) | GPU lowering pipeline |

---

## Call Path Examples

### Simple Operation: `tidx = gpu.thread_id(gpu.Dimension.x)`

```
Python:
  gpu.thread_id(gpu.Dimension.x)
    ↓
Python Binding:
  _mlirDialectsGPU.thread_id(dimension)  [auto-generated]
    ↓
C++ Builder:
  gpu::ThreadIdOp::build(builder, result, gpu::Dimension::x)
    ↓
MLIR Operation:
  %tidx = gpu.thread_id x : index
    ↓
After gpu-to-nvvm:
  %tidx = nvvm.read.ptx.sreg.tid.x : i32
    ↓
PTX:
  mov.u32 %r0, %tid.x;
```

### Complex Operation: `D += A @ B`

```
Python:
  D += A @ B  # WGMMAMatrix objects
    ↓
Operator Overloading:
  A.__matmul__(B)  → returns [desc_a, desc_b]
  D.__iadd__([desc_a, desc_b])
    ↓
Python Binding:
  nvgpu.WarpgroupMmaOp(...)
    ↓
C++ Builder:
  nvgpu::WarpgroupMmaOp::build(...)
    ↓
MLIR Operation:
  %d_new = nvgpu.warpgroup.mma %desc_a, %desc_b, %d_old {transposeB = true}
    ↓
After convert-nvgpu-to-nvvm:
  %d_new = nvvm.wgmma.mma_async.m64n128k16.f32.f16.f16 ...
    ↓
After convert-nvvm-to-llvm:
  %d_new = llvm.inline_asm "wgmma.mma_async.m64n128k16.f32.f16.f16 ..."
    ↓
PTX:
  wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {...};
```

---

## Type System Flow

### NumPy Array → MLIR Type → GPU Memory

```
NumPy Array:
  dtype: float16
  shape: (128, 64)
  data: contiguous buffer
    ↓
Python Type Detection:
  get_mlir_ty(array) → memref<128x64xf16>
    ↓
MLIR Type:
  ir.MemRefType.get([128, 64], T.f16())
    ↓
C API:
  mlirMemRefTypeGet(ctx, element_type=f16, rank=2, shape=[128,64])
    ↓
C++ Type:
  MemRefType::get({128, 64}, Float16Type::get(ctx))
    ↓
Runtime Descriptor:
  MemRefDescriptor {
    allocated: ptr to NumPy buffer
    aligned: ptr to NumPy buffer
    offset: 0
    sizes: [128, 64]
    strides: [64, 1]
  }
    ↓
GPU Allocation:
  gpu.alloc() → cuMemAlloc(128 * 64 * 2 = 16384 bytes)
    ↓
GPU Memory:
  Device pointer: 0xdeadbeef0000
  Size: 16384 bytes
  Layout: row-major, stride 64
```

---

## Performance Features

### TMA (Tensor Memory Accelerator)

**Benefits**:
- Hardware-accelerated memory transfers
- Automatic swizzling for bank conflict avoidance
- Asynchronous execution overlapped with compute
- Multi-dimensional addressing in hardware

**Configuration**:
```python
TMA([128, 64], memref_ty,
    swizzle=SWIZZLE_128B,       # 128-byte swizzling
    l2promo=L2PROMO_NONE,       # L2 cache hints
    oob=OOB_ZERO,               # Out-of-bounds handling
    interleave=INTERLEAVE_NONE) # Interleaving pattern
```

### WGMMA (Warpgroup Matrix Multiply-Accumulate)

**Benefits**:
- 128 threads cooperate on matrix multiply
- Tensor Core acceleration (Hopper architecture)
- Asynchronous execution
- High throughput: 1024 FLOPs per cycle

**Shapes Supported**:
- M: 64
- N: 8, 16, 24, 32, 40, 48, 56, 64, ..., 256
- K: 8, 16, 24, 32

### Mbarriers (Memory Barriers)

**Benefits**:
- Fine-grained synchronization
- Transaction counting for async operations
- Phase parity for pipeline reuse
- Minimal overhead

**Usage Patterns**:
- Pipeline staging
- Producer-consumer synchronization
- Async operation completion tracking

---

## Debugging Tips

### View Generated IR

Add to nvdsl.py:
```python
with open("debug.mlir", "w") as f:
    print(module, file=f)
```

### Enable MLIR Debugging

```bash
export MLIR_ENABLE_DUMP=1
```

### View PTX

```bash
cuobjdump --dump-ptx kernel.cubin
```

### Profile Execution

```bash
# System-wide profiling
nsys profile --stats=true python Ch3.py

# Kernel-level profiling
ncu --set full --export report python Ch3.py
```

---

## Related Documentation

### MLIR Documentation

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
- [NVGPU Dialect](https://mlir.llvm.org/docs/Dialects/NVGPU/)
- [NVVM Dialect](https://mlir.llvm.org/docs/Dialects/NVVM/)

### NVIDIA Documentation

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
- [Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)

---

## Contributing

To extend NVDSL:

1. **Add new operations**: Create wrapper classes in nvdsl.py
2. **Extend decorators**: Modify `@mlir_func` or `@mlir_gpu_launch`
3. **Add dialect ops**: Contribute to MLIR upstream
4. **Optimize pipeline**: Modify gpu-lower-to-nvvm-pipeline

---

## Summary

NVDSL provides a **complete stack** for GPU programming in Python:

- ✅ **High-level**: Pythonic API with operator overloading
- ✅ **Performant**: JIT compilation to optimized CUDA code
- ✅ **Modern**: Supports latest Hopper features (TMA, WGMMA)
- ✅ **Composable**: Build on MLIR's extensible infrastructure
- ✅ **Debuggable**: Access to IR at all levels

**Read the docs**: Start with [00_overview.md](00_overview.md) for system architecture!
