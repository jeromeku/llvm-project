# NVDSL: Complete System Architecture

## Overview

This document provides a **literate code walkthrough** of the NVDSL (NVIDIA DSL) system - a Python DSL that enables writing high-performance CUDA kernels using Python syntax that compiles down to executable GPU code through MLIR.

We trace the complete journey from:
```
Python Code → MLIR IR → LLVM IR → PTX → CUDA Binary → Executable
```

## What is NVDSL?

NVDSL is a Domain-Specific Language embedded in Python that allows you to write GPU kernels using:
- High-level Python syntax
- MLIR dialect operations (GPU, NVGPU, NVVM)
- Automatic compilation to executable CUDA code

### Key Features

1. **Python-native DSL**: Write GPU kernels in Python with operator overloading
2. **MLIR-powered**: Leverages MLIR's GPU and NVGPU dialects
3. **JIT Compilation**: Compiles and executes at runtime
4. **Tensor Core Support**: First-class support for Hopper architecture features (TMA, WGMMA)

## Example Files Analyzed

We'll trace three progressively complex examples:

| File | Description | Key Features |
|------|-------------|--------------|
| **Ch3.py** | Basic GEMM 128×128×64 | TMA load, Tensor Core GEMM, basic warpgroup operations |
| **Ch4.py** | Multistage GEMM | Software pipelining, multiple stages, overlapped TMA/MMA |
| **Ch5.py** | Warp-Specialized GEMM | Producer/consumer warpgroups, advanced synchronization |

### File Locations

```
mlir/test/Examples/NVGPU/
├── Ch3.py                          ← Simple GEMM example
├── Ch4.py                          ← Multistage GEMM
├── Ch5.py                          ← Warp-specialized GEMM
└── tools/
    ├── nvdsl.py                    ← NVDSL Python DSL implementation
    └── nvgpucompiler.py            ← Compilation pipeline
```

## System Architecture Layers

The NVDSL system consists of 5 distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: User Python Code (Ch3.py, Ch4.py, Ch5.py)         │
│           - NumPy arrays as inputs                          │
│           - Python functions decorated with @NVDSL          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: NVDSL Python DSL (nvdsl.py)                        │
│           - High-level Python API (TMA, Mbarriers, etc.)    │
│           - Operator overloading for Python ops             │
│           - Decorators: @mlir_func, @mlir_gpu_launch        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: MLIR Python Bindings (mlir.dialects.*)             │
│           - Python wrapper around C API                     │
│           - IR building: ir.Module, ir.Context              │
│           - Dialect ops: gpu.*, nvgpu.*, nvvm.*             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: MLIR C API (mlir-c/IR.h, mlir-c/Dialect/*.h)      │
│           - C ABI for Python bindings                       │
│           - Opaque handles: MlirContext, MlirType, etc.     │
│           - Generated from TableGen                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: MLIR C++ Core (mlir/IR/*, mlir/Dialect/*)         │
│           - Operation definitions (TableGen .td files)      │
│           - Type system, Attributes, Builders               │
│           - Pass infrastructure, Conversions                │
└─────────────────────────────────────────────────────────────┘
```

### Layer Communication

Each layer communicates with the layer below through well-defined interfaces:

1. **Layer 5 → Layer 4**: Python function calls
2. **Layer 4 → Layer 3**: Python module imports (`from mlir.dialects import ...`)
3. **Layer 3 → Layer 2**: `ctypes`/`cffi` calls to shared library (`.so`)
4. **Layer 2 → Layer 1**: C++ function calls (ABI-stable interface)

## Compilation Pipeline

After IR construction, the code goes through:

```
MLIR IR (GPU Dialect)
    ↓ [gpu-lower-to-nvvm-pipeline]
MLIR IR (NVVM Dialect)
    ↓ [translate-to-llvmir]
LLVM IR
    ↓ [LLVM Optimizer]
PTX Assembly
    ↓ [ptxas - NVIDIA PTX Assembler]
CUDA Binary (.cubin)
    ↓ [GPU Runtime Loader]
Executable GPU Code
```

## Document Structure

This documentation is organized as follows:

| Document | Content |
|----------|---------|
| **00_overview.md** (this file) | System architecture overview |
| **01_python_user_code.md** | Layer 5: User Python code walkthrough |
| **02_nvdsl_layer.md** | Layer 4: NVDSL DSL implementation |
| **03_python_bindings.md** | Layer 3: MLIR Python bindings |
| **04_c_api.md** | Layer 2: MLIR C API |
| **05_cpp_implementation.md** | Layer 1: MLIR C++ core |
| **06_compilation_pipeline.md** | End-to-end compilation to executable |
| **07_example_traces.md** | Complete traces for Ch3, Ch4, Ch5 |

## Key Concepts

### IR Building

MLIR IR is built **programmatically** in Python:

```python
# Python code
with ir.Context():
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        func_op = func.FuncOp("kernel", ...)
        with ir.InsertionPoint(func_op.body):
            # Build operations here
            gpu.thread_id(gpu.Dimension.x)
```

This Python code calls into C++ to construct the IR tree.

### Decorators

NVDSL uses decorators to transform Python functions:

```python
@NVDSL.mlir_func                    # Outer decorator: Creates MLIR function
def gemm(a, b, d):
    @NVDSL.mlir_gpu_launch(...)     # Inner decorator: Creates GPU kernel
    def kernel():
        # Kernel code here
        pass
```

The decorators:
1. Intercept function execution
2. Build MLIR IR instead of executing Python
3. Compile IR to executable
4. Execute and return results

### Type Conversion

Data flows through multiple type systems:

```
NumPy ndarray (Python)
    ↓
memref<128x64xf16> (MLIR Type)
    ↓
MlirType (C API Handle)
    ↓
MemRefType (C++ Object)
    ↓
LLVM struct (LLVM IR)
    ↓
GPU memory pointer (Runtime)
```

## Navigation

Each subsequent document provides:
- **Clickable file references** (e.g., [nvdsl.py:308](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L308))
- **Code snippets** with line numbers
- **Call graphs** showing function flow
- **Data flow diagrams** showing transformations
- **Table mappings** between layers

---

**Next**: [01_python_user_code.md](01_python_user_code.md) - Deep dive into the user-facing Python code
