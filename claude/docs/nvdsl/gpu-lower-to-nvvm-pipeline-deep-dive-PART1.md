# GPU Lower to NVVM Pipeline - Deep Dive (Part 1)

## Overview

The `gpu-lower-to-nvvm-pipeline` is a compound pass pipeline that transforms high-level GPU operations down to LLVM IR suitable for NVIDIA GPUs. This document traces each pass implementation line-by-line, showing before/after transformations.

### Pipeline Structure

```
gpu-lower-to-nvvm-pipeline expands to 19 passes:

1. convert-nvgpu-to-nvvm           ← NVGPU dialect → NVVM dialect
2. gpu-kernel-outlining             ← Extract GPU kernels into separate modules
3. convert-vector-to-scf            ← Vector ops → SCF loops
4. convert-scf-to-cf                ← SCF (Structured Control Flow) → CF (Control Flow)
5. convert-nvvm-to-llvm             ← NVVM intrinsics → LLVM intrinsics
6. convert-func-to-llvm             ← func dialect → LLVM dialect
7. expand-strided-metadata          ← Expand memref descriptors
8. nvvm-attach-target               ← Attach NVVM compilation attributes
9. lower-affine                     ← Affine dialect → standard ops
10. convert-arith-to-llvm           ← Arithmetic → LLVM
11. convert-index-to-llvm           ← Index type → i64
12. canonicalize                    ← Simplification/cleanup
13. cse                             ← Common Subexpression Elimination
14. gpu.module(convert-gpu-to-nvvm) ← GPU ops → NVVM (inside GPU module)
15. gpu.module(canonicalize)        ← Cleanup inside GPU module
16. gpu.module(cse)                 ← CSE inside GPU module
17. gpu.module(reconcile-unrealized-casts) ← Cleanup casts
18. gpu-to-llvm                     ← GPU runtime → LLVM calls
19. gpu-module-to-binary            ← Compile GPU module → PTX/cubin
20. convert-math-to-llvm            ← Math dialect → LLVM
21. canonicalize                    ← Final cleanup
22. cse                             ← Final CSE
23. reconcile-unrealized-casts      ← Final cast cleanup
```

### Goals

1. **Lower GPU-specific operations** to NVVM/LLVM
2. **Outline GPU kernels** into separate modules
3. **Convert high-level abstractions** (nvgpu, vector, scf) to low-level IR
4. **Compile GPU modules** to PTX assembly
5. **Generate host code** to launch kernels

---

## MLIR Operation Structure Primer

Before diving into passes, let's understand MLIR operation anatomy:

### Operation Structure

```mlir
%result = dialect.operation(%operand1, %operand2) {attribute = value} : (type1, type2) -> result_type
```

**Components**:
- `%result` - SSA value produced
- `dialect.operation` - Operation name (dialect.opname)
- `(%operand1, %operand2)` - Input operands (SSA values)
- `{attribute = value}` - Attributes (compile-time constants)
- `: (type1, type2) -> result_type` - Type signature

### Example from our IR

```mlir
%4 = nvgpu.tma.create.descriptor %cast box[%c128, %c64] : memref<*xf16>
     -> <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, ...>
```

**Breaking it down**:
- `%4` - Result: TMA descriptor handle
- `nvgpu.tma.create.descriptor` - Op: Create Tensor Memory Accelerator descriptor
- `%cast` - Operand 1: memref to create descriptor for
- `box[%c128, %c64]` - Attribute: box dimensions
- `: memref<*xf16>` - Input type
- `-> <tensor = ...>` - Result type (NVGPU TMA descriptor)

### Operation Traits

**Traits** define compile-time properties:

```cpp
// From TableGen (.td files)
def SomeOp : Op<...> {
  let traits = [Pure, SameOperandsAndResultType];
}
```

Common traits:
- `Pure` - No side effects (can be CSE'd, moved, eliminated)
- `SameOperandsAndResultType` - All operands and results have same type
- `Terminator` - Ends a basic block
- `IsolatedFromAbove` - Can't reference SSA values from parent regions

### Operation Interfaces

**Interfaces** define runtime behavior that passes can query:

```cpp
def SomeOp : Op<...> {
  let interfaces = [MemoryEffectsOpInterface, InferTypeOpInterface];
}
```

Common interfaces:
- `MemoryEffectsOpInterface` - Describes memory reads/writes
- `InferTypeOpInterface` - Can infer result types from operands
- `DestinationStyleOpInterface` - Has explicit destination operands
- `LoopLikeOpInterface` - Represents a loop

### Regions and Blocks

Operations can contain **regions** which contain **blocks**:

```mlir
scf.if %cond -> (i32) {
  // ^block1:  (implicit block label)
  %result = arith.constant 1 : i32
  scf.yield %result : i32
} else {
  // ^block2:
  %result = arith.constant 2 : i32
  scf.yield %result : i32
}
```

**Structure**:
- `scf.if` has 2 regions (then-region, else-region)
- Each region has 1 block
- Blocks end with terminator (`scf.yield`)

---

## Pass 0: Input IR (module_before.mlir)

**File**: `/home/jeromeku/llvm-project/mlir/test/Examples/NVGPU/module_before.mlir`

### Key Operations Present

1. **func.func** - Function definition (host code)
2. **gpu.wait/alloc/memcpy** - Async GPU memory operations
3. **nvgpu.tma.create.descriptor** - Create TMA descriptor for tensor loads
4. **gpu.launch** - Launch GPU kernel inline
5. **nvgpu.mbarrier.*** - Barrier operations
6. **nvgpu.tma.async.load** - Async tensor loads via TMA
7. **nvgpu.warpgroup.mma** - Warpgroup matrix multiply-accumulate
8. **nvgpu.warpgroup.mma.store** - Store MMA results

### Operation Anatomy Examples

#### nvgpu.tma.create.descriptor

```mlir
%4 = nvgpu.tma.create.descriptor %cast box[%c128, %c64] : memref<*xf16>
     -> <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b,
         l2promo = none, oob = zero, interleave = none>
```

**What it does**: Creates a Tensor Memory Accelerator descriptor for efficient data movement.

**Operands**:
- `%cast` - memref<*xf16> (unranked memref)

**Attributes**:
- `box[%c128, %c64]` - Dimensions of the box to load

**Result type**: `!nvgpu.tma.descriptor<...>` - Custom NVGPU type

**Implementation**:
- File: `mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td`
- C++ Class: `nvgpu::TmaCreateDescriptorOp`

#### gpu.launch

```mlir
gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1_7, %arg11 = %c1_8)
           threads(%arg6, %arg7, %arg8) in (%arg12 = %c128_9, %arg13 = %c1_10, %arg14 = %c1_11)
           dynamic_shared_memory_size %c32768_i32 {
  // Kernel body
  gpu.terminator
}
```

**What it does**: Defines a GPU kernel launch region inline.

**Operands**:
- Block/thread index operands: `%arg3-%arg8`
- Grid/block size operands: `%arg9-%arg14` (bound to constants)
- Dynamic shared memory size: `%c32768_i32`

**Regions**:
- 1 region containing kernel body
- Must end with `gpu.terminator`

**Traits**:
- `IsolatedFromAbove` - Kernel can't reference parent SSA values directly
- `AutomaticAllocationScope` - Handles memory allocation scoping

#### nvgpu.warpgroup.mma

```mlir
%17 = nvgpu.warpgroup.mma %15, %16, %14 {transposeB} :
      <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>,
      <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>,
      <fragmented = vector<128x128xf32>>
      -> <fragmented = vector<128x128xf32>>
```

**What it does**: Performs warpgroup-wide matrix multiply-accumulate using tensor cores.

**Operands**:
- `%15` - Descriptor for matrix A (128x64xf16)
- `%16` - Descriptor for matrix B (64x128xf16)
- `%14` - Accumulator (128x128xf32)

**Attributes**:
- `{transposeB}` - Transpose matrix B during multiplication

**Result**:
- `<fragmented = vector<128x128xf32>>` - Updated accumulator

**Hardware mapping**: Maps to WGMMA.MMA PTX instruction on Hopper (sm_90a)

---

## Pass 1: convert-nvgpu-to-nvvm

**Implementation**: `/home/jeromeku/llvm-project/mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp`

**Purpose**: Lower NVGPU dialect operations to NVVM dialect (NVIDIA's LLVM IR dialect).

### Pass Registration

```cpp
// File: mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp
namespace {
struct ConvertNVGPUToNVVMPass
    : public impl::ConvertNVGPUToNVVMBase<ConvertNVGPUToNVVMPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext());

    // Populate conversion patterns
    populateNVGPUToNVVMConversionPatterns(converter, patterns);

    // Apply patterns
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
}
```

### Key Conversion Patterns

Let me read the actual implementation to show you the patterns:

This is getting quite large. Let me create this as a multi-part document. I'll focus on the most important GPU-specific passes with full detail.
