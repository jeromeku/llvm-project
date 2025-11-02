# Detailed Pass Trace: convert-gpu-to-nvvm (Pass 14)

## Overview

The `convert-gpu-to-nvvm` pass runs **inside** `gpu.module` regions, converting generic GPU dialect operations to NVVM (NVIDIA LLVM) intrinsics. This is the device-side equivalent of LLVM lowering.

**Source**: `mlir/lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp`

**Key Difference from Pass 0**:
- **Pass 0** (`convert-nvgpu-to-nvvm`): High-level NVGPU ops → NVVM (before kernel outlining)
- **Pass 14** (`convert-gpu-to-nvvm`): Generic GPU ops → NVVM (inside `gpu.module` after outlining)

**What it converts**:
```
gpu.thread_id x     →  nvvm.read.ptx.sreg.tid.x
gpu.block_id x      →  nvvm.read.ptx.sreg.ctaid.x
gpu.block_dim x     →  nvvm.read.ptx.sreg.ntid.x
gpu.grid_dim x      →  nvvm.read.ptx.sreg.nctaid.x
gpu.barrier         →  nvvm.barrier0
gpu.return          →  llvm.return
```

---

## Part 1: Nested Pass Manager Execution

### Why Nested Pass Manager?

**Pipeline Structure**:
```python
"builtin.module(                       # Top-level module
  gpu-kernel-outlining,                # Creates gpu.module
  ...,
  gpu.module(                          # Runs ONLY on gpu.module!
    convert-gpu-to-nvvm,               # ← Pass 14
    canonicalize,
    cse,
    reconcile-unrealized-casts
  ),
  ...,
  gpu-module-to-binary                 # Compiles gpu.module
)"
```

**Pass Manager Nesting**:
```cpp
// Top-level pass manager
PassManager pm(module.getContext());

// Add pass that runs on builtin.module
pm.addPass(createGpuKernelOutliningPass());

// Add NESTED pass manager for gpu.module operations
OpPassManager &gpuModulePM = pm.nest<gpu::GPUModuleOp>();
gpuModulePM.addPass(createConvertGpuOpsToNVVMOps());
gpuModulePM.addPass(createCanonicalizerPass());
gpuModulePM.addPass(createCSEPass());
```

**Why nested?**
- Different operations need different passes
- `gpu.module` is device code (NVVM lowering)
- `builtin.module` is host code (LLVM lowering)
- Isolation: Passes run only on relevant code

---

## Part 2: Pass Infrastructure

### Pass Class

```cpp
// Line 596-650+: Pattern population
void mlir::populateGpuToNVVMConversionPatterns(
    const LLVMTypeConverter &converter,
    RewritePatternSet &patterns,
    PatternBenefit benefit) {

  // Thread/Block/Grid index operations
  patterns.add<
      gpu::index_lowering::OpLowering<
        gpu::ThreadIdOp,               // Source
        NVVM::ThreadIdXOp,             // Target X
        NVVM::ThreadIdYOp,             // Target Y
        NVVM::ThreadIdZOp              // Target Z
      >
  >(converter, IndexKind::Block, IntrType::Id, benefit);

  patterns.add<
      gpu::index_lowering::OpLowering<
        gpu::BlockIdOp,
        NVVM::BlockIdXOp,
        NVVM::BlockIdYOp,
        NVVM::BlockIdZOp
      >
  >(converter, IndexKind::Grid, IntrType::Id, benefit);

  // ... similar for BlockDim, GridDim, etc.

  patterns.add<
      GPULaneIdOpToNVVM,
      GPUShuffleOpLowering,
      GPUReturnOpLowering
  >(converter, benefit);

  patterns.add<GPUDynamicSharedMemoryOpLowering>(
      converter, NVVM::kSharedMemoryAlignmentBit, benefit);

  patterns.add<GPUFuncOpLowering>(
      converter,
      GPUFuncOpLoweringOptions{
          /*allocaAddrSpace=*/0,  // Private memory → alloca
          /*workgroupAddrSpace=*/3  // Shared memory address space
      },
      benefit);
}
```

**Pattern Registration Strategy**:
- Template-based patterns for similar operations (thread_id, block_id, etc.)
- Dimension-specific lowering (x, y, z)
- Configuration via options (address spaces, alignment)

---

## Part 3: Index Lowering Pattern

### Template Pattern for Index Operations

```cpp
// Simplified from GPUCommon/IndexIntrinsicsOpLowering.h
template<typename SourceOp, typename XOp, typename YOp, typename ZOp>
struct OpLowering : public ConvertOpToLLVMPattern<SourceOp> {
  OpLowering(LLVMTypeConverter &converter,
             IndexKind kind,    // Block, Grid, Other
             IntrType type,     // Id, Dim
             PatternBenefit benefit)
    : ConvertOpToLLVMPattern<SourceOp>(converter, benefit),
      kind(kind), type(type) {}

  LogicalResult matchAndRewrite(
      SourceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Get dimension (x, y, or z)
    gpu::Dimension dim = op.getDimension();

    // Choose appropriate NVVM operation
    Type resultType = rewriter.getI32Type();
    Operation *nvvmOp = nullptr;

    switch (dim) {
    case gpu::Dimension::x:
      nvvmOp = rewriter.create<XOp>(op.getLoc(), resultType);
      break;
    case gpu::Dimension::y:
      nvvmOp = rewriter.create<YOp>(op.getLoc(), resultType);
      break;
    case gpu::Dimension::z:
      nvvmOp = rewriter.create<ZOp>(op.getLoc(), resultType);
      break;
    }

    Value result = nvvmOp->getResult(0);

    // Cast to index if needed
    if (op.getType().isa<IndexType>())
      result = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), op.getType(), result);

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  IndexKind kind;
  IntrType type;
};
```

**Why Template?**
- Code reuse: All index ops have same pattern
- Type safety: Compiler checks dimension ops exist
- Maintainability: Add new index op = instantiate template

**Example Instantiation**:
```cpp
// gpu.thread_id → nvvm.read.ptx.sreg.tid.{x,y,z}
using ThreadIdLowering = OpLowering<
    gpu::ThreadIdOp,        // Source operation
    NVVM::ThreadIdXOp,      // X dimension target
    NVVM::ThreadIdYOp,      // Y dimension target
    NVVM::ThreadIdZOp       // Z dimension target
>;
```

---

## Part 4: Detailed Pattern Traces

### Pattern 1: gpu.thread_id → nvvm.read.ptx.sreg.tid.x

**Input IR** (after kernel outlining):
```mlir
gpu.func @kernel(...) kernel {
  %thread_id_x = gpu.thread_id x
  // ... use %thread_id_x ...
  gpu.return
}
```

**Pattern Execution**:
```cpp
// Matched by OpLowering<gpu::ThreadIdOp, ...>
matchAndRewrite(gpu::ThreadIdOp op, ...) {
  // Step 1: Get dimension
  gpu::Dimension dim = op.getDimension();  // Returns x

  // Step 2: Choose NVVM op based on dimension
  Type i32Type = rewriter.getI32Type();
  Value nvvmRead = rewriter.create<NVVM::ThreadIdXOp>(
      op.getLoc(), i32Type);

  // Step 3: NVVM ops return i32, GPU ops return index
  Type resultType = op.getType();  // index type
  Value result = nvvmRead;

  if (resultType.isa<IndexType>()) {
    result = rewriter.create<arith::IndexCastOp>(
        op.getLoc(), resultType, nvvmRead);
  }

  // Step 4: Replace original operation
  rewriter.replaceOp(op, result);
  return success();
}
```

**Output IR** (inside gpu.module):
```mlir
gpu.func @kernel(...) kernel {
  %0 = nvvm.read.ptx.sreg.tid.x : i32
  %thread_id_x = arith.index_cast %0 : i32 to index
  // ... use %thread_id_x ...
  gpu.return
}
```

**NVVM Intrinsic**: `nvvm.read.ptx.sreg.tid.x`
- Reads PTX special register `%tid.x`
- Returns thread index within block
- Hardware instruction: `mov.u32 %r0, %tid.x;`

**Why Index Cast?**
- MLIR uses `index` type (platform-dependent width)
- NVVM uses `i32` (GPU threads are 32-bit indexed)
- Must convert: i32 → index

---

### Pattern 2: gpu.barrier → nvvm.barrier0

**Input IR**:
```mlir
gpu.func @kernel(...) kernel {
  // ... some computation ...
  gpu.barrier
  // ... more computation ...
  gpu.return
}
```

**Pattern**: Direct replacement (no operands)

```cpp
struct GPUBarrierLowering : public ConvertOpToLLVMPattern<gpu::BarrierOp> {
  using ConvertOpToLLVMPattern<gpu::BarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      gpu::BarrierOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Simple replacement: gpu.barrier → nvvm.barrier0
    rewriter.replaceOpWithNewOp<NVVM::Barrier0Op>(op);
    return success();
  }
};
```

**Output IR**:
```mlir
gpu.func @kernel(...) kernel {
  // ... some computation ...
  nvvm.barrier0
  // ... more computation ...
  gpu.return
}
```

**NVVM Intrinsic**: `nvvm.barrier0`
- Synchronizes all threads in a thread block
- Hardware instruction: `bar.sync 0;` (barrier #0)
- Waits until all threads in block reach this point

**PTX Semantics**:
```ptx
bar.sync 0;   // Synchronize on barrier 0
              // All threads must execute this instruction
              // Undefined behavior if not all threads reach it
```

---

### Pattern 3: gpu.dynamic.shared.memory → NVVM address cast

**Input IR**:
```mlir
gpu.func @kernel(...) kernel {
  %shmem = gpu.dynamic.shared.memory : memref<?xi8, #gpu.address_space<workgroup>>
  // Cast to actual type
  %typed = memref.view %shmem[...](...) : memref<?xi8> to memref<128x64xf16, 3>
  gpu.return
}
```

**Pattern**: Get pointer to dynamic shared memory base

```cpp
// Line 640-641: GPUDynamicSharedMemoryOpLowering
struct GPUDynamicSharedMemoryOpLowering
    : public ConvertOpToLLVMPattern<gpu::DynamicSharedMemoryOp> {

  GPUDynamicSharedMemoryOpLowering(
      const LLVMTypeConverter &converter,
      unsigned alignmentBit,  // NVVM::kSharedMemoryAlignmentBit
      PatternBenefit benefit)
    : ConvertOpToLLVMPattern(converter, benefit),
      alignmentBit(alignmentBit) {}

  LogicalResult matchAndRewrite(
      gpu::DynamicSharedMemoryOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Step 1: Get base address of dynamic shared memory
    // This is a magic NVVM intrinsic that returns the base pointer
    Type llvmPtrType = LLVM::LLVMPointerType::get(
        rewriter.getContext(), 3);  // Address space 3 = shared

    // Create extern shared variable reference
    auto global = rewriter.create<LLVM::AddressOfOp>(
        op.getLoc(), llvmPtrType,
        NVVM::kDynamicSharedMemorySymbol);  // "__dynamic_shmem__"

    // Step 2: Convert to memref descriptor
    MemRefType memrefType = op.getType().cast<MemRefType>();
    Type convertedType = typeConverter->convertType(memrefType);

    Value memrefDesc = MemRefDescriptor::undef(
        rewriter, op.getLoc(), convertedType);

    // Set allocated pointer
    memrefDesc = MemRefDescriptor::setAllocatedPtr(
        rewriter, op.getLoc(), memrefDesc, convertedType, global);

    // Set aligned pointer (same as allocated for shmem)
    memrefDesc = MemRefDescriptor::setAlignedPtr(
        rewriter, op.getLoc(), memrefDesc, convertedType, global);

    // Set offset to 0
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(), 0);
    memrefDesc = MemRefDescriptor::setOffset(
        rewriter, op.getLoc(), memrefDesc, convertedType, zero);

    // Set size (dynamic, from 3rd operand of gpu.launch_func)
    // Note: Size already passed to GPU runtime

    rewriter.replaceOp(op, memrefDesc);
    return success();
  }

private:
  unsigned alignmentBit;
};
```

**Output IR**:
```mlir
gpu.func @kernel(...) kernel {
  // Get address of dynamic shared memory
  %0 = llvm.mlir.addressof @__dynamic_shmem__ : !llvm.ptr<3>

  // Build memref descriptor
  %1 = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1xi64>, array<1xi64>)>
  %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<...>  // allocated
  %3 = llvm.insertvalue %0, %2[1] : !llvm.struct<...>  // aligned
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %4 = llvm.insertvalue %c0, %3[2] : !llvm.struct<...>  // offset
  // ... (sizes and strides set elsewhere)

  %shmem = builtin.unrealized_conversion_cast %4 : ... to memref<?xi8, 3>
  gpu.return
}
```

**PTX Generated**:
```ptx
.extern .shared .align 16 .b8 __dynamic_shmem__[];

// In kernel:
mov.u64 %rd1, __dynamic_shmem__;  // Get base address
```

**Why "Dynamic"?**
- Size determined at runtime (3rd argument to `cudaLaunchKernel`)
- Allocated by CUDA runtime before kernel launch
- Shared by entire thread block
- Fast on-chip SRAM (100x faster than global memory)

---

### Pattern 4: gpu.func → llvm.func (with NVVM attributes)

**Input IR**:
```mlir
gpu.func @kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, ...) kernel {
  // ... kernel body ...
  gpu.return
}
```

**Pattern**: Convert function signature and add NVVM attributes

```cpp
// Line 646-650: GPUFuncOpLowering
struct GPUFuncOpLowering : public ConvertOpToLLVMPattern<gpu::GPUFuncOp> {
  GPUFuncOpLowering(const LLVMTypeConverter &converter,
                    GPUFuncOpLoweringOptions options,
                    PatternBenefit benefit)
    : ConvertOpToLLVMPattern(converter, benefit),
      options(options) {}

  LogicalResult matchAndRewrite(
      gpu::GPUFuncOp gpuFunc, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = gpuFunc.getLoc();

    // Step 1: Convert function type
    TypeConverter::SignatureConversion signatureConversion(
        gpuFunc.getNumArguments());

    for (auto [i, argType] : llvm::enumerate(gpuFunc.getArgumentTypes())) {
      Type convertedType = typeConverter->convertType(argType);
      signatureConversion.addInputs(i, convertedType);
    }

    // Step 2: Create LLVM function
    auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(
        loc,
        gpuFunc.getName(),
        LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(rewriter.getContext()),
            signatureConversion.getConvertedTypes()));

    // Step 3: Add NVVM kernel attribute
    if (gpuFunc->getAttr(gpu::GPUDialect::getKernelFuncAttrName())) {
      llvmFunc->setAttr(NVVM::NVVMDialect::getKernelFuncAttrName(),
                        rewriter.getUnitAttr());
    }

    // Step 4: Copy function body
    rewriter.inlineRegionBefore(gpuFunc.getBody(), llvmFunc.getBody(),
                                llvmFunc.end());

    // Step 5: Convert block arguments
    if (failed(rewriter.convertRegionTypes(&llvmFunc.getBody(),
                                          *typeConverter,
                                          &signatureConversion)))
      return failure();

    rewriter.replaceOp(gpuFunc, llvmFunc);
    return success();
  }

private:
  GPUFuncOpLoweringOptions options;
};
```

**Output IR**:
```mlir
llvm.func @kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, ...)
    attributes {nvvm.kernel} {
  // ... kernel body (with LLVM/NVVM ops) ...
  llvm.return
}
```

**NVVM Attribute**: `nvvm.kernel`
- Marks function as kernel entry point (not device function)
- Changes PTX generation:
  - Kernel: `.visible .entry kernel_name(...)`
  - Device function: `.visible .func device_func_name(...)`
- Affects calling convention

---

### Pattern 5: gpu.return → llvm.return

**Input IR**:
```mlir
gpu.func @kernel(...) kernel {
  // ... operations ...
  gpu.return
}
```

**Pattern**: Simple terminator replacement

```cpp
struct GPUReturnOpLowering : public ConvertOpToLLVMPattern<gpu::ReturnOp> {
  using ConvertOpToLLVMPattern<gpu::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      gpu::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // GPU kernels return void
    assert(op.getOperands().empty() && "GPU kernels cannot return values");

    // Replace with LLVM return
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange{});
    return success();
  }
};
```

**Output IR**:
```mlir
llvm.func @kernel(...) attributes {nvvm.kernel} {
  // ... operations ...
  llvm.return  // void return
}
```

**PTX Generated**:
```ptx
ret;  // Return from kernel
```

---

## Part 5: Complete Transformation Example

### Input (after Pass 1 - kernel outlining)

```mlir
gpu.module @gemm_128_128_64_kernel {
  gpu.func @gemm_128_128_64_kernel(
      %arg0: !llvm.ptr,    // TMA descriptor A
      %arg1: !llvm.ptr,    // TMA descriptor B
      %arg2: !llvm.ptr,    // Output: allocated
      %arg3: !llvm.ptr,    // Output: aligned
      %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64
  ) kernel attributes {gpu.known_block_size = array<i32: 128, 1, 1>} {

    // Generic GPU operations
    %thread_id = gpu.thread_id x
    %block_id = gpu.block_id x

    // Dynamic shared memory
    %shmem = gpu.dynamic.shared.memory : memref<?xi8, 3>

    // Barrier
    gpu.barrier

    // ... more operations ...

    gpu.return
  }
}
```

### Output (after Pass 14 - convert-gpu-to-nvvm)

```mlir
gpu.module @gemm_128_128_64_kernel {
  llvm.func @gemm_128_128_64_kernel(
      %arg0: !llvm.ptr, %arg1: !llvm.ptr,
      %arg2: !llvm.ptr, %arg3: !llvm.ptr,
      %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64
  ) attributes {
      nvvm.kernel,
      gpu.known_block_size = array<i32: 128, 1, 1>
  } {

    // NVVM intrinsics for thread/block ID
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    %thread_id = arith.index_cast %0 : i32 to index

    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    %block_id = arith.index_cast %1 : i32 to index

    // Dynamic shared memory
    %2 = llvm.mlir.addressof @__dynamic_shmem__ : !llvm.ptr<3>
    %3 = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, ...)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<...>
    // ... (build memref descriptor)
    %shmem = builtin.unrealized_conversion_cast %4 : ... to memref<?xi8, 3>

    // Barrier
    nvvm.barrier0

    // ... more operations ...

    llvm.return
  }
}
```

---

## Part 6: PTX Special Registers

NVVM intrinsics map directly to PTX special registers:

| MLIR Operation | NVVM Intrinsic | PTX Register | Description |
|----------------|----------------|--------------|-------------|
| `gpu.thread_id x` | `nvvm.read.ptx.sreg.tid.x` | `%tid.x` | Thread index in block (X) |
| `gpu.thread_id y` | `nvvm.read.ptx.sreg.tid.y` | `%tid.y` | Thread index in block (Y) |
| `gpu.thread_id z` | `nvvm.read.ptx.sreg.tid.z` | `%tid.z` | Thread index in block (Z) |
| `gpu.block_id x` | `nvvm.read.ptx.sreg.ctaid.x` | `%ctaid.x` | Block index in grid (X) |
| `gpu.block_dim x` | `nvvm.read.ptx.sreg.ntid.x` | `%ntid.x` | Block size (X) |
| `gpu.grid_dim x` | `nvvm.read.ptx.sreg.nctaid.x` | `%nctaid.x` | Grid size (X) |
| `gpu.lane_id` | `nvvm.read.ptx.sreg.laneid` | `%laneid` | Thread index in warp (0-31) |
| `gpu.warp_id` | `nvvm.read.ptx.sreg.warpid` | `%warpid` | Warp index in block |

**PTX Assembly Example**:
```ptx
mov.u32 %r0, %tid.x;        // Read thread ID X
mov.u32 %r1, %ctaid.x;      // Read block ID X
mov.u32 %r2, %ntid.x;       // Read block size X
mov.u32 %r3, %laneid;       // Read lane ID (0-31)
```

**Hardware Execution**:
- Special registers are read-only
- Values set by GPU hardware at kernel launch
- Zero-latency access (no memory fetch)
- Used for computing thread-specific addresses and data partitioning

---

## Part 7: Key MLIR/LLVM Data Structures

### 1. Template-Based Pattern Registration

```cpp
// Generic pattern for similar operations
template<typename SourceOp, typename XOp, typename YOp, typename ZOp>
struct OpLowering { ... };

// Instantiate for each GPU index operation
using ThreadIdLowering = OpLowering<gpu::ThreadIdOp, NVVM::ThreadIdXOp, ...>;
using BlockIdLowering = OpLowering<gpu::BlockIdOp, NVVM::BlockIdXOp, ...>;
using BlockDimLowering = OpLowering<gpu::BlockDimOp, NVVM::BlockDimXOp, ...>;
using GridDimLowering = OpLowering<gpu::GridDimOp, NVVM::GridDimXOp, ...>;

// Register all in one call
patterns.add<ThreadIdLowering, BlockIdLowering, BlockDimLowering, GridDimLowering>(...);
```

**Why?**
- DRY principle: Don't repeat pattern logic
- Compile-time safety: Type checking
- Easy to extend: New operation = new instantiation

### 2. MemRefDescriptor Helper

```cpp
// Helper for building memref descriptors
struct MemRefDescriptor {
  // Create undefined descriptor
  static Value undef(OpBuilder &builder, Location loc, Type descriptorType);

  // Set fields
  static Value setAllocatedPtr(OpBuilder &builder, Location loc,
                                Value descriptor, Type descriptorType, Value ptr);
  static Value setAlignedPtr(OpBuilder &builder, Location loc,
                              Value descriptor, Type descriptorType, Value ptr);
  static Value setOffset(OpBuilder &builder, Location loc,
                          Value descriptor, Type descriptorType, Value offset);
  static Value setSize(OpBuilder &builder, Location loc,
                        Value descriptor, Type descriptorType,
                        unsigned index, Value size);
  static Value setStride(OpBuilder &builder, Location loc,
                          Value descriptor, Type descriptorType,
                          unsigned index, Value stride);
};
```

**What it generates**:
```mlir
%0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
%1 = llvm.insertvalue %ptr, %0[0] : !llvm.struct<...>  // allocated
%2 = llvm.insertvalue %ptr, %1[1] : !llvm.struct<...>  // aligned
%3 = llvm.insertvalue %offset, %2[2] : !llvm.struct<...>  // offset
%4 = llvm.insertvalue %size0, %3[3, 0] : !llvm.struct<...>  // sizes[0]
// ... (more insertvalue ops)
```

**Why helper?**
- Tedious to build manually
- Easy to make mistakes (wrong indices)
- Consistent across conversions

### 3. SignatureConversion

```cpp
// Converts function signatures during lowering
class SignatureConversion {
  SmallVector<std::optional<TypeRange>, 4> argMapping;

  // Map original argument to converted type(s)
  void addInputs(unsigned origIdx, Type convertedType);
  void addInputs(unsigned origIdx, ArrayRef<Type> convertedTypes);

  // Get converted signature
  ArrayRef<Type> getConvertedTypes() const;
};
```

**Example Usage**:
```cpp
// Original: func(%arg0: memref<10xf32>)
// Converted: func(%arg0: ptr, %arg1: ptr, %arg2: i64, %arg3: i64, %arg4: i64)
//            (allocated, aligned, offset, size, stride)

SignatureConversion conversion(1);  // 1 original argument

// Memref unpacks to 5 LLVM arguments
SmallVector<Type> unpackedTypes = {
  ptrType, ptrType, i64Type, i64Type, i64Type
};
conversion.addInputs(0, unpackedTypes);
```

**Why?**
- High-level types (memref) → Low-level types (struct of pointers)
- Tracks which converted args map to which original arg
- Used during region cloning to remap block arguments

---

## Part 8: Why This Pass Exists

### Separation of Concerns

**GPU Dialect** (generic):
- Target-independent operations
- Portable across AMD/NVIDIA/Intel GPUs
- High-level abstractions

**NVVM Dialect** (NVIDIA-specific):
- PTX special registers
- NVIDIA-specific intrinsics
- Low-level, close to hardware

**Design Philosophy**:
- Write kernels in generic GPU dialect
- Lower to target-specific dialect (NVVM, ROCDL, SPIR-V)
- Same source, multiple backends

### Enabling PTX Generation

**After this pass**:
- All operations are LLVM/NVVM
- No generic GPU ops remain
- Ready for LLVM NVPTX backend

**Next step** (Pass 19):
- Translate LLVM IR → PTX assembly
- Use LLVM's NVPTX backend
- Embed PTX in `gpu.binary` operation

---

## Summary: convert-gpu-to-nvvm

**Input**: `gpu.module` with generic GPU operations
**Output**: `gpu.module` with NVVM intrinsics and LLVM ops

**Key Transformations**:
- `gpu.thread_id` → `nvvm.read.ptx.sreg.tid.x` (+ index cast)
- `gpu.block_id` → `nvvm.read.ptx.sreg.ctaid.x`
- `gpu.block_dim` → `nvvm.read.ptx.sreg.ntid.x`
- `gpu.grid_dim` → `nvvm.read.ptx.sreg.nctaid.x`
- `gpu.barrier` → `nvvm.barrier0`
- `gpu.dynamic.shared.memory` → `llvm.mlir.addressof @__dynamic_shmem__`
- `gpu.func` → `llvm.func` (+ `nvvm.kernel` attribute)
- `gpu.return` → `llvm.return`

**Key Techniques**:
- Template-based patterns for similar operations
- Dimension-specific lowering (x, y, z)
- MemRefDescriptor helper for building descriptors
- SignatureConversion for argument unpacking
- Nested pass managers (runs only on `gpu.module`)

**Why Critical?**
- Bridges generic GPU → target-specific NVVM
- Maps to PTX special registers
- Enables PTX compilation in next pass
- Target abstraction layer

---

**Next**: Pass 19 (`gpu-module-to-binary`) - Compiling LLVM/NVVM IR to PTX assembly
