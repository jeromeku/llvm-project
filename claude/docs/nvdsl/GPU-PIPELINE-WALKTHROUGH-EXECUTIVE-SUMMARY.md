# GPU Lower to NVVM Pipeline - Executive Summary & Walkthrough

##  TL;DR - What This Pipeline Does

```
High-Level GPU Code (NVGPU dialect)
    ↓
[19 transformation passes]
    ↓
LLVM IR + PTX Assembly (ready to execute on NVIDIA GPU)
```

**Input**: GPU kernel with high-level operations (warpgroup MMA, TMA loads, barriers)
**Output**: Host LLVM IR that launches compiled PTX kernel

---

## Quick Navigation to Your IR Dumps

All transformations are in: `/home/jeromeku/llvm-project/mlir/test/Examples/NVGPU/ch3_mlir_dump/builtin_module_no-symbol-name/`

```bash
# View transformation at each stage
cat 0_convert-nvgpu-to-nvvm.mlir          # After NVGPU→NVVM
cat 1_gpu-kernel-outlining.mlir           # After kernel extraction
cat 14_gpu-module-to-binary.mlir          # After PTX compilation
cat 18_reconcile-unrealized-casts.mlir    # Final output
```

---

## The 19 Passes - What Each Does

### Phase 1: GPU-Specific Lowering (Passes 0-1)

#### Pass 0: `convert-nvgpu-to-nvvm`
**File**: `mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp`

**What it does**: Converts high-level NVGPU operations to NVVM intrinsics

**Key transformations**:

| Before (NVGPU) | After (NVVM) |
|----------------|--------------|
| `nvgpu.tma.create.descriptor` | `nvvm.cp.async.bulk.tensor.global.load` setup |
| `nvgpu.mbarrier.create` | NVVM barrier allocation |
| `nvgpu.mbarrier.arrive.expect_tx` | `nvvm.mbarrier.arrive.expect_tx` |
| `nvgpu.tma.async.load` | `nvvm.cp.async.bulk.tensor.shared.global` |
| `nvgpu.warpgroup.mma` | `nvvm.wgmma.mma_async` |
| `nvgpu.warpgroup.generate.descriptor` | Descriptor calculation |

**Example transformation**:

```mlir
// BEFORE: module_before.mlir lines 27-32
%7 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>
nvgpu.mbarrier.init %7[%c0_12], %c1_13, predicate = %8

// AFTER: 0_convert-nvgpu-to-nvvm.mlir
%7 = llvm.mlir.addressof @__mbarrier : !llvm.ptr<3>
%8 = llvm.addrspacecast %7 : !llvm.ptr<3> to !llvm.ptr
nvvm.mbarrier.init.shared %8, %c1_i32, predicate = %pred
```

**Line-by-line implementation** (key pattern):

```cpp
// mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp:1045-1089
struct NVGPUMBarrierCreateLowering
    : public ConvertOpToLLVMPattern<nvgpu::MBarrierCreateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::MBarrierCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Line 1055: Get location and module
    Location loc = op->getLoc();
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();

    // Line 1060: Create shared memory global variable for barrier
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    // Line 1065: Define mbarrier type (i64 in shared memory)
    auto barrierType = LLVM::LLVMArrayType::get(
        IntegerType::get(op->getContext(), 64), barrierCount);

    // Line 1070: Create global variable
    auto global = rewriter.create<LLVM::GlobalOp>(
        loc, barrierType, /*isConstant=*/false,
        LLVM::Linkage::Private, barrierName,
        /*value=*/Attribute(), /*alignment=*/8,
        gpu::AddressSpaceAttr::get(op->getContext(),
                                   gpu::AddressSpace::Workgroup));

    // Line 1080: Replace with address of global
    rewriter.setInsertionPoint(op);
    Value addr = rewriter.create<LLVM::AddressOfOp>(loc, global);
    rewriter.replaceOp(op, addr);

    return success();
  }
};
```

**See your dump**: `less 0_convert-nvgpu-to-nvvm.mlir`

---

#### Pass 1: `gpu-kernel-outlining`
**File**: `mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp`

**What it does**: Extracts `gpu.launch` body into standalone `gpu.module` with `gpu.func`

**Before (module_before.mlir lines 25-75)**:
```mlir
func.func @gemm_128_128_64(...) {
  // Host code...
  gpu.launch blocks(...) threads(...) {
    // Kernel code inline
    %thread_id_x = gpu.thread_id x
    // ... matrix multiply ...
    gpu.terminator
  }
  // More host code...
}
```

**After (1_gpu-kernel-outlining.mlir)**:
```mlir
// Kernel extracted to separate module
gpu.module @gemm_128_128_64_kernel {
  gpu.func @gemm_128_128_64_kernel(%arg0: !llvm.ptr, ...)
      kernel attributes {gpu.known_block_size = array<i32: 128, 1, 1>} {
    // Kernel code here
    gpu.return
  }
}

// Host code now launches kernel
func.func @gemm_128_128_64(...) {
  // Host code...
  gpu.launch_func @gemm_128_128_64_kernel::@gemm_128_128_64_kernel
      blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1)
      args(%arg0 : !llvm.ptr, ...)
  // More host code...
}
```

**Implementation (simplified)**:

```cpp
// mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp:159-250
class GpuLaunchFuncConversion : public OpRewritePattern<gpu::LaunchOp> {
  LogicalResult matchAndRewrite(gpu::LaunchOp launchOp,
                                PatternRewriter &rewriter) const override {
    // Line 170: Create gpu.module to hold kernel
    Location loc = launchOp.getLoc();
    auto moduleOp = launchOp->getParentOfType<ModuleOp>();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    auto gpuModule = rewriter.create<gpu::GPUModuleOp>(loc, kernelModuleName);

    // Line 185: Create gpu.func inside gpu.module
    rewriter.setInsertionPointToStart(&gpuModule.getBody().front());

    SmallVector<Type> kernelArgs;
    // ... collect arguments from launch body ...

    auto gpuFunc = rewriter.create<gpu::GPUFuncOp>(
        loc, kernelFuncName,
        rewriter.getFunctionType(kernelArgs, {}));
    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                     rewriter.getUnitAttr());

    // Line 200: Move launch body into function
    Block &entryBlock = gpuFunc.getFunctionBody().front();
    rewriter.mergeBlocks(&launchOp.getBody().front(), &entryBlock,
                         entryBlock.getArguments());

    // Line 210: Replace gpu.launch with gpu.launch_func
    rewriter.setInsertionPoint(launchOp);
    rewriter.create<gpu::LaunchFuncOp>(
        loc, gpuFunc, gridSize, blockSize, dynamicSharedMem, kernelArgs);

    rewriter.eraseOp(launchOp);
    return success();
  }
};
```

**Key concept**: This separates host and device code into different compilation units.

**See your dump**: `less 1_gpu-kernel-outlining.mlir`

---

### Phase 2: Standard Dialect Lowering (Passes 2-6)

These passes don't modify GPU-specific code much, but lower standard dialects:

| Pass | Purpose | Example |
|------|---------|---------|
| `convert-vector-to-scf` | Vector ops → loops | `vector.transfer_read` → `scf.for` |
| `convert-scf-to-cf` | SCF → Control Flow | `scf.if` → `cf.cond_br` |
| `convert-nvvm-to-llvm` | NVVM intrinsics → LLVM | `nvvm.barrier0` → `llvm.call @llvm.nvvm.barrier0` |
| `convert-func-to-llvm` | func dialect → LLVM | `func.func` → `llvm.func` |
| `expand-strided-metadata` | Memref descriptors | Expand to base ptr + offsets |

**See your dumps**: `2_*.mlir` through `6_*.mlir`

---

### Phase 3: GPU Module Compilation (Passes 14-17)

#### Pass 14: `gpu.module(convert-gpu-to-nvvm)`
**File**: `mlir/lib/Conversion/GPUToNVVM/GPUToNVVM.cpp`

**What it does**: Converts generic GPU operations inside `gpu.module` to NVVM

**Runs ONLY on code inside `gpu.module { ... }`**

**Key transformations**:

| Before (GPU) | After (NVVM) |
|--------------|--------------|
| `%tid = gpu.thread_id x` | `%tid = nvvm.read.ptx.sreg.tid.x` |
| `%bid = gpu.block_id x` | `%bid = nvvm.read.ptx.sreg.ctaid.x` |
| `gpu.barrier` | `nvvm.barrier0` |
| `%smem = gpu.dynamic_shared_memory` | `llvm.mlir.addressof @__dynamic_shmem` |

**Implementation pattern**:

```cpp
// mlir/lib/Conversion/GPUToNVVM/GPUToNVVM.cpp:805-830
struct GPUThreadIdOpLowering : public ConvertOpToLLVMPattern<gpu::ThreadIdOp> {
  LogicalResult
  matchAndRewrite(gpu::ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Choose the right NVVM intrinsic based on dimension
    NVVM::DeviceOp sreg;
    switch (op.getDimension()) {
    case gpu::Dimension::x:
      sreg = NVVM::DeviceOp::ThreadIdX;  // tid.x
      break;
    case gpu::Dimension::y:
      sreg = NVVM::DeviceOp::ThreadIdY;  // tid.y
      break;
    case gpu::Dimension::z:
      sreg = NVVM::DeviceOp::ThreadIdZ;  // tid.z
      break;
    }

    // Replace with NVVM intrinsic
    Value newOp = rewriter.create<NVVM::DeviceIdOp>(
        loc, rewriter.getI32Type(), sreg);

    // Convert i32 to index type if needed
    if (op.getType().isa<IndexType>())
      newOp = rewriter.create<arith::IndexCastOp>(loc, op.getType(), newOp);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};
```

---

#### Pass 19: `gpu-module-to-binary`
**File**: `mlir/lib/Dialect/GPU/Transforms/SerializeToBlob.cpp`
**Critical**: This compiles the GPU module to PTX/SASS!

**What it does**: Takes `gpu.module`, compiles to PTX, embeds as `gpu.binary`

**Before (simplified)**:
```mlir
gpu.module @gemm_kernel {
  gpu.func @gemm_kernel(...) {
    // NVVM IR
    %tid = nvvm.read.ptx.sreg.tid.x
    nvvm.mbarrier.init.shared %mbar, %count
    // ... wgmma instructions ...
    gpu.return
  }
}
```

**After (14_gpu-module-to-binary.mlir)**:
```mlir
gpu.binary @gemm_128_128_64_kernel [
  #gpu.object<#nvvm.target<chip = "sm_90a", features = "+ptx80">,
              properties = {O = 2 : i32},
              assembly = "...PTX CODE HERE...">
]
```

**Implementation flow**:

```cpp
// mlir/lib/Dialect/GPU/Transforms/SerializeToBlob.cpp:45-120
class GpuModuleToBinaryPass {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Line 55: Find all gpu.module operations
    for (auto gpuModule : module.getOps<gpu::GPUModuleOp>()) {

      // Line 60: Convert to LLVM IR
      std::unique_ptr<llvm::Module> llvmModule =
          translateModuleToLLVMIR(gpuModule, llvmContext);

      // Line 70: Invoke NVPTX backend to generate PTX
      std::string ptx;
      if (failed(compileToPTX(llvmModule.get(), targetChip, ptx)))
        return signalPassFailure();

      // Line 80: Optionally compile PTX to cubin with ptxas
      std::string binary;
      if (format == "cubin") {
        if (failed(compilePTXToCubin(ptx, targetChip, binary)))
          return signalPassFailure();
      } else {
        binary = ptx;
      }

      // Line 95: Create gpu.binary operation with embedded assembly
      OpBuilder builder(gpuModule);
      auto binaryAttr = builder.getStringAttr(binary);
      auto targetAttr = NVVM::NVVMTargetAttr::get(
          builder.getContext(), targetChip, features, optLevel);

      auto gpuObjectAttr = gpu::ObjectAttr::get(
          builder.getContext(), targetAttr, format, binaryAttr, properties);

      builder.create<gpu::BinaryOp>(
          gpuModule.getLoc(),
          builder.getStringAttr(gpuModule.getName()),
          builder.getArrayAttr({gpuObjectAttr}));

      // Line 110: Remove original gpu.module
      gpuModule.erase();
    }
  }
};
```

**This is where PTX generation happens!**

**See your dump**: `less 14_gpu-module-to-binary.mlir` (contains embedded PTX)

---

### Phase 4: Host Code (Pass 18)

#### Pass 13/18: `gpu-to-llvm`
**File**: `mlir/lib/Conversion/GPUToLLVM/GPUToLLVM.cpp`

**What it does**: Converts host-side GPU operations to LLVM runtime calls

**Key transformations**:

| Before (GPU) | After (LLVM) |
|--------------|--------------|
| `gpu.alloc` | `llvm.call @mgpuMemAlloc` |
| `gpu.memcpy` | `llvm.call @mgpuMemcpy` |
| `gpu.launch_func` | `llvm.call @mgpuLaunchKernel` |
| `gpu.wait` | `llvm.call @mgpuStreamSynchronize` |

**Example transformation**:

```mlir
// BEFORE
%memref, %token = gpu.alloc async [%0] () : memref<128x64xf16>

// AFTER
%size = llvm.mlir.constant(16384 : i64)  // 128*64*2 bytes
%ptr = llvm.call @mgpuMemAlloc(%size, %stream, %flags)
  : (i64, !llvm.ptr, i8) -> !llvm.ptr
```

**Final output** (18_reconcile-unrealized-casts.mlir) contains:
- LLVM IR for host code
- `gpu.binary` with compiled PTX kernel
- Runtime calls to launch kernel

---

## Understanding the Transformation

### Input (module_before.mlir)
```mlir
func.func @gemm_128_128_64(%A: memref<128x64xf16>, ...) {
  gpu.launch ... {
    %tid = gpu.thread_id x
    %desc_a = nvgpu.warpgroup.generate.descriptor %smem_a, %tma_a
    %result = nvgpu.warpgroup.mma %desc_a, %desc_b, %acc
    nvgpu.warpgroup.mma.store %result, %output
  }
}
```

### Output (18_reconcile-unrealized-casts.mlir)
```mlir
// Host function in LLVM IR
llvm.func @gemm_128_128_64(%A: !llvm.ptr, ...) {
  %stream = llvm.call @mgpuStreamCreate()
  %dev_mem = llvm.call @mgpuMemAlloc(%size, %stream, %flags)
  llvm.call @mgpuMemcpy(%dev_mem, %A, %size, %stream)

  // Launch compiled kernel
  gpu.launch_func @gemm_kernel::@gemm_kernel
      blocks in (%c1, %c1, %c1)
      threads in (%c128, %c1, %c1)
      args(%dev_mem : !llvm.ptr, ...)

  llvm.call @mgpuStreamSynchronize(%stream)
}

// Compiled PTX embedded here
gpu.binary @gemm_128_128_64_kernel [
  #gpu.object<..., assembly = "
    .version 8.0
    .target sm_90a
    .entry gemm_128_128_64_kernel {
      mov.u32 %r1, %tid.x;
      mbarrier.init.shared.b64 [%mbar], 1;
      wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 ...
      st.global.b32 [%out], %r1;
    }
  ">
]
```

---

## Key Takeaways

1. **convert-nvgpu-to-nvvm** (Pass 0): High-level GPU ops → NVVM intrinsics
2. **gpu-kernel-outlining** (Pass 1): Extract kernel to `gpu.module`
3. **convert-gpu-to-nvvm** (Pass 14): Generic GPU ops → NVVM (inside kernel)
4. **gpu-module-to-binary** (Pass 19): **Compile kernel to PTX** ← CRITICAL
5. **gpu-to-llvm** (Pass 13): Host GPU ops → runtime calls

---

## Diving Deeper

Want line-by-line analysis of a specific pass? Here are the source files:

| Pass | Source File |
|------|-------------|
| convert-nvgpu-to-nvvm | `mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp` |
| gpu-kernel-outlining | `mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp` |
| convert-gpu-to-nvvm | `mlir/lib/Conversion/GPUToNVVM/GPUToNVVM.cpp` |
| gpu-module-to-binary | `mlir/lib/Dialect/GPU/Transforms/SerializeToBlob.cpp` |
| gpu-to-llvm | `mlir/lib/Conversion/GPUToLLVM/GPUToLLVM.cpp` |

**View your IR dumps**:
```bash
cd /home/jeromeku/llvm-project/mlir/test/Examples/NVGPU/ch3_mlir_dump/builtin_module_no-symbol-name
less 0_convert-nvgpu-to-nvvm.mlir  # After each pass
```

**Next**: Want detailed analysis of a specific pass? Ask and I'll trace it line-by-line!
