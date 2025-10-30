# NVGPU to NVVM Conversion Pass Analysis

## Overview

This document explains the relationship between `ConvertNVGPUToNVVMPass` and `populateNVGPUToNVVMConversionPatterns`, and traces the complete call stack for the NVGPU to NVVM dialect conversion in MLIR.

---

## Key Components

### 1. ConvertNVGPUToNVVMPass

**Location**: [mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp:395-478](../../../mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp#L395-L478)

**What**: A complete Pass class that orchestrates the entire NVGPU → NVVM conversion.

**Role**:
- Sets up the conversion infrastructure (type converters, conversion targets)
- Registers custom type conversions for NVGPU-specific types
- Populates conversion patterns
- Drives the actual conversion using the dialect conversion framework

**When Used**:
- Registered with pass manager and executed via `-convert-nvgpu-to-nvvm`
- Part of the GPU-to-NVVM lowering pipeline

### 2. populateNVGPUToNVVMConversionPatterns

**Location**: [mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp:1742-1765](../../../mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp#L1742-L1765)

**What**: A pattern population function that registers individual conversion patterns.

**Role**:
- Adds 20+ specific operation-by-operation rewrite patterns
- Each pattern knows how to convert one NVGPU op to NVVM intrinsics
- Reusable by other passes or pipelines

**Patterns Registered**:
```cpp
patterns.add<
    NVGPUMBarrierCreateLowering,           // nvgpu.mbarrier.create
    NVGPUMBarrierInitLowering,             // nvgpu.mbarrier.init
    NVGPUMBarrierGetLowering,              // nvgpu.mbarrier.get
    NVGPUMBarrierArriveLowering,           // nvgpu.mbarrier.arrive
    NVGPUMBarrierArriveNoCompleteLowering, // nvgpu.mbarrier.arrive.no_complete
    NVGPUMBarrierTestWaitLowering,         // nvgpu.mbarrier.test_wait_parity
    NVGPUMBarrierTryWaitParityLowering,    // nvgpu.mbarrier.try_wait_parity
    NVGPUTmaAsyncLoadOpLowering,           // nvgpu.tma.async.load
    NVGPUTmaAsyncStoreOpLowering,          // nvgpu.tma.async.store
    NVGPUTmaCreateDescriptorOpLowering,    // nvgpu.tma.create.descriptor
    NVGPUTmaPrefetchOpLowering,            // nvgpu.tma.prefetch.descriptor
    NVGPUTmaFenceOpLowering,               // nvgpu.tma.fence.descriptor
    NVGPUMBarrierArriveExpectTxLowering,   // nvgpu.mbarrier.arrive.expect_tx
    NVGPUGenerateWarpgroupDescriptorLowering, // nvgpu.warpgroup.generate.descriptor
    NVGPUWarpgroupMmaOpLowering,              // nvgpu.warpgroup.mma
    NVGPUWarpgroupMmaStoreOpLowering,         // nvgpu.warpgroup.mma.store
    NVGPUWarpgroupMmaInitAccumulatorOpLowering, // nvgpu.warpgroup.mma.init.accumulator
    MmaSyncOptoNVVM,                       // nvgpu.mma.sync
    MmaLdMatrixOpToNVVM,                   // nvgpu.ldmatrix
    NVGPUAsyncCopyLowering,                // nvgpu.device_async_copy
    NVGPUAsyncCreateGroupLowering,         // nvgpu.device_async_create_group
    NVGPUAsyncWaitLowering,                // nvgpu.device_async_wait
    NVGPUMmaSparseSyncLowering,            // nvgpu.mma.sp.sync
    NVGPURcpOpLowering                     // nvgpu.rcp
>(converter);
```

---

## Complete Call Stack

### UPWARD: What Calls the Pass

```
┌─────────────────────────────────────┐
│ User Command / Pipeline             │
│ mlir-opt -convert-nvgpu-to-nvvm     │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Pass Manager                        │
│ (Infrastructure)                    │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ ConvertNVGPUToNVVMPass              │
│ ::runOnOperation() ← YOU ARE HERE   │
│ (Line 399)                          │
└─────────────────────────────────────┘
```

**Specific Callers:**

1. **Command Line**:
   ```bash
   mlir-opt input.mlir -convert-nvgpu-to-nvvm
   ```

2. **GPU Pipeline** ([mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp:43](../../../mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp#L43)):
   ```cpp
   void buildCommonPassPipeline(
       OpPassManager &pm, const GPUToNVVMPipelineOptions &options) {
     pm.addPass(createConvertNVGPUToNVVMPass());
     pm.addPass(createGpuKernelOutliningPass());
     // ... more passes
   }
   ```

3. **Test Files** ([mlir/test/Conversion/NVGPUToNVVM/nvgpu-to-nvvm.mlir](../../../mlir/test/Conversion/NVGPUToNVVM/nvgpu-to-nvvm.mlir)):
   ```mlir
   // RUN: mlir-opt %s -convert-nvgpu-to-nvvm | FileCheck %s
   ```

### DOWNWARD: What the Pass Calls

```
ConvertNVGPUToNVVMPass::runOnOperation()
│
├─ Step 1: Create Infrastructure (lines 400-403)
│  └─ LowerToLLVMOptions, RewritePatternSet, LLVMTypeConverter
│
├─ Step 2: Setup GPU Memory Space Conversions (lines 404-418)
│  └─ populateGpuMemorySpaceAttributeConversions()
│     • Global → NVVM Global Memory Space
│     • Workgroup → NVVM Shared Memory Space
│     • Private → 0
│
├─ Step 3: Register Custom Type Conversions (lines 422-465)
│  ├─ DeviceAsyncTokenType → i32 (line 422)
│  ├─ WarpgroupAccumulatorType → LLVM struct (line 425)
│  ├─ MBarrierTokenType → i64 (line 452)
│  ├─ WarpgroupMatrixDescriptorType → i64 (line 455)
│  ├─ MBarrierGroupType → memref (line 459)
│  └─ TensorMapDescriptorType → LLVM pointer (line 463)
│
├─ Step 4: Populate Conversion Patterns (line 466)
│  └─ populateNVGPUToNVVMConversionPatterns(converter, patterns)
│     │
│     └─ Registers 20+ conversion patterns (see patterns list above)
│        Each pattern implements matchAndRewrite() for specific ops
│
├─ Step 5: Create Conversion Target (lines 467-471)
│  └─ LLVMConversionTarget
│     ├─ addLegalDialect<LLVM::LLVMDialect>()
│     ├─ addLegalDialect<arith::ArithDialect>()
│     ├─ addLegalDialect<memref::MemRefDialect>()
│     └─ addLegalDialect<NVVM::NVVMDialect>()
│
├─ Step 6: Add SCF Type Conversions (line 472)
│  └─ scf::populateSCFStructuralTypeConversionsAndLegality()
│
└─ Step 7: Apply Conversion (lines 474-476)
   └─ applyPartialConversion(getOperation(), target, patterns)
      │
      └─ Dialect Conversion Driver
         ├─ Traverses IR tree
         ├─ Checks legality against target
         ├─ Applies matching patterns
         ├─ Converts types using TypeConverter
         └─ Returns LogicalResult (success/failure)
```

---

## Key Function: applyPartialConversion

**Location**: [mlir/include/mlir/Transforms/DialectConversion.h:1448](../../../mlir/include/mlir/Transforms/DialectConversion.h#L1448)

**Signature**:
```cpp
LogicalResult applyPartialConversion(
    Operation *op,                        // Root operation
    const ConversionTarget &target,       // Legal/illegal dialect specs
    const FrozenRewritePatternSet &patterns // Registered patterns
);
```

**What It Does**:
1. Traverses the IR tree starting from the root operation
2. For each operation, checks if it's legal according to `target`
3. If illegal, attempts to apply matching patterns from `patterns`
4. Converts types using the `LLVMTypeConverter`
5. **"Partial"** means: Allows some ops to remain unconverted
   - Only fails if ops are explicitly marked illegal AND can't be converted
   - Unlike `applyFullConversion` which requires all ops to be converted

**Related Functions**:
- `applyFullConversion`: Requires complete conversion (all ops must be legal)
- `applyAnalysisConversion`: Analyzes what would be converted without actually converting

---

## Detailed Pass Execution Flow

### runOnOperation() Implementation

```cpp
void ConvertNVGPUToNVVMPass::runOnOperation() {
  // 1. Setup infrastructure
  LowerToLLVMOptions options(&getContext());
  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext(), options);
  IRRewriter rewriter(&getContext());

  // 2. Configure GPU memory space mappings
  populateGpuMemorySpaceAttributeConversions(
      converter, [](gpu::AddressSpace space) -> unsigned {
        switch (space) {
        case gpu::AddressSpace::Global:
          return static_cast<unsigned>(
              NVVM::NVVMMemorySpace::kGlobalMemorySpace);
        case gpu::AddressSpace::Workgroup:
          return static_cast<unsigned>(
              NVVM::NVVMMemorySpace::kSharedMemorySpace);
        case gpu::AddressSpace::Private:
          return 0;
        }
      });

  // 3. Register custom NVGPU type conversions

  // Device async tokens → dummy i32 (dropped during conversion)
  converter.addConversion([&](nvgpu::DeviceAsyncTokenType type) -> Type {
    return converter.convertType(IntegerType::get(type.getContext(), 32));
  });

  // Warpgroup accumulators → LLVM struct
  converter.addConversion([&](nvgpu::WarpgroupAccumulatorType type) -> Type {
    Type elemType = type.getFragmented().getElementType();
    int64_t sizeM = type.getFragmented().getDimSize(0);
    int64_t sizeN = type.getFragmented().getDimSize(1);

    unsigned numMembers;
    if (elemType.isF32() || elemType.isInteger(32))
      numMembers = sizeN / 2;
    else if (elemType.isF16())
      numMembers = sizeN / 4;

    SmallVector<Type> innerStructBody;
    for (unsigned i = 0; i < numMembers; i++)
      innerStructBody.push_back(elemType);
    auto innerStructType =
        LLVM::LLVMStructType::getLiteral(type.getContext(), innerStructBody);

    SmallVector<Type> structBody;
    for (int i = 0; i < sizeM; i += kWgmmaSizeM)
      structBody.push_back(innerStructType);

    auto convertedType =
        LLVM::LLVMStructType::getLiteral(type.getContext(), structBody);
    return converter.convertType(convertedType);
  });

  // Barrier tokens → i64
  converter.addConversion([&](nvgpu::MBarrierTokenType type) -> Type {
    return converter.convertType(IntegerType::get(type.getContext(), 64));
  });

  // Matrix descriptors → i64
  converter.addConversion([&](nvgpu::WarpgroupMatrixDescriptorType type) -> Type {
    return converter.convertType(IntegerType::get(type.getContext(), 64));
  });

  // Barrier groups → memref
  converter.addConversion([&](nvgpu::MBarrierGroupType type) -> Type {
    return converter.convertType(
        nvgpu::getMBarrierMemrefType(rewriter.getContext(), type));
  });

  // TMA descriptors → LLVM pointer
  converter.addConversion([&](nvgpu::TensorMapDescriptorType type) -> Type {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  // 4. Populate all NVGPU→NVVM conversion patterns
  populateNVGPUToNVVMConversionPatterns(converter, patterns);

  // 5. Define what's legal in the target
  LLVMConversionTarget target(getContext());
  target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<::mlir::arith::ArithDialect>();
  target.addLegalDialect<::mlir::memref::MemRefDialect>();
  target.addLegalDialect<::mlir::NVVM::NVVMDialect>();

  // 6. Handle SCF structural conversions
  mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
      converter, patterns, target);

  // 7. Run the conversion
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns))))
    signalPassFailure();
}
```

---

## Example: Complete Conversion Flow

### Input MLIR
```mlir
func.func @example(%arg0: vector<4x2xf16>,
                   %arg1: vector<2x2xf16>,
                   %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
  %result = nvgpu.mma.sync(%arg0, %arg1, %arg2)
            {mmaShape = [16, 8, 16]}
            : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>)
            -> vector<2x2xf16>
  return %result : vector<2x2xf16>
}
```

### Execution Steps

1. **Pass Manager** invokes `ConvertNVGPUToNVVMPass::runOnOperation()`

2. **Pass Setup**:
   - Creates `LLVMTypeConverter` with custom type conversions
   - Registers conversion patterns via `populateNVGPUToNVVMConversionPatterns()`
   - Creates `LLVMConversionTarget` marking LLVM/NVVM dialects as legal

3. **applyPartialConversion** execution:
   - Traverses the function body
   - Encounters `nvgpu.mma.sync` operation
   - Checks legality: NOT legal (NVGPU dialect not in target)
   - Searches for matching pattern: Finds `MmaSyncOptoNVVM`
   - Calls `MmaSyncOptoNVVM::matchAndRewrite()`

4. **Pattern Application** (`MmaSyncOptoNVVM`):
   - Extracts vector operands and converts to LLVM structs
   - Generates NVVM intrinsic call: `nvvm.wmma.mma`
   - Converts result back to expected type

### Output MLIR
```mlir
func.func @example(%arg0: !llvm.array<4 x vector<2xf16>>,
                   %arg1: !llvm.array<2 x vector<2xf16>>,
                   %arg2: !llvm.array<2 x vector<2xf16>>)
                   -> !llvm.array<2 x vector<2xf16>> {
  // Extraction of elements from arrays
  %0 = llvm.extractvalue %arg0[0] : !llvm.array<4 x vector<2xf16>>
  %1 = llvm.extractvalue %arg0[1] : !llvm.array<4 x vector<2xf16>>
  // ... more extractions

  // NVVM intrinsic call
  %result = nvvm.wmma.mma %0, %1, ..., %arg2
            {layoutA = #nvvm.mma_layout<row>,
             layoutB = #nvvm.mma_layout<col>,
             shape = #nvvm.shape<m = 16, n = 8, k = 16>}
            : !llvm.struct<...>

  // Reconstruction
  %ret = llvm.insertvalue %result, ... : !llvm.array<2 x vector<2xf16>>
  return %ret : !llvm.array<2 x vector<2xf16>>
}
```

---

## Why Separate Pass and Populate Functions?

### Design Benefits

1. **Modularity**: `populateNVGPUToNVVMConversionPatterns` can be reused by other passes or custom pipelines without the full pass infrastructure.

2. **Flexibility**: Custom lowering pipelines can combine patterns from multiple populate functions:
   ```cpp
   RewritePatternSet patterns(&ctx);
   LLVMTypeConverter converter(&ctx);

   // Combine multiple pattern sets
   populateNVGPUToNVVMConversionPatterns(converter, patterns);
   populateGPUToNVVMConversionPatterns(converter, patterns);
   populateMemRefToLLVMConversionPatterns(converter, patterns);

   // Apply with custom target
   applyPartialConversion(op, myCustomTarget, patterns);
   ```

3. **Testability**: Patterns can be tested independently of pass infrastructure.

4. **Composition**: Different passes can use the same patterns with different configurations:
   - Different type converters
   - Different conversion targets
   - Different pass options

### Common Pattern in MLIR

This design pattern appears throughout MLIR conversions:
- `populateGPUToNVVMConversionPatterns` + `ConvertGpuOpsToNVVMOps`
- `populateMemRefToLLVMConversionPatterns` + `FinalizeMemRefToLLVMConversionPass`
- `populateArithToLLVMConversionPatterns` + `ArithToLLVMConversionPass`

---

## Key Files and Locations

| File | Lines | Purpose |
|------|-------|---------|
| [mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp](../../../mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp) | 395-478 | `ConvertNVGPUToNVVMPass` implementation |
| [mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp](../../../mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp) | 1742-1765 | `populateNVGPUToNVVMConversionPatterns` |
| [mlir/include/mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h](../../../mlir/include/mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h) | 37-38 | Public function declarations |
| [mlir/include/mlir/Conversion/Passes.td](../../../mlir/include/mlir/Conversion/Passes.td) | 965-977 | TableGen pass definition |
| [mlir/include/mlir/Transforms/DialectConversion.h](../../../mlir/include/mlir/Transforms/DialectConversion.h) | 1448-1450 | `applyPartialConversion` declaration |
| [mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp](../../../mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp) | 43 | Pass usage in pipeline |
| [mlir/test/Conversion/NVGPUToNVVM/nvgpu-to-nvvm.mlir](../../../mlir/test/Conversion/NVGPUToNVVM/nvgpu-to-nvvm.mlir) | - | Test cases |

---

## Pass Definition (TableGen)

**Location**: [mlir/include/mlir/Conversion/Passes.td:965-977](../../../mlir/include/mlir/Conversion/Passes.td#L965-L977)

```tablegen
def ConvertNVGPUToNVVMPass : Pass<"convert-nvgpu-to-nvvm"> {
  let summary = "Convert NVGPU dialect to NVVM dialect";
  let description = [{
    This pass converts supported NVGPU ops to NVVM dialect intrinsics.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "NVVM::NVVMDialect"
  ];
}
```

This TableGen definition generates:
- `createConvertNVGPUToNVVMPass()` function
- `ConvertNVGPUToNVVMPassBase` base class
- Pass registration boilerplate

---

## Usage Examples

### Command Line
```bash
# Standalone conversion
mlir-opt input.mlir -convert-nvgpu-to-nvvm -o output.mlir

# As part of full GPU pipeline
mlir-opt input.mlir -gpu-lower-to-nvvm-pipeline
```

### Programmatic (C++)
```cpp
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"

// Add to pass manager
void buildPipeline(OpPassManager &pm) {
  pm.addPass(createConvertNVGPUToNVVMPass());
  // ... other passes
}
```

### Custom Pattern Usage
```cpp
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"

void customConversion(Operation *op) {
  MLIRContext *ctx = op->getContext();
  RewritePatternSet patterns(ctx);
  LLVMTypeConverter converter(ctx);

  // Reuse patterns without full pass
  populateNVGPUToNVVMConversionPatterns(converter, patterns);

  ConversionTarget target(*ctx);
  target.addLegalDialect<NVVM::NVVMDialect>();

  if (failed(applyPartialConversion(op, target, patterns)))
    // Handle failure
}
```

---

## Summary

- **`ConvertNVGPUToNVVMPass`** = Complete orchestrator (pass infrastructure + conversion)
- **`populateNVGPUToNVVMConversionPatterns`** = Pattern registration (reusable component)
- **`applyPartialConversion`** = Conversion engine (applies patterns to IR)

The pass calls the populate function, which calls the conversion engine with registered patterns. This modular design enables flexible composition and reuse across different conversion scenarios.
