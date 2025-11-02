# GPU Lower to NVVM Pipeline - Documentation Index

## Quick Start

**Want to understand the pipeline?** Start here:
1. Read [GPU-PIPELINE-WALKTHROUGH-EXECUTIVE-SUMMARY.md](GPU-PIPELINE-WALKTHROUGH-EXECUTIVE-SUMMARY.md) (15 min)
2. Review [GPU-PIPELINE-BEFORE-AFTER-EXAMPLES.md](GPU-PIPELINE-BEFORE-AFTER-EXAMPLES.md) (20 min)
3. Explore your IR dumps in `/home/jeromeku/llvm-project/mlir/test/Examples/NVGPU/ch3_mlir_dump/builtin_module_no-symbol-name/`

---

## Documentation Files

### 1. Executive Summary (START HERE)
**File**: [GPU-PIPELINE-WALKTHROUGH-EXECUTIVE-SUMMARY.md](GPU-PIPELINE-WALKTHROUGH-EXECUTIVE-SUMMARY.md)

**What it covers**:
- Overview of all 19+ passes
- What each pass does (high-level)
- Key transformations with examples
- Source file locations for each pass
- C++ implementation patterns

**Best for**: Understanding the big picture quickly

---

### 2. Before/After Examples (DETAILED TRANSFORMS)
**File**: [GPU-PIPELINE-BEFORE-AFTER-EXAMPLES.md](GPU-PIPELINE-BEFORE-AFTER-EXAMPLES.md)

**What it covers**:
- Actual transformations from your IR dumps
- Line-by-line before/after comparisons
- Explanation of what changed and why
- Exact line numbers in your dumps
- Key insights from each transformation

**Best for**: Understanding specific transformations in detail

---

### 3. MLIR Operation Structure Primer
**File**: [gpu-lower-to-nvvm-pipeline-deep-dive-PART1.md](gpu-lower-to-nvvm-pipeline-deep-dive-PART1.md)

**What it covers**:
- MLIR operation anatomy
- Traits and interfaces
- Regions and blocks
- Attributes vs. operands
- Example operations explained

**Best for**: Learning MLIR fundamentals

---

## The 19 Passes at a Glance

| # | Pass | Phase | Purpose |
|---|------|-------|---------|
| 0 | `convert-nvgpu-to-nvvm` | GPU | NVGPU → NVVM intrinsics |
| 1 | `gpu-kernel-outlining` | GPU | Extract kernel to module |
| 2 | `convert-vector-to-scf` | Standard | Vector ops → loops |
| 3 | `convert-scf-to-cf` | Standard | SCF → control flow |
| 4 | `convert-nvvm-to-llvm` | GPU | NVVM → LLVM intrinsics |
| 5 | `convert-func-to-llvm` | Standard | func dialect → LLVM |
| 6 | `expand-strided-metadata` | Standard | Expand memref descriptors |
| 7 | `nvvm-attach-target` | GPU | Attach compilation attrs |
| 8 | `lower-affine` | Standard | Affine → standard |
| 9 | `convert-arith-to-llvm` | Standard | Arithmetic → LLVM |
| 10 | `convert-index-to-llvm` | Standard | Index → i64 |
| 11 | `canonicalize` | Cleanup | Simplification |
| 12 | `cse` | Cleanup | Common subexpression elim |
| 13 | `gpu-to-llvm` | GPU | GPU runtime → LLVM calls |
| 14 | `gpu.module(convert-gpu-to-nvvm)` | GPU | GPU ops → NVVM (in kernel) |
| 15 | `gpu.module(canonicalize)` | Cleanup | Kernel cleanup |
| 16 | `gpu.module(cse)` | Cleanup | Kernel CSE |
| 17 | `gpu.module(reconcile-unrealized-casts)` | Cleanup | Cast cleanup |
| 18 | `gpu-to-llvm` (repeat) | GPU | Final host lowering |
| 19 | **`gpu-module-to-binary`** | **GPU** | **Compile to PTX!** |
| 20-23 | cleanup passes | Cleanup | Final cleanup |

**Critical passes for GPU**:
- Pass 0: Lowers high-level NVGPU ops
- Pass 1: Separates host/device code
- Pass 14: Converts kernel GPU ops
- **Pass 19: Generates PTX assembly** ← This is where the magic happens!

---

## Your IR Dumps

All transformations saved to:
```
/home/jeromeku/llvm-project/mlir/test/Examples/NVGPU/ch3_mlir_dump/builtin_module_no-symbol-name/
```

| File | After Pass | Key Changes |
|------|------------|-------------|
| `module_before.mlir` | (input) | High-level NVGPU ops |
| `0_convert-nvgpu-to-nvvm.mlir` | convert-nvgpu-to-nvvm | NVVM intrinsics appear |
| `1_gpu-kernel-outlining.mlir` | gpu-kernel-outlining | Kernel extracted |
| `2_convert-vector-to-scf.mlir` | convert-vector-to-scf | Vector loops |
| `3_convert-scf-to-cf.mlir` | convert-scf-to-cf | Control flow |
| `4_convert-nvvm-to-llvm.mlir` | convert-nvvm-to-llvm | LLVM intrinsics |
| `5_convert-func-to-llvm.mlir` | convert-func-to-llvm | LLVM functions |
| `6_expand-strided-metadata.mlir` | expand-strided-metadata | Memref expansion |
| `7_nvvm-attach-target.mlir` | nvvm-attach-target | Target attributes |
| `8_lower-affine.mlir` | lower-affine | Affine lowered |
| `9_convert-arith-to-llvm.mlir` | convert-arith-to-llvm | LLVM arithmetic |
| `10_convert-index-to-llvm.mlir` | convert-index-to-llvm | Index → i64 |
| `11_canonicalize.mlir` | canonicalize | Simplified |
| `12_cse.mlir` | cse | Common subexpr elim |
| `13_gpu-to-llvm.mlir` | gpu-to-llvm | Runtime calls |
| **`14_gpu-module-to-binary.mlir`** | **gpu-module-to-binary** | **PTX embedded!** |
| `15_convert-math-to-llvm.mlir` | convert-math-to-llvm | Math ops |
| `16_canonicalize.mlir` | canonicalize | Final cleanup |
| `17_cse.mlir` | cse | Final CSE |
| `18_reconcile-unrealized-casts.mlir` | reconcile-unrealized-casts | **Final output** |

---

## Key Source Files

### Pass Implementations

| Pass | Source File | Line Count |
|------|-------------|------------|
| convert-nvgpu-to-nvvm | `mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp` | ~1750 lines |
| gpu-kernel-outlining | `mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp` | ~400 lines |
| convert-gpu-to-nvvm | `mlir/lib/Conversion/GPUToNVVM/GPUToNVVM.cpp` | ~1200 lines |
| gpu-module-to-binary | `mlir/lib/Dialect/GPU/Transforms/SerializeToBlob.cpp` | ~200 lines |
| gpu-to-llvm | `mlir/lib/Conversion/GPUToLLVM/GPUToLLVM.cpp` | ~600 lines |

### Dialect Definitions (TableGen)

| Dialect | ODS File | Key Ops |
|---------|----------|---------|
| NVGPU | `mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td` | `tma.create.descriptor`, `warpgroup.mma`, `mbarrier.*` |
| GPU | `mlir/include/mlir/Dialect/GPU/IR/GPUOps.td` | `launch`, `launch_func`, `thread_id`, `barrier` |
| NVVM | `mlir/include/mlir/Dialect/NVVM/IR/NVVM.td` | `read.ptx.sreg.*`, `mbarrier.*`, `wgmma.*` |
| LLVM | `mlir/include/mlir/Dialect/LLVMIR/LLVMOps.td` | `func`, `call`, `load`, `store`, `getelementptr` |

---

## Exploring the Code

### View a Pass Implementation

```bash
# convert-nvgpu-to-nvvm
less mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp

# Look for pattern classes
grep "struct.*ToNVVM" mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp
```

### View Operation Definitions

```bash
# NVGPU dialect operations
less mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td

# Search for specific op
grep "def.*MBarrierCreate" mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td
```

### Compare IR Dumps

```bash
cd mlir/test/Examples/NVGPU/ch3_mlir_dump/builtin_module_no-symbol-name

# Compare consecutive passes
diff -u 0_convert-nvgpu-to-nvvm.mlir 1_gpu-kernel-outlining.mlir | less

# Find where PTX appears
grep -n "assembly =" 14_gpu-module-to-binary.mlir
```

---

## Understanding MLIR Operations

### Operation Structure

```mlir
%result = dialect.operation(%operand1, %operand2) {attribute = value} : (type1, type2) -> result_type
```

**Example**:
```mlir
%4 = nvgpu.tma.create.descriptor %cast box[%c128, %c64] : memref<*xf16>
     -> <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, ...>
```

- `%4` - Result SSA value
- `nvgpu.tma.create.descriptor` - Operation name
- `%cast` - Operand (memref)
- `box[%c128, %c64]` - Attribute (dimensions)
- `: memref<*xf16>` - Operand type
- `-> <tensor = ...>` - Result type (TMA descriptor)

### Traits

Operations have **traits** (compile-time properties):
- `Pure` - No side effects
- `SameOperandsAndResultType` - Type constraints
- `Terminator` - Ends basic block
- `IsolatedFromAbove` - Cannot reference parent SSA values

### Interfaces

Operations implement **interfaces** (runtime behavior):
- `MemoryEffectsOpInterface` - Memory read/write effects
- `InferTypeOpInterface` - Type inference
- `DestinationStyleOpInterface` - Has destination operand
- `LoopLikeOpInterface` - Loop semantics

---

## Common Patterns

### Pattern 1: High-Level Op → Runtime Library Call

```cpp
// Example: nvgpu.tma.create.descriptor → llvm.call @mgpuTensorMapEncodeTiledMemref

struct TMACreateDescriptorLowering : public ConvertOpToLLVMPattern<...> {
  LogicalResult matchAndRewrite(...) const override {
    // 1. Extract operands
    Value memref = op.getMemref();

    // 2. Convert to LLVM types
    Type llvmPtrTy = converter.convertType(memref.getType());

    // 3. Create runtime call
    Value result = rewriter.create<LLVM::CallOp>(
        loc, runtimeFunction, llvmArgs);

    // 4. Replace original op
    rewriter.replaceOp(op, result);
    return success();
  }
};
```

### Pattern 2: Op → NVVM Intrinsic

```cpp
// Example: gpu.thread_id → nvvm.read.ptx.sreg.tid.x

struct GPUThreadIdOpLowering : public ConvertOpToLLVMPattern<...> {
  LogicalResult matchAndRewrite(...) const override {
    // 1. Choose intrinsic based on dimension
    NVVM::ReadSRegOp::PtxSReg sreg;
    switch (op.getDimension()) {
    case gpu::Dimension::x:
      sreg = NVVM::ReadSRegOp::PtxSReg::tid_x;
      break;
    // ...
    }

    // 2. Create NVVM intrinsic
    Value result = rewriter.create<NVVM::ReadSRegOp>(loc, i32Type, sreg);

    // 3. Cast if needed
    if (needsCast)
      result = rewriter.create<arith::IndexCastOp>(loc, indexType, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};
```

### Pattern 3: Outlining (Region Extraction)

```cpp
// Example: gpu.launch → gpu.module + gpu.launch_func

void outlineKernel(gpu::LaunchOp launchOp) {
  // 1. Create gpu.module
  auto gpuModule = builder.create<gpu::GPUModuleOp>(loc, moduleName);

  // 2. Create gpu.func inside module
  auto gpuFunc = builder.create<gpu::GPUFuncOp>(loc, funcName, funcType);

  // 3. Move launch body into function
  builder.mergeBlocks(&launchOp.getBody(), &gpuFunc.getBody());

  // 4. Replace launch with launch_func
  builder.create<gpu::LaunchFuncOp>(loc, gpuFunc, gridSize, blockSize, args);

  // 5. Erase original launch
  launchOp.erase();
}
```

---

## FAQ

**Q: Where does PTX generation actually happen?**
A: In `gpu-module-to-binary` pass (Pass 19). It:
1. Translates `gpu.module` to LLVM IR
2. Invokes LLVM NVPTX backend
3. Generates PTX assembly
4. Embeds PTX as string in `gpu.binary` operation

**Q: Why so many passes?**
A: Layered lowering:
- High-level abstractions → Mid-level intrinsics
- Mid-level intrinsics → LLVM IR
- LLVM IR → Target assembly
Each layer is reusable for different targets.

**Q: What's the difference between NVGPU, GPU, and NVVM dialects?**
- **NVGPU**: High-level NVIDIA-specific ops (TMA, warpgroup MMA)
- **GPU**: Generic GPU ops (thread_id, barrier, launch)
- **NVVM**: NVIDIA LLVM-based intrinsics (low-level PTX operations)

**Q: Can I skip passes?**
A: Some passes are optional (like canonicalize, cse), but the core GPU passes must run in order.

**Q: How do I debug a pass failure?**
A: Enable IR printing:
```python
pm.enable_ir_printing(
    print_before_all=True,
    print_after_all=True,
    print_after_failure=True,
    tree_printing_dir_path="./debug_dumps"
)
```

---

## Next Steps

1. **Understand the flow**: Read the Executive Summary
2. **See transformations**: Review Before/After Examples
3. **Explore your IR**: Browse the ch3_mlir_dump directory
4. **Deep dive**: Pick a pass and read its source code
5. **Experiment**: Modify the pipeline and observe changes

**Need line-by-line C++ trace of a specific pass?** Just ask!

---

## Related Documentation

### Pass Implementation Deep Dives (Line-by-Line with Data Structures)

**ALL 4 CRITICAL GPU PASSES NOW DOCUMENTED!**

#### Pass 0: convert-nvgpu-to-nvvm
**[DETAILED-PASS-TRACE-1-NVGPU-TO-NVVM.md](DETAILED-PASS-TRACE-1-NVGPU-TO-NVVM.md)** - COMPREHENSIVE (1,473 lines)
- Complete line-by-line trace of convert-nvgpu-to-nvvm pass
- Full call stacks for all pattern executions
- In-depth explanation of MLIR/LLVM data structures:
  - TrailingObjects pattern for Operation layout
  - Intrusive linked lists for Block/Operation
  - SymbolTable for name resolution
  - SSA use-def chains
  - Attribute uniquing system
  - MemRef descriptors and getStridedElementPtr
- Detailed patterns covered:
  - NVGPUMBarrierCreateLowering (global variable generation)
  - NVGPUMBarrierInitLowering (pointer computation, GEP)
- Memory layouts and address space explanations

#### Pass 1: gpu-kernel-outlining
**[DETAILED-PASS-TRACE-2-GPU-KERNEL-OUTLINING.md](DETAILED-PASS-TRACE-2-GPU-KERNEL-OUTLINING.md)** - COMPREHENSIVE
- Transforms inline `gpu.launch` → separate `gpu.module` + `gpu.launch_func`
- Complete dependency analysis and argument marshalling
- Thread/block ID injection pattern
- Region cloning with IRMapping
- Symbol table management and nested references
- IsolatedFromAbove trait enforcement
- Static vs dynamic kernel launch configuration
- Why kernel outlining enables separate compilation

#### Pass 14: convert-gpu-to-nvvm (Nested in gpu.module)
**[DETAILED-PASS-TRACE-3-GPU-TO-NVVM.md](DETAILED-PASS-TRACE-3-GPU-TO-NVVM.md)** - COMPREHENSIVE
- Generic GPU ops → NVVM intrinsics (runs inside `gpu.module`)
- Template-based pattern registration for index operations
- Complete PTX special register mapping:
  - `gpu.thread_id` → `nvvm.read.ptx.sreg.tid.x`
  - `gpu.block_id` → `nvvm.read.ptx.sreg.ctaid.x`
  - `gpu.barrier` → `nvvm.barrier0`
- Dynamic shared memory handling
- GPU function → LLVM function with `nvvm.kernel` attribute
- Nested pass manager execution
- MemRefDescriptor building pattern
- Why target abstraction layer matters

#### Pass 19: gpu-module-to-binary (PTX GENERATION!)
**[DETAILED-PASS-TRACE-4-GPU-MODULE-TO-BINARY.md](DETAILED-PASS-TRACE-4-GPU-MODULE-TO-BINARY.md)** - COMPREHENSIVE
- **WHERE PTX IS ACTUALLY GENERATED! ⭐**
- Complete compilation pipeline:
  1. MLIR → LLVM IR translation
  2. Libdevice linking (math functions)
  3. LLVM optimization passes (O2)
  4. LLVM NVPTX backend → PTX assembly
  5. Optional: ptxas → cubin binary
- Target attribute interface (polymorphic compilation)
- PTX assembly structure and instruction categories
- Register allocation and memory spaces
- Embedding PTX in `gpu.object` attribute
- `gpu.module` → `gpu.binary` transformation
- Runtime kernel loading explanation
- Why this pass is the "magic moment"

### Python/C++ Bindings and Context Management

- **PTX Extraction**: [PTX-EXTRACTION-QUICKSTART.md](PTX-EXTRACTION-QUICKSTART.md)
- **Location Tracking**: [../context_management/location-tracking-guide.md](../context_management/location-tracking-guide.md)
- **PassManager Deep Dive**: [passmanager-deep-dive.md](passmanager-deep-dive.md)
- **Value Caster Registration**: [../context_management/value-caster-registration.md](../context_management/value-caster-registration.md)
- **InsertionPoint Deep Dive**: [../context_management/insertion-point-deep-dive.md](../context_management/insertion-point-deep-dive.md)
