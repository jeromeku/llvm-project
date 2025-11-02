# Detailed Pass Trace: gpu-kernel-outlining (Pass 1)

## Overview

The `gpu-kernel-outlining` pass transforms inline GPU kernels into separate `gpu.module` operations with standalone `gpu.func` kernels. This separates host and device code, enabling independent compilation.

**Source**: `mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp`

**What it does**:
```
BEFORE:
  func.func @host(...) {
    gpu.launch blocks(...) threads(...) {
      // Kernel body inline
      %result = gpu.thread_id x
      // ... kernel operations ...
      gpu.terminator
    }
  }

AFTER:
  gpu.module @kernel_module {
    gpu.func @kernel(...) kernel {
      // Kernel body extracted
      %result = gpu.thread_id x
      // ... kernel operations ...
      gpu.return
    }
  }

  func.func @host(...) {
    gpu.launch_func @kernel_module::@kernel
      blocks in (...) threads in (...)
      args(...)
  }
```

---

## Part 1: Pass Infrastructure

### Pass Registration

```cpp
// Line 32-34: Pass definition from TableGen
namespace mlir {
#define GEN_PASS_DEF_GPUKERNELOUTLININGPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
}
```

**TableGen Pass Definition** (`GPU/Transforms/Passes.td`):
```tablegen
def GpuKernelOutliningPass : Pass<"gpu-kernel-outlining", "ModuleOp"> {
  let summary = "Outline gpu.launch bodies to kernel functions";
  let constructor = "mlir::createGpuKernelOutliningPass()";
}
```

**Why TableGen?**
- Declarative pass definition
- Automatic boilerplate generation (registration, statistics, options)
- Consistent pass interface

### Pass Class Structure

```cpp
// Simplified pass structure
class GpuKernelOutliningPass : public impl::GpuKernelOutliningPassBase<...> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Walk all gpu.launch operations
    WalkResult walkResult = module->walk([&](gpu::LaunchOp launchOp) {
      // Outline each launch into a kernel
      if (failed(outlineKernel(launchOp)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};
```

**MLIR Data Structure**: `WalkResult`
- `advance()` - Continue walking
- `interrupt()` - Stop immediately (failure)
- `skip()` - Skip nested regions

**Why walk pattern?**
- Processes all operations of a type
- Allows early termination on failure
- Maintains IR consistency

---

## Part 2: Main Outlining Logic

### Finding LaunchOp Operations

```cpp
// Line ~300+: Main outlining entry point
LogicalResult outlineKernel(gpu::LaunchOp launchOp) {
  Location loc = launchOp.getLoc();
  OpBuilder builder(launchOp.getContext());

  // Step 1: Determine kernel arguments
  // Step 2: Create gpu.module
  // Step 3: Create gpu.func inside module
  // Step 4: Move kernel body
  // Step 5: Replace gpu.launch with gpu.launch_func
  // Step 6: Erase original launch
}
```

### Step 1: Analyzing Kernel Dependencies

**Problem**: Which values used inside the kernel need to be passed as arguments?

```cpp
// Values used inside kernel but defined outside
SetVector<Value> kernelArgs;

// Walk all operations inside launch body
launchOp.getBody().walk([&](Operation *op) {
  for (Value operand : op->getOperands()) {
    // Check if operand defined outside kernel
    if (!launchOp.getBody().isAncestor(operand.getParentRegion()))
      kernelArgs.insert(operand);
  }
});
```

**MLIR Data Structure**: `SetVector`
```cpp
template <typename T>
class SetVector {
  std::vector<T> vector;  // Maintains insertion order
  DenseSet<T> set;        // Fast membership testing
};
```

**Why SetVector?**
- Deterministic iteration order (vector)
- O(1) membership test (set)
- No duplicates

**Example from user's IR**:

Input kernel uses:
- `%4` - TMA descriptor A (defined outside)
- `%5` - TMA descriptor B (defined outside)
- `%3#0` - Output memref (defined outside)

These become kernel arguments.

---

### Step 2: Creating GPU Module

```cpp
// Create gpu.module at module scope
SymbolTable symbolTable(module);
builder.setInsertionPoint(&module.getBody()->front());

// Generate unique name
std::string kernelName = generateKernelName(launchOp);
// e.g., "gemm_128_128_64_kernel"

auto gpuModule = builder.create<gpu::GPUModuleOp>(
    loc, kernelName + "_module");

symbolTable.insert(gpuModule);
```

**MLIR Data Structure**: `gpu::GPUModuleOp`
```cpp
// Defined in GPU/IR/GPUOps.td
def GPU_GPUModuleOp : GPU_Op<"module",
    [IsolatedFromAbove, SymbolTable, Symbol]> {
  let summary = "A top-level compilation unit for GPU code";

  let regions = (region SizedRegion<1>:$bodyRegion);
}
```

**Key Traits**:
- `IsolatedFromAbove` - Cannot reference parent SSA values
- `SymbolTable` - Can contain named operations (kernels)
- `Symbol` - Has a symbol name for referencing

**Why gpu.module?**
- Compilation boundary: Separate compilation unit
- Target specification: Can attach GPU-specific attributes
- Binary container: Will contain PTX/SASS after compilation

---

### Step 3: Creating GPU Function

```cpp
// Build function type from kernel arguments
SmallVector<Type> argTypes;
for (Value arg : kernelArgs)
  argTypes.push_back(arg.getType());

FunctionType funcType = builder.getFunctionType(argTypes, /*results=*/{});

// Create gpu.func inside gpu.module
builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

auto gpuFunc = builder.create<gpu::GPUFuncOp>(
    loc, kernelName, funcType);

// Mark as kernel (not device function)
gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                 builder.getUnitAttr());
```

**MLIR Data Structure**: `gpu::GPUFuncOp`
```cpp
// GPU kernel/device function
def GPU_GPUFuncOp : GPU_Op<"func", [
    IsolatedFromAbove, FunctionOpInterface, Symbol
  ]> {
  let summary = "Function executable on GPU";

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<StrAttr>:$kernel  // "kernel" attribute marks it as entry point
  );
}
```

**`kernel` attribute**:
- Marks function as kernel entry point (vs device function)
- Affects calling convention in PTX
- Required for `gpu.launch_func` to call it

---

### Step 4: Block/Thread Index Injection

**Problem**: Kernel body uses `%arg3-%arg14` (block/thread indices), but these don't exist as function arguments.

**Solution**: Inject operations that query hardware registers:

```cpp
// Line 50-70: injectGpuIndexOperations
void injectGpuIndexOperations(Region &kernelBody, Region &launchBody,
                              IRMapping &map) {
  OpBuilder builder(kernelBody.front().begin());
  SmallVector<Value> indexOps;

  // Create operations that read hardware registers
  // Order matches gpu.launch arguments: blocks, threads, grid, block dims

  // Block indices (x, y, z)
  indexOps.push_back(builder.create<gpu::BlockIdOp>(loc, builder.getIndexType(),
                                                    gpu::Dimension::x));
  indexOps.push_back(builder.create<gpu::BlockIdOp>(loc, builder.getIndexType(),
                                                    gpu::Dimension::y));
  indexOps.push_back(builder.create<gpu::BlockIdOp>(loc, builder.getIndexType(),
                                                    gpu::Dimension::z));

  // Thread indices (x, y, z)
  indexOps.push_back(builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(),
                                                     gpu::Dimension::x));
  // ... similar for y, z

  // Grid dimensions (x, y, z)
  indexOps.push_back(builder.create<gpu::GridDimOp>(loc, builder.getIndexType(),
                                                    gpu::Dimension::x));
  // ... similar for y, z

  // Block dimensions (x, y, z)
  indexOps.push_back(builder.create<gpu::BlockDimOp>(loc, builder.getIndexType(),
                                                     gpu::Dimension::x));
  // ... similar for y, z

  // Map original block arguments to these operations
  for (auto [index, op] : enumerate(indexOps))
    map.map(launchBody.front().getArgument(index), op);
}
```

**Generated IR**:
```mlir
gpu.func @kernel(...) kernel {
  %block_id_x = gpu.block_id x
  %block_id_y = gpu.block_id y
  %block_id_z = gpu.block_id z
  %thread_id_x = gpu.thread_id x
  %thread_id_y = gpu.thread_id y
  %thread_id_z = gpu.thread_id z
  %grid_dim_x = gpu.grid_dim x
  %grid_dim_y = gpu.grid_dim y
  %grid_dim_z = gpu.grid_dim z
  %block_dim_x = gpu.block_dim x
  %block_dim_y = gpu.block_dim y
  %block_dim_z = gpu.block_dim z

  // Original kernel body follows
  // References to %arg3-%arg14 remapped to above values
}
```

**MLIR Data Structure**: `IRMapping`
```cpp
class IRMapping {
  DenseMap<Value, Value> valueMap;
  DenseMap<Block*, Block*> blockMap;

  void map(Value from, Value to) { valueMap[from] = to; }
  Value lookup(Value from) { return valueMap[from]; }
};
```

**Why IRMapping?**
- Used during region cloning
- Remaps references to cloned operations
- Handles both values and blocks

---

### Step 5: Moving Kernel Body

```cpp
// Clone launch body into kernel function body
Region &kernelBody = gpuFunc.getBody();
IRMapping map;

// Map block/thread args
injectGpuIndexOperations(kernelBody, launchOp.getBody(), map);

// Map kernel arguments to function arguments
for (auto [i, arg] : enumerate(kernelArgs))
  map.map(arg, kernelBody.front().getArgument(i + 12));  // +12 for indices

// Clone all operations
for (Operation &op : launchOp.getBody().front()) {
  if (isa<gpu::TerminatorOp>(op))
    continue;  // Skip terminator, will add gpu.return

  builder.clone(op, map);
}

// Add return terminator
builder.create<gpu::ReturnOp>(loc);
```

**MLIR Data Structure**: `Operation::clone()`
```cpp
Operation *Operation::clone(IRMapping &map) {
  // Create new operation with same name
  Operation *newOp = Operation::create(
      getLoc(), getName(), getResultTypes(), ...);

  // Clone operands (remapped)
  for (Value operand : getOperands())
    newOp->setOperand(i, map.lookupOrDefault(operand));

  // Clone attributes
  newOp->setAttrs(getAttrs());

  // Recursively clone nested regions
  for (Region &region : getRegions())
    region.cloneInto(&newOp->getRegion(i), map);

  return newOp;
}
```

**Why clone instead of move?**
- Cloning allows remapping values
- Original launch preserved until replacement complete
- Safer: Can roll back on failure

---

### Step 6: Creating launch_func

```cpp
// Replace gpu.launch with gpu.launch_func
builder.setInsertionPoint(launchOp);

// Extract grid/block sizes from gpu.launch
Value gridSizeX = launchOp.getGridSizeX();
Value gridSizeY = launchOp.getGridSizeY();
Value gridSizeZ = launchOp.getGridSizeZ();
Value blockSizeX = launchOp.getBlockSizeX();
Value blockSizeY = launchOp.getBlockSizeY();
Value blockSizeZ = launchOp.getBlockSizeZ();

// Create launch_func
auto launchFuncOp = builder.create<gpu::LaunchFuncOp>(
    loc,
    gpuFunc,                                    // Kernel to call
    gpu::KernelDim3{gridSizeX, gridSizeY, gridSizeZ},   // Grid size
    gpu::KernelDim3{blockSizeX, blockSizeY, blockSizeZ}, // Block size
    launchOp.getDynamicSharedMemorySize(),      // Shared memory
    kernelArgs);                                 // Arguments
```

**MLIR Data Structure**: `gpu::LaunchFuncOp`
```cpp
def GPU_LaunchFuncOp : GPU_Op<"launch_func", [
    AttrSizedOperandSegments, GPU_AsyncOpInterface
  ]> {
  let summary = "Launches a function as a GPU kernel";

  let arguments = (ins
    FlatSymbolRefAttr:$kernel,      // @module::@kernel
    Variadic<Index>:$gridSizeX,     // Can be dynamic
    Variadic<Index>:$gridSizeY,
    Variadic<Index>:$gridSizeZ,
    Variadic<Index>:$blockSizeX,
    Variadic<Index>:$blockSizeY,
    Variadic<Index>:$blockSizeZ,
    Optional<I32>:$dynamicSharedMemorySize,
    Variadic<AnyType>:$kernelOperands  // Actual kernel arguments
  );
}
```

**Symbol Reference Format**: `@module_name::@kernel_name`
- Nested symbol reference
- Module contains kernel
- Enables separate compilation

---

### Step 7: Static vs Dynamic Sizes

**Problem**: Grid/block sizes can be compile-time constants OR runtime values.

**Solution**: Store static sizes as attributes when possible:

```cpp
// Check if size is constant
if (auto constantOp = gridSizeX.getDefiningOp<arith::ConstantOp>()) {
  IntegerAttr attr = constantOp.getValue().cast<IntegerAttr>();
  launchFuncOp->setAttr("gridSizeX", attr);
  // Can remove operand, use attribute instead
}
```

**From user's IR** (1_gpu-kernel-outlining.mlir):
```mlir
gpu.func @gemm_128_128_64_kernel(...) kernel
  attributes {
    gpu.known_block_size = array<i32: 128, 1, 1>  // Static block size!
  }
```

**Why static sizes matter?**
- Optimization: Compiler knows thread count
- Register allocation: Can optimize per-thread register usage
- Occupancy: Better scheduling decisions

---

## Part 3: Complete Transformation Example

### Input IR (module_before.mlir:25-75)

```mlir
func.func @gemm_128_128_64(%A: memref<128x64xf16>, ...) {
  // ... host setup code ...

  gpu.launch blocks(%bx, %by, %bz) in (%gbx = %c1, %gby = %c1, %gbz = %c1)
             threads(%tx, %ty, %tz) in (%btx = %c128, %bty = %c1, %btz = %c1)
             dynamic_shared_memory_size %c32768 {

    // Kernel body
    %thread_id_x = gpu.thread_id x
    %7 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>
    // ... matrix multiply operations ...
    gpu.terminator
  }

  // ... more host code ...
}
```

**Dependencies identified**:
- `%4` - TMA descriptor for A
- `%5` - TMA descriptor for B
- `%3#0`, `%3#1`, `%3#2`, `%3#3`, `%3#4` - Output memref descriptor components

### Output IR (1_gpu-kernel-outlining.mlir)

**Part 1: GPU Module and Kernel**
```mlir
gpu.module @gemm_128_128_64_kernel {
  gpu.func @gemm_128_128_64_kernel(
      %arg0: !llvm.ptr,    // TMA descriptor A
      %arg1: !llvm.ptr,    // TMA descriptor B
      %arg2: !llvm.ptr,    // Output: allocated pointer
      %arg3: !llvm.ptr,    // Output: aligned pointer
      %arg4: i64,          // Output: offset
      %arg5: i64,          // Output: size[0]
      %arg6: i64,          // Output: size[1]
      %arg7: i64,          // Output: stride[0]
      %arg8: i64           // Output: stride[1]
  ) kernel attributes {
      gpu.known_block_size = array<i32: 128, 1, 1>
  } {
    // Inject index operations
    %0 = gpu.thread_id x

    // Original kernel body (cloned and remapped)
    %7 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>
    // ... operations use %arg0-%arg8 and %0 ...

    gpu.return
  }
}
```

**Part 2: Host Function**
```mlir
func.func @gemm_128_128_64(%A: memref<128x64xf16>, ...) {
  // ... host setup code (unchanged) ...

  // Launch function call replaces gpu.launch
  gpu.launch_func @gemm_128_128_64_kernel::@gemm_128_128_64_kernel
      blocks in (%c1, %c1, %c1)
      threads in (%c128, %c1, %c1)
      dynamic_shared_memory_size %c32768
      args(%4 : !llvm.ptr,           // TMA descriptor A
           %5 : !llvm.ptr,           // TMA descriptor B
           %3#0 : memref<128x128xf32>,  // Output memref
           %3#1 : ...,  // Descriptor components unpacked
           ...)

  // ... more host code (unchanged) ...
}
```

---

## Part 4: Key MLIR Data Structures & Techniques

### 1. IsolatedFromAbove Trait

```cpp
// gpu.module and gpu.func have this trait
def IsolatedFromAbove : TraitList<[
  NativeOpTrait<"IsolatedFromAbove">
]> {
  let summary = "Operation cannot reference SSA values from parent regions";
}
```

**What it enforces**:
- All values used in region must be:
  - Block arguments
  - Defined within the region
  - Constants
- **Cannot** capture parent SSA values

**Why needed?**
- Enables separate compilation
- GPU kernels run in different address space
- Clear interface (arguments only)

**Verification**:
```cpp
// MLIR verifier checks this
LogicalResult verify(Operation *op) {
  if (!op->hasTrait<OpTrait::IsolatedFromAbove>())
    return success();

  for (Region &region : op->getRegions()) {
    for (Operation &nestedOp : region.getOps()) {
      for (Value operand : nestedOp.getOperands()) {
        if (!region.isAncestor(operand.getParentRegion()))
          return emitError("value defined outside isolated region");
      }
    }
  }
  return success();
}
```

### 2. SymbolTable and Symbol References

**SymbolTable** manages named operations within a scope:

```cpp
class SymbolTable {
  DenseMap<StringAttr, Operation*> symbolMap;

  // Operations with Symbol trait have sym_name attribute
  void insert(Operation *op) {
    StringAttr name = op->getAttrOfType<StringAttr>("sym_name");
    symbolMap[name] = op;
  }

  Operation* lookup(StringAttr name) {
    return symbolMap[name];
  }
};
```

**Symbol References** refer to symbols:
- `FlatSymbolRefAttr`: `@symbol_name` (within current scope)
- `SymbolRefAttr`: `@outer::@inner` (nested scopes)

**Example**:
```mlir
// SymbolTable in builtin.module
builtin.module {
  // Symbol: @gemm_128_128_64_kernel
  gpu.module @gemm_128_128_64_kernel {
    // Symbol: @gemm_128_128_64_kernel (nested)
    gpu.func @gemm_128_128_64_kernel(...) kernel { ... }
  }

  func.func @host(...) {
    // Nested reference: @module::@kernel
    gpu.launch_func @gemm_128_128_64_kernel::@gemm_128_128_64_kernel ...
  }
}
```

**Why SymbolTable?**
- O(1) symbol lookups
- Uniqueness checking
- Lazy construction (only when needed)

### 3. WalkResult Pattern for Tree Walking

```cpp
enum class WalkResult {
  advance,    // Continue walking
  interrupt,  // Stop immediately
  skip        // Skip nested regions
};

template <typename OpT>
WalkResult Operation::walk(function_ref<WalkResult(OpT)> callback) {
  // Preorder traversal
  for (Region &region : getRegions()) {
    for (Operation &op : region.getOps()) {
      if (auto specificOp = dyn_cast<OpT>(&op)) {
        WalkResult result = callback(specificOp);
        if (result.wasInterrupted())
          return result;
        if (result == WalkResult::skip)
          continue;  // Don't walk nested regions
      }

      // Recurse into nested operations
      WalkResult result = op.walk(callback);
      if (result.wasInterrupted())
        return result;
    }
  }
  return WalkResult::advance();
}
```

**Why WalkResult?**
- Early termination on errors
- Skip expensive nested walks when not needed
- Functional style (no explicit iteration)

### 4. Argument Marshalling

**Problem**: MemRefs have descriptor structure, must be unpacked.

**From**: `memref<128x128xf32>` (high-level)
**To**: Multiple arguments (low-level):
- `ptr` - allocated pointer
- `ptr` - aligned pointer
- `i64` - offset
- `i64, i64` - sizes (2D)
- `i64, i64` - strides (2D)

**Unpacking logic**:
```cpp
SmallVector<Value> unpackMemRef(Value memref) {
  SmallVector<Value> components;
  auto memrefType = memref.getType().cast<MemRefType>();

  // Extract descriptor fields
  components.push_back(builder.create<LLVM::ExtractValueOp>(loc, memref, 0));  // allocated
  components.push_back(builder.create<LLVM::ExtractValueOp>(loc, memref, 1));  // aligned
  components.push_back(builder.create<LLVM::ExtractValueOp>(loc, memref, 2));  // offset

  // Extract sizes
  for (int i = 0; i < memrefType.getRank(); i++)
    components.push_back(builder.create<LLVM::ExtractValueOp>(loc, memref, 3 + i));

  // Extract strides
  for (int i = 0; i < memrefType.getRank(); i++)
    components.push_back(builder.create<LLVM::ExtractValueOp>(loc, memref, 3 + rank + i));

  return components;
}
```

---

## Part 5: Why This Pass Exists

### Compilation Model

**Single Module** (before outlining):
- Host and device code mixed
- Cannot compile separately
- No clear ABI boundary

**After Outlining**:
- `gpu.module` is independent compilation unit
- Can serialize to PTX/cubin
- Clear kernel ABI (arguments, launch config)
- Host code calls kernel via runtime

### Enabling Transformations

**What becomes possible**:
1. **Pass 14** (`convert-gpu-to-nvvm`) runs ONLY on `gpu.module`
2. **Pass 19** (`gpu-module-to-binary`) compiles ONLY `gpu.module` to PTX
3. Host code uses different passes (LLVM lowering)

**Nested Pass Managers**:
```python
# This is why you see: gpu.module(convert-gpu-to-nvvm)
pipeline = "builtin.module(gpu-lower-to-nvvm-pipeline)"

# Expands to:
pipeline = """
  builtin.module(
    convert-nvgpu-to-nvvm,
    gpu-kernel-outlining,        # <-- Creates gpu.module
    ...,
    gpu.module(                  # <-- Runs ONLY on gpu.module!
      convert-gpu-to-nvvm,
      canonicalize,
      cse
    ),
    ...,
    gpu-module-to-binary         # <-- Compiles gpu.module
  )
"""
```

**Why nested passes?**
- Different compilation units need different transformations
- Host: LLVM lowering, runtime calls
- Device: NVVM intrinsics, PTX generation
- Separation of concerns

---

## Summary: Kernel Outlining

**Input**: Inline `gpu.launch` with captured values
**Output**: Separate `gpu.module` with `gpu.func` kernel, host code with `gpu.launch_func`

**Key Steps**:
1. Identify kernel dependencies (captured values)
2. Create `gpu.module` at module scope
3. Create `gpu.func` with captured values as arguments
4. Inject block/thread ID operations
5. Clone kernel body with value remapping
6. Replace `gpu.launch` with `gpu.launch_func`
7. Preserve grid/block sizes (static when possible)

**Key Data Structures**:
- `IRMapping` - Value remapping during cloning
- `SymbolTable` - Symbol name management
- `IsolatedFromAbove` trait - Enforces no captured values
- `WalkResult` - Tree traversal with early termination
- Nested `SymbolRefAttr` - Cross-module symbol references

**Why Critical?**
- Enables separate compilation (host vs device)
- Creates clear ABI boundary
- Allows nested pass managers
- Required for PTX generation

---

**Next**: Pass 14 (`convert-gpu-to-nvvm`) - Converting generic GPU ops to NVVM intrinsics inside kernels.
