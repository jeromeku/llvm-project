# Detailed Pass Trace: gpu-module-to-binary (Pass 19 - PTX Generation)

## Overview

The `gpu-module-to-binary` pass is **where the magic happens** - it compiles `gpu.module` operations containing LLVM/NVVM IR into actual PTX assembly (or cubin binary). This is the final device-side compilation step.

**Sources**:
- `mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp` - Pass infrastructure
- `mlir/lib/Target/LLVM/NVVM/Target.cpp` - NVVM target serialization
- LLVM NVPTX backend - PTX code generation

**What it does**:
```
INPUT:
  gpu.module @kernel_module {
    llvm.func @kernel(...) attributes {nvvm.kernel} {
      // LLVM + NVVM intrinsics
      %0 = nvvm.read.ptx.sreg.tid.x : i32
      nvvm.barrier0
      llvm.return
    }
  }

OUTPUT:
  gpu.binary @kernel_module [
    #gpu.object<
      #nvvm.target<chip = "sm_90a">,
      assembly = ".version 8.0\n.target sm_90a\n..."  ← PTX!
    >
  ]
```

---

## Part 1: Pass Infrastructure

### Pass Entry Point

```cpp
// File: ModuleToBinary.cpp:38-73
void GpuModuleToBinaryPass::runOnOperation() {
  // Step 1: Parse compilation target (assembly, binary, etc.)
  auto targetFormat =
      llvm::StringSwitch<std::optional<CompilationTarget>>(compilationTarget)
          .Cases("offloading", "llvm", CompilationTarget::Offload)
          .Cases("assembly", "isa", CompilationTarget::Assembly)  // PTX
          .Cases("binary", "bin", CompilationTarget::Binary)      // cubin
          .Cases("fatbinary", "fatbin", CompilationTarget::Fatbin)
          .Default(std::nullopt);

  // Step 2: Build target options
  TargetOptions targetOptions(
      toolkitPath,       // CUDA toolkit path
      librariesToLink,   // libdevice.bc, etc.
      cmdOptions,        // Additional compiler flags
      elfSection,        // ELF section name (if applicable)
      *targetFormat,     // What to generate
      lazyTableBuilder   // Symbol table callback
  );

  // Step 3: Transform all gpu.module ops
  if (failed(transformGpuModulesToBinaries(
          getOperation(),
          OffloadingLLVMTranslationAttrInterface(nullptr),
          targetOptions)))
    return signalPassFailure();
}
```

**Key Design**: Plugin-based target system
- Different targets implement `TargetAttrInterface`
- NVVM target: PTX/cubin generation
- ROCDL target: AMD GCN ISA generation
- SPIR-V target: Vulkan/OpenCL SPIR-V

---

### Module Serializer

```cpp
// File: ModuleToBinary.cpp:76-115
LogicalResult moduleSerializer(
    GPUModuleOp op,
    OffloadingLLVMTranslationAttrInterface handler,
    const TargetOptions &targetOptions) {

  OpBuilder builder(op->getContext());
  SmallVector<Attribute> objects;

  // Step 1: Check for target attributes
  if (!op.getTargetsAttr())
    return op.emitError("the module has no target attributes");

  // Step 2: Serialize for each target (can have multiple!)
  for (auto targetAttr : op.getTargetsAttr()) {
    auto target = dyn_cast<gpu::TargetAttrInterface>(targetAttr);

    // Step 3: Call target-specific serialization
    std::optional<SmallVector<char, 0>> serializedModule =
        target.serializeToObject(op, targetOptions);

    if (!serializedModule) {
      op.emitError("An error happened while serializing the module.");
      return failure();
    }

    // Step 4: Create object attribute
    Attribute object =
        target.createObject(op, *serializedModule, targetOptions);

    if (!object) {
      op.emitError("An error happened while creating the object.");
      return failure();
    }

    objects.push_back(object);
  }

  // Step 5: Replace gpu.module with gpu.binary
  builder.setInsertionPointAfter(op);
  gpu::BinaryOp::create(builder, op.getLoc(), op.getName(), handler,
                        builder.getArrayAttr(objects));

  // Step 6: Erase original module
  op->erase();
  return success();
}
```

**Key Steps**:
1. Find all `#nvvm.target` attributes attached to `gpu.module`
2. For each target, serialize to PTX/cubin
3. Wrap PTX in `gpu.object` attribute
4. Create `gpu.binary` operation with embedded PTX
5. Delete original `gpu.module`

---

## Part 2: NVVM Target Serialization

### Target Attribute Interface

**Defined in**: `mlir/lib/Target/LLVM/NVVM/Target.cpp`

```cpp
// Line 58-69: Target interface implementation
class NVVMTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<NVVMTargetAttrImpl> {
public:
  // Serialize gpu.module to PTX/cubin
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  // Create gpu.object attribute with embedded PTX
  Attribute createObject(Attribute attribute, Operation *module,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
```

**Why Interface?**
- Extensibility: Add new GPU targets without modifying core
- Polymorphism: Same pass works for NVVM, ROCDL, SPIR-V
- Decoupling: Target logic separate from pass logic

---

### Serialization Pipeline

**Complete Flow**:
```
gpu.module (LLVM/NVVM IR)
  ↓
serializeToObject()
  ├─> translateModuleToLLVMIR()      // MLIR → LLVM IR
  ├─> linkBitcodeFiles()             // Link libdevice.bc
  ├─> optimizeLLVM()                 // LLVM optimization passes
  ├─> compileToPTX()                 // LLVM NVPTX backend
  └─> assembleToCubin() [optional]   // ptxas assembler
  ↓
SmallVector<char> (PTX or cubin bytes)
  ↓
createObject()
  └─> gpu.object<#nvvm.target, assembly/object="...">
```

---

### Step 1: MLIR → LLVM IR Translation

```cpp
// From SerializeGPUModuleBase class
std::optional<std::unique_ptr<llvm::Module>>
translateModuleToLLVMIR(Operation *module) {
  // Create LLVM context
  llvm::LLVMContext llvmContext;

  // Translate MLIR → LLVM IR
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(module, llvmContext);

  if (!llvmModule) {
    module->emitError("Failed to translate module to LLVM IR");
    return std::nullopt;
  }

  return llvmModule;
}
```

**What happens**:
```mlir
// MLIR
llvm.func @kernel(%arg0: !llvm.ptr) attributes {nvvm.kernel} {
  %0 = nvvm.read.ptx.sreg.tid.x : i32
  llvm.return
}
```

**Becomes LLVM IR**:
```llvm
define void @kernel(ptr %arg0) #0 {
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "kernel" }
attributes #1 = { nounwind readnone }
```

**MLIR Data Structure**: Translation Interface
```cpp
class LLVMTranslationDialectInterface {
  virtual LogicalResult
  convertOperation(Operation *op,
                   llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const;
};
```

**Why separate translation?**
- Each dialect provides its own translation logic
- NVVM dialect: Translates NVVM ops → LLVM intrinsics
- LLVM dialect: Translates LLVM ops → LLVM IR
- Composable: Multiple dialects in same module

---

### Step 2: Link Libdevice

```cpp
// Line 196-200: Load bitcode files
std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
loadBitcodeFiles(llvm::Module &module) {
  SmallVector<std::unique_ptr<llvm::Module>> bcFiles;

  // Load libdevice.bc and other bitcode files
  if (failed(loadBitcodeFilesFromList(
          module.getContext(), librariesToLink, bcFiles, true)))
    return std::nullopt;

  return bcFiles;
}
```

**What is libdevice?**
- NVIDIA's device-side standard library
- Implements math functions (sin, cos, exp, log, etc.)
- Compiled to LLVM bitcode (`libdevice.10.bc`)
- ~1.5 MB of optimized GPU code

**Example**:
```mlir
// User code
%result = math.sin %x : f32

// After convert-math-to-llvm
%result = llvm.call @__nv_sinf(%x) : (f32) -> f32

// __nv_sinf implementation comes from libdevice.bc
```

**Linking Process**:
```cpp
// Pseudo-code
for (auto &bcFile : bcFiles) {
  // Link bitcode module into main module
  if (llvm::Linker::linkModules(
          module, std::move(bcFile),
          llvm::Linker::Flags::OverrideFromSrc)) {
    return failure();
  }
}

// After linking, __nv_sinf definition is in module
// Dead code elimination will remove unused functions
```

---

### Step 3: LLVM Optimization

```cpp
std::optional<std::unique_ptr<llvm::Module>>
optimizeModule(std::unique_ptr<llvm::Module> llvmModule) {
  // Create optimization pipeline
  llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O2;

  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  // Create pass builder
  llvm::PassBuilder PB;

  // Register analyses
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Build optimization pipeline
  llvm::ModulePassManager MPM =
      PB.buildPerModuleDefaultPipeline(optLevel);

  // Run optimizations
  MPM.run(*llvmModule, MAM);

  return llvmModule;
}
```

**Key Optimizations for GPU**:
1. **Inlining**: Inline device functions
2. **Dead Code Elimination**: Remove unused code
3. **Constant Propagation**: Fold constants
4. **Loop Unrolling**: Unroll small loops
5. **Vectorization**: Use vector instructions
6. **SROA**: Scalar Replacement of Aggregates

**Example Optimization**:
```llvm
// Before
%1 = alloca i32
store i32 %tid, i32* %1
%2 = load i32, i32* %1
%3 = add i32 %2, 1

// After SROA + mem2reg
%3 = add i32 %tid, 1
```

---

### Step 4: PTX Compilation (LLVM NVPTX Backend)

**This is the critical step!**

```cpp
std::optional<std::vector<char>>
compileToPTX(llvm::Module &module, StringRef chip, StringRef features) {
  // Step 1: Get NVPTX target
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget("nvptx64", error);

  if (!target) {
    llvm::errs() << "Failed to lookup NVPTX target: " << error;
    return std::nullopt;
  }

  // Step 2: Create target machine
  llvm::TargetOptions options;
  options.AllowFPOpFusion = llvm::FPOpFusion::Fast;

  std::unique_ptr<llvm::TargetMachine> targetMachine(
      target->createTargetMachine(
          "nvptx64-nvidia-cuda",    // Triple
          chip,                      // e.g., "sm_90a"
          features,                  // e.g., "+ptx80"
          options,
          llvm::Reloc::PIC_,
          std::nullopt,
          llvm::CodeGenOpt::Aggressive));

  if (!targetMachine) {
    llvm::errs() << "Failed to create target machine";
    return std::nullopt;
  }

  // Step 3: Set data layout
  module.setDataLayout(targetMachine->createDataLayout());

  // Step 4: Create output stream
  llvm::SmallVector<char, 0> ptxBuffer;
  llvm::raw_svector_ostream ptxStream(ptxBuffer);

  // Step 5: Create pass manager for code generation
  llvm::legacy::PassManager codegenPasses;

  // Step 6: Add code generation passes
  if (targetMachine->addPassesToEmitFile(
          codegenPasses, ptxStream,
          nullptr,  // DwoOut
          llvm::CodeGenFileType::AssemblyFile)) {  // Generate assembly (PTX)
    llvm::errs() << "Target machine cannot emit PTX";
    return std::nullopt;
  }

  // Step 7: Run code generation
  codegenPasses.run(module);

  return std::vector<char>(ptxBuffer.begin(), ptxBuffer.end());
}
```

**LLVM NVPTX Backend Pipeline**:
```
LLVM IR
  ↓
Instruction Selection (Select PTX instructions)
  ↓
Register Allocation (Assign virtual registers)
  ↓
Instruction Scheduling (Reorder for ILP)
  ↓
PTX Emission (Print PTX assembly)
```

**Example Translation**:

**LLVM IR**:
```llvm
%1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%2 = add i32 %1, %arg0
```

**PTX Output**:
```ptx
mov.u32 %r1, %tid.x;
add.s32 %r2, %r1, %r0;
```

---

### Step 5: PTX to Cubin (Optional)

```cpp
std::optional<std::vector<char>>
assembleToCubin(const std::vector<char> &ptx,
                StringRef chip) {
  // Find ptxas (NVIDIA PTX assembler)
  std::string ptxasPath = findPtxas(toolkitPath);

  if (ptxasPath.empty()) {
    llvm::errs() << "Failed to find ptxas";
    return std::nullopt;
  }

  // Create temp files for PTX input and cubin output
  llvm::SmallString<128> ptxFile;
  llvm::sys::fs::createTemporaryFile("kernel", "ptx", ptxFile);

  llvm::SmallString<128> cubinFile;
  llvm::sys::fs::createTemporaryFile("kernel", "cubin", cubinFile);

  // Write PTX to file
  std::error_code EC;
  llvm::raw_fd_ostream ptxOut(ptxFile, EC);
  ptxOut.write(ptx.data(), ptx.size());
  ptxOut.close();

  // Run ptxas
  std::string chipArg = ("--gpu-name=" + chip).str();
  SmallVector<StringRef> args = {
      ptxasPath,
      chipArg,            // --gpu-name=sm_90a
      "-o", cubinFile,    // Output file
      ptxFile             // Input file
  };

  std::string errorMsg;
  int result = llvm::sys::ExecuteAndWait(
      ptxasPath, args, std::nullopt, {}, 0, 0, &errorMsg);

  if (result != 0) {
    llvm::errs() << "ptxas failed: " << errorMsg;
    return std::nullopt;
  }

  // Read cubin file
  auto cubinBuffer = llvm::MemoryBuffer::getFile(cubinFile);
  if (!cubinBuffer) {
    return std::nullopt;
  }

  return std::vector<char>(
      cubinBuffer.get()->getBufferStart(),
      cubinBuffer.get()->getBufferEnd());
}
```

**What is cubin?**
- **PTX**: Text assembly (portable across GPU architectures)
- **cubin**: Binary executable (specific to GPU architecture)
- **ptxas**: NVIDIA's PTX assembler
- **JIT compilation**: PTX → SASS at runtime (flexible)
- **AOT compilation**: cubin → SASS at build time (faster startup)

**File Sizes** (approximate):
- PTX (text): ~100 KB for complex kernel
- cubin (binary): ~50 KB (compressed)
- SASS (actual GPU instructions): Generated at load time from cubin

---

## Part 3: Creating GPU Binary Operation

### Step 6: Create gpu.object Attribute

```cpp
Attribute createObject(Attribute attribute, Operation *module,
                       const SmallVector<char, 0> &object,
                       const gpu::TargetOptions &options) const {
  auto target = cast<NVVMTargetAttr>(attribute);

  DenseMap<StringAttr, Attribute> properties;

  // Add optimization level
  properties[StringAttr::get(context, "O")] =
      IntegerAttr::get(IntegerType::get(context, 32), target.getO());

  // Add timing info (if available)
  if (options.getCompilationTimeProperty())
    properties[StringAttr::get(context, "LLVMIRToISATimeInMs")] =
        options.getCompilationTimeProperty();

  // Create object attribute based on target format
  StringRef data(object.data(), object.size());

  if (options.getFormat() == CompilationTarget::Assembly) {
    // PTX is stored as string
    return gpu::ObjectAttr::get(
        context, target,
        gpu::KernelTableAttr(),  // No kernel table for PTX
        StringAttr::get(context, data),  // assembly attribute
        DictAttr::get(context, properties));
  } else {
    // cubin is stored as binary blob
    return gpu::ObjectAttr::get(
        context, target,
        gpu::KernelTableAttr(),
        nullptr,  // No assembly
        DictAttr::get(context, properties),
        data);    // object attribute (binary)
  }
}
```

**GPU Object Attribute Structure**:
```mlir
#gpu.object<
  #nvvm.target<chip = "sm_90a", features = "+ptx80">,
  properties = {O = 2 : i32, LLVMIRToISATimeInMs = 15 : i64},
  assembly = ".version 8.0\n.target sm_90a\n..."  // PTX string
>
```

---

### Step 7: Create gpu.binary Operation

```cpp
// File: ModuleToBinary.cpp:111-112
builder.setInsertionPointAfter(op);
gpu::BinaryOp::create(builder, op.getLoc(), op.getName(), handler,
                      builder.getArrayAttr(objects));

op->erase();  // Delete original gpu.module
```

**Generated IR**:
```mlir
gpu.binary @gemm_128_128_64_kernel [
  #gpu.object<
    #nvvm.target<chip = "sm_90a", features = "+ptx80">,
    properties = {LLVMIRToISATimeInMs = 15 : i64, O = 2 : i32},
    assembly = "//\n// Generated by LLVM NVPTX Back-End\n//\n\n
      .version 8.0\n
      .target sm_90a\n
      .address_size 64\n
      \n
      // External declarations\n
      .extern .shared .align 16 .b8 __dynamic_shmem__0[];\n
      .shared .align 8 .b8 __mbarrier[8];\n
      \n
      .visible .entry gemm_128_128_64_kernel(...) {\n
        .reg .pred %p<2>;\n
        .reg .b32 %r<145>;\n
        .reg .b64 %rd<26>;\n
        \n
        mov.u32 %r137, %tid.x;\n
        @%p1 mbarrier.init.shared.b64 [%r1], %r2;\n
        wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {...};\n
        st.global.b32 [%rd25], %r9;\n
        ret;\n
      }\n
    "
  >
]
```

**Key Points**:
- Symbol name preserved: `@gemm_128_128_64_kernel`
- Can have multiple objects (for multi-targeting)
- PTX is embedded as a string literal
- Properties track compilation metadata

---

## Part 4: PTX Assembly Structure

### Generated PTX Anatomy

```ptx
//
// Generated by LLVM NVPTX Back-End
//

// PTX ISA version
.version 8.0

// Target GPU architecture
.target sm_90a

// Pointer size
.address_size 64

// External shared memory declaration
.extern .shared .align 16 .b8 __dynamic_shmem__0[];

// Static shared memory
.shared .align 8 .b8 __mbarrier[8];

// Kernel entry point
.visible .entry gemm_128_128_64_kernel(
    .param .u64 gemm_128_128_64_kernel_param_0,  // TMA desc A
    .param .u64 gemm_128_128_64_kernel_param_1,  // TMA desc B
    .param .u64 gemm_128_128_64_kernel_param_2,  // Output ptr
    // ... more parameters ...
) {
    // Register declarations
    .reg .pred %p<2>;       // Predicate registers
    .reg .b32 %r<145>;      // 32-bit registers
    .reg .b64 %rd<26>;      // 64-bit registers

    // Kernel body
    mov.u32 %r137, %tid.x;  // Get thread ID

    // Barrier initialization
    @%p1 mbarrier.init.shared.b64 [%r1], %r2;

    // Matrix multiply
    wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
        {%r0, %r1, ..., %r63},  // Accumulators (64 registers!)
        {%r64, %r65},           // Matrix A descriptor
        {%r66, %r67},           // Matrix B descriptor
        %p0,                    // Predicate
        %r68;                   // Scale

    // Store result
    st.global.b32 [%rd25], %r9;

    // Return
    ret;
}
```

### PTX Instruction Categories

| Category | Example | Description |
|----------|---------|-------------|
| **Data Movement** | `mov.u32 %r1, %r2` | Register-to-register |
| **Arithmetic** | `add.s32 %r3, %r1, %r2` | Integer/FP arithmetic |
| **Memory** | `ld.global.f32 %r1, [%rd2]` | Load/store global memory |
| **Sync** | `bar.sync 0` | Thread synchronization |
| **TMA** | `cp.async.bulk.tensor.2d.shared.global` | Tensor Memory Accelerator |
| **WGMMA** | `wgmma.mma_async.sync.aligned` | Warpgroup Matrix Multiply |
| **Control** | `bra L1` | Branch to label |
| **Predication** | `@%p1 add.s32 %r1, %r2, %r3` | Conditional execution |

### Register Types

- **Predicate**: `.reg .pred %p<N>` - 1-bit boolean (for conditionals)
- **32-bit**: `.reg .b32 %r<N>` - General purpose
- **64-bit**: `.reg .b64 %rd<N>` - Pointers, 64-bit values
- **Float**: `.reg .f32 %f<N>` - Single precision
- **Double**: `.reg .f64 %fd<N>` - Double precision

### Memory Spaces

- **Global**: `.global` - Device DRAM (slow, large)
- **Shared**: `.shared` - On-chip SRAM (fast, small)
- **Local**: `.local` - Per-thread stack
- **Constant**: `.const` - Read-only cached
- **Texture**: `.tex` - Texture cache optimized

---

## Part 5: Complete Transformation Example

### Input (Before Pass 19)

```mlir
module {
  gpu.module @gemm_128_128_64_kernel {
    llvm.func @gemm_128_128_64_kernel(
        %arg0: !llvm.ptr, %arg1: !llvm.ptr, ...
    ) attributes {nvvm.kernel} {
      %0 = nvvm.read.ptx.sreg.tid.x : i32
      %1 = llvm.mlir.addressof @__dynamic_shmem__ : !llvm.ptr<3>
      nvvm.barrier0
      llvm.return
    }
  }

  func.func @host(...) {
    gpu.launch_func @gemm_128_128_64_kernel::@gemm_128_128_64_kernel ...
  }
}
```

### Output (After Pass 19)

```mlir
module {
  gpu.binary @gemm_128_128_64_kernel [
    #gpu.object<
      #nvvm.target<chip = "sm_90a", features = "+ptx80">,
      properties = {LLVMIRToISATimeInMs = 15 : i64, O = 2 : i32},
      assembly = "
.version 8.0
.target sm_90a
.address_size 64

.extern .shared .align 16 .b8 __dynamic_shmem__0[];

.visible .entry gemm_128_128_64_kernel(...) {
  .reg .b32 %r<10>;
  .reg .b64 %rd<5>;

  mov.u32 %r0, %tid.x;
  mov.u64 %rd0, __dynamic_shmem__0;
  bar.sync 0;
  ret;
}
      "
    >
  ]

  func.func @host(...) {
    gpu.launch_func @gemm_128_128_64_kernel::@gemm_128_128_64_kernel ...
  }
}
```

**Key Changes**:
- `gpu.module` → `gpu.binary`
- LLVM/NVVM IR → PTX assembly string
- Compilation metadata attached (timing, optimization level)

---

## Part 6: Runtime Usage

### How PTX is Loaded at Runtime

**Host code** (generated by `gpu-to-llvm` pass):
```cpp
// 1. Get binary data
gpu::BinaryOp binaryOp = ...;
StringRef ptxData = binaryOp.getObjects()[0].getAssembly();

// 2. Create CUDA module from PTX
CUmodule cuModule;
cuModuleLoadDataEx(&cuModule, ptxData.data(), 0, nullptr, nullptr);

// 3. Get kernel function
CUfunction cuKernel;
cuModuleGetFunction(&cuKernel, cuModule, "gemm_128_128_64_kernel");

// 4. Launch kernel
void *args[] = {&arg0, &arg1, ...};
cuLaunchKernel(
    cuKernel,
    gridDim.x, gridDim.y, gridDim.z,    // Grid size
    blockDim.x, blockDim.y, blockDim.z, // Block size
    dynamicSharedMem,                    // Shared memory
    stream,                              // CUDA stream
    args,                                // Kernel arguments
    nullptr);
```

**JIT Compilation** (what CUDA driver does):
```
PTX (text assembly)
  ↓ CUDA Driver JIT
SASS (binary GPU instructions for specific GPU)
  ↓
GPU Execution
```

---

## Summary: gpu-module-to-binary

**Input**: `gpu.module` with LLVM/NVVM IR
**Output**: `gpu.binary` with embedded PTX assembly (or cubin)

**Complete Pipeline**:
1. **Translate**: MLIR → LLVM IR
2. **Link**: Add libdevice.bc (math functions)
3. **Optimize**: LLVM optimization passes (O2)
4. **Compile**: LLVM NVPTX backend → PTX assembly
5. **Assemble** [optional]: ptxas → cubin binary
6. **Embed**: Wrap PTX in `gpu.object` attribute
7. **Replace**: `gpu.module` → `gpu.binary`

**Key Techniques**:
- Target attribute interface (polymorphic compilation)
- LLVM backend integration (reuse LLVM infrastructure)
- Embedded resources (libdevice, PTX strings)
- Multi-targeting (can generate for multiple architectures)

**Why Critical?**
- **This is where PTX is actually generated!**
- Uses LLVM's mature NVPTX backend
- Embeds result in MLIR for further processing
- Enables runtime kernel loading

**What's Next?**
- Host code uses `gpu.binary` to load and launch kernel
- Pass 13/18 (`gpu-to-llvm`) lowers `gpu.launch_func` to CUDA API calls

---

**The Journey Complete**:
- Pass 0: NVGPU → NVVM (high-level barriers/TMA)
- Pass 1: Kernel outlining (separate host/device)
- Pass 14: GPU → NVVM (generic GPU ops)
- **Pass 19: NVVM → PTX (THE MAGIC!)** ← We are here
- Pass 18: Host runtime (launch kernels)
