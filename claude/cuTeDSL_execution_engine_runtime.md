# MLIR ExecutionEngine: Module Lowering, Compilation & Runtime Invocation

## Table of Contents
1. [Overview](#overview)
2. [Complete Compilation Pipeline](#complete-compilation-pipeline)
3. [Frame-by-Frame: MLIR → LLVM IR Translation](#frame-by-frame-mlir--llvm-ir-translation)
4. [Frame-by-Frame: LLVM IR → Native Code](#frame-by-frame-llvm-ir--native-code)
5. [Symbol Table Construction & Export](#symbol-table-construction--export)
6. [CuTeDSL's Compilation Flow](#cutedsl-compilation-flow)
7. [Calling Exported Symbols from Python](#calling-exported-symbols-from-python)
8. [Output Artifacts](#output-artifacts)
9. [GPU-Specific Compilation](#gpu-specific-compilation)

---

## Overview

The MLIR ExecutionEngine performs a multi-stage compilation process that transforms MLIR IR into executable native machine code using LLVM's JIT infrastructure. This document traces the complete lowering and compilation pipeline, showing how symbols are exposed, exported, and invoked from Python.

**High-Level Pipeline:**
```
MLIR IR → LLVM IR → Optimized LLVM IR → Native Machine Code → Symbol Lookup → Execution
```

**Key Components:**
- **ModuleTranslation**: Converts MLIR dialects to LLVM IR
- **LLJIT**: LLVM's "Lazy" JIT engine for on-demand compilation
- **TMOwningSimpleCompiler**: Target machine compiler with optimization
- **RTDyldObjectLinkingLayer**: Dynamic linker for loading object code
- **SimpleObjectCache**: Optional caching to avoid recompilation

---

## Complete Compilation Pipeline

### Pipeline Stages

The ExecutionEngine performs the following steps during `ExecutionEngine::create()`:

```cpp
// Location: mlir/lib/ExecutionEngine/ExecutionEngine.cpp:232-403
Expected<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::create(Operation *m, const ExecutionEngineOptions &options,
                        std::unique_ptr<llvm::TargetMachine> tm) {
  // STAGE 1: Create ExecutionEngine instance
  auto engine = std::make_unique<ExecutionEngine>(
      options.enableObjectDump, options.enableGDBNotificationListener,
      options.enablePerfNotificationListener);

  // STAGE 2: Translate MLIR → LLVM IR
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  auto llvmModule = options.llvmModuleBuilder
                        ? options.llvmModuleBuilder(m, *ctx)
                        : translateModuleToLLVMIR(m, *ctx);  // ← Key translation step
  if (!llvmModule)
    return makeStringError("could not convert to LLVM IR");

  // STAGE 3: Detect/create target machine
  if (!tm) {
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    auto tmOrError = tmBuilderOrError->createTargetMachine();
    tm = std::move(tmOrError.get());
  }

  // STAGE 4: Set target triple and data layout
  setupTargetTripleAndDataLayout(llvmModule.get(), tm.get());

  // STAGE 5: Create wrapper functions with packed arguments
  packFunctionArguments(llvmModule.get());  // ← Creates void(void**) wrappers

  // STAGE 6: Load shared libraries and extract init/destroy functions
  // ... (shared library loading code)

  // STAGE 7: Create LLJIT with compiler and linker
  auto jit = cantFail(llvm::orc::LLJITBuilder()
                   .setCompileFunctionCreator(compileFunctionCreator)  // ← Optimization
                   .setObjectLinkingLayerCreator(objectLinkingLayerCreator)  // ← Linking
                   .setDataLayout(dataLayout)
                   .create());

  // STAGE 8: Apply optional LLVM IR transformations
  ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
  if (options.transformer)
    cantFail(tsm.withModuleDo(
        [&](llvm::Module &module) { return options.transformer(&module); }));

  // STAGE 9: Add IR module to JIT (triggers lazy compilation)
  cantFail(jit->addIRModule(std::move(tsm)));  // ← Module now ready for JIT
  engine->jit = std::move(jit);

  // STAGE 10: Register runtime symbols
  // ... (symbol registration code)

  return std::move(engine);
}
```

**Important Notes:**
- Compilation is **lazy** by default - native code is only generated when a function is first looked up
- The `transformer` callback allows applying custom LLVM optimization passes
- Shared libraries can provide init/destroy callbacks for runtime symbol injection

---

## Frame-by-Frame: MLIR → LLVM IR Translation

### Frame 1: Entry Point - `translateModuleToLLVMIR()`

**File:** [mlir/include/mlir/Target/LLVMIR/Export.h:28-31](mlir/include/mlir/Target/LLVMIR/Export.h#L28-L31)

```cpp
/// Translates a given LLVM dialect `module` into an LLVM IR module living in
/// the given context. Operates on any operation from dialects that provide a
/// registered implementation of the LLVMTranslationDialectInterface.
std::unique_ptr<llvm::Module>
translateModuleToLLVMIR(Operation *module, llvm::LLVMContext &llvmContext,
                        llvm::StringRef name = "LLVMDialectModule",
                        bool disableVerification = false);
```

**What Happens:**
- Takes an MLIR `Operation*` (typically a `ModuleOp`) and an LLVM context
- Looks up registered dialect translation interfaces
- Returns a fully-formed `llvm::Module` containing LLVM IR

**Dialects Registered by CuTeDSL's ExecutionEngine:**

From [mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp:34-37](mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp#L34-L37):
```cpp
auto &ctx = *unwrap(op)->getContext();
mlir::registerBuiltinDialectTranslation(ctx);
mlir::registerLLVMDialectTranslation(ctx);
mlir::registerOpenMPDialectTranslation(ctx);
```

For GPU workloads, CuTeDSL also registers:
- NVVM dialect translation
- ROCDL dialect translation (for AMD GPUs)
- GPU dialect translation

### Frame 2: Dialect Translation Interfaces

**Concept:** MLIR uses a **plugin architecture** for dialect translation. Each dialect provides a `LLVMTranslationDialectInterface` that knows how to convert its operations to LLVM IR.

**Interface Definition:**
```cpp
class LLVMTranslationDialectInterface {
public:
  virtual LogicalResult convertOperation(
      Operation *op, llvm::IRBuilderBase &builder,
      LLVM::ModuleTranslation &moduleTranslation) const;
};
```

**Example: GPU Launch Operation Translation**

When translating a `gpu.launch_func` operation:
```mlir
gpu.launch_func @kernel blocks in (%bx, %by, %bz)
                        threads in (%tx, %ty, %tz)
                        args(%arg0 : memref<?xf32>)
```

The GPU dialect interface generates LLVM IR that:
1. Loads the kernel binary from a global constant
2. Calls `cuLaunchKernel()` runtime function
3. Passes block/thread dimensions and arguments

### Frame 3: Module Translation Process

**File:** [mlir/lib/Target/LLVMIR/ModuleTranslation.cpp](mlir/lib/Target/LLVMIR/ModuleTranslation.cpp)

The translation process walks the MLIR module in topological order:

```cpp
// Pseudo-code showing the translation flow
class ModuleTranslation {
  llvm::Module *convertModule(Operation *mlirModule) {
    // 1. Create LLVM module
    llvm::Module *llvmModule = new llvm::Module("module", context);

    // 2. Convert types (pre-pass)
    for (auto type : mlirModule->getAllTypes())
      typeTranslation.translate(type);

    // 3. Convert globals and function declarations
    for (auto global : mlirModule->getGlobals())
      convertGlobal(global);

    // 4. Convert function bodies
    for (auto func : mlirModule->getFunctions())
      convertFunction(func);

    // 5. Verify LLVM module
    if (!disableVerification)
      llvm::verifyModule(*llvmModule);

    return llvmModule;
  }
};
```

**Type Translation:**
- `!llvm.ptr` → `i8*` (opaque pointer)
- `f32` → `float`
- `memref<?xf32>` → `{ ptr, ptr, i64, [1 x i64], [1 x i64] }` (descriptor)
- `tensor<4xf32>` → `<4 x float>` (vector)

**Operation Translation:**
- `llvm.add` → `add` instruction
- `llvm.load` → `load` instruction
- `func.call @foo` → `call @foo`
- GPU operations → runtime library calls

### Frame 4: Function Argument Packing

**File:** [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:144-200](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L144-L200)

After MLIR → LLVM IR translation, the ExecutionEngine creates **wrapper functions** to enable uniform invocation from Python. This is critical for the Python bindings.

**Original Function Signature (LLVM IR):**
```llvm
define void @my_kernel(ptr %arg0, i64 %arg1, float %arg2)
```

**Generated Wrapper (after `packFunctionArguments()`):**
```llvm
define void @_mlir_my_kernel(ptr %argList) {
  ; Extract arg0: ptr
  %arg0_ptr_ptr = getelementptr ptr, ptr %argList, i64 0
  %arg0_ptr = load ptr, ptr %arg0_ptr_ptr
  %arg0 = load ptr, ptr %arg0_ptr

  ; Extract arg1: i64
  %arg1_ptr_ptr = getelementptr ptr, ptr %argList, i64 1
  %arg1_ptr = load ptr, ptr %arg1_ptr_ptr
  %arg1 = load i64, ptr %arg1_ptr

  ; Extract arg2: float
  %arg2_ptr_ptr = getelementptr ptr, ptr %argList, i64 2
  %arg2_ptr = load ptr, ptr %arg2_ptr_ptr
  %arg2 = load float, ptr %arg2_ptr

  ; Call original function
  call void @my_kernel(ptr %arg0, i64 %arg1, float %arg2)

  ret void
}
```

**C++ Implementation:**
```cpp
static void packFunctionArguments(Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);

  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration()) continue;

    // Create wrapper: void _mlir_funcName(i8**)
    auto *newType = llvm::FunctionType::get(
        builder.getVoidTy(), builder.getPtrTy(), /*isVarArg=*/false);
    auto newName = makePackedFunctionName(func.getName());  // "_mlir_" + name
    auto interfaceFunc = module->getOrInsertFunction(newName, newType);

    // Create function body that unpacks arguments
    auto *bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);

    llvm::Value *argList = interfaceFunc->arg_begin();
    SmallVector<llvm::Value *, 8> args;

    // Unpack each argument
    for (auto [index, arg] : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = ConstantInt::get(builder.getInt64Ty(), index);
      llvm::Value *argPtrPtr = builder.CreateGEP(builder.getPtrTy(), argList, argIndex);
      llvm::Value *argPtr = builder.CreateLoad(builder.getPtrTy(), argPtrPtr);
      llvm::Value *load = builder.CreateLoad(arg.getType(), argPtr);
      args.push_back(load);
    }

    // Call original function
    llvm::Value *result = builder.CreateCall(&func, args);

    // Store result if non-void
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = ConstantInt::get(builder.getInt64Ty(), func.arg_size());
      llvm::Value *retPtrPtr = builder.CreateGEP(builder.getPtrTy(), argList, retIndex);
      llvm::Value *retPtr = builder.CreateLoad(builder.getPtrTy(), retPtrPtr);
      builder.CreateStore(result, retPtr);
    }

    builder.CreateRetVoid();
  }
}
```

**Why This Matters:**
- Python bindings can call **any** function with the same signature: `void(void**)`
- No need for Python to know the exact function signature at compile time
- Enables dynamic function lookup and invocation

---

## Frame-by-Frame: LLVM IR → Native Code

### Frame 5: LLJIT Creation and Configuration

**File:** [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:361-384](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L361-L384)

The ExecutionEngine creates an LLVM ORC LLJIT (Lazy JIT) instance:

```cpp
// Callback to create the compiler with optimization
auto compileFunctionCreator = [&](JITTargetMachineBuilder jtmb)
    -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
  if (options.jitCodeGenOptLevel)
    jtmb.setCodeGenOptLevel(*options.jitCodeGenOptLevel);
  return std::make_unique<TMOwningSimpleCompiler>(
      std::move(tm), engine->cache.get());  // ← Uses target machine + cache
};

// Create LLJIT
auto jit = cantFail(llvm::orc::LLJITBuilder()
               .setCompileFunctionCreator(compileFunctionCreator)
               .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
               .setDataLayout(dataLayout)
               .create());

// Add the LLVM IR module (does NOT compile yet - lazy compilation)
ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
if (options.transformer)
  cantFail(tsm.withModuleDo(
      [&](llvm::Module &module) { return options.transformer(&module); }));
cantFail(jit->addIRModule(std::move(tsm)));
```

**Components:**
- **LLJIT**: LLVM's lazy JIT engine - compiles functions on first lookup
- **TMOwningSimpleCompiler**: Wrapper around LLVM's code generator
- **SimpleObjectCache**: Stores compiled object code to avoid recompilation
- **transformer**: Optional callback for LLVM-level optimizations

### Frame 6: LLVM Optimization Pipeline

When the `transformer` option is provided (which it is by default), LLVM optimization passes run on the IR:

**File:** [mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp:54-60](mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp#L54-L60)

```cpp
// Create a transformer to run all LLVM optimization passes at the
// specified optimization level.
auto transformer = mlir::makeOptimizingTransformer(
    optLevel, /*sizeLevel=*/0, /*targetMachine=*/tmOrError->get());
ExecutionEngineOptions jitOptions;
jitOptions.transformer = transformer;
jitOptions.jitCodeGenOptLevel = static_cast<llvm::CodeGenOptLevel>(optLevel);
```

**Optimization Levels:**
- **O0**: No optimization, fast compilation
- **O1**: Basic optimizations (constant folding, DCE)
- **O2**: Standard optimizations (inlining, loop unrolling, vectorization) ← **Default for CuTeDSL**
- **O3**: Aggressive optimizations (may increase code size)

**Example Transformations:**
```llvm
; Before optimization
define float @foo(float %x) {
  %1 = fmul float %x, 2.0
  %2 = fadd float %1, 0.0
  ret float %2
}

; After O2 optimization
define float @foo(float %x) {
  %1 = fmul float %x, 2.0  ; Dead add removed
  ret float %1
}
```

### Frame 7: Lazy Compilation Trigger

Compilation to native code happens **lazily** when a function is first looked up:

```cpp
// Location: mlir/lib/ExecutionEngine/ExecutionEngine.cpp:414-434
Expected<void *> ExecutionEngine::lookup(StringRef name) const {
  auto expectedSymbol = jit->lookup(name);  // ← Triggers compilation if needed

  if (!expectedSymbol) {
    // Handle error...
  }

  if (void *fptr = expectedSymbol->toPtr<void *>())
    return fptr;
  return makeStringError("looked up function is null");
}
```

**What `jit->lookup()` Does:**
1. Checks if symbol is already compiled (symbol table lookup)
2. If not, triggers compilation:
   - Run LLVM code generator on the function's LLVM IR
   - Emit target-specific assembly (x86, ARM, etc.)
   - Assemble to machine code
   - Link with runtime symbols
   - Store in object cache (if enabled)
3. Returns function pointer to native code

### Frame 8: Code Generation Details

**Inside LLVM's Code Generator:**

```
LLVM IR → SelectionDAG → MachineIR → Assembly → Machine Code
```

1. **SelectionDAG**: Convert LLVM IR to target-agnostic DAG
2. **Instruction Selection**: Map DAG nodes to target instructions
3. **Register Allocation**: Assign virtual registers to physical registers
4. **Code Emission**: Generate final machine code bytes

**Example for x86-64:**
```llvm
; LLVM IR
%add = add i32 %a, %b
```
↓
```asm
; x86-64 assembly
addl %esi, %edi
movl %edi, %eax
```
↓
```
; Machine code (hex)
01 F7 89 F8
```

### Frame 9: Object Caching

**File:** [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:65-99](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L65-L99)

To avoid recompiling the same code, the ExecutionEngine can cache compiled object code:

```cpp
void SimpleObjectCache::notifyObjectCompiled(const Module *m,
                                             MemoryBufferRef objBuffer) {
  cachedObjects[m->getModuleIdentifier()] =
      MemoryBuffer::getMemBufferCopy(objBuffer.getBuffer(),
                                     objBuffer.getBufferIdentifier());
}

std::unique_ptr<MemoryBuffer> SimpleObjectCache::getObject(const Module *m) {
  auto i = cachedObjects.find(m->getModuleIdentifier());
  if (i == cachedObjects.end()) {
    LLVM_DEBUG(dbgs() << "No object for " << m->getModuleIdentifier()
                      << " in cache. Compiling.\n");
    return nullptr;  // Triggers recompilation
  }
  return MemoryBuffer::getMemBuffer(i->second->getMemBufferRef());
}
```

**Benefits:**
- Avoid recompilation across ExecutionEngine instances
- Can dump to `.o` file for offline inspection
- Significant speedup for repeated compilations

---

## Symbol Table Construction & Export

### Symbol Mangling and Naming

The ExecutionEngine uses three naming conventions for symbols:

1. **Original Name**: The MLIR function name as written
   ```mlir
   func.func @my_kernel(%arg0: memref<?xf32>) { ... }
   ```
   Symbol: `my_kernel`

2. **Packed Wrapper Name**: Prefixed with `_mlir_`
   ```cpp
   static std::string makePackedFunctionName(StringRef name) {
     return "_mlir_" + name.str();
   }
   ```
   Symbol: `_mlir_my_kernel`

3. **C Interface Name**: For ABI compatibility with C
   ```mlir
   func.func @my_kernel(...) attributes { llvm.emit_c_interface } { ... }
   ```
   Symbol: `_mlir_ciface_my_kernel`

### Symbol Export Mechanisms

The ExecutionEngine exposes symbols through multiple layers:

#### 1. LLVM Symbol Table

After JIT compilation, all symbols are registered in LLVM's ORC symbol table:

```cpp
// Location: mlir/lib/ExecutionEngine/ExecutionEngine.cpp:387-391
llvm::orc::JITDylib &mainJD = engine->jit->getMainJITDylib();
mainJD.addGenerator(
    cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
        dataLayout.getGlobalPrefix())));
```

**Symbol Resolution Order:**
1. Compiled module symbols (from JIT)
2. Shared library symbols (from `shared_libs` parameter)
3. Current process symbols (libc, CUDA runtime, etc.)

#### 2. Custom Runtime Symbols

Users can register custom runtime symbols that will be available to JIT'd code:

```cpp
// Location: mlir/lib/ExecutionEngine/ExecutionEngine.cpp:123-129
void ExecutionEngine::registerSymbols(
    llvm::function_ref<SymbolMap(MangleAndInterner)> symbolMap) {
  auto &mainJitDylib = jit->getMainJITDylib();
  cantFail(mainJitDylib.define(
      absoluteSymbols(symbolMap(llvm::orc::MangleAndInterner(
          mainJitDylib.getExecutionSession(), jit->getDataLayout())))));
}
```

**From Python:**
```python
# CuTeDSL wrapper: cutlass/_mlir/execution_engine.py:35-41
def register_runtime(self, name, ctypes_callback):
    """Register a runtime function available to the jitted code
    under the provided `name`. The `ctypes_callback` must be a
    `CFuncType` that outlives the execution engine.
    """
    callback = ctypes.cast(ctypes_callback, ctypes.c_void_p)
    self.raw_register_runtime("_mlir_ciface_" + name, callback)
```

#### 3. Shared Library Symbols

Shared libraries can provide init/destroy callbacks for dynamic symbol injection:

**File:** [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:287-311](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L287-L311)

```cpp
// Callback function types
using LibraryInitFn = void (*)(llvm::StringMap<void *> &);
using LibraryDestroyFn = void (*)();

// Loading shared library
for (auto &libPath : sharedLibPaths) {
  auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(libPath.c_str());
  void *initSym = lib.getAddressOfSymbol("__mlir_execution_engine_init");
  void *destroySim = lib.getAddressOfSymbol("__mlir_execution_engine_destroy");

  if (initSym && destroySim) {
    // Library provides callbacks - use them
    auto initFn = reinterpret_cast<LibraryInitFn>(initSym);
    initFn(exportSymbols);  // ← Library populates symbol map

    auto destroyFn = reinterpret_cast<LibraryDestroyFn>(destroySim);
    destroyFns.push_back(destroyFn);
  } else {
    // No callbacks - rely on symbol visibility
    jitDyLibPaths.push_back(libPath);
  }
}
```

**Use Case:** CUDA runtime libraries use this mechanism to inject GPU-related functions

### Symbol Lookup from Python

The Python bindings provide methods to look up and invoke symbols:

**File:** [cutlass/_mlir/execution_engine.py:12-41](cutlass/_mlir/execution_engine.py#L12-L41)

```python
class ExecutionEngine(_execution_engine.ExecutionEngine):
    def lookup(self, name):
        """Lookup a function emitted with the `llvm.emit_c_interface`
        attribute and returns a ctype callable.
        Raise a RuntimeError if the function isn't found.
        """
        func = self.raw_lookup("_mlir_ciface_" + name)  # ← Adds prefix
        if not func:
            raise RuntimeError("Unknown function " + name)
        prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        return prototype(func)

    def invoke(self, name, *ctypes_args):
        """Invoke a function with the list of ctypes arguments.
        All arguments must be pointers.
        Raise a RuntimeError if the function isn't found.
        """
        func = self.lookup(name)
        packed_args = (ctypes.c_void_p * len(ctypes_args))()
        for argNum in range(len(ctypes_args)):
            packed_args[argNum] = ctypes.cast(ctypes_args[argNum], ctypes.c_void_p)
        func(packed_args)  # ← Calls the void(void**) wrapper

    def register_runtime(self, name, ctypes_callback):
        """Register a runtime function available to the jitted code
        under the provided `name`. The `ctypes_callback` must be a
        `CFuncType` that outlives the execution engine.
        """
        callback = ctypes.cast(ctypes_callback, ctypes.c_void_p)
        self.raw_register_runtime("_mlir_ciface_" + name, callback)
```

---

## CuTeDSL's Compilation Flow

### Complete End-to-End Example

Let's trace a simple CuTeDSL kernel from Python source to GPU execution:

**Python Source:**
```python
import cutlass as cute

@cute.jit
def my_kernel(x: cute.Tensor[1024, float32]):
    tid = cute.threadIdx.x
    x[tid] = x[tid] * 2.0

# JIT compile and execute
x = cute.Tensor([1.0] * 1024)
my_kernel(x)
```

### Step 1: DSL → MLIR IR Generation

**File:** CuTeDSL's DSL layer generates MLIR IR

```mlir
module {
  func.func @my_kernel(%arg0: memref<1024xf32>) {
    %tid = gpu.thread_id x
    %0 = memref.load %arg0[%tid] : memref<1024xf32>
    %c2 = arith.constant 2.0 : f32
    %1 = arith.mulf %0, %c2 : f32
    memref.store %1, %arg0[%tid] : memref<1024xf32>
    return
  }
}
```

### Step 2: Pass Pipeline Execution

**File:** [cutlass/base_dsl/compiler.py:136-164](cutlass/base_dsl/compiler.py#L136-L164)

```python
def compile(self, module, pipeline: str, cuda_toolkit: str = "",
            arch: str = "", enable_verifier=False):
    """Compiles the module by invoking the pipeline."""
    try:
        pm = self.passmanager.PassManager.parse(pipeline)
        pm.enable_verifier(enable_verifier)
        pm.run(module.operation)  # ← Runs passes
    except Exception as e:
        # Error handling...
```

**Typical CuTeDSL Pipeline:**
```
builtin.module(
  convert-to-gpu,           // Convert host code to GPU operations
  gpu-kernel-outlining,     // Extract GPU kernel into separate module
  lower-affine,             // Lower affine dialect
  convert-scf-to-cf,        // Convert SCF to control flow
  convert-to-llvm,          // Convert to LLVM dialect
  gpu-to-cubin{            // Compile GPU kernel to CUBIN
    cubin-chip=sm_80
    cubin-format=fatbin
  }
)
```

**After Pipeline, MLIR IR becomes:**
```mlir
module {
  llvm.func @my_kernel(%arg0: !llvm.ptr) {
    // ... LLVM dialect code ...
  }

  // GPU kernel embedded as binary
  gpu.binary @my_kernel_kernel [
    #gpu.object<#nvvm.target, "...CUBIN_DATA...">
  ]
}
```

### Step 3: JIT Compilation

**File:** [cutlass/base_dsl/compiler.py:166-174](cutlass/base_dsl/compiler.py#L166-L174)

```python
def jit(self, module, opt_level: int = 2, shared_libs: Sequence[str] = ()):
    """Wraps the module in a JIT execution engine."""
    self._check_cuda_dependencies_once(shared_libs)
    return self.execution_engine.ExecutionEngine(
        module, opt_level=opt_level, shared_libs=shared_libs
    )
```

**What Happens:**
1. MLIR module → LLVM IR (via `translateModuleToLLVMIR`)
2. Function argument packing (creates `_mlir_my_kernel` wrapper)
3. LLJIT creation with optimization pipeline
4. Module added to JIT (lazy compilation - not compiled yet!)

### Step 4: Runtime Initialization

**File:** [cutlass/base_dsl/jit_executor.py:600-626](cutlass/base_dsl/jit_executor.py#L600-L626)

Before calling the kernel, CuTeDSL loads GPU binaries:

```python
def _load_binary_execution_engine(self):
    """Load the cuda module from the binary execution engine."""
    cubin_suffix = "cubin"
    if self.prefix is None:
        raise DSLRuntimeError("prefix is required to be set for binary loading")

    # Lookup the CUBIN data symbol
    cubin_data = self.engine.lookup("_".join([self.prefix, cubin_suffix]))
    if not cubin_data:
        raise RuntimeError("Unknown function " + "_".join([self.prefix, cubin_suffix]))

    # Load CUBIN into CUDA driver
    cubin_module = cuda_helpers.load_library_data(cubin_data)

    # Get kernel function pointers
    kernel_modules = collections.OrderedDict()
    for sym, attrs in self.kernel_info.items():
        kernel = cuda_helpers.get_library_kernel(cubin_module, sym)
        kernel_modules[sym] = CudaModuleAndKernel(sym, cubin_module, kernel, attrs)

    return list(kernel_modules.values())
```

**Key Points:**
- GPU kernel binaries are **embedded as global constants** in the LLVM IR
- The symbol name is constructed from the kernel prefix + `"cubin"`
- `cuModuleLoadData()` is called to load the binary into GPU memory

### Step 5: Kernel Invocation

**File:** [cutlass/base_dsl/jit_executor.py:490-508](cutlass/base_dsl/jit_executor.py#L490-L508)

```python
class JitExecutor:
    def run_compiled_program(self, exe_args):
        try:
            # Pack arguments into void** array
            packed_args = self.profiler(self._get_invoke_packed_args)(exe_args)

            # Invoke the C API function (the _mlir_my_kernel wrapper)
            self.profiler(self.jit_module.capi_func)(packed_args)

            # Check for CUDA errors
            if self.cuda_result is not None:
                if self.cuda_result.value != 0:
                    error_code = self.cuda_result.value
                    error_name = cuda_helpers._cudaGetErrorEnum(
                        cuda_helpers.cuda.CUresult(error_code)
                    )
                    raise DSLCudaRuntimeError(error_code, error_name)
                return self.cuda_result.value
            return None
        except DSLCudaRuntimeError as e:
            raise e
        except Exception as e:
            raise DSLRuntimeError(f"💥💥💥 Runtime Crash 💥💥💥", cause=e)
```

**Call Stack:**
```
Python: my_kernel(x)
  ↓
JitCompiledFunction.__call__(*args)
  ↓
JitExecutor.run_compiled_program(exe_args)
  ↓
jit_module.capi_func(packed_args)  ← ctypes function pointer
  ↓
[Native Code] _mlir_my_kernel(void** argList)  ← Argument unpacking wrapper
  ↓
[Native Code] my_kernel(memref descriptor)  ← Original function
  ↓
[Native Code] cuLaunchKernel(kernel, blocks, threads, args)  ← GPU launch
  ↓
[GPU] Kernel executes on device
```

---

## Calling Exported Symbols from Python

### Method 1: Using `ExecutionEngine.invoke()`

**Simplest method for functions with `llvm.emit_c_interface` attribute:**

```python
import ctypes
import numpy as np
from cutlass._mlir import ir, execution_engine

# Assume we have a compiled module
engine = execution_engine.ExecutionEngine(module, opt_level=2)

# Create input data
x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

# Create ctypes pointer
x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Invoke function (note: no "_mlir_ciface_" prefix needed)
engine.invoke("my_function", x_ptr)

print(x)  # Modified by the function
```

### Method 2: Using `ExecutionEngine.lookup()` + Manual Call

**For more control over argument passing:**

```python
import ctypes

# Lookup the function
func = engine.lookup("my_function")  # Returns ctypes.CFUNCTYPE(None, ctypes.c_void_p)

# Pack arguments manually
args = [x_ptr, ctypes.c_int32(42)]
packed_args = (ctypes.c_void_p * len(args))()
for i, arg in enumerate(args):
    packed_args[i] = ctypes.cast(arg, ctypes.c_void_p)

# Call function
func(packed_args)
```

### Method 3: Using `raw_lookup()` for Custom Signatures

**For functions without the packed interface:**

```python
# Get raw function pointer (integer address)
func_addr = engine.raw_lookup("custom_function")

# Create custom ctypes prototype
# Example: float custom_function(int, float)
prototype = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_int, ctypes.c_float)
func = prototype(func_addr)

# Call with typed arguments
result = func(42, 3.14)
print(f"Result: {result}")
```

### Method 4: CuTeDSL's High-Level API

**CuTeDSL provides a high-level wrapper that handles all the details:**

```python
import cutlass as cute

@cute.jit
def my_kernel(x: cute.Tensor[1024, float32], alpha: float32):
    tid = cute.threadIdx.x
    x[tid] = x[tid] * alpha

# Create input
x = cute.Tensor([1.0] * 1024)

# Direct invocation - CuTeDSL handles:
# - Argument packing
# - Type conversion
# - GPU memory management
# - Kernel launch
my_kernel(x, 2.0)

# Result is in x
print(x[:10])  # [2.0, 2.0, 2.0, ...]
```

**Under the hood, CuTeDSL:**
1. Converts Python objects to ctypes pointers
2. Packs pointers into `void**` array
3. Calls the `_mlir_ciface_my_kernel` wrapper
4. Wrapper unpacks and calls original function
5. Original function launches GPU kernel
6. GPU kernel executes
7. Results written back to memory

---

## Output Artifacts

### 1. LLVM IR (Intermediate)

**Not saved by default**, but can be obtained:

```python
# During ExecutionEngine::create(), the LLVM module is available
# Can be printed or saved before JIT compilation

# Example: Save LLVM IR to file
llvm_module.print(output_file)
```

**Example LLVM IR for packed function:**
```llvm
define void @_mlir_my_kernel(ptr %0) {
  %2 = getelementptr ptr, ptr %0, i64 0
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr ptr, ptr %0, i64 1
  %6 = load ptr, ptr %5, align 8
  %7 = load i64, ptr %6, align 4
  call void @my_kernel(ptr %4, i64 %7)
  ret void
}

define void @my_kernel(ptr %arg0, i64 %arg1) {
  ; ... function body ...
}
```

### 2. Native Object Code (.o file)

**Can be dumped using `dump_to_object_file()`:**

```python
engine = execution_engine.ExecutionEngine(module, opt_level=2)

# Trigger compilation by looking up a function
engine.lookup("my_function")

# Dump compiled object code
engine.dump_to_object_file("output.o")
```

**File:** [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:101-121](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L101-L121)

```cpp
void ExecutionEngine::dumpToObjectFile(StringRef filename) {
  if (cache == nullptr) {
    llvm::errs() << "cannot dump ExecutionEngine object code to file: "
                    "object cache is disabled\n";
    return;
  }

  // Force compilation if not already done
  if (cache->isEmpty()) {
    for (std::string &functionName : functionNames) {
      auto result = lookupPacked(functionName);
      if (!result) {
        llvm::errs() << "Could not compile " << functionName << ":\n  "
                     << result.takeError() << "\n";
        return;
      }
    }
  }

  cache->dumpToObjectFile(filename);
}
```

**Inspecting Object File:**
```bash
# View symbols
nm output.o

# Disassemble
objdump -d output.o

# View sections
readelf -S output.o
```

**Example nm output:**
```
0000000000000000 T _mlir_my_kernel
0000000000000050 T my_kernel
                 U cuLaunchKernel
                 U printf
```

### 3. GPU Binaries (PTX/CUBIN)

**CuTeDSL-specific: GPU kernels embedded in IR**

**Enable dumping with compile options:**
```python
from cutlass.base_dsl.compiler import KeepPTX, KeepCUBIN

@cute.jit(options=(KeepPTX, KeepCUBIN))
def my_kernel(...):
    # ...
```

**Or via environment variables:**
```bash
export CUTE_KEEP_PTX=1
export CUTE_KEEP_CUBIN=1
export CUTE_DUMP_DIR="./artifacts"
```

**File:** [cutlass/base_dsl/compiler.py:327-333](cutlass/base_dsl/compiler.py#L327-L333)

```python
class KeepCUBIN(BooleanBasedFileDumpOption):
    option_name = "dump-cubin-path"

class KeepPTX(BooleanBasedFileDumpOption):
    option_name = "dump-ptx-path"
```

**Generated Files:**
```
./artifacts/
  my_kernel.sm_80.ptx     # PTX assembly
  my_kernel.sm_80.cubin   # Compiled CUBIN binary
```

**PTX Example:**
```ptx
.version 7.5
.target sm_80
.address_size 64

.visible .entry my_kernel(.param .u64 ptr, .param .u32 size) {
  .reg .pred %p<2>;
  .reg .b32 %r<4>;
  .reg .b64 %rd<3>;
  .reg .f32 %f<3>;

  mov.u32 %r0, %tid.x;
  ld.param.u64 %rd0, [ptr];
  mul.wide.u32 %rd1, %r0, 4;
  add.u64 %rd2, %rd0, %rd1;
  ld.global.f32 %f0, [%rd2];
  mul.f32 %f1, %f0, 0f40000000;  // *2.0
  st.global.f32 [%rd2], %f1;
  ret;
}
```

### 4. In-Memory Function Pointers

**The primary artifact - function pointers to executable code:**

```python
# Get function pointer
func_addr = engine.raw_lookup("my_function")  # Returns integer address

# Convert to ctypes callable
prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
func = prototype(func_addr)

# Now callable from Python
func(args)
```

---

## GPU-Specific Compilation

### GPU Compilation Pipeline

CuTeDSL kernels undergo additional compilation stages for GPU execution:

```
MLIR GPU Dialect → NVVM Dialect → PTX Assembly → CUBIN Binary → GPU Memory
```

### Stage 1: GPU Kernel Outlining

**Pass:** `gpu-kernel-outlining`

**Before:**
```mlir
func.func @host_function(%arg0: memref<1024xf32>) {
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, ...)
             threads(%tx, %ty, %tz) in (%block_x = %c1024, ...) {
    %tid = gpu.thread_id x
    %val = memref.load %arg0[%tid] : memref<1024xf32>
    %doubled = arith.mulf %val, 2.0 : f32
    memref.store %doubled, %arg0[%tid] : memref<1024xf32>
    gpu.terminator
  }
  return
}
```

**After:**
```mlir
// Host function
func.func @host_function(%arg0: memref<1024xf32>) {
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index

  gpu.launch_func @my_kernel_module::@my_kernel
      blocks in (%c1, %c1, %c1)
      threads in (%c1024, %c1, %c1)
      args(%arg0 : memref<1024xf32>)
  return
}

// Kernel module (separate)
gpu.module @my_kernel_module {
  gpu.func @my_kernel(%arg0: memref<1024xf32>) kernel {
    %tid = gpu.thread_id x
    %val = memref.load %arg0[%tid] : memref<1024xf32>
    %doubled = arith.mulf %val, 2.0 : f32
    memref.store %doubled, %arg0[%tid] : memref<1024xf32>
    gpu.return
  }
}
```

### Stage 2: Lowering to NVVM

**Pass:** `convert-gpu-to-nvvm`

**After:**
```mlir
gpu.module @my_kernel_module {
  llvm.func @my_kernel(%arg0: !llvm.ptr) attributes {nvvm.kernel} {
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    %1 = llvm.sext %0 : i32 to i64
    %2 = llvm.getelementptr %arg0[%1] : (!llvm.ptr, i64) -> !llvm.ptr
    %3 = llvm.load %2 : !llvm.ptr -> f32
    %4 = llvm.fmul %3, 2.0 : f32
    llvm.store %4, %2 : f32, !llvm.ptr
    llvm.return
  }
}
```

### Stage 3: NVVM → PTX Compilation

**Pass:** `gpu-to-cubin`

**Configuration:**
```mlir
builtin.module(
  gpu-to-cubin{
    cubin-chip=sm_80           // Target architecture
    cubin-format=fatbin        // Format (PTX, CUBIN, or fatbin)
    cubin-features=+ptx70      // Optional PTX features
  }
)
```

**What Happens:**
1. NVVM dialect → LLVM IR with NVVM intrinsics
2. LLVM IR → PTX assembly (via LLVM NVPTX backend)
3. PTX → CUBIN binary (via `ptxas` compiler)
4. CUBIN embedded as `gpu.binary` attribute in IR

**Resulting MLIR:**
```mlir
gpu.binary @my_kernel_module [
  #gpu.object<#nvvm.target<chip = "sm_80">,
    properties = {O = 3 : i32},
    bin = "...BINARY_DATA...">
]

func.func @host_function(%arg0: memref<1024xf32>) {
  %kernel = gpu.binary.get @my_kernel_module::@my_kernel
  gpu.launch_func %kernel
      blocks in (%c1, %c1, %c1)
      threads in (%c1024, %c1, %c1)
      args(%arg0 : memref<1024xf32>)
  return
}
```

### Stage 4: Binary Embedding

The GPU binary is embedded as a **global constant** in the final LLVM IR:

```llvm
@my_kernel_module_cubin = private unnamed_addr constant [4096 x i8] c"\7FELF..."

define void @host_function(ptr %arg0) {
  ; Load binary
  %binary_ptr = getelementptr [4096 x i8], ptr @my_kernel_module_cubin, i64 0, i64 0

  ; Call runtime to load module
  %module = call ptr @cuModuleLoadData(ptr %binary_ptr)

  ; Get kernel function
  %kernel = call ptr @cuModuleGetFunction(ptr %module, ptr @kernel_name)

  ; Launch kernel
  call void @cuLaunchKernel(
      ptr %kernel,
      i32 1, i32 1, i32 1,      ; grid dims
      i32 1024, i32 1, i32 1,   ; block dims
      i32 0,                     ; shared mem
      ptr null,                  ; stream
      ptr %args,                 ; kernel args
      ptr null                   ; extra
  )

  ret void
}
```

### Stage 5: Runtime Loading via `initialize()`

**File:** [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:449-457](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L449-L457)

```cpp
void ExecutionEngine::initialize() {
  if (isInitialized)
    return;

  // Run global constructors (loads GPU binaries!)
  if (!jit->getTargetTriple().isAArch64())
    cantFail(jit->initialize(jit->getMainJITDylib()));

  isInitialized = true;
}
```

**What `jit->initialize()` does:**
1. Finds functions marked as constructors (`llvm.global_ctors`)
2. Executes them before main code runs
3. These constructors call `cuModuleLoadData()` with embedded CUBIN
4. GPU driver loads binary into device memory
5. Kernel functions become available for launch

**Critical for GPU:** `initialize()` **must** be called before any GPU kernel launch! CuTeDSL does this automatically in `invokePacked()`.

---

## Summary

### Key Takeaways

1. **MLIR → LLVM IR Translation:**
   - Performed by `translateModuleToLLVMIR()` using dialect-specific translation interfaces
   - Each MLIR dialect registers a translator (GPU → NVVM, Func → LLVM Func, etc.)
   - Results in a fully-formed LLVM IR module

2. **Function Argument Packing:**
   - Every function gets a `_mlir_` prefixed wrapper with signature `void(void**)`
   - Enables uniform invocation from Python without knowing exact signature
   - Python passes `ctypes.c_void_p` array, wrapper unpacks to typed arguments

3. **Lazy JIT Compilation:**
   - Native code only generated when function is first looked up
   - LLVM optimization passes run on IR (O0-O3 levels)
   - Compiled object code can be cached to avoid recompilation

4. **Symbol Export:**
   - Three naming conventions: original, `_mlir_` packed, `_mlir_ciface_` C interface
   - Symbols registered in LLVM ORC JIT symbol table
   - Custom symbols can be injected via `registerSymbols()`

5. **GPU-Specific Flow:**
   - GPU kernels outlined into separate modules
   - Compiled to PTX/CUBIN via NVVM backend
   - Embedded as binary constants in LLVM IR
   - Loaded into GPU memory via `initialize()` call
   - Launched using CUDA driver API (`cuLaunchKernel`)

6. **Python Invocation:**
   - `engine.invoke(name, *args)` - High-level, automatic argument packing
   - `engine.lookup(name)` - Returns ctypes callable
   - `engine.raw_lookup(name)` - Returns raw function pointer (integer)
   - CuTeDSL provides additional high-level wrappers for seamless integration

### Compilation Timeline

```
┌─────────────────────────────────────────────────────────────────┐
│ ExecutionEngine::create()                                       │
├─────────────────────────────────────────────────────────────────┤
│ 1. MLIR → LLVM IR            [translateModuleToLLVMIR]         │
│ 2. Function packing          [packFunctionArguments]           │
│ 3. LLJIT creation            [LLJITBuilder]                    │
│ 4. Add IR to JIT             [jit->addIRModule]                │
│                              ↓                                  │
│                         [LAZY - NOT YET COMPILED]              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ engine.lookup("my_function") / engine.invoke("my_function")    │
├─────────────────────────────────────────────────────────────────┤
│ 5. JIT compilation          [LLVM CodeGen]                     │
│    - Optimization passes     [O0/O1/O2/O3]                     │
│    - Instruction selection   [SelectionDAG]                    │
│    - Register allocation                                       │
│    - Code emission          [x86/ARM machine code]             │
│ 6. Object caching           [SimpleObjectCache]                │
│ 7. Symbol registration      [Symbol Table]                     │
│                              ↓                                  │
│                         [FUNCTION POINTER RETURNED]            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ engine.initialize()          [GPU-specific only]               │
├─────────────────────────────────────────────────────────────────┤
│ 8. Run global constructors  [llvm.global_ctors]                │
│ 9. Load GPU binaries        [cuModuleLoadData]                 │
│ 10. Register kernels        [cuModuleGetFunction]              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ func(*packed_args)           [Python invocation]               │
├─────────────────────────────────────────────────────────────────┤
│ 11. Argument unpacking      [_mlir_ wrapper]                   │
│ 12. Original function call  [Native code]                      │
│ 13. GPU kernel launch       [cuLaunchKernel]                   │
│ 14. Kernel execution        [GPU hardware]                     │
└─────────────────────────────────────────────────────────────────┘
```

### File Reference Index

**Core MLIR Files:**
- [mlir/lib/ExecutionEngine/ExecutionEngine.cpp](mlir/lib/ExecutionEngine/ExecutionEngine.cpp) - Main implementation
- [mlir/include/mlir/ExecutionEngine/ExecutionEngine.h](mlir/include/mlir/ExecutionEngine/ExecutionEngine.h) - C++ API
- [mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp](mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp) - C API wrapper
- [mlir/lib/Bindings/Python/ExecutionEngineModule.cpp](mlir/lib/Bindings/Python/ExecutionEngineModule.cpp) - Python bindings
- [mlir/lib/Target/LLVMIR/ConvertToLLVMIR.cpp](mlir/lib/Target/LLVMIR/ConvertToLLVMIR.cpp) - MLIR→LLVM translation
- [mlir/lib/Target/LLVMIR/ModuleTranslation.cpp](mlir/lib/Target/LLVMIR/ModuleTranslation.cpp) - Translation implementation

**CuTeDSL Files:**
- [cutlass/_mlir/execution_engine.py](cutlass/_mlir/execution_engine.py) - Python wrapper with convenience methods
- [cutlass/base_dsl/compiler.py](cutlass/base_dsl/compiler.py) - Compilation pipeline
- [cutlass/base_dsl/jit_executor.py](cutlass/base_dsl/jit_executor.py) - JIT execution and GPU loading

---

*This document provides a complete trace of the MLIR ExecutionEngine's compilation pipeline, from MLIR IR to native machine code execution, with specific focus on CuTeDSL's GPU compilation flow and Python invocation patterns.*
