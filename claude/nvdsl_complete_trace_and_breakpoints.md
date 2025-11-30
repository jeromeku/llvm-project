# NVDSL Complete Call Trace & Debugging Guide

This document provides frame-by-frame traces from Python → Bindings → MLIR-C → MLIR C++ → LLVM for three critical operations:

1. **ExecutionEngine Construction** (Module → LLVM IR → Executable)
2. **engine.initialize()** (Running global constructors / GPU binary loading)
3. **engine.invoke()** (Executing the compiled function)

---

## Part 1: ExecutionEngine Construction & Compilation

### Python Entry Point

**File**: [mlir/test/Examples/NVGPU/tools/nvgpucompiler.py:36-40](mlir/test/Examples/NVGPU/tools/nvgpucompiler.py#L36-L40)

```python
def jit(self, module: ir.Module) -> execution_engine.ExecutionEngine:
    """Wraps the module in a JIT execution engine."""
    return execution_engine.ExecutionEngine(
        module, opt_level=self.opt_level, shared_libs=self.shared_libs
    )
```

**GDB Breakpoint**: N/A (Python level)

---

### Frame 1: Python ExecutionEngine Wrapper Constructor

**File**: [mlir/python/mlir/execution_engine.py:14](mlir/python/mlir/execution_engine.py#L14)

```python
class ExecutionEngine(_execution_engine.ExecutionEngine):
    # This inherits from the C++ extension module
```

The constructor delegates directly to the nanobind C++ extension.

**GDB Breakpoint**: N/A (Python wrapper, no code)

---

### Frame 2: Nanobind C++ Binding Layer

**File**: [mlir/lib/Bindings/Python/ExecutionEngineModule.cpp:74-89](mlir/lib/Bindings/Python/ExecutionEngineModule.cpp#L74-L89)

```cpp
nb::class_<PyExecutionEngine>(m, "ExecutionEngine")
    .def(
        "__init__",
        [](PyExecutionEngine &self, MlirModule module, int optLevel,
           const std::vector<std::string> &sharedLibPaths,
           bool enableObjectDump) {
            llvm::SmallVector<MlirStringRef, 4> libPaths;
            for (const std::string &path : sharedLibPaths)
              libPaths.push_back({path.c_str(), path.length()});

            // Call the C API to create the execution engine
            MlirExecutionEngine executionEngine =
                mlirExecutionEngineCreate(module, optLevel, libPaths.size(),
                                          libPaths.data(), enableObjectDump);

            if (mlirExecutionEngineIsNull(executionEngine))
              throw std::runtime_error(
                  "Failure while creating the ExecutionEngine.");
            new (&self) PyExecutionEngine(executionEngine);
        },
        nb::arg("module"), nb::arg("opt_level") = 2,
        nb::arg("shared_libs") = nb::list(),
        nb::arg("enable_object_dump") = true,
        "Create a new ExecutionEngine instance for the given Module...")
```

**What happens**:
- Converts Python module to `MlirModule` (opaque C struct wrapper)
- Converts shared library paths to C-style strings
- Calls the C API function `mlirExecutionEngineCreate()`
- Wraps the returned pointer in a PyExecutionEngine object

**GDB Breakpoint**:
```gdb
break mlir::python::ExecutionEngineModule.cpp:83
# Or use the lambda location
break 'mlir::python::(anonymous namespace)::$_0::operator()'
```

---

### Frame 3: MLIR C API Layer

**File**: [mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp:22-69](mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp#L22-L69)

```cpp
extern "C" MlirExecutionEngine
mlirExecutionEngineCreate(MlirModule op, int optLevel, int numPaths,
                          const MlirStringRef *sharedLibPaths,
                          bool enableObjectDump) {
  // One-time initialization of LLVM native target
  static bool initOnce = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser(); // needed for inline_asm
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)initOnce;

  // Register dialect translations (MLIR → LLVM IR)
  auto &ctx = *unwrap(op)->getContext();
  mlir::registerBuiltinDialectTranslation(ctx);
  mlir::registerLLVMDialectTranslation(ctx);
  mlir::registerOpenMPDialectTranslation(ctx);

  // Create JIT target machine for the host
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
    return MlirExecutionEngine{nullptr};
  }
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Failed to create a TargetMachine for the host\n";
    return MlirExecutionEngine{nullptr};
  }

  // Convert shared library paths
  SmallVector<StringRef> libPaths;
  for (unsigned i = 0; i < static_cast<unsigned>(numPaths); ++i)
    libPaths.push_back(sharedLibPaths[i].data);

  // Create LLVM optimization transformer pipeline
  auto transformer = mlir::makeOptimizingTransformer(
      optLevel, /*sizeLevel=*/0, /*targetMachine=*/tmOrError->get());

  // Set up ExecutionEngine options
  ExecutionEngineOptions jitOptions;
  jitOptions.transformer = transformer;
  jitOptions.jitCodeGenOptLevel = static_cast<llvm::CodeGenOptLevel>(optLevel);
  jitOptions.sharedLibPaths = libPaths;
  jitOptions.enableObjectDump = enableObjectDump;

  // Call the C++ ExecutionEngine::create() method
  auto jitOrError = ExecutionEngine::create(unwrap(op), jitOptions);
  if (!jitOrError) {
    consumeError(jitOrError.takeError());
    return MlirExecutionEngine{nullptr};
  }
  return wrap(jitOrError->release());
}
```

**What happens**:
1. **One-time LLVM initialization**: Initializes native target (x86_64, NVPTX, etc.)
2. **Dialect translation registration**: Registers MLIR → LLVM IR translators
3. **Target machine creation**: Detects host CPU and creates TargetMachine
4. **Optimizer setup**: Creates optimization pipeline at specified level
5. **Calls C++ layer**: Invokes `ExecutionEngine::create()`

**GDB Breakpoints**:
```gdb
break mlirExecutionEngineCreate
break mlir::registerBuiltinDialectTranslation
break mlir::ExecutionEngine::create
```

---

### Frame 4: MLIR C++ ExecutionEngine Creation

**File**: [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:232-404](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L232-L404)

```cpp
Expected<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::create(Operation *m, const ExecutionEngineOptions &options,
                        std::unique_ptr<llvm::TargetMachine> tm) {
  auto engine = std::make_unique<ExecutionEngine>(
      options.enableObjectDump, options.enableGDBNotificationListener,
      options.enablePerfNotificationListener);

  // Remember all entry-points if object dumping is enabled
  if (options.enableObjectDump) {
    for (auto funcOp : m->getRegion(0).getOps<LLVM::LLVMFuncOp>()) {
      StringRef funcName = funcOp.getSymName();
      engine->functionNames.push_back(funcName.str());
    }
  }

  // ============================================
  // CRITICAL STEP 1: MLIR → LLVM IR Translation
  // ============================================
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  auto llvmModule = options.llvmModuleBuilder
                        ? options.llvmModuleBuilder(m, *ctx)
                        : translateModuleToLLVMIR(m, *ctx);
  if (!llvmModule)
    return makeStringError("could not convert to LLVM IR");

  // If no valid TargetMachine was passed, create a default TM
  if (!tm) {
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError)
      return tmBuilderOrError.takeError();

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError)
      return tmOrError.takeError();
    tm = std::move(tmOrError.get());
  }

  // ============================================
  // CRITICAL STEP 2: Set Target Triple & Data Layout
  // ============================================
  setupTargetTripleAndDataLayout(llvmModule.get(), tm.get());

  // ============================================
  // CRITICAL STEP 3: Pack Function Arguments
  // Creates _mlir_ciface_* wrapper functions
  // ============================================
  packFunctionArguments(llvmModule.get());

  auto dataLayout = llvmModule->getDataLayout();

  // Use absolute library path so that gdb can find the symbol table
  SmallVector<SmallString<256>, 4> sharedLibPaths;
  transform(
      options.sharedLibPaths, std::back_inserter(sharedLibPaths),
      [](StringRef libPath) {
        SmallString<256> absPath(libPath.begin(), libPath.end());
        cantFail(llvm::errorCodeToError(llvm::sys::fs::make_absolute(absPath)));
        return absPath;
      });

  // Process shared libraries for init/destroy callbacks
  llvm::StringMap<void *> exportSymbols;
  SmallVector<LibraryDestroyFn> destroyFns;
  SmallVector<StringRef> jitDyLibPaths;

  for (auto &libPath : sharedLibPaths) {
    auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(
        libPath.str().str().c_str());
    void *initSym = lib.getAddressOfSymbol(kLibraryInitFnName);
    void *destroySim = lib.getAddressOfSymbol(kLibraryDestroyFnName);

    if (!initSym || !destroySim) {
      jitDyLibPaths.push_back(libPath);
      continue;
    }

    auto initFn = reinterpret_cast<LibraryInitFn>(initSym);
    initFn(exportSymbols);

    auto destroyFn = reinterpret_cast<LibraryDestroyFn>(destroySim);
    destroyFns.push_back(destroyFn);
  }
  engine->destroyFns = std::move(destroyFns);

  // ============================================
  // CRITICAL STEP 4: Create Object Linking Layer
  // Sets up symbol resolution to current process and dynamic libraries
  // ============================================
  auto objectLinkingLayerCreator = [&](ExecutionSession &session) {
    auto objectLayer = std::make_unique<RTDyldObjectLinkingLayer>(
        session, [sectionMemoryMapper =
                      options.sectionMemoryMapper](const MemoryBuffer &) {
          return std::make_unique<SectionMemoryManager>(sectionMemoryMapper);
        });

    // Register JIT event listeners if enabled
    if (engine->gdbListener)
      objectLayer->registerJITEventListener(*engine->gdbListener);
    if (engine->perfListener)
      objectLayer->registerJITEventListener(*engine->perfListener);

    // COFF format handling for Windows
    const llvm::Triple &targetTriple = llvmModule->getTargetTriple();
    if (targetTriple.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    // Resolve symbols from shared libraries
    for (auto &libPath : jitDyLibPaths) {
      auto mb = llvm::MemoryBuffer::getFile(libPath);
      if (!mb) {
        errs() << "Failed to create MemoryBuffer for: " << libPath
               << "\nError: " << mb.getError().message() << "\n";
        continue;
      }
      auto &jd = session.createBareJITDylib(std::string(libPath));
      auto loaded = DynamicLibrarySearchGenerator::Load(
          libPath.str().c_str(), dataLayout.getGlobalPrefix());
      if (!loaded) {
        errs() << "Could not load " << libPath << ":\n  " << loaded.takeError()
               << "\n";
        continue;
      }
      jd.addGenerator(std::move(*loaded));
      cantFail(objectLayer->add(jd, std::move(mb.get())));
    }

    return objectLayer;
  };

  // ============================================
  // CRITICAL STEP 5: Create Compilation Function
  // This sets up lazy JIT compilation
  // ============================================
  auto compileFunctionCreator = [&](JITTargetMachineBuilder jtmb)
      -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
    if (options.jitCodeGenOptLevel)
      jtmb.setCodeGenOptLevel(*options.jitCodeGenOptLevel);
    return std::make_unique<TMOwningSimpleCompiler>(std::move(tm),
                                                    engine->cache.get());
  };

  // ============================================
  // CRITICAL STEP 6: Create LLJIT Instance
  // This is the LLVM ORC JIT engine
  // ============================================
  auto jit =
      cantFail(llvm::orc::LLJITBuilder()
                   .setCompileFunctionCreator(compileFunctionCreator)
                   .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                   .setDataLayout(dataLayout)
                   .create());

  // ============================================
  // CRITICAL STEP 7: Add LLVM IR Module to JIT
  // Module is NOT compiled yet - compilation is lazy!
  // ============================================
  ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
  if (options.transformer)
    cantFail(tsm.withModuleDo(
        [&](llvm::Module &module) { return options.transformer(&module); }));
  cantFail(jit->addIRModule(std::move(tsm)));
  engine->jit = std::move(jit);

  // ============================================
  // CRITICAL STEP 8: Register Current Process Symbols
  // Allows JIT'd code to call functions in the host process
  // ============================================
  llvm::orc::JITDylib &mainJD = engine->jit->getMainJITDylib();
  mainJD.addGenerator(
      cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix())));

  // Build a runtime symbol map from the exported symbols and register them
  auto runtimeSymbolMap = [&](llvm::orc::MangleAndInterner interner) {
    auto symbolMap = llvm::orc::SymbolMap();
    for (auto &exportSymbol : exportSymbols)
      symbolMap[interner(exportSymbol.getKey())] = {
          llvm::orc::ExecutorAddr::fromPtr(exportSymbol.getValue()),
          llvm::JITSymbolFlags::Exported};
    return symbolMap;
  };
  engine->registerSymbols(runtimeSymbolMap);

  return std::move(engine);
}
```

**What happens**:

1. **MLIR → LLVM IR Translation** (Line 248-252):
   - Calls `translateModuleToLLVMIR(m, *ctx)`
   - Converts MLIR dialects (gpu, nvgpu, nvvm, llvm) to LLVM IR
   - For GPU: Converts `gpu.module` with kernels into embedded CUBIN blobs
   - Creates `@llvm.global_ctors` array with `@kernel_load()` functions

2. **Set Target Triple & Data Layout** (Line 272):
   - Sets module target triple (e.g., x86_64-unknown-linux-gnu)
   - Sets data layout for pointer sizes, alignment, etc.

3. **Pack Function Arguments** (Line 273):
   - Creates `_mlir_ciface_*` wrapper functions
   - Transforms `func(a, b, c)` → `_mlir_ciface_func(void** args)`
   - Allows uniform invocation interface from Python

4. **Create Object Linking Layer** (Line 316-359):
   - Sets up RTDyldObjectLinkingLayer for loading JIT'd code
   - Registers GDB/perf listeners for debugging
   - Links shared libraries (e.g., libmlir_cuda_runtime.so)

5. **Create Compilation Function** (Line 363-369):
   - Sets up `TMOwningSimpleCompiler` with object cache
   - **Compilation is LAZY**: Not executed until first function lookup!

6. **Create LLJIT** (Line 372-377):
   - Creates LLVM ORC LLJIT instance
   - Sets up layered compilation: IR → Object → Linking

7. **Add IR Module to JIT** (Line 380-385):
   - Wraps LLVM module in ThreadSafeModule
   - Applies transformer (optimization passes)
   - **NO COMPILATION YET** - just stores the module

8. **Register Process Symbols** (Line 388-402):
   - Makes host process symbols available to JIT'd code
   - Required for calling `mgpuModuleLoadJIT()`, etc.

**GDB Breakpoints**:
```gdb
break mlir::ExecutionEngine::create
break mlir::translateModuleToLLVMIR
break mlir::ExecutionEngine::setupTargetTripleAndDataLayout
break mlir::ExecutionEngine::packFunctionArguments
break llvm::orc::LLJITBuilder::create
```

---

### Frame 5: MLIR → LLVM IR Translation (GPU-Specific)

**File**: [mlir/lib/Target/LLVMIR/ModuleTranslation.cpp](mlir/lib/Target/LLVMIR/ModuleTranslation.cpp)

**What happens** (conceptual, not showing full code due to complexity):

1. **Module Translation**:
   - Walks MLIR operations
   - Converts each operation to equivalent LLVM IR

2. **GPU Module Handling**:
   - Finds `gpu.module` operations containing GPU kernels
   - Serializes each GPU kernel to PTX/CUBIN binary
   - Embeds binary as global constant in LLVM IR:
     ```llvm
     @kernel_binary = internal constant [8192 x i8] c"\7FELF..."
     ```

3. **Global Constructor Creation**:
   - Creates `@kernel_load()` function:
     ```llvm
     define internal void @kernel_load() {
     entry:
       %0 = call ptr @mgpuModuleLoadJIT(ptr @kernel_binary, i32 2)
       store ptr %0, ptr @kernel_module, align 8
       ret void
     }
     ```
   - Adds to `@llvm.global_ctors` array:
     ```llvm
     @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [
       { i32 65535, ptr @kernel_load, ptr null }
     ]
     ```

4. **Kernel Launch Code**:
   - Creates host-side code that:
     - Loads module handle from global variable
     - Gets kernel function via `mgpuModuleGetFunction()`
     - Launches kernel via `mgpuLaunchKernel()`

**GDB Breakpoints**:
```gdb
break mlir::ModuleTranslation::convertOperation
break mlir::gpu::SerializeToCubinPass::runOnOperation
break mlir::translateModuleToLLVMIR
```

---

### Frame 6: Function Argument Packing

**File**: [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:144-201](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L144-L201)

```cpp
static void packFunctionArguments(Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  DenseSet<llvm::Function *> interfaceFunctions;

  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration()) {
      continue;
    }
    if (interfaceFunctions.count(&func)) {
      continue;
    }

    // Given a function `foo(<...>)`, define the interface function
    // `_mlir_ciface_foo(i8**)`.
    auto *newType =
        llvm::FunctionType::get(builder.getVoidTy(), builder.getPtrTy(),
                                /*isVarArg=*/false);
    auto newName = makePackedFunctionName(func.getName());  // "_mlir_" + name
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc = cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    // Extract the arguments from the type-erased argument list and cast them
    auto *bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    SmallVector<llvm::Value *, 8> args;
    args.reserve(llvm::size(func.args()));

    // Unpack each argument: args[i] = *(argList[i])
    for (auto [index, arg] : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), APInt(64, index));
      llvm::Value *argPtrPtr =
          builder.CreateGEP(builder.getPtrTy(), argList, argIndex);
      llvm::Value *argPtr = builder.CreateLoad(builder.getPtrTy(), argPtrPtr);
      llvm::Type *argTy = arg.getType();
      llvm::Value *load = builder.CreateLoad(argTy, argPtr);
      args.push_back(load);
    }

    // Call the implementation function with the extracted arguments
    llvm::Value *result = builder.CreateCall(&func, args);

    // If function returns a value, store it in args[N]
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), APInt(64, llvm::size(func.args())));
      llvm::Value *retPtrPtr =
          builder.CreateGEP(builder.getPtrTy(), argList, retIndex);
      llvm::Value *retPtr = builder.CreateLoad(builder.getPtrTy(), retPtrPtr);
      builder.CreateStore(result, retPtr);
    }

    // The interface function returns void
    builder.CreateRetVoid();
  }
}
```

**What happens**:
- For each function `foo(int a, float b)`:
  - Creates wrapper `_mlir_ciface_foo(void** args)`
  - Wrapper unpacks arguments: `a = *(int*)args[0], b = *(float*)args[1]`
  - Calls original `foo(a, b)`
  - Packs return value: `*(return_type*)args[2] = result`

**Why needed**:
- Python ctypes can only pass `void**` arrays
- Allows uniform invocation interface for any function signature

**GDB Breakpoint**:
```gdb
break mlir::ExecutionEngine::packFunctionArguments
```

---

### Summary of Compilation Phase

**When does actual native code compilation happen?**

**NOT during `ExecutionEngine` construction!**

Compilation is **LAZY** and happens on first function lookup:

1. `engine.invoke("main", args)` is called
2. Python wrapper calls `engine.lookup("main")`
3. Lookup calls `ExecutionEngine::lookupPacked("main")`
4. LookupPacked searches for `_mlir_ciface_main` symbol
5. **Symbol not found** → Triggers JIT compilation:
   - LLVM IR → SelectionDAG
   - SelectionDAG → MachineIR
   - MachineIR → Assembly
   - Assembly → Object code
   - Object code → Loaded into memory
6. Symbol address returned
7. Function can now be invoked

**Artifacts Created**:
- **In-memory**: Native x86_64 machine code
- **If object dump enabled**: `.o` file with compiled code
- **NOT created yet**: GPU binaries (loaded during `initialize()`)

**Symbols Exported**:
- `_mlir_ciface_<function_name>` - Packed wrapper function
- `<function_name>` - Original function (may not be exported)
- Global variables like `@kernel_module` (initially NULL)

---

## Part 2: engine.initialize() - Running Global Constructors

### Python Entry Point

**File**: [mlir/test/Examples/NVGPU/exec_engine.py:74](mlir/test/Examples/NVGPU/exec_engine.py#L74)

```python
engine.initialize()
```

**GDB Breakpoint**: N/A (Python level)

---

### Frame 1: Python ExecutionEngine Method

**File**: [mlir/python/mlir/execution_engine.py:14](mlir/python/mlir/execution_engine.py#L14)

This is just a wrapper class that inherits from `_mlirExecutionEngine.ExecutionEngine`. The `initialize()` method is defined in the C++ extension.

**GDB Breakpoint**: N/A (no Python code)

---

### Frame 2: Nanobind C++ Binding

**File**: [mlir/lib/Bindings/Python/ExecutionEngineModule.cpp:128-137](mlir/lib/Bindings/Python/ExecutionEngineModule.cpp#L128-L137)

```cpp
.def(
    "initialize",
    [](PyExecutionEngine &executionEngine) {
        mlirExecutionEngineInitialize(executionEngine.get());
    },
    "Initialize the ExecutionEngine. Global constructors specified by "
    "`llvm.mlir.global_ctors` will be run. One common scenario is that "
    "kernel binary compiled from `gpu.module` gets loaded during "
    "initialization. Make sure all symbols are resolvable before "
    "initialization by calling `register_runtime` or including "
    "shared libraries.")
```

**What happens**:
- Unwraps `PyExecutionEngine` to get `MlirExecutionEngine` handle
- Calls C API function `mlirExecutionEngineInitialize()`

**GDB Breakpoint**:
```gdb
break 'mlir::python::(anonymous namespace)::$_2::operator()'
# Or use line number
break mlir/lib/Bindings/Python/ExecutionEngineModule.cpp:130
```

---

### Frame 3: MLIR C API Layer

**File**: [mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp:71-73](mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp#L71-L73)

```cpp
extern "C" void mlirExecutionEngineInitialize(MlirExecutionEngine jit) {
  unwrap(jit)->initialize();
}
```

**What happens**:
- Unwraps opaque C pointer to C++ `ExecutionEngine*`
- Calls C++ `initialize()` method

**GDB Breakpoint**:
```gdb
break mlirExecutionEngineInitialize
```

---

### Frame 4: MLIR C++ ExecutionEngine::initialize()

**File**: [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:449-457](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L449-L457)

```cpp
void ExecutionEngine::initialize() {
  if (isInitialized)
    return;

  // TODO: Allow JIT initialize for AArch64. Currently there's a bug causing a
  // crash for AArch64 see related issue #71963.
  if (!jit->getTargetTriple().isAArch64())
    cantFail(jit->initialize(jit->getMainJITDylib()));

  isInitialized = true;
}
```

**What happens**:
- Checks if already initialized (idempotent)
- Calls LLVM ORC `LLJIT::initialize()` with the main JITDylib
- Sets `isInitialized` flag

**GDB Breakpoint**:
```gdb
break mlir::ExecutionEngine::initialize
```

---

### Frame 5: LLVM ORC LLJIT::initialize()

**File**: [llvm/include/llvm/ExecutionEngine/Orc/LLJIT.h:197-204](llvm/include/llvm/ExecutionEngine/Orc/LLJIT.h#L197-L204)

```cpp
/// Run the initializers for the given JITDylib.
Error initialize(JITDylib &JD) {
  DEBUG_WITH_TYPE("orc", {
    dbgs() << "LLJIT running initializers for JITDylib \"" << JD.getName()
           << "\"\n";
  });
  assert(PS && "PlatformSupport must be set to run initializers.");
  return PS->initialize(JD);
}
```

**What happens**:
- Delegates to `PlatformSupport::initialize()`
- PlatformSupport is typically `GenericLLVMIRPlatformSupport`

**GDB Breakpoint**:
```gdb
break llvm::orc::LLJIT::initialize
```

---

### Frame 6: GenericLLVMIRPlatformSupport::initialize()

**File**: [llvm/lib/ExecutionEngine/Orc/LLJIT.cpp:230-248](llvm/lib/ExecutionEngine/Orc/LLJIT.cpp#L230-L248)

```cpp
Error initialize(JITDylib &JD) override {
  LLVM_DEBUG({
    dbgs() << "GenericLLVMIRPlatformSupport getting initializers to run\n";
  });

  // Get list of initializer function addresses
  if (auto Initializers = getInitializers(JD)) {
    LLVM_DEBUG(
        { dbgs() << "GenericLLVMIRPlatformSupport running initializers\n"; });

    // Execute each initializer function
    for (auto InitFnAddr : *Initializers) {
      LLVM_DEBUG({
        dbgs() << "  Running init " << formatv("{0:x16}", InitFnAddr)
               << "...\n";
      });

      // Cast address to function pointer and call it
      auto *InitFn = InitFnAddr.toPtr<void (*)()>();
      InitFn();  // ← THIS IS WHERE @kernel_load() RUNS!
    }
  } else
    return Initializers.takeError();
  return Error::success();
}
```

**What happens**:
1. **Calls `getInitializers(JD)`** to find all init functions
2. **Looks up symbols** in `@llvm.global_ctors` array
3. **Gets function addresses** for each initializer
4. **Calls each function pointer** (e.g., `@kernel_load()`)

**This is where GPU binaries get loaded!**

**GDB Breakpoint**:
```gdb
break 'llvm::orc::(anonymous namespace)::GenericLLVMIRPlatformSupport::initialize'
break llvm/lib/ExecutionEngine/Orc/LLJIT.cpp:237
break llvm/lib/ExecutionEngine/Orc/LLJIT.cpp:242
```

---

### Frame 7: getInitializers() - Finding Constructor Functions

**File**: [llvm/lib/ExecutionEngine/Orc/LLJIT.cpp:283-333](llvm/lib/ExecutionEngine/Orc/LLJIT.cpp#L283-L333)

```cpp
Expected<std::vector<ExecutorAddr>> getInitializers(JITDylib &JD) {
  // Issue lookups to materialize any init symbols
  if (auto Err = issueInitLookups(JD))
    return std::move(Err);

  DenseMap<JITDylib *, SymbolLookupSet> LookupSymbols;
  std::vector<JITDylibSP> DFSLinkOrder;

  // Get list of JITDylibs in dependency order
  if (auto Err = getExecutionSession().runSessionLocked([&]() -> Error {
        if (auto DFSLinkOrderOrErr = JD.getDFSLinkOrder())
          DFSLinkOrder = std::move(*DFSLinkOrderOrErr);
        else
          return DFSLinkOrderOrErr.takeError();

        // Collect all init functions from each JITDylib
        for (auto &NextJD : DFSLinkOrder) {
          auto IFItr = InitFunctions.find(NextJD.get());
          if (IFItr != InitFunctions.end()) {
            LookupSymbols[NextJD.get()] = std::move(IFItr->second);
            InitFunctions.erase(IFItr);
          }
        }
        return Error::success();
      }))
    return std::move(Err);

  LLVM_DEBUG({
    dbgs() << "Running order for " << JD.getName() << ":\n";
    for (auto &NextJD : DFSLinkOrder)
      dbgs() << "  " << NextJD->getName() << "\n";
  });

  std::vector<ExecutorAddr> Initializers;

  // Look up addresses of all init functions
  for (auto *NextJD : DFSLinkOrder) {
    auto LMItr = LookupSymbols.find(NextJD);
    if (LMItr == LookupSymbols.end())
      continue;

    // Perform symbol lookup to get function addresses
    auto InitSymbols = J.getExecutionSession().lookup(
        makeJITDylibSearchOrder(NextJD,
                                JITDylibLookupFlags::MatchExportedSymbolsOnly),
        std::move(LMItr->second));
    if (!InitSymbols)
      return InitSymbols.takeError();

    // Add addresses to list in order
    for (auto &KV : *InitSymbols)
      Initializers.push_back(KV.second.getAddress());
  }

  return Initializers;
}
```

**What happens**:
1. **Scans `InitFunctions` map** for symbols matching `__orc_init_func.*` or from `@llvm.global_ctors`
2. **Looks up symbol addresses** via `ExecutionSession::lookup()`
   - **This triggers JIT compilation** if not already compiled
   - Native code for `@kernel_load()` is generated
3. **Returns vector of function pointers** to execute

**GDB Breakpoint**:
```gdb
break 'llvm::orc::(anonymous namespace)::GenericLLVMIRPlatformSupport::getInitializers'
break llvm::orc::ExecutionSession::lookup
```

---

### Frame 8: Native Code - @kernel_load() Execution

**Generated LLVM IR**: [mlir/test/Examples/NVGPU/translated_module.ll:86-89](mlir/test/Examples/NVGPU/translated_module.ll#L86-L89)

```llvm
define internal void @gemm_128_128_64_kernel_load() section ".text.startup" {
entry:
  ; Call mgpuModuleLoadJIT with kernel binary and opt level
  %0 = call ptr @mgpuModuleLoadJIT(ptr @gemm_128_128_64_kernel_binary, i32 2)

  ; Store the loaded module handle in global variable
  store ptr %0, ptr @gemm_128_128_64_kernel_module, align 8
  ret void
}
```

**What this function does** (native x86_64 code):
1. **Calls `mgpuModuleLoadJIT()`** with pointer to embedded CUBIN binary
2. **Receives CUmodule handle** back
3. **Stores handle** in global variable `@gemm_128_128_64_kernel_module`

**GDB Breakpoint**:
```gdb
# Can't set by name (generated symbol), use address after lookup
# Or break on the called function:
break mgpuModuleLoadJIT
```

---

### Frame 9: mgpuModuleLoadJIT() - GPU Binary Loading

**File**: [mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp:127-135](mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp#L127-L135)

```cpp
extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule
mgpuModuleLoadJIT(void *data, int optLevel) {
  ScopedContext scopedContext;  // ← Initializes CUDA here!

  CUjit_option jitOptions[] = {CU_JIT_OPTIMIZATION_LEVEL};
  void *jitOptionsValues[] = {reinterpret_cast<void *>(optLevel)};

  CUmodule module = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleLoadDataEx(&module, data, 1, jitOptions,
                                          jitOptionsValues));
  return module;
}
```

**What happens**:
1. **Creates `ScopedContext`** - This is CRITICAL!
2. **Calls `cuModuleLoadDataEx()`** to load CUBIN into GPU memory
3. **Returns CUmodule handle**

**GDB Breakpoint**:
```gdb
break mgpuModuleLoadJIT
break cuModuleLoadDataEx
```

---

### Frame 10: ScopedContext Constructor - CUDA Initialization

**File**: [mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp:83-101](mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp#L83-L101)

```cpp
class ScopedContext {
public:
  ScopedContext() {
    // Static reference to CUDA primary context for device
    static CUcontext context = [] {
      // ============================================
      // THIS IS WHERE CUDA GETS INITIALIZED!
      // ============================================
      CUDA_REPORT_IF_ERROR(cuInit(/*flags=*/0));

      CUcontext ctx;
      CUDA_REPORT_IF_ERROR(
          cuDevicePrimaryCtxRetain(&ctx, getDefaultCuDevice()));
      return ctx;
    }();

    CUDA_REPORT_IF_ERROR(cuCtxPushCurrent(context));
  }

  ~ScopedContext() {
    CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr));
  }
};
```

**What happens**:
1. **First call only** (static lambda):
   - **Calls `cuInit(0)`** - Initializes CUDA driver!
   - **Gets device 0** via `getDefaultCuDevice()`
   - **Retains primary context** via `cuDevicePrimaryCtxRetain()`
   - **Stores context** in static variable
2. **Every call**:
   - **Pushes context** onto thread's context stack via `cuCtxPushCurrent()`
3. **On destruction**:
   - **Pops context** via `cuCtxPopCurrent()`

**This is THE critical function that initializes CUDA!**

**GDB Breakpoints**:
```gdb
break 'ScopedContext::ScopedContext'
break cuInit
break cuDevicePrimaryCtxRetain
break cuCtxPushCurrent
```

---

### Frame 11: cuInit() - CUDA Driver Initialization

**CUDA Driver API** (proprietary, no source available)

**What happens** (conceptual):
1. **Loads CUDA driver** (e.g., /usr/lib/x86_64-linux-gnu/libcuda.so)
2. **Initializes driver state**
3. **Enumerates GPUs** on the system
4. **Sets up device handles**
5. **Prepares for context creation**

**GDB Breakpoint**:
```gdb
break cuInit
```

**To see CUDA calls**, set environment variable:
```bash
export MLIR_CUDA_DEBUG=1
```

---

### Summary of initialize() Flow

```
Python: engine.initialize()
  ↓
Nanobind: [](PyExecutionEngine &ee) { mlirExecutionEngineInitialize(ee.get()); }
  ↓
C API: mlirExecutionEngineInitialize(jit) { unwrap(jit)->initialize(); }
  ↓
C++: ExecutionEngine::initialize() { jit->initialize(getMainJITDylib()); }
  ↓
LLVM: LLJIT::initialize(JD) { PS->initialize(JD); }
  ↓
LLVM: GenericLLVMIRPlatformSupport::initialize(JD)
  ↓ getInitializers(JD)
  ↓ ExecutionSession::lookup(..., InitFunctions)
  ↓ [JIT COMPILATION OF @kernel_load HAPPENS HERE]
  ↓
LLVM: for (auto InitFn : Initializers) { InitFn(); }  // Call function pointers
  ↓
Native: @kernel_load() { mgpuModuleLoadJIT(...); }
  ↓
CUDA Wrapper: mgpuModuleLoadJIT() { ScopedContext ctx; cuModuleLoadDataEx(); }
  ↓
CUDA Wrapper: ScopedContext::ScopedContext() { cuInit(0); cuDevicePrimaryCtxRetain(); }
  ↓
CUDA Driver: cuInit(0)  // ← CUDA is now initialized!
  ↓
CUDA Driver: cuDevicePrimaryCtxRetain()  // ← Context created for GPU 0
  ↓
CUDA Driver: cuCtxPushCurrent()  // ← Context activated
  ↓
CUDA Driver: cuModuleLoadDataEx()  // ← CUBIN binary loaded into GPU memory
  ↓
Native: store CUmodule handle in @kernel_module  // ← Global variable now has valid handle
  ↓
Return to Python
```

**Result**:
- CUDA driver is initialized
- GPU context is created and active
- GPU kernel binaries are loaded into GPU memory
- Module handles are stored in global variables
- Ready to launch kernels!

---

## Part 3: engine.invoke() - Executing Compiled Function

### Python Entry Point

**File**: [mlir/test/Examples/NVGPU/exec_engine.py:78](mlir/test/Examples/NVGPU/exec_engine.py#L78)

```python
engine.invoke(function_name, *newArgs)
```

**GDB Breakpoint**: N/A (Python level)

---

### Frame 1: Python ExecutionEngine.invoke()

**File**: [mlir/python/mlir/execution_engine.py:26-35](mlir/python/mlir/execution_engine.py#L26-L35)

```python
def invoke(self, name, *ctypes_args):
    """Invoke a function with the list of ctypes arguments.
    All arguments must be pointers.
    Raise a RuntimeError if the function isn't found.
    """
    # Look up the _mlir_ciface_* wrapper function
    func = self.lookup(name)

    # Pack arguments into void** array
    packed_args = (ctypes.c_void_p * len(ctypes_args))()
    for argNum in range(len(ctypes_args)):
        packed_args[argNum] = ctypes.cast(ctypes_args[argNum], ctypes.c_void_p)

    # Call the function with packed arguments
    func(packed_args)
```

**What happens**:
1. **Calls `self.lookup(name)`** to get ctype callable
2. **Packs arguments** into `void**` array
3. **Calls function** via ctypes

**GDB Breakpoint**: N/A (Python level, but can trace ctypes call)

---

### Frame 2: Python ExecutionEngine.lookup()

**File**: [mlir/python/mlir/execution_engine.py:15-24](mlir/python/mlir/execution_engine.py#L15-L24)

```python
def lookup(self, name):
    """Lookup a function emitted with the `llvm.emit_c_interface`
    attribute and returns a ctype callable.
    Raise a RuntimeError if the function isn't found.
    """
    # Call the C++ extension's raw_lookup method
    func = self.raw_lookup("_mlir_ciface_" + name)
    if not func:
        raise RuntimeError("Unknown function " + name)

    # Create ctypes function prototype: void(void*)
    prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    return prototype(func)
```

**What happens**:
1. **Prepends `_mlir_ciface_`** to function name
2. **Calls C++ `raw_lookup()`** to get function pointer (as integer)
3. **Wraps in ctypes.CFUNCTYPE** to make it callable from Python
4. **Returns callable**

**GDB Breakpoint**: N/A (Python level)

---

### Frame 3: Nanobind C++ raw_lookup()

**File**: [mlir/lib/Bindings/Python/ExecutionEngineModule.cpp:103-112](mlir/lib/Bindings/Python/ExecutionEngineModule.cpp#L103-L112)

```cpp
.def(
    "raw_lookup",
    [](PyExecutionEngine &executionEngine, const std::string &func) {
        auto *res = mlirExecutionEngineLookupPacked(
            executionEngine.get(),
            mlirStringRefCreate(func.c_str(), func.size()));
        return reinterpret_cast<uintptr_t>(res);
    },
    nb::arg("func_name"),
    "Lookup function `func` in the ExecutionEngine.")
```

**What happens**:
- Calls C API `mlirExecutionEngineLookupPacked()`
- Returns function pointer as `uintptr_t` (Python integer)

**GDB Breakpoint**:
```gdb
break 'mlir::python::(anonymous namespace)::$_1::operator()'
break mlir/lib/Bindings/Python/ExecutionEngineModule.cpp:106
```

---

### Frame 4: MLIR C API mlirExecutionEngineLookupPacked()

**File**: [mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp:90-97](mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp#L90-L97)

```cpp
extern "C" void *mlirExecutionEngineLookupPacked(MlirExecutionEngine jit,
                                                 MlirStringRef name) {
  auto optionalFPtr =
      llvm::expectedToOptional(unwrap(jit)->lookupPacked(unwrap(name)));
  if (!optionalFPtr)
    return nullptr;
  return reinterpret_cast<void *>(*optionalFPtr);
}
```

**What happens**:
- Unwraps C types to C++ types
- Calls `ExecutionEngine::lookupPacked()`
- Returns function pointer or nullptr

**GDB Breakpoint**:
```gdb
break mlirExecutionEngineLookupPacked
```

---

### Frame 5: MLIR C++ ExecutionEngine::lookupPacked()

**File**: [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:406-412](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L406-L412)

```cpp
Expected<void (*)(void **)>
ExecutionEngine::lookupPacked(StringRef name) const {
  // Call generic lookup with packed name
  auto result = lookup(makePackedFunctionName(name));
  if (!result)
    return result.takeError();

  // Cast to correct function pointer type: void(*)(void**)
  return reinterpret_cast<void (*)(void **)>(result.get());
}
```

**What happens**:
- Calls `makePackedFunctionName(name)` → `"_mlir_" + name`
- Calls `lookup("_mlir_<name>")`
- Casts result to `void(*)(void**)` function pointer
- Returns function pointer

**GDB Breakpoint**:
```gdb
break mlir::ExecutionEngine::lookupPacked
```

---

### Frame 6: MLIR C++ ExecutionEngine::lookup()

**File**: [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:414-434](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L414-L434)

```cpp
Expected<void *> ExecutionEngine::lookup(StringRef name) const {
  // Look up symbol in JIT
  auto expectedSymbol = jit->lookup(name);

  // JIT lookup may return an Error referring to strings stored internally by
  // the JIT. If the Error outlives the ExecutionEngine, it would want have a
  // dangling reference, which is currently caught by an assertion inside JIT
  // thanks to hand-rolled reference counting. Rewrap the error message into a
  // string before returning.
  if (!expectedSymbol) {
    std::string errorMessage;
    llvm::raw_string_ostream os(errorMessage);
    llvm::handleAllErrors(expectedSymbol.takeError(),
                          [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
    return makeStringError(errorMessage);
  }

  if (void *fptr = expectedSymbol->toPtr<void *>())
    return fptr;
  return makeStringError("looked up function is null");
}
```

**What happens**:
- Calls `jit->lookup(name)` (LLVM ORC JIT lookup)
- **This triggers JIT compilation if not already compiled**
- Returns function pointer address
- Error handling for missing symbols

**GDB Breakpoint**:
```gdb
break mlir::ExecutionEngine::lookup
```

---

### Frame 7: LLVM ORC LLJIT::lookup()

**File**: [llvm/lib/ExecutionEngine/Orc/LLJIT.cpp](llvm/lib/ExecutionEngine/Orc/LLJIT.cpp) (complex, conceptual)

```cpp
Expected<ExecutorAddr> LLJIT::lookup(StringRef UnmangledName) {
  return ES->lookup(makeJITDylibSearchOrder(&getMainJITDylib()),
                    ES->intern(mangle(UnmangledName)));
}
```

**What happens**:
1. **Mangles symbol name** (e.g., `_mlir_main` stays `_mlir_main` on Linux)
2. **Searches JITDylibs** in order (main, process symbols, shared libs)
3. **If symbol not materialized yet**:
   - **Triggers lazy compilation**:
     - LLVM IR → SelectionDAG
     - SelectionDAG → MachineIR
     - MachineIR → Assembly
     - Assembly → Object code
     - Object code → Memory
   - **Links object code** with RTDyldObjectLinkingLayer
   - **Resolves relocations**
   - **Returns symbol address**
4. **If symbol already materialized**:
   - Returns cached address

**This is where actual x86_64 machine code compilation happens!**

**GDB Breakpoints**:
```gdb
break llvm::orc::LLJIT::lookup
break llvm::orc::ExecutionSession::lookup
break llvm::orc::IRCompileLayer::emit
break llvm::orc::SimpleCompiler::operator()
```

---

### Frame 8: JIT Compilation (Conceptual)

**LLVM JIT Compilation Pipeline**:

1. **LLVM IR** (already in memory from `ExecutionEngine::create()`)
   ```llvm
   define void @_mlir_ciface_main(ptr %0) {
     ; Unpack arguments
     %2 = getelementptr ptr, ptr %0, i64 0
     %3 = load ptr, ptr %2
     %4 = load i32, ptr %3

     ; Call original function
     call void @main(i32 %4)
     ret void
   }
   ```

2. **SelectionDAG** (LLVM IR → DAG representation)
   - Instruction selection
   - Register allocation preparation

3. **MachineIR** (Target-specific instructions)
   - x86_64 instructions
   - Virtual registers

4. **Register Allocation**
   - Virtual registers → Physical registers (rax, rbx, etc.)

5. **Assembly Code**
   ```asm
   _mlir_ciface_main:
     push   rbp
     mov    rbp, rsp
     mov    rdi, QWORD PTR [rdi]
     mov    edi, DWORD PTR [rdi]
     call   main
     pop    rbp
     ret
   ```

6. **Object Code** (Machine code bytes)
   ```
   55 48 89 e5 48 8b 3f 8b 3f e8 xx xx xx xx 5d c3
   ```

7. **Loaded into Memory**
   - Allocated executable memory page
   - Code copied to memory
   - Relocations resolved

**GDB Breakpoints** (detailed compilation):
```gdb
break llvm::IRCompileLayer::emit
break llvm::IRCompiler::operator()
break llvm::SelectionDAGISel::runOnMachineFunction
break llvm::AsmPrinter::runOnMachineFunction
```

---

### Frame 9: Python ctypes Function Call

After `lookup()` returns, Python has a ctypes-wrapped function pointer:

```python
# func is a ctypes.CFUNCTYPE(None, ctypes.c_void_p) wrapping the pointer
func(packed_args)
```

**What happens**:
- ctypes marshals arguments
- Calls native function via FFI
- Control transfers to native x86_64 code

**GDB Breakpoint**:
```gdb
# Set breakpoint on the native function
break _mlir_ciface_main
# Or break on the original function
break main
```

---

### Frame 10: Native Code - _mlir_ciface_main()

**Generated from**: [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:144-201](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L144-L201) (packFunctionArguments)

**LLVM IR**:
```llvm
define void @_mlir_ciface_main(ptr %0) {
entry:
  ; Unpack argument 0: alpha
  %1 = getelementptr ptr, ptr %0, i64 0
  %2 = load ptr, ptr %1
  %3 = load i32, ptr %2

  ; Call the original main function
  call void @main(i32 %3)

  ret void
}
```

**x86_64 Assembly** (conceptual):
```asm
_mlir_ciface_main:
    push    rbp
    mov     rbp, rsp

    ; Load argument array pointer (rdi = args)
    ; Load first pointer: ptr* args[0]
    mov     rdi, QWORD PTR [rdi]

    ; Load value: int* args[0]
    mov     edi, DWORD PTR [rdi]

    ; Call original function with unpacked argument
    call    main

    pop     rbp
    ret
```

**What happens**:
1. **Receives `void** args`** in first argument register (rdi on x86_64)
2. **Unpacks arguments**:
   - `args[0]` is pointer to first argument
   - Dereference to get actual value
3. **Calls original `@main()`** with unpacked arguments
4. **Returns**

**GDB Breakpoint**:
```gdb
break _mlir_ciface_main
```

---

### Frame 11: Native Code - main() Function

**User's NVDSL Python code** compiled to native:

```python
@NVDSL.mlir_func
def main(alpha):
    @NVDSL.mlir_gpu_launch(grid=(1, 1, 1), block=(4, 1, 1))
    def kernel():
        i = gpu.thread_id(gpu.Dimension.x)
        alpha[i] = alpha[i] + 100
```

**LLVM IR** (simplified):
```llvm
define void @main(i32 %alpha) {
entry:
  ; Allocate memory for alpha array
  %0 = alloca ptr, align 8

  ; ... setup code ...

  ; Load GPU module handle (set by @kernel_load during initialize())
  %module = load ptr, ptr @kernel_module, align 8

  ; Get kernel function from module
  %kernel_func = call ptr @mgpuModuleGetFunction(ptr %module, ptr @kernel_name)

  ; Launch kernel
  call void @mgpuLaunchKernel(ptr %kernel_func, i64 1, i64 1, i64 1,
                              i64 4, i64 1, i64 1, i32 0, ptr null, ptr %args)

  ret void
}
```

**What happens**:
1. **Loads module handle** from `@kernel_module` global variable
   - **This was set by `@kernel_load()` during `initialize()`**
   - **If `initialize()` wasn't called, this is NULL!** → Error!
2. **Calls `mgpuModuleGetFunction()`** to get kernel function handle
3. **Calls `mgpuLaunchKernel()`** to launch GPU kernel
4. **Returns**

**GDB Breakpoint**:
```gdb
break main
```

---

### Frame 12: mgpuModuleGetFunction() - Get Kernel Function

**File**: [mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp:152-157](mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp#L152-L157)

```cpp
extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUfunction
mgpuModuleGetFunction(CUmodule module, const char *name) {
  CUfunction function = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleGetFunction(&function, module, name));
  return function;
}
```

**What happens**:
- Calls CUDA driver API `cuModuleGetFunction()`
- Gets function handle from loaded module
- **If module is NULL** → `CUDA_ERROR_NOT_INITIALIZED`

**GDB Breakpoint**:
```gdb
break mgpuModuleGetFunction
break cuModuleGetFunction
```

---

### Frame 13: mgpuLaunchKernel() - Launch GPU Kernel

**File**: [mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp:172-202](mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp#L172-L202)

```cpp
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                 intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                 intptr_t blockZ, int32_t smem, CUstream stream, void **params,
                 void **extra, size_t paramsCount) {

  if (getEnvMLIRCUDADebugFlag()) {
    fprintf(stderr,
            "%s:%d:%s(): Launching kernel, grid=%ld,%ld,%ld, "
            "threads: %ld, %ld, %ld, smem: %dkb\n",
            __FILE__, __LINE__, __func__, gridX, gridY, gridZ, blockX, blockY,
            blockZ, smem / 1000);
  }

  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                      blockY, blockZ, smem, stream, params,
                                      extra));
}
```

**What happens**:
- Prints debug message (if `MLIR_CUDA_DEBUG=1`)
- Calls CUDA driver API `cuLaunchKernel()`
- **Kernel executes on GPU**
- Returns (kernel launch is asynchronous by default)

**GDB Breakpoint**:
```gdb
break mgpuLaunchKernel
break cuLaunchKernel
```

---

### Frame 14: cuLaunchKernel() - CUDA Kernel Launch

**CUDA Driver API** (proprietary)

**What happens**:
1. **Validates parameters** (grid, block dimensions, memory)
2. **Allocates GPU resources** (registers, shared memory)
3. **Copies parameters** to GPU constant memory
4. **Enqueues kernel** on GPU stream
5. **Returns immediately** (async launch)

**Kernel executes on GPU**:
```cuda
// Conceptual CUDA C code (actual is PTX/SASS)
__global__ void kernel(int* alpha) {
    int i = threadIdx.x;
    alpha[i] = alpha[i] + 100;
}
```

**GDB Breakpoint**:
```gdb
break cuLaunchKernel
# Can't break inside GPU kernel with GDB (need cuda-gdb)
```

---

### Summary of invoke() Flow

```
Python: engine.invoke("main", alpha)
  ↓
Python: func = engine.lookup("main")  // Returns ctypes callable
  ↓
Python: func(packed_args)  // Call via ctypes
  ↓
Nanobind: raw_lookup() { mlirExecutionEngineLookupPacked(...); }
  ↓
C API: mlirExecutionEngineLookupPacked() { unwrap(jit)->lookupPacked(...); }
  ↓
C++: ExecutionEngine::lookupPacked() { lookup(makePackedFunctionName(name)); }
  ↓
C++: ExecutionEngine::lookup() { jit->lookup(name); }
  ↓
LLVM: LLJIT::lookup() { ES->lookup(...); }
  ↓
LLVM: ExecutionSession::lookup()
  ↓ [LAZY JIT COMPILATION HAPPENS HERE IF NOT YET COMPILED]
  ↓ LLVM IR → SelectionDAG → MachineIR → Assembly → Object Code → Memory
  ↓
LLVM: Returns function pointer address
  ↓
Python: Wraps pointer in ctypes.CFUNCTYPE
  ↓
Python: Calls function via ctypes FFI
  ↓
Native x86_64: _mlir_ciface_main(void** args)
  ↓ Unpack arguments from args array
  ↓
Native x86_64: main(int alpha)
  ↓ Load module handle from @kernel_module global (set by initialize())
  ↓
CUDA Wrapper: mgpuModuleGetFunction(module, "kernel")
  ↓
CUDA Driver: cuModuleGetFunction(&function, module, "kernel")
  ↓
CUDA Wrapper: mgpuLaunchKernel(function, grid, block, params)
  ↓
CUDA Driver: cuLaunchKernel(function, grid, block, smem, stream, params)
  ↓
GPU: Kernel executes in parallel across threads
  ↓ Each thread: alpha[threadIdx.x] += 100
  ↓
Return to Python
```

---

## Complete Breakpoint Reference

### GDB Commands

Start GDB with Python:
```bash
cd /home/jeromeku/llvm-project
export PYTHONPATH=./build/tools/mlir/python_packages/mlir_core
export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH
export MLIR_CUDA_DEBUG=1  # Enable CUDA debug output

gdb --args python mlir/test/Examples/NVGPU/exec_engine.py
```

### Breakpoints by Phase

#### Phase 1: ExecutionEngine Construction

```gdb
# Python → C API
break mlir/lib/Bindings/Python/ExecutionEngineModule.cpp:83
break mlirExecutionEngineCreate

# MLIR → LLVM IR Translation
break mlir::ExecutionEngine::create
break mlir::translateModuleToLLVMIR

# GPU-specific translation
break mlir::gpu::SerializeToCubinPass::runOnOperation

# Function packing
break mlir::ExecutionEngine::packFunctionArguments

# LLJIT creation
break llvm::orc::LLJITBuilder::create
```

#### Phase 2: engine.initialize()

```gdb
# Entry points
break mlir/lib/Bindings/Python/ExecutionEngineModule.cpp:130
break mlirExecutionEngineInitialize
break mlir::ExecutionEngine::initialize

# LLVM ORC initialization
break llvm::orc::LLJIT::initialize
break 'llvm::orc::(anonymous namespace)::GenericLLVMIRPlatformSupport::initialize'

# Constructor execution
break llvm/lib/ExecutionEngine/Orc/LLJIT.cpp:237  # Loop calling init functions
break llvm/lib/ExecutionEngine/Orc/LLJIT.cpp:242  # InitFn() call

# GPU binary loading
break mgpuModuleLoadJIT
break 'ScopedContext::ScopedContext'

# CUDA initialization
break cuInit
break cuDevicePrimaryCtxRetain
break cuCtxPushCurrent
break cuModuleLoadDataEx
```

#### Phase 3: engine.invoke()

```gdb
# Lookup phase
break mlir/lib/Bindings/Python/ExecutionEngineModule.cpp:106
break mlirExecutionEngineLookupPacked
break mlir::ExecutionEngine::lookupPacked
break mlir::ExecutionEngine::lookup

# JIT compilation (if not yet compiled)
break llvm::orc::LLJIT::lookup
break llvm::orc::ExecutionSession::lookup
break llvm::orc::IRCompileLayer::emit
break llvm::orc::SimpleCompiler::operator()

# Native execution
break _mlir_ciface_main  # Wrapper function
break main  # User's function

# GPU kernel launch
break mgpuModuleGetFunction
break cuModuleGetFunction
break mgpuLaunchKernel
break cuLaunchKernel
```

### Useful GDB Commands

```gdb
# Run program
run

# Continue after breakpoint
continue

# Step into function
step

# Step over function
next

# Print backtrace
bt

# Print variable
print variable_name

# Print expression
print/x expression  # Hex format

# Set conditional breakpoint
break file.cpp:123 if condition

# Enable LLVM debug output
set environment LLVM_DEBUG orc

# Enable CUDA debug output
set environment MLIR_CUDA_DEBUG 1
```

### LLDB Commands (Alternative)

```bash
lldb -- python mlir/test/Examples/NVGPU/exec_engine.py
```

```lldb
# Set breakpoint
breakpoint set --name mlir::ExecutionEngine::initialize
breakpoint set --file ExecutionEngine.cpp --line 449

# Run
run

# Continue
continue

# Backtrace
bt

# Print variable
print variable_name

# Set environment
settings set target.env-vars MLIR_CUDA_DEBUG=1
```

---

## Environment Variables for Debugging

### MLIR/LLVM

```bash
# Enable MLIR CUDA wrapper debug prints
export MLIR_CUDA_DEBUG=1

# Enable MLIR dump methods
export MLIR_ENABLE_DUMP=1

# Enable LLVM dump methods
export LLVM_ENABLE_DUMP=1

# Enable LLVM debug output for specific subsystems
export LLVM_DEBUG="orc,jit"

# Enable verbose LLVM output
export LLVM_VERBOSE=1
```

### CUDA

```bash
# Make kernel launches synchronous (easier debugging)
export CUDA_LAUNCH_BLOCKING=1

# Enable CUDA API tracing (requires CUDA toolkit)
export CUDA_PROFILE=1

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Enable CUDA error checking
export CUDA_DEVICE_DEBUG=1
```

### GDB/Debugging

```bash
# Disable address space randomization (consistent addresses)
setarch $(uname -m) -R gdb --args python script.py

# Or in GDB:
# (gdb) set disable-randomization on
```

---

## Source File Reference Guide

### Python Layer

| File | Purpose |
|------|---------|
| [mlir/python/mlir/execution_engine.py](mlir/python/mlir/execution_engine.py) | Python wrapper for ExecutionEngine |
| [mlir/test/Examples/NVGPU/tools/nvdsl.py](mlir/test/Examples/NVGPU/tools/nvdsl.py) | NVDSL decorator framework |
| [mlir/test/Examples/NVGPU/tools/nvgpucompiler.py](mlir/test/Examples/NVGPU/tools/nvgpucompiler.py) | NVGPU compilation wrapper |

### Python Bindings (C++)

| File | Purpose |
|------|---------|
| [mlir/lib/Bindings/Python/ExecutionEngineModule.cpp](mlir/lib/Bindings/Python/ExecutionEngineModule.cpp) | Nanobind C++ bindings for ExecutionEngine |

### MLIR C API

| File | Purpose |
|------|---------|
| [mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp](mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp) | C API layer for ExecutionEngine |
| [mlir-c/ExecutionEngine.h](mlir-c/ExecutionEngine.h) | C API header |

### MLIR C++ ExecutionEngine

| File | Purpose |
|------|---------|
| [mlir/lib/ExecutionEngine/ExecutionEngine.cpp](mlir/lib/ExecutionEngine/ExecutionEngine.cpp) | Core ExecutionEngine implementation |
| [mlir/include/mlir/ExecutionEngine/ExecutionEngine.h](mlir/include/mlir/ExecutionEngine/ExecutionEngine.h) | ExecutionEngine header |

### MLIR → LLVM IR Translation

| File | Purpose |
|------|---------|
| [mlir/lib/Target/LLVMIR/ModuleTranslation.cpp](mlir/lib/Target/LLVMIR/ModuleTranslation.cpp) | MLIR to LLVM IR translation |
| [mlir/lib/Target/LLVMIR/Dialect/GPU/GPUToLLVMIR.cpp](mlir/lib/Target/LLVMIR/Dialect/GPU/GPUToLLVMIR.cpp) | GPU dialect translation |

### CUDA Runtime Wrappers

| File | Purpose |
|------|---------|
| [mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp](mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp) | CUDA wrapper functions (mgpu*) |

### LLVM ORC JIT

| File | Purpose |
|------|---------|
| [llvm/lib/ExecutionEngine/Orc/LLJIT.cpp](llvm/lib/ExecutionEngine/Orc/LLJIT.cpp) | LLJIT implementation |
| [llvm/include/llvm/ExecutionEngine/Orc/LLJIT.h](llvm/include/llvm/ExecutionEngine/Orc/LLJIT.h) | LLJIT header |
| [llvm/lib/ExecutionEngine/Orc/ExecutionUtils.cpp](llvm/lib/ExecutionEngine/Orc/ExecutionUtils.cpp) | ORC execution utilities |

---

## Key Insights

### 1. Lazy Compilation

**ExecutionEngine construction does NOT compile to native code!**

- MLIR → LLVM IR happens during construction
- LLVM IR → Native code happens on **first lookup**
- Global constructors compile when `initialize()` is called

### 2. Critical Initialization Order

**MUST call in this order**:

```python
engine = ExecutionEngine(module, ...)  # MLIR → LLVM IR
engine.initialize()                     # Load GPU binaries, init CUDA
engine.invoke("func", args)             # Compile & execute
```

**Skipping `initialize()` causes**:
- Global constructors never run
- GPU binaries never loaded
- `@kernel_module` stays NULL
- `mgpuModuleGetFunction(NULL, ...)` fails
- `CUDA_ERROR_NOT_INITIALIZED`

### 3. Function Name Mangling

| Python Name | Lookup Name | Wrapper Name |
|-------------|-------------|--------------|
| `"main"` | `"_mlir_ciface_main"` | `"_mlir_main"` |

- Python `invoke("main")` looks up `"_mlir_ciface_main"`
- C interface wrappers have `_mlir_ciface_` prefix
- Internal packed wrappers have `_mlir_` prefix

### 4. GPU-Specific Flow

**GPU kernels are NOT JIT compiled at runtime!**

- GPU code compiled during MLIR → LLVM IR (AOT within JIT)
- Kernels serialized to PTX/CUBIN during translation
- Binaries embedded in LLVM IR as constants
- Global constructors load binaries during `initialize()`
- Host code is JIT compiled (x86_64)

### 5. ScopedContext is Critical

**Every CUDA wrapper function creates a ScopedContext!**

```cpp
CUmodule mgpuModuleLoadJIT(void *data, int optLevel) {
  ScopedContext scopedContext;  // ← Initializes CUDA on first call
  // ... CUDA operations ...
}
```

- First `ScopedContext()` calls `cuInit(0)`
- Subsequent calls reuse static context
- RAII pattern ensures context is active

---

## Testing the Traces

### Verify Constructor Trace

```bash
gdb --args python mlir/test/Examples/NVGPU/exec_engine.py

(gdb) break mlir::ExecutionEngine::create
(gdb) break mlir::translateModuleToLLVMIR
(gdb) break mlir::ExecutionEngine::packFunctionArguments
(gdb) run
(gdb) bt  # At each breakpoint, examine backtrace
```

### Verify initialize() Trace

```bash
gdb --args python mlir/test/Examples/NVGPU/exec_engine.py

(gdb) break mlir::ExecutionEngine::initialize
(gdb) break 'llvm::orc::(anonymous namespace)::GenericLLVMIRPlatformSupport::initialize'
(gdb) break mgpuModuleLoadJIT
(gdb) break cuInit
(gdb) run
# Continue to line 74 in exec_engine.py (engine.initialize())
(gdb) bt  # At each breakpoint
```

### Verify invoke() Trace

```bash
gdb --args python mlir/test/Examples/NVGPU/exec_engine.py

(gdb) break mlir::ExecutionEngine::lookup
(gdb) break llvm::orc::ExecutionSession::lookup
(gdb) break _mlir_ciface_main
(gdb) break mgpuLaunchKernel
(gdb) run
# Continue to line 78 (engine.invoke())
(gdb) bt  # At each breakpoint
```

---

## Summary

This document provides **complete frame-by-frame traces** for:

1. ✅ **ExecutionEngine Construction**: Python → Bindings → C API → C++ → LLVM IR translation
2. ✅ **engine.initialize()**: Python → Bindings → C API → C++ → LLVM ORC → Global constructors → GPU binary loading → CUDA init
3. ✅ **engine.invoke()**: Python → Bindings → C API → C++ → LLVM ORC → JIT compilation → Native execution → GPU kernel launch

All traces include:
- Source file locations with line numbers
- Code snippets showing what happens
- GDB/LLDB breakpoint commands
- Explanations of each frame
- Links to relevant source code

Use this guide to:
- Set precise breakpoints for debugging
- Understand the full call stack at any point
- Trace issues through all layers (Python → C++ → LLVM → CUDA)
- Learn how MLIR ExecutionEngine works internally

**For the GPU-specific case**: The key insight is that GPU binaries are compiled AOT (during MLIR→LLVM IR) but loaded JIT (during `initialize()`), while host code is fully JIT compiled (during first `invoke()` lookup).
