# ExecutionEngine Call Trace: Frame-by-Frame Analysis

A comprehensive walkthrough of `ExecutionEngine(module, opt_level, shared_libs)` from Python → C++ bindings → MLIR-C → MLIR C++, with special focus on CUDA/GPU execution.

---

## Table of Contents

1. [Call Stack Overview](#call-stack-overview)
2. [Frame 0: Python User Code](#frame-0-python-user-code)
3. [Frame 1: Python ExecutionEngine Wrapper](#frame-1-python-executionengine-wrapper)
4. [Frame 2: Native Extension (_mlirExecutionEngine)](#frame-2-native-extension-_mlirexecutionengine)
5. [Frame 3: Python Bindings (Nanobind/C++)](#frame-3-python-bindings-nanobin dC++)
6. [Frame 4: MLIR-C API](#frame-4-mlir-c-api)
7. [Frame 5: MLIR C++ ExecutionEngine](#frame-5-mlir-c-executionengine)
8. [Frame 6: LLVM ORC JIT](#frame-6-llvm-orc-jit)
9. [CUDA-Specific: Runtime Wrappers](#cuda-specific-runtime-wrappers)
10. [Complete Call Graph](#complete-call-graph)

---

## Call Stack Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Frame 0: Python User Code                                       │
│   nvgpucompiler.py:40                                           │
│   execution_engine.ExecutionEngine(module, opt_level, ...)     │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ Frame 1: Python Wrapper                                         │
│   mlir/python/mlir/execution_engine.py:14                       │
│   class ExecutionEngine(_execution_engine.ExecutionEngine)     │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ Frame 2: Native Extension Module                                │
│   _mlir_libs._mlirExecutionEngine.ExecutionEngine.__init__     │
│   (C++ extension loaded dynamically)                            │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ Frame 3: Python Bindings (Nanobind)                             │
│   mlir/lib/Bindings/Python/ExecutionEngineModule.cpp:75        │
│   PyExecutionEngine.__init__                                    │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ Frame 4: MLIR-C API                                              │
│   mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp:23         │
│   mlirExecutionEngineCreate()                                   │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ Frame 5: MLIR C++ ExecutionEngine                               │
│   mlir/lib/ExecutionEngine/ExecutionEngine.cpp:~300            │
│   ExecutionEngine::create()                                     │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ Frame 6: LLVM ORC JIT                                            │
│   llvm::orc::LLJIT::create()                                    │
│   JIT compilation, linking, code generation                     │
└─────────────────────────────────────────────────────────────────┘
                        ↓
                   [CUDA Runtime]
           (loaded via shared libs / global ctors)
```

---

## Frame 0: Python User Code

### Source Location
[mlir/test/Examples/NVGPU/tools/nvgpucompiler.py:38-42](/home/jeromeku/llvm-project/mlir/test/Examples/NVGPU/tools/nvgpucompiler.py#L38-L42)

### Code
```python
def jit(self, module: ir.Module) -> execution_engine.ExecutionEngine:
    """Wraps the module in a JIT execution engine."""
    return execution_engine.ExecutionEngine(
        module, opt_level=self.opt_level, shared_libs=self.shared_libs
    )
```

### What Happens
1. **Inputs**:
   - `module`: An `ir.Module` object (MLIR module in Python)
   - `opt_level`: Integer (e.g., 2 for -O2 optimization)
   - `shared_libs`: List of shared library paths (e.g., `["libmlir_cuda_runtime.so"]`)

2. **Action**: Calls the `ExecutionEngine` constructor

3. **Next**: Python looks up `execution_engine.ExecutionEngine` and resolves it to the wrapper class

---

## Frame 1: Python ExecutionEngine Wrapper

### Source Location
[mlir/python/mlir/execution_engine.py:6-14](/home/jeromeku/llvm-project/mlir/python/mlir/execution_engine.py#L6-L14)

### Code
```python
from ._mlir_libs import _mlirExecutionEngine as _execution_engine

class ExecutionEngine(_execution_engine.ExecutionEngine):
    def lookup(self, name):
        """Lookup a function emitted with the `llvm.emit_c_interface` attribute..."""
        func = self.raw_lookup("_mlir_ciface_" + name)
        # ... Python-side utilities
```

### What Happens
1. **Import**: Loads the native extension `_mlirExecutionEngine`
   - This is a compiled `.so` file: `_mlirExecutionEngine.cpython-312-x86_64-linux-gnu.so`
   - Located in: `build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/`

2. **Inheritance**: Python class inherits from the native extension's `ExecutionEngine`

3. **Constructor Delegation**: Since `__init__` is not overridden, Python calls the parent's `__init__`

4. **Next**: Call enters the native extension module

---

## Frame 2: Native Extension Module

### Source Location
Dynamic C++ extension: `_mlirExecutionEngine.cpython-312-x86_64-linux-gnu.so`

### Type Stub (for reference)
[mlir/python/mlir/_mlir_libs/_mlirExecutionEngine.pyi:15-16](/home/jeromeku/llvm-project/mlir/python/mlir/_mlir_libs/_mlirExecutionEngine.pyi#L15-L16)

```python
class ExecutionEngine:
    def __init__(self, module: _ir.Module, opt_level: int = 2,
                 shared_libs: Sequence[str] = ...) -> None: ...
```

### What Happens
1. **Dynamic Lookup**: Python's import system loaded the `.so` and registered the `ExecutionEngine` class

2. **Nanobind Glue**: The C++ binding code (using Nanobind) receives the call

3. **Type Conversion**: Python objects are converted to C++ types:
   - `module` (Python `ir.Module`) → `MlirModule` (C struct)
   - `opt_level` (Python int) → C++ `int`
   - `shared_libs` (Python list of strings) → C++ `std::vector<std::string>`

4. **Next**: Call enters the C++ binding implementation

---

## Frame 3: Python Bindings (Nanobind/C++)

### Source Location
[mlir/lib/Bindings/Python/ExecutionEngineModule.cpp:73-98](/home/jeromeku/llvm-project/mlir/lib/Bindings/Python/ExecutionEngineModule.cpp#L73-L98)

### Code
```cpp
nb::class_<PyExecutionEngine>(m, "ExecutionEngine")
    .def(
        "__init__",
        [](PyExecutionEngine &self, MlirModule module, int optLevel,
           const std::vector<std::string> &sharedLibPaths,
           bool enableObjectDump) {
          // Convert std::vector<std::string> to SmallVector<MlirStringRef>
          llvm::SmallVector<MlirStringRef, 4> libPaths;
          for (const std::string &path : sharedLibPaths)
            libPaths.push_back({path.c_str(), path.length()});

          // Call MLIR-C API
          MlirExecutionEngine executionEngine =
              mlirExecutionEngineCreate(module, optLevel, libPaths.size(),
                                        libPaths.data(), enableObjectDump);

          // Check for errors
          if (mlirExecutionEngineIsNull(executionEngine))
            throw std::runtime_error(
                "Failure while creating the ExecutionEngine.");

          // Placement new: construct PyExecutionEngine in-place
          new (&self) PyExecutionEngine(executionEngine);
        },
        nb::arg("module"), nb::arg("opt_level") = 2,
        nb::arg("shared_libs") = nb::list(),
        nb::arg("enable_object_dump") = true,
        "Create a new ExecutionEngine instance...")
```

### What Happens
1. **Lambda Constructor**: Nanobind invokes the lambda that serves as the `__init__` implementation

2. **Type Conversion**:
   - `MlirModule module`: Already a C-compatible struct wrapping the MLIR module
   - `int optLevel`: Optimization level (0-3, typically 2)
   - `std::vector<std::string> &sharedLibPaths`: Shared libraries to load
   - `bool enableObjectDump`: Whether to enable object dumping

3. **String Conversion**: Converts `std::vector<std::string>` to `SmallVector<MlirStringRef>`
   - `MlirStringRef` is a C struct: `{const char *data; size_t length}`
   - Allows passing strings without copying to the C API

4. **Call MLIR-C API**: Invokes `mlirExecutionEngineCreate()`

5. **Error Handling**: Checks if returned `MlirExecutionEngine` is null

6. **Wrapper Construction**: Uses placement new to construct `PyExecutionEngine`:
   ```cpp
   class PyExecutionEngine {
   public:
     PyExecutionEngine(MlirExecutionEngine executionEngine)
         : executionEngine(executionEngine) {}
     ~PyExecutionEngine() {
       if (!mlirExecutionEngineIsNull(executionEngine))
         mlirExecutionEngineDestroy(executionEngine);
     }
   private:
     MlirExecutionEngine executionEngine;
     std::vector<nb::object> referencedObjects;  // For GC
   };
   ```

7. **Next**: Call enters the MLIR-C API layer

---

## Frame 4: MLIR-C API

### Source Location
[mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp:22-69](/home/jeromeku/llvm-project/mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp#L22-L69)

### Header Definition
[mlir/include/mlir-c/ExecutionEngine.h:31-47](/home/jeromeku/llvm-project/mlir/include/mlir-c/ExecutionEngine.h#L31-L47)

```c
// Opaque C struct wrapping a void* to the C++ object
DEFINE_C_API_STRUCT(MlirExecutionEngine, void);

MLIR_CAPI_EXPORTED MlirExecutionEngine mlirExecutionEngineCreate(
    MlirModule op, int optLevel, int numPaths,
    const MlirStringRef *sharedLibPaths, bool enableObjectDump);
```

### Implementation
```cpp
extern "C" MlirExecutionEngine
mlirExecutionEngineCreate(MlirModule op, int optLevel, int numPaths,
                          const MlirStringRef *sharedLibPaths,
                          bool enableObjectDump) {
  // [1] ONE-TIME INITIALIZATION
  static bool initOnce = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser();  // For inline asm
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)initOnce;

  // [2] REGISTER DIALECT TRANSLATIONS
  auto &ctx = *unwrap(op)->getContext();
  mlir::registerBuiltinDialectTranslation(ctx);
  mlir::registerLLVMDialectTranslation(ctx);
  mlir::registerOpenMPDialectTranslation(ctx);

  // [3] CREATE TARGET MACHINE BUILDER
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Failed to create JITTargetMachineBuilder\n";
    return MlirExecutionEngine{nullptr};
  }

  // [4] CREATE TARGET MACHINE
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Failed to create TargetMachine\n";
    return MlirExecutionEngine{nullptr};
  }

  // [5] CONVERT SHARED LIBRARY PATHS
  SmallVector<StringRef> libPaths;
  for (unsigned i = 0; i < static_cast<unsigned>(numPaths); ++i)
    libPaths.push_back(sharedLibPaths[i].data);

  // [6] CREATE OPTIMIZER TRANSFORMER
  auto transformer = mlir::makeOptimizingTransformer(
      optLevel, /*sizeLevel=*/0, /*targetMachine=*/tmOrError->get());

  // [7] BUILD EXECUTION ENGINE OPTIONS
  ExecutionEngineOptions jitOptions;
  jitOptions.transformer = transformer;
  jitOptions.jitCodeGenOptLevel = static_cast<llvm::CodeGenOptLevel>(optLevel);
  jitOptions.sharedLibPaths = libPaths;
  jitOptions.enableObjectDump = enableObjectDump;

  // [8] CREATE EXECUTION ENGINE (C++)
  auto jitOrError = ExecutionEngine::create(unwrap(op), jitOptions);
  if (!jitOrError) {
    consumeError(jitOrError.takeError());
    return MlirExecutionEngine{nullptr};
  }

  // [9] WRAP AND RETURN
  return wrap(jitOrError->release());
}
```

### What Happens

#### [1] One-Time LLVM Initialization
```cpp
static bool initOnce = [] {
  llvm::InitializeNativeTarget();        // Initialize x86/ARM/etc codegen
  llvm::InitializeNativeTargetAsmParser(); // Parse inline assembly
  llvm::InitializeNativeTargetAsmPrinter(); // Generate assembly
  return true;
}();
```
- **Purpose**: Initialize LLVM's code generation backends
- **Lazy**: Static variable ensures this runs only once per process
- **Platform-Specific**: Initializes only the native target (e.g., x86_64 on Intel)

#### [2] Register Dialect Translations
```cpp
auto &ctx = *unwrap(op)->getContext();
mlir::registerBuiltinDialectTranslation(ctx);
mlir::registerLLVMDialectTranslation(ctx);
mlir::registerOpenMPDialectTranslation(ctx);
```
- **Purpose**: Register dialect → LLVM IR translation interfaces
- **Why**: MLIR dialects need translators to convert to LLVM IR for code generation
- **Example**: `gpu.module` ops get translated to LLVM IR with CUDA runtime calls

#### [3] Detect Host Target
```cpp
auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
```
- **Purpose**: Auto-detect the host architecture
- **Returns**: Builder with host triple (e.g., `x86_64-unknown-linux-gnu`)
- **Includes**: CPU features (AVX2, SSE4.2, etc.)

#### [4] Create Target Machine
```cpp
auto tmOrError = tmBuilderOrError->createTargetMachine();
```
- **Purpose**: Instantiate LLVM's TargetMachine for code generation
- **Result**: Configured for host architecture with all features

#### [5] Convert Library Paths
```cpp
SmallVector<StringRef> libPaths;
for (unsigned i = 0; i < numPaths; ++i)
  libPaths.push_back(sharedLibPaths[i].data);
```
- **Purpose**: Convert C-style `MlirStringRef` array to LLVM's `SmallVector<StringRef>`
- **Example**: `["libmlir_cuda_runtime.so", "libmlir_c_runner_utils.so"]`

#### [6] Create Optimizer Transformer
```cpp
auto transformer = mlir::makeOptimizingTransformer(
    optLevel, /*sizeLevel=*/0, /*targetMachine=*/tmOrError->get());
```
- **Purpose**: Create LLVM optimization pipeline
- **Returns**: Lambda that runs LLVM optimizations on an `llvm::Module`
- **Optimization Level**:
  - `0` = No optimization
  - `1` = -O1 (basic opts)
  - `2` = -O2 (default, aggressive opts)
  - `3` = -O3 (maximum opts)

#### [7] Build Options Struct
```cpp
ExecutionEngineOptions jitOptions;
jitOptions.transformer = transformer;
jitOptions.jitCodeGenOptLevel = static_cast<llvm::CodeGenOptLevel>(optLevel);
jitOptions.sharedLibPaths = libPaths;
jitOptions.enableObjectDump = enableObjectDump;
```
- **Purpose**: Package options for C++ ExecutionEngine
- **Fields**:
  - `transformer`: LLVM optimization pipeline
  - `jitCodeGenOptLevel`: Codegen optimization (None/Less/Default/Aggressive)
  - `sharedLibPaths`: Libraries to load (e.g., CUDA runtime)
  - `enableObjectDump`: Save compiled object code

#### [8] Create C++ ExecutionEngine
```cpp
auto jitOrError = ExecutionEngine::create(unwrap(op), jitOptions);
```
- **Purpose**: Call the actual C++ implementation
- **unwrap(op)**: Converts `MlirModule` (C struct) to `mlir::Operation*` (C++)

#### [9] Wrap and Return
```cpp
return wrap(jitOrError->release());
```
- **Purpose**: Convert C++ pointer to opaque C struct
- **wrap()**: Macro that wraps `ExecutionEngine*` in `MlirExecutionEngine{ptr}`
- **release()**: Release unique_ptr ownership (C API caller owns it now)

### Next
Call enters the MLIR C++ ExecutionEngine implementation

---

## Frame 5: MLIR C++ ExecutionEngine

### Source Location
[mlir/lib/ExecutionEngine/ExecutionEngine.cpp:~200-450](/home/jeromeku/llvm-project/mlir/lib/ExecutionEngine/ExecutionEngine.cpp)

### Header Definition
[mlir/include/mlir/ExecutionEngine/ExecutionEngine.h:113-146](/home/jeromeku/llvm-project/mlir/include/mlir/ExecutionEngine/ExecutionEngine.h#L113-L146)

```cpp
class ExecutionEngine {
public:
  static llvm::Expected<std::unique_ptr<ExecutionEngine>>
  create(Operation *op, const ExecutionEngineOptions &options = {},
         std::unique_ptr<llvm::TargetMachine> tm = nullptr);

  ~ExecutionEngine();

  llvm::Expected<void (*)(void **)> lookupPacked(StringRef name) const;
  llvm::Expected<void *> lookup(StringRef name) const;
  llvm::Error invokePacked(StringRef name, MutableArrayRef<void *> args = {});

private:
  /// Shared libraries to load.
  SmallVector<LibraryHandle> loadedSharedLibs;

  /// JIT execution session.
  std::unique_ptr<llvm::orc::LLJIT> jit;

  /// Cache for compiled object code.
  std::shared_ptr<SimpleObjectCache> cache;

  /// Function names for lazy compilation.
  std::vector<std::string> functionNames;

  bool isInitialized = false;
};
```

### Implementation: ExecutionEngine::create()

```cpp
llvm::Expected<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::create(Operation *op,
                        const ExecutionEngineOptions &options,
                        std::unique_ptr<llvm::TargetMachine> tm) {

  // [1] CREATE EXECUTION ENGINE OBJECT
  auto engine = std::make_unique<ExecutionEngine>(
      options.enableObjectDump,
      options.enableGDBNotificationListener,
      options.enablePerfNotificationListener);

  // [2] TRANSLATE MLIR TO LLVM IR
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  auto llvmModule = options.llvmModuleBuilder
      ? options.llvmModuleBuilder(op, *ctx)
      : translateModuleToLLVMIR(op, *ctx);
  if (!llvmModule)
    return makeStringError("could not convert to LLVM IR");

  // [3] SET UP TARGET MACHINE
  if (!tm) {
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError)
      return tmBuilderOrError.takeError();
    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError)
      return tmOrError.takeError();
    tm = std::move(*tmOrError);
  }

  // [4] SET MODULE DATA LAYOUT AND TRIPLE
  engine->setupTargetTripleAndDataLayout(llvmModule.get(), tm.get());

  // [5] COLLECT FUNCTION NAMES FOR PACKED WRAPPERS
  for (auto func : op->getRegion(0).front().getOps<LLVM::LLVMFuncOp>()) {
    engine->functionNames.push_back(func.getName().str());
  }

  // [6] CREATE PACKED FUNCTION WRAPPERS
  packFunctionArguments(llvmModule.get());

  // [7] CREATE OBJECT CACHE (IF ENABLED)
  if (options.enableObjectDump) {
    engine->cache = std::make_shared<SimpleObjectCache>();
  }

  // [8] CREATE LLJIT INSTANCE
  auto jitOrError = llvm::orc::LLJITBuilder()
      .setJITTargetMachineBuilder(std::move(*tmBuilder))
      .setObjectLinkingLayerCreator([&](ExecutionSession &ES, const Triple &TT) {
        auto objectLayer = std::make_unique<RTDyldObjectLinkingLayer>(
            ES, []() { return std::make_unique<SectionMemoryManager>(); });

        // Register JIT event listeners
        if (options.enableGDBNotificationListener)
          objectLayer->registerJITEventListener(
              *JITEventListener::createGDBRegistrationListener());
        if (options.enablePerfNotificationListener)
          objectLayer->registerJITEventListener(
              *JITEventListener::createPerfJITEventListener());

        return objectLayer;
      })
      .create();

  if (!jitOrError)
    return jitOrError.takeError();
  engine->jit = std::move(*jitOrError);

  // [9] APPLY LLVM OPTIMIZATIONS
  if (options.transformer) {
    auto err = options.transformer(llvmModule.get());
    if (err)
      return std::move(err);
  }

  // [10] ADD LLVM MODULE TO JIT
  llvm::orc::ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
  if (auto err = engine->jit->addIRModule(std::move(tsm)))
    return std::move(err);

  // [11] LOAD SHARED LIBRARIES
  SmallVector<LibraryHandle> loadedLibs;
  for (StringRef libPath : options.sharedLibPaths) {
    auto lib = loadSharedLibrary(libPath);
    if (!lib)
      return lib.takeError();
    loadedLibs.push_back(*lib);

    // Call library init function if present
    auto initSym = lib->getAddressOfSymbol(kLibraryInitFnName);
    if (initSym) {
      auto init = reinterpret_cast<LibraryInitFn>(initSym);
      llvm::StringMap<void *> exportedSymbols;
      init(exportedSymbols);

      // Register exported symbols with JIT
      engine->registerSymbols([&](MangleAndInterner interner) {
        SymbolMap symbolMap;
        for (auto &[name, ptr] : exportedSymbols) {
          symbolMap[interner(name)] = {
            llvm::orc::ExecutorAddr::fromPtr(ptr),
            llvm::JITSymbolFlags::Exported
          };
        }
        return symbolMap;
      });
    }
  }
  engine->loadedSharedLibs = std::move(loadedLibs);

  // [12] INITIALIZE JIT (RUN GLOBAL CONSTRUCTORS)
  engine->initialize();

  return std::move(engine);
}
```

### Detailed Breakdown

#### [1] Create ExecutionEngine Object
```cpp
auto engine = std::make_unique<ExecutionEngine>(
    options.enableObjectDump,
    options.enableGDBNotificationListener,
    options.enablePerfNotificationListener);
```
- Creates the wrapper object
- Sets up notification listeners for debugging

#### [2] Translate MLIR → LLVM IR
```cpp
auto llvmModule = options.llvmModuleBuilder
    ? options.llvmModuleBuilder(op, *ctx)
    : translateModuleToLLVMIR(op, *ctx);
```
- **Default Path**: `translateModuleToLLVMIR()`:
  - Walks MLIR module
  - Calls dialect translation interfaces
  - Converts each operation to LLVM IR
  - **Example**: `gpu.launch` → `cuLaunchKernel` calls

- **Result**: `llvm::Module` ready for optimization and codegen

#### [3-4] Target Machine Setup
```cpp
engine->setupTargetTripleAndDataLayout(llvmModule.get(), tm.get());
```
- Sets module's target triple (e.g., `x86_64-unknown-linux-gnu`)
- Sets data layout (struct alignment, pointer sizes, etc.)
- **Critical**: Ensures IR matches host architecture

#### [5-6] Function Wrapper Generation
```cpp
for (auto func : op->getRegion(0).front().getOps<LLVM::LLVMFuncOp>()) {
  engine->functionNames.push_back(func.getName().str());
}
packFunctionArguments(llvmModule.get());
```

**What `packFunctionArguments()` does:**

For every function `foo(arg0: T0, arg1: T1) -> T2`, generates:

```cpp
void _mlir_foo(void **packed_args) {
  // Unpack arguments
  T0 arg0 = *reinterpret_cast<T0*>(packed_args[0]);
  T1 arg1 = *reinterpret_cast<T1*>(packed_args[1]);

  // Call original function
  T2 result = foo(arg0, arg1);

  // Pack result
  *reinterpret_cast<T2*>(packed_args[2]) = result;
}
```

**Why?** Provides a uniform ABI for Python to call any function:
```python
args = [ctypes.pointer(arg0), ctypes.pointer(arg1), ctypes.pointer(result)]
engine.invoke("foo", args)
```

#### [7] Object Cache Setup
```cpp
if (options.enableObjectDump) {
  engine->cache = std::make_shared<SimpleObjectCache>();
}
```
- Caches compiled native code
- Allows dumping to `.o` files for inspection

#### [8] Create LLJIT
```cpp
auto jitOrError = llvm::orc::LLJITBuilder()
    .setJITTargetMachineBuilder(...)
    .setObjectLinkingLayerCreator(...)
    .create();
engine->jit = std::move(*jitOrError);
```

**LLJIT** is LLVM's JIT compilation engine:
- **IR Compile Layer**: Compiles LLVM IR → machine code
- **Object Linking Layer**: Links object files, resolves symbols
- **Execution Layer**: Maps code into memory, makes it executable

**Event Listeners**:
- **GDB**: Notifies GDB about JIT-compiled code for debugging
- **Perf**: Notifies Linux `perf` tool for profiling

#### [9] Apply Optimizations
```cpp
if (options.transformer) {
  auto err = options.transformer(llvmModule.get());
}
```
- Runs LLVM optimization passes (e.g., -O2)
- **Example passes**: inlining, constant folding, loop unrolling, vectorization

#### [10] Add Module to JIT
```cpp
llvm::orc::ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
if (auto err = engine->jit->addIRModule(std::move(tsm)))
  return std::move(err);
```
- Transfers ownership of LLVM IR to JIT
- **Lazy Compilation**: Code is compiled when first looked up/invoked

#### [11] Load Shared Libraries
```cpp
for (StringRef libPath : options.sharedLibPaths) {
  auto lib = loadSharedLibrary(libPath);  // dlopen()
  loadedLibs.push_back(*lib);

  // Call library init function
  auto initSym = lib->getAddressOfSymbol("__mlir_execution_engine_init");
  if (initSym) {
    auto init = reinterpret_cast<LibraryInitFn>(initSym);
    llvm::StringMap<void *> exportedSymbols;
    init(exportedSymbols);  // Library provides symbols

    // Register with JIT
    engine->registerSymbols(...);
  }
}
```

**Example**: Loading `libmlir_cuda_runtime.so`:
1. **dlopen()** loads the library
2. **getAddressOfSymbol()** looks up `__mlir_execution_engine_init`
3. **init()** calls the library's init function
4. **Library returns symbols**: `mgpuModuleLoad`, `mgpuLaunchKernel`, etc.
5. **registerSymbols()** makes them available to JIT-compiled code

#### [12] Initialize JIT
```cpp
engine->initialize();
```

**What happens** (from [ExecutionEngine.cpp:449-456](/home/jeromeku/llvm-project/mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L449-L456)):

```cpp
void ExecutionEngine::initialize() {
  if (isInitialized)
    return;

  // Run global constructors
  cantFail(jit->initialize(jit->getMainJITDylib()));
  isInitialized = true;
}
```

**Global Constructors**:
- LLVM IR can have `llvm.mlir.global_ctors` array
- Contains functions to run at initialization
- **GPU Use Case**: Loads GPU kernel binaries

**Example**: After translating `gpu.module` to LLVM IR:
```llvm
@__cuda_module_binary = constant [1024 x i8] c"...<PTX/CUBIN>..."
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [
  { i32 65535, void ()* @__cuda_module_init, i8* null }
]

define void @__cuda_module_init() {
  call void @mgpuModuleLoad(i8* @__cuda_module_binary)
  ret void
}
```

When `initialize()` runs:
1. JIT finds `@llvm.global_ctors`
2. Executes `@__cuda_module_init`
3. Calls `mgpuModuleLoad()` (from `libmlir_cuda_runtime.so`)
4. CUDA driver loads the kernel binary

### Return
Returns `std::unique_ptr<ExecutionEngine>` wrapped in `llvm::Expected`

---

## Frame 6: LLVM ORC JIT

### LLVM ORC Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       LLJIT                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  IR Transform Layer (optional)                       │  │
│  │    - Optimization passes                             │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       ↓                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  IR Compile Layer                                    │  │
│  │    - LLVM IR → Object Code                          │  │
│  │    - Uses TargetMachine for codegen                 │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       ↓                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Object Linking Layer (RTDyldObjectLinkingLayer)    │  │
│  │    - Link object files                               │  │
│  │    - Resolve symbols                                 │  │
│  │    - Apply relocations                               │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       ↓                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Memory Manager (SectionMemoryManager)               │  │
│  │    - Allocate memory for code                        │  │
│  │    - Make pages executable                           │  │
│  │    - Track allocations for cleanup                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### What Happens During Compilation

When `jit->addIRModule()` is called:

1. **Module Registration**: IR module is added to the JIT dylib
2. **Symbol Export**: Function names are exported for lookup
3. **Lazy Compilation**: Code is NOT compiled yet!

When `lookupPacked("foo")` is called:

1. **Symbol Lookup**: JIT searches for `_mlir_foo` symbol
2. **Compilation Triggered**:
   - IR Compile Layer compiles LLVM IR to object code
   - Object Linking Layer resolves symbols and applies relocations
3. **Memory Allocation**: SectionMemoryManager allocates executable memory
4. **Code Placement**: Object code copied to executable pages
5. **Protection Change**: `mprotect(PROT_READ | PROT_EXEC)`
6. **Return Pointer**: Returns function pointer

---

## CUDA-Specific: Runtime Wrappers

### Overview

GPU kernels compiled from `gpu.module` need runtime support for:
- Loading kernel binaries
- Allocating device memory
- Launching kernels
- Copying data

### CUDA Runtime Wrappers Library

**Source**: [mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp](/home/jeromeku/llvm-project/mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp)

**Compiled to**: `libmlir_cuda_runtime.so`

### Key Functions

#### 1. Module Loading: `mgpuModuleLoad`

```cpp
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuModuleLoad(void *data) {
  ScopedContext scopedContext;

  CUmodule module;
  CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));

  // Store module handle
  loadedModules.push_back(module);

  debug_print("Loaded CUDA module: %p\n", module);
}
```

**Called By**: Global constructor generated during MLIR → LLVM translation

**What It Does**:
1. Establishes CUDA context
2. Calls `cuModuleLoadData()` to load PTX/CUBIN
3. Stores module handle for later kernel lookups

#### 2. Kernel Launch: `mgpuLaunchKernel`

```cpp
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                 intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                 intptr_t blockZ, int32_t smem, CUstream stream,
                 void **params, void **extra) {
  ScopedContext scopedContext;

  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ,
                                      blockX, blockY, blockZ,
                                      smem, stream, params, extra));

  debug_print("Launched kernel with grid=(%ld,%ld,%ld) block=(%ld,%ld,%ld)\n",
              gridX, gridY, gridZ, blockX, blockY, blockZ);
}
```

**Called By**: JIT-compiled code from `gpu.launch_func`

**Example Translation**:
```mlir
gpu.launch_func @kernel
  blocks in (%c1024, %c1, %c1)
  threads in (%c256, %c1, %c1)
  args(%arg0 : memref<f32>)
```

↓ Translated to LLVM IR ↓

```llvm
define void @launch_kernel() {
  %module = load %CUmodule, %CUmodule* @module_handle
  %kernel = call %CUfunction @cuModuleGetFunction(%module, "kernel")

  %params = alloca [1 x i8*]
  %params[0] = bitcast memref %arg0 to i8*

  call void @mgpuLaunchKernel(
    %kernel,
    i64 1024, i64 1, i64 1,  ; grid
    i64 256, i64 1, i64 1,    ; block
    i32 0,                     ; smem
    %CUstream null,            ; stream
    i8** %params,              ; params
    i8** null                  ; extra
  )
  ret void
}
```

#### 3. Memory Management

```cpp
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *mgpuMemAlloc(uint64_t sizeBytes) {
  ScopedContext scopedContext;
  CUdeviceptr ptr;
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&ptr, sizeBytes));
  debug_print("Allocated %lu bytes on device: %p\n", sizeBytes, (void*)ptr);
  return reinterpret_cast<void *>(ptr);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuMemFree(void *ptr) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
  debug_print("Freed device memory: %p\n", ptr);
}
```

#### 4. Data Transfer

```cpp
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemcpy(void *dst, void *src, size_t sizeBytes, CUstream stream) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuMemcpyAsync(
      reinterpret_cast<CUdeviceptr>(dst),
      reinterpret_cast<CUdeviceptr>(src),
      sizeBytes, stream));
}
```

### How CUDA Wrappers Are Loaded

#### Step 1: Shared Library Specification
```python
# In nvgpucompiler.py
compiler = NvgpuCompiler(
    shared_libs=[
        f"{support_lib_dir}/libmlir_cuda_runtime.so",
        f"{support_lib_dir}/libmlir_c_runner_utils.so",
    ]
)
```

#### Step 2: Library Loading During ExecutionEngine Creation
```cpp
// In ExecutionEngine::create()
for (StringRef libPath : options.sharedLibPaths) {
  auto lib = loadSharedLibrary(libPath);  // dlopen()

  // Look for init function
  auto initSym = lib->getAddressOfSymbol("__mlir_execution_engine_init");
  if (initSym) {
    auto init = reinterpret_cast<LibraryInitFn>(initSym);
    llvm::StringMap<void *> exportedSymbols;
    init(exportedSymbols);  // Library provides its symbols

    // Register symbols with JIT
    registerSymbols([&](MangleAndInterner interner) {
      SymbolMap symbolMap;
      for (auto &[name, ptr] : exportedSymbols) {
        symbolMap[interner(name)] = {
          llvm::orc::ExecutorAddr::fromPtr(ptr),
          llvm::JITSymbolFlags::Exported
        };
      }
      return symbolMap;
    });
  }
}
```

#### Step 3: Symbol Resolution
When JIT-compiled code calls `@mgpuModuleLoad`:
1. **Linker looks up symbol** in registered libraries
2. **Finds `mgpuModuleLoad` in `libmlir_cuda_runtime.so`**
3. **Resolves relocation** in generated code
4. **Call executes** CUDA API

### ScopedContext Pattern

```cpp
class ScopedContext {
public:
  ScopedContext() {
    static CUcontext context = [] {
      CUDA_REPORT_IF_ERROR(cuInit(0));
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

**Purpose**: RAII wrapper for CUDA context
- **Constructor**: Pushes CUDA context onto current thread
- **Destructor**: Pops CUDA context
- **Thread-Safe**: Each thread gets its own context stack

**Why Needed**: CUDA driver API requires an active context for all operations

---

## Complete Call Graph

### Visual Representation

```
Python                                C++ Bindings                MLIR-C                          MLIR C++                      LLVM
═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

execution_engine.ExecutionEngine()
  │
  ├─→ __init__(module, opt_level, shared_libs)
  │     ↓
  │   _mlirExecutionEngine.ExecutionEngine.__init__
  │     │
  │     └─→ PyExecutionEngine.__init__()
  │           │
  │           ├─→ Convert args to C types
  │           │   • module → MlirModule
  │           │   • opt_level → int
  │           │   • shared_libs → SmallVector<MlirStringRef>
  │           │
  │           └─→ mlirExecutionEngineCreate()
  │                 │
  │                 ├─→ [1] InitializeNativeTarget()
  │                 │
  │                 ├─→ [2] Register dialect translations
  │                 │   • Builtin → LLVM IR
  │                 │   • LLVM → LLVM IR
  │                 │   • OpenMP → LLVM IR
  │                 │
  │                 ├─→ [3] JITTargetMachineBuilder::detectHost()
  │                 │
  │                 ├─→ [4] createTargetMachine()
  │                 │
  │                 ├─→ [5] makeOptimizingTransformer(optLevel)
  │                 │
  │                 └─→ ExecutionEngine::create()
  │                       │
  │                       ├─→ [6] translateModuleToLLVMIR()
  │                       │     • Walk MLIR operations
  │                       │     • Call dialect translators
  │                       │     • Generate LLVM IR
  │                       │
  │                       ├─→ [7] setupTargetTripleAndDataLayout()
  │                       │
  │                       ├─→ [8] packFunctionArguments()
  │                       │     • Generate _mlir_* wrappers
  │                       │
  │                       ├─→ [9] LLJITBuilder::create()
  │                       │     │
  │                       │     └─→ LLJIT::LLJIT()
  │                       │           │
  │                       │           ├─→ IRCompileLayer
  │                       │           ├─→ RTDyldObjectLinkingLayer
  │                       │           └─→ SectionMemoryManager
  │                       │
  │                       ├─→ [10] transformer(llvmModule)
  │                       │      • Run LLVM optimization passes
  │                       │
  │                       ├─→ [11] jit->addIRModule(tsm)
  │                       │
  │                       ├─→ [12] Load shared libraries
  │                       │      For each lib in shared_libs:
  │                       │        ├─→ loadSharedLibrary()  (dlopen)
  │                       │        ├─→ getAddressOfSymbol("__mlir_execution_engine_init")
  │                       │        ├─→ Call init function
  │                       │        │     │
  │                       │        │     └─→ libmlir_cuda_runtime.so init:
  │                       │        │           • cuInit()
  │                       │        │           • cuDevicePrimaryCtxRetain()
  │                       │        │           • Return symbol map:
  │                       │        │             - mgpuModuleLoad
  │                       │        │             - mgpuLaunchKernel
  │                       │        │             - mgpuMemAlloc
  │                       │        │             - mgpuMemFree
  │                       │        │             - ...
  │                       │        │
  │                       │        └─→ registerSymbols(symbolMap)
  │                       │              • Add to JIT symbol table
  │                       │
  │                       └─→ [13] initialize()
  │                             │
  │                             └─→ jit->initialize(mainJITDylib)
  │                                   │
  │                                   └─→ Run @llvm.global_ctors
  │                                         │
  │                                         └─→ @__cuda_module_init()
  │                                               │
  │                                               └─→ mgpuModuleLoad(binary)
  │                                                     │
  │                                                     └─→ cuModuleLoadData()
  │                                                           • Parse PTX/CUBIN
  │                                                           • JIT compile to SASS
  │                                                           • Store handle
  │
  └─→ Return PyExecutionEngine wrapper
```

### Numbered Steps Summary

| # | Layer | Function | What Happens |
|---|-------|----------|--------------|
| 0 | Python | `ExecutionEngine(module, ...)` | User creates ExecutionEngine |
| 1 | Python Wrapper | Class delegation | Inherits from native extension |
| 2 | Nanobind | `__init__` lambda | Convert Python → C++ types |
| 3 | MLIR-C | `mlirExecutionEngineCreate` | Initialize LLVM, setup target |
| 4 | MLIR C++ | `ExecutionEngine::create` | Main orchestration |
| 5 | MLIR C++ | `translateModuleToLLVMIR` | MLIR → LLVM IR |
| 6 | MLIR C++ | `packFunctionArguments` | Generate ABI wrappers |
| 7 | LLVM | `LLJITBuilder::create` | Create JIT engine |
| 8 | MLIR C++ | `transformer(module)` | Run optimizations |
| 9 | LLVM | `jit->addIRModule` | Register IR for lazy compilation |
| 10 | MLIR C++ | `loadSharedLibrary` | dlopen runtime libs |
| 11 | Runtime Lib | `__mlir_execution_engine_init` | Library exports symbols |
| 12 | MLIR C++ | `registerSymbols` | Add symbols to JIT |
| 13 | MLIR C++ | `initialize()` | Run global constructors |
| 14 | LLVM | Run `@llvm.global_ctors` | Execute initialization code |
| 15 | CUDA Runtime | `mgpuModuleLoad` | Load GPU kernel binary |
| 16 | CUDA Driver | `cuModuleLoadData` | Parse and JIT PTX |

---

## Key Takeaways

### 1. **Layered Architecture**
- Python → Nanobind → MLIR-C → MLIR C++ → LLVM → Native
- Each layer has clear responsibilities
- C API provides ABI stability

### 2. **Lazy Compilation**
- Code is NOT compiled when ExecutionEngine is created
- Compilation happens on first `lookup()` or `invoke()`
- Enables fast startup

### 3. **Packed Function Wrappers**
- Generated automatically for all functions
- Provides uniform calling convention
- Enables Python interop without manual marshaling

### 4. **Shared Library Integration**
- Libraries can provide init functions
- Symbols are registered with JIT
- Enables runtime support (CUDA, MKL, etc.)

### 5. **Global Constructors**
- `llvm.global_ctors` runs during `initialize()`
- Used to load GPU kernel binaries
- Critical for GPU execution

### 6. **CUDA Integration**
- `libmlir_cuda_runtime.so` provides wrappers
- Loaded via `shared_libs` parameter
- Symbols resolved at link time
- Kernels loaded during initialization

---

## Debugging Tips

### 1. Enable CUDA Debug Output
```bash
export MLIR_CUDA_DEBUG=1
```
Prints debug messages from CudaRuntimeWrappers

### 2. Dump LLVM IR
```python
# Before creating ExecutionEngine
print(module)
```

### 3. Dump Object Code
```python
ee = ExecutionEngine(module, enable_object_dump=True)
ee.dump_to_object_file("output.o")
```

Then disassemble:
```bash
objdump -d output.o
```

### 4. Use GDB with JIT
```bash
gdb --args python my_script.py
(gdb) set environment MLIR_CUDA_DEBUG 1
(gdb) break mgpuModuleLoad
(gdb) run
```

### 5. Check Symbol Resolution
```python
ptr = ee.raw_lookup("_mlir_my_function")
print(f"Function at: 0x{ptr:x}")
```

---

## References

- **Python Layer**: [mlir/python/mlir/execution_engine.py](/home/jeromeku/llvm-project/mlir/python/mlir/execution_engine.py)
- **Nanobind Bindings**: [mlir/lib/Bindings/Python/ExecutionEngineModule.cpp](/home/jeromeku/llvm-project/mlir/lib/Bindings/Python/ExecutionEngineModule.cpp)
- **MLIR-C API**: [mlir/include/mlir-c/ExecutionEngine.h](/home/jeromeku/llvm-project/mlir/include/mlir-c/ExecutionEngine.h)
- **MLIR-C Implementation**: [mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp](/home/jeromeku/llvm-project/mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp)
- **MLIR C++ Header**: [mlir/include/mlir/ExecutionEngine/ExecutionEngine.h](/home/jeromeku/llvm-project/mlir/include/mlir/ExecutionEngine/ExecutionEngine.h)
- **MLIR C++ Implementation**: [mlir/lib/ExecutionEngine/ExecutionEngine.cpp](/home/jeromeku/llvm-project/mlir/lib/ExecutionEngine/ExecutionEngine.cpp)
- **CUDA Runtime Wrappers**: [mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp](/home/jeromeku/llvm-project/mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp)

---

This trace provides a complete picture of how Python code compiles and executes MLIR/GPU code through multiple abstraction layers, culminating in native CUDA kernel execution! 🚀
