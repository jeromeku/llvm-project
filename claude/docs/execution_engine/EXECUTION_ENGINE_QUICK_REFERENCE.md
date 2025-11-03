# ExecutionEngine Quick Reference

Quick lookup for ExecutionEngine call stack and key concepts.

---

## Quick Call Stack

```python
# Python
execution_engine.ExecutionEngine(module, opt_level=2, shared_libs=[...])
  ↓
# Nanobind (C++)
PyExecutionEngine.__init__(MlirModule, int, vector<string>)
  ↓
# MLIR-C
mlirExecutionEngineCreate(MlirModule, int, int, MlirStringRef*, bool)
  ↓
# MLIR C++
ExecutionEngine::create(Operation*, ExecutionEngineOptions, TargetMachine*)
  ↓
# LLVM
LLJIT::create() → addIRModule() → initialize()
  ↓
# CUDA (if GPU code)
@llvm.global_ctors → mgpuModuleLoad() → cuModuleLoadData()
```

---

## File Map

| Layer | File | Purpose |
|-------|------|---------|
| Python User | `nvgpucompiler.py:40` | Calls ExecutionEngine |
| Python Wrapper | `mlir/python/mlir/execution_engine.py` | Thin Python wrapper |
| Nanobind | `mlir/lib/Bindings/Python/ExecutionEngineModule.cpp` | Python→C++ bridge |
| MLIR-C API | `mlir/include/mlir-c/ExecutionEngine.h` | C API declarations |
| MLIR-C Impl | `mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp` | C API implementation |
| MLIR C++ API | `mlir/include/mlir/ExecutionEngine/ExecutionEngine.h` | C++ interface |
| MLIR C++ Impl | `mlir/lib/ExecutionEngine/ExecutionEngine.cpp` | Main implementation |
| CUDA Runtime | `mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp` | CUDA helpers |

---

## Key Functions

### Python → C++
```python
# Python
ee = ExecutionEngine(module, opt_level=2, shared_libs=["libmlir_cuda_runtime.so"])
```

### C++ Binding
```cpp
// ExecutionEngineModule.cpp:75
.def("__init__", [](PyExecutionEngine &self, MlirModule module, ...) {
  MlirExecutionEngine ee = mlirExecutionEngineCreate(...);
  new (&self) PyExecutionEngine(ee);
})
```

### MLIR-C API
```c
// ExecutionEngine.h:45
MlirExecutionEngine mlirExecutionEngineCreate(
    MlirModule op, int optLevel, int numPaths,
    const MlirStringRef *sharedLibPaths, bool enableObjectDump);
```

### MLIR C++
```cpp
// ExecutionEngine.cpp:~300
llvm::Expected<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::create(Operation *op, const ExecutionEngineOptions &options);
```

---

## Key Data Structures

### Python Side
```python
class ExecutionEngine:
    def __init__(self, module: ir.Module, opt_level: int, shared_libs: List[str]): ...
    def lookup(self, name: str) -> callable: ...
    def invoke(self, name: str, *args): ...
```

### C++ Side
```cpp
class PyExecutionEngine {
  MlirExecutionEngine executionEngine;  // C API handle
  std::vector<nb::object> referencedObjects;  // Python objects (GC)
};

class ExecutionEngine {
  std::unique_ptr<llvm::orc::LLJIT> jit;  // JIT compiler
  SmallVector<LibraryHandle> loadedSharedLibs;  // dlopen handles
  std::shared_ptr<SimpleObjectCache> cache;  // Compiled code
  std::vector<std::string> functionNames;  // For packed wrappers
  bool isInitialized = false;
};
```

### MLIR-C
```c
struct MlirExecutionEngine {
  void *ptr;  // Points to C++ ExecutionEngine*
};
```

---

## Execution Flow

### 1. Construction
```
ExecutionEngine(module, opt_level, shared_libs)
  ├─→ Initialize LLVM target
  ├─→ Register dialect translators
  ├─→ Create TargetMachine
  ├─→ Translate MLIR → LLVM IR
  ├─→ Generate packed function wrappers
  ├─→ Create LLJIT
  ├─→ Run LLVM optimizations
  ├─→ Add IR to JIT (not compiled yet!)
  ├─→ Load shared libraries (dlopen)
  │   └─→ Call __mlir_execution_engine_init
  │       └─→ Register symbols with JIT
  └─→ Initialize (run global constructors)
      └─→ Load GPU kernels
```

### 2. Function Lookup
```
ee.lookup("my_function")
  ├─→ raw_lookup("_mlir_ciface_my_function")
  │   └─→ mlirExecutionEngineLookupPacked()
  │       └─→ ExecutionEngine::lookupPacked()
  │           └─→ jit->lookup("_mlir_my_function")
  │               ├─→ **Compilation triggered here!**
  │               ├─→ IR → Object code
  │               ├─→ Link & resolve symbols
  │               ├─→ Allocate executable memory
  │               └─→ Return function pointer
  └─→ Wrap in ctypes.CFUNCTYPE
```

### 3. Function Invocation
```
ee.invoke("my_function", arg1, arg2, result)
  ├─→ Pack arguments: [ptr(arg1), ptr(arg2), ptr(result)]
  ├─→ Call _mlir_my_function(packed_args)
  │   ├─→ Unpack: arg1, arg2
  │   ├─→ Call original: my_function(arg1, arg2)
  │   └─→ Pack result
  └─→ Extract result from result pointer
```

---

## CUDA-Specific Flow

### Module Translation
```mlir
gpu.module @kernels {
  gpu.func @my_kernel(...) {
    // Kernel code
  }
}
```
↓ Translate to LLVM IR ↓
```llvm
@__cuda_module_binary = constant [N x i8] c"...<PTX/CUBIN>..."
@llvm.global_ctors = [{ i32 65535, void()* @__cuda_init, i8* null }]

define void @__cuda_init() {
  call void @mgpuModuleLoad(i8* @__cuda_module_binary)
}
```

### Initialization
```
ExecutionEngine::initialize()
  └─→ jit->initialize(mainJITDylib)
      └─→ Run @llvm.global_ctors
          └─→ @__cuda_init()
              └─→ mgpuModuleLoad(binary)  [from libmlir_cuda_runtime.so]
                  └─→ cuModuleLoadData(binary)  [CUDA Driver API]
                      ├─→ Parse PTX/CUBIN
                      ├─→ JIT compile to SASS
                      └─→ Store CUmodule handle
```

### Kernel Launch
```python
ee.invoke("launch_my_kernel", grid, block, args)
```
↓ Compiled code ↓
```llvm
define void @launch_my_kernel(...) {
  call void @mgpuLaunchKernel(
    %CUfunction kernel,
    i64 gridX, i64 gridY, i64 gridZ,
    i64 blockX, i64 blockY, i64 blockZ,
    i32 smem, %CUstream stream,
    i8** params, i8** extra
  )
}
```
↓ Runtime ↓
```cpp
void mgpuLaunchKernel(...) {
  ScopedContext ctx;  // Push CUDA context
  cuLaunchKernel(function, gridX, ...);  // CUDA Driver API
}
```

---

## Shared Library Loading

### Specification
```python
shared_libs = [
    "libmlir_cuda_runtime.so",
    "libmlir_c_runner_utils.so"
]
```

### Loading Process
```cpp
for (StringRef path : shared_libs) {
  auto lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);

  // Look for init function
  auto init = dlsym(lib, "__mlir_execution_engine_init");
  if (init) {
    StringMap<void*> symbols;
    ((LibraryInitFn)init)(symbols);  // Library provides symbols

    // Register with JIT
    for (auto [name, ptr] : symbols) {
      jit->addSymbol(name, ptr);
    }
  }
}
```

### CUDA Runtime Symbols
```cpp
// libmlir_cuda_runtime.so provides:
symbols["mgpuModuleLoad"] = &mgpuModuleLoad;
symbols["mgpuLaunchKernel"] = &mgpuLaunchKernel;
symbols["mgpuMemAlloc"] = &mgpuMemAlloc;
symbols["mgpuMemFree"] = &mgpuMemFree;
symbols["mgpuMemcpy"] = &mgpuMemcpy;
// ... more symbols
```

---

## Common Patterns

### Pattern 1: Basic Usage
```python
from mlir import execution_engine, ir

# Create ExecutionEngine
ee = execution_engine.ExecutionEngine(module, opt_level=2)

# Lookup function
my_func = ee.lookup("my_function")

# Call it
result = ctypes.c_float()
my_func(ctypes.pointer(result))
print(result.value)
```

### Pattern 2: With CUDA
```python
ee = execution_engine.ExecutionEngine(
    module,
    opt_level=2,
    shared_libs=["libmlir_cuda_runtime.so"]
)

# Kernels are loaded automatically during initialization
# Just invoke the launch function
ee.invoke("launch_kernel", grid_x, grid_y, grid_z, ...)
```

### Pattern 3: Debugging
```python
import os
os.environ['MLIR_CUDA_DEBUG'] = '1'

ee = execution_engine.ExecutionEngine(
    module,
    opt_level=0,  # No optimization for easier debugging
    enable_object_dump=True
)

ee.dump_to_object_file("debug.o")
```

---

## Important Functions

### ExecutionEngine Methods

| Method | Purpose | Example |
|--------|---------|---------|
| `__init__(module, opt_level, shared_libs)` | Create engine | `ExecutionEngine(m, 2, [...])` |
| `lookup(name)` | Get callable | `func = ee.lookup("foo")` |
| `invoke(name, *args)` | Call function | `ee.invoke("foo", arg1, arg2)` |
| `raw_lookup(name)` | Get function pointer | `ptr = ee.raw_lookup("_mlir_foo")` |
| `register_runtime(name, callback)` | Add symbol | `ee.register_runtime("my_sym", cb)` |
| `dump_to_object_file(path)` | Save compiled code | `ee.dump_to_object_file("out.o")` |

### CUDA Runtime Wrappers

| Function | CUDA API | Purpose |
|----------|----------|---------|
| `mgpuModuleLoad(data)` | `cuModuleLoadData` | Load kernel binary |
| `mgpuLaunchKernel(...)` | `cuLaunchKernel` | Launch kernel |
| `mgpuMemAlloc(size)` | `cuMemAlloc` | Allocate device memory |
| `mgpuMemFree(ptr)` | `cuMemFree` | Free device memory |
| `mgpuMemcpy(dst, src, size)` | `cuMemcpyAsync` | Copy data |
| `mgpuStreamCreate()` | `cuStreamCreate` | Create stream |
| `mgpuStreamSynchronize(stream)` | `cuStreamSynchronize` | Wait for stream |

---

## Key Concepts

### 1. Lazy Compilation
- Code is NOT compiled during `ExecutionEngine()` construction
- Compilation happens on first `lookup()` or `invoke()`
- Each function compiled independently

### 2. Packed Function Wrappers
For every function `foo(T0, T1) -> T2`, generates:
```cpp
void _mlir_foo(void **args) {
  T0 arg0 = *static_cast<T0*>(args[0]);
  T1 arg1 = *static_cast<T1*>(args[1]);
  T2 result = foo(arg0, arg1);
  *static_cast<T2*>(args[2]) = result;
}
```
Enables uniform Python calling convention.

### 3. Global Constructors
- Array `@llvm.global_ctors` in LLVM IR
- Run during `ExecutionEngine::initialize()`
- Used to load GPU kernel binaries
- Example: `@__cuda_module_init` loads PTX

### 4. Symbol Resolution
JIT resolves symbols in this order:
1. Symbols from loaded shared libraries
2. Symbols explicitly registered via `register_runtime()`
3. Process symbols (if accessible)

### 5. CUDA Context Management
- `ScopedContext` RAII wrapper
- Pushes context on construction
- Pops context on destruction
- Thread-local context stack

---

## Optimization Levels

| Level | Description | When to Use |
|-------|-------------|-------------|
| 0 | No optimization | Debugging, fast compile |
| 1 | Basic optimizations | Quick testing |
| 2 | Aggressive optimizations (default) | Production |
| 3 | Maximum optimizations | Performance-critical |

**LLVM Passes at -O2**:
- Inlining
- Constant folding
- Dead code elimination
- Loop unrolling
- Vectorization
- ...

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Failure while creating ExecutionEngine" | Invalid MLIR/LLVM IR | Check module validity |
| "Unknown function X" | Function not found | Check function name, may need `_mlir_ciface_` prefix |
| Symbol resolution error | Missing library | Add to `shared_libs` |
| CUDA error during init | CUDA unavailable | Check CUDA installation, `nvidia-smi` |
| Segfault in JIT code | Type mismatch | Check argument types match |

### Debugging Checklist

1. ✅ Check module is valid: `module.operation.verify()`
2. ✅ Enable debug output: `MLIR_CUDA_DEBUG=1`
3. ✅ Use opt_level=0 for easier debugging
4. ✅ Dump object code: `ee.dump_to_object_file("debug.o")`
5. ✅ Check symbols: `ee.raw_lookup("function_name")`
6. ✅ Use GDB: `gdb --args python script.py`

---

## Summary

**ExecutionEngine = MLIR JIT Compiler**

```
Python Code
    ↓
MLIR Module
    ↓ (Translation)
LLVM IR
    ↓ (Optimization)
Optimized LLVM IR
    ↓ (Codegen)
Native Machine Code
    ↓ (Execution)
Results
```

**Key Points:**
- Lazy compilation (fast startup)
- LLVM-backed (state-of-the-art optimizations)
- Extensible (shared libraries, custom symbols)
- GPU-ready (CUDA runtime integration)
- Python-friendly (uniform ABI via packed wrappers)

---

For the complete detailed trace, see: [EXECUTION_ENGINE_FRAME_BY_FRAME.md](/home/jeromeku/llvm-project/EXECUTION_ENGINE_FRAME_BY_FRAME.md)
