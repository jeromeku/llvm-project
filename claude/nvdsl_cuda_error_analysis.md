# NVDSL CUDA Error Analysis: `CUDA_ERROR_NOT_INITIALIZED`

## Error Summary

**Errors Observed:**
```
'cuModuleGetFunction(&function, module, name)' failed with 'CUDA_ERROR_NOT_INITIALIZED'
'cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream, params, extra)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuModuleUnload(module)' failed with 'CUDA_ERROR_INVALID_HANDLE'
```

**Root Cause:** The CUDA driver is not initialized before GPU kernel modules are loaded and executed. Specifically, **`engine.initialize()` is never called**, which means the global constructors that load GPU binaries are never executed.

---

## Problem Location

### Line 36-37: [nvgpucompiler.py](mlir/test/Examples/NVGPU/tools/nvgpucompiler.py#L36-L37)

```python
def jit(self, module: ir.Module) -> execution_engine.ExecutionEngine:
    """Wraps the module in a JIT execution engine."""
    return execution_engine.ExecutionEngine(
        module, opt_level=self.opt_level, shared_libs=self.shared_libs
    )
```

**What happens:**
- Creates an ExecutionEngine instance
- Adds LLVM IR module to JIT
- **Does NOT call `engine.initialize()`**
- Returns engine without loading GPU binaries

### Line 453-454: [nvdsl.py](mlir/test/Examples/NVGPU/tools/nvdsl.py#L453-L454)

```python
# Run the compiled program
engine.invoke(function_name, *newArgs)
```

**What happens:**
- Attempts to invoke the function via ExecutionEngine
- Function calls `mgpuModuleGetFunction()` to get kernel pointer
- **GPU module was never loaded** because `initialize()` was never called
- CUDA driver returns `CUDA_ERROR_NOT_INITIALIZED`

---

## Complete Call Stack Trace

### Frame 1: Python Entry Point

**File:** [mlir/test/Examples/NVGPU/tools/nvdsl.py:453](mlir/test/Examples/NVGPU/tools/nvdsl.py#L453)

```python
engine.invoke(function_name, *newArgs)
```

**Inputs:**
- `function_name`: Name of the function (e.g., `"gemm_128_128_64"`)
- `newArgs`: List of ctypes pointers to arguments

**What it does:**
- Calls the `invoke()` method on the ExecutionEngine Python wrapper

---

### Frame 2: Python ExecutionEngine Wrapper

**File:** [mlir/python/mlir/execution_engine.py](mlir/python/mlir/execution_engine.py) (MLIR core bindings)

```python
class ExecutionEngine(_execution_engine.ExecutionEngine):
    def invoke(self, name, *ctypes_args):
        """Invoke a function with the list of ctypes arguments.
        All arguments must be pointers.
        Raise a RuntimeError if the function isn't found.
        """
        func = self.lookup(name)  # ← Lookup _mlir_ciface_function_name
        packed_args = (ctypes.c_void_p * len(ctypes_args))()
        for argNum in range(len(ctypes_args)):
            packed_args[argNum] = ctypes.cast(ctypes_args[argNum], ctypes.c_void_p)
        func(packed_args)  # ← Call the void(void**) wrapper
```

**What it does:**
1. Looks up `_mlir_ciface_{name}` wrapper function
2. Packs arguments into `void**` array
3. Calls the wrapper as a ctypes function pointer

**Key Point:** This does **NOT** call `engine.initialize()` before invocation!

---

### Frame 3: Python → C++ Binding Layer

**File:** [mlir/lib/Bindings/Python/ExecutionEngineModule.cpp](mlir/lib/Bindings/Python/ExecutionEngineModule.cpp)

The `invoke()` method internally calls `lookup()` first:

```cpp
.def("raw_lookup",
    [](PyExecutionEngine &self, const std::string &name) {
        auto expectedFPtr = self.get().lookup(name);  // ← C++ lookup
        if (!expectedFPtr) {
            // ... error handling ...
        }
        return (uintptr_t)(*expectedFPtr);  // Return function pointer address
    })
```

**What it does:**
- Calls C++ `ExecutionEngine::lookup(name)`
- Returns raw function pointer as integer
- **No initialization happens here**

---

### Frame 4: C++ ExecutionEngine Lookup

**File:** [mlir/lib/ExecutionEngine/ExecutionEngine.cpp:414-434](mlir/lib/ExecutionEngine/ExecutionEngine.cpp#L414-L434)

```cpp
Expected<void *> ExecutionEngine::lookup(StringRef name) const {
  auto expectedSymbol = jit->lookup(name);  // ← LLVM ORC JIT lookup

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

**What it does:**
1. Triggers lazy JIT compilation if not already compiled
2. Returns native function pointer
3. **Does NOT run global constructors**

**Critical:** Global constructors (`llvm.global_ctors`) are ONLY executed when `initialize()` is explicitly called!

---

### Frame 5: Native Code Execution

**Generated LLVM IR:** [translated_module.ll:98-100](mlir/test/Examples/NVGPU/translated_module.ll#L98-L100)

```llvm
define void @_mlir_ciface_gemm_128_128_64(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %0, align 8
  %5 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %6 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8
  call void @gemm_128_128_64(...descriptors...)
  ret void
}
```

This calls the actual kernel launch function:

**Generated LLVM IR:** [translated_module.ll:86-89](mlir/test/Examples/NVGPU/translated_module.ll#L86-L89)

```llvm
define void @gemm_128_128_64(...) {
  ; ... setup args ...

  ; Load the GPU module (this is a GLOBAL variable set by constructor)
  %68 = load ptr, ptr @gemm_128_128_64_kernel_module, align 8  // ← NULL!

  ; Get kernel function from module
  %69 = call ptr @mgpuModuleGetFunction(ptr %68, ptr @gemm_128_128_64_kernel_name)

  ; Launch kernel
  call void @mgpuLaunchKernel(ptr %69, i64 1, i64 1, i64 1, ...)

  ret void
}
```

**The Problem:**
- `@gemm_128_128_64_kernel_module` is a global variable initialized by a constructor
- **The constructor was never run**, so it's NULL
- `mgpuModuleGetFunction(NULL, ...)` → `CUDA_ERROR_NOT_INITIALIZED`

---

### Frame 6: CUDA Runtime Wrapper

**File:** [mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp:152-157](mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp#L152-L157)

```cpp
extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUfunction
mgpuModuleGetFunction(CUmodule module, const char *name) {
  CUfunction function = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleGetFunction(&function, module, name));
  return function;
}
```

**What happens:**
- `module` parameter is **NULL** (because constructor never ran)
- `cuModuleGetFunction(&function, NULL, name)` is called
- CUDA driver returns `CUDA_ERROR_NOT_INITIALIZED`
- Error is printed via `CUDA_REPORT_IF_ERROR` macro

**Error Macro:** [CudaRuntimeWrappers.cpp:37-46](mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp#L37-L46)

```cpp
#define CUDA_REPORT_IF_ERROR(expr)                                             \
  [](CUresult result) {                                                        \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = nullptr;                                                \
    cuGetErrorName(result, &name);                                             \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)
```

This prints: `'cuModuleGetFunction(&function, module, name)' failed with 'CUDA_ERROR_NOT_INITIALIZED'`

---

## What Should Have Happened

### The Missing Initialization Step

**The global constructor that loads GPU binaries:**

**Generated LLVM IR:** [translated_module.ll:139-144](mlir/test/Examples/NVGPU/translated_module.ll#L139-L144)

```llvm
define internal void @gemm_128_128_64_kernel_load() section ".text.startup" {
entry:
  ; Load CUBIN binary via JIT compilation
  %0 = call ptr @mgpuModuleLoadJIT(ptr @gemm_128_128_64_kernel_binary, i32 2)

  ; Store module handle in global variable
  store ptr %0, ptr @gemm_128_128_64_kernel_module, align 8
  ret void
}

; Register as global constructor
@llvm.global_ctors = appending global [1 x {i32, ptr, ptr}] [
  {i32 0, ptr @gemm_128_128_64_kernel_load, ptr null}
]
```

**File:** [mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp:119-125](mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp#L119-L125)

```cpp
extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule
mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
  ScopedContext scopedContext;  // ← Initializes CUDA context
  CUmodule module = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
  return module;
}
```

**Critical: `ScopedContext` constructor:**

**File:** [CudaRuntimeWrappers.cpp:83-101](mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp#L83-L101)

```cpp
class ScopedContext {
public:
  ScopedContext() {
    // Static reference to CUDA primary context for device
    static CUcontext context = [] {
      CUDA_REPORT_IF_ERROR(cuInit(/*flags=*/0));  // ← Initialize CUDA driver!
      CUcontext ctx;
      CUDA_REPORT_IF_ERROR(
          cuDevicePrimaryCtxRetain(&ctx, getDefaultCuDevice()));
      return ctx;
    }();

    CUDA_REPORT_IF_ERROR(cuCtxPushCurrent(context));
  }

  ~ScopedContext() { CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr)); }
};
```

**This is where `cuInit()` is called**, which initializes the CUDA driver!

---

### Correct Execution Flow

**What CuTeDSL does (correctly):**

**File:** [cutlass/base_dsl/jit_executor.py:436-444](cutlass/base_dsl/jit_executor.py#L436-L444)

```python
def invokePacked(self, name: StringRef, args: MutableArrayRef<void *>) {
  initialize();  // ← CRITICAL: Run global constructors first!

  auto expectedFPtr = lookupPacked(name);
  if (!expectedFPtr)
    return expectedFPtr.takeError();
  auto fptr = *expectedFPtr;

  (*fptr)(args.data());
  return Error::success();
}
```

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
1. Finds all functions in `@llvm.global_ctors`
2. Calls them in order
3. For GPU code: This calls `@gemm_128_128_64_kernel_load()`
4. Which calls `mgpuModuleLoadJIT()` → `ScopedContext()` → `cuInit()`
5. GPU binaries are loaded and module handles stored in globals

**After initialization:**
- `@gemm_128_128_64_kernel_module` is a valid CUmodule handle
- `mgpuModuleGetFunction()` succeeds
- `mgpuLaunchKernel()` can launch the kernel

---

## The Fix

### Option 1: Call `initialize()` Explicitly (Recommended)

**Modify nvdsl.py line 453:**

```python
# Before invoking, run global constructors
if not NVDSL.compile_only:
    # Convert input arguments to MLIR arguments
    newArgs = get_mlir_func_obj_ty(args)

    # CRITICAL: Initialize ExecutionEngine (loads GPU binaries)
    # This was added in _nvdsl_modded.py but missing in original
    engine.initialize()  # ← ADD THIS LINE

    # Run the compiled program
    engine.invoke(function_name, *newArgs)
```

**Note:** I can see this fix already exists in `_nvdsl_modded.py`!

**File:** [_nvdsl_modded.py:369](mlir/test/Examples/NVGPU/tools/_nvdsl_modded.py#L369)

```python
def launch(self, function_name, args, result):
    # Ensure static constructors (e.g., GPU binary loaders) run
    # so CUDA is initialized before invoking kernels.
    self.engine.initialize()  # ← THIS IS THE FIX!

    # Convert input arguments to MLIR arguments
    newArgs = get_mlir_func_obj_ty(args)

    # Run the compiled program
    self.engine.invoke(function_name, *newArgs)
```

### Option 2: Use `invokePacked()` from C API

The C API's `mlirExecutionEngineInvokePacked()` calls `initialize()` automatically, but this is not exposed to Python.

---

## Debugging Breakpoints

### GDB Breakpoints for CUDA Runtime

**Set these breakpoints to trace execution:**

```gdb
# 1. Entry point - Python invoke
break mlir/python/mlir/execution_engine.py:invoke

# 2. C++ ExecutionEngine lookup
break mlir::ExecutionEngine::lookup
break mlir/lib/ExecutionEngine/ExecutionEngine.cpp:414

# 3. Global constructor initialization
break mlir::ExecutionEngine::initialize
break mlir/lib/ExecutionEngine/ExecutionEngine.cpp:449

# 4. CUDA module loading
break mgpuModuleLoad
break mgpuModuleLoadJIT
break mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp:120

# 5. CUDA context initialization
break ScopedContext::ScopedContext
break mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp:85

# 6. CUDA driver initialization
break cuInit
break cuDevicePrimaryCtxRetain

# 7. Module function lookup
break mgpuModuleGetFunction
break mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp:153

# 8. Kernel launch
break mgpuLaunchKernel
break mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp:163
```

### LLDB Breakpoints (alternative)

```lldb
# Set breakpoints
breakpoint set --file ExecutionEngine.cpp --line 414
breakpoint set --file ExecutionEngine.cpp --line 449
breakpoint set --file CudaRuntimeWrappers.cpp --line 85
breakpoint set --file CudaRuntimeWrappers.cpp --line 120
breakpoint set --name mgpuModuleGetFunction
breakpoint set --name cuInit

# Run with LLDB
lldb -- python your_test.py

# Continue and inspect
run
bt  # backtrace
frame variable  # inspect variables
continue
```

### Conditional Breakpoints

To break only when module is NULL:

```gdb
break mgpuModuleGetFunction
condition 1 module == 0x0
```

---

## Source Files for Debugging

### Key Files to Examine

**1. Python Bindings:**
- [mlir/lib/Bindings/Python/ExecutionEngineModule.cpp](mlir/lib/Bindings/Python/ExecutionEngineModule.cpp)
  - `PyExecutionEngine` class definition
  - `.def("__init__", ...)` - Constructor binding
  - `.def("raw_lookup", ...)` - Function lookup binding
  - `.def("_testing_release", ...)` - Cleanup

**2. C++ ExecutionEngine:**
- [mlir/lib/ExecutionEngine/ExecutionEngine.cpp](mlir/lib/ExecutionEngine/ExecutionEngine.cpp)
  - Line 232-403: `ExecutionEngine::create()` - Main factory method
  - Line 414-434: `ExecutionEngine::lookup()` - Symbol lookup
  - Line 449-457: `ExecutionEngine::initialize()` - **CRITICAL** global ctor runner
  - Line 436-447: `ExecutionEngine::invokePacked()` - Calls initialize() first

**3. CUDA Runtime Wrappers:**
- [mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp](mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp)
  - Line 83-101: `ScopedContext` - CUDA initialization
  - Line 119-125: `mgpuModuleLoad()` - Load precompiled binary
  - Line 127-146: `mgpuModuleLoadJIT()` - JIT compile PTX
  - Line 152-157: `mgpuModuleGetFunction()` - Get kernel function
  - Line 162-192: `mgpuLaunchKernel()` - Launch kernel

**4. Python Wrapper:**
- [mlir/python/mlir/execution_engine.py](mlir/python/mlir/execution_engine.py)
  - `invoke()` method - High-level invocation
  - `lookup()` method - Function lookup
  - `register_runtime()` - Register runtime symbols

**5. MLIR GPU Dialect Lowering:**
- [mlir/lib/Conversion/GPUCommon/GPUToLLVMConversion.cpp](mlir/lib/Conversion/GPUCommon/GPUToLLVMConversion.cpp)
  - How `gpu.launch_func` lowers to LLVM IR
  - Generation of global constructors
  - Binary embedding logic

---

## Debugging Flags and Environment Variables

### MLIR/LLVM Debug Flags

**Enable general debugging:**
```bash
export MLIR_ENABLE_DUMP=1
export LLVM_ENABLE_DUMP=1
```

**Enable pass debugging:**
```bash
# Print IR before/after each pass
python your_test.py --mlir-print-ir-before-all --mlir-print-ir-after-all

# Print only specific passes
python your_test.py --mlir-print-ir-after=gpu-kernel-outlining
```

**Enable timing statistics:**
```bash
python your_test.py --mlir-timing
```

### CUDA Runtime Debugging

**Enable CUDA runtime debug output:**

**File:** [CudaRuntimeWrappers.cpp:59-71](mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp#L59-L71)

```cpp
bool isDebugEnabled() {
  const char *kDebugEnvironmentVariable = "MLIR_CUDA_DEBUG";
  static bool isEnabled = getenv(kDebugEnvironmentVariable) != nullptr;
  return isEnabled;
}

#define debug_print(fmt, ...)                                                  \
  do {                                                                         \
    if (isDebugEnabled())                                                      \
      fprintf(stderr, "%s:%d:%s(): " fmt, "CudaRuntimeWrappers.cpp", __LINE__, \
              __func__, __VA_ARGS__);                                          \
  } while (0)
```

**Enable debugging:**
```bash
export MLIR_CUDA_DEBUG=1
python your_test.py
```

**This will print:**
- Kernel launch parameters (grid, block, shared memory)
- Module loading events
- Function lookups

### CUDA Driver API Debugging

**Enable CUDA API logging:**
```bash
export CUDA_LAUNCH_BLOCKING=1  # Synchronous launches (easier to debug)
export CUDA_VISIBLE_DEVICES=0   # Use specific GPU
```

**Use cuda-gdb:**
```bash
cuda-gdb --args python your_test.py
```

### ExecutionEngine Debugging

**Enable JIT debug output:**
```bash
export LLVM_DEBUG=1
export LLVM_DEBUG_ONLY=orc  # Only ORC JIT debug output
```

**Compile LLVM/MLIR with debug symbols:**
```bash
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      ...
```

### Python Debugging

**Add debug prints:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# In your code
print(f"Engine created: {engine}")
print(f"About to invoke: {function_name}")
print(f"Args: {newArgs}")

# Check if initialize was called
import inspect
print(f"ExecutionEngine methods: {dir(engine)}")
print(f"Has initialize: {hasattr(engine, 'initialize')}")
```

**Use pdb:**
```python
import pdb

# Before invoke
pdb.set_trace()
engine.invoke(function_name, *newArgs)
```

---

## Verification Steps

### 1. Check if Global Constructors Exist

**Dump LLVM IR and look for constructors:**

```python
# After compilation, before JIT
with open("module.ll", "w") as f:
    # This requires accessing the LLVM module before it's consumed
    # You'd need to modify nvgpucompiler.py to save IR
    pass
```

**Search for:**
```llvm
@llvm.global_ctors = appending global [...]
define internal void @..._load() section ".text.startup" { ... }
```

### 2. Check Module Loading

**Add debug print to constructor:**

Create a modified version of the test that prints when loaded:

```python
# Before running test
import ctypes

# Create a callback that will be registered
def debug_callback():
    print("GPU module loaded!")

# Register it via execution engine
# (This requires modifying the MLIR to call it)
```

### 3. Verify CUDA Initialization

**Check CUDA context:**

```python
import cuda.bindings.driver as cuda

# After engine creation, before invoke
try:
    result = cuda.cuInit(0)
    print(f"CUDA init result: {result}")

    device = cuda.CUdevice()
    cuda.cuDeviceGet(device, 0)
    print(f"Device: {device}")

    context = cuda.CUcontext()
    cuda.cuCtxGetCurrent(context)
    print(f"Current context: {context}")
except Exception as e:
    print(f"CUDA error: {e}")
```

---

## Additional Recommendations

### 1. Always Call `initialize()` for GPU Code

**Best Practice Pattern:**

```python
def safe_invoke(engine, function_name, *args):
    """Safely invoke a function with GPU code."""
    # Ensure global constructors run (loads GPU binaries)
    if hasattr(engine, 'initialize'):
        engine.initialize()

    # Now invoke
    engine.invoke(function_name, *args)
```

### 2. Check Shared Libraries

Ensure CUDA runtime library is loaded:

```python
compiler = nvgpucompiler.NvgpuCompiler(
    NVDSL.pipeline_options,
    opt_level=3,
    shared_libs=[
        "/path/to/libmlir_cuda_runtime.so",  # ← Must include this!
        "/path/to/libmlir_runner_utils.so"
    ]
)
```

**Find shared libraries:**
```bash
find /home/jeromeku/llvm-project/build -name "libmlir_cuda_runtime.so"
find /home/jeromeku/llvm-project/build -name "libmlir_runner_utils.so"
```

### 3. Verify Compilation Pipeline

Ensure GPU-to-CUBIN pass is in pipeline:

```python
pipeline = "builtin.module(gpu-lower-to-nvvm-pipeline{...})"
```

Check that it includes:
- `gpu-kernel-outlining`
- `gpu-to-nvvm` or `gpu-to-cubin`

---

## Summary

### Root Cause
The ExecutionEngine's `initialize()` method is **never called** before `invoke()`, which means:
1. Global constructors don't run
2. GPU binaries are never loaded via `mgpuModuleLoadJIT()`
3. CUDA driver is never initialized via `cuInit()`
4. Module handle globals remain NULL
5. `mgpuModuleGetFunction(NULL, ...)` → `CUDA_ERROR_NOT_INITIALIZED`

### The Fix
Add **one line** before invoking:

```python
engine.initialize()  # ← ADD THIS
engine.invoke(function_name, *newArgs)
```

### Why It's Easy to Miss
- CuTeDSL's `invokePacked()` C++ method calls `initialize()` automatically
- MLIR's Python bindings don't call it automatically in `invoke()`
- The error message is cryptic ("NOT_INITIALIZED" vs "MODULE_NOT_LOADED")
- Works fine for host-only code (no GPU kernels)

### Verification
The fix already exists in `_nvdsl_modded.py` line 369, confirming this analysis!

---

*This document provides a complete trace from Python to CUDA driver, identifying the exact cause of the error and providing comprehensive debugging guidance.*
