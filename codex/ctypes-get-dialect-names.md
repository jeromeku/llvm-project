# Calling mlir::DialectRegistry::getDialectNames() from Python via ctypes

The MLIR Python bindings expose an `ir.DialectRegistry`, but its stub/typeshed does not surface C++ methods like `getDialectNames()`. Since `ctypes` can only call C‑ABI symbols, you need a tiny C shim that wraps the C++ API and returns C‑friendly data.

Below is a minimal, non‑intrusive approach: build a small shared library against your existing build that exports a C function to collect dialect names, and call it from Python with `ctypes`.

---

## 1) C/C++ shim that exposes a C ABI

File: `codex/dialect_names_shim.cpp`

```cpp
// Build against your existing MLIR build tree.
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mlir/IR/DialectRegistry.h"               // DialectRegistry
#include "mlir/InitAllDialects.h"                  // registerAllDialects

extern "C" {

// Collect dialect names from a fresh registry with all built‑in dialects.
// Returns 0 on success. On success, `*outNames` points to an array of `count`
// C strings. Caller must free via `mlir_free_dialect_names`.
int mlir_collect_all_dialect_names(char ***, size_t *);

// Variant that accepts an existing registry C handle.
int mlir_collect_dialect_names_from_registry(MlirDialectRegistry,
                                             char ***, size_t *);

// Free the array returned by the above
void mlir_free_dialect_names(char **, size_t);

}
```

Notes
- This directly calls the C++ method [`mlir/include/mlir/IR/DialectRegistry.h:204-210`](mlir/include/mlir/IR/DialectRegistry.h#L204-L210) via `registry.getDialectNames()`.
- It avoids the C API entirely and does not modify MLIR sources.

---

## 2) Build the shim against your build tree

Use CMake so it picks the right headers/libs from your local build.

File: `codex/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.20)
project(mlir_dialect_names_shim LANGUAGES CXX)

# Point MLIR_DIR to your build’s CMake config dir.
# Example: -DMLIR_DIR=/home/jeromeku/llvm-project/build/lib/cmake/mlir
find_package(MLIR REQUIRED CONFIG)

add_library(mlir_dialect_names_shim SHARED dialect_names_shim.cpp)

target_include_directories(mlir_dialect_names_shim PRIVATE
  ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(mlir_dialect_names_shim PRIVATE ${MLIR_DEFINITIONS})
target_compile_features(mlir_dialect_names_shim PRIVATE cxx_std_17)

# Link the pieces needed for DialectRegistry + registerAllDialects.
target_link_libraries(mlir_dialect_names_shim PRIVATE
  MLIRIR               # core IR
  MLIRRegisterAllDialects
)

# Ensure default symbol visibility for the extern "C" functions.
set_target_properties(mlir_dialect_names_shim PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN NO
)
```

Build

```bash
cd /home/jeromeku/llvm-project/codex
cmake -S . -B build-shim -DMLIR_DIR=/home/jeromeku/llvm-project/build/lib/cmake/mlir
cmake --build build-shim -j

# Result: build-shim/libmlir_dialect_names_shim.so
```

Tip: If you prefer a one‑liner without CMake, you can reproduce what `find_package(MLIR)` sets up by using include/link dirs from your build, but CMake is less error‑prone.

---

## 3) Call from Python with ctypes

```python
import ctypes, os

# Paths
LLVM_ROOT = "/home/jeromeku/llvm-project"
shim_path = os.path.join(LLVM_ROOT, "codex", "build-shim", "libmlir_dialect_names_shim.so")

# Load shim
shim = ctypes.CDLL(shim_path)

# Prototypes
CharPP = ctypes.POINTER(ctypes.c_char_p)
shim.mlir_collect_all_dialect_names.argtypes = [ctypes.POINTER(CharPP), ctypes.POINTER(ctypes.c_size_t)]
shim.mlir_collect_all_dialect_names.restype = ctypes.c_int
shim.mlir_free_dialect_names.argtypes = [CharPP, ctypes.c_size_t]
shim.mlir_free_dialect_names.restype = None

# Call
names_ptr = CharPP()
count = ctypes.c_size_t()
err = shim.mlir_collect_all_dialect_names(ctypes.byref(names_ptr), ctypes.byref(count))
if err != 0:
    raise RuntimeError(f"collect failed: {err}")

try:
    names = [names_ptr[i].decode("utf-8") for i in range(count.value)]
finally:
    shim.mlir_free_dialect_names(names_ptr, count)

print(f"{len(names)} dialects:")
for n in sorted(names):
    print(" ", n)
```

This yields the same names you’d get from `DialectRegistry::getDialectNames()` in C++.

---

### Using an existing MlirDialectRegistry (from a Python capsule)

If you have an `ir.DialectRegistry` Python object, you can obtain its C API handle from the capsule property `_CAPIPtr`, extract the raw pointer using Python’s C-API via `ctypes`, wrap it as a `MlirDialectRegistry` struct, and pass it to the registry-accepting shim function:

```python
from mlir.ir import DialectRegistry
import ctypes

reg = DialectRegistry()

# 1) Extract the raw pointer from the capsule using the known capsule name
#    from mlir-c/Bindings/Python/Interop.h
capsule = reg._CAPIPtr
capsule_name = b"mlir.ir.DialectRegistry._CAPIPtr"
pycapsule_getptr = ctypes.pythonapi.PyCapsule_GetPointer
pycapsule_getptr.restype = ctypes.c_void_p
pycapsule_getptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
ptr = pycapsule_getptr(capsule, capsule_name)

# 2) Mirror the C struct: typedef struct { void *ptr; } MlirDialectRegistry;
class MlirDialectRegistry(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p)]

mlir_reg = MlirDialectRegistry(ptr)

# 3) Call the shim that consumes an existing registry
CharPP = ctypes.POINTER(ctypes.c_char_p)
shim.mlir_collect_dialect_names_from_registry.argtypes = [MlirDialectRegistry,
                                                          ctypes.POINTER(CharPP),
                                                          ctypes.POINTER(ctypes.c_size_t)]
shim.mlir_collect_dialect_names_from_registry.restype = ctypes.c_int

names_ptr = CharPP()
count = ctypes.c_size_t()
err = shim.mlir_collect_dialect_names_from_registry(mlir_reg,
                                                    ctypes.byref(names_ptr),
                                                    ctypes.byref(count))
if err != 0:
    raise RuntimeError(f"collect failed: {err}")
try:
    names = [names_ptr[i].decode("utf-8") for i in range(count.value)]
finally:
    shim.mlir_free_dialect_names(names_ptr, count)

print(sorted(names)[:10])
```

This approach lets you enumerate dialects from any registry object you already have (including registries populated by your own code), instead of creating a new one inside the shim.

---

## Why a shim is needed (and alternatives)

- `ctypes` only calls C‑ABI functions. C++ member functions like `DialectRegistry::getDialectNames()` use C++ name mangling and calling conventions. Even if you resolved the mangled symbol, the return type is a templated range of `StringRef`, which is not a C type.
- The shim converts to an array of C strings and exports a stable C ABI, which `ctypes` can consume.

Alternatives
- If you specifically need to operate on an existing `MlirDialectRegistry` (C API handle), use the provided `mlir_collect_dialect_names_from_registry` and pass a handle extracted from the Python capsule as shown above. Under the hood, it uses `unwrap(registry)` to call `getDialectNames()` (see related C-API usage in [`mlir/lib/CAPI/IR/IR.cpp:145-151`](mlir/lib/CAPI/IR/IR.cpp#L145-L151)).
- If you just need names available in a context, MLIR’s C API exposes counts and dialect handles on the context; however, it does not currently expose a direct “list all registered dialect names” function. The above shim is the most direct path for enumeration.
