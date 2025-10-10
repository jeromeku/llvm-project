# MLIR Python Bindings Architecture

## Purpose

This document describes the architecture of MLIR's Python bindings, providing a map for understanding how Python code interacts with MLIR's C++ infrastructure. It serves as a guide for developers who need to understand, debug, or extend the bindings.

## High-Level Architecture

MLIR's Python bindings follow a three-layer architecture:

```
┌─────────────────────────────────┐
│     Python User Code            │
│  (import mlir.ir, dialects...)  │
└─────────────┬───────────────────┘
              │
┌─────────────▼───────────────────┐
│    Python Binding Layer         │
│  (pybind11/nanobind modules)    │
│  lib/Bindings/Python/*.cpp      │
└─────────────┬───────────────────┘
              │
┌─────────────▼───────────────────┐
│        C API Layer              │
│   (mlir-c/*.h interfaces)       │
│    Stable C interface           │
└─────────────┬───────────────────┘
              │
┌─────────────▼───────────────────┐
│     MLIR C++ Core               │
│   (mlir/IR/*, Dialects/*)       │
│    Actual implementation        │
└─────────────────────────────────┘
```

## Key Design Principles

1. **C API as Stability Layer**: The C API provides a stable ABI boundary, avoiding C++ ABI compatibility issues
2. **Composable Modules**: Python bindings are organized into composable modules for downstream integration
3. **Ownership Model**: Python objects maintain ownership of underlying MLIR objects through reference counting
4. **Type Safety**: Automatic conversions between Python types and MLIR C API handles

## Core Components

### 1. C++ Implementation Layer

The foundation - MLIR's C++ classes that implement all functionality:

**Location**: `mlir/include/mlir/IR/`, `mlir/lib/IR/`

**Key Classes**:
- `mlir::MLIRContext` - The context owning all IR objects
- `mlir::Operation` - Base class for all operations
- `mlir::Type`, `mlir::Attribute` - Type system and metadata
- `mlir::Module`, `mlir::Block`, `mlir::Region` - IR structure

### 2. C API Layer

Provides stable C interfaces to C++ functionality:

**Location**: `mlir/include/mlir-c/`

**Key Headers**:
```c
// mlir-c/IR.h - Core IR interfaces
MlirContext mlirContextCreate();
MlirModule mlirModuleCreateEmpty(MlirLocation location);
MlirOperation mlirOperationCreate(MlirOperationState *state);

// mlir-c/BuiltinTypes.h - Type system
MlirType mlirIntegerTypeGet(MlirContext ctx, unsigned bitwidth);
MlirType mlirF32TypeGet(MlirContext ctx);
```

**Wrapping Pattern**: C++ objects are wrapped/unwrapped:
```c++
// In mlir/lib/CAPI/IR/IR.cpp
MlirContext mlirContextCreate() {
  return wrap(new MLIRContext());  // wrap() converts C++ to C handle
}

MLIRContext *unwrap(MlirContext c) {
  return reinterpret_cast<MLIRContext *>(c.ptr);
}
```

### 3. Python Binding Layer

Uses pybind11/nanobind to expose C API to Python:

**Location**: `mlir/lib/Bindings/Python/`

**Key Files**:
- `IRCore.cpp` - Core IR classes (Context, Module, Operation)
- `IRTypes.cpp` - Type system bindings
- `IRAttributes.cpp` - Attribute bindings
- `PybindAdaptors.h` - Type conversion utilities

### 4. Python Wrapper Classes

Pure Python layer providing high-level API:

**Location**: `mlir/python/mlir/`

**Structure**:
```
mlir/
├── ir.py           # Core IR classes
├── dialects/       # Dialect-specific bindings
│   ├── arith.py
│   ├── func.py
│   └── ...
└── _mlir_libs/     # Native extension modules
    └── _mlir.so    # Compiled C++ bindings
```

## Binding Implementation Details

### Type Conversions

MLIR uses pybind11 adaptors to automatically convert between Python objects and C API handles. The conversion happens through type casters defined in PybindAdaptors.h.

**Example from `PybindAdaptors.h`**:
```c++
namespace pybind11::detail {
  // Type caster for MlirAttribute
  template <>
  struct type_caster<MlirAttribute> {
    PYBIND11_TYPE_CASTER(MlirAttribute, _("MlirAttribute"));
    
    bool load(handle src, bool) {
      py::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToAttribute(capsule.ptr());
      return !mlirAttributeIsNull(value);
    }
    
    static handle cast(MlirAttribute v, return_value_policy, handle) {
      py::object capsule = py::reinterpret_steal<py::object>(
        mlirPythonAttributeToCapsule(v));
      return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Attribute")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
    }
  };
}
```

### Python Class Binding Example

From `IRCore.cpp`, here's how `PyOperation` is exposed:

```c++
// In IRCore.cpp
void populateIRCore(py::module &m) {
  // Define Operation class
  py::class_<PyOperation>(m, "Operation")
    .def_static("create", &PyOperation::create,
                py::arg("name"), py::arg("results") = py::none(),
                py::arg("operands") = py::none(),
                py::arg("attributes") = py::none(),
                py::arg("successors") = py::none(),
                py::arg("regions") = 0,
                py::arg("loc") = py::none(),
                py::arg("ip") = py::none(),
                kOperationCreateDocstring)
    .def("clone", &PyOperation::clone)
    .def_property_readonly("name", 
        [](PyOperation &self) {
          MlirOperation op = self.get();
          MlirStringRef name = mlirOperationGetName(op);
          return py::str(name.data, name.length);
        })
    .def_property_readonly("operands",
        [](PyOperation &self) {
          return PyOpOperandList(self.getRef());
        })
    .def("__str__", [](PyOperation &self) {
      return self.print(/*binary=*/false);
    });
}
```

## MLIR Concepts Primer

### Operations

Operations are the core unit of abstraction in MLIR. An operation has a name (like "toy.transpose"), operands (input values), results (output values), attributes (compile-time constants), and nested regions containing blocks of other operations.

**Example MLIR Operation**:
```mlir
%result = "dialect.operation"(%operand) {attribute = 42} : (f32) -> f32
```

### Dialects

Dialects provide a grouping mechanism for operations, types, and attributes under a unique namespace. MLIR is completely extensible - there is no closed set of operations or types.

### Types and Attributes

- **Types**: Describe the data type of values (e.g., `tensor<2x3xf64>`, `i32`)
- **Attributes**: Compile-time constant metadata (e.g., integer values, strings)

### IR Structure Hierarchy

```
Module
└── Region (implicitly created)
    └── Block
        └── Operation
            ├── Operands (Values)
            ├── Results (Values)
            ├── Attributes
            └── Regions (nested)
                └── Blocks...
```

## Tracing Python to C++

### Method 1: Using Python Introspection

```python
import mlir.ir as ir

# Check what C functions are called
ctx = ir.Context()
print(type(ctx))  # <class 'mlir.ir.Context'>
print(ctx.__class__.__module__)  # mlir._mlir_libs._mlir.ir

# The _mlir.so module contains the actual bindings
import mlir._mlir_libs._mlir as _mlir
print(dir(_mlir.ir.Context))  # Shows available methods
```

### Method 2: Following Source Code

1. **Python call**: `module = ir.Module.parse(mlir_text)`

2. **Binding layer** (`IRCore.cpp`):
```c++
.def_static("parse", [](const std::string &moduleAsm) {
  MlirModule module = mlirModuleCreateParse(
    mlirContextGetGlobalRegistry(), 
    mlirStringRefCreate(moduleAsm.data(), moduleAsm.size()));
  // ... error handling ...
  return PyModule::forModule(module);
})
```

3. **C API** (`mlir-c/IR.h`):
```c
MlirModule mlirModuleCreateParse(MlirContext context, 
                                  MlirStringRef module);
```

4. **C++ Implementation** (`mlir/lib/CAPI/IR/IR.cpp`):
```c++
MlirModule mlirModuleCreateParse(MlirContext context, 
                                  MlirStringRef module) {
  OwningOpRef<ModuleOp> owning = parseSourceString<ModuleOp>(
    unwrap(module), unwrap(context));
  return wrap(owning.release().getOperation());
}
```

### Method 3: Using Debugger

```bash
# Run Python under gdb
gdb python3
(gdb) run -c "import mlir.ir; ctx = mlir.ir.Context()"
(gdb) break mlirContextCreate
(gdb) continue
```

## Complete Examples

### Example 1: Creating IR in Python

```python
import mlir.ir as ir
import mlir.dialects.func as func
import mlir.dialects.arith as arith

# Create context and module
with ir.Context() as ctx, ir.Location.unknown():
    module = ir.Module.create()
    
    # Create a function
    with ir.InsertionPoint(module.body):
        f32 = ir.F32Type.get()
        
        # func.func @add(%arg0: f32, %arg1: f32) -> f32
        @func.FuncOp.from_py_func(f32, f32)
        def add(arg0, arg1):
            result = arith.AddFOp(arg0, arg1)
            func.ReturnOp([result])
    
    print(module)
```

### Example 2: Equivalent in C++

```c++
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

void createModule() {
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect, 
                        mlir::arith::ArithDialect>();
    
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(
        builder.getUnknownLoc());
    
    // Create function type: (f32, f32) -> f32
    auto f32 = builder.getF32Type();
    auto funcType = builder.getFunctionType({f32, f32}, {f32});
    
    // Create function
    builder.setInsertionPointToEnd(module.getBody());
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "add", funcType);
    
    // Create entry block with arguments
    auto *entry = func.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    
    // Create add operation
    auto add = builder.create<mlir::arith::AddFOp>(
        builder.getUnknownLoc(),
        entry->getArgument(0),
        entry->getArgument(1));
    
    // Create return
    builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), add.getResult());
    
    module.dump();
}
```

### Example 3: Using C API Directly

```c
#include "mlir-c/IR.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Dialect/Func.h"

void createModuleWithCAPI() {
    MlirContext ctx = mlirContextCreate();
    mlirDialectHandleRegisterDialect(
        mlirGetDialectHandle__func__(), ctx);
    
    MlirLocation loc = mlirLocationUnknownGet(ctx);
    MlirModule module = mlirModuleCreateEmpty(loc);
    
    // Create function type
    MlirType f32 = mlirF32TypeGet(ctx);
    MlirType inputs[] = {f32, f32};
    MlirType outputs[] = {f32};
    MlirType funcType = mlirFunctionTypeGet(ctx, 2, inputs, 
                                             1, outputs);
    
    // Would continue with operation creation...
    // (C API is more verbose than C++)
    
    mlirModuleDestroy(module);
    mlirContextDestroy(ctx);
}
```

## Custom Dialect Bindings

To expose a custom dialect to Python, you need to: provide C API functions for the dialect, create pybind11/nanobind bindings that wrap the C API, and register Python classes for operations and types.

### Step 1: Define C API

```c
// In include/mlir-c/Dialect/MyDialect.h
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MyDialect, my_dialect);

MlirType myDialectCustomTypeGet(MlirContext ctx, intptr_t param);
```

### Step 2: Create Python Bindings

```c++
// In lib/Bindings/Python/MyDialectModule.cpp
#include "mlir-c/Dialect/MyDialect.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

PYBIND11_MODULE(_myDialect, m) {
  m.doc() = "My custom dialect bindings";
  
  m.def("register_dialect", [](MlirContext context) {
    mlirDialectHandleRegisterDialect(
      mlirGetDialectHandle__my_dialect__(), context);
  });
  
  // Bind custom type
  m.def("CustomType", [](MlirContext ctx, int param) {
    return myDialectCustomTypeGet(ctx, param);
  });
}
```

### Step 3: Python Wrapper

```python
# In python/mlir/dialects/my_dialect.py
from ._my_dialect import *
from .._mlir_libs import _myDialect

def register_dialect(ctx):
    _myDialect.register_dialect(ctx)

class CustomType:
    @staticmethod
    def get(param, context=None):
        return _myDialect.CustomType(context, param)
```

## Building and Debugging

### Build Configuration

```cmake
# Enable Python bindings in CMake
cmake -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DPython3_EXECUTABLE=$(which python3) \
      ...
```

### Finding Symbol Definitions

```bash
# Find C API implementation
grep -r "mlirOperationCreate" mlir/lib/CAPI/

# Find Python binding
grep -r "PyOperation::create" mlir/lib/Bindings/Python/

# Check exported symbols
nm lib/libMLIRCAPI.so | grep mlirOperation
```

### Common Issues and Solutions

1. **ImportError**: Check that `_mlir_libs` is in Python path
2. **Missing dialect**: Ensure dialect is registered with context
3. **Type errors**: Verify C API handle conversions in PybindAdaptors
4. **Segfaults**: Often due to lifetime issues - Python objects may outlive C++ objects

## Current Evolution

MLIR is transitioning from pybind11 to nanobind for improved compile times and performance. The binding architecture remains similar but with updated syntax and better efficiency.

## References

- [MLIR Python Bindings Documentation](https://mlir.llvm.org/docs/Bindings/Python/)
- [MLIR C API Headers](https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir-c)
- [Binding Implementation](https://github.com/llvm/llvm-project/tree/main/mlir/lib/Bindings/Python)
- [Python API Source](https://github.com/llvm/llvm-project/tree/main/mlir/python/mlir)