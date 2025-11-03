# MLIR Interfaces Tutorial

A comprehensive guide to understanding, implementing, and using MLIR interfaces.

---

## Table of Contents

1. [What Are MLIR Interfaces?](#what-are-mlir-interfaces)
2. [Types of Interfaces](#types-of-interfaces)
3. [Defining Interfaces in TableGen](#defining-interfaces-in-tablegen)
4. [Implementation Strategies](#implementation-strategies)
5. [Complete Examples](#complete-examples)
6. [Common Patterns](#common-patterns)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## What Are MLIR Interfaces?

**Interfaces** in MLIR provide a way to define **polymorphic behavior** for operations, types, and attributes without using traditional C++ virtual functions. They enable:

- ✅ **Duck typing**: Operations/types/attributes can implement behaviors without inheritance
- ✅ **Non-intrusive extension**: Add interfaces to existing types without modifying them
- ✅ **Zero-cost abstraction**: Uses static polymorphism (CRTP) instead of virtual functions
- ✅ **Compile-time type safety**: Interface conformance checked at compile time

### Key Concept: Separation of Interface and Implementation

```
┌─────────────────────────┐
│   Interface Definition  │ ← What methods must be implemented
│   (TableGen .td file)   │
└───────────┬─────────────┘
            │
            ├─────────────────────────────────────┐
            │                                     │
            ↓                                     ↓
┌───────────────────────┐         ┌────────────────────────────┐
│ Native Implementation │         │  External Implementation   │
│ (in the type itself)  │         │  (separate model class)    │
└───────────────────────┘         └────────────────────────────┘
```

---

## Types of Interfaces

MLIR provides three main interface types:

| Interface Type | Applies To | Base Class | Example |
|----------------|------------|------------|---------|
| **OpInterface** | Operations | `OpInterface<...>` | `CastOpInterface` |
| **AttrInterface** | Attributes | `AttrInterface<...>` | `TargetAttrInterface` |
| **TypeInterface** | Types | `TypeInterface<...>` | `ShapedTypeInterface` |

### Where Interfaces Are Used

```mlir
// Operation interface
%result = arith.addi %a, %b : i32
//        ^^^^^^^^^^^ CastOpInterface, ArithmeticFastMathInterface

// Type interface
tensor<4x4xf32>
^^^^^^ ShapedType implements ShapedTypeInterface

// Attribute interface
#nvvm.target<chip = "sm_80">
^^^^^^^^^^^ Implements TargetAttrInterface
```

---

## Defining Interfaces in TableGen

### Basic Structure

From [Interfaces.td:87-122](/home/jeromeku/llvm-project/mlir/include/mlir/IR/Interfaces.td#L87-L122):

```tablegen
class Interface<string name, list<Interface> baseInterfacesArg = []> {
  // Human-readable description
  string description = "";

  // C++ class name
  string cppInterfaceName = name;

  // C++ namespace
  string cppNamespace = "";

  // List of methods
  list<InterfaceMethod> methods = [];

  // Extra C++ code in interface class
  code extraClassDeclaration = "";

  // Extra C++ code in both interface and trait
  code extraSharedClassDeclaration = "";

  // Optional base interfaces (inheritance)
  list<Interface> baseInterfaces = baseInterfacesArg;
}
```

### Example 1: Simple OpInterface

From [CastInterfaces.td:18-48](/home/jeromeku/llvm-project/mlir/include/mlir/Interfaces/CastInterfaces.td#L18-L48):

```tablegen
def CastOpInterface : OpInterface<"CastOpInterface"> {
  let description = [{
    A cast-like operation converts from input types to output types.
    Cast operations are removable when they produce a No-op.
  }];

  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<[{
        Returns true if input and result types are compatible.
      }],
      "bool",                              // Return type
      "areCastCompatible",                 // Method name
      (ins "::mlir::TypeRange":$inputs,    // Arguments
           "::mlir::TypeRange":$outputs)
    >,
  ];

  let extraTraitClassDeclaration = [{
    /// Attempt to fold the given cast operation.
    static LogicalResult foldTrait(Operation *op,
                                   ArrayRef<Attribute> operands,
                                   SmallVectorImpl<OpFoldResult> &results) {
      return impl::foldCastInterfaceOp(op, operands, results);
    }
  }];

  let verify = [{
    return impl::verifyCastInterfaceOp($_op);
  }];
}
```

### Example 2: TypeInterface with Methods

From [TestInterfaces.td:39-68](/home/jeromeku/llvm-project/mlir/test/lib/Dialect/Test/TestInterfaces.td#L39-L68):

```tablegen
def TestTypeInterface : TypeInterface<"TestTypeInterface"> {
  let cppNamespace = "::test";

  let methods = [
    // Method without default implementation (must be implemented)
    InterfaceMethod<"Prints the type name.",
      "void",                              // Return type
      "printTypeC",                        // Method name
      (ins "::mlir::Location":$loc)        // Arguments
    >,

    // Method with default implementation (can be overridden)
    InterfaceMethod<"Prints the type name and returns the type.",
      "TestTypeInterface",                 // Can use interface as return type
      "printTypeRet",
      (ins "::mlir::Location":$loc),
      [{}],                                // Empty body (for override)
      /*defaultImplementation=*/[{
        emitRemark(loc) << $_type << " - TestRet";
        return $_type;                     // $_type refers to the implementing type
      }]
    >,
  ];

  // Extra methods in interface class (not in trait)
  let extraClassDeclaration = [{
    void printTypeD(::mlir::Location loc) const {
      emitRemark(loc) << *this << " - TestD";
    }
  }];

  // Extra methods in trait (available to implementing types)
  let extraTraitClassDeclaration = [{
    void printTypeE(::mlir::Location loc) const {
      emitRemark(loc) << $_type << " - TestE";
    }
  }];
}
```

### Example 3: AttrInterface for GPU Compilation

From [CompilationAttrInterfaces.td:22-56](/home/jeromeku/llvm-project/mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td#L22-L56):

```tablegen
def GPUTargetAttrInterface : AttrInterface<"TargetAttrInterface"> {
  let description = [{
    Interface for GPU target attributes that compile GPU modules into binaries.
  }];

  let cppNamespace = "::mlir::gpu";

  let methods = [
    InterfaceMethod<[{
        Serializes a GPU module to a binary object.
        Returns std::nullopt on failure.
      }],
      "std::optional<::mlir::SmallVector<char, 0>>",
      "serializeToObject",
      (ins "::mlir::Operation*":$module,
           "const ::mlir::gpu::TargetOptions&":$options)
    >,

    InterfaceMethod<[{
        Creates a GPU object attribute from a binary string.
      }],
      "::mlir::Attribute",
      "createObject",
      (ins "::mlir::Operation *":$module,
           "const ::llvm::SmallVector<char, 0> &":$object,
           "const ::mlir::gpu::TargetOptions &":$options)
    >
  ];
}
```

---

## Implementation Strategies

MLIR provides three ways to implement interfaces:

### Strategy 1: Native Implementation (Intrusive)

The type/op **inherently** implements the interface.

#### Step 1: Declare Interface in TableGen

```tablegen
// MyDialectOps.td
def MyCastOp : Op<MyDialect, "cast", [CastOpInterface]> {
  //                                  ^^^^^^^^^^^^^^^^
  //                                  Declare interface conformance
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);

  // Implement interface methods directly in the op
  let extraClassDeclaration = [{
    // Implementation of CastOpInterface::areCastCompatible
    static bool areCastCompatible(TypeRange inputs, TypeRange outputs);
  }];
}
```

#### Step 2: Implement in C++

```cpp
// MyDialectOps.cpp
bool MyCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  return inputs[0] != outputs[0];  // Allow casts between different types
}
```

#### Usage

```cpp
// User code
auto castOp = builder.create<MyCastOp>(loc, newType, value);

// Access via interface
if (auto castInterface = dyn_cast<CastOpInterface>(castOp.getOperation())) {
  if (CastOpInterface::areCastCompatible(inputs, outputs)) {
    // ...
  }
}
```

---

### Strategy 2: External Model (Non-Intrusive)

Implement an interface for a type **without modifying its definition**.

#### Step 1: Define External Model Class

```cpp
// In a separate compilation unit (e.g., Target.cpp)
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

namespace {
class NVVMTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<NVVMTargetAttrImpl> {
    //                                  ^^^^^^^^^^^^
    //                                  External model base (CRTP)
public:
  // Implement interface methods
  // Note: Takes Attribute as parameter (not 'this')
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute,
                    Operation *module,
                    const gpu::TargetOptions &options) const {
    auto target = cast<NVVMTargetAttr>(attribute);

    // Compile the GPU module to PTX/CUBIN
    // ... compilation logic ...

    return binary;
  }

  Attribute createObject(Attribute attribute,
                         Operation *module,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const {
    auto target = cast<NVVMTargetAttr>(attribute);
    return gpu::ObjectAttr::get(target, format, object, options);
  }
};
} // namespace
```

#### Step 2: Register the External Model

```cpp
void registerNVVMTargetInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, NVVMDialect *dialect) {
    // Attach the external model to NVVMTargetAttr
    NVVMTargetAttr::attachInterface<NVVMTargetAttrImpl>(*ctx);
  });
}

void registerNVVMTargetInterface(MLIRContext &context) {
  DialectRegistry registry;
  registerNVVMTargetInterface(registry);
  context.appendDialectRegistry(registry);
}
```

#### Step 3: Initialize in Your Application

```cpp
int main() {
  MLIRContext context;

  // Register dialects
  context.getOrLoadDialect<NVVMDialect>();

  // Register external interfaces
  registerNVVMTargetInterface(context);

  // Now NVVMTargetAttr implements TargetAttrInterface!
  auto target = NVVMTargetAttr::get(...);
  auto interface = cast<gpu::TargetAttrInterface>(target);
  auto binary = interface.serializeToObject(module, options);
}
```

---

### Strategy 3: ExternalModel (Convenience Variant)

A more explicit version of FallbackModel.

```cpp
class MyExternalModelImpl
    : public SomeInterface::ExternalModel<MyExternalModelImpl, ConcreteType> {
    //                      ^^^^^^^^^^^^^
    //                      Provides ConcreteEntity typedef
public:
  // Same as FallbackModel
  ReturnType methodName(Attribute attr, ...) const {
    // Implementation
  }
};

// Register the same way
ConcreteType::attachInterface<MyExternalModelImpl>(ctx);
```

---

## Complete Examples

### Example 1: Implementing a Simple OpInterface

Let's create a `ComputeOpInterface` for operations that perform computation.

#### Step 1: Define the Interface

```tablegen
// ComputeInterfaces.td
def ComputeOpInterface : OpInterface<"ComputeOpInterface"> {
  let description = [{
    Interface for operations that perform computation and can report
    their estimated cost.
  }];

  let cppNamespace = "::mydialect";

  let methods = [
    InterfaceMethod<[{
        Returns the estimated computational cost of this operation.
        Higher numbers indicate more expensive operations.
      }],
      "int64_t",           // Return type
      "getComputeCost",    // Method name
      (ins),               // No arguments
      /*methodBody=*/[{}], // No default body (must be implemented)
      /*defaultImplementation=*/[{
        // Default: estimate based on operand count
        return $_op.getNumOperands() + $_op.getNumResults();
      }]
    >,

    InterfaceMethod<[{
        Returns true if this operation is compute-intensive.
      }],
      "bool",
      "isComputeIntensive",
      (ins),
      [{}],
      /*defaultImplementation=*/[{
        return $_op.getComputeCost() > 100;
      }]
    >,
  ];
}
```

#### Step 2: Implement in Operations

```tablegen
// MyDialectOps.td
def MatMulOp : Op<MyDialect, "matmul", [ComputeOpInterface]> {
  let arguments = (ins
    F32Tensor:$lhs,
    F32Tensor:$rhs
  );
  let results = (outs F32Tensor:$result);

  let extraClassDeclaration = [{
    // Override getComputeCost
    int64_t getComputeCost() {
      auto lhsType = getLhs().getType().cast<TensorType>();
      auto rhsType = getRhs().getType().cast<TensorType>();

      if (!lhsType.hasRank() || !rhsType.hasRank())
        return 1000;  // Unknown, assume expensive

      // Cost = M * N * K for matmul
      auto lhsShape = lhsType.getShape();
      auto rhsShape = rhsType.getShape();
      return lhsShape[0] * lhsShape[1] * rhsShape[1];
    }
  }];
}

def AddOp : Op<MyDialect, "add", [ComputeOpInterface]> {
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);

  // Uses default implementation (3 = 2 operands + 1 result)
}
```

#### Step 3: Use in Passes

```cpp
// OptimizeComputePass.cpp
#include "ComputeInterfaces.h"

struct OptimizeComputePass : public PassWrapper<...> {
  void runOnOperation() override {
    getOperation()->walk([](Operation *op) {
      auto computeOp = dyn_cast<mydiala::ComputeOpInterface>(op);
      if (!computeOp)
        return;

      int64_t cost = computeOp.getComputeCost();

      if (computeOp.isComputeIntensive()) {
        // Schedule for GPU execution
        llvm::errs() << "Expensive op: " << *op
                     << " (cost: " << cost << ")\n";
      }
    });
  }
};
```

---

### Example 2: Implementing an AttrInterface Externally

Let's add a custom compilation interface to an existing attribute.

#### Step 1: Define the Interface

```tablegen
// MyTargetInterface.td
def MyTargetInterface : AttrInterface<"MyTargetInterface"> {
  let cppNamespace = "::mytarget";

  let methods = [
    InterfaceMethod<"Compile to binary",
      "std::optional<std::string>",
      "compileToBinary",
      (ins "Operation*":$module)
    >,
  ];
}
```

#### Step 2: External Implementation

```cpp
// MyTargetImpl.cpp
#include "mlir/IR/BuiltinAttributes.h"
#include "MyTargetInterface.h"

namespace {
class StringAttrTargetImpl
    : public mytarget::MyTargetInterface::FallbackModel<StringAttrTargetImpl> {
public:
  std::optional<std::string>
  compileToBinary(Attribute attribute, Operation *module) const {
    auto strAttr = cast<StringAttr>(attribute);
    std::string targetName = strAttr.getValue().str();

    if (targetName == "x86") {
      return compileForX86(module);
    } else if (targetName == "arm") {
      return compileForARM(module);
    }

    return std::nullopt;  // Unsupported target
  }

private:
  std::string compileForX86(Operation *module) const {
    // Compilation logic
    return "binary code for x86";
  }

  std::string compileForARM(Operation *module) const {
    return "binary code for arm";
  }
};
} // namespace

void registerMyTargetInterface(MLIRContext &ctx) {
  StringAttr::attachInterface<StringAttrTargetImpl>(ctx);
}
```

#### Step 3: Usage

```cpp
int main() {
  MLIRContext ctx;
  registerMyTargetInterface(ctx);

  // Create a string attribute as a "target"
  auto targetAttr = StringAttr::get(&ctx, "x86");

  // Cast to interface
  auto target = cast<mytarget::MyTargetInterface>(targetAttr);

  // Use the interface
  if (auto binary = target.compileToBinary(module)) {
    llvm::outs() << "Compiled: " << *binary << "\n";
  }
}
```

---

## Common Patterns

### Pattern 1: Interfaces with Default Implementations

```tablegen
def MyInterface : OpInterface<"MyInterface"> {
  let methods = [
    InterfaceMethod<"Get size estimate",
      "int64_t", "getSizeEstimate", (ins),
      [{}],  // No body for overriding
      /*defaultImplementation=*/[{
        // Default: count operands and results
        return $_op.getNumOperands() + $_op.getNumResults();
      }]
    >,
  ];
}
```

**When to use**: Provide sensible defaults that most implementations can use.

---

### Pattern 2: Static Interface Methods

```tablegen
def MyInterface : OpInterface<"MyInterface"> {
  let methods = [
    StaticInterfaceMethod<"Check compatibility",
      "bool", "isCompatible",
      (ins "Type":$lhs, "Type":$rhs),
      [{}],
      /*defaultImplementation=*/[{
        return lhs == rhs;
      }]
    >,
  ];
}
```

**Usage**: For methods that don't need access to the operation instance.

```cpp
bool compat = MyInterface::isCompatible(type1, type2);
```

---

### Pattern 3: Interface Inheritance

```tablegen
def BaseInterface : OpInterface<"BaseInterface"> {
  let methods = [
    InterfaceMethod<"Base method", "void", "baseMethod", (ins)>,
  ];
}

def DerivedInterface : OpInterface<"DerivedInterface", [BaseInterface]> {
  //                                                     ^^^^^^^^^^^^^
  //                                                     Inherit from base
  let methods = [
    InterfaceMethod<"Derived method", "void", "derivedMethod", (ins)>,
  ];
}
```

**Ops implementing DerivedInterface must implement both `baseMethod` and `derivedMethod`.**

---

### Pattern 4: Verification in Interfaces

```tablegen
def MyInterface : OpInterface<"MyInterface"> {
  let verify = [{
    // $_op refers to the operation being verified
    if ($_op.getNumOperands() < 2)
      return $_op.emitError("must have at least 2 operands");
    return success();
  }];
}
```

---

### Pattern 5: Extra Declarations

```tablegen
def MyInterface : OpInterface<"MyInterface"> {
  // In the interface class only
  let extraClassDeclaration = [{
    bool hasProperty() const {
      return getImpl()->checkProperty(this);
    }
  }];

  // In both interface and trait
  let extraSharedClassDeclaration = [{
    static constexpr int MagicNumber = 42;
  }];

  // In the trait only
  let extraTraitClassDeclaration = [{
    void helperMethod() {
      // Can access $_op or $_type or $_attr
    }
  }];
}
```

---

## Best Practices

### 1. **Use Interfaces for Common Behavior**

✅ **Good**: Define `ShapedTypeInterface` for all types with shapes
```cpp
if (auto shaped = dyn_cast<ShapedType>(type))
  return shaped.getShape();
```

❌ **Bad**: Check for specific types
```cpp
if (auto tensor = dyn_cast<TensorType>(type))
  return tensor.getShape();
else if (auto memref = dyn_cast<MemRefType>(type))
  return memref.getShape();
// ... endless chain
```

### 2. **Prefer Default Implementations**

```tablegen
InterfaceMethod<"Get size",
  "int64_t", "getSize", (ins),
  [{}],
  /*defaultImplementation=*/[{
    return $_type.getIntOrFloatBitWidth() / 8;
  }]
>
```

This allows implementations to override only when needed.

### 3. **Use External Models for Optional Features**

When a feature is optional (e.g., GPU compilation), use external models so:
- Core dialect doesn't depend on optional libraries
- Users can opt-in by registering the interface

### 4. **Document Interface Contracts**

```tablegen
def MyInterface : OpInterface<"MyInterface"> {
  let description = [{
    Interface for operations that...

    Implementing operations must satisfy:
    - Have at least one result
    - All operands must be of the same type
    - Results can be queried via `getOutputType()`
  }];
}
```

### 5. **Use Verification**

```tablegen
let verify = [{
  if ($_op.getNumResults() == 0)
    return $_op.emitError("interface requires at least one result");
  return success();
}];
```

### 6. **Namespace Your Interfaces**

```tablegen
let cppNamespace = "::myproject::mydiala";
```

Avoids naming conflicts.

---

## Troubleshooting

### Problem 1: "Interface not found" at runtime

**Symptom**: `dyn_cast<Interface>(op)` returns nullptr even though the op declares the interface.

**Solution**: Ensure the interface is properly included and linked:

```cpp
#include "MyInterface.h.inc"     // Interface definition
#include "MyInterface.cpp.inc"   // Interface implementation
```

In CMake:
```cmake
mlir_tablegen(MyInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(MyInterface.cpp.inc -gen-op-interface-defs)
```

---

### Problem 2: External model not working

**Symptom**: External interface methods are not called.

**Solution**: Ensure registration happens before use:

```cpp
MLIRContext ctx;
// Register BEFORE loading dialect
registerMyExternalInterfaces(ctx);
ctx.getOrLoadDialect<MyDialect>();
```

---

### Problem 3: CRTP compilation errors with FallbackModel

**Symptom**: "incomplete type" or "cannot cast" errors.

**Solution**: Ensure the derived class is fully defined:

```cpp
namespace {
class MyImpl : public Interface::FallbackModel<MyImpl> {
//   ^^^^^^^                                    ^^^^^^^
//   Must match exactly!
public:
  ReturnType method(Attribute attr, ...) const;
  //                ^^^^^^^^^^
  //                Must take attribute as first parameter
};
} // namespace
```

---

### Problem 4: Method signature mismatch

**Symptom**: Linker errors or "no matching method" errors.

**Check**:
- **Native Model**: Method is on the op/type/attr itself, takes NO attribute parameter
- **External Model**: Method is on the model class, takes attribute as FIRST parameter

```cpp
// Native (method on MyCastOp)
bool MyCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) { ... }

// External (method on MyImpl, taking attribute)
bool MyImpl::areCastCompatible(Attribute attr, TypeRange inputs, TypeRange outputs) { ... }
```

---

## Quick Reference

### TableGen Basics

```tablegen
def MyOpInterface : OpInterface<"MyOpInterface"> {
  let description = "...";
  let cppNamespace = "::myns";
  let methods = [...];
  let verify = [{ ... }];
  let extraClassDeclaration = [{ ... }];
}

def MyAttrInterface : AttrInterface<"MyAttrInterface"> { ... }
def MyTypeInterface : TypeInterface<"MyTypeInterface"> { ... }
```

### Interface Methods

```tablegen
InterfaceMethod<"description", "ReturnType", "methodName",
  (ins "ArgType":$argName, ...),
  /*methodBody=*/[{}],
  /*defaultImplementation=*/[{ ... }]
>

StaticInterfaceMethod<"description", "ReturnType", "methodName",
  (ins "ArgType":$argName, ...), ...>
```

### Implementation Strategies

| Strategy | Use When | Base Class |
|----------|----------|------------|
| Native | Defining new ops/types | Declare interface in TableGen |
| External (FallbackModel) | Adding to existing types | `Interface::FallbackModel<Derived>` |
| External (ExternalModel) | Explicit model/entity separation | `Interface::ExternalModel<Model, Entity>` |

### Registration

```cpp
// External models
ConcreteType::attachInterface<ModelImpl>(ctx);

// In registry
registry.addExtension(+[](MLIRContext *ctx, MyDialect *dialect) {
  MyAttr::attachInterface<MyImpl>(*ctx);
});
```

---

## Further Reading

- **Official Docs**: [MLIR Interfaces](https://mlir.llvm.org/docs/Interfaces/)
- **Design Pattern Guide**: [MLIR_INTERFACE_DESIGN_PATTERN.md](/home/jeromeku/llvm-project/MLIR_INTERFACE_DESIGN_PATTERN.md)
- **Test Examples**: `mlir/test/lib/Dialect/Test/TestInterfaces.td`
- **Real Examples**:
  - Simple: `mlir/include/mlir/Interfaces/CastInterfaces.td`
  - Complex: `mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td`

---

## Summary

**MLIR Interfaces** provide a powerful mechanism for polymorphic behavior:

1. **Define once** in TableGen
2. **Implement** natively or externally
3. **Use** via uniform interface

They enable:
- ✅ Non-intrusive extension of existing types
- ✅ Zero-cost abstraction (no virtual functions)
- ✅ Compile-time type safety
- ✅ Clean separation of concerns

**Key takeaway**: Interfaces are MLIR's answer to duck typing with compile-time guarantees and zero runtime overhead!
