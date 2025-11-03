# MLIR Interfaces Quick Reference Cheat Sheet

## Interface Types

```tablegen
// Operation interface
def MyOpInterface : OpInterface<"MyOpInterface"> { ... }

// Attribute interface
def MyAttrInterface : AttrInterface<"MyAttrInterface"> { ... }

// Type interface
def MyTypeInterface : TypeInterface<"MyTypeInterface"> { ... }
```

---

## Defining Methods

### Instance Method (Non-Static)

```tablegen
InterfaceMethod<
  "Description",                    // Documentation
  "ReturnType",                     // Return type
  "methodName",                     // Method name
  (ins "ArgType":$arg1, ...),       // Arguments
  /*methodBody=*/[{}],              // Body for override (optional)
  /*defaultImplementation=*/[{      // Default impl (optional)
    return $_op.someMethod();       // $_op, $_type, or $_attr available
  }]
>
```

### Static Method

```tablegen
StaticInterfaceMethod<
  "Description",
  "ReturnType",
  "methodName",
  (ins "ArgType":$arg1, ...),
  [{}],                            // Body
  /*defaultImplementation=*/[{ ... }]
>
```

---

## Implementation Strategies

### Strategy 1: Native (Intrusive)

**In TableGen:**
```tablegen
def MyCastOp : Op<MyDialect, "cast", [CastOpInterface]> {
  //                                  ^^^^^^^^^^^^^^^^ Declare interface

  let extraClassDeclaration = [{
    // Implement interface methods
    static bool areCastCompatible(TypeRange inputs, TypeRange outputs);
  }];
}
```

**In C++:**
```cpp
bool MyCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return inputs.size() == 1 && outputs.size() == 1;
}
```

---

### Strategy 2: External Model (Non-Intrusive)

**Define Model:**
```cpp
namespace {
class MyAttrImpl
    : public MyInterface::FallbackModel<MyAttrImpl> {
    //                                  ^^^^^^^^^^^ CRTP: pass yourself
public:
  // Methods take attribute as FIRST parameter
  ReturnType methodName(Attribute attribute, ...) const {
    auto concreteAttr = cast<ConcreteAttr>(attribute);
    // Implementation
    return ...;
  }
};
} // namespace
```

**Register:**
```cpp
void registerMyInterface(MLIRContext &ctx) {
  ConcreteAttr::attachInterface<MyAttrImpl>(ctx);
}

// Or via registry
void registerMyInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, MyDialect *dialect) {
    ConcreteAttr::attachInterface<MyAttrImpl>(*ctx);
  });
}
```

**Initialize:**
```cpp
int main() {
  MLIRContext ctx;
  registerMyInterface(ctx);  // BEFORE using the interface
  // ...
}
```

---

## Method Signature Differences

| Implementation | Method Location | Signature |
|----------------|-----------------|-----------|
| **Native** | On the op/type/attr | `ReturnType method(Args...)` |
| **External** | On the model class | `ReturnType method(Attribute attr, Args...)` |

### Example:

**Native:**
```cpp
class MyCastOp {
  bool areCastCompatible(TypeRange in, TypeRange out);
  //   ^^^^^^^^^^^^^^^^ No attribute parameter
};
```

**External:**
```cpp
class MyImpl : public Interface::FallbackModel<MyImpl> {
  bool areCastCompatible(Attribute attr, TypeRange in, TypeRange out);
  //                     ^^^^^^^^^^^^^^ Attribute as first param
};
```

---

## Common Patterns

### Pattern 1: Verification

```tablegen
def MyInterface : OpInterface<"MyInterface"> {
  let verify = [{
    if ($_op.getNumOperands() == 0)
      return $_op.emitError("must have operands");
    return success();
  }];
}
```

### Pattern 2: Default Implementation

```tablegen
InterfaceMethod<"Get cost", "int64_t", "getCost", (ins),
  [{}],
  /*defaultImplementation=*/[{
    return $_op.getNumOperands();  // Default behavior
  }]
>
```

### Pattern 3: Interface Inheritance

```tablegen
def BaseInterface : OpInterface<"BaseInterface"> {
  let methods = [
    InterfaceMethod<"Base method", "void", "baseMethod", (ins)>,
  ];
}

def DerivedInterface : OpInterface<"DerivedInterface", [BaseInterface]> {
  //                                                     ^^^^^^^^^^^^^^
  let methods = [
    InterfaceMethod<"Derived method", "void", "derivedMethod", (ins)>,
  ];
}
```

### Pattern 4: Extra Declarations

```tablegen
def MyInterface : OpInterface<"MyInterface"> {
  // In interface class only
  let extraClassDeclaration = [{
    void helperMethod() const { ... }
  }];

  // In both interface and trait
  let extraSharedClassDeclaration = [{
    static constexpr int Value = 42;
  }];

  // In trait only
  let extraTraitClassDeclaration = [{
    void traitHelper() { ... }
  }];
}
```

---

## Special Variables in Interface Bodies

| Variable | Available In | Description |
|----------|--------------|-------------|
| `$_op` | OpInterface methods | The operation being accessed |
| `$_type` | TypeInterface methods | The type being accessed |
| `$_attr` | AttrInterface methods | The attribute being accessed |
| `$argName` | Any method | Named argument from `(ins "Type":$argName)` |

### Example:

```tablegen
InterfaceMethod<"Get bit width", "unsigned", "getBitWidth", (ins),
  [{}],
  /*defaultImplementation=*/[{
    return $_type.getIntOrFloatBitWidth();
    //     ^^^^^^ Refers to the type implementing the interface
  }]
>
```

---

## Using Interfaces

### Checking if an Entity Implements an Interface

```cpp
Operation *op = ...;
if (auto interface = dyn_cast<MyOpInterface>(op)) {
  // Use the interface
  auto result = interface.methodName(...);
}

// Or with Type/Attribute
Type type = ...;
if (auto interface = dyn_cast<MyTypeInterface>(type)) { ... }

Attribute attr = ...;
if (auto interface = dyn_cast<MyAttrInterface>(attr)) { ... }
```

### Calling Static Methods

```cpp
bool compat = MyOpInterface::staticMethod(arg1, arg2);
```

### Walking All Ops with an Interface

```cpp
module.walk([](Operation *op) {
  if (auto interface = dyn_cast<MyInterface>(op)) {
    interface.doSomething();
  }
});
```

---

## CMake Configuration

### TableGen Generation

```cmake
set(LLVM_TARGET_DEFINITIONS MyInterfaces.td)
mlir_tablegen(MyInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(MyInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MyInterfacesIncGen)

# For attribute interfaces:
mlir_tablegen(MyInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(MyInterfaces.cpp.inc -gen-attr-interface-defs)

# For type interfaces:
mlir_tablegen(MyInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(MyInterfaces.cpp.inc -gen-type-interface-defs)
```

### Include in C++

```cpp
// MyInterfaces.h
#include "MyInterfaces.h.inc"

// MyInterfaces.cpp
#include "MyInterfaces.cpp.inc"
```

---

## Common Use Cases

### 1. Cast-like Operations

```tablegen
def CastOpInterface : OpInterface<"CastOpInterface"> {
  let methods = [
    StaticInterfaceMethod<"Check cast compatibility",
      "bool", "areCastCompatible",
      (ins "TypeRange":$inputs, "TypeRange":$outputs)>,
  ];
}
```

### 2. Shaped Types

```tablegen
def ShapedTypeInterface : TypeInterface<"ShapedType"> {
  let methods = [
    InterfaceMethod<"Get shape", "ArrayRef<int64_t>", "getShape">,
    InterfaceMethod<"Has rank", "bool", "hasRank">,
    InterfaceMethod<"Get rank", "int64_t", "getRank">,
  ];
}
```

### 3. Memory Effects

```tablegen
def MemoryEffectOpInterface : OpInterface<"MemoryEffectOpInterface"> {
  let methods = [
    InterfaceMethod<"Get memory effects",
      "void", "getEffects",
      (ins "SmallVectorImpl<MemoryEffects::EffectInstance>&":$effects)>,
  ];
}
```

### 4. Side Effect-Free Operations

```tablegen
def Pure : OpInterface<"Pure"> {
  let description = [{
    Operation has no side effects and can be safely eliminated if unused.
  }];
}
```

---

## Debugging Tips

### 1. Print Generated Code

```bash
mlir-tblgen -gen-op-interface-decls MyInterfaces.td -o -
mlir-tblgen -gen-op-interface-defs MyInterfaces.td -o -
```

### 2. Check Interface Registration

```cpp
bool isRegistered = ConcreteType::hasInterface<MyInterface>(ctx);
```

### 3. Inspect at Runtime

```cpp
op->getInterfaceMap().lookup(MyInterface::getInterfaceID());
```

### 4. Verify Interface Conformance

```cpp
if (auto interface = dyn_cast<MyInterface>(op)) {
  llvm::errs() << "Op implements interface\n";
} else {
  llvm::errs() << "Op does NOT implement interface\n";
}
```

---

## Common Errors

### Error: "Interface not found at runtime"

**Solution**: Ensure interface is properly included and generated:
```cpp
#include "MyInterface.h.inc"
#include "MyInterface.cpp.inc"
```

### Error: "Method signature mismatch"

**Solution**: Check Native vs External signature:
- Native: `ReturnType method(Args...)`
- External: `ReturnType method(Attribute attr, Args...)`

### Error: "CRTP incomplete type"

**Solution**: Ensure model class name matches template parameter:
```cpp
class MyImpl : public Interface::FallbackModel<MyImpl> {
  //  ^^^^^^                                    ^^^^^^ Must match!
};
```

### Error: "External interface not working"

**Solution**: Register BEFORE use:
```cpp
registerMyInterface(ctx);  // First
ctx.getOrLoadDialect<MyDialect>();  // Then
```

---

## File Organization

### Typical Project Structure

```
MyDialect/
├── IR/
│   ├── MyInterfaces.td          # Interface definitions
│   ├── MyInterfaces.h           # Include generated .h.inc
│   ├── MyInterfaces.cpp         # Include generated .cpp.inc
│   ├── MyOps.td                 # Ops implementing interfaces
│   └── MyOps.cpp
├── Target/
│   └── MyTarget.cpp             # External interface implementations
└── CMakeLists.txt
```

---

## Quick Decision Tree

**Should I use an interface?**

```
Do multiple ops/types need similar behavior?
├─ Yes: Define an interface
└─ No: Use regular class methods

Can I modify the type definition?
├─ Yes: Use native implementation (intrusive)
└─ No: Use external model (non-intrusive)

Is the feature optional?
├─ Yes: Use external model + conditional registration
└─ No: Use native implementation

Do I need static dispatch?
├─ Yes: Use StaticInterfaceMethod
└─ No: Use InterfaceMethod
```

---

## Resources

- **Full Tutorial**: [MLIR_INTERFACES_TUTORIAL.md](/home/jeromeku/llvm-project/MLIR_INTERFACES_TUTORIAL.md)
- **Design Patterns**: [MLIR_INTERFACE_DESIGN_PATTERN.md](/home/jeromeku/llvm-project/MLIR_INTERFACE_DESIGN_PATTERN.md)
- **Official Docs**: https://mlir.llvm.org/docs/Interfaces/
- **Test Examples**: `mlir/test/lib/Dialect/Test/TestInterfaces.td`
- **Real Examples**: `mlir/include/mlir/Interfaces/`

---

## Summary

**3 Steps to Use Interfaces:**

1. **Define** in TableGen (`.td` file)
2. **Implement** natively (in op def) or externally (via FallbackModel)
3. **Use** via `dyn_cast<Interface>(op/type/attr)`

**Key Points:**
- ✅ Zero-cost abstraction (CRTP, no virtual functions)
- ✅ Non-intrusive extension (external models)
- ✅ Compile-time type safety
- ✅ Uniform interface across different types
