# MLIR Interface Design Pattern: External Model & FallbackModel

## Your Question

From [Target.cpp:59-61](/home/jeromeku/llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp#L59-L61):

```cpp
class NVVMTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<NVVMTargetAttrImpl> {
```

**Is this CRTP?** Yes! But it's a sophisticated variant used for implementing **external interface models** in MLIR.

---

## The Complete Inheritance Hierarchy

Let me trace through the full chain:

### 1. The Interface Definition (TableGen)

From [CompilationAttrInterfaces.td:22-56](/home/jeromeku/llvm-project/mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td#L22-L56):

```tablegen
def GPUTargetAttrInterface : AttrInterface<"TargetAttrInterface"> {
  let cppNamespace = "::mlir::gpu";
  let methods = [
    InterfaceMethod<"serializeToObject", ...>,
    InterfaceMethod<"createObject", ...>
  ];
}
```

### 2. Generated Code Structure

TableGen generates [CompilationAttrInterfaces.h.inc](/home/jeromeku/llvm-project/build/tools/mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.h.inc):

```cpp
namespace mlir::gpu::detail {
  struct TargetAttrInterfaceInterfaceTraits {
    // The pure virtual interface
    struct Concept {
      std::optional<SmallVector<char, 0>> (*serializeToObject)(...);
      Attribute (*createObject)(...);
    };

    // For types that NATIVELY implement the interface
    template<typename ConcreteAttr>
    class Model : public Concept { ... };

    // For EXTERNAL implementations (what you're using!)
    template<typename ConcreteAttr>
    class FallbackModel : public Concept { ... };

    // Convenience: ExternalModel extends FallbackModel
    template<typename ConcreteModel, typename ConcreteAttr>
    class ExternalModel : public FallbackModel<ConcreteModel> { ... };
  };
}
```

### 3. The Public Interface Class

```cpp
class TargetAttrInterface
    : public AttributeInterface<
        TargetAttrInterface,                           // CRTP: self-type
        detail::TargetAttrInterfaceInterfaceTraits     // Traits
      >
{
public:
  // User-facing methods that delegate to the Concept
  std::optional<SmallVector<char, 0>> serializeToObject(...) const;
  Attribute createObject(...) const;
};
```

### 4. Your Implementation (External Model)

From [Target.cpp:59-69](/home/jeromeku/llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp#L59-L69):

```cpp
class NVVMTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<NVVMTargetAttrImpl>
//           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^           ^^^^^^^^^^^^^^^^^
//           Interface class                        CRTP: derived type!
{
public:
  // Implement the interface methods with THIS signature:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute,  // <-- Note: takes Attribute, not module
                    Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, ...) const;
};
```

---

## What's Happening: The Three Models

MLIR provides three different ways to implement an interface:

### Model 1: **Model<ConcreteAttr>** - Native Implementation

For types that **inherently implement** the interface (defined in the same dialect):

```cpp
// In NVVMAttrDefs.td:
def NVVMTargetAttr : ... {
  let extraClassDeclaration = [{
    std::optional<SmallVector<char, 0>> serializeToObject(...);
    //                                  ^^^^^^ operates on 'this'
  }];
}

// Generated implementation:
template<typename ConcreteAttr>
std::optional<SmallVector<char, 0>>
Model<ConcreteAttr>::serializeToObject(..., Attribute tablegen_opaque_val, ...) {
  return (llvm::cast<ConcreteAttr>(tablegen_opaque_val))
      .serializeToObject(module, options);
  //    ^^^^^^^^^^^^^^^^^ Cast the attribute and call its method
}
```

**Key**: The attribute **object itself** has the method.

### Model 2: **FallbackModel<ConcreteModel>** - External Implementation (YOU!)

For **external implementations** that don't modify the original type:

```cpp
class NVVMTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<NVVMTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute,  // <-- Takes the attribute as parameter!
                    Operation *module,
                    const gpu::TargetOptions &options) const;
};

// Generated implementation:
template<typename ConcreteAttr>
std::optional<SmallVector<char, 0>>
FallbackModel<ConcreteAttr>::serializeToObject(..., Attribute tablegen_opaque_val, ...) {
  return static_cast<const ConcreteAttr *>(impl)
      ->serializeToObject(tablegen_opaque_val, module, options);
  //    ^^^^^^^^^^^^^^^^^ Cast 'this' to the derived type (CRTP!)
  //                      and pass the attribute as a parameter
}
```

**Key**: The **model object** has the method, not the attribute.

### Model 3: **ExternalModel<Model, Attr>** - Syntactic Sugar

Just extends `FallbackModel`:

```cpp
template<typename ConcreteModel, typename ConcreteAttr>
class ExternalModel : public FallbackModel<ConcreteModel> {
public:
  using ConcreteEntity = ConcreteAttr;
};
```

Provides a `ConcreteEntity` typedef for convenience.

---

## The CRTP Pattern in Detail

### Step 1: Derive from FallbackModel

```cpp
class NVVMTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<NVVMTargetAttrImpl>
//                                                    ^^^^^^^^^^^^^^^^^
//                                                    Pass yourself as template param!
```

### Step 2: FallbackModel Uses Static Polymorphism

From the generated code (line 223):

```cpp
template<typename ConcreteAttr>
std::optional<SmallVector<char, 0>>
FallbackModel<ConcreteAttr>::serializeToObject(
    const Concept *impl,                    // 'impl' is actually a pointer to the DERIVED type
    Attribute tablegen_opaque_val,
    Operation* module,
    const gpu::TargetOptions& options)
{
  return static_cast<const ConcreteAttr *>(impl)  // <-- CRTP downcast!
      ->serializeToObject(tablegen_opaque_val, module, options);
}
```

### Step 3: Registration Links Everything

From [Target.cpp:73-78](/home/jeromeku/llvm-project/mlir/lib/Target/LLVM/NVVM/Target.cpp#L73-L78):

```cpp
void mlir::NVVM::registerNVVMTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, NVVM::NVVMDialect *dialect) {
    NVVMTargetAttr::attachInterface<NVVMTargetAttrImpl>(*ctx);
    //              ^^^^^^^^^^^^^^^ Registers the external model
  });
}
```

This creates an instance of `NVVMTargetAttrImpl` and stores it in the context, associated with `NVVMTargetAttr`.

---

## How It Works at Runtime

### Call Flow

```
User code:
  auto attr = NVVMTargetAttr::get(...);
  auto interface = attr.cast<gpu::TargetAttrInterface>();
  interface.serializeToObject(module, options);

┌──────────────────────────────────────────────────────────────────┐
│ TargetAttrInterface::serializeToObject(...)                      │
│   ↓                                                               │
│   return getImpl()->serializeToObject(this, module, options);    │
│          ^^^^^^^^^^                                               │
│          Returns the registered Concept* for this attribute       │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ FallbackModel<NVVMTargetAttrImpl>::serializeToObject(           │
│     const Concept *impl,              // Points to the instance  │
│     Attribute tablegen_opaque_val,    // The NVVMTargetAttr     │
│     Operation *module,                                           │
│     const TargetOptions &options)                                │
│   ↓                                                               │
│   return static_cast<const NVVMTargetAttrImpl *>(impl)           │
│       ->serializeToObject(tablegen_opaque_val, module, options); │
│          ^^^^^^^^^^^^^^^^^                                        │
│          CRTP downcast to derived type!                          │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ NVVMTargetAttrImpl::serializeToObject(                          │
│     Attribute attribute,                                         │
│     Operation *module,                                           │
│     const TargetOptions &options) const                          │
│   ↓                                                               │
│   // Your actual implementation                                  │
│   auto nvvmAttr = cast<NVVMTargetAttr>(attribute);              │
│   ... compile the module ...                                     │
│   return binary;                                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Why This Design?

### Problem 1: Non-Intrusive Extension

You want to add interface implementations to types **without modifying their definition**.

Example: `NVVMTargetAttr` is defined in the NVVM dialect, but compilation logic lives in `mlir/lib/Target/LLVM/NVVM/`. These are separate libraries!

### Problem 2: Optional Functionality

Not all builds include all targets. The interface should work even if:
- NVVM target isn't available
- ROCDL target isn't available
- User adds custom GPU targets

### Solution: External Models

```cpp
// In the dialect (always compiled):
def NVVMTargetAttr : ... {
  // No implementation here!
}

// In the target library (optionally compiled):
class NVVMTargetAttrImpl
    : public TargetAttrInterface::FallbackModel<NVVMTargetAttrImpl> {
  // Implementation here!
};

// Registration happens only if target library is linked:
NVVMTargetAttr::attachInterface<NVVMTargetAttrImpl>(*ctx);
```

---

## Comparison: Model vs FallbackModel

### Native Model (Intrusive)

```cpp
// Attribute MUST have this method:
class NVVMTargetAttr {
  std::optional<SmallVector<char, 0>>
  serializeToObject(Operation *module,
                    const TargetOptions &options) const;
  //                ^^^^^^^ 'this' is the attribute
};

// Generated Model calls it:
return cast<NVVMTargetAttr>(attribute).serializeToObject(module, options);
```

### FallbackModel (Non-Intrusive)

```cpp
// External implementation:
class NVVMTargetAttrImpl {
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute,  // <-- attribute passed IN
                    Operation *module,
                    const TargetOptions &options) const;
};

// Generated FallbackModel calls it:
return static_cast<const NVVMTargetAttrImpl*>(this)
    ->serializeToObject(attribute, module, options);
```

**Key Difference**:
- **Model**: Method is ON the attribute (intrusive)
- **FallbackModel**: Method is ON the model, attribute is a parameter (non-intrusive)

---

## The CRTP Aspect

Yes, this is **CRTP** (Curiously Recurring Template Pattern):

```cpp
// The pattern:
template<typename Derived>
class Base {
  void interface() {
    static_cast<Derived*>(this)->implementation();
    //          ^^^^^^^ Downcast to derived type at compile time
  }
};

class Concrete : public Base<Concrete> {
  void implementation() { /* ... */ }
};

// In MLIR:
template<typename ConcreteModel>
class FallbackModel : public Concept {
  ReturnType method(..., const Concept *impl, ...) {
    return static_cast<const ConcreteModel*>(impl)->method(...);
    //                      ^^^^^^^^^^^^^ CRTP downcast
  }
};

class NVVMTargetAttrImpl : public FallbackModel<NVVMTargetAttrImpl> {
  //                                            ^^^^^^^^^^^^^^^^^^
  //                                            Pass yourself!
  ReturnType method(...) { /* implementation */ }
};
```

**Why CRTP?**
- **Static polymorphism**: No virtual function overhead
- **Type safety**: Compile-time checking of method signatures
- **Flexibility**: Derived class methods are called directly

---

## Visual Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                   Your Code                                  │
│  class NVVMTargetAttrImpl                                   │
│      : public TargetAttrInterface::FallbackModel<           │
│                NVVMTargetAttrImpl> { ... }                  │
│                ^^^^^^^^^^^^^^^^^^                            │
│                CRTP parameter!                               │
└────────────────────┬────────────────────────────────────────┘
                     │ derives from
                     ↓
┌─────────────────────────────────────────────────────────────┐
│         Generated: FallbackModel<NVVMTargetAttrImpl>       │
│  template<typename ConcreteModel>                           │
│  class FallbackModel : public Concept {                     │
│    ReturnType method(..., const Concept *impl, ...) {       │
│      return static_cast<const ConcreteModel*>(impl)         │
│          ->method(...);  // <-- CRTP downcast               │
│    }                                                         │
│  }                                                           │
└────────────────────┬────────────────────────────────────────┘
                     │ derives from
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              Generated: Concept (vtable)                    │
│  struct Concept {                                            │
│    ReturnType (*method)(...);  // Function pointers         │
│  };                                                          │
└────────────────────┬────────────────────────────────────────┘
                     │ used by
                     ↓
┌─────────────────────────────────────────────────────────────┐
│         Public Interface: TargetAttrInterface               │
│  class TargetAttrInterface {                                │
│    ReturnType method(...) const {                           │
│      return getImpl()->method(this, ...);                   │
│    }                                                         │
│  private:                                                   │
│    Concept *conceptImpl;  // Pointer to the model           │
│  };                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Yes, it's CRTP**: `FallbackModel<NVVMTargetAttrImpl>` uses CRTP to call derived methods without virtual functions

2. **External Model Pattern**: Allows implementing interfaces for types **without modifying their definition**

3. **Three implementation strategies**:
   - **Model**: For native implementations (method on the attribute)
   - **FallbackModel**: For external implementations (method on the model, CRTP-based)
   - **ExternalModel**: Syntactic sugar for FallbackModel

4. **Type-erased at runtime**: The `Concept` struct uses function pointers (like a vtable), but resolved at **registration time**, not call time

5. **Static polymorphism**: CRTP provides compile-time polymorphism, avoiding virtual function overhead

6. **Separation of concerns**:
   - Dialect defines the type
   - Target library implements the interface
   - Registration connects them at runtime

---

## Related Patterns

- **Type Erasure**: `Concept` struct provides type-erased interface
- **CRTP**: Static polymorphism via `FallbackModel<Derived>`
- **Policy-Based Design**: Different models for different use cases
- **External Polymorphism**: Adding behavior without modifying original types

This is a sophisticated design that combines multiple patterns to achieve:
- ✅ Non-intrusive extension
- ✅ Zero-cost abstraction
- ✅ Compile-time type safety
- ✅ Optional functionality

Very elegant C++ design! 🎯
