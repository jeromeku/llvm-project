# MLIR Documentation Index

A curated collection of guides and tutorials for working with MLIR.

---

## 📚 Core Documentation

### ExecutionEngine

Deep dive into MLIR's JIT compilation and execution.

| Document | Description | Best For |
|----------|-------------|----------|
| [**ExecutionEngine Frame-by-Frame**](EXECUTION_ENGINE_FRAME_BY_FRAME.md) | Complete call trace from Python to LLVM | Understanding internals |
| [**ExecutionEngine Quick Reference**](EXECUTION_ENGINE_QUICK_REFERENCE.md) | Quick lookup and common patterns | Daily development |

**Topics Covered:**
- Python → Nanobind → MLIR-C → MLIR C++ → LLVM call stack
- Detailed trace with code snippets and line numbers
- CUDA-specific runtime wrappers and kernel loading
- Lazy compilation and packed function wrappers
- Shared library loading and symbol resolution
- Global constructors and GPU kernel initialization

---

### Interfaces

Understanding and implementing MLIR's interface system.

| Document | Description | Best For |
|----------|-------------|----------|
| [**Interfaces Tutorial**](MLIR_INTERFACES_TUTORIAL.md) | Comprehensive guide to interfaces | Learning from scratch |
| [**Interface Design Pattern**](MLIR_INTERFACE_DESIGN_PATTERN.md) | Deep dive into CRTP and external models | Understanding implementation |
| [**Interfaces Cheat Sheet**](MLIR_INTERFACES_CHEATSHEET.md) | Quick reference for common patterns | Quick lookup |

**Topics Covered:**
- What are MLIR interfaces?
- OpInterface, AttrInterface, TypeInterface
- Native vs External implementation
- FallbackModel and ExternalModel patterns
- Complete examples and usage patterns

---

### Python Bindings

Understanding and discovering MLIR Python APIs.

| Document | Description | Best For |
|----------|-------------|----------|
| [**Python Bindings Guide**](mlir-python-bindings-guide.md) | How MLIR Python bindings work | Understanding architecture |
| [**GPU Dialect Python API**](GPU_DIALECT_PYTHON_API.md) | Complete GPU dialect reference | Working with GPU dialect |
| [**Discovery Script**](discover_dialect_api.sh) | Tool to discover dialect APIs | Exploring new dialects |

**Topics Covered:**
- Three-layer architecture (TableGen + C++ bindings)
- Why `.pyi` stubs are incomplete
- Discovering APIs systematically
- `gpu.ObjectAttr` and other dialect-specific APIs
- Runtime inspection techniques

**Quick Start:**
```bash
# Discover APIs for any dialect
./discover_dialect_api.sh gpu
./discover_dialect_api.sh nvgpu
./discover_dialect_api.sh linalg
```

---

## 🎯 Quick Reference

### By Task

| I Want To... | Read This |
|--------------|-----------|
| Understand how ExecutionEngine works | [ExecutionEngine Frame-by-Frame](EXECUTION_ENGINE_FRAME_BY_FRAME.md) |
| Look up ExecutionEngine methods | [ExecutionEngine Quick Reference](EXECUTION_ENGINE_QUICK_REFERENCE.md) |
| Debug JIT-compiled code | [ExecutionEngine Quick Reference - Debugging](EXECUTION_ENGINE_QUICK_REFERENCE.md#debugging-checklist) |
| Understand CUDA kernel loading | [ExecutionEngine Frame-by-Frame - CUDA Specific](EXECUTION_ENGINE_FRAME_BY_FRAME.md#cuda-specific-runtime-wrappers) |
| Implement an interface for my ops | [Interfaces Tutorial - Native Implementation](MLIR_INTERFACES_TUTORIAL.md#strategy-1-native-implementation-intrusive) |
| Add an interface to existing types | [Interfaces Tutorial - External Model](MLIR_INTERFACES_TUTORIAL.md#strategy-2-external-model-non-intrusive) |
| Understand CRTP in MLIR | [Interface Design Pattern](MLIR_INTERFACE_DESIGN_PATTERN.md) |
| Look up interface syntax quickly | [Interfaces Cheat Sheet](MLIR_INTERFACES_CHEATSHEET.md) |
| Find Python APIs for a dialect | [Python Bindings Guide - Discovery Methods](mlir-python-bindings-guide.md#systematic-discovery-methods) |
| Use `gpu.ObjectAttr` in Python | [GPU Dialect Python API](GPU_DIALECT_PYTHON_API.md#gpuobjectattr) |

---

## 📖 Learning Paths

### Path 1: Working with Interfaces (C++)

1. **Start**: [Interfaces Tutorial - What Are Interfaces?](MLIR_INTERFACES_TUTORIAL.md#what-are-mlir-interfaces)
2. **Define**: [Interfaces Tutorial - Defining in TableGen](MLIR_INTERFACES_TUTORIAL.md#defining-interfaces-in-tablegen)
3. **Implement**: [Interfaces Tutorial - Implementation Strategies](MLIR_INTERFACES_TUTORIAL.md#implementation-strategies)
4. **Practice**: [Interfaces Tutorial - Complete Examples](MLIR_INTERFACES_TUTORIAL.md#complete-examples)
5. **Reference**: [Interfaces Cheat Sheet](MLIR_INTERFACES_CHEATSHEET.md)

**Deep Dive**: [Interface Design Pattern](MLIR_INTERFACE_DESIGN_PATTERN.md) - Understand the underlying CRTP mechanism

---

### Path 2: MLIR Python Development

1. **Understand**: [Python Bindings Guide - Architecture](mlir-python-bindings-guide.md#how-mlir-python-bindings-are-structured)
2. **Discover**: Run `./discover_dialect_api.sh <dialect>`
3. **Reference**: [GPU Dialect Python API](GPU_DIALECT_PYTHON_API.md)
4. **Learn**: [Python Bindings Guide - Discovery Methods](mlir-python-bindings-guide.md#systematic-discovery-methods)

**Pro Tip**: Keep [GPU Dialect Python API](GPU_DIALECT_PYTHON_API.md) open as a template for other dialects

---

### Path 3: Extending MLIR with External Models

1. **Motivation**: [Interface Design Pattern - Why This Design?](MLIR_INTERFACE_DESIGN_PATTERN.md#why-this-design)
2. **Pattern**: [Interface Design Pattern - FallbackModel](MLIR_INTERFACE_DESIGN_PATTERN.md#model-2-fallbackmodelconcretemodel---external-implementation-you)
3. **Implementation**: [Interfaces Tutorial - External Model Strategy](MLIR_INTERFACES_TUTORIAL.md#strategy-2-external-model-non-intrusive)
4. **Example**: [Interfaces Tutorial - Example 2](MLIR_INTERFACES_TUTORIAL.md#example-2-implementing-an-attrinterface-externally)

---

## 🔍 By Topic

### Interfaces

**Basics:**
- [What are interfaces?](MLIR_INTERFACES_TUTORIAL.md#what-are-mlir-interfaces)
- [Types of interfaces](MLIR_INTERFACES_TUTORIAL.md#types-of-interfaces)
- [Defining interfaces](MLIR_INTERFACES_TUTORIAL.md#defining-interfaces-in-tablegen)

**Implementation:**
- [Native implementation](MLIR_INTERFACES_TUTORIAL.md#strategy-1-native-implementation-intrusive)
- [External model](MLIR_INTERFACES_TUTORIAL.md#strategy-2-external-model-non-intrusive)
- [FallbackModel vs ExternalModel](MLIR_INTERFACE_DESIGN_PATTERN.md#comparison-model-vs-fallbackmodel)

**Advanced:**
- [CRTP in interfaces](MLIR_INTERFACE_DESIGN_PATTERN.md#the-crtp-pattern-in-detail)
- [Inheritance hierarchy](MLIR_INTERFACE_DESIGN_PATTERN.md#the-complete-inheritance-hierarchy)
- [Runtime dispatch](MLIR_INTERFACE_DESIGN_PATTERN.md#how-it-works-at-runtime)

**Reference:**
- [Quick syntax](MLIR_INTERFACES_CHEATSHEET.md#defining-methods)
- [Common patterns](MLIR_INTERFACES_CHEATSHEET.md#common-patterns)
- [Troubleshooting](MLIR_INTERFACES_TUTORIAL.md#troubleshooting)

---

### Python Bindings

**Architecture:**
- [Three-layer system](mlir-python-bindings-guide.md#how-mlir-python-bindings-are-structured)
- [TableGen generation](mlir-python-bindings-guide.md#1-tableg-generated-operations)
- [C++ bindings](mlir-python-bindings-guide.md#3-c-nanobind-extensions)

**Discovery:**
- [Reading C++ bindings](mlir-python-bindings-guide.md#method-1-read-the-c-binding-files)
- [Using test files](mlir-python-bindings-guide.md#method-2-read-python-test-files)
- [Runtime inspection](mlir-python-bindings-guide.md#method-3-inspect-at-runtime)
- [Discovery script](mlir-python-bindings-guide.md#complete-discovery-workflow-for-any-dialect)

**Specific APIs:**
- [gpu.ObjectAttr](GPU_DIALECT_PYTHON_API.md#gpuobjectattr)
- [gpu.AsyncTokenType](GPU_DIALECT_PYTHON_API.md#gpuasynctokentype)
- [Compilation enums](GPU_DIALECT_PYTHON_API.md#enums-tablegen-generated)

**Problems:**
- [Why no autocomplete?](mlir-python-bindings-guide.md#why-are-pyi-files-incomplete)
- [Missing type hints](GPU_DIALECT_PYTHON_API.md#why-ide-autocomplete-doesnt-work)

---

## 💡 Code Examples

### Interface Definition (TableGen)

```tablegen
// Simple OpInterface
def CastOpInterface : OpInterface<"CastOpInterface"> {
  let description = "Interface for cast-like operations";
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<"Check cast compatibility",
      "bool", "areCastCompatible",
      (ins "TypeRange":$inputs, "TypeRange":$outputs)>,
  ];
}
```

**Full example**: [Interfaces Tutorial - Example 1](MLIR_INTERFACES_TUTORIAL.md#example-1-simple-opinterface)

---

### External Model Implementation

```cpp
class NVVMTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<NVVMTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const {
    // Implementation
  }
};

// Registration
NVVMTargetAttr::attachInterface<NVVMTargetAttrImpl>(*ctx);
```

**Full example**: [Interfaces Tutorial - Example 2](MLIR_INTERFACES_TUTORIAL.md#example-2-implementing-an-attrinterface-externally)

---

### Python API Usage

```python
from mlir.dialects import gpu
from mlir.ir import Attribute

# Create ObjectAttr
target = Attribute.parse("#nvvm.target")
obj = gpu.ObjectAttr.get(target, gpu.CompilationTarget.Fatbin, b"binary")

# Access properties
print(f"Format: {obj.format}")
print(f"Binary size: {len(obj.object)}")
```

**Full example**: [GPU Dialect Python API - Complete Example](GPU_DIALECT_PYTHON_API.md#complete-example-extracting-ptx-from-binaryop)

---

## 🛠️ Tools

| Tool | Description | Usage |
|------|-------------|-------|
| `discover_dialect_api.sh` | Discover Python APIs for any dialect | `./discover_dialect_api.sh gpu` |
| `mlir-tblgen` | Generate interface code from TableGen | `mlir-tblgen -gen-op-interface-decls MyInterface.td` |

---

## 📝 File Structure

### Project Organization

```
llvm-project/
├── MLIR_DOCUMENTATION_INDEX.md          ← You are here
├── MLIR_INTERFACES_TUTORIAL.md          ← Comprehensive tutorial
├── MLIR_INTERFACES_CHEATSHEET.md        ← Quick reference
├── MLIR_INTERFACE_DESIGN_PATTERN.md     ← Deep dive into CRTP
├── mlir-python-bindings-guide.md        ← Python bindings guide
├── GPU_DIALECT_PYTHON_API.md            ← GPU dialect reference
├── discover_dialect_api.sh              ← Discovery tool
└── mlir/
    ├── include/mlir/IR/Interfaces.td    ← Interface base definitions
    ├── include/mlir/Interfaces/         ← Core interfaces
    ├── test/lib/Dialect/Test/TestInterfaces.td  ← Test examples
    └── lib/Bindings/Python/             ← Python bindings
```

---

## 🔗 External Resources

### Official MLIR Documentation

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [MLIR Interfaces](https://mlir.llvm.org/docs/Interfaces/)
- [MLIR Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/)
- [TableGen Documentation](https://llvm.org/docs/TableGen/)

### Source Code Examples

- **Simple Interface**: `mlir/include/mlir/Interfaces/CastInterfaces.td`
- **Complex Interface**: `mlir/include/mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td`
- **Test Interfaces**: `mlir/test/lib/Dialect/Test/TestInterfaces.td`
- **External Models**: `mlir/lib/Target/LLVM/NVVM/Target.cpp`

---

## 🎓 Learning Tips

### For C++ Developers

1. **Start with simple interfaces** like `CastOpInterface`
2. **Read the generated code** to understand what TableGen produces
3. **Use test examples** in `mlir/test/lib/Dialect/Test/`
4. **Reference the cheat sheet** for syntax

### For Python Developers

1. **Run the discovery script** first
2. **Read test files** in `mlir/test/python/dialects/`
3. **Check C++ bindings** for missing APIs
4. **Use runtime inspection** with `dir()` and `help()`

### General Tips

- 💡 **Examples first**: Start with working examples, then read theory
- 🔍 **Use the discovery script**: Don't guess, discover systematically
- 📖 **Read test files**: They're often the best documentation
- 🎯 **Start simple**: Master basic interfaces before external models

---

## 🆘 Getting Help

### Troubleshooting

- **Interfaces**: [Troubleshooting section](MLIR_INTERFACES_TUTORIAL.md#troubleshooting)
- **Common errors**: [Cheat sheet errors](MLIR_INTERFACES_CHEATSHEET.md#common-errors)
- **Python issues**: [Python guide FAQ](mlir-python-bindings-guide.md#practical-tips)

### When Stuck

1. Check the **cheat sheet** for syntax
2. Look at **test files** for working examples
3. Read the **tutorial** for detailed explanations
4. Examine **generated code** to understand the machinery

---

## 📊 Document Comparison

| Document | Scope | Depth | Best For |
|----------|-------|-------|----------|
| Tutorial | Comprehensive | Medium | Learning |
| Cheat Sheet | Quick reference | Shallow | Quick lookup |
| Design Pattern | CRTP & External Models | Deep | Understanding internals |
| Python Guide | Python bindings | Medium | Python development |
| GPU API | GPU dialect | Medium | GPU-specific work |

---

## ✅ Quick Start Checklist

### Implementing an Interface

- [ ] Define interface in `.td` file
- [ ] Add TableGen rules to `CMakeLists.txt`
- [ ] Include generated `.h.inc` and `.cpp.inc`
- [ ] Implement methods (native or external)
- [ ] Register external models (if applicable)
- [ ] Test with `dyn_cast<Interface>(...)`

### Using Python Bindings

- [ ] Set `PYTHONPATH` and `LD_LIBRARY_PATH`
- [ ] Run discovery script: `./discover_dialect_api.sh <dialect>`
- [ ] Read test files for examples
- [ ] Check C++ bindings for missing APIs
- [ ] Use `dir()` for runtime inspection

---

## 📅 Last Updated

This index was created alongside the tutorial materials and reflects the structure of MLIR as of January 2025.

---

## 🙏 Credits

These documents are based on:
- Official MLIR documentation
- Source code analysis of LLVM/MLIR
- Real-world implementation examples
- Test suite examples

---

## Happy MLIR Development! 🚀
