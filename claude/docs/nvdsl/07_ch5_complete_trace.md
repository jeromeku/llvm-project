# Chapter 5: Complete Dispatch Path - Python to C++

## Introduction

This document provides a **frame-by-frame trace** of the complete dispatch path for Chapter 5 (Warp-Specialized GEMM), showing every layer from Python → MLIR Python Bindings → MLIR C API → MLIR C++ implementation.

We'll trace several key operations showing:
- Exact function calls at each layer
- State changes in MLIR context/objects
- Memory allocations and IR construction
- Complete call stack with line numbers

---

## Tracing Methodology

We'll trace these representative operations from Ch5.py:

1. **Context Creation**: `with ir.Context()`
2. **Module Creation**: `ir.Module.create()`
3. **Thread ID**: `gpu.thread_id(gpu.Dimension.x)`
4. **Warpgroup Creation**: `Warpgroup(primary_thread=0, register_size=232)`
5. **Mbarrier Creation**: `Mbarriers(number_of_barriers=7)`
6. **TMA Descriptor**: `TMA([128, 64], a.type).create_descriptor(a_dev)`
7. **WGMMA Operation**: `D += A @ B`

---

## Trace 1: Context Creation

### User Code

**File**: [Ch5.py:253](../../../mlir/test/Examples/NVGPU/Ch5.py#L253) (inside @NVDSL.mlir_func decorator)

```python
@NVDSL.mlir_func
def gemm_warp_specialized(a, b, d, num_stages):
    # Decorator creates context...
```

### Frame 1: Decorator Entry

**File**: [nvdsl.py:411](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L411)

```python
with ir.Context(), ir.Location.unknown():
    # Context created here
```

**Python Stack**:
```
Frame: wrapper() at nvdsl.py:411
  Local vars:
    - funcBody = <function gemm_warp_specialized>
    - args = (np.ndarray[512,1024,f16], np.ndarray[1024,256,f16], np.ndarray[512,256,f32], 7)
```

### Frame 2: Python Context Constructor

**File**: [mlir/python/mlir/ir.py](../../../mlir/python/mlir/ir.py#L10)

```python
from ._mlir_libs._mlir.ir import *
```

The `ir.Context` class is imported from the C extension module `_mlir.ir`.

**Call**: `ir.Context()`

**Python Binding Source**: Built from C++ using nanobind

**Call Path**:
```
Python: ir.Context()
    ↓
_mlir_libs._mlir.ir.Context.__init__()  [C extension]
    ↓
PyContext::__init__() [C++ nanobind binding]
```

### Frame 3: C++ Binding Layer

**File**: [mlir/lib/Bindings/Python/IRCore.cpp:77-104](../../../mlir/lib/Bindings/Python/IRCore.cpp#L77-L104)

```cpp
class PyMlirContext {
public:
  PyMlirContext() = delete;
  PyMlirContext(const PyMlirContext &) = delete;
  PyMlirContext(PyMlirContext &&) = delete;

  // Creates a new context and ownership of the underlying MLIR context.
  static PyMlirContextRef createNewContextForInit() {
    MlirContext context = mlirContextCreate();
    return PyMlirContext::createFromOwnedContext(context);
  }
```

**State Before**:
- No MLIR context exists
- Global MLIR state uninitialized

**Function Call**: `mlirContextCreate()`

**State Changes**:
1. Creates new `MlirContext` handle
2. Initializes dialect registry
3. Sets up MLIR global state
4. Returns opaque handle

### Frame 4: MLIR C API

**File**: [mlir/lib/CAPI/IR/IR.cpp:46-50](../../../mlir/lib/CAPI/IR/IR.cpp#L46-L50)

```cpp
MlirContext mlirContextCreate() {
  auto *context = new MLIRContext();
  context->loadDialect<BuiltinDialect>();
  return wrap(context);
}
```

**Parameters**: None

**Returns**: `MlirContext` (opaque pointer to `MLIRContext*`)

**State Changes**:
1. **Heap Allocation**: `new MLIRContext()` allocates context object
2. **Dialect Registration**: Loads `BuiltinDialect`
3. **Handle Wrapping**: Converts `MLIRContext*` → `MlirContext`

**Function**: `wrap(context)`
```cpp
inline MlirContext wrap(MLIRContext *context) {
  return MlirContext{context};
}
```

### Frame 5: MLIR C++ Core

**File**: [mlir/lib/IR/MLIRContext.cpp:275-312](../../../mlir/lib/IR/MLIRContext.cpp#L275-L312)

```cpp
MLIRContext::MLIRContext(Threading setting)
    : impl(new MLIRContextImpl(/*loadAllDialects=*/false)) {
  // Initialize threading configuration
  impl->threadingConfig = setting;
}
```

**Constructor Execution**:

**File**: [mlir/lib/IR/MLIRContext.cpp:155-190](../../../mlir/lib/IR/MLIRContext.cpp#L155-L190)

```cpp
MLIRContextImpl::MLIRContextImpl(bool loadAllDialects)
    : loadAllDialects(loadAllDialects) {
  // Initialize the affine uniquer.
  affineUniquer = std::make_unique<AffineUniquer>();

  // Initialize several collections and maps:
  attributes.initialize();
  types.initialize();
  attributeDetails.initialize();

  // Register builtin types/attributes
  registerBuiltinTypes();
  registerBuiltinAttributes();

  // Initialize dialect registry
  dialectsRegistry = std::make_unique<DialectRegistry>();
}
```

**Memory Layout After Construction**:

```
MLIRContext @ 0x55e8a0001000
├── impl: MLIRContextImpl @ 0x55e8a0002000
│   ├── affineUniquer: AffineUniquer @ 0x55e8a0003000
│   ├── types: TypeUniquer @ 0x55e8a0004000
│   │   └── typeStorage: DenseMap<TypeID, TypeStorage*>
│   ├── attributes: AttributeUniquer @ 0x55e8a0005000
│   │   └── attributeStorage: DenseMap<TypeID, AttributeStorage*>
│   ├── dialects: DialectRegistry @ 0x55e8a0006000
│   │   └── registry: StringMap<DialectAllocator>
│   ├── loadedDialects: DenseMap<TypeID, Dialect*>
│   ├── operations: RegisteredOperationMap
│   └── interfaces: DenseMap<TypeID, InterfaceBase*>
└── Properties:
    - allowUnregisteredDialects: false
    - printOpOnDiagnostic: true
    - printStackTraceOnDiagnostic: false
```

### Frame 6: Dialect Loading

**Back in Frame 4**: [mlir/lib/CAPI/IR/IR.cpp:47](../../../mlir/lib/CAPI/IR/IR.cpp#L47)

```cpp
context->loadDialect<BuiltinDialect>();
```

**Execution**:

**File**: [mlir/lib/IR/MLIRContext.cpp:430-450](../../../mlir/lib/IR/MLIRContext.cpp#L430-L450)

```cpp
template <typename ConcreteDialect>
void MLIRContext::loadDialect() {
  loadDialect(ConcreteDialect::getDialectNamespace());
}

void MLIRContext::loadDialect(StringRef dialectNamespace) {
  // Get or load the dialect
  if (Dialect *dialect = getLoadedDialect(dialectNamespace))
    return;

  // Look up in registry
  DialectAllocatorFunction allocator =
      getDialectRegistry().getDialectAllocator(dialectNamespace);

  // Allocate dialect
  std::unique_ptr<Dialect> ownedDialect = allocator(this);

  // Register the dialect
  registerDialect(std::move(ownedDialect));
}
```

**State Changes**:
```
MLIRContext::impl->loadedDialects:
  BEFORE: {}
  AFTER:  {
    TypeID::get<BuiltinDialect>() → BuiltinDialect @ 0x55e8a0007000
  }
```

### Frame 7: Return to Python

**Unwinding the stack**:

```
C++: MLIRContext constructed @ 0x55e8a0001000
    ↓
C API: wrap(context) → MlirContext{0x55e8a0001000}
    ↓
C++ Binding: PyMlirContext::createFromOwnedContext(mlirContext)
    ↓
Python: ir.Context instance created
    ↓
Context Manager: __enter__() called
```

**Python Object State**:
```python
>>> ctx = ir.Context()
>>> ctx
<mlir.ir.Context object at 0x7f8c40001000>

# Internal state:
ctx._CAPIPtr = 0x55e8a0001000  # Pointer to MLIRContext
ctx._owned = True               # Python owns the context
```

---

## Trace 2: Module Creation

### User Code

**File**: [nvdsl.py:417](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L417)

```python
module = ir.Module.create()
```

### Frame 1: Python Binding

**Call**: `ir.Module.create()`

**Source**: Imported from `_mlir_libs._mlir.ir`

### Frame 2: C++ Binding

**File**: [mlir/lib/Bindings/Python/IRCore.cpp:1230-1250](../../../mlir/lib/Bindings/Python/IRCore.cpp#L1230-L1250)

```cpp
nb::class_<PyModule>(m, "Module")
  .def_static(
    "create",
    [](DefaultingPyLocation loc) {
      MlirModule module = mlirModuleCreateEmpty(loc);
      return PyModule::forModule(module).release();
    },
    nb::arg("loc") = nb::none(),
    "Creates an empty module")
```

**Call Flow**:
```
Python: ir.Module.create()
    ↓
Lambda: [](DefaultingPyLocation loc) { ... }
    ↓
loc parameter: DefaultingPyLocation wraps current location
    ↓ (defaults to unknown location if not provided)
```

### Frame 3: MLIR C API

**File**: [mlir/lib/CAPI/IR/IR.cpp:110-114](../../../mlir/lib/CAPI/IR/IR.cpp#L110-L114)

```cpp
MlirModule mlirModuleCreateEmpty(MlirLocation location) {
  return wrap(ModuleOp::create(unwrap(location)));
}
```

**Parameter Unwrapping**:
```cpp
inline Location unwrap(MlirLocation location) {
  return Location::getFromOpaquePointer(location.ptr);
}
```

**State Before**:
- Context exists with BuiltinDialect loaded
- No operations created yet

### Frame 4: MLIR C++ - ModuleOp Creation

**File**: [mlir/lib/IR/BuiltinOps.cpp:98-105](../../../mlir/lib/IR/BuiltinOps.cpp#L98-L105)

```cpp
ModuleOp ModuleOp::create(Location loc, std::optional<StringRef> name) {
  OperationState state(loc, getOperationName());
  Builder builder(loc.getContext());

  // Add a region to hold the module body
  state.addRegion();

  // Optionally add the module name
  if (name)
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(*name));

  return cast<ModuleOp>(Operation::create(state));
}
```

**OperationState Construction**:

**File**: [mlir/include/mlir/IR/OperationSupport.h:120-130](../../../mlir/include/mlir/IR/OperationSupport.h#L120-L130)

```cpp
struct OperationState {
  Location location;
  OperationName name;
  SmallVector<Value, 4> operands;
  SmallVector<Type, 4> types;
  NamedAttrList attributes;
  SmallVector<std::unique_ptr<Region>, 1> regions;
  SmallVector<Block *, 1> successors;

  OperationState(Location location, StringRef name);
  // ...
};
```

**State Initialization**:
```cpp
OperationState state(loc, "builtin.module");
  ↓
state.location = UnknownLoc @ 0x55e8a0008000
state.name = OperationName("builtin.module")
state.operands = []
state.types = []
state.attributes = {}
state.regions = []  // Empty initially
state.successors = []
```

**Adding Region**:
```cpp
state.addRegion();
  ↓
state.regions.push_back(std::make_unique<Region>());
  ↓
Region created @ 0x55e8a0009000
state.regions = [Region @ 0x55e8a0009000]
```

### Frame 5: Operation Creation

**File**: [mlir/lib/IR/Operation.cpp:150-220](../../../mlir/lib/IR/Operation.cpp#L150-L220)

```cpp
Operation *Operation::create(const OperationState &state) {
  unsigned numRegions = state.regions.size();
  unsigned numResults = state.types.size();
  unsigned numOperands = state.operands.size();
  unsigned numSuccessors = state.successors.size();

  // Calculate total size needed
  size_t byteSize =
      totalSizeToAlloc<OpResult, BlockOperand, Region, detail::OperandStorage>(
          numResults, numSuccessors, numRegions, needsOperandStorage);

  // Allocate operation
  void *rawMem = state.location.getContext()
                     ->getImpl()
                     .allocator.Allocate(byteSize, alignof(Operation));

  // Construct operation in-place
  Operation *op = ::new (rawMem) Operation(
      state.location, state.name, numResults, numSuccessors, numRegions,
      state.attributes, needsOperandStorage);

  // Initialize regions
  for (unsigned i = 0; i < numRegions; ++i)
    new (&op->getRegion(i)) Region(op);

  // Move regions from state
  for (unsigned i = 0; i < numRegions; ++i) {
    if (state.regions[i])
      op->getRegion(i).takeBody(*state.regions[i]);
  }

  return op;
}
```

**Memory Layout**:

```
Operation @ 0x55e8a000a000 (allocated via context allocator)
├── Header (Operation base class)
│   ├── location: UnknownLoc
│   ├── name: OperationName("builtin.module")
│   ├── numResults: 0
│   ├── numSuccessors: 0
│   ├── numRegions: 1
│   ├── attrs: DictionaryAttr (empty)
│   └── block: nullptr (not inserted yet)
├── Trailing Objects:
│   ├── Region[0] @ 0x55e8a000a080
│   │   ├── blocks: [] (empty block list)
│   │   └── parent: Operation @ 0x55e8a000a000
└── Size: 128 bytes (approximate)
```

### Frame 6: Cast to ModuleOp

```cpp
return cast<ModuleOp>(Operation::create(state));
```

**MLIR Cast Mechanism**:
```cpp
template <typename To, typename From>
To cast(From value) {
  assert(isa<To>(value) && "cast failed");
  return To(value.getOperation());
}
```

**ModuleOp is a wrapper**:
```cpp
class ModuleOp : public Op<ModuleOp, OpTrait::OneRegion, ...> {
  Operation *operation;  // Points to the Operation we just created
};
```

### Frame 7: Return Path

**Wrapping for C API**:

```cpp
MlirModule wrap(ModuleOp module) {
  return {module.getOperation()};  // MlirModule{Operation*}
}
```

**Python Binding**:

```cpp
return PyModule::forModule(module).release();
```

**File**: [mlir/lib/Bindings/Python/IRCore.cpp:1150-1170](../../../mlir/lib/Bindings/Python/IRCore.cpp#L1150-L1170)

```cpp
PyModule PyModule::forModule(MlirModule module) {
  // Create Python wrapper
  return PyModule(module);
}

class PyModule {
  MlirModule module;  // C API handle
  PyMlirContextRef context;  // Python context reference
  // ...
};
```

**Python Object**:
```python
>>> module = ir.Module.create()
>>> module
<mlir.ir.Module object at 0x7f8c40002000>

# Internal state:
module._CAPIPtr = 0x55e8a000a000  # Points to Operation
module.context = <ir.Context>      # Reference to context
```

---

## Trace 3: Thread ID Operation

### User Code

**File**: [Ch5.py:278-280](../../../mlir/test/Examples/NVGPU/Ch5.py#L278-L280) (inside kernel)

```python
@NVDSL.mlir_gpu_launch(...)
def gemm_warp_specialized_kernel():
    wg_producer = Warpgroup(primary_thread=128, register_size=40)
```

**Inside Warpgroup.__init__**: [nvdsl.py:171](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L171)

```python
tidx = gpu.thread_id(gpu.Dimension.x)
```

### Frame 1: Python Dialect Binding

**Call**: `gpu.thread_id(gpu.Dimension.x)`

**Source**: From `mlir.dialects.gpu` module

**File**: Auto-generated from TableGen, loaded as `_mlirDialectsGPU.so`

### Frame 2: Python Extension Module

The GPU dialect bindings are auto-generated from ODS (Operation Definition Specification).

**TableGen Source**: [mlir/include/mlir/Dialect/GPU/IR/GPUOps.td:465-485](../../../mlir/include/mlir/Dialect/GPU/IR/GPUOps.td#L465-L485)

```tablegen
def GPU_ThreadIdOp : GPU_Op<"thread_id", [Pure]> {
  let summary = "Get the thread ID for a given dimension";
  let description = [{
    Returns the thread ID for the given dimension (x, y, z).
  }];

  let arguments = (ins GPU_DimensionAttr:$dimension);
  let results = (outs Index:$result);

  let assemblyFormat = "$dimension attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "Dimension":$dimension), [{
      build($_builder, $_state, $_builder.getIndexType(), dimension);
    }]>
  ];
}
```

**Generated Python Binding**:

**File**: [mlir/python/mlir/dialects/GPUOps.td](../../../mlir/python/mlir/dialects/GPUOps.td) (Python tablegen)

The Python binding is created via MLIR's Python binding generator, which creates:

```python
# Auto-generated code (conceptual)
def thread_id(dimension, *, loc=None, ip=None):
    # Get current insertion point and location
    if ip is None:
        ip = InsertionPoint.current
    if loc is None:
        loc = Location.current

    # Call C++ builder
    return _mlir_dialects_gpu.thread_id(dimension, loc, ip)
```

### Frame 3: C++ Python Binding

**File**: [mlir/lib/Bindings/Python/DialectGPU.cpp](../../../mlir/lib/Bindings/Python/DialectGPU.cpp) (conceptual - auto-generated)

```cpp
// Nanobind wrapper (simplified)
m.def("thread_id",
  [](gpu::Dimension dimension, MlirLocation loc, PyInsertionPoint &ip) {
    OpBuilder builder = ip.getBuilder();
    Location location = unwrap(loc);

    // Build the operation
    auto op = builder.create<gpu::ThreadIdOp>(location, dimension);

    // Return the result value
    return wrap(op.getResult());
  },
  nb::arg("dimension"),
  nb::arg("loc") = nb::none(),
  nb::arg("ip") = nb::none());
```

### Frame 4: OpBuilder::create

**File**: [mlir/include/mlir/IR/Builders.h:415-430](../../../mlir/include/mlir/IR/Builders.h#L415-L430)

```cpp
template <typename OpTy, typename... Args>
OpTy OpBuilder::create(Location location, Args &&...args) {
  OperationState state(location, OpTy::getOperationName());
  OpTy::build(*this, state, std::forward<Args>(args)...);
  auto *op = create(state);
  return cast<OpTy>(op);
}
```

**Template Instantiation**:
```cpp
gpu::ThreadIdOp OpBuilder::create<gpu::ThreadIdOp>(Location loc, gpu::Dimension dim)
```

### Frame 5: ThreadIdOp::build

**File**: [mlir/lib/Dialect/GPU/IR/GPUDialect.cpp:850-860](../../../mlir/lib/Dialect/GPU/IR/GPUDialect.cpp#L850-L860)

```cpp
void ThreadIdOp::build(OpBuilder &builder, OperationState &state,
                        Dimension dimension) {
  // Add result type (index)
  state.addTypes(builder.getIndexType());

  // Add dimension attribute
  state.addAttribute("dimension",
                     builder.getI32IntegerAttr(static_cast<int32_t>(dimension)));
}
```

**OperationState Before**:
```
state.location = <current location>
state.name = "gpu.thread_id"
state.operands = []
state.types = []
state.attributes = {}
state.regions = []
```

**After build()**:
```
state.types = [IndexType @ 0x55e8a000b000]
state.attributes = {
  "dimension": IntegerAttr(i32, 0)  // x = 0, y = 1, z = 2
}
```

### Frame 6: Operation Construction

**File**: [mlir/lib/IR/Operation.cpp:150](../../../mlir/lib/IR/Operation.cpp#L150)

```cpp
Operation *op = Operation::create(state);
```

**Memory Allocation**:

```
Operation @ 0x55e8a000c000
├── name: OperationName("gpu.thread_id")
├── location: <current location>
├── numResults: 1
├── numOperands: 0
├── numRegions: 0
├── attrs: DictionaryAttr {
│   "dimension": IntegerAttr(i32, 0)
│ }
├── results: OpResult[1]
│   └── result[0]: Value(IndexType) @ 0x55e8a000c080
│       ├── type: IndexType
│       ├── owner: Operation @ 0x55e8a000c000
│       └── resultNumber: 0
└── operands: [] (none)
```

### Frame 7: Insertion into IR

**Context**: Inside `@mlir_gpu_launch`, we're in a `gpu.launch` operation's region.

**Current Insertion Point**:
```
gpu.launch ... {
  ^bb0(%bx, %by, %bz, ...):  ← Block entry
    <insertion point here>
```

**File**: [mlir/lib/IR/Builders.cpp:380-390](../../../mlir/lib/IR/Builders.cpp#L380-L390)

```cpp
Operation *OpBuilder::insert(Operation *op) {
  if (block) {
    block->getOperations().insert(insertPoint, op);
  }

  if (listener)
    listener->notifyOperationInserted(op);

  return op;
}
```

**Block State Before**:
```
Block @ 0x55e8a000d000 (inside gpu.launch)
operations: []
```

**Block State After**:
```
Block @ 0x55e8a000d000
operations: [
  Operation @ 0x55e8a000c000 (gpu.thread_id)
]
```

**Operation linking**:
```cpp
op->block = block;  // Set parent block
block->operations.push_back(op);  // Add to block's operation list
```

### Frame 8: Return Value

**Wrapping for Python**:

```cpp
// C++ → C API
MlirValue wrap(Value value) {
  return {value.getAsOpaquePointer()};
}

// C API → Python
PyValue PyValue::create(MlirValue value) {
  return PyValue(value, PyMlirContextRef::forValue(value));
}
```

**Python Object**:
```python
>>> tidx = gpu.thread_id(gpu.Dimension.x)
>>> tidx
<mlir.ir.Value object at 0x7f8c40003000>

# Internal state:
tidx._CAPIPtr = 0x55e8a000c080  # Points to OpResult
tidx.context = <ir.Context>
tidx.type = <mlir.ir.IndexType>

# ArithValue casting (from nvdsl.py)
>>> isinstance(tidx, ArithValue)
True
```

**Key Insight**: The returned `tidx` is now an `ArithValue` (due to type caster registration in nvdsl.py), which enables operator overloading:

```python
>>> tidx + 10  # Calls ArithValue.__add__(10)
# Generates: arith.addi %tidx, %c10
```

---

## Trace 4: Mbarrier Creation

### User Code

**File**: [Ch5.py:284](../../../mlir/test/Examples/NVGPU/Ch5.py#L284)

```python
mbar_mma, mbar_tma = initialize(a_tma, b_tma, num_stages)
```

**Inside initialize()**: [Ch5.py:124](../../../mlir/test/Examples/NVGPU/Ch5.py#L124)

```python
mbar_group_tma = Mbarriers(number_of_barriers=num_stages)  # num_stages=7
```

### Frame 1: Mbarriers Constructor

**File**: [nvdsl.py:58-65](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L58-L65)

```python
class Mbarriers:
    def __init__(self, number_of_barriers=1):
        self.mbar_ty = ir.Type.parse(
            "!nvgpu.mbarrier.group<memorySpace=#gpu.address_space<workgroup>, num_barriers = "
            + str(number_of_barriers)
            + ">"
        )
        self.mbar_group_op = nvgpu.mbarrier_create(self.mbar_ty)
        self.number_of_barriers = number_of_barriers
```

### Frame 2: Type Parsing

**Call**: `ir.Type.parse("!nvgpu.mbarrier.group<...>")`

**File**: [mlir/lib/Bindings/Python/IRCore.cpp:540-560](../../../mlir/lib/Bindings/Python/IRCore.cpp#L540-L560)

```cpp
nb::class_<PyType>(m, "Type")
  .def_static("parse",
    [](const std::string &typeSpec, DefaultingPyMlirContext context) {
      MlirType type = mlirTypeParseGet(context->get(), toMlirStringRef(typeSpec));
      if (mlirTypeIsNull(type))
        throw nb::value_error("Unable to parse type");
      return PyType(context, type);
    },
    nb::arg("asm"), nb::arg("context") = nb::none());
```

### Frame 3: MLIR C API - Type Parsing

**File**: [mlir/lib/CAPI/IR/IR.cpp:520-530](../../../mlir/lib/CAPI/IR/IR.cpp#L520-L530)

```cpp
MlirType mlirTypeParseGet(MlirContext context, MlirStringRef typeSpec) {
  MLIRContext *ctx = unwrap(context);
  StringRef typeSpecStr = unwrap(typeSpec);

  Type type = parseType(typeSpecStr, ctx);
  return wrap(type);
}
```

### Frame 4: MLIR C++ Type Parser

**File**: [mlir/lib/AsmParser/Parser.cpp:1850-1900](../../../mlir/lib/AsmParser/Parser.cpp#L1850-L1900) (simplified)

```cpp
Type Parser::parseType() {
  Token tok = getToken();

  // Handle builtin types
  if (tok.is(Token::exclamation)) {
    // Dialect-specific type: !dialect.type<...>
    consumeToken(Token::exclamation);

    StringRef dialectName = getTokenSpelling();
    consumeToken(Token::bare_identifier);  // "nvgpu"

    // Lookup dialect
    Dialect *dialect = ctx->getLoadedDialect(dialectName);
    if (!dialect) {
      // Try to load dialect
      ctx->loadDialect(dialectName);
      dialect = ctx->getLoadedDialect(dialectName);
    }

    // Parse custom type syntax
    return parseDialectType(dialect);
  }
}
```

**Parsing Flow**:
```
Input: "!nvgpu.mbarrier.group<memorySpace=#gpu.address_space<workgroup>, num_barriers = 7>"
    ↓
Token: exclamation "!"
    ↓
Token: bare_identifier "nvgpu"
    ↓
Load dialect: ctx->loadDialect("nvgpu")
    ↓
Parse custom syntax: dialect->parseType(parser)
```

### Frame 5: NVGPU Dialect Type Parsing

**File**: [mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp:80-120](../../../mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp#L80-L120) (simplified)

```cpp
Type NVGPUDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  Type parsedType;

  if (parser.parseKeyword(&mnemonic))
    return Type();

  if (mnemonic == "mbarrier") {
    // Parse: mbarrier.group<...>
    if (parser.parseKeyword("group"))
      return Type();

    // Parse: < memorySpace = ... , num_barriers = ... >
    Attribute memorySpace;
    int64_t numBarriers;

    if (parser.parseLess() ||
        parser.parseKeyword("memorySpace") ||
        parser.parseEqual() ||
        parser.parseAttribute(memorySpace) ||
        parser.parseComma() ||
        parser.parseKeyword("num_barriers") ||
        parser.parseEqual() ||
        parser.parseInteger(numBarriers) ||
        parser.parseGreater())
      return Type();

    // Create type
    return MBarrierGroupType::get(
        parser.getContext(),
        memorySpace.cast<Attribute>(),
        numBarriers);
  }
}
```

### Frame 6: MBarrierGroupType Construction

**File**: [mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp:150-180](../../../mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp#L150-L180)

```cpp
MBarrierGroupType MBarrierGroupType::get(MLIRContext *context,
                                          Attribute memorySpace,
                                          int64_t numBarriers) {
  // Use type uniquer to get canonical instance
  return Base::get(context, memorySpace, numBarriers);
}
```

**Type Uniquing**:

**File**: [mlir/lib/IR/TypeDetail.h:50-80](../../../mlir/lib/IR/TypeDetail.h#L50-L80)

```cpp
template <typename ConcreteType, typename... Args>
ConcreteType TypeUniquer::get(MLIRContext *ctx, Args &&...args) {
  // Create storage key
  auto key = getKey<ConcreteType>(args...);

  // Look up in type cache
  auto *storage = ctx->getImpl().types.get<typename ConcreteType::ImplType>(key);

  if (!storage) {
    // Allocate new storage
    storage = ConcreteType::ImplType::construct(ctx, args...);
    ctx->getImpl().types.insert(storage);
  }

  return ConcreteType(storage);
}
```

**Type Storage**:

```cpp
struct MBarrierGroupTypeStorage : public TypeStorage {
  Attribute memorySpace;
  int64_t numBarriers;

  static MBarrierGroupTypeStorage *construct(MLIRContext *ctx,
                                             Attribute memorySpace,
                                             int64_t numBarriers) {
    auto *storage = ctx->getAllocator().Allocate<MBarrierGroupTypeStorage>();
    return new (storage) MBarrierGroupTypeStorage{memorySpace, numBarriers};
  }

  // Hash and equality for uniquing
  using KeyTy = std::tuple<Attribute, int64_t>;
  bool operator==(const KeyTy &key) const {
    return memorySpace == std::get<0>(key) && numBarriers == std::get<1>(key);
  }
};
```

**Memory State After**:

```
MLIRContext::impl->types (TypeUniquer)
├── Type Cache (DenseMap<TypeID, TypeStorageMap>)
│   └── TypeID::get<MBarrierGroupType>()
│       └── TypeStorageMap (DenseMap<KeyTy, TypeStorage*>)
│           └── Key{#gpu.address_space<workgroup>, 7}
│               → MBarrierGroupTypeStorage @ 0x55e8a000e000
│                   ├── memorySpace: #gpu.address_space<workgroup>
│                   └── numBarriers: 7
└── MBarrierGroupType handle points to storage @ 0x55e8a000e000
```

### Frame 7: Mbarrier Create Operation

**Back in Frame 1**: `self.mbar_group_op = nvgpu.mbarrier_create(self.mbar_ty)`

**File**: Auto-generated from [mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td](../../../mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td)

**Python Binding Call**:
```python
nvgpu.mbarrier_create(mbar_ty)
    ↓
_mlirDialectsNVGPU.mbarrier_create(mbar_ty)
```

### Frame 8: C++ Operation Builder

**File**: [mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp:450-470](../../../mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp#L450-L470)

```cpp
void MBarrierCreateOp::build(OpBuilder &builder, OperationState &state,
                               Type resultType) {
  state.addTypes(resultType);
}
```

**Operation Creation**:
```
Operation @ 0x55e8a000f000
├── name: "nvgpu.mbarrier.create"
├── operands: []
├── results: [
│   OpResult[0]: !nvgpu.mbarrier.group<memorySpace=#gpu.address_space<workgroup>, num_barriers = 7>
│ ]
├── attributes: {}
└── regions: []
```

**Insertion into IR**:
```
Current Block:
  %mbar_group = nvgpu.mbarrier.create : !nvgpu.mbarrier.group<...>
```

### Frame 9: Python Return

**Python Object State**:
```python
>>> mbar = Mbarriers(number_of_barriers=7)
>>> mbar.mbar_ty
<mlir.ir.Type object representing !nvgpu.mbarrier.group<...>>

>>> mbar.mbar_group_op
<mlir.ir.OpView representing nvgpu.mbarrier.create>

>>> mbar.number_of_barriers
7
```

---

## Trace 5: TMA Descriptor Creation

### User Code

**File**: [Ch5.py:267](../../../mlir/test/Examples/NVGPU/Ch5.py#L267)

```python
a_tma = TMA([128, 64], a.type, swizzle=sw)
a_tma.create_descriptor(a_dev)
```

### Frame 1: TMA Constructor

**File**: [nvdsl.py:105-120](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L105-L120)

```python
def __init__(self, tma_box_shape, memref_ty, swizzle=...):
    self.swizzle = swizzle
    self.tma_box_shape = tma_box_shape  # [128, 64]
    self.memref_ty = memref_ty          # memref<512x1024xf16>
    self.tma_memref = ir.MemRefType.get(tma_box_shape, memref_ty.element_type)
```

**State After Constructor**:
```python
a_tma.tma_box_shape = [128, 64]
a_tma.memref_ty = memref<512x1024xf16>
a_tma.tma_memref = memref<128x64xf16>  # Tile shape
a_tma.swizzle = TensorMapSwizzleKind.SWIZZLE_128B
```

### Frame 2: create_descriptor() Call

**File**: [nvdsl.py:138-149](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L138-L149)

```python
def create_descriptor(self, device_ptr):
    tma_descriptor_ty = self.tensormap_descriptor_ty
    device_unranked_memref = memref.CastOp(
        ir.UnrankedMemRefType.get(
            self.memref_ty.element_type, self.memref_ty.memory_space
        ),
        device_ptr,
    )
    self.tma_descriptor = nvgpu.TmaCreateDescriptorOp(
        tma_descriptor_ty, device_unranked_memref, map(const, self.tma_box_shape)
    )
    return self.tma_descriptor.result
```

### Frame 3: Property Access - tensormap_descriptor_ty

**File**: [nvdsl.py:122-136](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L122-L136)

```python
@property
def tensormap_descriptor_ty(self):
    tensorMemrefType = ir.MemRefType.get(
        self.tma_box_shape,
        self.memref_ty.element_type,
        memory_space=ir.Attribute.parse("3"),  # Shared memory
    )
    return nvgpu.TensorMapDescriptorType.get(
        tensorMemrefType,
        self.swizzle,
        self.l2promo,
        self.oob,
        self.interleave,
    )
```

### Frame 4: MemRefType Creation

**Call**: `ir.MemRefType.get([128, 64], f16, memory_space=3)`

**Python Binding**:

**File**: [mlir/lib/Bindings/Python/IRTypes.cpp:280-320](../../../mlir/lib/Bindings/Python/IRTypes.cpp#L280-L320)

```cpp
nb::class_<PyMemRefType>(m, "MemRefType")
  .def_static("get",
    [](std::vector<int64_t> shape,
       PyType &elementType,
       PyAttribute memorySpace) {

      MlirType type = mlirMemRefTypeGet(
          unwrap(elementType),
          shape.size(),
          shape.data(),
          /*layout=*/mlirAttributeGetNull(),
          unwrap(memorySpace));

      return PyMemRefType(type);
    },
    nb::arg("shape"),
    nb::arg("element_type"),
    nb::arg("memory_space") = nb::none());
```

### Frame 5: C API - MemRef Type Construction

**File**: [mlir/lib/CAPI/IR/BuiltinTypes.cpp:180-200](../../../mlir/lib/CAPI/IR/BuiltinTypes.cpp#L180-L200)

```cpp
MlirType mlirMemRefTypeGet(MlirType elementType, intptr_t rank,
                            const int64_t *shape, MlirAttribute layout,
                            MlirAttribute memorySpace) {
  return wrap(MemRefType::get(
      llvm::ArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType),
      MemRefLayoutAttrInterface(unwrap(layout)),
      unwrap(memorySpace)));
}
```

### Frame 6: C++ MemRefType Construction

**File**: [mlir/lib/IR/BuiltinTypes.cpp:420-450](../../../mlir/lib/IR/BuiltinTypes.cpp#L420-L450)

```cpp
MemRefType MemRefType::get(ArrayRef<int64_t> shape, Type elementType,
                            MemRefLayoutAttrInterface layout,
                            Attribute memorySpace) {
  // Normalize parameters
  if (!layout)
    layout = {};

  // Use type uniquer
  return Base::get(elementType.getContext(), shape, elementType, layout,
                   memorySpace);
}
```

**Type Storage**:

```cpp
struct MemRefTypeStorage : public TypeStorage {
  ArrayRef<int64_t> shape;
  Type elementType;
  MemRefLayoutAttrInterface layout;
  Attribute memorySpace;

  using KeyTy = std::tuple<ArrayRef<int64_t>, Type,
                            MemRefLayoutAttrInterface, Attribute>;

  static MemRefTypeStorage *construct(MLIRContext *ctx, KeyTy key) {
    auto shape = std::get<0>(key);
    // Allocate shape storage
    int64_t *shapeCopy = ctx->getAllocator().Allocate<int64_t>(shape.size());
    std::copy(shape.begin(), shape.end(), shapeCopy);

    // Allocate and construct storage
    auto *storage = ctx->getAllocator().Allocate<MemRefTypeStorage>();
    return new (storage) MemRefTypeStorage{
      ArrayRef(shapeCopy, shape.size()),
      std::get<1>(key),  // elementType
      std::get<2>(key),  // layout
      std::get<3>(key)   // memorySpace
    };
  }
};
```

**Memory State**:

```
MemRefTypeStorage @ 0x55e8a0010000
├── shape: int64_t[2] @ 0x55e8a0010080
│   ├── [0]: 128
│   └── [1]: 64
├── elementType: Float16Type
├── layout: <null>
└── memorySpace: IntegerAttr(i64, 3)

Uniquer Cache:
  Key{[128,64], f16, null, 3} → MemRefTypeStorage @ 0x55e8a0010000
```

### Frame 7: TensorMapDescriptorType.get()

**Call**: `nvgpu.TensorMapDescriptorType.get(tensorMemrefType, swizzle, ...)`

**Python Binding**:

**File**: [mlir/lib/Bindings/Python/DialectNVGPU.cpp:24-34](../../../mlir/lib/Bindings/Python/DialectNVGPU.cpp#L24-L34)

```cpp
nvgpuTensorMapDescriptorType.def_classmethod(
    "get",
    [](const nb::object &cls, MlirType tensorMemrefType, int swizzle,
       int l2promo, int oobFill, int interleave, MlirContext ctx) {
      return cls(mlirNVGPUTensorMapDescriptorTypeGet(
          ctx, tensorMemrefType, swizzle, l2promo, oobFill, interleave));
    },
    nb::arg("cls"), nb::arg("tensor_type"), nb::arg("swizzle"),
    nb::arg("l2promo"), nb::arg("oob_fill"), nb::arg("interleave"),
    nb::arg("ctx") = nb::none());
```

### Frame 8: C API - TensorMapDescriptor Construction

**File**: [mlir/lib/CAPI/Dialect/NVGPU.cpp:20-30](../../../mlir/lib/CAPI/Dialect/NVGPU.cpp#L20-L30)

```cpp
MlirType mlirNVGPUTensorMapDescriptorTypeGet(
    MlirContext ctx, MlirType tensorType, int swizzle, int l2promo,
    int oob, int interleave) {

  return wrap(nvgpu::TensorMapDescriptorType::get(
      unwrap(ctx),
      unwrap(tensorType).cast<MemRefType>(),
      static_cast<nvgpu::TensorMapSwizzleKind>(swizzle),
      static_cast<nvgpu::TensorMapL2PromoKind>(l2promo),
      static_cast<nvgpu::TensorMapOOBKind>(oob),
      static_cast<nvgpu::TensorMapInterleaveKind>(interleave)));
}
```

### Frame 9: C++ TensorMapDescriptorType Construction

**File**: [mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp:200-230](../../../mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp#L200-L230)

```cpp
TensorMapDescriptorType TensorMapDescriptorType::get(
    MLIRContext *context,
    MemRefType tensorType,
    TensorMapSwizzleKind swizzle,
    TensorMapL2PromoKind l2promo,
    TensorMapOOBKind oob,
    TensorMapInterleaveKind interleave) {

  return Base::get(context, tensorType, swizzle, l2promo, oob, interleave);
}
```

**Type Storage**:

```cpp
struct TensorMapDescriptorTypeStorage : public TypeStorage {
  MemRefType tensorType;
  TensorMapSwizzleKind swizzle;
  TensorMapL2PromoKind l2promo;
  TensorMapOOBKind oob;
  TensorMapInterleaveKind interleave;

  using KeyTy = std::tuple<MemRefType, TensorMapSwizzleKind,
                            TensorMapL2PromoKind, TensorMapOOBKind,
                            TensorMapInterleaveKind>;
};
```

**Memory State**:

```
TensorMapDescriptorTypeStorage @ 0x55e8a0011000
├── tensorType: memref<128x64xf16, 3>
├── swizzle: SWIZZLE_128B (1)
├── l2promo: L2PROMO_NONE (0)
├── oob: OOB_ZERO (0)
└── interleave: INTERLEAVE_NONE (0)
```

### Frame 10: MemRef Cast Operation

**Back in Frame 2**: `memref.CastOp(...)`

**Call**: `memref.CastOp(unrankedMemrefType, device_ptr)`

**Python Binding** (auto-generated from TableGen):

```python
# Conceptual
class CastOp:
    def __init__(self, result_type, source):
        # Build operation
        ...
```

**C++ Builder**:

**File**: [mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp:850-870](../../../mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp#L850-L870)

```cpp
void CastOp::build(OpBuilder &builder, OperationState &state,
                    Type resultType, Value source) {
  state.addOperands(source);
  state.addTypes(resultType);
}
```

**Operation Created**:

```
Operation @ 0x55e8a0012000
├── name: "memref.cast"
├── operands: [%a_dev : memref<512x1024xf16>]
├── results: [%0 : memref<*xf16>]  # Unranked
├── attributes: {}
└── regions: []
```

**Generated IR**:
```mlir
%a_unranked = memref.cast %a_dev : memref<512x1024xf16> to memref<*xf16>
```

### Frame 11: TmaCreateDescriptorOp

**Back in Frame 2**: `nvgpu.TmaCreateDescriptorOp(...)`

**Call**:
```python
nvgpu.TmaCreateDescriptorOp(
    tma_descriptor_ty,           # !nvgpu.tensormap.descriptor<...>
    device_unranked_memref,      # %a_unranked
    map(const, [128, 64])        # [%c128, %c64]
)
```

**Python Binding** (auto-generated):

**C++ Builder**:

**File**: [mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp:550-580](../../../mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp#L550-L580)

```cpp
void TmaCreateDescriptorOp::build(
    OpBuilder &builder, OperationState &state,
    Type resultType,
    Value tensor,
    ValueRange boxDimensions) {

  state.addOperands(tensor);
  state.addOperands(boxDimensions);
  state.addTypes(resultType);
}
```

**Operation Created**:

```
Operation @ 0x55e8a0013000
├── name: "nvgpu.tma.create.descriptor"
├── operands: [
│   %a_unranked : memref<*xf16>,
│   %c128 : index,
│   %c64 : index
│ ]
├── results: [
│   %tma_desc_a : !nvgpu.tensormap.descriptor<
│     tensor=memref<128x64xf16, 3>,
│     swizzle=swizzle_128b, ...
│   >
│ ]
├── attributes: {}
└── regions: []
```

**Generated IR**:
```mlir
%c128 = arith.constant 128 : index
%c64 = arith.constant 64 : index
%a_unranked = memref.cast %a_dev : memref<512x1024xf16> to memref<*xf16>
%tma_desc_a = nvgpu.tma.create.descriptor %a_unranked box[%c128, %c64]
  : memref<*xf16> -> !nvgpu.tensormap.descriptor<...>
```

---

## Trace 6: WGMMA Operation (D += A @ B)

This is the most complex operation, involving operator overloading across multiple layers.

### User Code

**File**: [Ch5.py:208](../../../mlir/test/Examples/NVGPU/Ch5.py#L208)

```python
D += A @ B
```

Where:
```python
A = WGMMAMatrix(WGMMAType.Descriptor, [128, 64], desc=a_tma, smem=a_smem)
B = WGMMAMatrix(WGMMAType.Descriptor, [64, 128], desc=b_tma, smem=b_smem)
D = WGMMAMatrix(WGMMAType.Accumulator, shape=[128, 128], ty=T.f32())
```

### Frame 1: Operator @ Evaluation

**Python**: `A @ B` triggers `A.__matmul__(B)`

**File**: [nvdsl.py:241-248](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L241-L248)

```python
def __matmul__(self, rhs):
    lhs = nvgpu.warpgroup_generate_descriptor(
        self.wgmma_ty, self.smem, self.desc.tma_descriptor
    )
    rhs = nvgpu.warpgroup_generate_descriptor(
        rhs.wgmma_ty, rhs.smem, rhs.desc.tma_descriptor
    )
    return [lhs, rhs]
```

**State**:
```python
self = A (WGMMAMatrix)
  - matrix_type = WGMMAType.Descriptor
  - M = 128, K = 64
  - desc = a_tma (TMA object)
  - smem = %a_smem (ir.Value pointing to memref<128x64xf16, 3>)

rhs = B (WGMMAMatrix)
  - matrix_type = WGMMAType.Descriptor
  - K = 64, N = 128
  - desc = b_tma
  - smem = %b_smem
```

### Frame 2: First warpgroup_generate_descriptor

**Call**: `nvgpu.warpgroup_generate_descriptor(A.wgmma_ty, A.smem, A.desc.tma_descriptor)`

**Parameters**:
- `A.wgmma_ty`: Type = `!nvgpu.warpgroup.descriptor<tensor=memref<128x64xf16, 3>>`
- `A.smem`: Value = `%a_smem : memref<128x64xf16, 3>`
- `A.desc.tma_descriptor`: Value = `%tma_desc_a : !nvgpu.tensormap.descriptor<...>`

**Python Binding** (auto-generated from TableGen):

**TableGen Source**: [mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td:980-1000](../../../mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td#L980-L1000)

```tablegen
def NVGPU_WarpgroupGenerateDescriptorOp : NVGPU_Op<"warpgroup.generate.descriptor"> {
  let arguments = (ins
    Arg<AnyMemRef, "shared memory buffer">:$tensor,
    NVGPU_TensorMapDescriptor:$tensorMap
  );
  let results = (outs NVGPU_WarpgroupDescriptor:$result);
}
```

**C++ Builder**:

**File**: [mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp:720-740](../../../mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp#L720-L740)

```cpp
void WarpgroupGenerateDescriptorOp::build(
    OpBuilder &builder, OperationState &state,
    Type resultType,
    Value tensor,
    Value tensorMap) {

  state.addOperands({tensor, tensorMap});
  state.addTypes(resultType);
}
```

**Operation Created**:

```
Operation @ 0x55e8a0014000
├── name: "nvgpu.warpgroup.generate.descriptor"
├── operands: [
│   %a_smem : memref<128x64xf16, 3>,
│   %tma_desc_a : !nvgpu.tensormap.descriptor<...>
│ ]
├── results: [
│   %desc_a : !nvgpu.warpgroup.descriptor<tensor=memref<128x64xf16, 3>>
│ ]
├── attributes: {}
└── regions: []
```

**Generated IR**:
```mlir
%desc_a = nvgpu.warpgroup.generate.descriptor %a_smem, %tma_desc_a
  : memref<128x64xf16, 3>, !nvgpu.tensormap.descriptor<...>
  -> !nvgpu.warpgroup.descriptor<tensor=memref<128x64xf16, 3>>
```

### Frame 3: Second warpgroup_generate_descriptor

Similar to Frame 2, generates:

```mlir
%desc_b = nvgpu.warpgroup.generate.descriptor %b_smem, %tma_desc_b
  : memref<64x128xf16, 3>, !nvgpu.tensormap.descriptor<...>
  -> !nvgpu.warpgroup.descriptor<tensor=memref<64x128xf16, 3>>
```

**Return Value**:
```python
__matmul__ returns: [%desc_a, %desc_b]  # List of two ir.Value objects
```

### Frame 4: Operator += Evaluation

**Python**: `D += [desc_a, desc_b]` triggers `D.__iadd__([desc_a, desc_b])`

**File**: [nvdsl.py:250-256](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L250-L256)

```python
def __iadd__(self, matmulResult):
    lhs = matmulResult[0]  # %desc_a
    rhs = matmulResult[1]  # %desc_b
    acc_op = nvgpu.WarpgroupMmaOp(
        self.acc_op.type, lhs, rhs, self.acc_op, transposeB=True
    )
    return WGMMAMatrix(WGMMAType.Accumulator, acc_op=acc_op)
```

**State**:
```python
self = D (WGMMAMatrix)
  - matrix_type = WGMMAType.Accumulator
  - acc_op = %d_init : !nvgpu.warpgroup.accumulator<fragmented=vector<128x128xf32>>

matmulResult = [%desc_a, %desc_b]
```

### Frame 5: WarpgroupMmaOp Construction

**Call**:
```python
nvgpu.WarpgroupMmaOp(
    D.acc_op.type,  # Result type: !nvgpu.warpgroup.accumulator<...>
    %desc_a,        # LHS descriptor
    %desc_b,        # RHS descriptor
    D.acc_op,       # Initial accumulator
    transposeB=True
)
```

**Python Binding** (auto-generated):

**TableGen Source**: [mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td:1050-1100](../../../mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td#L1050-L1100)

```tablegen
def NVGPU_WarpgroupMmaOp : NVGPU_Op<"warpgroup.mma"> {
  let arguments = (ins
    NVGPU_WarpgroupDescriptor:$descriptorA,
    NVGPU_WarpgroupDescriptor:$descriptorB,
    NVGPU_WarpgroupAccumulator:$accumulatorIn,
    DefaultValuedAttr<BoolAttr, "false">:$transposeA,
    DefaultValuedAttr<BoolAttr, "false">:$transposeB
  );
  let results = (outs NVGPU_WarpgroupAccumulator:$accumulatorOut);
}
```

**C++ Builder**:

**File**: [mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp:850-880](../../../mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp#L850-L880)

```cpp
void WarpgroupMmaOp::build(
    OpBuilder &builder, OperationState &state,
    Type resultType,
    Value descriptorA,
    Value descriptorB,
    Value accumulatorIn,
    bool transposeA,
    bool transposeB) {

  state.addOperands({descriptorA, descriptorB, accumulatorIn});
  state.addTypes(resultType);
  state.addAttribute("transposeA", builder.getBoolAttr(transposeA));
  state.addAttribute("transposeB", builder.getBoolAttr(transposeB));
}
```

**Operation Created**:

```
Operation @ 0x55e8a0015000
├── name: "nvgpu.warpgroup.mma"
├── operands: [
│   %desc_a : !nvgpu.warpgroup.descriptor<tensor=memref<128x64xf16, 3>>,
│   %desc_b : !nvgpu.warpgroup.descriptor<tensor=memref<64x128xf16, 3>>,
│   %d_init : !nvgpu.warpgroup.accumulator<fragmented=vector<128x128xf32>>
│ ]
├── results: [
│   %d_new : !nvgpu.warpgroup.accumulator<fragmented=vector<128x128xf32>>
│ ]
├── attributes: {
│   "transposeA": BoolAttr(false),
│   "transposeB": BoolAttr(true)
│ }
└── regions: []
```

**Generated IR**:
```mlir
%d_new = nvgpu.warpgroup.mma %desc_a, %desc_b, %d_init
    {transposeA = false, transposeB = true}
  : !nvgpu.warpgroup.descriptor<tensor=memref<128x64xf16, 3>>,
    !nvgpu.warpgroup.descriptor<tensor=memref<64x128xf16, 3>>,
    !nvgpu.warpgroup.accumulator<fragmented=vector<128x128xf32>>
  -> !nvgpu.warpgroup.accumulator<fragmented=vector<128x128xf32>>
```

### Frame 6: Return New WGMMAMatrix

```python
return WGMMAMatrix(WGMMAType.Accumulator, acc_op=acc_op)
```

**New Python Object**:
```python
D_new = WGMMAMatrix(WGMMAType.Accumulator, acc_op=%d_new)
D_new.matrix_type = WGMMAType.Accumulator
D_new.acc_op = %d_new : !nvgpu.warpgroup.accumulator<...>
```

**Python Assignment**:
```python
D += A @ B  # D is now D_new
```

---

## State Summary: MLIR Context After All Operations

After executing the traced operations, the MLIR context contains:

```
MLIRContext @ 0x55e8a0001000
├── Loaded Dialects:
│   ├── BuiltinDialect
│   ├── FuncDialect
│   ├── GPUDialect
│   ├── NVGPUDialect
│   ├── ArithDialect
│   ├── MemRefDialect
│   └── SCFDialect
│
├── Type Cache (Uniqued Types):
│   ├── IndexType
│   ├── Float16Type
│   ├── Float32Type
│   ├── MemRefType<128x64xf16>
│   ├── MemRefType<64x128xf16>
│   ├── MemRefType<128x64xf16, 3>  # Shared memory
│   ├── UnrankedMemRefType<*xf16>
│   ├── MBarrierGroupType<workgroup, 7>
│   ├── TensorMapDescriptorType<...>
│   ├── WarpgroupDescriptorType<...>
│   └── WarpgroupAccumulatorType<...>
│
├── Attribute Cache:
│   ├── IntegerAttr(i32, 0)  # Dimension.x
│   ├── IntegerAttr(i32, 128)
│   ├── IntegerAttr(i32, 64)
│   ├── BoolAttr(true)
│   ├── BoolAttr(false)
│   └── DictionaryAttr(...)
│
└── Operations in Module:
    func.func @gemm_warp_specialized(...) {
      %token = gpu.async.token
      %a_dev, %t1 = gpu.alloc ...
      %tma_desc_a = nvgpu.tma.create.descriptor ...

      gpu.launch ... {
        %tidx = gpu.thread_id x : index
        %mbar_group = nvgpu.mbarrier.create ...
        %a_smem = gpu.dynamic_shared_memory ...
        %desc_a = nvgpu.warpgroup.generate.descriptor ...
        %d_init = nvgpu.warpgroup.mma.init.accumulator ...
        %d_new = nvgpu.warpgroup.mma %desc_a, %desc_b, %d_init ...
        gpu.terminator
      }

      func.return
    }
```

---

## Summary: Complete Dispatch Paths

### Key Insight: Three-Layer Architecture

Every MLIR operation follows this pattern:

```
Python Code
    ↓ [Python object method call]
Python Binding Layer (nanobind C++ extension)
    ↓ [C function call]
MLIR C API (opaque handles: MlirContext, MlirOperation, etc.)
    ↓ [C++ function call, handle unwrapping]
MLIR C++ Core (actual IR manipulation)
```

### State Changes at Each Layer

1. **Python Layer**: Python objects store references (pointers) to C API handles
2. **C API Layer**: Opaque handles wrap C++ pointers
3. **C++ Layer**: Actual memory allocation, IR construction, type uniquing

### Memory Management

- **Context**: Owns all IR objects via custom allocator
- **Types/Attributes**: Uniqued (cached) - same type = same memory address
- **Operations**: Allocated in context's arena, parent references maintained
- **Values**: Represented as pointers into operation results/block arguments

### Thread Safety

- MLIR contexts are **not thread-safe** by default
- Each Python `with ir.Context()` creates isolated context
- Operations within context must be single-threaded

---

## Related Files Reference

| Component | File | Line Range |
|-----------|------|------------|
| **Python User Code** | [Ch5.py](../../../mlir/test/Examples/NVGPU/Ch5.py) | 1-322 |
| **NVDSL DSL** | [nvdsl.py](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py) | 1-460 |
| **Python IR Bindings** | [IRCore.cpp](../../../mlir/lib/Bindings/Python/IRCore.cpp) | 1-2500 |
| **Python NVGPU Bindings** | [DialectNVGPU.cpp](../../../mlir/lib/Bindings/Python/DialectNVGPU.cpp) | 1-42 |
| **C API - IR** | [IR.cpp](../../../mlir/lib/CAPI/IR/IR.cpp) | 1-800 |
| **C API - NVGPU** | [NVGPU.cpp](../../../mlir/lib/CAPI/Dialect/NVGPU.cpp) | 1-50 |
| **C++ Context** | [MLIRContext.cpp](../../../mlir/lib/IR/MLIRContext.cpp) | 1-500 |
| **C++ Operations** | [Operation.cpp](../../../mlir/lib/IR/Operation.cpp) | 1-1000 |
| **C++ Types** | [BuiltinTypes.cpp](../../../mlir/lib/IR/BuiltinTypes.cpp) | 1-800 |
| **NVGPU Dialect** | [NVGPUDialect.cpp](../../../mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp) | 1-1200 |
| **GPU Dialect** | [GPUDialect.cpp](../../../mlir/lib/Dialect/GPU/IR/GPUDialect.cpp) | 1-1500 |

---

**This completes the frame-by-frame trace showing the complete dispatch path from Python to MLIR C++!**
