# MLIR InsertionPoint: Complete Technical Deep Dive

This document provides a comprehensive trace of how `ir.InsertionPoint` works in MLIR's Python bindings, from the Python API through C++ implementation.

## Table of Contents
- [Overview](#overview)
- [Layer-by-Layer Trace](#layer-by-layer-trace)
- [Thread-Local Context Stack](#thread-local-context-stack)
- [How Operations are Inserted](#how-operations-are-inserted)
- [Insertion Point Behavior](#insertion-point-behavior)
- [Common Pitfalls](#common-pitfalls)

---

## Overview

When you write Python code like:

```python
with ir.InsertionPoint(launch_op.body.blocks[0]):
    op1 = arith.ConstantOp(...)
    op2 = arith.AddIOp(...)
```

This triggers a multi-layer system involving:
1. **Python context manager protocol** (`__enter__` / `__exit__`)
2. **Thread-local stack management** for tracking active contexts
3. **C API wrappers** for interfacing with C++
4. **C++ intrusive data structures** for efficient operation storage

---

## Layer-by-Layer Trace

### Layer 1: Python API

**File**: [mlir/python/mlir/ir.py:10](../../mlir/python/mlir/ir.py#L10)

The `InsertionPoint` class is imported from C++ bindings:

```python
from ._mlir_libs._mlir.ir import *
```

**Type Stubs**: [mlir/python/mlir/_mlir_libs/_mlir/ir.pyi:1848-1885](../../mlir/python/mlir/_mlir_libs/_mlir/ir.pyi#L1848-L1885)

```python
class InsertionPoint:
    current: ClassVar[InsertionPoint]  # Thread-local current insertion point

    def __enter__(self) -> InsertionPoint: ...
    def __exit__(self, arg0: Any, arg1: Any, arg2: Any) -> None: ...

    @overload
    def __init__(self, block: Block) -> None:
        """Inserts after the last operation but still inside the block."""

    @overload
    def __init__(self, beforeOperation: _OperationBase) -> None:
        """Inserts before a referenced operation."""

    def insert(self, operation: _OperationBase) -> None:
        """Inserts an operation."""

    @property
    def block(self) -> Block:
        """Returns the block that this InsertionPoint points to."""

    @property
    def ref_operation(self) -> _OperationBase | None:
        """The reference operation before which new operations are inserted,
        or None if the insertion point is at the end of the block"""
```

**Usage Example**:
```python
# Insert at end of block
with ir.InsertionPoint(block):
    op = create_some_op()

# Insert before specific operation
with ir.InsertionPoint(existing_op):
    new_op = create_another_op()

# Access current insertion point
current_ip = ir.InsertionPoint.current
```

---

### Layer 2: C++ Python Bindings (Nanobind)

**Header**: [mlir/lib/Bindings/Python/IRModule.h:820-856](../../mlir/lib/Bindings/Python/IRModule.h#L820-L856)

#### PyInsertionPoint Class

```cpp
/// An insertion point maintains a pointer to a Block and a reference operation.
/// Calls to insert() will insert a new operation before the reference operation.
/// If the reference operation is null, then appends to the end of the block.
class PyInsertionPoint {
public:
  /// Creates an insertion point positioned after the last operation in the
  /// block, but still inside the block.
  PyInsertionPoint(const PyBlock &block);

  /// Creates an insertion point positioned before a reference operation.
  PyInsertionPoint(PyOperationBase &beforeOperationBase);

  /// Shortcut to create an insertion point at the beginning of the block.
  static PyInsertionPoint atBlockBegin(PyBlock &block);

  /// Shortcut to create an insertion point before the block terminator.
  static PyInsertionPoint atBlockTerminator(PyBlock &block);

  /// Shortcut to create an insertion point after the specified operation.
  static PyInsertionPoint after(PyOperationBase &op);

  /// Inserts an operation.
  void insert(PyOperationBase &operationBase);

  /// Enter and exit the context manager.
  static nanobind::object contextEnter(nanobind::object insertionPoint);
  void contextExit(const nanobind::object &excType,
                   const nanobind::object &excVal,
                   const nanobind::object &excTb);

  PyBlock &getBlock() { return block; }
  std::optional<PyOperationRef> &getRefOperation() { return refOperation; }

private:
  std::optional<PyOperationRef> refOperation;  // Reference operation (if any)
  PyBlock block;                                // The block to insert into
};
```

#### Context Manager Implementation

**File**: [mlir/lib/Bindings/Python/IRCore.cpp:2087-2095](../../mlir/lib/Bindings/Python/IRCore.cpp#L2087-L2095)

```cpp
nb::object PyInsertionPoint::contextEnter(nb::object insertPoint) {
  return PyThreadContextEntry::pushInsertionPoint(insertPoint);
}

void PyInsertionPoint::contextExit(const nb::object &excType,
                                   const nb::object &excVal,
                                   const nb::object &excTb) {
  PyThreadContextEntry::popInsertionPoint(*this);
}
```

**Key Insight**: The context manager delegates to `PyThreadContextEntry` for managing a thread-local stack.

#### Constructor Implementations

**File**: [mlir/lib/Bindings/Python/IRCore.cpp:2022-2026](../../mlir/lib/Bindings/Python/IRCore.cpp#L2022-L2026)

```cpp
// Constructor: Insert at end of block
PyInsertionPoint::PyInsertionPoint(const PyBlock &block) : block(block) {}

// Constructor: Insert before operation
PyInsertionPoint::PyInsertionPoint(PyOperationBase &beforeOperationBase)
    : refOperation(beforeOperationBase.getOperation().getRef()),
      block((*refOperation)->getBlock()) {}
```

#### Insert Implementation

**File**: [mlir/lib/Bindings/Python/IRCore.cpp:2028-2052](../../mlir/lib/Bindings/Python/IRCore.cpp#L2028-L2052)

```cpp
void PyInsertionPoint::insert(PyOperationBase &operationBase) {
  PyOperation &operation = operationBase.getOperation();
  if (operation.isAttached())
    throw nb::value_error(
        "Attempt to insert operation that is already attached");

  block.getParentOperation()->checkValid();
  MlirOperation beforeOp = {nullptr};

  if (refOperation) {
    // Insert before operation.
    (*refOperation)->checkValid();
    beforeOp = (*refOperation)->get();
  } else {
    // Insert at end (before null) is only valid if the block does not
    // already end in a known terminator (violating this will cause assertion
    // failures later).
    if (!mlirOperationIsNull(mlirBlockGetTerminator(block.get()))) {
      throw nb::index_error("Cannot insert operation at the end of a block "
                            "that already has a terminator. Did you mean to "
                            "use 'InsertionPoint.at_block_terminator(block)' "
                            "versus 'InsertionPoint(block)'?");
    }
  }

  mlirBlockInsertOwnedOperationBefore(block.get(), beforeOp, operation);
  operation.setAttached();
}
```

#### Binding Registration

**File**: [mlir/lib/Bindings/Python/IRCore.cpp:3846-3866](../../mlir/lib/Bindings/Python/IRCore.cpp#L3846-L3866)

```cpp
nb::class_<PyInsertionPoint>(m, "InsertionPoint")
    .def("__enter__", &PyInsertionPoint::contextEnter)
    .def("__exit__", &PyInsertionPoint::contextExit,
         nb::arg("exc_type").none(), nb::arg("exc_value").none(),
         nb::arg("traceback").none())
    .def_prop_ro_static(
        "current",
        [](nb::object & /*class*/) {
          auto *ip = PyThreadContextEntry::getDefaultInsertionPoint();
          if (!ip)
            throw nb::value_error("No current InsertionPoint");
          return ip;
        },
        "Gets the InsertionPoint bound to the current thread or raises "
        "ValueError if none has been set")
    .def(nb::init<PyBlock &>(), nb::arg("block"),
         "Inserts after the last operation but still inside the block.")
    .def(nb::init<PyOperationBase &>(), nb::arg("beforeOperation"),
         "Inserts before a referenced operation.")
    .def_static("at_block_begin", &PyInsertionPoint::atBlockBegin,
                nb::arg("block"), "Inserts at the beginning of the block.")
    .def_static("at_block_terminator", &PyInsertionPoint::atBlockTerminator,
                nb::arg("block"), "Inserts before the block terminator.")
    .def_static("after", &PyInsertionPoint::after, nb::arg("operation"),
                "Inserts after the given operation.")
    .def("insert", &PyInsertionPoint::insert, nb::arg("operation"))
    .def_prop_ro("block", &PyInsertionPoint::getBlock)
    .def_prop_ro("ref_operation",
                 [](PyInsertionPoint &self) -> nb::object {
                   auto &refOperation = self.getRefOperation();
                   if (refOperation)
                     return nb::cast(refOperation->getObject());
                   return nb::none();
                 });
```

---

### Layer 3: Thread-Local Context Stack

**Header**: [mlir/lib/Bindings/Python/IRModule.h:109-161](../../mlir/lib/Bindings/Python/IRModule.h#L109-L161)

#### PyThreadContextEntry Class

MLIR maintains a **thread-local stack** of context entries. Each entry can hold:
- An `MLIRContext`
- An `InsertionPoint`
- A `Location`

```cpp
/// Tracks an entry in the thread context stack. New entries are pushed onto
/// here for each with block that activates a new InsertionPoint, Context or
/// Location.
class PyThreadContextEntry {
public:
  enum class FrameKind {
    Context,
    InsertionPoint,
    Location,
  };

  PyThreadContextEntry(FrameKind frameKind, nanobind::object context,
                       nanobind::object insertionPoint,
                       nanobind::object location)
      : context(std::move(context)), insertionPoint(std::move(insertionPoint)),
        location(std::move(location)), frameKind(frameKind) {}

  /// Gets the top of stack context and return nullptr if not defined.
  static PyMlirContext *getDefaultContext();

  /// Gets the top of stack insertion point and return nullptr if not defined.
  static PyInsertionPoint *getDefaultInsertionPoint();

  /// Gets the top of stack location and returns nullptr if not defined.
  static PyLocation *getDefaultLocation();

  /// Stack management.
  static nanobind::object pushInsertionPoint(nanobind::object insertionPoint);
  static void popInsertionPoint(PyInsertionPoint &insertionPoint);

  /// Gets the thread local stack.
  static std::vector<PyThreadContextEntry> &getStack();

private:
  nanobind::object context;           // PyContext reference
  nanobind::object insertionPoint;    // Current insertion point
  nanobind::object location;          // Current location
  FrameKind frameKind;                // What was pushed
};
```

#### Push/Pop Implementation

**File**: [mlir/lib/Bindings/Python/IRCore.cpp:885-906](../../mlir/lib/Bindings/Python/IRCore.cpp#L885-L906)

```cpp
PyThreadContextEntry::pushInsertionPoint(nb::object insertionPointObj) {
  PyInsertionPoint &insertionPoint =
      nb::cast<PyInsertionPoint &>(insertionPointObj);

  // Extract the context from the insertion point's block
  nb::object contextObj =
      insertionPoint.getBlock().getParentOperation()->getContext().getObject();

  // Push a new frame onto the thread-local stack
  push(FrameKind::InsertionPoint,
       /*context=*/contextObj,
       /*insertionPoint=*/insertionPointObj,
       /*location=*/nb::object());

  return insertionPointObj;
}

void PyThreadContextEntry::popInsertionPoint(PyInsertionPoint &insertionPoint) {
  auto &stack = getStack();
  if (stack.empty())
    throw std::runtime_error("Unbalanced InsertionPoint enter/exit");

  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::InsertionPoint &&
      tos.getInsertionPoint() != &insertionPoint)
    throw std::runtime_error("Unbalanced InsertionPoint enter/exit");

  stack.pop_back();
}
```

**Stack Visualization**:

```
Thread-Local Stack (per Python thread):
┌─────────────────────────────────────┐
│ Top of Stack (TOS)                  │
│ FrameKind: InsertionPoint           │
│ context: PyMlirContext              │
│ insertionPoint: PyInsertionPoint    │ ← Current
│ location: None                      │
├─────────────────────────────────────┤
│ FrameKind: Context                  │
│ context: PyMlirContext              │
│ insertionPoint: None                │
│ location: None                      │
└─────────────────────────────────────┘
```

When you nest `with` blocks:

```python
with ir.Context():
    with ir.Location.unknown():
        with ir.InsertionPoint(block):
            # Stack has 3 entries
            pass
        # Stack has 2 entries
    # Stack has 1 entry
# Stack is empty
```

---

### Layer 4: C API

**File**: [mlir/lib/CAPI/IR/IR.cpp:1031-1041](../../mlir/lib/CAPI/IR/IR.cpp#L1031-L1041)

The C API provides a C-compatible interface to C++ functionality:

```cpp
void mlirBlockInsertOwnedOperationBefore(MlirBlock block,
                                         MlirOperation reference,
                                         MlirOperation operation) {
  if (mlirOperationIsNull(reference))
    return mlirBlockAppendOwnedOperation(block, operation);

  assert(unwrap(reference)->getBlock() == unwrap(block) &&
         "expected reference operation to belong to the block");

  // Unwrap opaque handles to C++ objects and insert
  unwrap(block)->getOperations().insert(Block::iterator(unwrap(reference)),
                                        unwrap(operation));
}
```

**Key Functions**:
- `wrap()`: Convert C++ pointer to opaque C handle
- `unwrap()`: Convert opaque C handle back to C++ pointer

**Type Definitions** (from [mlir/include/mlir-c/IR.h](../../mlir/include/mlir-c/IR.h)):
```c
// Opaque handles
typedef struct MlirBlock_* MlirBlock;
typedef struct MlirOperation_* MlirOperation;
typedef struct MlirContext_* MlirContext;
```

---

### Layer 5: C++ Core Implementation

#### OpBuilder::InsertPoint

**File**: [mlir/include/mlir/IR/Builders.h:327-345](../../mlir/include/mlir/IR/Builders.h#L327-L345)

```cpp
class OpBuilder : public Builder {
public:
  class InsertPoint {
  public:
    /// Creates a new insertion point which doesn't point to anything.
    InsertPoint() = default;

    /// Creates a new insertion point at the given location.
    InsertPoint(Block *insertBlock, Block::iterator insertPt)
        : block(insertBlock), point(insertPt) {}

    /// Returns true if this insert point is set.
    bool isSet() const { return (block != nullptr); }

    Block *getBlock() const { return block; }
    Block::iterator getPoint() const { return point; }

  private:
    Block *block = nullptr;      // The block to insert into
    Block::iterator point;       // Iterator position in the block
  };

  // ... builder methods ...
};
```

**Structure**:
- `block`: Raw pointer to the `Block` where operations will be inserted
- `point`: Bidirectional iterator into the block's intrusive operation list

#### Setting Insertion Point

**File**: [mlir/include/mlir/IR/Builders.h:398-408](../../mlir/include/mlir/IR/Builders.h#L398-L408)

```cpp
/// Set the insertion point to the specified location.
void setInsertionPoint(Block *block, Block::iterator insertPoint) {
  // TODO: check that insertPoint is in this rather than some other block.
  this->block = block;
  this->insertPoint = insertPoint;
}

/// Set the insertion point to before the specified operation.
void setInsertionPoint(Operation *op) {
  setInsertionPoint(op->getBlock(), Block::iterator(op));
}

/// Set the insertion point to after the specified operation.
void setInsertionPointAfter(Operation *op) {
  setInsertionPoint(op->getBlock(), ++Block::iterator(op));
}

/// Set the insertion point to the end of the specified block.
void setInsertionPointToEnd(Block *block) {
  setInsertionPoint(block, block->end());
}

/// Set the insertion point to the start of the specified block.
void setInsertionPointToStart(Block *block) {
  setInsertionPoint(block, block->begin());
}
```

#### Operation Insertion

**File**: [mlir/lib/IR/Builders.cpp:420-427](../../mlir/lib/IR/Builders.cpp#L420-L427)

```cpp
Operation *OpBuilder::insert(Operation *op) {
  if (block) {
    // Insert operation at current insertion point
    block->getOperations().insert(insertPoint, op);

    // Notify listener if registered
    if (listener)
      listener->notifyOperationInserted(op, /*previous=*/{});
  }
  return op;
}
```

**File**: [mlir/lib/IR/Builders.cpp:456-458](../../mlir/lib/IR/Builders.cpp#L456-L458)

```cpp
Operation *OpBuilder::create(const OperationState &state) {
  return insert(Operation::create(state));
}
```

**Two-Phase Process**:
1. `Operation::create(state)` allocates and constructs the operation
2. `insert(op)` inserts it into the IR at the current insertion point

---

## How Operations are Inserted

### The Intrusive List

Operations within a block are stored in an **intrusive doubly-linked list** (`llvm::iplist<Operation>`).

**Key Properties**:
- List nodes are **embedded in** `Operation` objects
- No separate allocation for list nodes
- Iterators point to `Operation*` directly

**From** [mlir/include/mlir/IR/Block.h](../../mlir/include/mlir/IR/Block.h):
```cpp
class Block : public IRObjectWithUseList<BlockOperand>,
              public llvm::ilist_node_with_parent<Block, Region> {
public:
  using OpListType = llvm::iplist<Operation>;
  OpListType &getOperations() { return operations; }

private:
  OpListType operations;  // Intrusive list of operations
};
```

### Insertion Process

When you call:
```cpp
block->getOperations().insert(insertPoint, op);
```

**What happens**:
1. `insertPoint` is an iterator pointing to some operation (or `end()`)
2. `op` is inserted **before** the operation pointed to by `insertPoint`
3. The iterator `insertPoint` **remains unchanged** - it still points to the same operation

**Visual Example**:

```
Before insertion:
┌─────────────────────────────────┐
│ Block                           │
│  ├── op_a                       │
│  ├── op_b  ← insertPoint        │
│  └── op_c                       │
└─────────────────────────────────┘

After insert(insertPoint, new_op):
┌─────────────────────────────────┐
│ Block                           │
│  ├── op_a                       │
│  ├── new_op                     │  ← Inserted here
│  ├── op_b  ← insertPoint        │  ← Still points here!
│  └── op_c                       │
└─────────────────────────────────┘

After insert(insertPoint, new_op2):
┌─────────────────────────────────┐
│ Block                           │
│  ├── op_a                       │
│  ├── new_op2                    │  ← Inserted here
│  ├── new_op                     │
│  ├── op_b  ← insertPoint        │  ← Still unchanged!
│  └── op_c                       │
└─────────────────────────────────┘
```

**Result**: If you insert multiple operations without advancing the insertion point, they appear in **reverse order**!

---

## Insertion Point Behavior

### Critical Insight: Insertion Point Does NOT Auto-Advance

When you insert an operation, the `OpBuilder`'s insertion point **does not automatically advance**.

```cpp
OpBuilder builder(block, block->end());

Operation *op1 = builder.create<ConstantOp>(...);  // Inserted at end
// insertPoint still points to end()

Operation *op2 = builder.create<AddIOp>(...);      // Also inserted at end
// insertPoint still points to end()
```

### Python Binding Behavior

In Python, when using `InsertionPoint` context manager:

```python
with ir.InsertionPoint(block):
    op1 = arith.ConstantOp(...)  # Uses current insertion point
    op2 = arith.AddIOp(...)      # Uses SAME insertion point
```

**Each operation**:
1. Retrieves current insertion point from thread-local stack
2. Inserts at that position
3. Does NOT modify the insertion point

### To Advance the Insertion Point

**In Python**:
```python
# Method 1: Create new insertion point after operation
with ir.InsertionPoint(block):
    op1 = arith.ConstantOp(...)

# After 'with' block, create new insertion point after op1
with ir.InsertionPoint.after(op1):
    op2 = arith.AddIOp(...)

# Method 2: Manual insertion
ip = ir.InsertionPoint(block)
op1 = create_op1()
ip.insert(op1)
# Now manually create new insertion point for next operation
```

**In C++**:
```cpp
// Method 1: Explicitly set after each operation
builder.create<ConstantOp>(...);
builder.setInsertionPointToEnd(block);
builder.create<AddIOp>(...);

// Method 2: Use InsertionGuard
{
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);
  builder.create<...>();
  // Restoration happens automatically on scope exit
}

// Method 3: Use setInsertionPointAfter
Operation *op1 = builder.create<ConstantOp>(...);
builder.setInsertionPointAfter(op1);
Operation *op2 = builder.create<AddIOp>(...);
```

---

## Common Pitfalls

### Pitfall 1: Reverse Order Insertion

**Problem**:
```python
with ir.InsertionPoint(block):
    for i in range(3):
        create_operation(i)
```

If insertion point is at the beginning, operations appear in **reverse order**: op2, op1, op0.

**Solution**:
```python
# Insert at end instead
with ir.InsertionPoint(block):
    # Internally uses block->end() as insertion point
    for i in range(3):
        create_operation(i)  # Ops appear in correct order

# Or: explicitly insert at block terminator
with ir.InsertionPoint.at_block_terminator(block):
    create_operation()
```

### Pitfall 2: Inserting After Terminator

**Problem**:
```python
# Block already has terminator
with ir.InsertionPoint(block):  # Tries to insert at end
    op = arith.ConstantOp(...)  # ERROR!
```

**Error**:
```
IndexError: Cannot insert operation at the end of a block that already has a
terminator. Did you mean to use 'InsertionPoint.at_block_terminator(block)'?
```

**Solution**:
```python
with ir.InsertionPoint.at_block_terminator(block):
    op = arith.ConstantOp(...)  # Inserts BEFORE terminator
```

### Pitfall 3: Unbalanced Context Managers

**Problem**:
```python
def bad_function():
    ip = ir.InsertionPoint(block)
    ip.__enter__()
    # Forgot to call __exit__!
    return

# Stack is corrupted!
```

**Solution**: Always use `with` statement:
```python
def good_function():
    with ir.InsertionPoint(block):
        create_ops()
    # __exit__ called automatically
```

### Pitfall 4: Thread-Local Confusion

**Problem**:
```python
# Thread 1
with ir.InsertionPoint(block1):
    # Thread 2 (won't see this insertion point)
    with ir.InsertionPoint(block2):
        pass
```

**Key**: Each thread has its **own** context stack. Context managers only affect the current thread.

### Pitfall 5: Using InsertionPoint Outside Context

**Problem**:
```python
ip = ir.InsertionPoint(block)
# Not in 'with' block - this insertion point is NOT active!
current = ir.InsertionPoint.current  # ERROR: No current InsertionPoint
```

**Solution**:
```python
with ir.InsertionPoint(block) as ip:
    # Now it's active
    current = ir.InsertionPoint.current  # Works!
```

---

## Complete Call Chain Summary

```
┌─────────────────────────────────────────────────────────────┐
│ Python Layer                                                 │
│   with ir.InsertionPoint(block):                            │
│       operation = dialect.SomeOp(...)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Nanobind Layer (C++ Python Bindings)                        │
│   PyInsertionPoint::contextEnter()                          │
│     └─> PyThreadContextEntry::pushInsertionPoint()          │
│           └─> push() onto thread-local stack                │
│                                                              │
│   [Operations created using InsertionPoint.current]         │
│                                                              │
│   PyInsertionPoint::contextExit()                           │
│     └─> PyThreadContextEntry::popInsertionPoint()           │
│           └─> pop() from thread-local stack                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Operation Creation                                           │
│   PyOperation::create() or builder.create<OpType>()        │
│     └─> Retrieve current insertion point from TLS           │
│           └─> mlirBlockInsertOwnedOperationBefore()         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ C API Layer                                                  │
│   mlirBlockInsertOwnedOperationBefore(block, ref, op)       │
│     └─> unwrap() opaque handles                             │
│           └─> block->getOperations().insert(iterator, op)   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ C++ Core Layer                                               │
│   llvm::iplist<Operation>::insert(iterator, Operation*)     │
│     └─> Intrusive list insertion                            │
│           └─> Update prev/next pointers in Operation        │
└─────────────────────────────────────────────────────────────┘
```

---

## Advanced Topics

### InsertionGuard (C++ Only)

The C++ API provides an RAII guard to automatically restore insertion points:

```cpp
OpBuilder builder(...);
builder.setInsertionPoint(someOp);

{
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  builder.create<...>();
  // Insertion point automatically restored on scope exit
}

// builder's insertion point is back to someOp
```

**Implementation** [mlir/include/mlir/IR/Builders.h:348-374](../../mlir/include/mlir/IR/Builders.h#L348-L374):
```cpp
class InsertionGuard {
public:
  InsertionGuard(OpBuilder &builder)
      : builder(&builder), ip(builder.saveInsertionPoint()) {}

  ~InsertionGuard() {
    if (builder)
      builder->restoreInsertionPoint(ip);
  }

  // Move-only semantics
  InsertionGuard(const InsertionGuard &) = delete;
  InsertionGuard(InsertionGuard &&other) noexcept
      : builder(other.builder), ip(other.ip) {
    other.builder = nullptr;  // Prevent double-restore
  }

private:
  OpBuilder *builder;
  OpBuilder::InsertPoint ip;
};
```

### Listener Pattern for Operation Insertion

OpBuilder supports registering listeners that get notified when operations are inserted:

```cpp
struct MyListener : public OpBuilder::Listener {
  void notifyOperationInserted(Operation *op, InsertPoint previous) override {
    llvm::errs() << "Operation inserted: " << op->getName() << "\n";
  }
};

MyListener listener;
OpBuilder builder(context, &listener);
builder.create<ConstantOp>(...);  // Triggers notifyOperationInserted
```

**Use cases**:
- Rewriter patterns (tracking rewrites)
- Debugging/logging
- Metrics collection
- Custom IR validation

---

## Best Practices

### 1. Always Use Context Managers

```python
# Good
with ir.InsertionPoint(block):
    create_operations()

# Bad
ip = ir.InsertionPoint(block)
ip.__enter__()
create_operations()
ip.__exit__(None, None, None)  # Easy to forget!
```

### 2. Be Explicit About Insertion Position

```python
# Unclear: where does this insert?
with ir.InsertionPoint(block):
    op = create_op()

# Clear: insert at end
with ir.InsertionPoint(block):  # Defaults to end
    op = create_op()

# Clear: insert before terminator
with ir.InsertionPoint.at_block_terminator(block):
    op = create_op()

# Clear: insert at beginning
with ir.InsertionPoint.at_block_begin(block):
    op = create_op()
```

### 3. Understand Insertion Order

```python
# If you need sequential order at block start, build backwards:
ops = []
for i in range(3):
    ops.append(create_op(i))

with ir.InsertionPoint.at_block_begin(block):
    for op in reversed(ops):
        insert(op)

# Or insert at end:
with ir.InsertionPoint(block):  # At end by default
    for i in range(3):
        create_op(i)  # Natural order preserved
```

### 4. Check for Terminators

```python
def safe_insert(block):
    if block.operations and block.operations[-1].name == "terminator":
        # Insert before terminator
        with ir.InsertionPoint.at_block_terminator(block):
            create_op()
    else:
        # Insert at end
        with ir.InsertionPoint(block):
            create_op()
```

### 5. Nested Contexts

```python
with ir.Context() as ctx:
    with ir.Location.unknown(ctx):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            # All contexts properly stacked
            func = create_function()
```

---

## Related Documentation

- [Memory Management Patterns](../optimizations/memory-management-patterns.md) - Operation allocation and lifecycle
- [OpBuilder C++ API](../../mlir/include/mlir/IR/Builders.h) - C++ builder interface
- [Python Bindings Architecture](../../mlir/lib/Bindings/Python/) - Complete Python binding implementation

---

## Debugging Tips

### Print Thread-Local Stack

Add debug code in your Python:

```python
import sys
import mlir.ir as ir

def debug_stack():
    try:
        ctx = ir.Context.current
        print(f"Current context: {ctx}")
    except:
        print("No current context")

    try:
        ip = ir.InsertionPoint.current
        print(f"Current insertion point: {ip.block}")
    except:
        print("No current insertion point")

    try:
        loc = ir.Location.current
        print(f"Current location: {loc}")
    except:
        print("No current location")

with ir.Context():
    debug_stack()  # Shows context
    with ir.InsertionPoint(block):
        debug_stack()  # Shows context + insertion point
```

### Verify Insertion Order

```python
def verify_order(block, expected_ops):
    actual = [op.name for op in block.operations]
    assert actual == expected_ops, f"Expected {expected_ops}, got {actual}"
```

### Check for Memory Leaks

Python bindings properly manage reference counts, but verify with:

```python
import gc
import weakref

def check_leak():
    ops = []
    with ir.InsertionPoint(block):
        for i in range(100):
            op = create_op()
            ops.append(weakref.ref(op))

    gc.collect()
    alive = sum(1 for ref in ops if ref() is not None)
    print(f"Operations still alive: {alive}")
```

---

## Conclusion

The `InsertionPoint` context manager is a sophisticated multi-layer system that:

1. **Manages thread-local state** for tracking where operations should be inserted
2. **Wraps C++ functionality** through nanobind for Python accessibility
3. **Uses intrusive data structures** for efficient operation storage
4. **Does NOT auto-advance** - you control insertion position explicitly
5. **Integrates with** other context managers (Context, Location) via a unified stack

Understanding these layers is crucial for:
- Correctly generating IR in the right order
- Debugging insertion issues
- Writing efficient IR construction code
- Extending MLIR with custom operations and patterns
