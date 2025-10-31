# MLIR Memory Management Patterns and Optimizations

This document describes the core memory management idioms and optimization patterns used throughout MLIR.

## Table of Contents
- [Two-Phase Construction Pattern](#two-phase-construction-pattern)
- [Core Memory Management Idioms](#core-memory-management-idioms)
- [Performance Implications](#performance-implications)

---

## Two-Phase Construction Pattern

### Overview: `OperationState` + `Operation::create()`

MLIR uses a two-phase construction pattern for operations:
1. **Build phase**: Accumulate operation data in an `OperationState`
2. **Allocation phase**: Call `Operation::create(state)` to allocate and construct

### Why This Pattern?

#### 1. Variable-Sized Allocation with Trailing Objects

Operations have dynamic memory requirements that can't be known at compile time. The `OperationState` collects all necessary information before computing the final size.

**Location**: [mlir/lib/IR/Operation.cpp:66-119](../../mlir/lib/IR/Operation.cpp#L66-L119)

```cpp
// Compute the byte size for the operation and the operand storage.
size_t byteSize =
    totalSizeToAlloc<detail::OperandStorage, detail::OpProperties,
                     BlockOperand, Region, OpOperand>(
        needsOperandStorage ? 1 : 0, opPropertiesAllocSize, numSuccessors,
        numRegions, numOperands);
size_t prefixByteSize = llvm::alignTo(
    Operation::prefixAllocSize(numTrailingResults, numInlineResults),
    alignof(Operation));
char *mallocMem = reinterpret_cast<char *>(malloc(byteSize + prefixByteSize));
void *rawMem = mallocMem + prefixByteSize;

// Create the new Operation using placement new
Operation *op = ::new (rawMem) Operation(
    location, name, numResults, numSuccessors, numRegions,
    opPropertiesAllocSize, attributes, properties, needsOperandStorage);
```

**Key Benefits**:
- **Single allocation**: Everything (operation + operands + results + regions) in one `malloc` call
- **Cache locality**: All related data in contiguous memory
- **No fragmentation**: Reduces heap fragmentation vs. many small allocations

#### 2. Immutability Enforcement

**Location**: [mlir/include/mlir/IR/OperationSupport.h:948-993](../../mlir/include/mlir/IR/OperationSupport.h#L948-L993)

```cpp
struct OperationState {
  Location location;
  OperationName name;
  SmallVector<Value, 4> operands;
  SmallVector<Type, 4> types;           // Result types
  NamedAttrList attributes;
  SmallVector<Block *, 1> successors;
  SmallVector<std::unique_ptr<Region>, 1> regions;

  // Properties handling
  Attribute propertiesAttr;

  // Move-only semantics
  OperationState(OperationState &&other) = default;
  OperationState(const OperationState &other) = delete;  // No copies!
  OperationState &operator=(const OperationState &other) = delete;
};
```

Once an `Operation` is created, many properties become **immutable**:
- Number of results
- Number of operands (though values can be updated)
- Operation name

This immutability enables optimizations and prevents accidental misuse.

#### 3. Memory Layout Optimization with TrailingObjects

**Location**: [mlir/include/mlir/IR/Operation.h:84-88](../../mlir/include/mlir/IR/Operation.h#L84-L88)

```cpp
class alignas(8) Operation final
    : public llvm::ilist_node_with_parent<Operation, Block>,
      private llvm::TrailingObjects<Operation, detail::OperandStorage,
                                    detail::OpProperties, BlockOperand, Region,
                                    OpOperand> {
```

The `llvm::TrailingObjects` pattern stores variable-sized data **inline** after the fixed-size `Operation` object:

```
Memory Layout:
┌────────────────────────────────────────────────────────────┐
│ Fixed Operation Header                                      │
├────────────────────────────────────────────────────────────┤
│ OperandStorage (if needed)                                 │
├────────────────────────────────────────────────────────────┤
│ OpProperties (aligned to 8 bytes)                          │
├────────────────────────────────────────────────────────────┤
│ BlockOperand[numSuccessors]                                │
├────────────────────────────────────────────────────────────┤
│ Region[numRegions]                                         │
├────────────────────────────────────────────────────────────┤
│ OpOperand[numOperands]                                     │
└────────────────────────────────────────────────────────────┘
     All in ONE contiguous allocation!
```

**Benefits**:
- Zero pointer indirections for accessing operands/results
- Excellent cache locality
- Memory overhead only for actual data, not bookkeeping

---

## Core Memory Management Idioms

### 1. Immortal Uniquing (Interning) Pattern

**Used for**: `Type`, `Attribute`, `Identifier`

**Implementation**: `StorageUniquer` class in [mlir/include/mlir/Support/StorageUniquer.h](../../mlir/include/mlir/Support/StorageUniquer.h)

**Concept**:
- All `Type` and `Attribute` instances are **hash-consed** (uniqued) in tables owned by `MLIRContext`
- Creating the same type/attribute twice returns **the same pointer**
- Objects are **immortal** - they live until the context is destroyed
- Enables **pointer equality** for structural equality checks

**Example**:
```cpp
MLIRContext ctx;
Type t1 = IntegerType::get(&ctx, 32);
Type t2 = IntegerType::get(&ctx, 32);

// Pointer comparison is sufficient!
assert(t1.getAsOpaquePointer() == t2.getAsOpaquePointer());  // Same object!

// No need for deep structural comparison
bool equal = (t1 == t2);  // Just compares pointers
```

**Benefits**:
- **O(1) equality comparison**: Just pointer comparison
- **Memory savings**: Duplicate types/attributes share storage
- **Thread-safe**: `StorageUniquer` provides locking
- **Canonical representation**: Only one instance per unique value

**Trade-offs**:
- Memory never freed until context destruction
- Initial allocation requires hash table lookup
- Context must outlive all types/attributes

### 2. Arena/Bulk Deallocation for Operations

**Pattern**: Operations are allocated individually but deallocated in bulk

**Lifecycle**:
1. `Operation::create()` does individual `malloc`
2. Operation inserted into parent `Block`
3. When `Region` (containing `Block`) is destroyed, all operations destroyed together
4. No individual `free()` calls - bulk cleanup

**Benefits**:
- Fast deallocation: Just destroy the parent region
- Simplifies ownership: Operations owned by blocks/regions
- Reduces allocator overhead: No per-operation free

**Example**:
```cpp
Region *region = new Region();
Block *block = new Block();
region->push_back(block);

// Create many operations
for (int i = 0; i < 10000; ++i) {
  Operation *op = builder.create<SomeOp>(...);
  block->push_back(op);
}

// Destroy all 10000 operations at once
delete region;  // Fast bulk deallocation
```

### 3. Intrusive Data Structures

**Pattern**: Embed list nodes directly in objects rather than using external node allocations

**Example**: Operations in Blocks
```cpp
class Operation final
    : public llvm::ilist_node_with_parent<Operation, Block> {
    // List node embedded in Operation!
```

**Benefits**:
- **Zero allocation overhead**: No separate node allocations
- **Better cache locality**: Node and data together
- **Automatic parent tracking**: `getParentOp()` is free

**Standard library comparison**:
```cpp
// Standard approach (std::list): TWO allocations per operation
std::list<Operation*> ops;  // Node allocated separately

// MLIR approach: ONE allocation
Block block;
block.push_back(op);  // No extra allocation, op contains list node
```

### 4. Value/Use Chains (Def-Use Analysis)

**Pattern**: Intrusive linked list for tracking where values are used

**Key Classes**:
- `Value`: Lightweight handle to an SSA value (just a pointer + index)
- `OpOperand`: Represents a use of a value, forms intrusive linked list
- `OpResult`: Represents a value defined by an operation

**Structure**:
```cpp
class Value {
  // Just a pointer to the defining operation/block argument
  detail::ValueImpl *impl;
};

class OpOperand : public IROperand<OpOperand, OpResult> {
  // Intrusive list node for use chain
  OpOperand *nextUse;
  OpOperand **back;  // Previous pointer in the list
  Value value;       // The value being used
};
```

**Benefits**:
- **O(1) value replacement**: Just update use chain pointers
- **O(N) use iteration**: Walk linked list of uses
- **No separate allocation**: Uses stored in operation's trailing objects

**Example**:
```cpp
// Walk all uses of a value
for (OpOperand &use : value.getUses()) {
  Operation *user = use.getOwner();
  unsigned operandNum = use.getOperandNumber();
  // Process use...
}

// Replace all uses
value.replaceAllUsesWith(newValue);  // O(N) in number of uses
```

### 5. SmallVector Throughout

**Pattern**: Use `llvm::SmallVector<T, N>` instead of `std::vector<T>`

**Concept**: Store first N elements **inline** (no heap allocation), only allocate when exceeding capacity

```cpp
SmallVector<Value, 4> operands;  // First 4 operands inline!

// 0-4 operands: no heap allocation
operands.push_back(val1);  // Stored inline
operands.push_back(val2);  // Stored inline

// 5+ operands: falls back to heap allocation
operands.push_back(val5);  // Triggers malloc
```

**Usage in MLIR**:
- Operation operands
- Operation results
- Block arguments
- Attribute lists
- Most temporary collections

**Benefits**:
- **Common case optimization**: Most operations have few operands/results
- **Reduced allocations**: ~80% of cases avoid heap allocation
- **Cache-friendly**: Small data inline with containing object

**Trade-offs**:
- Larger object size (includes inline storage)
- Choose N based on profiling/common case

### 6. Opaque Properties with Type Erasure

**Pattern**: Store operation-specific data without template bloat

**Location**: [mlir/include/mlir/IR/OperationSupport.h:71-82](../../mlir/include/mlir/IR/OperationSupport.h#L71-L82)

```cpp
class OpaqueProperties {
public:
  OpaqueProperties(void *prop) : properties(prop) {}

  template <typename Dest>
  Dest as() const {
    return static_cast<Dest>(const_cast<void *>(properties));
  }

private:
  void *properties;  // Type-erased storage
};
```

**Usage**:
```cpp
// Each operation can have custom properties
struct MyOpProperties {
  int64_t someValue;
  StringRef name;
};

// Stored opaquely in Operation
OpaqueProperties props = op.getProperties();
auto *myProps = props.as<MyOpProperties*>();
```

**Benefits**:
- **No templates**: `Operation` class not templated
- **Flexible**: Each op type can have different properties
- **Type-safe**: Access through registered interface

**Trade-offs**:
- Requires casting (type-checked via `TypeID`)
- Less type-safe than templates (runtime vs compile-time)

### 7. Custom Allocator Hooks via StorageUniquer

**Pattern**: Pluggable allocation strategy for custom storage classes

**Interface**:
```cpp
class StorageUniquer {
  // Allocate storage for a type/attribute
  template <typename Storage, typename... Args>
  Storage *allocate(Args &&...args);

  // Get or create uniqued storage
  template <typename Storage>
  Storage *get(/* construction parameters */);
};
```

**Features**:
- Thread-safe uniquing with fine-grained locking
- Custom hash/equality functions
- Support for mutable vs immutable storage
- Parameterized storage construction

**Example** (Attribute storage):
```cpp
struct IntegerAttrStorage : public AttributeStorage {
  using KeyTy = std::pair<Type, APInt>;

  // Hash and equality for uniquing
  static unsigned hashKey(const KeyTy &key);
  bool operator==(const KeyTy &key) const;

  // Construction
  static IntegerAttrStorage *construct(
      StorageAllocator &allocator, const KeyTy &key);
};
```

---

## Performance Implications

### Memory Efficiency

| Pattern | Memory Saved | Scenario |
|---------|--------------|----------|
| Uniquing | 50-90% | Many duplicate types/attributes |
| Trailing objects | 24-48 bytes | Per operation (pointer overhead) |
| Intrusive lists | 16-24 bytes | Per list node |
| SmallVector (N=4) | 16-32 bytes | When ≤N elements (80% of cases) |

### Cache Performance

**Operation traversal** (hot path in optimization passes):
```cpp
// Excellent cache behavior
for (Operation &op : block) {
  // Operation data contiguous:
  // - Operand storage inline
  // - Results inline
  // - Next pointer inline (intrusive list)
  // All likely in same cache line!
}
```

**Type comparison** (extremely hot path):
```cpp
// Before uniquing: deep structural comparison
bool equal = type1.deepCompare(type2);  // Walk entire type tree

// With uniquing: pointer comparison
bool equal = (type1 == type2);  // Single pointer comparison!
```

### Allocation Performance

Typical compiler pass over 10,000 operations:
- **Without optimizations**: ~50,000 allocations (ops + lists + vectors)
- **With MLIR patterns**: ~10,000 allocations (ops only, rest inline/arena)
- **3-5x reduction** in allocator calls

---

## Summary Table

| Pattern | Purpose | Key Benefit | Trade-off |
|---------|---------|-------------|-----------|
| **Two-phase construction** | Variable-size allocation | Single malloc per operation | Requires state object |
| **Trailing objects** | Inline variable data | Zero pointer indirections | Complex memory layout |
| **Uniquing/interning** | Canonical representation | O(1) equality via pointers | Immortal (memory not freed) |
| **Intrusive containers** | Zero-overhead lists | No node allocations | Embedded state |
| **Type erasure** | Generic interfaces | No template bloat | Runtime casting |
| **Arena semantics** | Bulk deallocation | Fast cleanup | Delayed memory reclamation |
| **SmallVector** | Inline small collections | ~80% avoid heap | Larger object size |

---

## When to Use Each Pattern

### Use Two-Phase Construction When:
- Creating objects with variable-sized trailing data
- Need to compute size before allocation
- Want single allocation for complex objects

### Use Uniquing When:
- Values have structural equality semantics
- Equality comparison is frequent (hot path)
- Memory overhead of duplicates is significant
- Values can be immortal (tied to context lifetime)

### Use Intrusive Containers When:
- Objects naturally belong to one container
- Cache locality is critical
- Want automatic parent tracking

### Use SmallVector When:
- Collection size typically small (< 8 elements)
- Frequently allocate/deallocate
- 80/20 rule applies (80% of cases fit in N)

### Use Type Erasure When:
- Need generic interface without templates
- Each instance may have different structure
- Want stable ABI across types

---

## Further Reading

- `llvm::TrailingObjects` documentation: [llvm/ADT/TrailingObjects.h](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/TrailingObjects.h)
- MLIR Operation design: [mlir/include/mlir/IR/Operation.h](../../mlir/include/mlir/IR/Operation.h)
- StorageUniquer implementation: [mlir/lib/Support/StorageUniquer.cpp](../../mlir/lib/Support/StorageUniquer.cpp)
- LLVM Programmer's Manual on data structures: https://llvm.org/docs/ProgrammersManual.html
