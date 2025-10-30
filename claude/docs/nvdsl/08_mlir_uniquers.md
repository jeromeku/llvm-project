# MLIR Uniquers: Type, Attribute, and Affine Uniquing

## Introduction

MLIR uses **uniquers** to ensure that identical types, attributes, and affine expressions share the same memory location. This is a critical optimization called **structural sharing** or **hash-consing** that:

- Reduces memory usage (one copy per unique structure)
- Enables fast equality checks (pointer comparison instead of deep comparison)
- Maintains canonical forms (same value = same pointer)

This document explains the implementation of MLIR's three uniquer systems shown in the selected code.

---

## Overview: The Uniquing Problem

### Without Uniquing

```cpp
// Without uniquing - multiple copies of same type
Type t1 = Float32Type::get(ctx);  // Allocates @ 0x1000
Type t2 = Float32Type::get(ctx);  // Allocates @ 0x2000
Type t3 = Float32Type::get(ctx);  // Allocates @ 0x3000

// Problem 1: Wastes memory (3 copies)
// Problem 2: Slow comparison
t1 == t2  // Must compare structure: "float<32>" == "float<32>"
```

### With Uniquing

```cpp
// With uniquing - one canonical copy
Type t1 = Float32Type::get(ctx);  // Allocates @ 0x1000
Type t2 = Float32Type::get(ctx);  // Returns @ 0x1000 (cached!)
Type t3 = Float32Type::get(ctx);  // Returns @ 0x1000 (cached!)

// Benefit 1: Single allocation
// Benefit 2: Fast comparison
t1 == t2  // Just pointer comparison: 0x1000 == 0x1000
```

---

## Selected Code Analysis

**File**: [mlir/lib/IR/MLIRContext.cpp:310-357](../../../mlir/lib/IR/MLIRContext.cpp#L310-L357)

```cpp
//// Types.
/// Floating-point Types.
impl->bf16Ty = TypeUniquer::get<BFloat16Type>(this);
impl->f16Ty = TypeUniquer::get<Float16Type>(this);
impl->tf32Ty = TypeUniquer::get<FloatTF32Type>(this);
impl->f32Ty = TypeUniquer::get<Float32Type>(this);
impl->f64Ty = TypeUniquer::get<Float64Type>(this);
impl->f80Ty = TypeUniquer::get<Float80Type>(this);
impl->f128Ty = TypeUniquer::get<Float128Type>(this);

/// Index Type.
impl->indexTy = TypeUniquer::get<IndexType>(this);

/// Integer Types.
impl->int1Ty = TypeUniquer::get<IntegerType>(this, 1, IntegerType::Signless);
impl->int8Ty = TypeUniquer::get<IntegerType>(this, 8, IntegerType::Signless);
impl->int16Ty = TypeUniquer::get<IntegerType>(this, 16, IntegerType::Signless);
impl->int32Ty = TypeUniquer::get<IntegerType>(this, 32, IntegerType::Signless);
impl->int64Ty = TypeUniquer::get<IntegerType>(this, 64, IntegerType::Signless);
impl->int128Ty = TypeUniquer::get<IntegerType>(this, 128, IntegerType::Signless);

/// None Type.
impl->noneType = TypeUniquer::get<NoneType>(this);

//// Attributes.
impl->unknownLocAttr = AttributeUniquer::get<UnknownLoc>(this);
impl->falseAttr = IntegerAttr::getBoolAttrUnchecked(impl->int1Ty, false);
impl->trueAttr = IntegerAttr::getBoolAttrUnchecked(impl->int1Ty, true);
impl->unitAttr = AttributeUniquer::get<UnitAttr>(this);
impl->emptyDictionaryAttr = DictionaryAttr::getEmptyUnchecked(this);
impl->emptyStringAttr = StringAttr::getEmptyStringAttrUnchecked(this);

// Register affine storage types
impl->affineUniquer.registerParametricStorageType<AffineBinaryOpExprStorage>();
impl->affineUniquer.registerParametricStorageType<AffineConstantExprStorage>();
impl->affineUniquer.registerParametricStorageType<AffineDimExprStorage>();
impl->affineUniquer.registerParametricStorageType<AffineMapStorage>();
impl->affineUniquer.registerParametricStorageType<IntegerSetStorage>();
```

**Purpose**: Pre-create and cache commonly-used types and attributes during context initialization.

---

## 1. TypeUniquer

### Architecture

**File**: [mlir/lib/IR/StorageUniquerSupport.cpp](../../../mlir/lib/IR/StorageUniquerSupport.cpp)

```
TypeUniquer (per MLIRContext)
├── StorageUniquer (underlying implementation)
│   ├── impl: StorageUniquerImpl
│   │   ├── storageTypes: DenseMap<TypeID, ParametricStorageUniquer*>
│   │   │   └── Per-type cache
│   │   └── singletonStorageTypes: DenseMap<TypeID, BaseStorage*>
│   │       └── For parameterless types (Float32Type, IndexType, etc.)
│   └── allocator: StorageAllocator
│       └── BumpPtrAllocator for storage objects
```

### Type Categories

**1. Singleton Types** (no parameters):
- `Float32Type`, `IndexType`, `NoneType`
- Stored directly in `singletonStorageTypes` map
- Only one instance ever exists

**2. Parametric Types** (have parameters):
- `IntegerType(width, signedness)`
- `MemRefType(shape, elementType, layout, memorySpace)`
- Stored in `storageTypes` with parameter-based key
- One instance per unique parameter combination

### Implementation Deep Dive

#### Storage Classes

Every uniqued type has an associated storage class:

**File**: [mlir/include/mlir/IR/TypeSupport.h:30-50](../../../mlir/include/mlir/IR/TypeSupport.h#L30-L50)

```cpp
/// Base storage class for all types
class TypeStorage : public StorageUniquer::BaseStorage {
public:
  MLIRContext *getContext() const { return context; }

protected:
  TypeStorage(MLIRContext *context) : context(context) {}

private:
  MLIRContext *context;
};
```

**Example - IntegerType Storage**:

**File**: [mlir/lib/IR/BuiltinTypes.cpp:50-80](../../../mlir/lib/IR/BuiltinTypes.cpp#L50-L80)

```cpp
struct IntegerTypeStorage : public TypeStorage {
  IntegerTypeStorage(unsigned width, IntegerType::SignednessSemantics signedness)
      : width(width), signedness(signedness) {}

  /// The hash key used for uniquing
  using KeyTy = std::tuple<unsigned, IntegerType::SignednessSemantics>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(width, signedness);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  /// Construction interface
  static IntegerTypeStorage *construct(StorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<IntegerTypeStorage>())
        IntegerTypeStorage(std::get<0>(key), std::get<1>(key));
  }

  unsigned width;
  IntegerType::SignednessSemantics signedness;
};
```

#### TypeUniquer::get Implementation

**File**: [mlir/include/mlir/IR/StorageUniquerSupport.h:180-220](../../../mlir/include/mlir/IR/StorageUniquerSupport.h#L180-L220)

```cpp
template <typename ConcreteType, typename... Args>
ConcreteType TypeUniquer::get(MLIRContext *ctx, Args &&...args) {
  // Step 1: Create lookup key from arguments
  using StorageType = typename ConcreteType::ImplType;
  typename StorageType::KeyTy key(args...);

  // Step 2: Look up in cache
  StorageUniquer &uniquer = ctx->getImpl().types;
  BaseStorage *storage = uniquer.get<StorageType>(
      [&] { return StorageType::construct(uniquer.getAllocator(), key); },
      TypeID::get<ConcreteType>(),
      key);

  // Step 3: Wrap storage in type handle
  return ConcreteType(static_cast<StorageType*>(storage));
}
```

**Detailed Execution Flow**:

```cpp
// Call: TypeUniquer::get<IntegerType>(ctx, 32, Signless)

1. Create Key:
   KeyTy key = tuple(32, Signless)
   hash = hash_combine(32, Signless) = 0xABCD1234

2. Look up in storageTypes:
   TypeID id = TypeID::get<IntegerType>()
   map = storageTypes[id]  // Get cache for IntegerType

   if (map.contains(key)):
       return map[key]  // Cache hit!

3. Cache miss - construct new:
   allocator = uniquer.getAllocator()
   void *mem = allocator.allocate<IntegerTypeStorage>()
   storage = new (mem) IntegerTypeStorage(32, Signless)

   map[key] = storage  // Cache for future

4. Return wrapped type:
   return IntegerType(storage)
```

### Memory Layout Example

After calling:
```cpp
Type i32_1 = IntegerType::get(ctx, 32, Signless);
Type i32_2 = IntegerType::get(ctx, 32, Signless);
Type i64 = IntegerType::get(ctx, 64, Signless);
```

**Memory State**:

```
StorageUniquer @ 0x1000
├── storageTypes: DenseMap<TypeID, ParametricStorageUniquer*>
│   └── TypeID::get<IntegerType>() → ParametricStorageUniquer @ 0x2000
│       └── cache: DenseMap<KeyTy, BaseStorage*>
│           ├── Key{32, Signless} → IntegerTypeStorage @ 0x3000
│           │   ├── width: 32
│           │   └── signedness: Signless
│           └── Key{64, Signless} → IntegerTypeStorage @ 0x4000
│               ├── width: 64
│               └── signedness: Signless
│
└── allocator: BumpPtrAllocator @ 0x5000
    ├── Slab 1: [IntegerTypeStorage @ 0x3000, ...]
    └── Slab 2: [IntegerTypeStorage @ 0x4000, ...]

Result:
  i32_1.impl = 0x3000  ← Same pointer!
  i32_2.impl = 0x3000  ← Same pointer!
  i64.impl   = 0x4000  ← Different pointer
```

### Singleton Types

**File**: [mlir/lib/IR/StorageUniquerSupport.cpp:120-150](../../../mlir/lib/IR/StorageUniquerSupport.cpp#L120-L150)

For singleton types (parameterless), the uniquer uses a simpler path:

```cpp
template <typename ConcreteType>
ConcreteType TypeUniquer::get(MLIRContext *ctx) {
  TypeID id = TypeID::get<ConcreteType>();
  StorageUniquer &uniquer = ctx->getImpl().types;

  // Check singleton cache
  BaseStorage *&storage = uniquer.getSingletonStorage(id);

  if (!storage) {
    // First time - allocate
    using StorageType = typename ConcreteType::ImplType;
    storage = StorageType::construct(uniquer.getAllocator());
  }

  return ConcreteType(static_cast<typename ConcreteType::ImplType*>(storage));
}
```

**Example**:
```cpp
Type f32_1 = Float32Type::get(ctx);  // Allocates @ 0x6000
Type f32_2 = Float32Type::get(ctx);  // Returns @ 0x6000 (cached)

// Storage:
singletonStorageTypes[TypeID::get<Float32Type>()] = 0x6000
```

### Pre-cached Types in MLIRContext

The selected code pre-caches common types:

```cpp
impl->f32Ty = TypeUniquer::get<Float32Type>(this);
impl->int32Ty = TypeUniquer::get<IntegerType>(this, 32, IntegerType::Signless);
```

**Why?**
1. **Performance**: Types like `i32`, `f32` are used everywhere
2. **Fast access**: Direct member access instead of hash lookup
3. **Initialization guarantee**: Types exist before any dialect code runs

**Access**:
```cpp
// Fast path - no hash lookup!
Type f32 = Float32Type::get(ctx);
// Equivalent to:
Type f32 = ctx->getImpl().f32Ty;  // Direct member access
```

---

## 2. AttributeUniquer

### Architecture

Similar to TypeUniquer, but for attributes (constant data attached to operations/types).

```
AttributeUniquer (per MLIRContext)
├── StorageUniquer
│   ├── storageTypes: DenseMap<TypeID, ParametricStorageUniquer*>
│   │   ├── IntegerAttr cache
│   │   ├── StringAttr cache
│   │   ├── ArrayAttr cache
│   │   └── DictionaryAttr cache
│   └── singletonStorageTypes: DenseMap<TypeID, BaseStorage*>
│       ├── UnitAttr (singleton)
│       └── UnknownLoc (singleton)
```

### Attribute Storage Example

**IntegerAttr Storage**:

**File**: [mlir/lib/IR/BuiltinAttributes.cpp:120-160](../../../mlir/lib/IR/BuiltinAttributes.cpp#L120-L160)

```cpp
struct IntegerAttrStorage : public AttributeStorage {
  IntegerAttrStorage(Type type, const APInt &value)
      : type(type), value(value) {}

  using KeyTy = std::pair<Type, APInt>;

  bool operator==(const KeyTy &key) const {
    return key.first == type && key.second == value;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static IntegerAttrStorage *construct(StorageAllocator &allocator,
                                       const KeyTy &key) {
    // Copy APInt into context allocator
    APInt valueCopy = key.second;
    if (!valueCopy.isSingleWord()) {
      // Allocate space for multi-word APInt
      size_t numWords = valueCopy.getNumWords();
      uint64_t *words = allocator.allocate<uint64_t>(numWords);
      std::copy_n(valueCopy.getRawData(), numWords, words);
      valueCopy = APInt(valueCopy.getBitWidth(), ArrayRef(words, numWords));
    }

    return new (allocator.allocate<IntegerAttrStorage>())
        IntegerAttrStorage(key.first, valueCopy);
  }

  Type type;
  APInt value;  // Arbitrary-precision integer
};
```

### Pre-cached Attributes

The selected code shows commonly-used attributes cached at initialization:

```cpp
impl->unknownLocAttr = AttributeUniquer::get<UnknownLoc>(this);
impl->falseAttr = IntegerAttr::getBoolAttrUnchecked(impl->int1Ty, false);
impl->trueAttr = IntegerAttr::getBoolAttrUnchecked(impl->int1Ty, true);
impl->unitAttr = AttributeUniquer::get<UnitAttr>(this);
impl->emptyDictionaryAttr = DictionaryAttr::getEmptyUnchecked(this);
impl->emptyStringAttr = StringAttr::getEmptyStringAttrUnchecked(this);
```

**Why pre-cache?**
- `true`/`false`: Used in every boolean attribute
- `UnitAttr`: Common marker attribute (presence = true)
- Empty dictionary/string: Default values for many constructs

### Usage Example

```cpp
// Creating integer attributes
Attribute i32_42_a = IntegerAttr::get(i32Type, 42);  // First call - allocates
Attribute i32_42_b = IntegerAttr::get(i32Type, 42);  // Cache hit!
Attribute i32_100 = IntegerAttr::get(i32Type, 100);  // Different value - allocates

// Result:
i32_42_a.impl == i32_42_b.impl  // true - same pointer!
i32_42_a.impl != i32_100.impl   // true - different values
```

**Memory Layout**:

```
AttributeUniquer
└── cache[TypeID::get<IntegerAttr>()]
    ├── Key{i32Type, APInt(42)} → IntegerAttrStorage @ 0x7000
    │   ├── type: i32Type
    │   └── value: APInt(42)
    └── Key{i32Type, APInt(100)} → IntegerAttrStorage @ 0x8000
        ├── type: i32Type
        └── value: APInt(100)
```

---

## 3. AffineUniquer

### Purpose

Affine expressions and maps are heavily used in MLIR for loop transformations, memory access patterns, and polyhedral optimizations. Examples:

```mlir
// Affine map: (d0, d1)[s0] -> (d0 + 2*d1 + s0)
#map = affine_map<(d0, d1)[s0] -> (d0 + 2*d1 + s0)>

// Affine expressions:
d0 + 2*d1     // AffineBinaryOpExpr (d0, +, 2*d1)
2*d1          // AffineBinaryOpExpr (2, *, d1)
s0            // AffineSymbolExpr
```

### Architecture

```
AffineUniquer (per MLIRContext)
├── StorageUniquer
│   └── Registered parametric types:
│       ├── AffineBinaryOpExprStorage (for +, -, *, ceildiv, floordiv, mod)
│       ├── AffineConstantExprStorage (for constant values)
│       ├── AffineDimExprStorage (for dimension references d0, d1, ...)
│       ├── AffineMapStorage (for complete affine maps)
│       └── IntegerSetStorage (for affine constraints)
```

### Storage Registration

The selected code shows registration of storage types:

```cpp
impl->affineUniquer.registerParametricStorageType<AffineBinaryOpExprStorage>();
impl->affineUniquer.registerParametricStorageType<AffineConstantExprStorage>();
impl->affineUniquer.registerParametricStorageType<AffineDimExprStorage>();
impl->affineUniquer.registerParametricStorageType<AffineMapStorage>();
impl->affineUniquer.registerParametricStorageType<IntegerSetStorage>();
```

**What does registration do?**

```cpp
template <typename Storage>
void StorageUniquer::registerParametricStorageType() {
  TypeID id = TypeID::get<Storage>();

  // Create a new cache for this storage type
  impl->storageTypes[id] = std::make_unique<ParametricStorageUniquer>();
}
```

Creates an empty cache that will be populated on-demand.

### Example: AffineBinaryOpExprStorage

**File**: [mlir/lib/IR/AffineExpr.cpp:80-120](../../../mlir/lib/IR/AffineExpr.cpp#L80-L120)

```cpp
struct AffineBinaryOpExprStorage : public AffineExprStorage {
  AffineExpr lhs;
  AffineExpr rhs;

  using KeyTy = std::tuple<unsigned, AffineExpr, AffineExpr>;  // kind, lhs, rhs

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == static_cast<unsigned>(getKind()) &&
           std::get<1>(key) == lhs &&
           std::get<2>(key) == rhs;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  static AffineBinaryOpExprStorage *construct(StorageAllocator &allocator,
                                              const KeyTy &key) {
    auto *result = allocator.allocate<AffineBinaryOpExprStorage>();
    // Initialize with placement new
    return new (result) AffineBinaryOpExprStorage(
        static_cast<AffineExprKind>(std::get<0>(key)),
        std::get<1>(key),  // lhs
        std::get<2>(key)); // rhs
  }
};
```

### Building Affine Expressions

```cpp
MLIRContext *ctx = ...;

// Build: d0 + d1
AffineExpr d0 = getAffineDimExpr(0, ctx);  // Uniqued
AffineExpr d1 = getAffineDimExpr(1, ctx);  // Uniqued
AffineExpr sum = d0 + d1;                  // Creates AffineBinaryOpExpr - uniqued!

// Build: d0 + d1 (again)
AffineExpr d0_2 = getAffineDimExpr(0, ctx);  // Returns cached d0
AffineExpr d1_2 = getAffineDimExpr(1, ctx);  // Returns cached d1
AffineExpr sum_2 = d0_2 + d1_2;              // Returns cached sum!

// Result:
sum.impl == sum_2.impl  // true - same pointer!
```

**Uniquing Process**:

```
1. d0 + d1 triggers:
   AffineBinaryOpExpr::get(Add, d0, d1)

2. Create key:
   KeyTy key = {Add, d0, d1}
   hash = hash_combine(Add, d0.impl, d1.impl)

3. Look up in affineUniquer cache:
   if (cache.contains(key)):
       return cache[key]  // Cache hit!

4. Cache miss:
   storage = AffineBinaryOpExprStorage::construct(allocator, key)
   storage->lhs = d0
   storage->rhs = d1
   cache[key] = storage

5. Return:
   return AffineBinaryOpExpr(storage)
```

### Complex Example: Affine Map

```mlir
// MLIR: #map = affine_map<(d0, d1)[s0] -> (d0 + 2*d1 + s0)>
```

**Construction**:

```cpp
AffineExpr d0 = getAffineDimExpr(0, ctx);
AffineExpr d1 = getAffineDimExpr(1, ctx);
AffineExpr s0 = getAffineSymbolExpr(0, ctx);

AffineExpr two = getAffineConstantExpr(2, ctx);
AffineExpr twoD1 = two * d1;         // AffineBinaryOpExpr(Mul, 2, d1)
AffineExpr sum1 = d0 + twoD1;        // AffineBinaryOpExpr(Add, d0, twoD1)
AffineExpr result = sum1 + s0;       // AffineBinaryOpExpr(Add, sum1, s0)

AffineMap map = AffineMap::get(
    /*dimCount=*/2,
    /*symbolCount=*/1,
    /*results=*/{result},
    ctx);
```

**Memory Layout**:

```
AffineUniquer
├── AffineDimExprStorage[0] @ 0x9000 (d0)
├── AffineDimExprStorage[1] @ 0x9100 (d1)
├── AffineSymbolExprStorage[0] @ 0x9200 (s0)
├── AffineConstantExprStorage{2} @ 0x9300
├── AffineBinaryOpExprStorage @ 0x9400
│   ├── kind: Mul
│   ├── lhs: 0x9300 (const 2)
│   └── rhs: 0x9100 (d1)
├── AffineBinaryOpExprStorage @ 0x9500
│   ├── kind: Add
│   ├── lhs: 0x9000 (d0)
│   └── rhs: 0x9400 (2*d1)
└── AffineBinaryOpExprStorage @ 0x9600
    ├── kind: Add
    ├── lhs: 0x9500 (d0 + 2*d1)
    └── rhs: 0x9200 (s0)

AffineMapStorage @ 0x9700
├── numDims: 2
├── numSymbols: 1
└── results: [0x9600]  // Points to root expression
```

---

## Implementation Details

### Hash-Based Lookup

All uniquers use **hash tables** (LLVM's `DenseMap`):

```cpp
template <typename KeyTy, typename Storage>
class ParametricStorageUniquer {
  DenseMap<KeyTy, Storage*> cache;

  Storage *get(const KeyTy &key, llvm::function_ref<Storage*()> constructorFn) {
    auto it = cache.find(key);
    if (it != cache.end())
      return it->second;  // Cache hit!

    // Cache miss - construct
    Storage *storage = constructorFn();
    cache[key] = storage;
    return storage;
  }
};
```

### Memory Allocation: BumpPtrAllocator

**File**: [llvm/include/llvm/Support/Allocator.h](../../../../llvm/include/llvm/Support/Allocator.h)

All storage objects are allocated from a **bump pointer allocator**:

```cpp
class BumpPtrAllocator {
  struct Slab {
    char *ptr;
    size_t size;
    size_t allocated;
  };

  std::vector<Slab> slabs;
  Slab *current;

public:
  void *allocate(size_t size, size_t alignment) {
    // Align pointer
    char *result = alignPtr(current->ptr + current->allocated, alignment);

    // Check if slab has space
    if (result + size > current->ptr + current->size) {
      // Need new slab
      allocateNewSlab(size);
      result = current->ptr;
    }

    current->allocated = (result + size) - current->ptr;
    return result;
  }

  // No individual deallocation!
  // All memory freed when allocator is destroyed
};
```

**Benefits**:
- **Fast allocation**: Just bump a pointer
- **No fragmentation**: Linear allocation
- **Bulk deallocation**: Free all at once when context destroyed
- **Cache friendly**: Related objects allocated contiguously

**Tradeoff**: Cannot free individual objects (but that's fine - types/attributes live for entire context lifetime).

### Thread Safety

**File**: [mlir/lib/IR/StorageUniquerSupport.cpp:50-80](../../../mlir/lib/IR/StorageUniquerSupport.cpp#L50-L80)

```cpp
class ParametricStorageUniquer {
  DenseMap<KeyTy, Storage*> cache;
  llvm::sys::SmartRWMutex<true> mutex;  // Reader-writer lock

  Storage *get(const KeyTy &key, ConstructorFn constructorFn) {
    // Fast path: read lock for lookup
    {
      llvm::sys::SmartScopedReader<true> lock(mutex);
      auto it = cache.find(key);
      if (it != cache.end())
        return it->second;
    }

    // Slow path: write lock for insertion
    {
      llvm::sys::SmartScopedWriter<true> lock(mutex);

      // Double-check (another thread might have inserted)
      auto it = cache.find(key);
      if (it != cache.end())
        return it->second;

      // Construct and insert
      Storage *storage = constructorFn();
      cache[key] = storage;
      return storage;
    }
  }
};
```

**Pattern**: Double-checked locking
1. Read lock: Check if already cached (common case - fast)
2. Write lock: Insert new entry (rare case)
3. Double-check after acquiring write lock (avoid race)

---

## Performance Characteristics

### Time Complexity

| Operation | First Call | Subsequent Calls |
|-----------|-----------|------------------|
| Singleton type (f32) | O(1) | O(1) - direct member access |
| Parametric type (i32) | O(1) hash + allocate | O(1) hash lookup |
| Affine expression | O(n) for n sub-expressions | O(1) if entire expression cached |
| Type comparison | - | O(1) - pointer comparison |

### Space Complexity

- **One copy per unique structure**: O(number of unique types/attrs)
- **No duplication**: Same parameters = same pointer
- **Cache overhead**: Hash table ~1.3x the number of unique entries

### Benchmark Example

```cpp
// Without uniquing:
for (int i = 0; i < 1000000; i++) {
  Type t = IntegerType::get(ctx, 32, Signless);  // 1M allocations!
}
// Memory: 1M * sizeof(IntegerTypeStorage) = ~16 MB

// With uniquing:
for (int i = 0; i < 1000000; i++) {
  Type t = IntegerType::get(ctx, 32, Signless);  // 1 allocation!
}
// Memory: 1 * sizeof(IntegerTypeStorage) = ~16 bytes
```

---

## Summary

### TypeUniquer

- **Purpose**: Ensure identical types share memory
- **Structure**: Hash table of type storages, keyed by parameters
- **Access**: `TypeUniquer::get<ConcreteType>(ctx, params...)`
- **Allocation**: BumpPtrAllocator (fast, no individual free)
- **Thread-safe**: Reader-writer locks

### AttributeUniquer

- **Purpose**: Ensure identical attributes share memory
- **Structure**: Same as TypeUniquer, but for constant data
- **Examples**: IntegerAttr, StringAttr, DictionaryAttr
- **Pre-cached**: true, false, unit, empty collections

### AffineUniquer

- **Purpose**: Unique affine expressions and maps
- **Structure**: Separate uniquer for affine constructs
- **Registration**: Must register storage types at init
- **Composition**: Complex expressions built from uniqued sub-expressions

### Key Benefits

1. **Memory efficiency**: 1000s of uses → 1 allocation
2. **Fast comparison**: Pointer equality instead of deep compare
3. **Canonicalization**: Same structure = same pointer
4. **Cache friendly**: Linear allocation in bump allocator
5. **Thread-safe**: Concurrent reads, safe writes

### Common Pattern

```cpp
// 1. Define storage class with KeyTy and construct()
struct MyTypeStorage : TypeStorage {
  using KeyTy = std::tuple<...>;
  static MyTypeStorage *construct(allocator, key) { ... }
  bool operator==(const KeyTy &key) { ... }
};

// 2. Type wrapper
class MyType : public Type {
  using ImplType = MyTypeStorage;
  static MyType get(MLIRContext *ctx, ...) {
    return TypeUniquer::get<MyType>(ctx, ...);
  }
};

// 3. Users get automatic uniquing
MyType t1 = MyType::get(ctx, params);  // Allocates
MyType t2 = MyType::get(ctx, params);  // Cached!
assert(t1 == t2);  // Pointer comparison: true
```

---

## Related Files

| Component | File | Description |
|-----------|------|-------------|
| **Type Uniquer** | [StorageUniquerSupport.h](../../../mlir/include/mlir/IR/StorageUniquerSupport.h) | Template interface |
| **Storage Uniquer** | [StorageUniquer.h](../../../mlir/include/mlir/Support/StorageUniquer.h) | Core implementation |
| **Type Storage** | [TypeSupport.h](../../../mlir/include/mlir/IR/TypeSupport.h) | Base classes |
| **Builtin Types** | [BuiltinTypes.cpp](../../../mlir/lib/IR/BuiltinTypes.cpp) | Storage implementations |
| **Attributes** | [BuiltinAttributes.cpp](../../../mlir/lib/IR/BuiltinAttributes.cpp) | Attribute storages |
| **Affine** | [AffineExpr.cpp](../../../mlir/lib/IR/AffineExpr.cpp) | Affine storages |
| **Context Init** | [MLIRContext.cpp:310-357](../../../mlir/lib/IR/MLIRContext.cpp#L310-L357) | Pre-cached types/attrs |

This uniquing system is fundamental to MLIR's performance and is used by every dialect!
