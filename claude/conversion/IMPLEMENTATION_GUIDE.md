# MLIR Dialect Conversion - Implementation Deep Dive

This guide explains how MLIR's dialect conversion infrastructure works internally, with links to source code and detailed explanations of the algorithms and data structures involved.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [The Conversion Driver](#the-conversion-driver)
3. [Value Remapping](#value-remapping)
4. [Type Conversion](#type-conversion)
5. [Materialization](#materialization)
6. [Rollback and Transactions](#rollback-and-transactions)
7. [Pattern Application](#pattern-application)

## Architecture Overview

The dialect conversion infrastructure is implemented primarily in:
- [mlir/include/mlir/Transforms/DialectConversion.h](../../mlir/include/mlir/Transforms/DialectConversion.h) - Public APIs
- [mlir/lib/Transforms/Utils/DialectConversion.cpp](../../mlir/lib/Transforms/Utils/DialectConversion.cpp) - Implementation (4000+ lines)

### Key Classes

```
OperationConverter              # Main conversion driver
├── ConversionPatternRewriterImpl   # Rewriter implementation
│   ├── Mapping                     # Value remapping (old → new)
│   ├── BlockActions                # Track IR modifications
│   └── RewriteRecordEntry          # Record for rollback
├── TypeConverter                   # Type conversion logic
└── ConversionTarget                # Legality specification
```

## The Conversion Driver

The main entry point is `OperationConverter::convert()`:

**Source:** [mlir/lib/Transforms/Utils/DialectConversion.cpp:2876](../../mlir/lib/Transforms/Utils/DialectConversion.cpp)

### Algorithm Overview

```cpp
LogicalResult OperationConverter::convert() {
  // 1. Initialize: Build list of all operations to consider
  // 2. Compute conversion set: Which operations need conversion?
  // 3. Iterate until fixed point:
  //    a. Try to apply patterns to illegal operations
  //    b. Check if converted operations are now legal
  //    c. If any pattern fails, rollback that conversion
  // 4. If all operations legal: SUCCESS
  // 5. Otherwise: FAILURE (rollback everything)
}
```

### Implementation Details

#### Step 1: Initialization

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:2900
SmallVector<Operation *> worklist;
// Walk the IR and add all operations to worklist
rootOp->walk([&](Operation *op) {
  worklist.push_back(op);
});
```

The worklist contains all operations that might need conversion. Operations are processed in a specific order to ensure dependencies are handled correctly.

#### Step 2: Compute Conversion Set

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:2509
LogicalResult computeConversionSet(
    Operation *rootOp,
    ConversionTarget &target,
    DenseSet<Operation *> &opsToConvert) {

  // For each operation:
  for (Operation *op : worklist) {
    // Check legality via ConversionTarget
    if (!target.isLegal(op)) {
      opsToConvert.insert(op);
    }
  }
}
```

**Legality checking:**

The `ConversionTarget` maintains several data structures:

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:340
class ConversionTarget {
  // Legal/Illegal operation tracking
  DenseMap<OperationName, LegalityAction> legalOperations;

  // Dynamic legality callbacks
  DenseMap<OperationName, LegalityPredicate> dynamicLegalityPredicates;

  // Dialect-level legality
  DenseSet<Dialect *> legalDialects;
  DenseSet<Dialect *> illegalDialects;
};
```

When checking if an operation is legal:

1. Check dialect-level legality first
2. Check operation-specific legality
3. If marked as dynamically legal, invoke callback
4. Default: illegal if not explicitly marked legal

#### Step 3: Pattern Application Loop

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:2950
for (Operation *op : opsToConvert) {
  // Try to match and apply patterns
  if (failed(legalizeOp(op, rewriterImpl))) {
    // Pattern application failed
    // Rollback is automatic via RAII
    return failure();
  }
}
```

**Pattern matching:**

Patterns are stored in a `RewritePatternSet` and organized by:

```cpp
// Source: mlir/lib/IR/PatternMatch.cpp:390
struct RewritePatternSet {
  // Patterns are stored in a map keyed by operation name
  DenseMap<OperationName, SmallVector<RewritePattern *>> patterns;

  // Patterns are tried in order of benefit (highest first)
  // Benefit is set when registering the pattern
};
```

When applying patterns:

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:2650
LogicalResult legalizeOp(Operation *op, ...) {
  // Get patterns for this operation
  auto patterns = getPatterns(op->getName());

  // Sort by benefit (highest first)
  llvm::stable_sort(patterns, [](auto *lhs, auto *rhs) {
    return lhs->getBenefit() > rhs->getBenefit();
  });

  // Try each pattern
  for (auto *pattern : patterns) {
    if (succeeded(pattern->matchAndRewrite(op, rewriter))) {
      return success();  // Pattern succeeded
    }
  }

  return failure();  // No pattern matched
}
```

## Value Remapping

One of the most critical aspects of conversion is tracking how values change during conversion. This is handled by the **value mapping** system.

**Source:** [mlir/lib/Transforms/Utils/DialectConversion.cpp:686](../../mlir/lib/Transforms/Utils/DialectConversion.cpp)

### Data Structure

```cpp
class ConversionPatternRewriterImpl {
  // Maps original SSA values to their converted forms
  // Key: Original value
  // Value: Converted value (possibly type-converted)
  DenseMap<Value, Value> mapping;

  // During conversion, this is updated whenever:
  // 1. A pattern replaces an operation
  // 2. Type conversion creates new values
  // 3. Materialization inserts casts
};
```

### How Remapping Works

When you call `OpAdaptor::getLhs()` in a conversion pattern:

```cpp
// Your pattern code:
struct MyPattern : OpConversionPattern<MyOp> {
  matchAndRewrite(MyOp op, OpAdaptor adaptor, ...) {
    Value lhs = adaptor.getLhs();  // ← What happens here?
  }
};
```

Internally, `OpAdaptor` does:

```cpp
// Generated by TableGen
class MyOpAdaptor {
  Value getLhs() {
    // Get the original operand
    Value originalLhs = op.getLhs();

    // Look up in conversion mapping
    Value convertedLhs = rewriter.getRemappedValue(originalLhs);

    return convertedLhs;  // May be type-converted!
  }
};
```

The `getRemappedValue` implementation:

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:1843
Value ConversionPatternRewriter::getRemappedValue(Value key) {
  // Look up in mapping
  auto it = impl->mapping.find(key);
  if (it != impl->mapping.end())
    return it->second;

  // If not found, return original
  // (This can happen for values that don't need conversion)
  return key;
}
```

### Updating Mappings

When a pattern replaces an operation:

```cpp
// Your pattern code:
rewriter.replaceOp(op, newValue);
```

Internally:

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:1925
void ConversionPatternRewriter::replaceOp(Operation *op, ValueRange newValues) {
  // Update mappings: old values → new values
  for (auto [oldResult, newValue] : llvm::zip(op->getResults(), newValues)) {
    impl->mapping[oldResult] = newValue;
  }

  // Schedule old operation for deletion
  // (Actual deletion happens after conversion succeeds)
  impl->notifyOperationRemoved(op);
}
```

## Type Conversion

Type conversion is handled by the `TypeConverter` class, which maintains conversion rules and materialization callbacks.

**Source:** [mlir/include/mlir/Transforms/DialectConversion.h:555](../../mlir/include/mlir/Transforms/DialectConversion.h)

### Conversion Functions

The `TypeConverter` stores conversion functions in a vector:

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:480
class TypeConverter {
  // Conversion functions, tried in REVERSE order of registration
  SmallVector<ConversionCallbackFn, 4> conversions;

  // Materialization callbacks
  MaterializationCallbackFn targetMaterialization;
  MaterializationCallbackFn sourceMaterialization;
  MaterializationCallbackFn argumentMaterialization;
};
```

When converting a type:

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:520
Type TypeConverter::convertType(Type t) {
  // Try conversion functions in REVERSE order
  // (Most recently registered tried first)
  for (auto it = conversions.rbegin(); it != conversions.rend(); ++it) {
    if (Type result = (*it)(t))
      return result;
  }

  // If no conversion, return original type
  return t;
}
```

**Why reverse order?** This allows you to override earlier registrations by registering more specific conversions later.

### Example Flow

```cpp
TypeConverter converter;

// Register general conversion
converter.addConversion([](Type type) { return type; });  // Identity

// Register specific conversion (tried first due to reverse order)
converter.addConversion([](FixedPointType type) {
  return IntegerType::get(type.getContext(), type.getWidth());
});

// Usage:
Type fp = FixedPointType::get(context, 16, 8);
Type converted = converter.convertType(fp);
// Result: i16
```

## Materialization

Materialization handles type mismatches by inserting cast operations. This is necessary when converted and unconverted IR need to interact.

**Source:** [mlir/lib/Transforms/Utils/DialectConversion.cpp:1157](../../mlir/lib/Transforms/Utils/DialectConversion.cpp)

### When Materialization Happens

Consider this scenario:

```mlir
// Before conversion:
%fp1 = some_unconverted_op : !typed.fixed<16,8>
%fp2 = typed.fixed_add %fp1, %fp1 : !typed.fixed<16,8>

// During conversion:
// - typed.fixed_add is converted
// - some_unconverted_op remains (partial conversion)

// Problem: typed.fixed_add expects i16 operands after conversion
//          but some_unconverted_op produces !typed.fixed<16,8>
```

### Materialization Algorithm

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:1200
Value materializeConversion(Type targetType, Value sourceValue) {
  Type sourceType = sourceValue.getType();

  // If types match, no materialization needed
  if (sourceType == targetType)
    return sourceValue;

  // Try target materialization (source → target)
  if (auto callback = typeConverter->getTargetMaterialization()) {
    if (Value mat = callback(builder, targetType, sourceValue, loc))
      return mat;
  }

  // If no materialization available, fail
  return nullptr;
}
```

### Three Types of Materialization

1. **Target Materialization**: Source type → Target type

   ```cpp
   // Example: FixedPoint → Integer
   converter.addTargetMaterialization([](OpBuilder &builder, Type targetType,
                                         ValueRange inputs, Location loc) {
     if (targetType.isa<IntegerType>() && inputs[0].getType().isa<FixedPointType>())
       return builder.create<FixedToIntOp>(loc, targetType, inputs[0]);
     return nullptr;
   });
   ```

   **When used**: Converted operation needs target-type operands, but has source-type values.

2. **Source Materialization**: Target type → Source type

   ```cpp
   // Example: Integer → FixedPoint
   converter.addSourceMaterialization([](OpBuilder &builder, Type sourceType,
                                         ValueRange inputs, Location loc) {
     if (sourceType.isa<FixedPointType>() && inputs[0].getType().isa<IntegerType>())
       return builder.create<IntToFixedOp>(loc, sourceType, inputs[0]);
     return nullptr;
   });
   ```

   **When used**: Unconverted operation needs source-type operands, but has target-type values.

3. **Argument Materialization**: For block/function arguments

   ```cpp
   converter.addArgumentMaterialization([](OpBuilder &builder, Type resultType,
                                           ValueRange inputs, Location loc) {
     // Similar to source materialization, but specifically for arguments
   });
   ```

   **When used**: During signature conversion when block arguments change type.

### Materialization Resolution

After conversion, the framework may have inserted temporary materialization operations. These are resolved in a post-processing step:

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:1300
LogicalResult legalizeUnresolvedMaterializations() {
  // For each materialization:
  for (auto *mat : pendingMaterializations) {
    // Check if materialization is still needed
    if (mat->use_empty()) {
      // No uses, can be removed
      mat->erase();
      continue;
    }

    // Verify materialization is legal according to target
    if (!target.isLegal(mat)) {
      // Illegal materialization - conversion fails
      return failure();
    }
  }
}
```

## Rollback and Transactions

The conversion infrastructure supports **rollback** - if conversion fails, all changes are undone. This is implemented using a transaction/action system.

**Source:** [mlir/lib/Transforms/Utils/DialectConversion.cpp:800](../../mlir/lib/Transforms/Utils/DialectConversion.cpp)

### Action Recording

Every IR modification is recorded as an action:

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:850
enum class ActionKind {
  CreateOperation,   // New operation created
  EraseOperation,    // Operation erased
  ReplaceOperation,  // Operation replaced
  MoveOperation,     // Operation moved
  InlineBlock,       // Block inlined
  SplitBlock,        // Block split
  // ... more action types
};

struct RewriteRecordEntry {
  ActionKind kind;
  Operation *op;
  // Additional data depending on action type
};
```

When you call `rewriter.create()`:

```cpp
// Your code:
Value result = rewriter.create<arith::AddIOp>(loc, lhs, rhs);

// Internally:
Operation *ConversionPatternRewriter::createOperation(...) {
  // Actually create the operation
  Operation *op = OpBuilder::create(state);

  // Record the action for potential rollback
  impl->recordAction(RewriteRecordEntry{
    .kind = ActionKind::CreateOperation,
    .op = op
  });

  return op;
}
```

### Rollback Implementation

If a pattern fails or conversion is unsuccessful:

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:1500
void ConversionPatternRewriterImpl::rollback() {
  // Process actions in REVERSE order
  for (auto it = actions.rbegin(); it != actions.rend(); ++it) {
    RewriteRecordEntry &entry = *it;

    switch (entry.kind) {
      case ActionKind::CreateOperation:
        // Undo creation: erase the operation
        entry.op->erase();
        break;

      case ActionKind::EraseOperation:
        // Undo erasure: re-insert the operation
        entry.block->getOperations().insert(entry.position, entry.op);
        break;

      case ActionKind::ReplaceOperation:
        // Undo replacement: restore old values
        for (auto [result, oldValue] : entry.replacements) {
          result.replaceAllUsesWith(oldValue);
        }
        break;

      // ... handle other action types
    }
  }

  // Clear all actions
  actions.clear();
}
```

### Transaction Scopes

Patterns can use transaction scopes for speculative rewriting:

```cpp
struct MyPattern : OpConversionPattern<MyOp> {
  matchAndRewrite(MyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Start transaction
    ConversionPatternRewriter::Transaction transaction(rewriter);

    // Try to do something
    Value result = tryConversion(op, rewriter);

    if (!result) {
      // Failed - rollback this transaction
      return failure();
    }

    // Succeeded - commit transaction
    transaction.commit();
    return success();
  }
};
```

## Pattern Application

Patterns are applied using a sophisticated matching and rewriting system.

**Source:** [mlir/lib/Transforms/Utils/DialectConversion.cpp:2650](../../mlir/lib/Transforms/Utils/DialectConversion.cpp)

### Pattern Matching

```cpp
LogicalResult legalizeOp(Operation *op, RewritePatternSet &patterns,
                        ConversionPatternRewriter &rewriter) {

  // Get patterns that might apply to this operation
  auto candidates = patterns.getMatchingPatterns(op->getName());

  // Sort by benefit (descending)
  llvm::stable_sort(candidates, [](auto *lhs, auto *rhs) {
    return lhs->getBenefit() > rhs->getBenefit();
  });

  // Try each pattern
  for (auto *pattern : candidates) {
    // Check if pattern matches
    if (!pattern->match(op))
      continue;

    // Try to apply the pattern (with transaction for rollback)
    ConversionPatternRewriter::Transaction transaction(rewriter);

    if (succeeded(pattern->matchAndRewrite(op, rewriter))) {
      // Pattern succeeded - commit changes
      transaction.commit();
      return success();
    }

    // Pattern failed - transaction auto-rollbacks on destruction
  }

  // No pattern succeeded
  return failure();
}
```

### Pattern Benefit

Patterns have a **benefit** score that determines application order:

```cpp
patterns.add<MyPattern>(context, /*benefit=*/10);  // High priority
patterns.add<FallbackPattern>(context, /*benefit=*/1);  // Low priority
```

**Guidelines:**
- Default benefit: 1
- Specific patterns: 5-10
- General fallback patterns: 1
- Optimization patterns: 10+

Higher benefit patterns are tried first. This allows you to:
- Prioritize optimized lowerings over simple ones
- Ensure specific patterns are tried before general ones
- Control pattern application order explicitly

### Pattern Debugging

Enable debug output:

```bash
# Method 1: Debug flag
MLIR_ENABLE_DUMP=1 your-tool input.mlir -your-pass -debug-only=dialect-conversion

# Method 2: Pattern rewriter tracing
MLIR_ENABLE_DUMP=1 your-tool input.mlir -your-pass -debug-only=pattern-match
```

This shows:
- Which patterns are being tried
- Why patterns fail to match
- Which pattern successfully applies
- Value remappings
- Rollback information

## Advanced Topics

### Nested Pattern Application

Sometimes a pattern needs to apply other patterns. This is supported:

```cpp
struct OuterPattern : OpConversionPattern<OuterOp> {
  matchAndRewrite(OuterOp op, ...) const override {
    // Create intermediate operations
    auto innerOp = rewriter.create<InnerOp>(...);

    // The conversion framework will later try to convert innerOp
    // using registered patterns

    return success();
  }
};
```

The conversion driver will detect the new `InnerOp` and try to legalize it.

### Signature Conversion

Function signature conversion is a special case handled by `TypeConverter::SignatureConversion`:

```cpp
// Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:700
class SignatureConversion {
  // Maps original argument indices to new argument indices
  // Supports 1:N mapping (one arg can become multiple args)
  SmallVector<SmallVector<unsigned>> argMapping;

  // Converted types
  SmallVector<Type> convertedTypes;
};
```

Example usage:

```cpp
TypeConverter::SignatureConversion conversion(funcOp.getNumArguments());

for (auto [idx, argType] : llvm::enumerate(funcOp.getArgumentTypes())) {
  Type converted = typeConverter->convertType(argType);
  conversion.addInputs(idx, converted);
}

// Apply to function
rewriter.applySignatureConversion(&funcOp.getBody(), conversion);
```

### Region Conversion

When converting regions (e.g., function bodies, loop bodies), the framework:

1. Converts block arguments
2. Inserts argument materializations if needed
3. Recursively converts operations in the region
4. Updates terminators

This is all handled automatically, but you can customize:

```cpp
struct MyRegionOp : OpConversionPattern<RegionOp> {
  matchAndRewrite(RegionOp op, ...) const override {
    // Convert region signature
    TypeConverter::SignatureConversion sigConversion(op.getNumArgs());
    // ... fill in sigConversion

    // Apply conversion
    rewriter.applySignatureConversion(&op.getRegion(), sigConversion);

    // Operations inside the region will be converted by the framework
    return success();
  }
};
```

## Performance Considerations

### Pattern Application Cost

Pattern matching is O(P × N) where:
- P = number of patterns
- N = number of operations

Optimize by:
- Using specific pattern types (OpConversionPattern<T>)
- Implementing `match()` to fail fast
- Avoiding expensive checks in `matchAndRewrite()`

### Memory Usage

The conversion framework maintains several data structures:
- Value mappings: O(V) where V = number of values
- Action records: O(A) where A = number of actions
- Materialization ops: O(M) where M = type mismatches

For large IR:
- Use partial conversion when possible
- Clean up materializations aggressively
- Consider streaming conversion for very large IR

### Incremental Conversion

Instead of converting everything at once:

```cpp
// Option 1: Multiple partial conversions
applyPartialConversion(module, target1, patterns1);
applyPartialConversion(module, target2, patterns2);
applyFullConversion(module, target3, patterns3);

// Option 2: Progressive lowering through dialects
// High → Medium → Low → LLVM
```

## Summary

The MLIR dialect conversion infrastructure is a sophisticated system that:

1. **Tracks IR modifications** for rollback
2. **Maintains value mappings** to handle type conversion
3. **Applies patterns iteratively** until all ops are legal
4. **Inserts materializations** to bridge type mismatches
5. **Supports transactions** for safe speculative rewriting

Understanding these internals helps you:
- Write more efficient conversion patterns
- Debug conversion issues effectively
- Design better dialect lowering strategies
- Optimize conversion performance

## Further Reading

- [Pattern Rewriter Documentation](https://mlir.llvm.org/docs/PatternRewriter/)
- [Dialect Conversion Documentation](https://mlir.llvm.org/docs/DialectConversion/)
- [MLIR Rationale](https://mlir.llvm.org/docs/Rationale/)
- Source code deep dives:
  - [DialectConversion.cpp](../../mlir/lib/Transforms/Utils/DialectConversion.cpp)
  - [PatternMatch.cpp](../../mlir/lib/IR/PatternMatch.cpp)
