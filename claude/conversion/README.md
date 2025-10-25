# MLIR Dialect Conversion Tutorial

A comprehensive guide to understanding and using MLIR's dialect conversion infrastructure, with runnable examples that demonstrate real-world conversion workflows.

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Key APIs](#key-apis)
4. [Examples](#examples)
5. [Building and Running](#building-and-running)

## Overview

Dialect conversion in MLIR is the process of transforming operations from one dialect to another. This is fundamental to MLIR's multi-level compilation strategy, where high-level operations are progressively lowered through intermediate representations to target-specific code.

The dialect conversion framework provides:
- **Pattern-based rewriting**: Define how operations transform
- **Type conversion**: Handle type transformations across dialects
- **Legality checking**: Specify what's legal in the target dialect
- **Partial conversion**: Allow mixed dialect IR during transformation
- **Materialization**: Handle type mismatches at IR boundaries

## Core Concepts

### 1. ConversionPattern

A `ConversionPattern` defines how to rewrite an operation. It's the fundamental building block of dialect conversion.

**Key source files:**
- [mlir/include/mlir/Transforms/DialectConversion.h](../../../mlir/include/mlir/Transforms/DialectConversion.h) - Main conversion APIs
- [mlir/lib/Transforms/Utils/DialectConversion.cpp](../../../mlir/lib/Transforms/Utils/DialectConversion.cpp) - Implementation

### 2. TypeConverter

A `TypeConverter` defines how types are converted between dialects. It handles:
- Type conversions (e.g., tensor<f32> → memref<f32>)
- Materialization of conversions when needed
- Signature conversions for function types

### 3. ConversionTarget

A `ConversionTarget` specifies the legality of operations:
- **Legal**: Operations allowed in target IR
- **Illegal**: Operations that must be converted
- **Dynamic**: Legality determined at runtime

### 4. Conversion Drivers

MLIR provides three main conversion drivers:

- **`applyFullConversion`**: All operations must be legal; fails otherwise
- **`applyPartialConversion`**: Some operations can remain illegal
- **`applyAnalysisConversion`**: Dry-run to test if conversion is possible

**Source:** [mlir/lib/Transforms/Utils/DialectConversion.cpp](../../../mlir/lib/Transforms/Utils/DialectConversion.cpp)

## Key APIs

### ConversionPattern

```cpp
class ConversionPattern : public RewritePattern {
public:
  // Match and rewrite with type-converted operands
  virtual LogicalResult matchAndRewrite(
      Operation *op,
      ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const = 0;
};
```

**Implementation details:**
- Defined in [mlir/include/mlir/Transforms/DialectConversion.h](../../../mlir/include/mlir/Transforms/DialectConversion.h)
- Extends `RewritePattern` from the general rewriting framework
- The `operands` parameter contains **type-converted** versions of the original operands
- Uses `ConversionPatternRewriter` which tracks conversions for rollback

### OpConversionPattern<T>

A templated helper for operation-specific patterns:

```cpp
template <typename SourceOp>
class OpConversionPattern : public ConversionPattern {
public:
  OpConversionPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : ConversionPattern(SourceOp::getOperationName(), benefit, context) {}

  // Type-safe matchAndRewrite
  virtual LogicalResult matchAndRewrite(
      SourceOp op,
      OpAdaptor adaptor,  // Contains converted operands
      ConversionPatternRewriter &rewriter) const = 0;
};
```

**Why use OpConversionPattern?**
- Type-safe: Works with specific operation types
- Convenient: `OpAdaptor` provides named accessors for converted operands
- Less boilerplate than raw `ConversionPattern`

**Source:** [mlir/include/mlir/Transforms/DialectConversion.h:1051](../../../mlir/include/mlir/Transforms/DialectConversion.h)

### TypeConverter

```cpp
class TypeConverter {
public:
  // Register a type conversion callback
  void addConversion(ConversionCallbackFn callback);

  // Register materialization callbacks
  void addSourceMaterialization(MaterializationCallbackFn callback);
  void addTargetMaterialization(MaterializationCallbackFn callback);
  void addArgumentMaterialization(MaterializationCallbackFn callback);

  // Convert a type
  Type convertType(Type t);

  // Convert function signatures
  LogicalResult convertSignatureArgs(TypeRange types,
                                     SignatureConversion &result);
};
```

**Materialization callbacks** are critical for handling type mismatches:

1. **Target materialization**: Converts values from source to target types
   - Used when a converted operation needs source-type values
   - Example: Insert cast from `f32` to `i32` when required

2. **Source materialization**: Converts values from target to source types
   - Used when unconverted operations need target-type values
   - Opposite of target materialization

3. **Argument materialization**: Special handling for block/function arguments
   - Used during signature conversion
   - Example: Converting function argument types

**Source:** [mlir/include/mlir/Transforms/DialectConversion.h:555](../../../mlir/include/mlir/Transforms/DialectConversion.h)

### ConversionTarget

```cpp
class ConversionTarget {
public:
  // Mark entire dialect as legal/illegal
  void addLegalDialect<DialectT>();
  void addIllegalDialect<DialectT>();

  // Mark specific operations
  void addLegalOp<OpT>();
  void addIllegalOp<OpT>();
  void addDynamicallyLegalOp<OpT>(DynamicLegalityCallbackFn callback);

  // Check if operation is legal
  bool isLegal(Operation *op) const;
};
```

**Dynamic legality** is powerful for conditional conversion:

```cpp
target.addDynamicallyLegalOp<SomeOp>([](SomeOp op) {
  // Only legal if operands meet certain conditions
  return op.getOperand().getType().isF32();
});
```

**Source:** [mlir/include/mlir/Transforms/DialectConversion.h:301](../../../mlir/include/mlir/Transforms/DialectConversion.h)

### ConversionPatternRewriter

Extends `PatternRewriter` with conversion-specific capabilities:

```cpp
class ConversionPatternRewriter : public PatternRewriter {
public:
  // Get converted value (after type conversion)
  Value getRemappedValue(Value key);

  // Replace operation with new values
  void replaceOp(Operation *op, ValueRange newValues);

  // Create new operations (automatically type-converted)
  Operation *createOperation(const OperationState &state);

  // Signature conversion for regions
  void applySignatureConversion(Region *region,
                                 TypeConverter::SignatureConversion &conversion);
};
```

**Key implementation detail:** The rewriter maintains a mapping from original values to converted values. When you use `getRemappedValue()`, it returns the type-converted version.

**Source:** [mlir/include/mlir/Transforms/DialectConversion.h:835](../../../mlir/include/mlir/Transforms/DialectConversion.h)

## Examples

This tutorial includes three progressive examples:

### Example 1: Simple Arithmetic Lowering

**Files:**
- [include/SimpleArith.td](include/SimpleArith.td) - Custom dialect definition
- [lib/SimpleArithLowering.cpp](lib/SimpleArithLowering.cpp) - Conversion patterns
- [tools/simple-opt.cpp](tools/simple-opt.cpp) - Testing tool

**What it demonstrates:**
- Basic `OpConversionPattern` usage
- Simple operation-to-operation lowering
- `ConversionTarget` configuration
- Full conversion with `applyFullConversion`

**Concepts covered:**
- Pattern registration
- Rewriter API usage
- Lowering passes

### Example 2: Type Conversion

**Files:**
- [include/TypedArith.td](include/TypedArith.td) - Dialect with custom types
- [lib/TypedArithLowering.cpp](lib/TypedArithLowering.cpp) - Type conversion patterns
- [tools/typed-opt.cpp](tools/typed-opt.cpp) - Testing tool

**What it demonstrates:**
- `TypeConverter` usage
- Materialization callbacks
- Signature conversion
- Handling type mismatches

**Concepts covered:**
- Custom type definitions
- Type conversion chains
- Source/target materialization
- Argument type conversion

### Example 3: Partial Conversion

**Files:**
- [include/MixedDialects.td](include/MixedDialects.td) - Multiple dialect definitions
- [lib/PartialLowering.cpp](lib/PartialLowering.cpp) - Partial conversion patterns
- [tools/partial-opt.cpp](tools/partial-opt.cpp) - Testing tool

**What it demonstrates:**
- `applyPartialConversion`
- Dynamic legality checking
- Mixed dialect IR
- Incremental lowering strategies

**Concepts covered:**
- Multi-stage lowering
- Conditional operation legality
- Progressive dialect lowering
- Real-world lowering pipelines

## Building and Running

### Prerequisites

- MLIR built from source
- CMake 3.20+
- C++17 compiler

### Build Instructions

```bash
# From llvm-project directory
cd claude/conversion
mkdir build && cd build

# Configure with your MLIR build
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm

# Build
ninja
```

### Running Examples

```bash
# Example 1: Simple arithmetic lowering
./build/bin/simple-opt test/simple-arith.mlir -convert-simple-to-arith

# Example 2: Type conversion
./build/bin/typed-opt test/typed-arith.mlir -convert-typed-to-std

# Example 3: Partial conversion
./build/bin/partial-opt test/mixed-dialects.mlir -partial-lower
```

## Deep Dive: How Conversion Works

### The Conversion Driver Loop

The conversion infrastructure uses a sophisticated iterative algorithm:

1. **Initial setup**: Mark all operations as unresolved
2. **Pattern application**: Try to match and apply patterns
3. **Legality checking**: Verify converted operations are legal
4. **Rollback on failure**: Undo changes if conversion fails
5. **Repeat**: Continue until fixed point or failure

**Source code:** [mlir/lib/Transforms/Utils/DialectConversion.cpp:2876](../../../mlir/lib/Transforms/Utils/DialectConversion.cpp) - `OperationConverter::convert()`

### Value Remapping

During conversion, the framework maintains mappings:

```
Original Value → Converted Value (after type conversion)
```

When a pattern replaces an operation:
1. New values are created with converted types
2. Mappings are updated: `old_value → new_value`
3. Subsequent patterns see converted values via `OpAdaptor`

**Implementation:** The `ConversionPatternRewriterImpl` class maintains these mappings in the `mapping` field.

**Source:** [mlir/lib/Transforms/Utils/DialectConversion.cpp:686](../../../mlir/lib/Transforms/Utils/DialectConversion.cpp)

### Materialization

When type conversions create mismatches, materialization operations are inserted:

```mlir
// Before conversion
%0 : tensor<4xf32>
%1 = some.op %0 : tensor<4xf32>

// After conversion (tensor → memref)
%0 : memref<4xf32>
%converted = "tensor.from_memref"(%0) : memref<4xf32> → tensor<4xf32>  // Source materialization
%1 = some.op %converted : tensor<4xf32>  // Unconverted op still needs tensor
```

The framework automatically inserts these casts based on your materialization callbacks.

**Source:** [mlir/lib/Transforms/Utils/DialectConversion.cpp:1157](../../../mlir/lib/Transforms/Utils/DialectConversion.cpp) - `legalizeUnresolvedMaterialization()`

### Pattern Ordering and Application

Patterns are applied in order of:
1. **Benefit**: Higher benefit patterns tried first
2. **Registration order**: Tie-breaker when benefits equal
3. **Specificity**: More specific patterns preferred

**Best practice:** Assign higher benefits to more specific patterns:

```cpp
// General pattern - low benefit
patterns.add<GeneralLowering>(context, /*benefit=*/1);

// Specific optimization pattern - high benefit
patterns.add<SpecialCaseLowering>(context, /*benefit=*/10);
```

**Source:** [mlir/lib/Transforms/Utils/DialectConversion.cpp:2509](../../../mlir/lib/Transforms/Utils/DialectConversion.cpp) - `OperationConverter::computeConversionSet()`

## Common Patterns and Best Practices

### 1. Simple Operation Lowering

```cpp
struct LowerMyOpPattern : public OpConversionPattern<MyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Get converted operands from adaptor
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Create replacement operation(s)
    Value result = rewriter.create<TargetOp>(
        op.getLoc(), lhs, rhs);

    // Replace original op
    rewriter.replaceOp(op, result);
    return success();
  }
};
```

### 2. Multi-Operation Lowering

```cpp
struct LowerComplexOpPattern : public OpConversionPattern<ComplexOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ComplexOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Lower to sequence of simpler operations
    Value v1 = rewriter.create<Op1>(op.getLoc(), adaptor.getInput());
    Value v2 = rewriter.create<Op2>(op.getLoc(), v1);
    Value v3 = rewriter.create<Op3>(op.getLoc(), v2);

    rewriter.replaceOp(op, v3);
    return success();
  }
};
```

### 3. Type Conversion with Materialization

```cpp
class MyTypeConverter : public TypeConverter {
public:
  MyTypeConverter() {
    // Convert tensor → memref
    addConversion([](TensorType type) {
      return MemRefType::get(type.getShape(), type.getElementType());
    });

    // Target materialization: memref → tensor (insert cast)
    addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (auto tensorType = resultType.dyn_cast<TensorType>())
        return builder.create<bufferization::ToTensorOp>(loc, inputs[0]);
      return nullptr;
    });

    // Source materialization: tensor → memref (insert cast)
    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (auto memrefType = resultType.dyn_cast<MemRefType>())
        return builder.create<bufferization::ToMemrefOp>(loc, memrefType, inputs[0]);
      return nullptr;
    });
  }
};
```

### 4. Dynamic Legality

```cpp
void populateConversionTarget(ConversionTarget &target) {
  // Always legal
  target.addLegalDialect<arith::ArithDialect>();

  // Always illegal
  target.addIllegalDialect<MySourceDialect>();

  // Conditionally legal
  target.addDynamicallyLegalOp<SomeOp>([](SomeOp op) {
    // Only legal if all operands are built-in types
    return llvm::all_of(op.getOperandTypes(), [](Type type) {
      return type.isIntOrFloat();
    });
  });
}
```

### 5. Signature Conversion for Functions

```cpp
struct ConvertFuncOpPattern : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp funcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto *typeConverter = getTypeConverter();

    // Convert function type
    TypeConverter::SignatureConversion signatureConversion(
        funcOp.getFunctionType().getNumInputs());

    for (auto [idx, argType] : llvm::enumerate(funcOp.getArgumentTypes())) {
      Type convertedType = typeConverter->convertType(argType);
      signatureConversion.addInputs(idx, convertedType);
    }

    // Create new function with converted signature
    auto newFuncType = FunctionType::get(
        funcOp.getContext(),
        signatureConversion.getConvertedTypes(),
        llvm::map_to_vector(funcOp.getResultTypes(),
                           [&](Type type) { return typeConverter->convertType(type); }));

    rewriter.modifyOpInPlace(funcOp, [&] {
      funcOp.setType(newFuncType);
      rewriter.applySignatureConversion(&funcOp.getBody(), signatureConversion);
    });

    return success();
  }
};
```

## Debugging Tips

### 1. Print Debug Information

```cpp
// In your pattern
LLVM_DEBUG(llvm::dbgs() << "Converting op: " << *op << "\n");
```

Run with: `MLIR_ENABLE_DUMP=1 ./your-tool -debug-only=dialect-conversion`

### 2. Use mlir-opt with -debug

```bash
mlir-opt input.mlir -convert-to-target -debug-only=dialect-conversion
```

### 3. Check IR After Each Pass

```bash
mlir-opt input.mlir \
  -pass1 -mlir-print-ir-after-all \
  -pass2 -mlir-print-ir-after-all
```

### 4. Verify IR Integrity

```cpp
// After conversion
if (failed(mlir::verify(module)))
  return signalPassFailure();
```

## Further Reading

### MLIR Documentation
- [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/) - Official docs
- [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/) - General rewriting framework
- [Table-Driven Rewriting](https://mlir.llvm.org/docs/DeclarativeRewrites/) - DRR for simpler patterns

### Source Code Deep Dives
- [mlir/include/mlir/Transforms/DialectConversion.h](../../../mlir/include/mlir/Transforms/DialectConversion.h) - All APIs
- [mlir/lib/Transforms/Utils/DialectConversion.cpp](../../../mlir/lib/Transforms/Utils/DialectConversion.cpp) - Core implementation
- [mlir/include/mlir/IR/PatternMatch.h](../../../mlir/include/mlir/IR/PatternMatch.h) - Base rewriting framework

### Real-World Examples in LLVM
- [TosaToLinalg](../../../mlir/lib/Conversion/TosaToLinalg/TosaToLinalg.cpp) - Complex tensor operations
- [SCFToControlFlow](../../../mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp) - Structured to unstructured control flow
- [ArithToLLVM](../../../mlir/lib/Conversion/ArithToLLVM/ArithToLLVM.cpp) - Arithmetic to LLVM dialect

## Summary

MLIR's dialect conversion infrastructure is a powerful framework for transforming IR between abstraction levels. Key takeaways:

1. **Patterns define transformations** - Use `OpConversionPattern` for type-safe operation rewriting
2. **Types need conversion too** - `TypeConverter` handles cross-dialect type mapping
3. **Legality drives conversion** - `ConversionTarget` determines what needs converting
4. **Materialization bridges gaps** - Automatic cast insertion for type mismatches
5. **Multiple strategies exist** - Full, partial, and analysis conversion for different needs

The examples in this tutorial demonstrate these concepts with runnable code that you can modify and experiment with. Start with Example 1 and progressively work through more complex scenarios.

Happy converting!
