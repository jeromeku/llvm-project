# Quick Start Guide

Get up and running with the MLIR Dialect Conversion tutorial in 5 minutes.

## Prerequisites

- MLIR built from source (this tutorial assumes you're already in the llvm-project directory with MLIR built)
- CMake 3.20+
- Ninja build system
- C++17 compiler

## Building

```bash
cd claude/conversion

# Build everything
./build.sh

# This will:
# 1. Detect your MLIR build directory
# 2. Configure CMake
# 3. Build all three examples
```

If the script can't find your MLIR build, specify it:

```bash
export LLVM_BUILD_DIR=/path/to/llvm-project/build
./build.sh
```

## Running the Examples

```bash
# Run all examples with explanations
./test.sh
```

Or run individual examples:

```bash
# Example 1: Simple arithmetic conversion
./build/bin/simple-opt test/simple-arith.mlir -convert-simple-to-arith

# Example 2: Type conversion with fixed-point arithmetic
./build/bin/typed-opt test/typed-arith.mlir -convert-typed-to-std

# Example 3: Partial conversion with dynamic legality
./build/bin/partial-opt test/partial-conversion.mlir -partial-lower-high
```

## Understanding the Examples

### Example 1: Basic Operation Lowering

**Goal**: Learn basic `OpConversionPattern` usage

**Files**:
- [include/SimpleArith.td](include/SimpleArith.td) - Dialect definition
- [lib/SimpleArithLowering.cpp](lib/SimpleArithLowering.cpp) - Conversion patterns (heavily commented)
- [test/simple-arith.mlir](test/simple-arith.mlir) - Test cases

**What you'll learn**:
- How to write conversion patterns
- Using `ConversionPatternRewriter`
- Setting up `ConversionTarget`
- Using `applyFullConversion`

**Try it**:
```bash
# Convert simple arithmetic operations to standard arith dialect
./build/bin/simple-opt test/simple-arith.mlir -convert-simple-to-arith

# See IR after each transformation
./build/bin/simple-opt test/simple-arith.mlir \
  -convert-simple-to-arith \
  --mlir-print-ir-after-all

# Debug conversion process
MLIR_ENABLE_DUMP=1 ./build/bin/simple-opt test/simple-arith.mlir \
  -convert-simple-to-arith \
  -debug-only=dialect-conversion
```

### Example 2: Type Conversion

**Goal**: Learn how to convert custom types using `TypeConverter`

**Files**:
- [include/TypedArith.td](include/TypedArith.td) - Dialect with custom types (FixedPoint)
- [lib/TypedArithLowering.cpp](lib/TypedArithLowering.cpp) - Type converter and patterns
- [test/typed-arith.mlir](test/typed-arith.mlir) - Type conversion examples

**What you'll learn**:
- Creating a `TypeConverter`
- Type conversion callbacks
- Materialization (target/source/argument)
- Signature conversion
- Using `applyPartialConversion`

**Try it**:
```bash
# Convert fixed-point types to integers
./build/bin/typed-opt test/typed-arith.mlir -convert-typed-to-std

# Notice how FixedPoint<16,8> → i16
# And fixed_mul becomes: mul + right-shift

# See the signature conversion in action
./build/bin/typed-opt test/typed-arith.mlir \
  -convert-typed-to-std \
  --mlir-print-ir-after-all
```

### Example 3: Partial Conversion & Dynamic Legality

**Goal**: Learn progressive lowering with mixed dialect IR

**Files**:
- [include/MixedDialects.td](include/MixedDialects.td) - High and Medium level dialects
- [lib/PartialLowering.cpp](lib/PartialLowering.cpp) - Partial conversion patterns
- [test/partial-conversion.mlir](test/partial-conversion.mlir) - Mixed dialect tests

**What you'll learn**:
- Dynamic legality (conditional conversion)
- Partial conversion (allowing illegal ops)
- Multi-stage lowering pipelines
- Using `applyAnalysisConversion`

**Try it**:
```bash
# Apply partial conversion
./build/bin/partial-opt test/partial-conversion.mlir -partial-lower-high

# Notice: Large operations remain high-level
#         Small operations get converted
#         This is INTENTIONAL with dynamic legality!

# Test if conversion would succeed (without modifying IR)
./build/bin/partial-opt test/partial-conversion.mlir -analyze-lowering
```

## Debugging Tips

### 1. Enable IR Printing

```bash
# Print IR after every pass
mlir-opt input.mlir -your-pass --mlir-print-ir-after-all

# Print IR before and after
mlir-opt input.mlir -your-pass --mlir-print-ir-before-all --mlir-print-ir-after-all
```

### 2. Enable Debug Output

```bash
# See detailed conversion information
MLIR_ENABLE_DUMP=1 mlir-opt input.mlir -your-pass -debug-only=dialect-conversion

# See pattern matching details
MLIR_ENABLE_DUMP=1 mlir-opt input.mlir -your-pass -debug-only=pattern-match

# See both
MLIR_ENABLE_DUMP=1 mlir-opt input.mlir -your-pass -debug
```

### 3. Verify IR

```bash
# Verify IR is well-formed after conversion
mlir-opt input.mlir -your-pass -verify-each
```

### 4. Dump Operations

Add to your patterns:

```cpp
// In your conversion pattern
LLVM_DEBUG({
  llvm::dbgs() << "Converting operation: ";
  op->dump();
  llvm::dbgs() << "Converted operands: ";
  for (Value v : adaptor.getOperands())
    llvm::dbgs() << "  " << v << " : " << v.getType() << "\n";
});
```

Run with:
```bash
MLIR_ENABLE_DUMP=1 your-tool -your-pass -debug-only=your-pattern-name
```

## Common Issues

### Issue: "Cannot find MLIR build directory"

**Solution**:
```bash
export LLVM_BUILD_DIR=/path/to/your/llvm-project/build
./build.sh
```

### Issue: "MLIRConfig.cmake not found"

**Solution**: Build MLIR first:
```bash
cd /path/to/llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DCMAKE_BUILD_TYPE=Release
ninja
```

### Issue: Conversion patterns not applying

**Check**:
1. Is the operation marked as illegal in `ConversionTarget`?
2. Is your pattern registered in the `RewritePatternSet`?
3. Does your pattern's `match()` succeed?
4. Is the pattern benefit high enough?

**Debug**:
```bash
MLIR_ENABLE_DUMP=1 your-tool input.mlir -your-pass \
  -debug-only=dialect-conversion \
  --mlir-print-ir-after-all
```

### Issue: Type conversion not working

**Check**:
1. Is `TypeConverter` passed to patterns?
2. Are conversion callbacks registered?
3. Are materialization callbacks defined?
4. Is the type actually being converted (check with debug output)?

**Debug**:
```cpp
// In your TypeConverter constructor
addConversion([](Type type) {
  llvm::errs() << "Converting type: " << type << "\n";
  return type;
});
```

## Next Steps

1. **Read the main tutorial**: [README.md](README.md) - Comprehensive guide with API documentation

2. **Study implementation details**: [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Deep dive into how conversion works

3. **Examine real MLIR conversions**:
   - [TosaToLinalg](../../mlir/lib/Conversion/TosaToLinalg/) - Complex tensor operations
   - [SCFToControlFlow](../../mlir/lib/Conversion/SCFToControlFlow/) - Control flow lowering
   - [ArithToLLVM](../../mlir/lib/Conversion/ArithToLLVM/) - Lowering to LLVM dialect

4. **Modify the examples**:
   - Add new operations to SimpleArith
   - Implement complex number lowering in TypedArith
   - Complete the matrix multiplication lowering in MixedDialects

5. **Create your own dialect**:
   - Define custom operations
   - Write conversion patterns
   - Test with the provided infrastructure

## Resources

### Documentation
- [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [Defining Dialects](https://mlir.llvm.org/docs/DefiningDialects/)

### Source Code
- [DialectConversion.h](../../mlir/include/mlir/Transforms/DialectConversion.h)
- [DialectConversion.cpp](../../mlir/lib/Transforms/Utils/DialectConversion.cpp)
- [PatternMatch.h](../../mlir/include/mlir/IR/PatternMatch.h)

### Community
- [MLIR Discourse](https://discourse.llvm.org/c/mlir/31)
- [MLIR Discord](https://discord.gg/xS7Z362)
- [MLIR GitHub](https://github.com/llvm/llvm-project/tree/main/mlir)

## Getting Help

If you encounter issues:

1. Check the debug output (see "Debugging Tips" above)
2. Read the extensive comments in the example source files
3. Consult the [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
4. Look at similar patterns in MLIR's conversion passes
5. Ask on MLIR Discourse or Discord

Happy converting!
