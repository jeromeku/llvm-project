# MLIR Dialect Conversion Tutorial - Complete Index

A comprehensive tutorial on MLIR's dialect conversion infrastructure with runnable examples and deep implementation insights.

## Documentation Structure

### 📚 Main Guides

1. **[QUICKSTART.md](QUICKSTART.md)** - *Start here!*
   - 5-minute setup and first runs
   - Basic debugging commands
   - Common issues and solutions
   - **Audience**: Newcomers, quick reference

2. **[README.md](README.md)** - *Comprehensive tutorial*
   - All dialect conversion concepts
   - Complete API documentation with source links
   - Three progressive examples explained
   - Best practices and patterns
   - **Audience**: Learning the full system, reference

3. **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - *Deep dive*
   - How conversion works internally
   - Algorithm walkthroughs with source code
   - Data structures and their purposes
   - Performance considerations
   - **Audience**: Understanding internals, debugging complex issues

## Example Programs

### Example 1: Simple Operation Lowering

**What it teaches**: Basic conversion pattern structure

**Files**:
```
include/SimpleArith.td          # Dialect definition (TableGen)
include/SimpleArith.h           # Generated C++ headers
lib/SimpleArithLowering.cpp     # Conversion patterns (100+ lines of comments)
tools/simple-opt.cpp            # Tool to run conversions
test/simple-arith.mlir          # Test cases with expected output
```

**Key concepts**:
- `OpConversionPattern<T>` - Type-safe pattern definition
- `OpAdaptor` - Accessing converted operands
- `ConversionPatternRewriter` - IR modification
- `ConversionTarget` - Specifying legality
- `applyFullConversion` - Complete lowering

**Run it**:
```bash
./build/bin/simple-opt test/simple-arith.mlir -convert-simple-to-arith
```

**Code highlights**:
- [ConvertAddOp pattern](lib/SimpleArithLowering.cpp#L47) - Basic operation replacement
- [ConvertNegOp pattern](lib/SimpleArithLowering.cpp#L163) - Multi-operation expansion
- [Pass definition](lib/SimpleArithLowering.cpp#L242) - Setting up conversion target

### Example 2: Type Conversion

**What it teaches**: Converting custom types with materialization

**Files**:
```
include/TypedArith.td          # Dialect with custom types (FixedPoint, Complex)
include/TypedArith.h           # Type definitions
lib/TypedArithLowering.cpp     # Type converter + patterns (200+ lines of comments)
tools/typed-opt.cpp            # Type conversion tool
test/typed-arith.mlir          # Type conversion tests
```

**Key concepts**:
- `TypeConverter` - Defining type conversions
- Type conversion callbacks
- **Materialization**:
  - Target materialization (source → target)
  - Source materialization (target → source)
  - Argument materialization (function args)
- Signature conversion
- `applyPartialConversion` - Allowing illegal ops

**Run it**:
```bash
./build/bin/typed-opt test/typed-arith.mlir -convert-typed-to-std
```

**Code highlights**:
- [TypedArithTypeConverter class](lib/TypedArithLowering.cpp#L69) - Complete type converter with materialization
- [Fixed-point multiplication](lib/TypedArithLowering.cpp#L213) - Lowering with scale adjustment
- [Materialization callbacks](lib/TypedArithLowering.cpp#L105) - Automatic cast insertion

### Example 3: Partial Conversion & Dynamic Legality

**What it teaches**: Progressive lowering with conditional conversion

**Files**:
```
include/MixedDialects.td       # High and Medium level dialects
include/MixedDialects.h        # Dialect declarations
lib/PartialLowering.cpp        # Partial conversion (300+ lines of comments)
tools/partial-opt.cpp          # Partial conversion tool
test/partial-conversion.mlir   # Mixed dialect IR examples
```

**Key concepts**:
- **Dynamic legality** - Conditional conversion based on properties
- **Partial conversion** - Mixed dialect IR (intentional!)
- Multi-stage lowering pipelines
- `applyAnalysisConversion` - Dry-run testing
- Progressive lowering strategies

**Run it**:
```bash
./build/bin/partial-opt test/partial-conversion.mlir -partial-lower-high
```

**Code highlights**:
- [Dynamic legality for MatMul](lib/PartialLowering.cpp#L231) - Size-based conversion
- [Partial conversion pass](lib/PartialLowering.cpp#L179) - Allowing illegal ops
- [Analysis conversion](lib/PartialLowering.cpp#L331) - Testing without modifying IR

## Source Code Map

### Dialect Definitions (TableGen)

| File | Dialect | Operations | Types | Purpose |
|------|---------|------------|-------|---------|
| [SimpleArith.td](include/SimpleArith.td) | `simple` | add, sub, mul, div, neg, constant | Standard | Basic operation lowering |
| [TypedArith.td](include/TypedArith.td) | `typed` | fixed_add, fixed_mul, complex_add, etc. | FixedPoint, Complex | Type conversion |
| [MixedDialects.td](include/MixedDialects.td) | `high`, `medium` | matmul, dot, loop, extract, insert | Standard | Partial conversion |

### Conversion Implementations

| File | Classes | LOC | Key Features |
|------|---------|-----|--------------|
| [SimpleArithLowering.cpp](lib/SimpleArithLowering.cpp) | 6 patterns, 1 pass | 350 | Basic patterns, target setup |
| [TypedArithLowering.cpp](lib/TypedArithLowering.cpp) | TypeConverter, 5 patterns, 1 pass | 500 | Type conversion, materialization |
| [PartialLowering.cpp](lib/PartialLowering.cpp) | 3 patterns, 3 passes | 400 | Dynamic legality, multi-stage |

### Tools

| Tool | Purpose | Input Dialect(s) | Output Dialect(s) |
|------|---------|------------------|-------------------|
| [simple-opt](tools/simple-opt.cpp) | Basic conversion testing | simple | arith |
| [typed-opt](tools/typed-opt.cpp) | Type conversion testing | typed | arith + standard |
| [partial-opt](tools/partial-opt.cpp) | Partial conversion testing | high | high + medium (mixed) |

### Tests

| Test File | Tests | Focus |
|-----------|-------|-------|
| [simple-arith.mlir](test/simple-arith.mlir) | 5 functions | Operation lowering, CHECK directives |
| [typed-arith.mlir](test/typed-arith.mlir) | 7 functions | Type conversion, signature conversion |
| [partial-conversion.mlir](test/partial-conversion.mlir) | 6 functions | Dynamic legality, mixed IR |

## Build System

```
CMakeLists.txt                 # Top-level CMake
├── include/CMakeLists.txt     # TableGen generation
├── lib/CMakeLists.txt         # Library builds
└── tools/CMakeLists.txt       # Tool builds

build.sh                       # Automated build script
test.sh                        # Automated test runner
```

## Key Concepts Index

### Conversion Patterns

- **OpConversionPattern<T>** - [Example](lib/SimpleArithLowering.cpp#L47), [Docs](README.md#opconversionpatternt)
- **matchAndRewrite()** - [Example](lib/SimpleArithLowering.cpp#L53), [Internals](IMPLEMENTATION_GUIDE.md#pattern-matching)
- **OpAdaptor** - [Example](lib/SimpleArithLowering.cpp#L64), [Docs](README.md#opconversionpatternt)
- **Pattern benefits** - [Example](README.md#pattern-ordering-and-application), [Internals](IMPLEMENTATION_GUIDE.md#pattern-benefit)

### Type Conversion

- **TypeConverter** - [Example](lib/TypedArithLowering.cpp#L69), [Docs](README.md#typeconverter)
- **Type conversion callbacks** - [Example](lib/TypedArithLowering.cpp#L81), [Internals](IMPLEMENTATION_GUIDE.md#conversion-functions)
- **Target materialization** - [Example](lib/TypedArithLowering.cpp#L105), [Docs](README.md#typeconverter)
- **Source materialization** - [Example](lib/TypedArithLowering.cpp#L139), [Docs](README.md#typeconverter)
- **Argument materialization** - [Example](lib/TypedArithLowering.cpp#L167), [Docs](README.md#typeconverter)
- **Signature conversion** - [Example](README.md#5-signature-conversion-for-functions), [Internals](IMPLEMENTATION_GUIDE.md#signature-conversion)

### Conversion Targets

- **ConversionTarget** - [Example](lib/SimpleArithLowering.cpp#L286), [Docs](README.md#conversiontarget)
- **Legal operations** - [Example](lib/SimpleArithLowering.cpp#L297), [Docs](README.md#conversiontarget)
- **Illegal operations** - [Example](lib/SimpleArithLowering.cpp#L302), [Docs](README.md#conversiontarget)
- **Dynamic legality** - [Example](lib/PartialLowering.cpp#L231), [Docs](README.md#4-dynamic-legality)

### Conversion Drivers

- **applyFullConversion** - [Example](lib/SimpleArithLowering.cpp#L320), [Docs](README.md#conversion-drivers)
- **applyPartialConversion** - [Example](lib/TypedArithLowering.cpp#L447), [Docs](README.md#conversion-drivers)
- **applyAnalysisConversion** - [Example](lib/PartialLowering.cpp#L355), [Docs](README.md#conversion-drivers)

### Advanced Topics

- **Value remapping** - [Internals](IMPLEMENTATION_GUIDE.md#value-remapping)
- **Rollback/transactions** - [Internals](IMPLEMENTATION_GUIDE.md#rollback-and-transactions)
- **Multi-stage lowering** - [Example](lib/PartialLowering.cpp#L300), [Docs](README.md#example-3-partial-conversion)
- **Pattern application** - [Internals](IMPLEMENTATION_GUIDE.md#pattern-application)

## MLIR Source Code References

All links point to the actual MLIR source code in your llvm-project directory.

### Key Headers

- [mlir/include/mlir/Transforms/DialectConversion.h](../../mlir/include/mlir/Transforms/DialectConversion.h) - Main conversion APIs
- [mlir/include/mlir/IR/PatternMatch.h](../../mlir/include/mlir/IR/PatternMatch.h) - Pattern infrastructure
- [mlir/include/mlir/IR/Builders.h](../../mlir/include/mlir/IR/Builders.h) - IR builder APIs

### Key Implementation Files

- [mlir/lib/Transforms/Utils/DialectConversion.cpp](../../mlir/lib/Transforms/Utils/DialectConversion.cpp) - Core conversion logic (4000+ lines)
- [mlir/lib/IR/PatternMatch.cpp](../../mlir/lib/IR/PatternMatch.cpp) - Pattern matching implementation

### Real-World Examples in MLIR

- [mlir/lib/Conversion/TosaToLinalg/](../../mlir/lib/Conversion/TosaToLinalg/) - Complex tensor operations
- [mlir/lib/Conversion/SCFToControlFlow/](../../mlir/lib/Conversion/SCFToControlFlow/) - Control flow lowering
- [mlir/lib/Conversion/ArithToLLVM/](../../mlir/lib/Conversion/ArithToLLVM/) - Lowering to LLVM dialect
- [mlir/lib/Conversion/VectorToLLVM/](../../mlir/lib/Conversion/VectorToLLVM/) - Vector operations to LLVM
- [mlir/lib/Conversion/MemRefToLLVM/](../../mlir/lib/Conversion/MemRefToLLVM/) - Memory operations to LLVM

## Learning Paths

### Path 1: Quick Start (1-2 hours)

1. Read [QUICKSTART.md](QUICKSTART.md)
2. Build the examples: `./build.sh`
3. Run the examples: `./test.sh`
4. Modify [test/simple-arith.mlir](test/simple-arith.mlir) and experiment

### Path 2: Complete Tutorial (4-6 hours)

1. Read [QUICKSTART.md](QUICKSTART.md)
2. Read [README.md](README.md) fully
3. Study [lib/SimpleArithLowering.cpp](lib/SimpleArithLowering.cpp) with comments
4. Study [lib/TypedArithLowering.cpp](lib/TypedArithLowering.cpp) with comments
5. Study [lib/PartialLowering.cpp](lib/PartialLowering.cpp) with comments
6. Implement your own conversion pattern

### Path 3: Deep Understanding (8-12 hours)

1. Follow Path 2
2. Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) fully
3. Read MLIR source code:
   - [DialectConversion.cpp](../../mlir/lib/Transforms/Utils/DialectConversion.cpp)
   - Focus on `OperationConverter::convert()`
   - Trace through value remapping
4. Study a real MLIR conversion (e.g., TosaToLinalg)
5. Implement a multi-stage lowering pipeline

### Path 4: Mastery (ongoing)

1. Follow Path 3
2. Read MLIR discourse discussions on conversion
3. Contribute to MLIR conversion passes
4. Implement complex dialects with custom types
5. Optimize conversion performance

## Quick Reference

### Building

```bash
./build.sh                    # Build everything
cd build && ninja             # Rebuild after changes
```

### Testing

```bash
./test.sh                                     # Run all examples
./build/bin/simple-opt test/simple-arith.mlir -convert-simple-to-arith
./build/bin/typed-opt test/typed-arith.mlir -convert-typed-to-std
./build/bin/partial-opt test/partial-conversion.mlir -partial-lower-high
```

### Debugging

```bash
# Print IR after transformations
./build/bin/simple-opt input.mlir -your-pass --mlir-print-ir-after-all

# Debug conversion process
MLIR_ENABLE_DUMP=1 ./build/bin/simple-opt input.mlir -your-pass \
  -debug-only=dialect-conversion

# Debug pattern matching
MLIR_ENABLE_DUMP=1 ./build/bin/simple-opt input.mlir -your-pass \
  -debug-only=pattern-match

# All debug info
MLIR_ENABLE_DUMP=1 ./build/bin/simple-opt input.mlir -your-pass -debug
```

### Common Commands

```bash
# Verify IR
mlir-opt input.mlir -verify-each

# Parse and print (format check)
mlir-opt input.mlir

# Run lit tests (if you add RUN: directives)
llvm-lit test/

# Generate documentation from TableGen
mlir-tblgen -gen-op-doc include/SimpleArith.td
```

## Getting Help

1. **Check the guides**: Start with [QUICKSTART.md](QUICKSTART.md), then [README.md](README.md)
2. **Read the comments**: Example files have 100+ lines of explanatory comments
3. **Use debug output**: See "Debugging" section above
4. **Consult implementation guide**: [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
5. **Ask the community**:
   - [MLIR Discourse](https://discourse.llvm.org/c/mlir/31)
   - [MLIR Discord](https://discord.gg/xS7Z362)

## Contributing

Found an issue or want to improve the tutorial?

1. The tutorial is in `llvm-project/claude/conversion/`
2. All files are heavily commented for learning
3. Feel free to extend examples or add new ones
4. Share improvements with the MLIR community

## License

This tutorial follows the LLVM Project license (Apache 2.0 with LLVM exception), same as MLIR.

---

**Happy learning! Start with [QUICKSTART.md](QUICKSTART.md) →**
