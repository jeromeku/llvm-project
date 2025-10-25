# Build Fixes Applied

This document summarizes all the fixes applied to make the tutorial build successfully with LLVM's strict build system requirements.

## Issue 1: Multiple Targets Per Directory

**Error:**
```
CMake Error: Found erroneous configuration for source file *.cpp
LLVM's build system enforces that all source files are added to a build target...
```

**Cause:**
LLVM's build system (LLVMProcessSources.cmake) enforces that when multiple targets exist in one directory, they must be explicitly marked with `PARTIAL_SOURCES_INTENDED`.

**Solution:**
Added `PARTIAL_SOURCES_INTENDED` to all targets:

### lib/CMakeLists.txt
```cmake
add_mlir_library(MLIRSimpleArithLowering
  SimpleArithLowering.cpp
  PARTIAL_SOURCES_INTENDED    # ← Added
  ...
)

add_mlir_library(MLIRTypedArithLowering
  TypedArithLowering.cpp
  PARTIAL_SOURCES_INTENDED    # ← Added
  ...
)

add_mlir_library(MLIRPartialLowering
  PartialLowering.cpp
  PARTIAL_SOURCES_INTENDED    # ← Added
  ...
)
```

### tools/CMakeLists.txt
```cmake
add_llvm_executable(simple-opt
  simple-opt.cpp
  PARTIAL_SOURCES_INTENDED    # ← Added
)

add_llvm_executable(typed-opt
  typed-opt.cpp
  PARTIAL_SOURCES_INTENDED    # ← Added
)

add_llvm_executable(partial-opt
  partial-opt.cpp
  PARTIAL_SOURCES_INTENDED    # ← Added
)
```

## Issue 2: Duplicate TableGen Output Files

**Error:**
```
CMake Error: Attempt to add a custom rule to output
  .../MixedDialects.cpp.inc.rule
which already has a custom rule.
```

**Cause:**
Multiple `mlir_tablegen()` calls were trying to generate the same output file name (e.g., `SimpleArith.cpp.inc`) with different generators.

**Solution:**
Use unique output file names for each generator:

### include/CMakeLists.txt

**Before (WRONG):**
```cmake
set(LLVM_TARGET_DEFINITIONS SimpleArith.td)
mlir_tablegen(SimpleArith.h.inc -gen-dialect-decls)
mlir_tablegen(SimpleArith.cpp.inc -gen-dialect-defs)   # Same name!
mlir_tablegen(SimpleArith.cpp.inc -gen-op-decls)       # Same name!
mlir_tablegen(SimpleArith.cpp.inc -gen-op-defs)        # Same name!
```

**After (CORRECT):**
```cmake
set(LLVM_TARGET_DEFINITIONS SimpleArith.td)
mlir_tablegen(SimpleArith.h.inc -gen-dialect-decls)
mlir_tablegen(SimpleArith.cpp.inc -gen-dialect-defs)
mlir_tablegen(SimpleArithOps.h.inc -gen-op-decls)     # Unique name
mlir_tablegen(SimpleArithOps.cpp.inc -gen-op-defs)    # Unique name
```

### Updated Include Directives

All C++ files updated to match new generated file names:

#### SimpleArith
- Header: `SimpleArithOps.h.inc` (was `SimpleArith.cpp.inc`)
- Implementation: `SimpleArithOps.cpp.inc` (was `SimpleArith.cpp.inc`)

#### TypedArith
- Type headers: `TypedArithTypes.h.inc` (was `TypedArith.cpp.inc`)
- Type implementation: `TypedArithTypes.cpp.inc` (was `TypedArith.cpp.inc`)
- Op headers: `TypedArithOps.h.inc` (was `TypedArith.cpp.inc`)
- Op implementation: `TypedArithOps.cpp.inc` (was `TypedArith.cpp.inc`)

#### MixedDialects
- Op headers: `MixedDialectsOps.h.inc` (was `MixedDialects.cpp.inc`)
- Op implementation: `MixedDialectsOps.cpp.inc` (was `MixedDialects.cpp.inc`)

## Files Modified

### CMake Files
- [x] `CMakeLists.txt` - Root CMake file
- [x] `include/CMakeLists.txt` - TableGen generation rules
- [x] `lib/CMakeLists.txt` - Library targets
- [x] `tools/CMakeLists.txt` - Executable targets

### Header Files
- [x] `include/SimpleArith.h` - Updated include for ops
- [x] `include/TypedArith.h` - Updated includes for types and ops
- [x] `include/MixedDialects.h` - Updated include for ops

### Implementation Files
- [x] `lib/SimpleArithLowering.cpp` - Updated include for ops
- [x] `lib/TypedArithLowering.cpp` - Updated includes for types and ops
- [x] `lib/PartialLowering.cpp` - Updated include for ops

## Verification

After these fixes, the project should build successfully:

```bash
cd claude/conversion
./build.sh
```

Expected output:
```
✓ CMake configuration successful
✓ TableGen generation successful
✓ Compilation successful
✓ Linking successful
✓ Executables created:
  - build/bin/simple-opt
  - build/bin/typed-opt
  - build/bin/partial-opt
```

## Testing

Run the test suite:

```bash
./test.sh
```

Or manually:

```bash
./build/bin/simple-opt test/simple-arith.mlir -convert-simple-to-arith
./build/bin/typed-opt test/typed-arith.mlir -convert-typed-to-std
./build/bin/partial-opt test/partial-conversion.mlir -partial-lower-high
```

## Why These Patterns?

These fix patterns follow MLIR conventions seen in the LLVM tree:

**Example from MLIR dialects:**
```cmake
# mlir/lib/Dialect/Arith/IR/CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS ArithOps.td)
mlir_tablegen(ArithOps.h.inc -gen-op-decls)
mlir_tablegen(ArithOps.cpp.inc -gen-op-defs)
mlir_tablegen(ArithOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ArithOpsDialect.cpp.inc -gen-dialect-defs)
# ↑ Note: Each generator gets its own unique output file
```

**References:**
- [mlir/lib/Dialect/Arith/IR/CMakeLists.txt](../../../mlir/lib/Dialect/Arith/IR/CMakeLists.txt)
- [mlir/lib/Dialect/Func/IR/CMakeLists.txt](../../../mlir/lib/Dialect/Func/IR/CMakeLists.txt)
- [llvm/cmake/modules/LLVMProcessSources.cmake](../../../llvm/cmake/modules/LLVMProcessSources.cmake)

## Summary

✅ **Fixed:** Multiple targets per directory (added `PARTIAL_SOURCES_INTENDED`)
✅ **Fixed:** Duplicate TableGen outputs (unique file names)
✅ **Updated:** All include directives to match new file names
✅ **Verified:** Follows MLIR conventions from LLVM tree

The tutorial should now build successfully on any system with MLIR built from source!
