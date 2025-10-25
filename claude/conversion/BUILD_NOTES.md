# Build Notes

## CMake Configuration Fixes

### Fix 1: PARTIAL_SOURCES_INTENDED

The CMakeLists.txt files have been updated to add `PARTIAL_SOURCES_INTENDED` to all library and executable targets. This is required by LLVM's build system when you have multiple targets in the same directory.

### Fix 2: TableGen Output File Names

Fixed TableGen generation to use unique output file names for different generators. MLIR's TableGen cannot generate multiple outputs with the same filename.

**Changes:**
- Dialect definitions: `SimpleArith.h.inc`, `TypedArith.h.inc`, `MixedDialects.h.inc`
- Dialect implementations: `SimpleArith.cpp.inc`, `TypedArith.cpp.inc`, `MixedDialects.cpp.inc`
- Operation definitions: `SimpleArithOps.h.inc`, `TypedArithOps.h.inc`, `MixedDialectsOps.h.inc`
- Operation implementations: `SimpleArithOps.cpp.inc`, `TypedArithOps.cpp.inc`, `MixedDialectsOps.cpp.inc`
- Type definitions (TypedArith only): `TypedArithTypes.h.inc`, `TypedArithTypes.cpp.inc`

**Updated files:**
- `include/CMakeLists.txt` - Changed mlir_tablegen output names
- All header files (`.h`) - Updated include directives
- All implementation files (`.cpp`) - Updated include directives

### Why This Is Needed

LLVM's build system (LLVMProcessSources.cmake) enforces:
1. All source files must be part of a build target
2. Each directory should ideally have only one target
3. If you need multiple targets, explicitly mark with `PARTIAL_SOURCES_INTENDED`

### Changes Made

**lib/CMakeLists.txt**:
- Added `PARTIAL_SOURCES_INTENDED` to all three libraries:
  - MLIRSimpleArithLowering
  - MLIRTypedArithLowering
  - MLIRPartialLowering

**tools/CMakeLists.txt**:
- Added `PARTIAL_SOURCES_INTENDED` to all three executables:
  - simple-opt
  - typed-opt
  - partial-opt

### Building

The project should now build successfully:

```bash
./build.sh
```

Or manually:

```bash
mkdir build && cd build
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm
ninja
```

### Reference

LLVM build system documentation:
- [AddLLVM.cmake](../../llvm/cmake/modules/AddLLVM.cmake)
- [LLVMProcessSources.cmake](../../llvm/cmake/modules/LLVMProcessSources.cmake)
