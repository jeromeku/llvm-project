# MLIR Dialect CMake Build System: Complete Walkthrough

This document provides a comprehensive guide to understanding the CMake build system for MLIR dialects, focusing on C API testing and Python bindings. We use the `mlir-list` dialect as a reference example.

## Table of Contents

- [Overview](#overview)
- [Part 1: CAPI Testing](#part-1-capi-testing)
- [Part 2: Python Bindings](#part-2-python-bindings)
- [Key Concepts](#key-concepts)
- [Reference](#reference)

---

## Overview

MLIR dialects require two main build artifacts:

1. **C API Test Library**: A self-contained shared library for testing the dialect from plain C
2. **Python Bindings**: Python modules that expose dialect functionality through a mix of auto-generated and hand-written code

The build system uses a **two-phase approach**:
- **Phase 1 (Declaration)**: Register files and metadata using CMake INTERFACE libraries
- **Phase 2 (Build)**: Walk the dependency graph and compile everything

---

## Part 1: CAPI Testing

**File**: [mlir-list/test/CAPI/CMakeLists.txt](../../../mlir-list/test/CAPI/CMakeLists.txt)

### Purpose

Create a C executable that tests your List dialect through the C API, proving the dialect works from plain C (not just C++).

### Architecture

```
listproject-capi-test (executable)
    ↓ links against
libListProjectCAPITestLib.so (aggregate library)
    ↓ embeds object files from
├── libMLIRCAPIIR.a           (Core MLIR IR C API)
├── libMLIRCAPIRegisterEverything.a  (Dialect registration)
└── libListProjectCAPI.a      (Your List dialect C API)
```

### Step-by-Step Code Walkthrough

#### Step 1: Create Aggregate Library (Lines 5-13)

```cmake
add_mlir_aggregate(ListProjectCAPITestLib
  SHARED                          # Build as a .so (shared library)
  EMBED_LIBS                      # List of libraries to embed follows
  MLIRCAPIIR                      # Core MLIR C API for IR manipulation
  MLIRCAPIRegisterEverything      # C API for dialect registration
  ListProjectCAPI                 # Your List dialect's C API
)
```

**Function**: `add_mlir_aggregate`
- **Location**: [mlir/cmake/modules/AddMLIR.cmake:497-629](../../../mlir/cmake/modules/AddMLIR.cmake#L497-L629)
- **Purpose**: Combines multiple libraries into a single self-contained shared library

**Internal Process**:

1. **Parse Arguments** (lines 498-502):
   - `SHARED` → create a `.so` file
   - `EMBED_LIBS` → list of libraries to bundle together

2. **For Each Library** (lines 515-583):
   ```cmake
   foreach(lib ${ARG_EMBED_LIBS})
     # Extract object files via CMake generator expressions
     set(_local_objects $<TARGET_PROPERTY:${lib},MLIR_AGGREGATE_OBJECTS>)
     # Extract link dependencies
     set(_local_deps $<TARGET_PROPERTY:${lib},MLIR_AGGREGATE_DEPS>)

     list(APPEND _objects ${_local_objects})
     list(APPEND _deps ${_local_deps})
   endforeach()
   ```

   For each library in `EMBED_LIBS`:
   - Extract the library's **object files** (`.o` files) via property `MLIR_AGGREGATE_OBJECTS`
   - Extract the library's **link dependencies** via property `MLIR_AGGREGATE_DEPS`
   - Store these for later aggregation

3. **Create Library** (lines 585-593):
   ```cmake
   add_mlir_library(${name}
     ${_libtype}
     PARTIAL_SOURCES_INTENDED
     EXCLUDE_FROM_LIBMLIR
     LINK_LIBS PRIVATE ${_deps} ${ARG_PUBLIC_LIBS}
   )
   ```

   Calls `add_mlir_library` ([AddMLIR.cmake:339-496](../../../mlir/cmake/modules/AddMLIR.cmake#L339-L496)) to create the actual library target.

4. **Add Object Files as Sources** (line 594):
   ```cmake
   target_sources(${name} PRIVATE ${_objects})
   ```

   **This is the key magic!** Instead of linking libraries, it directly includes their compiled object files, creating one monolithic `.so`.

5. **Enforce Self-Containment** (lines 602-606):
   ```cmake
   if((CMAKE_SYSTEM_NAME STREQUAL "Linux") AND (NOT LLVM_USE_SANITIZER))
     target_link_options(${name} PRIVATE "LINKER:-z,defs")
   endif()
   ```

   On Linux, forces the linker to report undefined symbols at build time (not runtime), ensuring the library is truly self-contained.

**Why Aggregate?**

For C APIs, you want a single self-contained `.so` that clients can link against without worrying about finding multiple dependent libraries at runtime.

---

#### Step 2: Create Test Executable (Lines 15-20)

```cmake
add_llvm_executable(listproject-capi-test
  listproject-capi-test.c         # The C source file
)
llvm_update_compile_flags(listproject-capi-test)
target_link_libraries(listproject-capi-test
  PRIVATE ListProjectCAPITestLib)
```

**Functions**:

1. **`add_llvm_executable`**: LLVM's wrapper around CMake's `add_executable`
   - Source: LLVM's `AddLLVM.cmake`
   - Adds LLVM-specific compile flags
   - Handles install rules

2. **`llvm_update_compile_flags`**: Updates compiler flags for LLVM conventions
   - Adds warning flags
   - Handles RTTI settings (LLVM defaults to `-fno-rtti`)
   - Sets up exception handling

3. **`target_link_libraries`**: Links against the aggregate library we created

**Result**: An executable `listproject-capi-test` that can:
- Create MLIR contexts and modules ([listproject-capi-test.c:26-34](../../../mlir-list/test/CAPI/listproject-capi-test.c#L26-L34))
- Register and use your List dialect ([listproject-capi-test.c:30](../../../mlir-list/test/CAPI/listproject-capi-test.c#L30))
- Parse and manipulate IR ([listproject-capi-test.c:32-44](../../../mlir-list/test/CAPI/listproject-capi-test.c#L32-L44))

---

## Part 2: Python Bindings

**File**: [mlir-list/python/CMakeLists.txt](../../../mlir-list/python/CMakeLists.txt)

### Purpose

Create Python bindings for your List dialect using nanobind (a modern, efficient alternative to pybind11).

### Architecture

```
Python Package Structure:
mlir_listproject/
├── dialects/
│   ├── list_nanobind.py          # Your handwritten wrapper
│   └── _list_ops_gen.py          # Auto-generated from TableGen
└── _mlir_libs/
    ├── _listprojectDialectsNanobind.so  # C++ extension module (nanobind)
    └── libListProjectPythonCAPI.so      # Aggregate C API library
```

### Build Process Overview

```
Phase 1: Declaration
├── declare_mlir_python_sources         (register Python files)
├── declare_mlir_dialect_python_bindings (run TableGen, register outputs)
└── declare_mlir_python_extension       (register C++ extension metadata)

Phase 2: Build
├── add_mlir_python_common_capi_library (compile aggregate C API)
└── add_mlir_python_modules             (walk dependency graph, compile everything)
```

### Detailed Walkthrough

#### Preamble (Lines 1-8)

```cmake
include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")
```

- Includes MLIR's Python CMake utilities
- Sets the package prefix for Python imports (e.g., `mlir_listproject.dialects.list`)

---

#### Step 1: Declare Python Sources (Lines 14-22)

```cmake
declare_mlir_python_sources(ListProjectPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ListProjectPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_listproject"
  TD_FILE dialects/ListOps.td        # TableGen file
  SOURCES
    dialects/list_nanobind.py        # Python wrapper
  DIALECT_NAME list)
```

##### `declare_mlir_python_sources`

- **Location**: [mlir/cmake/modules/AddMLIRPython.cmake:26-116](../../../mlir/cmake/modules/AddMLIRPython.cmake#L26-L116)
- **Purpose**: Creates a logical grouping of Python source files for installation

**Internal Process**:

```cmake
# Line 52: Create INTERFACE library (pseudo-target, no build artifact)
add_library(${name} INTERFACE)

# Lines 53-59: Store metadata as target properties
set_target_properties(${name} PROPERTIES
  mlir_python_SOURCES_TYPE pure        # Pure Python, not C++
  mlir_python_DEPENDS ""
)

# Lines 70-73: Setup include directories for build vs. install
target_include_directories(${name} INTERFACE
  "$<BUILD_INTERFACE:${ARG_ROOT_DIR}>"
  "$<INSTALL_INTERFACE:${_install_destination}>"
)
```

**Key Point**: This creates an `INTERFACE` library—a CMake pseudo-target that doesn't produce any build artifact but carries properties like source paths and dependencies.

##### `declare_mlir_dialect_python_bindings`

- **Location**: [mlir/cmake/modules/AddMLIRPython.cmake:293-340](../../../mlir/cmake/modules/AddMLIRPython.cmake#L293-L340)
- **Purpose**: Auto-generate Python wrappers from TableGen definitions

**Internal Process**:

1. **Register Python Sources** (lines 300-306):
   ```cmake
   set(_dialect_target "${ARG_ADD_TO_PARENT}.${ARG_DIALECT_NAME}")
   declare_mlir_python_sources(${_dialect_target}
     ROOT_DIR "${ARG_ROOT_DIR}"
     ADD_TO_PARENT "${ARG_ADD_TO_PARENT}"
     SOURCES "${ARG_SOURCES}"
   )
   ```

   Creates a child source group `ListProjectPythonSources.list` to organize files hierarchically.

2. **Run TableGen** (lines 309-320):
   ```cmake
   set(tblgen_target "${_dialect_target}.tablegen")
   set(td_file "${ARG_ROOT_DIR}/${ARG_TD_FILE}")
   set(dialect_filename "${relative_td_directory}/_${ARG_DIALECT_NAME}_ops_gen.py")

   set(LLVM_TARGET_DEFINITIONS ${td_file})
   mlir_tablegen("${dialect_filename}"
     -gen-python-op-bindings -bind-dialect=${ARG_DIALECT_NAME}
     DEPENDS ${ARG_DEPENDS}
   )
   add_public_tablegen_target(${tblgen_target})
   ```

   **The TableGen Pipeline**:
   ```
   ListOps.td (TableGen definition)
       ↓
   mlir-tblgen -gen-python-op-bindings -bind-dialect=list
       ↓
   _list_ops_gen.py (generated Python classes for operations)
   ```

   - **Input**: Your `ListOps.td` file with operation definitions
   - **Tool**: `mlir-tblgen` with the `-gen-python-op-bindings` backend
   - **Output**: `_list_ops_gen.py` with Python classes for each operation

3. **Register Generated Files** (lines 334-338):
   ```cmake
   declare_mlir_python_sources("${_dialect_target}.ops_gen"
     ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
     ADD_TO_PARENT "${_dialect_target}"
     SOURCES ${_sources}
   )
   ```

   Adds the generated `_list_ops_gen.py` to the Python package.

**Note**: This only *declares* files and runs TableGen. No C++ compilation yet!

---

#### Step 2: Declare C++ Extension (Lines 24-33)

```cmake
declare_mlir_python_extension(ListProjectPythonSources.NanobindExtension
  MODULE_NAME _listprojectDialectsNanobind
  ADD_TO_PARENT ListProjectPythonSources
  SOURCES
    ListProjectExtensionNanobind.cpp
  EMBED_CAPI_LINK_LIBS
    ListProjectCAPI
  PYTHON_BINDINGS_LIBRARY nanobind
  GENERATE_TYPE_STUBS
)
```

**Function**: `declare_mlir_python_extension`
- **Location**: [mlir/cmake/modules/AddMLIRPython.cmake:118-171](../../../mlir/cmake/modules/AddMLIRPython.cmake#L118-L171)
- **Purpose**: Declares metadata for a C++ Python extension module

**Internal Process**:

```cmake
# Line 134: Create INTERFACE library
add_library(${name} INTERFACE)

# Lines 135-144: Store extension metadata
set_target_properties(${name} PROPERTIES
  mlir_python_SOURCES_TYPE extension                    # Mark as C++, not pure Python
  mlir_python_EXTENSION_MODULE_NAME "${ARG_MODULE_NAME}"
  mlir_python_EMBED_CAPI_LINK_LIBS "${ARG_EMBED_CAPI_LINK_LIBS}"
  mlir_python_BINDINGS_LIBRARY "${ARG_PYTHON_BINDINGS_LIBRARY}"
)

# Lines 148-153: Store C++ source file paths
list(TRANSFORM ARG_SOURCES PREPEND "${ARG_ROOT_DIR}/" OUTPUT_VARIABLE _build_sources)
target_sources(${name} INTERFACE
  "$<BUILD_INTERFACE:${_build_sources}>"
  "$<INSTALL_INTERFACE:${_install_sources}>"
)
```

**What gets stored**:
- Source type: `extension` (not `pure` Python)
- Module name: `_listprojectDialectsNanobind`
- C API dependencies: `ListProjectCAPI`
- Binding library: `nanobind` (vs. pybind11)
- C++ source: `ListProjectExtensionNanobind.cpp`

**Still no compilation!** Just metadata storage in an INTERFACE library.

---

#### Step 3: Create Common CAPI Library (Lines 40-55)

```cmake
add_mlir_python_common_capi_library(ListProjectPythonCAPI
  INSTALL_COMPONENT ListProjectPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../../"
  DECLARED_SOURCES
    ListProjectPythonSources          # Your dialect
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.arith
    MLIRPythonSources.Dialects.memref
)
```

**Function**: `add_mlir_python_common_capi_library`
- **Location**: [mlir/cmake/modules/AddMLIRPython.cmake:480-533](../../../mlir/cmake/modules/AddMLIRPython.cmake#L480-L533)
- **Purpose**: Create the aggregate C API shared library that all Python extensions link against

**Internal Process**:

1. **Collect All CAPI Libraries** (lines 486-495):
   ```cmake
   set(_embed_libs ${ARG_EMBED_LIBS})
   _flatten_mlir_python_targets(_all_source_targets ${ARG_DECLARED_SOURCES})

   foreach(t ${_all_source_targets})
     get_target_property(_local_embed_libs ${t} mlir_python_EMBED_CAPI_LINK_LIBS)
     if(_local_embed_libs)
       list(APPEND _embed_libs ${_local_embed_libs})
     endif()
   endforeach()

   list(REMOVE_DUPLICATES _embed_libs)
   ```

   **What this does**:
   - Walks through all declared sources (your dialect + MLIR core dialects)
   - Recursively extracts their `mlir_python_EMBED_CAPI_LINK_LIBS` properties
   - Builds a complete list: `[ListProjectCAPI, MLIRCAPIIR, MLIRCAPIFunc, MLIRCAPIArith, ...]`

2. **Create Aggregate Library** (lines 497-502):
   ```cmake
   add_mlir_aggregate(${name}
     SHARED
     DISABLE_INSTALL
     EMBED_LIBS ${_embed_libs}
   )
   ```

   Uses the same `add_mlir_aggregate` mechanism as the CAPI test library!

   **Result**: `libListProjectPythonCAPI.so` containing all C API object files bundled together.

3. **Setup RPATH** (lines 525-527):
   ```cmake
   mlir_python_setup_extension_rpath(${name}
     RELATIVE_INSTALL_ROOT "${ARG_RELATIVE_INSTALL_ROOT}"
   )
   ```

   Ensures the shared library can find its dependencies at runtime. Critical for Python extensions to find each other.

4. **Install Rules** (lines 528-532):
   ```cmake
   install(TARGETS ${name}
     COMPONENT ${ARG_INSTALL_COMPONENT}
     LIBRARY DESTINATION "${ARG_INSTALL_DESTINATION}"
     RUNTIME DESTINATION "${ARG_INSTALL_DESTINATION}"
   )
   ```

   Copies the `.so` to `_mlir_libs/` directory in the install tree.

**Why This Library?**

Python extensions need a single, stable C API library to link against. This:
- Prevents symbol conflicts between extensions
- Ensures all extensions share the same MLIR context
- Simplifies dependency management (one `.so` instead of many)

**Visualization**:

```
Input Libraries (static .a archives):
├── libListProjectCAPI.a      (20 .o files)
├── libMLIRCAPIIR.a           (100 .o files)
├── libMLIRCAPIFunc.a         (50 .o files)
├── libMLIRCAPIArith.a        (30 .o files)
└── ... (more dialects)

Output (single shared library):
└── libListProjectPythonCAPI.so  (contains all .o files from above)
```

---

#### Step 4: Build Python Modules (Lines 61-76)

```cmake
add_mlir_python_modules(ListProjectPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    ListProjectPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.memref
    MLIRPythonSources.Dialects.arith
  COMMON_CAPI_LINK_LIBS
    ListProjectPythonCAPI
)
```

**Function**: `add_mlir_python_modules`
- **Location**: [mlir/cmake/modules/AddMLIRPython.cmake:209-266](../../../mlir/cmake/modules/AddMLIRPython.cmake#L209-L266)
- **Purpose**: **This is where compilation actually happens!**

**Internal Process**:

1. **Create Main Build Target** (line 253):
   ```cmake
   add_custom_target(${name} ALL)
   ```

   Creates `ListProjectPythonModules` target that depends on all sub-targets.

2. **Flatten Dependency Tree** (line 254):
   ```cmake
   _flatten_mlir_python_targets(_flat_targets ${ARG_DECLARED_SOURCES})
   ```

   Recursively walks all `DECLARED_SOURCES`, following their `mlir_python_DEPENDS` properties, producing a flat list of all sources to process.

3. **Process Each Source** (lines 255-257):
   ```cmake
   foreach(sources_target ${_flat_targets})
     _process_target(${name} ${sources_target})
   endforeach()
   ```

   For each source target, calls `_process_target` which checks the source type:

   **For Pure Python Sources** (lines 219-228):
   ```cmake
   if(_source_type STREQUAL "pure")
     set(_pure_sources_target "${modules_target}.sources.${sources_target}")
     add_mlir_python_sources_target(${_pure_sources_target}
       INSTALL_COMPONENT ${modules_target}
       INSTALL_DIR ${ARG_INSTALL_PREFIX}
       OUTPUT_DIRECTORY ${ARG_ROOT_PREFIX}
       SOURCES_TARGETS ${sources_target}
     )
     add_dependencies(${modules_target} ${_pure_sources_target})
   ```

   - Copies `.py` files to build and install trees
   - Creates symlinks (Linux/Mac) or copies (Windows)

   **For C++ Extensions** (lines 229-245):
   ```cmake
   elseif(_source_type STREQUAL "extension")
     get_target_property(_module_name ${sources_target} mlir_python_EXTENSION_MODULE_NAME)
     get_target_property(_bindings_library ${sources_target} mlir_python_BINDINGS_LIBRARY)

     set(_extension_target "${modules_target}.extension.${_module_name}.dso")
     add_mlir_python_extension(${_extension_target} "${_module_name}"
       INSTALL_COMPONENT ${modules_target}
       INSTALL_DIR "${ARG_INSTALL_PREFIX}/_mlir_libs"
       OUTPUT_DIRECTORY "${ARG_ROOT_PREFIX}/_mlir_libs"
       PYTHON_BINDINGS_LIBRARY ${_bindings_library}
       LINK_LIBS PRIVATE
         ${sources_target}
         ${ARG_COMMON_CAPI_LINK_LIBS}
     )
     add_dependencies(${modules_target} ${_extension_target})
     mlir_python_setup_extension_rpath(${_extension_target})
   ```

   - Compiles the C++ extension module
   - Links against `ListProjectPythonCAPI`
   - Outputs to `_mlir_libs/` directory

##### Building the C++ Extension

**Function**: `add_mlir_python_extension` (the build version)
- **Location**: [mlir/cmake/modules/AddMLIRPython.cmake:641-760](../../../mlir/cmake/modules/AddMLIRPython.cmake#L641-L760)
- **Purpose**: Compile the C++ extension module using nanobind

**Key Steps**:

1. **Choose Binding Library** (lines 663-714):
   ```cmake
   if(ARG_PYTHON_BINDINGS_LIBRARY STREQUAL "nanobind")
     nanobind_add_module(${libname}
       NB_DOMAIN ${MLIR_BINDINGS_PYTHON_NB_DOMAIN}
       FREE_THREADED
       ${ARG_SOURCES}
     )
   ```

   Calls nanobind's CMake function to create the Python extension module.

2. **Enable RTTI and Exceptions** (lines 651-658, 716):
   ```cmake
   set(eh_rtti_enable)
   if (MSVC)
     set(eh_rtti_enable /EHsc /GR)
   elseif(LLVM_COMPILER_IS_GCC_COMPATIBLE OR CLANG_CL)
     set(eh_rtti_enable -frtti -fexceptions)
   endif()

   target_compile_options(${libname} PRIVATE ${eh_rtti_enable})
   ```

   **Why?** LLVM normally disables RTTI (`-fno-rtti`) for smaller binaries, but Python bindings require RTTI and exceptions.

3. **Configure Output** (lines 719-724):
   ```cmake
   set_target_properties(${libname} PROPERTIES
     LIBRARY_OUTPUT_DIRECTORY ${ARG_OUTPUT_DIRECTORY}
     OUTPUT_NAME "${extname}"
     NO_SONAME ON
   )
   ```

   Sets the output name to `_listprojectDialectsNanobind.so` and output directory to `_mlir_libs/`.

4. **Link Against C API** (lines 736-739):
   ```cmake
   target_link_libraries(${libname}
     PRIVATE
     ${ARG_LINK_LIBS}
   )
   ```

   Links against:
   - The C++ source's interface library (contains source paths)
   - `ListProjectPythonCAPI` (the aggregate C API library)

5. **Hide Internal Symbols** (lines 741-746):
   ```cmake
   target_link_options(${libname}
     PRIVATE
       $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>
   )
   ```

   On Linux, hides symbols from statically linked libraries, preventing conflicts between Python extensions.

**Final Result**: `_listprojectDialectsNanobind.so` compiled and ready to be imported from Python!

---

## Key Concepts

### 1. CMake INTERFACE Libraries

**What are they?**

INTERFACE libraries are pseudo-targets that don't produce build artifacts but carry metadata:
- Source file paths (via `target_sources`)
- Include directories (via `target_include_directories`)
- Custom properties (via `set_target_properties`)
- Dependencies (via `add_dependencies`)

**Why use them?**

MLIR uses INTERFACE libraries to build a **dependency graph** before actual compilation. This allows:
- Declaring targets in any order (solves circular dependencies)
- Lazy evaluation (only compile what's needed)
- Exporting metadata for external projects

**Example**:
```cmake
add_library(MyDialectPythonSources INTERFACE)
set_target_properties(MyDialectPythonSources PROPERTIES
  mlir_python_SOURCES_TYPE pure
  mlir_python_DEPENDS ""
)
```

This creates `MyDialectPythonSources` which stores metadata but produces no build artifacts.

---

### 2. The Aggregate Pattern

**Problem**: MLIR is highly modular with dozens of dialect libraries. Python/C users want simple, monolithic APIs without managing many dependencies.

**Solution**: Bundle multiple libraries' object files into one `.so`:

```
Input (static archives):
├── libMLIRCAPIIR.a      (100 .o files)
├── libMLIRCAPIFunc.a    (50 .o files)
├── libListProjectCAPI.a (20 .o files)
        ↓
Output (single shared library):
└── libListProjectPythonCAPI.so  (contains all 170 .o files)
```

**Mechanism**:

`add_mlir_aggregate` uses generator expressions to collect object files:

```cmake
# Extract object files from each library
set(_local_objects $<TARGET_PROPERTY:${lib},MLIR_AGGREGATE_OBJECTS>)

# Add them directly as sources (not as link dependencies!)
target_sources(${name} PRIVATE ${_objects})
```

**Benefits**:
- Single library to link against
- No runtime dependency resolution needed
- Consistent symbol visibility
- All extensions share the same MLIR context

---

### 3. Two-Phase Build System

MLIR's Python bindings use a two-phase approach:

**Phase 1: Declaration**
```cmake
declare_mlir_python_sources(...)          # Register Python files
declare_mlir_dialect_python_bindings(...) # Run TableGen
declare_mlir_python_extension(...)        # Register C++ extension
```

**What happens**: Creates INTERFACE libraries storing metadata. No compilation.

**Phase 2: Build**
```cmake
add_mlir_python_common_capi_library(...)  # Compile aggregate C API
add_mlir_python_modules(...)              # Compile everything
```

**What happens**: Walks dependency graph, compiles all sources and extensions.

**Why two phases?**

1. **Dependency Resolution**: Targets can reference each other before being defined
2. **Flexibility**: External projects can inject their own sources
3. **Clarity**: Separation of "what to build" vs. "how to build it"

---

### 4. TableGen Integration

TableGen auto-generates Python wrappers from operation definitions:

```
┌─────────────┐
│ ListOps.td  │  (Your operation definitions)
└──────┬──────┘
       │
       ↓ mlir-tblgen -gen-python-op-bindings
       │
┌──────────────────┐
│ _list_ops_gen.py │  (Generated Python classes)
└────────┬─────────┘
         │
         ↓ imported by
┌────────────────────┐
│ list_nanobind.py   │  (Your handwritten wrapper)
└────────────────────┘
```

**What gets generated?**

For each operation in TableGen, Python classes are created with:
- Operation builders
- Attribute/operand accessors
- Type checking
- Documentation strings

**Example**:

TableGen definition:
```tablegen
def ListFooOp : List_Op<"foo"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType);
}
```

Generated Python (simplified):
```python
class FooOp(OpView):
    def __init__(self, input, *, loc=None, ip=None):
        # Builder logic
        ...

    @property
    def input(self):
        return self.operation.operands[0]
```

---

### 5. RTTI and Exception Handling

**LLVM Default**: `-fno-rtti -fno-exceptions`
- Smaller binaries
- Faster code
- No C++ dynamic_cast or try/catch overhead

**Python Bindings Requirement**: RTTI and exceptions **must** be enabled
- Python's type system relies on C++ RTTI
- nanobind/pybind11 use dynamic_cast extensively
- Python exceptions map to C++ exceptions

**Solution**:

```cmake
# Lines 651-658, 716 in AddMLIRPython.cmake
if(LLVM_COMPILER_IS_GCC_COMPATIBLE OR CLANG_CL)
  set(eh_rtti_enable -frtti -fexceptions)
endif()

target_compile_options(${libname} PRIVATE ${eh_rtti_enable})
```

Only Python extension modules get RTTI/exceptions. The rest of LLVM/MLIR stays `-fno-rtti`.

---

### 6. RPATH Configuration

**Problem**: Python extensions need to find shared libraries at runtime.

**Example**:
```
_listprojectDialectsNanobind.so
    ↓ needs
libListProjectPythonCAPI.so
    ↓ which might need
libMLIRSupport.so
```

**Solution**: Set RPATH (runtime library search path):

```cmake
mlir_python_setup_extension_rpath(${_extension_target}
  RELATIVE_INSTALL_ROOT "${ARG_RELATIVE_INSTALL_ROOT}"
)
```

**What this does**:
- On Linux: Sets `RPATH` in ELF header
- On macOS: Sets `@loader_path` relative references
- On Windows: Handled differently (DLLs in same directory)

**Result**: Extensions can find libraries relative to their install location, no `LD_LIBRARY_PATH` needed!

---

## Reference

### File Locations

| File | Path |
|------|------|
| CAPI CMakeLists | [mlir-list/test/CAPI/CMakeLists.txt](../../../mlir-list/test/CAPI/CMakeLists.txt) |
| Python CMakeLists | [mlir-list/python/CMakeLists.txt](../../../mlir-list/python/CMakeLists.txt) |
| CAPI Test Source | [mlir-list/test/CAPI/listproject-capi-test.c](../../../mlir-list/test/CAPI/listproject-capi-test.c) |
| Core CMake Functions | [mlir/cmake/modules/AddMLIR.cmake](../../../mlir/cmake/modules/AddMLIR.cmake) |
| Python CMake Functions | [mlir/cmake/modules/AddMLIRPython.cmake](../../../mlir/cmake/modules/AddMLIRPython.cmake) |

### Key Function Reference

| Function | Location | Purpose |
|----------|----------|---------|
| `add_mlir_aggregate` | [AddMLIR.cmake:497-629](../../../mlir/cmake/modules/AddMLIR.cmake#L497-L629) | Bundle multiple libraries' object files into one `.so` |
| `add_mlir_library` | [AddMLIR.cmake:339-496](../../../mlir/cmake/modules/AddMLIR.cmake#L339-L496) | Create MLIR library with object file support |
| `declare_mlir_python_sources` | [AddMLIRPython.cmake:26-116](../../../mlir/cmake/modules/AddMLIRPython.cmake#L26-L116) | Register Python source files as INTERFACE library |
| `declare_mlir_dialect_python_bindings` | [AddMLIRPython.cmake:293-340](../../../mlir/cmake/modules/AddMLIRPython.cmake#L293-L340) | Run TableGen, register generated Python bindings |
| `declare_mlir_python_extension` | [AddMLIRPython.cmake:118-171](../../../mlir/cmake/modules/AddMLIRPython.cmake#L118-L171) | Register C++ extension metadata |
| `add_mlir_python_common_capi_library` | [AddMLIRPython.cmake:480-533](../../../mlir/cmake/modules/AddMLIRPython.cmake#L480-L533) | Build aggregate C API library for Python |
| `add_mlir_python_modules` | [AddMLIRPython.cmake:209-266](../../../mlir/cmake/modules/AddMLIRPython.cmake#L209-L266) | Walk dependency graph, build all modules |
| `add_mlir_python_extension` (build) | [AddMLIRPython.cmake:641-760](../../../mlir/cmake/modules/AddMLIRPython.cmake#L641-L760) | Compile C++ extension using nanobind/pybind11 |

### Build Targets Created

#### CAPI Test

| Target | Type | Output |
|--------|------|--------|
| `ListProjectCAPITestLib` | Shared Library | `libListProjectCAPITestLib.so` |
| `listproject-capi-test` | Executable | `listproject-capi-test` |

#### Python Bindings

| Target | Type | Output |
|--------|------|--------|
| `ListProjectPythonSources` | INTERFACE | (metadata only) |
| `ListProjectPythonSources.list` | INTERFACE | (metadata only) |
| `ListProjectPythonSources.list.tablegen` | TableGen | `_list_ops_gen.py` |
| `ListProjectPythonSources.NanobindExtension` | INTERFACE | (metadata only) |
| `ListProjectPythonCAPI` | Shared Library | `libListProjectPythonCAPI.so` |
| `ListProjectPythonModules.extension._listprojectDialectsNanobind.dso` | Python Extension | `_listprojectDialectsNanobind.so` |
| `ListProjectPythonModules` | Custom Target | (umbrella target) |

### Common Patterns

#### Pattern 1: Adding a New Operation

1. Define in TableGen (`ListOps.td`)
2. TableGen auto-generates Python bindings
3. Optionally add handwritten wrappers in `list_nanobind.py`
4. No CMake changes needed!

#### Pattern 2: Adding a New Dialect

1. Create `declare_mlir_dialect_python_bindings` block
2. Add to `DECLARED_SOURCES` in `add_mlir_python_modules`
3. If has C API, add to `DECLARED_SOURCES` in `add_mlir_python_common_capi_library`

#### Pattern 3: Debugging Build Issues

```bash
# Verbose CMake output
cmake --build build --verbose

# See what's in the aggregate library
nm -D build/tools/mlir/python_packages/mlir_listproject/_mlir_libs/libListProjectPythonCAPI.so

# Check RPATH settings
readelf -d build/tools/mlir/python_packages/mlir_listproject/_mlir_libs/_listprojectDialectsNanobind.so | grep RPATH

# Test Python import
cd build/tools/mlir/python_packages
python -c "import mlir_listproject.dialects.list"
```

---

## Summary

The MLIR CMake build system for dialects is sophisticated but follows clear patterns:

1. **Aggregate C APIs** into single shared libraries for simple linking
2. **Use INTERFACE libraries** to build dependency graphs before compilation
3. **Two-phase approach**: declare first, build second
4. **TableGen integration** auto-generates Python bindings
5. **Special compiler flags** for Python extensions (RTTI, exceptions)
6. **RPATH configuration** ensures runtime library discovery

Understanding these patterns allows you to:
- Add new dialects with minimal boilerplate
- Debug build issues by tracing the CMake flow
- Customize the build process for your project's needs
- Export your dialect for use in other projects

For minimal CMake knowledge, remember:
- `INTERFACE` libraries = metadata storage
- `add_mlir_aggregate` = bundle object files
- `declare_*` functions = Phase 1 (registration)
- `add_*` functions = Phase 2 (compilation)
