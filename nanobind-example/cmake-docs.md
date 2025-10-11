# CMake and Ninja Build System Documentation

This document covers CMake, Ninja, and nanobind build system concepts learned through practical examples.

---

## Table of Contents

1. [Finding Python with uv](#finding-python-with-uv)
2. [CMake Path Utilities](#cmake-path-utilities)
3. [CMake Generator Expressions](#cmake-generator-expressions)
4. [Adding Compiler Flags](#adding-compiler-flags)
5. [Ninja Build System](#ninja-build-system)
6. [CMakeFiles Directory Structure](#cmakefiles-directory-structure)
7. [Disabling clang-tidy](#disabling-clang-tidy)
8. [Nanobind Macros](#nanobind-macros)
9. [RTTI (Run-Time Type Information)](#rtti-run-time-type-information)

---

## Finding Python with uv

### Problem

When using Python managed by `uv`, CMake's `find_package(Python ...)` needs to know where to find the Python installation.

### How Python3_EXECUTABLE Works

When you set `Python3_EXECUTABLE`, CMake uses it as an **anchor point** to discover all Python components:

#### 1. Interpreter Component
- CMake directly uses the executable you specified
- Validates it matches version requirements

#### 2. Development Component Discovery
CMake queries the interpreter to find development files:

```bash
# CMake internally runs:
${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
```

This returns:
- **Include directories**: Where `Python.h` and headers are located
- **Library directories**: Where `libpython3.12.so` is located
- **Library name**: The actual Python library to link against
- **Other metadata**: ABI flags, extension suffix, etc.

#### 3. uv Virtual Environment Structure

```
.venv/
├── bin/python          # The interpreter
├── include/python3.12/ # Development headers
└── lib/
    ├── python3.12/     # Standard library
    └── libpython3.12.so # Python library (if available)
```

#### 4. What CMake Sets

After `find_package(Python 3.12 COMPONENTS Interpreter Development)`:

```cmake
Python3_FOUND             # TRUE
Python3_EXECUTABLE        # /path/.venv/bin/python
Python3_INCLUDE_DIRS      # /path/.venv/include/python3.12
Python3_LIBRARIES         # /path/.venv/lib/libpython3.12.so (if available)
Python3_VERSION           # 3.12.x
Python3_VERSION_MAJOR     # 3
Python3_VERSION_MINOR     # 12
```

### Solution: Methods to Specify Python

#### Method 1: Set Python3_EXECUTABLE (Recommended for uv)

```cmake
set(Python3_EXECUTABLE "/path/to/uv/python3")
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
```

#### Method 2: Set Python3_ROOT_DIR

```cmake
set(Python3_ROOT_DIR "/path/to/uv/python")
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
```

#### Method 3: Use CMAKE_PREFIX_PATH

```cmake
list(APPEND CMAKE_PREFIX_PATH "/path/to/uv/python")
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
```

#### Method 4: For uv with .venv in parent directory

```cmake
# Get parent directory
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

# Set Python executable from parent's .venv
set(Python3_EXECUTABLE "${PARENT_DIR}/.venv/bin/python3")

# Now find_package will use this Python
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
```

### Development.Module vs Development

Your CMakeLists.txt should handle both:

```cmake
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
```

- **Development.Module** (CMake 3.18+): Only needs headers for extension modules
- **Development**: Needs both headers and libpython for embedding Python

For nanobind, you typically only need `Development.Module` since you're creating Python extensions, not embedding Python in a C++ application.

---

## CMake Path Utilities

### Getting Parent Directory

#### Method 1: get_filename_component (Works with older CMake)

```cmake
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
message(STATUS "Parent directory: ${PARENT_DIR}")
```

#### Method 2: cmake_path (CMake 3.20+, Recommended)

```cmake
cmake_path(GET CMAKE_SOURCE_DIR PARENT_PATH PARENT_DIR)
message(STATUS "Parent directory: ${PARENT_DIR}")
```

#### Going Up Multiple Levels

```cmake
# Go up one level
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

# Go up two levels
get_filename_component(GRANDPARENT_DIR ${PARENT_DIR} DIRECTORY)
```

---

## CMake Generator Expressions

Generator expressions use the syntax `$<...>` and are evaluated during **build system generation**, not during CMake configuration.

### Basic Forms

- `$<CONDITION:true_value>` - If CONDITION is true, use true_value; otherwise empty string
- `$<CONDITION:true_value:false_value>` - If true use true_value, else use false_value (CMake 3.8+)

### Example from nanobind

```cmake
function(nanobind_opt_size name)
  if (MSVC)
    target_compile_options(${name} PRIVATE $<${NB_OPT_SIZE}:$<$<COMPILE_LANGUAGE:CXX>:/Os>>)
  else()
    target_compile_options(${name} PRIVATE $<${NB_OPT_SIZE}:$<$<COMPILE_LANGUAGE:CXX>:-Os>>)
  endif()
endfunction()
```

### Breaking Down the Expression

Expression: `$<${NB_OPT_SIZE}:$<$<COMPILE_LANGUAGE:CXX>:-Os>>`

#### Level 1 (Outer): Conditional based on NB_OPT_SIZE
```cmake
$<${NB_OPT_SIZE}:...>
```
- `${NB_OPT_SIZE}` is evaluated at configuration time to something like `$<CONFIG:Release>`
- **Meaning**: "Only apply the inner expression if NB_OPT_SIZE condition is true"

#### Level 2 (Middle): Check compile language
```cmake
$<$<COMPILE_LANGUAGE:CXX>:-Os>
```
- **Condition**: `$<COMPILE_LANGUAGE:CXX>` evaluates to `1` if compiling C++ files, `0` otherwise
- **Value**: `-Os` (optimize for size)
- **Meaning**: "If compiling C++, use -Os"

#### Level 3 (Inner): The actual flag
```cmake
-Os    # GCC/Clang: optimize for size
/Os    # MSVC: optimize for size
```

### Full Expression Flow Example

Assume:
- `NB_OPT_SIZE` is set to `$<CONFIG:Release>`
- We're building in Release mode
- We're compiling a C++ file

#### Step-by-step evaluation:

1. **Configuration time** (when CMake runs):
   ```cmake
   $<${NB_OPT_SIZE}:$<$<COMPILE_LANGUAGE:CXX>:-Os>>
   # ${NB_OPT_SIZE} expands to: $<CONFIG:Release>
   $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CXX>:-Os>>
   ```

2. **Generation time** (when build files are created):
   ```cmake
   # We're in Release mode, so $<CONFIG:Release> = 1
   $<1:$<$<COMPILE_LANGUAGE:CXX>:-Os>>
   # Outer condition is true, so we keep the inner part:
   $<$<COMPILE_LANGUAGE:CXX>:-Os>
   ```

3. **Per-file compilation**:
   - For a `.cpp` file: `$<COMPILE_LANGUAGE:CXX>` = `1` → Result: `-Os`
   - For a `.c` file: `$<COMPILE_LANGUAGE:CXX>` = `0` → Result: `` (empty)

### Why This Complexity?

This multi-level nesting allows:
1. **Build configuration control**: Only optimize for size in certain configurations
2. **Language-specific flags**: Only apply to C++ files, not C or other languages
3. **Clean separation**: Different flags for different compilers

### Common Generator Expressions

```cmake
# Configuration checks
$<CONFIG:Release>                    # True if Release build
$<CONFIG:Debug,RelWithDebInfo>       # True if Debug OR RelWithDebInfo

# Compiler checks
$<CXX_COMPILER_ID:GNU>              # True if GCC
$<CXX_COMPILER_ID:Clang,AppleClang> # True if Clang or AppleClang

# Language checks
$<COMPILE_LANGUAGE:CXX>             # True when compiling C++
$<COMPILE_LANGUAGE:C>               # True when compiling C

# Boolean operations
$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>  # Both must be true
$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>> # Either can be true
$<NOT:$<CONFIG:Debug>>                          # Negation

# Platform checks
$<PLATFORM_ID:Linux>                # True on Linux
$<PLATFORM_ID:Windows>              # True on Windows

# Practical examples
$<$<CONFIG:Debug>:-g3>              # Add -g3 only in Debug
$<$<CXX_COMPILER_ID:GNU>:-Wall>     # Add -Wall only for GCC
```

### Equivalent Non-Generator Expression Code

For comparison, here's what the code would look like without generator expressions:

```cmake
function(nanobind_opt_size name)
  if (MSVC)
    if (CMAKE_BUILD_TYPE STREQUAL "Release")  # Less flexible
      target_compile_options(${name} PRIVATE /Os)  # Applies to ALL files
    endif()
  else()
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
      target_compile_options(${name} PRIVATE -Os)
    endif()
  endif()
endfunction()
```

The generator expression version is better because:
- Works with multi-config generators (Visual Studio, Xcode)
- Per-file language detection
- Evaluated at build time, not configure time

---

## Adding Compiler Flags

### Method 1: target_compile_options (Recommended)

```cmake
target_compile_options(my_target PRIVATE
  -O0
  -g
  -fno-omit-frame-pointer
)
```

**Each flag should be a separate item** in the list (not a single string).

### Method 2: Separate flags by type (More Explicit)

```cmake
# Optimization level
target_compile_options(my_target PRIVATE -O0)

# Debug symbols
target_compile_options(my_target PRIVATE -g)

# Frame pointer
target_compile_options(my_target PRIVATE -fno-omit-frame-pointer)
```

### Method 3: Handle different compilers

```cmake
target_compile_options(my_target PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O0>
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-g>
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-omit-frame-pointer>
  $<$<CXX_COMPILER_ID:MSVC>:/Od>
  $<$<CXX_COMPILER_ID:MSVC>:/Zi>
)
```

### Method 4: Configuration-specific (e.g., Debug only)

```cmake
target_compile_options(my_target PRIVATE
  $<$<CONFIG:Debug>:-O0>
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
)
```

### PRIVATE vs PUBLIC vs INTERFACE

- **PRIVATE**: Flags only apply to this target (most common for executables/modules)
- **INTERFACE**: Flags apply to targets that link to this target (for header-only libraries)
- **PUBLIC**: Flags apply to both this target AND targets that link to it

For a Python extension module, use **PRIVATE**.

### What NOT to Do

```cmake
# BAD: Modifies global flags for ALL targets
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -fno-omit-frame-pointer")

# BAD: Flags as single string
target_compile_options(my_target PRIVATE "-O0 -g -fno-omit-frame-pointer")

# BAD: add_compile_options affects ALL targets in directory
add_compile_options(-O0 -g -fno-omit-frame-pointer)
```

### Why target_compile_options is Better

- **Target-specific**: Only affects your extension, not dependencies
- **Build system agnostic**: Works correctly with all generators (Make, Ninja, Visual Studio)
- **Composable**: Flags are properly de-duplicated and ordered
- **Maintainable**: Easy to see what flags apply to which targets

---

## Ninja Build System

### Overview

Ninja is a small build system focused on speed. It's designed to have its input files (build.ninja) generated by higher-level build systems like CMake.

### Ninja vs Make

| Feature | Ninja | Make |
|---------|-------|------|
| **Purpose** | Fast execution | General build tool |
| **Config** | Generated (by CMake) | Written by hand |
| **Syntax** | Simple, strict | Flexible, complex |
| **Variables** | `=` only | `=`, `:=`, `?=`, etc. |
| **Dependencies** | 3 types | 1 type |
| **Speed** | Very fast | Slower |
| **Use case** | Large projects | Small projects, automation |

### Basic Ninja Syntax

The core of Ninja is the `build` statement:

```ninja
build OUTPUT: RULE INPUT | IMPLICIT_DEP || ORDER_ONLY_DEP
  VARIABLE = value
```

### Dependency Types

| Symbol | Type | Meaning |
|--------|------|---------|
| (space) | Normal | If input changes, rebuild |
| `|` | Implicit | If changes, rebuild (but not shown in command) |
| `||` | Order-only | Must exist, but changes don't trigger rebuild |

### Example: Object File Compilation

```ninja
build CMakeFiles/my_ext.dir/my_ext.cpp.o: CXX_COMPILER__my_ext_unscanned_RelWithDebInfo /home/jeromeku/llvm-project/nanobind-example/my_ext.cpp || cmake_object_order_depends_target_my_ext
  CONFIG = RelWithDebInfo
  DEFINES = -Dmy_ext_EXPORTS
  DEP_FILE = CMakeFiles/my_ext.dir/my_ext.cpp.o.d
  FLAGS = -O2 -g -DNDEBUG -fPIC -fvisibility=hidden -O0 -g -fno-omit-frame-pointer -ffunction-sections -fdata-sections
  INCLUDES = -I/home/jeromeku/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/include/python3.12 -I/home/jeromeku/llvm-project/nanobind-example/ext/nanobind/include
  OBJECT_DIR = CMakeFiles/my_ext.dir
  OBJECT_FILE_DIR = CMakeFiles/my_ext.dir
```

**Parsing this:**
- **OUTPUT**: `CMakeFiles/my_ext.dir/my_ext.cpp.o` (the .o file)
- **RULE**: `CXX_COMPILER__my_ext_unscanned_RelWithDebInfo` (defined in rules.ninja)
- **INPUT**: `/home/jeromeku/llvm-project/nanobind-example/my_ext.cpp` (source file)
- **ORDER-ONLY DEP** (`||`): `cmake_object_order_depends_target_my_ext`

**Indented lines** set variables for this specific build statement. These get passed to the rule.

### Example: Linking

```ninja
build my_ext.cpython-312-x86_64-linux-gnu.so: CXX_MODULE_LIBRARY_LINKER__my_ext_RelWithDebInfo CMakeFiles/my_ext.dir/my_ext.cpp.o | libnanobind-static.a || libnanobind-static.a
  LINK_FLAGS = -shared -Wl,--gc-sections
  LINK_LIBRARIES = libnanobind-static.a
```

**Parsing:**
- **OUTPUT**: `my_ext.cpython-312-x86_64-linux-gnu.so` (the Python extension)
- **RULE**: `CXX_MODULE_LIBRARY_LINKER__my_ext_RelWithDebInfo`
- **INPUT**: `CMakeFiles/my_ext.dir/my_ext.cpp.o` (the compiled object)
- **IMPLICIT DEP** (`|`): `libnanobind-static.a` (if this changes, rebuild)
- **ORDER-ONLY DEP** (`||`): `libnanobind-static.a` (must exist first)

### Phony Targets and Aliases

#### Why `phony` is Needed

Without `phony`, Ninja would think:
1. "I need to create a **file** called `my_ext`"
2. If a file named `my_ext` exists, Ninja checks its timestamp
3. Ninja might think the build is up-to-date when it's not

With `phony`:
```ninja
build my_ext: phony my_ext.cpython-312-x86_64-linux-gnu.so
```

1. ✅ "my_ext is **not a real file**, just a name"
2. ✅ "Always check if dependencies need rebuilding"
3. ✅ "Ignore any actual file named `my_ext` that might exist"

#### Real-World Example

```bash
# Without phony - BAD
$ touch my_ext                    # Accidentally create a file named "my_ext"
$ ninja my_ext                    # Ninja: "my_ext file exists and is newer, nothing to do"
ninja: no work to do.             # WRONG!

# With phony - GOOD
$ touch my_ext                    # Create a file named "my_ext"
$ ninja my_ext                    # Ninja: "my_ext is phony, check its dependencies"
[builds the .so if needed]        # CORRECT!
```

#### Common Uses of Phony

1. **Aliases**:
   ```ninja
   build my_ext: phony my_ext.cpython-312-x86_64-linux-gnu.so
   ```

2. **Aggregate targets**:
   ```ninja
   build all: phony target1 target2 target3
   ```

3. **Actions without outputs**:
   ```ninja
   build test: phony
     command = pytest
   ```

4. **Order-only dependencies**:
   ```ninja
   build cmake_object_order_depends_target_my_ext: phony || cmake_object_order_depends_target_nanobind-static
   ```

### Rules in rules.ninja

Rules define **HOW** to execute commands:

#### Rule 1: C++ Compilation

```ninja
rule CXX_COMPILER__my_ext_unscanned_RelWithDebInfo
  depfile = $DEP_FILE
  deps = gcc
  command = ${LAUNCHER}${CODE_CHECK}/usr/bin/c++ $DEFINES $INCLUDES $FLAGS -MD -MT $out -MF $DEP_FILE -o $out -c $in
  description = Building CXX object $out
```

**Breaking it down:**

| Line | What it does |
|------|-------------|
| `depfile = $DEP_FILE` | Where to write dependency info (.d file) |
| `deps = gcc` | Use GCC-style dependency tracking |
| `command = ...` | The actual compilation command |
| `-MD -MT $out -MF $DEP_FILE` | Generate dependencies for headers |
| `-o $out -c $in` | Output object file from input source |
| `description = ...` | What Ninja shows when running |

**Key variables:**
- `$in`: Input file (e.g., `my_ext.cpp`)
- `$out`: Output file (e.g., `my_ext.cpp.o`)
- `$DEP_FILE`: Dependency file (e.g., `my_ext.cpp.o.d`)

#### Rule 2: Linking Shared Module

```ninja
rule CXX_MODULE_LIBRARY_LINKER__my_ext_RelWithDebInfo
  command = $PRE_LINK && /usr/bin/c++ -fPIC $LANGUAGE_COMPILE_FLAGS $ARCH_FLAGS $LINK_FLAGS $SONAME_FLAG$SONAME -o $TARGET_FILE $in $LINK_PATH $LINK_LIBRARIES && $POST_BUILD
  description = Linking CXX shared module $TARGET_FILE
  restat = $RESTAT
```

**What this does:**
1. `$PRE_LINK`: Run commands before linking (usually `:` = nothing)
2. `/usr/bin/c++`: Use C++ compiler to link
3. `-fPIC`: Position Independent Code (required for shared libraries)
4. `-o $TARGET_FILE`: Output the `.so` file
5. `$in`: Input object files
6. `$LINK_LIBRARIES`: Libraries to link against
7. `$POST_BUILD`: Run commands after linking
8. `restat = $RESTAT`: Re-check file timestamp after linking

**Why restat?** Sometimes linking doesn't change the output file (e.g., only debug symbols changed). `restat` prevents unnecessary downstream rebuilds.

#### Rule 3: Static Library

```ninja
rule CXX_STATIC_LIBRARY_LINKER__nanobind-static_RelWithDebInfo
  command = $PRE_LINK && cmake -E rm -f $TARGET_FILE && /usr/bin/ar qc $TARGET_FILE $LINK_FLAGS $in && /usr/bin/ranlib $TARGET_FILE && $POST_BUILD
  description = Linking CXX static library $TARGET_FILE
  restat = $RESTAT
```

**Command breakdown:**
1. `cmake -E rm -f $TARGET_FILE`: Delete old library
2. `/usr/bin/ar qc $TARGET_FILE $in`: Create archive from object files
   - `q`: Quick append
   - `c`: Create if doesn't exist
3. `/usr/bin/ranlib $TARGET_FILE`: Create index for fast symbol lookup

### Useful Ninja Commands

```bash
ninja                  # Build default target (all)
ninja my_ext          # Build specific target
ninja -t targets      # List all targets
ninja -t clean        # Clean built files
ninja -n              # Dry run (show commands)
ninja -v              # Verbose (show full commands)
ninja -t graph        # Generate dependency graph
ninja -t compdb       # Generate compile_commands.json
```

### How Ninja Executes Builds

1. **Parse build.ninja**: Read all build statements
2. **Build dependency graph**: Determine order
3. **Check timestamps**: What's out of date?
4. **Execute in parallel**: Run independent builds simultaneously
5. **Track dependencies**: `.d` files track header dependencies

---

## CMakeFiles Directory Structure

### Overview

```
CMakeFiles/
├── 4.1.0/                              # CMake version-specific files
│   ├── CompilerIdC/                    # C compiler detection
│   │   ├── CMakeCCompilerId.c          # Test program
│   │   ├── a.out                       # Compiled test
│   │   └── tmp/                        # Temporary files
│   ├── CompilerIdCXX/                  # C++ compiler detection
│   │   ├── CMakeCXXCompilerId.cpp      # Test program
│   │   ├── a.out                       # Compiled test
│   │   └── tmp/
│   ├── CMakeDetermineCompilerABI_C.bin # C ABI test executable
│   ├── CMakeDetermineCompilerABI_CXX.bin # C++ ABI test executable
│   ├── CMakeCCompiler.cmake            # C compiler properties
│   ├── CMakeCXXCompiler.cmake          # C++ compiler properties
│   └── CMakeSystem.cmake               # System detection results
├── my_ext.dir/                         # Build directory for my_ext
├── nanobind-static.dir/                # Build directory for nanobind-static
│   └── ext/nanobind/src/               # Mirrors source tree
├── CMakeScratch/                       # Temporary scratch space
├── pkgRedirects/                       # Package config redirects
├── rules.ninja                         # Ninja rule definitions
├── TargetDirectories.txt               # List of target directories
├── cmake.check_cache                   # Cache validation
├── CMakeConfigureLog.yaml              # Detailed config log
└── InstallScripts.json                 # Installation manifest
```

### Root Level Files

#### TargetDirectories.txt

Lists all target build directories:
```
/home/jeromeku/llvm-project/nanobind-example/build/CMakeFiles/my_ext.dir
/home/jeromeku/llvm-project/nanobind-example/build/CMakeFiles/nanobind-static.dir
```

**Purpose:** Quick reference for CMake during incremental builds
**Used by:** CMake, IDEs

#### cmake.check_cache

```
# This file is generated by cmake for dependency checking of the CMakeCache.txt file
```

**Purpose:** Marker file to track if CMakeCache.txt is valid
**How it works:**
- If missing or older than CMakeLists.txt, CMake re-runs configuration
- Acts as a timestamp sentinel

#### InstallScripts.json

```json
{
  "InstallScripts": [
    "cmake_install.cmake",
    "ext/nanobind/cmake_install.cmake"
  ],
  "Parallel": false
}
```

**Purpose:** Tells CMake which install scripts to run during `ninja install`

#### CMakeConfigureLog.yaml

Detailed log of everything CMake did during configuration:

```yaml
events:
  - kind: "find-v1"
    mode: "program"
    variable: "CMAKE_UNAME"
    found: "/usr/bin/uname"
```

**What it logs:**
- Every `find_program()` call
- Every `find_package()` search
- Every compiler detection attempt
- Search paths and results

**Why useful:**
- Debugging "package not found" errors
- Understanding why CMake chose specific tools
- Audit trail of configuration decisions

### The 4.1.0/ Directory - Compiler Detection

#### CompilerIdCXX/CMakeCXXCompilerId.cpp

A **generated test program** that CMake compiles to detect your compiler:

```cpp
#if defined(__GNUC__)
# define COMPILER_ID "GNU"
# define COMPILER_VERSION_MAJOR DEC(__GNUC__)
# define COMPILER_VERSION_MINOR DEC(__GNUC_MINOR__)
#endif

#if defined(__clang__)
# define COMPILER_ID "Clang"
# define COMPILER_VERSION_MAJOR DEC(__clang_major__)
#endif
```

**The process:**
1. CMake generates this source with tons of `#if defined(...)` checks
2. Compiles it: `c++ CMakeCXXCompilerId.cpp -o a.out`
3. Runs or inspects the binary to extract:
   - Compiler ID (GNU, Clang, MSVC, Intel, etc.)
   - Compiler version (11.4.0)
   - Architecture (x86_64)

**The a.out binary** embeds strings that CMake can parse:
```bash
$ strings a.out | grep "INFO:"
INFO:compiler[GNU]
INFO:platform[Linux]
INFO:arch[x86_64]
```

#### CMakeDetermineCompilerABI_CXX.bin

**Purpose:** Detects compiler **ABI** (Application Binary Interface)

**What it tests:**
- Size of fundamental types
- Endianness
- Default library paths
- Implicit link directories
- Implicit link libraries

**How CMake uses it:**
```bash
$ /usr/bin/c++ -v CMakeDetermineCompilerABI_CXX.bin
```

The `-v` output reveals implicit paths and libraries, which go into `CMakeCXXCompiler.cmake`.

#### CMakeCXXCompiler.cmake

Result of compiler detection:

```cmake
set(CMAKE_CXX_COMPILER "/usr/bin/c++")
set(CMAKE_CXX_COMPILER_ID "GNU")
set(CMAKE_CXX_COMPILER_VERSION "11.4.0")
set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CXX_COMPILE_FEATURES "cxx_std_98;cxx_std_11;...;cxx_std_23")
```

**Used for:**
- Generator expressions: `$<CXX_COMPILER_ID:GNU>`
- Feature requirements: `target_compile_features(my_ext PRIVATE cxx_std_17)`

#### CMakeSystem.cmake

System detection results:

```cmake
set(CMAKE_HOST_SYSTEM "Linux-5.15.0-157-generic")
set(CMAKE_HOST_SYSTEM_NAME "Linux")
set(CMAKE_HOST_SYSTEM_PROCESSOR "x86_64")
set(CMAKE_CROSSCOMPILING "FALSE")
```

When cross-compiling, `CMAKE_HOST_*` differs from `CMAKE_SYSTEM_*`.

### Target Build Directories

#### my_ext.dir/

Holds build artifacts for the `my_ext` target:

```
my_ext.dir/
├── my_ext.cpp.o         # Compiled object file
├── my_ext.cpp.o.d       # Dependency file
```

Initially empty; populated during build.

#### nanobind-static.dir/ext/nanobind/src/

Mirrors source tree structure for object files:

```
Source:                          Build:
ext/nanobind/src/nb_func.cpp  → nanobind-static.dir/ext/nanobind/src/nb_func.cpp.o
```

**Benefits:**
- No name collisions
- Clear organization
- Out-of-source builds keep source tree clean

### Build Process Flow

#### Phase 1: Initial CMake Run

```
1. User: cmake -S . -B build -G Ninja
   ↓
2. CMake creates build/CMakeFiles/
   ↓
3. Compiler Detection:
   - Generate CompilerIdCXX/CMakeCXXCompilerId.cpp
   - Compile → a.out
   - Parse → CMakeCXXCompiler.cmake
   ↓
4. ABI Detection:
   - Compile CMakeDetermineCompilerABI_CXX.bin
   - Run with -v
   - Parse implicit paths
   ↓
5. System Detection:
   - Run uname → CMakeSystem.cmake
   ↓
6. Create target directories
   ↓
7. Generate build files:
   - build.ninja
   - rules.ninja
   - compile_commands.json
   ↓
8. Log → CMakeConfigureLog.yaml
```

#### Phase 2: Incremental CMake Run

```
1. User: cmake -S . -B build
   ↓
2. Check cmake.check_cache
   ↓
3. If CMakeLists.txt unchanged:
   - Skip compiler detection
   - Regenerate only build.ninja
   ↓
4. If changed:
   - Re-run configuration
```

#### Phase 3: Build

```
1. User: ninja
   ↓
2. Read build.ninja and rules.ninja
   ↓
3. Compile sources → .o files in target.dir/
   ↓
4. Link → .so files
```

### File Lifecycle Summary

| File/Directory | Created When | Updated When | Purpose |
|----------------|--------------|--------------|---------|
| `4.1.0/` | First cmake | Never (unless CMake version changes) | Compiler detection |
| `my_ext.dir/` | Configuration | Build time | Object files |
| `rules.ninja` | Every cmake | Every cmake | Rule definitions |
| `CMakeConfigureLog.yaml` | Every cmake | Every cmake | Config audit log |

---

## Disabling clang-tidy

### Method 1: Add NOLINT Comment (Recommended for nanobind)

```cpp
// NOLINTBEGIN
#include <nanobind/nanobind.h>

// Your code here

NB_MODULE(my_ext, m) {
    m.def("add", &add);
}
// NOLINTEND
```

### Method 2: CMake - Remove clang-tidy for Target

```cmake
# For entire target
set_target_properties(my_target PROPERTIES CXX_CLANG_TIDY "")

# For specific source files
set_source_files_properties(my_ext.cpp PROPERTIES
  CXX_CLANG_TIDY ""
)
```

---

## Nanobind Macros

### The NB_MODULE Macro

When you write:
```cpp
NB_MODULE(my_ext, m) {
    m.def("add", &add);
}
```

This expands into multiple generated functions.

### Macro Expansion

Located in `ext/nanobind/include/nanobind/nb_defs.h`:

```cpp
#define NB_MODULE(name, variable) NB_MODULE_IMPL(name, variable)
#define NB_MODULE_IMPL(name, variable) NB_MODULE_IMPL2(name, variable)
```

The double indirection ensures macro arguments are fully expanded before token pasting.

### What NB_MODULE_IMPL2 Generates

#### 1. Forward Declaration
```cpp
static void nanobind_my_ext_exec_impl(nanobind::module_);
```

#### 2. Wrapper Function
```cpp
static int nanobind_my_ext_exec(PyObject *m) {
    nanobind::detail::init(NB_DOMAIN_STR);
    try {
        nanobind_my_ext_exec_impl(
            nanobind::borrow<nanobind::module_>(m));
        return 0;
    } catch (nanobind::python_error &e) {
        e.restore();
        nanobind::chain_error(PyExc_ImportError,
            "Encountered an error while initializing the extension.");
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_ImportError, e.what());
    }
    return -1;
}
```

**What this does:**
- Wraps your binding code in exception handling
- Converts `PyObject *m` to `nanobind::module_`
- Calls the actual implementation function

#### 3. Module Definition Structures
```cpp
static PyModuleDef_Slot nanobind_my_ext_slots[] = {
    { Py_mod_exec, (void *) nanobind_my_ext_exec },
    NB_MODULE_SLOTS_2
};

static struct PyModuleDef nanobind_my_ext_module = {
    PyModuleDef_HEAD_INIT, "my_ext", nullptr, 0, nullptr,
    nanobind_my_ext_slots, nullptr, nullptr, nullptr
};
```

#### 4. Module Initialization Function
```cpp
extern "C" [[maybe_unused]] NB_EXPORT PyObject *PyInit_my_ext(void);
extern "C" PyObject *PyInit_my_ext(void) {
    return PyModuleDef_Init(&nanobind_my_ext_module);
}
```

This is the entry point Python calls when importing your module.

#### 5. Function Definition Start
```cpp
void nanobind_my_ext_exec_impl(nanobind::module_ m)
```

**Your code block becomes the body of this function!**

### Token Pasting

The `##` operator concatenates tokens:
```cpp
nanobind_##name##_exec_impl
// With name=my_ext becomes:
nanobind_my_ext_exec_impl
```

### The Flow When Python Imports

```
1. Python calls: PyInit_my_ext()
   ↓
2. PyInit_my_ext() returns module with slots
   ↓
3. Python's module system calls: nanobind_my_ext_exec(PyObject *m)
   ↓
4. nanobind_my_ext_exec() wraps in try/catch
   ↓
5. Calls: nanobind_my_ext_exec_impl(nanobind::borrow<nanobind::module_>(m))
   ↓
6. Your binding code executes: m.def("add", &add);
```

### Why This Design?

1. **Exception Safety**: Wrapper catches C++ and Python exceptions
2. **Type Safety**: Converts raw `PyObject*` to type-safe `nanobind::module_`
3. **Python C API Compliance**: Follows PEP 489 multi-phase initialization
4. **Clean Syntax**: You just write `NB_MODULE(name, m) { /* bindings */ }`

---

## compile_commands.json - The Compilation Database

### What It Is

JSON database of **exact commands** used to compile each file:

```json
{
  "directory": "/home/jeromeku/llvm-project/nanobind-example/build",
  "command": "/usr/bin/c++ -Dmy_ext_EXPORTS -I... -O0 -g -fno-omit-frame-pointer -o CMakeFiles/my_ext.dir/my_ext.cpp.o -c my_ext.cpp",
  "file": "/home/jeromeku/llvm-project/nanobind-example/my_ext.cpp",
  "output": "/home/jeromeku/llvm-project/nanobind-example/build/CMakeFiles/my_ext.dir/my_ext.cpp.o"
}
```

### Who Uses It

1. **clangd** (LSP): Code completion, go-to-definition
2. **clang-tidy**: Static analysis
3. **clang-format**: Code formatting
4. **IDEs**: VSCode, CLion, etc.

### How Tools Use It

```bash
# clangd looks for compile_commands.json in parent directories
clangd --compile-commands-dir=build

# clang-tidy uses it to know how to parse your code
clang-tidy my_ext.cpp -p build
```

---

## Summary

This documentation covers the essential CMake and Ninja build system concepts:

1. **Python Discovery**: How CMake finds Python via `Python3_EXECUTABLE`
2. **Path Utilities**: Getting parent directories in CMake
3. **Generator Expressions**: Conditional compilation flags evaluated at build time
4. **Compiler Flags**: Idiomatic ways to add flags with `target_compile_options`
5. **Ninja Build System**: How Ninja organizes builds with rules and build statements
6. **CMakeFiles Structure**: Internal organization of CMake's generated files
7. **Compiler Detection**: How CMake identifies compilers and their capabilities
8. **Nanobind Macros**: How `NB_MODULE` expands into Python C API code

These concepts form the foundation for understanding modern CMake-based build systems.

---

## RTTI (Run-Time Type Information)

### What is RTTI?

**RTTI (Run-Time Type Information)** is a C++ feature that allows programs to determine the type of an object at runtime. It's essential for:
- Type-safe downcasting with `dynamic_cast`
- Runtime type identification with `typeid`
- Type-based exception handling

### Why nanobind Uses -fno-rtti

nanobind and many Python binding libraries compile with `-fno-rtti` because:

1. **Binary size reduction**: 30-50% smaller binaries
2. **Python's type system**: Python has its own runtime type checking
3. **No need for dynamic_cast**: Type conversions are explicit in binding code
4. **Performance**: Slightly smaller vtables, better cache locality

### RTTI Features (Require RTTI Enabled)

#### 1. typeid Operator
```cpp
Animal* animal = new Dog();
std::cout << typeid(*animal).name();    // "3Dog" (mangled)
std::cout << typeid(*animal).hash_code(); // Unique hash

if (typeid(*animal) == typeid(Dog)) {
    // It's definitely a Dog
}
```

#### 2. dynamic_cast Operator
```cpp
Animal* animal = get_animal();

// Safe downcast - returns nullptr if wrong type
if (Dog* dog = dynamic_cast<Dog*>(animal)) {
    dog->bark();
}

// With references - throws std::bad_cast if wrong
Dog& dog = dynamic_cast<Dog&>(*animal);
```

### What Happens with -fno-rtti

#### Compilation Errors
```cpp
typeid(*animal)              // ERROR: cannot use 'typeid' with '-fno-rtti'
dynamic_cast<Dog*>(animal)   // ERROR: 'dynamic_cast' not permitted with '-fno-rtti'
```

#### What Still Works
```cpp
animal->speak()              // ✅ Virtual functions work
static_cast<Dog*>(animal)    // ✅ Static cast works (no runtime check!)
catch (MyException& e)       // ✅ Exception handling works
```

### Binary Size Impact

Example from our test (see `rtti-explained.md` for details):

```bash
$ ls -lh rtti-example-*
-rwxrwxr-x 1 user user 36K  rtti-example-with-rtti
-rwxrwxr-x 1 user user 20K  rtti-example-no-rtti
# Savings: 16K (44% reduction)

$ size rtti-example-*
   text    data     bss     dec     hex filename
  15405    1360     280   17045    4295 rtti-example-with-rtti
   8770    1000     280   10050    2742 rtti-example-no-rtti
```

**What gets removed:**
- `typeinfo` structures for each polymorphic class
- `typeinfo` name strings (mangled names)
- `typeinfo` pointers in vtables (8 bytes per vtable)
- Code for `dynamic_cast` and `typeid` operations

### Workarounds Without RTTI

When RTTI is disabled, use manual type identification:

#### Pattern 1: Type Enum (Recommended)
```cpp
class Animal {
public:
    enum class Type { ANIMAL, DOG, CAT };
    virtual Type get_type() const { return Type::ANIMAL; }
};

class Dog : public Animal {
    Type get_type() const override { return Type::DOG; }
};

// Usage
if (animal->get_type() == Animal::Type::DOG) {
    Dog* dog = static_cast<Dog*>(animal);  // Safe after check
    dog->bark();
}
```

#### Pattern 2: Virtual Type Checking
```cpp
class Animal {
public:
    virtual bool is_dog() const { return false; }
    virtual bool is_cat() const { return false; }
};

class Dog : public Animal {
    bool is_dog() const override { return true; }
};
```

#### Pattern 3: Compile-time Hash
```cpp
class Dog : public Animal {
    static constexpr size_t type_id = compute_hash("Dog");
public:
    size_t get_type_id() const override { return type_id; }
};

constexpr size_t compute_hash(const char* str) {
    size_t hash = 5381;
    while (*str) hash = ((hash << 5) + hash) + *str++;
    return hash;
}
```

### VTable Structure Difference

#### With RTTI
```
VTable for Dog:
+------------------------+
| offset_to_top          |
+------------------------+
| typeinfo for Dog*      | ← 8 bytes overhead
+------------------------+
| Dog::~Dog()            |
| Dog::speak()           |
| Dog::get_name()        |
+------------------------+
```

#### Without RTTI (-fno-rtti)
```
VTable for Dog:
+------------------------+
| offset_to_top          |
+------------------------+
| Dog::~Dog()            | ← No typeinfo pointer!
| Dog::speak()           |
| Dog::get_name()        |
+------------------------+
```

Each vtable saves 8 bytes (typeinfo pointer removed).

### Symbol Analysis

#### With RTTI - nm output
```bash
$ nm -C binary-with-rtti | grep -i typeinfo
0000000000006c08 V typeinfo for Dog
0000000000006c20 V typeinfo for Cat
00000000000042ed V typeinfo name for Dog
00000000000042e8 V typeinfo name for Cat
```

#### Without RTTI - nm output
```bash
$ nm -C binary-no-rtti | grep -i typeinfo
# No typeinfo symbols!
```

### When to Disable RTTI

**Disable RTTI (-fno-rtti) when:**
- Binary size is critical (embedded, mobile)
- Building Python bindings (Python has its own type system)
- Platform requires it (some game consoles)
- Want to enforce explicit type handling
- Linking with libraries compiled without RTTI

**Keep RTTI enabled when:**
- Using `dynamic_cast` or `typeid` in your code
- Need standard type introspection
- Using libraries that require RTTI
- Debugging polymorphic code (better error messages)
- Building serialization/reflection systems

### CMake Configuration

```cmake
# Disable RTTI
target_compile_options(my_target PRIVATE -fno-rtti)

# Enable RTTI explicitly (usually default)
target_compile_options(my_target PRIVATE -frtti)

# Conditional based on compiler
target_compile_options(my_target PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti>
  $<$<CXX_COMPILER_ID:MSVC>:/GR->
)
```

### Important Warnings

⚠️ **ABI Compatibility**: All linked code must use the same RTTI setting. Mixing RTTI and no-RTTI code causes undefined behavior.

⚠️ **Static vs Dynamic Cast**: Without RTTI, `static_cast` has no runtime checking. You must ensure type safety manually:

```cpp
// WITH RTTI - Safe
Dog* dog = dynamic_cast<Dog*>(animal);  // Returns nullptr if wrong
if (dog) dog->bark();

// WITHOUT RTTI - Unsafe if not checked first!
Dog* dog = static_cast<Dog*>(animal);   // NO runtime check!
dog->bark();  // Undefined behavior if animal is not a Dog!

// WITHOUT RTTI - Safe pattern
if (animal->get_type() == Animal::Type::DOG) {
    Dog* dog = static_cast<Dog*>(animal);  // Now safe
    dog->bark();
}
```

### See Also

For complete examples, binary analysis, and detailed explanations, see [rtti-explained.md](rtti-explained.md).

Example files:
- [rtti-example.cpp](rtti-example.cpp) - Full RTTI demo
- [rtti-example-no-rtti.cpp](rtti-example-no-rtti.cpp) - Workarounds without RTTI
