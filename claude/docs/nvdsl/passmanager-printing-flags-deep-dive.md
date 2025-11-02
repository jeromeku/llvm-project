# PassManager Printing Flags Deep Dive

## Overview

This document explains how `enableDebugInfo` and `printGenericOpForm` flags work when using PassManager's `enable_ir_printing()` method, tracing from Python through all implementation layers.

## High-Level Purpose

### `enableDebugInfo` Flag

**Purpose**: Includes source location information in the printed IR

**Effect**: When enabled, every operation in the printed MLIR shows where it came from (file, line, column)

**Use case**: Debugging passes to track which source code location generated which IR operations

**Example output difference**:
```mlir
// Without debug info:
%0 = arith.addi %arg0, %arg1 : i32

// With debug info:
%0 = arith.addi %arg0, %arg1 : i32 loc("file.mlir":10:5)
```

### `printGenericOpForm` Flag

**Purpose**: Prints operations in their generic form instead of using custom assembly format

**Effect**: Shows the underlying structure of every operation uniformly, without dialect-specific syntactic sugar

**Use case**:
- Seeing the "raw" operation structure
- Debugging custom assembly formats
- Understanding how ODS generates operation structure

**Example output difference**:
```mlir
// Normal (custom assembly format):
%0 = arith.addi %arg0, %arg1 : i32
func.func @foo(%arg0: i32) -> i32 {
  return %arg0 : i32
}

// Generic form:
%0 = "arith.addi"(%arg0, %arg1) : (i32, i32) -> i32
"func.func"() ({
  ^bb0(%arg0: i32):
    "func.return"(%arg0) : (i32) -> ()
}) {sym_name = "foo", function_type = (i32) -> i32} : () -> ()
```

---

## Complete Call Trace

### Layer 1: Python API (`mlir.passmanager.PassManager`)

**File**: [mlir/lib/Bindings/Python/Pass.cpp:85-117](mlir/lib/Bindings/Python/Pass.cpp#L85-L117)

```cpp
.def(
    "enable_ir_printing",
    [](PyPassManager &passManager, bool printBeforeAll,
       bool printAfterAll, bool printModuleScope, bool printAfterChange,
       bool printAfterFailure, std::optional<int64_t> largeElementsLimit,
       std::optional<int64_t> largeResourceLimit, bool enableDebugInfo,
       bool printGenericOpForm,
       std::optional<std::string> optionalTreePrintingPath) {
      // Step 1: Create OpPrintingFlags object
      MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();

      // Step 2: Configure large element/resource elision
      if (largeElementsLimit) {
        mlirOpPrintingFlagsElideLargeElementsAttrs(flags, *largeElementsLimit);
        mlirOpPrintingFlagsElideLargeResourceString(flags, *largeElementsLimit);
      }
      if (largeResourceLimit)
        mlirOpPrintingFlagsElideLargeResourceString(flags, *largeResourceLimit);

      // Step 3: Set enableDebugInfo flag
      if (enableDebugInfo)
        mlirOpPrintingFlagsEnableDebugInfo(flags, /*enable=*/true,
                                           /*prettyForm=*/false);

      // Step 4: Set printGenericOpForm flag
      if (printGenericOpForm)
        mlirOpPrintingFlagsPrintGenericOpForm(flags);

      // Step 5: Prepare tree printing path
      std::string treePrintingPath = "";
      if (optionalTreePrintingPath.has_value())
        treePrintingPath = optionalTreePrintingPath.value();

      // Step 6: Call C API to enable IR printing
      mlirPassManagerEnableIRPrinting(
          passManager.get(), printBeforeAll, printAfterAll,
          printModuleScope, printAfterChange, printAfterFailure, flags,
          mlirStringRefCreate(treePrintingPath.data(),
                              treePrintingPath.size()));

      // Step 7: Destroy flags (config copied by C++ side)
      mlirOpPrintingFlagsDestroy(flags);
    },
    "print_before_all"_a = false, "print_after_all"_a = true,
    "print_module_scope"_a = false, "print_after_change"_a = false,
    "print_after_failure"_a = false,
    "large_elements_limit"_a = nb::none(),
    "large_resource_limit"_a = nb::none(), "enable_debug_info"_a = false,
    "print_generic_op_form"_a = false,
    "tree_printing_dir_path"_a = nb::none(),
    "Configure printing of intermediate IR")
```

**Key Points**:
- Python parameters are converted to C API calls
- `OpPrintingFlags` object is created, configured, passed, then destroyed
- The C++ side makes a copy of the flags for later use

---

### Layer 2: C API Flag Configuration

#### C API Headers

**File**: [mlir/include/mlir-c/IR.h:537-546](mlir/include/mlir-c/IR.h#L537-L546)

```c
/// Enable or disable printing of debug information (based on `enable`). If
/// 'prettyForm' is set to true, debug information is printed in a more readable
/// 'pretty' form. Note: The IR generated with 'prettyForm' is not parsable.
MLIR_CAPI_EXPORTED void
mlirOpPrintingFlagsEnableDebugInfo(MlirOpPrintingFlags flags, bool enable,
                                   bool prettyForm);

/// Always print operations in the generic form.
MLIR_CAPI_EXPORTED void
mlirOpPrintingFlagsPrintGenericOpForm(MlirOpPrintingFlags flags);
```

#### C API Implementation

**File**: [mlir/lib/CAPI/IR/IR.cpp:220-227](mlir/lib/CAPI/IR/IR.cpp#L220-L227)

```cpp
// enableDebugInfo implementation
void mlirOpPrintingFlagsEnableDebugInfo(MlirOpPrintingFlags flags, bool enable,
                                        bool prettyForm) {
  // unwrap() converts MlirOpPrintingFlags (C handle) to OpPrintingFlags* (C++ object)
  unwrap(flags)->enableDebugInfo(enable, /*prettyForm=*/prettyForm);
}

// printGenericOpForm implementation
void mlirOpPrintingFlagsPrintGenericOpForm(MlirOpPrintingFlags flags) {
  // unwrap() converts MlirOpPrintingFlags (C handle) to OpPrintingFlags* (C++ object)
  unwrap(flags)->printGenericOpForm();
}
```

**Key Points**:
- Simple wrapper functions that unwrap C API handles to C++ objects
- Calls methods on the actual `OpPrintingFlags` C++ class

---

### Layer 3: C++ OpPrintingFlags Class

#### Class Declaration

**File**: [mlir/include/mlir/IR/OperationSupport.h:1176-1256](mlir/include/mlir/IR/OperationSupport.h#L1176-L1256)

```cpp
class OpPrintingFlags {
public:
  OpPrintingFlags();

  /// Enable or disable printing of debug information (based on `enable`). If
  /// 'prettyForm' is set to true, debug information is printed in a more
  /// readable 'pretty' form. Note: The IR generated with 'prettyForm' is not
  /// parsable.
  OpPrintingFlags &enableDebugInfo(bool enable = true, bool prettyForm = false);

  /// Always print operations in the generic form.
  OpPrintingFlags &printGenericOpForm(bool enable = true);

  /// Return if debug information should be printed.
  bool shouldPrintDebugInfo() const;

  /// Return if debug information should be printed in the pretty form.
  bool shouldPrintDebugInfoPrettyForm() const;

  /// Return if operations should be printed in the generic form.
  bool shouldPrintGenericOpForm() const;

  // ... other methods and private members
};
```

#### Class Implementation

**File**: [mlir/lib/IR/AsmPrinter.cpp:266-279](mlir/lib/IR/AsmPrinter.cpp#L266-L279)

```cpp
/// Enable printing of debug information. If 'prettyForm' is set to true,
/// debug information is printed in a more readable 'pretty' form.
OpPrintingFlags &OpPrintingFlags::enableDebugInfo(bool enable,
                                                   bool prettyForm) {
  // Set member flags
  printDebugInfoFlag = enable;
  printDebugInfoPrettyFormFlag = prettyForm;
  return *this;
}

/// Always print operations in the generic form.
OpPrintingFlags &OpPrintingFlags::printGenericOpForm(bool enable) {
  // Set member flag
  printGenericOpFormFlag = enable;
  return *this;
}
```

**Implementation of Query Methods** ([mlir/lib/IR/AsmPrinter.cpp:350-362](mlir/lib/IR/AsmPrinter.cpp#L350-L362)):

```cpp
/// Return if debug information should be printed.
bool OpPrintingFlags::shouldPrintDebugInfo() const {
  return printDebugInfoFlag;
}

/// Return if operations should be printed in the generic form.
bool OpPrintingFlags::shouldPrintGenericOpForm() const {
  return printGenericOpFormFlag;
}
```

**Key Points**:
- Flags are stored as simple boolean member variables
- Setter methods return `*this` for method chaining (builder pattern)
- Query methods provide read-only access to flag state

---

### Layer 4: PassManager Integration

#### C API PassManager

**File**: [mlir/lib/CAPI/IR/Pass.cpp:47-72](mlir/lib/CAPI/IR/Pass.cpp#L47-L72)

```cpp
void mlirPassManagerEnableIRPrinting(MlirPassManager passManager,
                                     bool printBeforeAll, bool printAfterAll,
                                     bool printModuleScope,
                                     bool printAfterOnlyOnChange,
                                     bool printAfterOnlyOnFailure,
                                     MlirOpPrintingFlags flags,
                                     MlirStringRef treePrintingPath) {
  // Create lambda predicates for when to print
  auto shouldPrintBeforePass = [printBeforeAll](Pass *, Operation *) {
    return printBeforeAll;
  };
  auto shouldPrintAfterPass = [printAfterAll](Pass *, Operation *) {
    return printAfterAll;
  };

  // Branch based on output destination
  if (unwrap(treePrintingPath).empty())
    // Print to stderr
    return unwrap(passManager)
        ->enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                           printModuleScope, printAfterOnlyOnChange,
                           printAfterOnlyOnFailure, /*out=*/llvm::errs(),
                           *unwrap(flags));  // <-- Flags passed here

  // Print to file tree
  unwrap(passManager)
      ->enableIRPrintingToFileTree(shouldPrintBeforePass, shouldPrintAfterPass,
                                   printModuleScope, printAfterOnlyOnChange,
                                   printAfterOnlyOnFailure,
                                   unwrap(treePrintingPath), *unwrap(flags));
}
```

#### C++ PassManager Methods

**File**: [mlir/lib/Pass/IRPrinting.cpp:364-387](mlir/lib/Pass/IRPrinting.cpp#L364-L387)

```cpp
/// Add an instrumentation to print the IR before and after pass execution.
void PassManager::enableIRPrinting(
    std::function<bool(Pass *, Operation *)> shouldPrintBeforePass,
    std::function<bool(Pass *, Operation *)> shouldPrintAfterPass,
    bool printModuleScope, bool printAfterOnlyOnChange,
    bool printAfterOnlyOnFailure, raw_ostream &out,
    OpPrintingFlags opPrintingFlags) {
  // Create config object that stores the OpPrintingFlags
  enableIRPrinting(std::make_unique<BasicIRPrinterConfig>(
      std::move(shouldPrintBeforePass), std::move(shouldPrintAfterPass),
      printModuleScope, printAfterOnlyOnChange, printAfterOnlyOnFailure,
      opPrintingFlags, out));  // <-- Flags stored in config
}

/// Add an instrumentation to print the IR before and after pass execution.
void PassManager::enableIRPrintingToFileTree(
    std::function<bool(Pass *, Operation *)> shouldPrintBeforePass,
    std::function<bool(Pass *, Operation *)> shouldPrintAfterPass,
    bool printModuleScope, bool printAfterOnlyOnChange,
    bool printAfterOnlyOnFailure, StringRef printTreeDir,
    OpPrintingFlags opPrintingFlags) {
  // Create file-tree config object that stores the OpPrintingFlags
  enableIRPrinting(std::make_unique<FileTreeIRPrinterConfig>(
      std::move(shouldPrintBeforePass), std::move(shouldPrintAfterPass),
      printModuleScope, printAfterOnlyOnChange, printAfterOnlyOnFailure,
      opPrintingFlags, printTreeDir));  // <-- Flags stored in config
}
```

**Key Points**:
- PassManager stores the `OpPrintingFlags` in an `IRPrinterConfig` object
- The config object is wrapped in an `IRPrinterInstrumentation`
- The instrumentation gets called before/after each pass

---

### Layer 5: IR Printing During Pass Execution

#### IRPrinterInstrumentation Class

**File**: [mlir/lib/Pass/IRPrinting.cpp:28-46](mlir/lib/Pass/IRPrinting.cpp#L28-L46)

```cpp
class IRPrinterInstrumentation : public PassInstrumentation {
public:
  IRPrinterInstrumentation(std::unique_ptr<PassManager::IRPrinterConfig> config)
      : config(std::move(config)) {}

private:
  /// Instrumentation hooks called by PassManager
  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;
  void runAfterPassFailed(Pass *pass, Operation *op) override;

  /// Configuration that stores the OpPrintingFlags
  std::unique_ptr<PassManager::IRPrinterConfig> config;

  /// Fingerprints for change detection
  DenseMap<Pass *, OperationFingerPrint> beforePassFingerPrints;
};
```

#### Core Printing Function

**File**: [mlir/lib/Pass/IRPrinting.cpp:49-68](mlir/lib/Pass/IRPrinting.cpp#L49-L68)

```cpp
static void printIR(Operation *op, bool printModuleScope, raw_ostream &out,
                    OpPrintingFlags flags) {
  // If not printing at module scope, print just this operation
  if (!printModuleScope)
    return op->print(out << " //----- //\n",
                     op->getBlock() ? flags.useLocalScope() : flags);
                     //               ^^^^^ Flags used here

  // If printing at module scope, print the entire top-level module
  out << " ('" << op->getName() << "' operation";
  if (auto symbolName =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    out << ": @" << symbolName.getValue();
  out << ") //----- //\n";

  // Find the top-level operation
  auto *topLevelOp = op;
  while (auto *parentOp = topLevelOp->getParentOp())
    topLevelOp = parentOp;
  topLevelOp->print(out, flags);  // <-- Flags passed to print
}
```

#### Before-Pass Hook

**File**: [mlir/lib/Pass/IRPrinting.cpp:71-85](mlir/lib/Pass/IRPrinting.cpp#L71-L85)

```cpp
void IRPrinterInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  if (isa<OpToOpPassAdaptor>(pass))
    return;

  // Record fingerprint for change detection
  if (config->shouldPrintAfterOnlyOnChange())
    beforePassFingerPrints.try_emplace(pass, op);

  // Print IR before pass
  config->printBeforeIfEnabled(pass, op, [&](raw_ostream &out) {
    out << "// -----// IR Dump Before " << pass->getName() << " ("
        << pass->getArgument() << ")";
    printIR(op, config->shouldPrintAtModuleScope(), out,
            config->getOpPrintingFlags());  // <-- Flags retrieved from config
    out << "\n\n";
  });
}
```

#### After-Pass Hook

**File**: [mlir/lib/Pass/IRPrinting.cpp:87-116](mlir/lib/Pass/IRPrinting.cpp#L87-L116)

```cpp
void IRPrinterInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  if (isa<OpToOpPassAdaptor>(pass))
    return;

  // Don't print if only printing on failure
  if (config->shouldPrintAfterOnlyOnFailure())
    return;

  // Check for changes if requested
  if (config->shouldPrintAfterOnlyOnChange()) {
    auto fingerPrintIt = beforePassFingerPrints.find(pass);
    assert(fingerPrintIt != beforePassFingerPrints.end());
    // Skip printing if IR unchanged
    if (fingerPrintIt->second == OperationFingerPrint(op)) {
      beforePassFingerPrints.erase(fingerPrintIt);
      return;
    }
    beforePassFingerPrints.erase(fingerPrintIt);
  }

  // Print IR after pass
  config->printAfterIfEnabled(pass, op, [&](raw_ostream &out) {
    out << "// -----// IR Dump After " << pass->getName() << " ("
        << pass->getArgument() << ")";
    printIR(op, config->shouldPrintAtModuleScope(), out,
            config->getOpPrintingFlags());  // <-- Flags retrieved from config
    out << "\n\n";
  });
}
```

**Key Points**:
- PassManager calls instrumentation hooks before/after each pass
- Hooks retrieve `OpPrintingFlags` from stored config
- Flags are passed to `Operation::print()` method

---

### Layer 6: Operation Printing with Flags

#### Where Flags Are Checked

**File**: [mlir/lib/IR/AsmPrinter.cpp](mlir/lib/IR/AsmPrinter.cpp)

**1. Debug Info Printing** ([line 2119-2121](mlir/lib/IR/AsmPrinter.cpp#L2119-L2121)):

```cpp
void AsmPrinter::Impl::printTrailingLocation(Location loc, bool allowAlias) {
  // Check to see if we are printing debug information.
  if (!printerFlags.shouldPrintDebugInfo())
    return;  // <-- Early return if debug info disabled

  os << " ";
  // ... print location information
}
```

**2. Generic vs Custom Form** ([line 703-712](mlir/lib/IR/AsmPrinter.cpp#L703-L712)):

```cpp
void OperationInitializer::printCustomOrGenericOp(Operation *op) override {
  // Visit the operation location (for debug info)
  if (printerFlags.shouldPrintDebugInfo())
    initializer.visit(op->getLoc(), /*canBeDeferred=*/true);

  // If requested, always print the generic form.
  if (!printerFlags.shouldPrintGenericOpForm()) {
    // Use custom assembly format defined by the operation
    op->getName().printAssembly(op, *this, /*defaultDialect=*/"");
    return;
  }

  // Otherwise print generic form
  printGenericOp(op, /*printOpName=*/true);
}
```

**3. Another Check for Generic Form** ([line 3688-3693](mlir/lib/IR/AsmPrinter.cpp#L3688-L3693)):

```cpp
void OperationPrinter::printCustomOrGenericOp(Operation *op) {
  // If requested, always print the generic form.
  if (!printerFlags.shouldPrintGenericOpForm()) {
    // Check to see if this is a known operation. If so, use the registered
    // custom printer hook.
    if (auto opInfo = op->getRegisteredInfo()) {
      // Call operation's custom print method
      opInfo->printAssembly(op, *this, defaultDialectStack.back());
      return;
    }
  }

  // Print generic form (either requested or unknown operation)
  printGenericOp(op, /*printOpName=*/true);
}
```

**4. Block Argument Locations** ([line 751-754](mlir/lib/IR/AsmPrinter.cpp#L751-L754)):

```cpp
void printBlockArgumentsAndNames(...) {
  // ...
  printType(arg.getType());

  // Visit the argument location.
  if (printerFlags.shouldPrintDebugInfo())
    // TODO: Allow deferring argument locations.
    initializer.visit(arg.getLoc(), /*canBeDeferred=*/false);
}
```

**5. Custom Naming Disabled in Generic Form** ([line 1610](mlir/lib/IR/AsmPrinter.cpp#L1610)):

```cpp
void AsmState::Impl::initializeRegionState(...) {
  // ...

  if (!printerFlags.shouldPrintGenericOpForm()) {
    // Only use custom block argument names in normal mode
    if (Operation *op = region.getParentOp()) {
      if (auto asmInterface = dyn_cast<OpAsmOpInterface>(op))
        asmInterface.getAsmBlockArgumentNames(region, setBlockArgNameFn);
    }
  }
  // In generic form, use default %arg0, %arg1, ... naming
}
```

---

## How Flags Change PassManager Behavior

### enableDebugInfo = true

**Effect on PassManager IR Dumps**:

1. **Location Information Appended**: Every operation, block argument, and region gets location info appended
2. **Format**: `loc("filename":line:column)` or custom location formats
3. **Overhead**: Increases output size significantly
4. **Parseability**: Still parsable (unless prettyForm=true)

**Example PassManager Output**:

```mlir
// -----// IR Dump Before Canonicalize (canonicalize)
module {
  func.func @example(%arg0: i32) -> i32 loc("test.mlir":10:1) {
    %0 = arith.constant 5 : i32 loc("test.mlir":11:5)
    %1 = arith.addi %arg0, %0 : i32 loc("test.mlir":12:5)
    return %1 : i32 loc("test.mlir":13:5)
  } loc("test.mlir":10:1)
} loc(unknown)

// -----// IR Dump After Canonicalize (canonicalize)
module {
  func.func @example(%arg0: i32) -> i32 loc("test.mlir":10:1) {
    %0 = arith.constant 5 : i32 loc("test.mlir":11:5)
    %1 = arith.addi %arg0, %0 : i32 loc("test.mlir":12:5)
    return %1 : i32 loc("test.mlir":13:5)
  } loc("test.mlir":10:1)
} loc(unknown)
```

### printGenericOpForm = true

**Effect on PassManager IR Dumps**:

1. **All Operations in Generic Form**: No dialect-specific syntax
2. **Uniform Structure**: `"dialect.opname"(...) {...} {attrs}`
3. **Explicit Regions**: All regions shown with explicit `^bb` labels
4. **Default Naming**: No custom SSA value names (always %0, %1, ...)
5. **Verbose**: Much larger output

**Example PassManager Output**:

```mlir
// -----// IR Dump Before Canonicalize (canonicalize)
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 5 : i32} : () -> i32
    %1 = "arith.addi"(%arg0, %0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {sym_name = "example", function_type = (i32) -> i32} : () -> ()
}) : () -> ()

// -----// IR Dump After Canonicalize (canonicalize)
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 5 : i32} : () -> i32
    %1 = "arith.addi"(%arg0, %0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {sym_name = "example", function_type = (i32) -> i32} : () -> ()
}) : () -> ()
```

### Both Flags Combined

**Example Output**:

```mlir
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32 loc("test.mlir":10:20)):
    %0 = "arith.constant"() {value = 5 : i32} : () -> i32 loc("test.mlir":11:5)
    %1 = "arith.addi"(%arg0, %0) : (i32, i32) -> i32 loc("test.mlir":12:5)
    "func.return"(%1) : (i32) -> () loc("test.mlir":13:5)
  }) {sym_name = "example", function_type = (i32) -> i32} : () -> () loc("test.mlir":10:1)
}) : () -> () loc(unknown)
```

---

## Summary Diagram

```
Python: pm.enable_ir_printing(enable_debug_info=True, print_generic_op_form=True)
    |
    v
Nanobind: Create MlirOpPrintingFlags, configure, pass to C API
    |
    v
C API: mlirOpPrintingFlagsEnableDebugInfo(flags, true, false)
       mlirOpPrintingFlagsPrintGenericOpForm(flags)
       mlirPassManagerEnableIRPrinting(..., flags, ...)
    |
    v
C++ OpPrintingFlags: Set printDebugInfoFlag = true
                     Set printGenericOpFormFlag = true
    |
    v
C++ PassManager: Store OpPrintingFlags in IRPrinterConfig
                 Wrap in IRPrinterInstrumentation
    |
    v
During Pass Execution:
    |
    +-> runBeforePass() -> printIR(op, ..., config->getOpPrintingFlags())
    |                          |
    |                          v
    |                      op->print(out, flags)
    |                          |
    |                          v
    |                      AsmPrinter checks flags:
    |                        - shouldPrintDebugInfo() -> append locations
    |                        - shouldPrintGenericOpForm() -> use generic syntax
    |
    +-> Pass executes and modifies IR
    |
    +-> runAfterPass() -> printIR(op, ..., config->getOpPrintingFlags())
                              |
                              v
                          op->print(out, flags)
                              |
                              v
                          AsmPrinter checks flags again
```

---

## Usage Example

```python
from mlir import ir
from mlir.passmanager import PassManager

# Create context and module
with ir.Context() as ctx, ir.Location.unknown():
    module = ir.Module.parse("""
        func.func @example(%arg0: i32) -> i32 {
            %c5 = arith.constant 5 : i32
            %0 = arith.addi %arg0, %c5 : i32
            return %0 : i32
        }
    """)

    # Create pass manager with pipeline
    pm = PassManager.parse("builtin.module(func.func(canonicalize))")

    # Enable IR printing with flags
    pm.enable_ir_printing(
        print_before_all=True,
        print_after_all=True,
        print_module_scope=True,
        enable_debug_info=True,        # <-- Show source locations
        print_generic_op_form=True,    # <-- Use generic syntax
        tree_printing_dir_path="./ir_dumps"  # Optional: dump to directory
    )

    # Run passes - IR will be dumped before/after each pass
    pm.run(module.operation)
```

---

## Key Takeaways

1. **OpPrintingFlags is a configuration object** passed through all layers of the stack
2. **Flags are stored in PassManager's IRPrinterConfig** and retrieved during each print
3. **enableDebugInfo adds location information** to every printed element
4. **printGenericOpForm forces uniform syntax** without dialect-specific formatting
5. **The actual printing logic is in AsmPrinter** which queries flags via `shouldPrint*()` methods
6. **PassInstrumentation hooks** are called before/after each pass to trigger printing
7. **Flags don't change pass behavior** - they only affect how IR is printed to output
