# MLIR Module Inspection and PTX Extraction Guide

## Overview

This guide covers how to inspect, examine, and extract information from MLIR modules using both C++ APIs and Python bindings. The specific use case focuses on extracting PTX assembly from `gpu.binary` operations in NVGPU-compiled modules.

---

## Part 1: Available APIs for Module Inspection

### Python Bindings Overview

The MLIR Python bindings provide several ways to inspect and manipulate modules:

**Core APIs**:
- `ir.Module` - Top-level module container
- `ir.Operation` - Individual operations in the IR
- `ir.Block` - Basic blocks containing operations
- `ir.Region` - Regions containing blocks
- `ir.Attribute` - Operation attributes (where PTX is stored!)
- `ir.Type` - Type system

**Navigation Pattern**:
```
Module → Operation → Regions → Blocks → Operations → Attributes
```

---

## Part 2: Minimal Examples of Module Inspection

### Example 1: Basic Module Loading and Iteration (Python)

```python
from mlir import ir

# Load a module from file
with ir.Context() as ctx:
    with open("module.mlir", "r") as f:
        module = ir.Module.parse(f.read())

    # Get the top-level module operation
    module_op = module.operation

    # Print module
    print(module_op)

    # Iterate through all operations in the module
    for op in module.body.operations:
        print(f"Operation: {op.name}")
        print(f"  Attributes: {op.attributes}")
```

**Output**:
```
Operation: llvm.mlir.global
  Attributes: {...}
Operation: llvm.func
  Attributes: {...}
Operation: gpu.binary
  Attributes: {'objects': ...}
```

### Example 2: Walking Operations Recursively

```python
def walk_operations(op, indent=0):
    """Recursively walk all operations in the IR."""
    print("  " * indent + f"Op: {op.name}")

    # Walk through regions
    for region in op.regions:
        # Walk through blocks
        for block in region.blocks:
            # Walk through operations
            for child_op in block.operations:
                walk_operations(child_op, indent + 1)

# Usage:
with ir.Context() as ctx:
    module = ir.Module.parse(mlir_text)
    walk_operations(module.operation)
```

### Example 3: Finding Specific Operations

```python
def find_operations_by_name(op, target_name):
    """Find all operations with a specific name."""
    results = []

    if op.name == target_name:
        results.append(op)

    for region in op.regions:
        for block in region.blocks:
            for child_op in block.operations:
                results.extend(find_operations_by_name(child_op, target_name))

    return results

# Usage: Find all gpu.binary operations
with ir.Context() as ctx:
    module = ir.Module.parse(mlir_text)
    gpu_binaries = find_operations_by_name(module.operation, "gpu.binary")

    for binary in gpu_binaries:
        print(f"Found: {binary}")
```

### Example 4: Accessing Operation Attributes

```python
with ir.Context() as ctx:
    module = ir.Module.parse(mlir_text)

    for op in module.body.operations:
        if op.name == "gpu.binary":
            # Access attributes dictionary
            attrs = op.attributes

            # Get specific attribute
            if "sym_name" in attrs:
                print(f"Symbol name: {attrs['sym_name']}")

            # Iterate all attributes
            for key, value in attrs.items():
                print(f"  {key} = {value}")
```

### Example 5: Extracting String Attributes

```python
from mlir import ir

def extract_string_attr(attr):
    """Extract string value from StringAttr."""
    if isinstance(attr, ir.StringAttr):
        return str(attr).strip('"')  # Remove quotes
    return None

with ir.Context() as ctx:
    module = ir.Module.parse(mlir_text)

    for op in module.body.operations:
        if "sym_name" in op.attributes:
            name = extract_string_attr(op.attributes["sym_name"])
            print(f"Operation {op.name} has name: {name}")
```

### Example 6: Modifying Operations

```python
from mlir import ir

with ir.Context() as ctx:
    module = ir.Module.parse(mlir_text)

    # Add a new attribute to an operation
    for op in module.body.operations:
        if op.name == "llvm.func":
            # Create a new string attribute
            new_attr = ir.StringAttr.get("my_marker")
            op.attributes["custom_tag"] = new_attr

    # Print modified module
    print(module)
```

---

## Part 3: PTX Extraction - Python Approach

### Understanding the gpu.binary Structure

The PTX is stored in a `gpu.binary` operation with this structure:

```mlir
gpu.binary @kernel_name [
  #gpu.object<
    #nvvm.target<chip = "sm_90a", features = "+ptx80">,
    properties = {...},
    assembly = "...PTX CODE HERE..."  // <-- This is what we want!
  >
]
```

The PTX is in the `assembly` field within the `gpu.object` attribute.

### Python Solution: Extract PTX

```python
#!/usr/bin/env python3
"""Extract PTX assembly from MLIR gpu.binary operations."""

from mlir import ir
import sys
import re


def extract_ptx_from_module(module_text):
    """
    Extract PTX assembly from gpu.binary operations in MLIR module.

    Args:
        module_text: String containing MLIR module text

    Returns:
        dict: Mapping of kernel names to PTX assembly strings
    """
    ptx_kernels = {}

    with ir.Context() as ctx:
        # Register GPU dialect
        ctx.load_all_available_dialects()

        # Parse module
        module = ir.Module.parse(module_text)

        # Find all gpu.binary operations
        for op in module.body.operations:
            if op.name == "gpu.binary":
                # Get the symbol name
                sym_name = None
                if "sym_name" in op.attributes:
                    sym_name_attr = op.attributes["sym_name"]
                    sym_name = str(sym_name_attr).strip('"')

                # Get the binary objects attribute
                # The objects are in an ArrayAttr
                if "objects" in op.attributes:
                    objects_attr = op.attributes["objects"]

                    # The PTX is embedded in the attribute string representation
                    # We need to parse it out
                    attr_str = str(objects_attr)

                    # Extract assembly field using regex
                    # Pattern: assembly = "..."
                    match = re.search(r'assembly = "([^"]*(?:\\.[^"]*)*)"', attr_str)
                    if match:
                        ptx_escaped = match.group(1)
                        # Unescape the PTX
                        ptx = ptx_escaped.replace('\\0A', '\n')
                        ptx = ptx.replace('\\09', '\t')
                        ptx = ptx.replace('\\"', '"')

                        ptx_kernels[sym_name or "unknown"] = ptx

    return ptx_kernels


def extract_ptx_from_file(mlir_file):
    """Extract PTX from MLIR file."""
    with open(mlir_file, 'r') as f:
        module_text = f.read()

    return extract_ptx_from_module(module_text)


# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_ptx.py <mlir_file> [output_dir]")
        sys.exit(1)

    mlir_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    # Extract PTX
    ptx_kernels = extract_ptx_from_file(mlir_file)

    # Save each kernel to a file
    for kernel_name, ptx in ptx_kernels.items():
        output_file = f"{output_dir}/{kernel_name}.ptx"
        with open(output_file, 'w') as f:
            f.write(ptx)
        print(f"Extracted PTX for '{kernel_name}' to {output_file}")
        print(f"  Lines: {len(ptx.splitlines())}")
```

**Running the script**:
```bash
python extract_ptx.py module_after.mlir ./ptx_output/
# Output:
# Extracted PTX for 'gemm_128_128_64_kernel' to ./ptx_output/gemm_128_128_64_kernel.ptx
#   Lines: 237
```

### Alternative Python Approach: Direct Regex

If you don't need the full MLIR parsing, you can use simple regex:

```python
#!/usr/bin/env python3
"""Simple regex-based PTX extractor."""

import re
import sys


def extract_ptx_simple(mlir_file):
    """Extract PTX using regex without parsing MLIR."""
    with open(mlir_file, 'r') as f:
        content = f.read()

    # Find gpu.binary operations
    # Pattern: gpu.binary @name [...assembly = "..."]
    pattern = r'gpu\.binary\s+@(\w+)\s+\[.*?assembly = "([^"]*(?:\\.[^"]*)*)"'

    matches = re.findall(pattern, content, re.DOTALL)

    ptx_kernels = {}
    for kernel_name, ptx_escaped in matches:
        # Unescape
        ptx = ptx_escaped.replace('\\0A', '\n')
        ptx = ptx.replace('\\09', '\t')
        ptx = ptx.replace('\\"', '"')
        ptx_kernels[kernel_name] = ptx

    return ptx_kernels


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_ptx_simple.py <mlir_file>")
        sys.exit(1)

    ptx_kernels = extract_ptx_simple(sys.argv[1])

    for name, ptx in ptx_kernels.items():
        print(f"\n{'='*70}")
        print(f"Kernel: {name}")
        print('='*70)
        print(ptx)
```

### Integrating with Your NVDSL Pipeline

Add PTX extraction to your compilation pipeline:

```python
from mlir import ir
from mlir.passmanager import PassManager
import re


def compile_and_extract_ptx(module: ir.Module):
    """Compile MLIR module and extract PTX."""

    # Run compilation pipeline
    with module.context:
        pipeline = "builtin.module(gpu-lower-to-nvvm-pipeline{...})"
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)

    # Extract PTX from compiled module
    module_str = str(module)

    # Find gpu.binary
    pattern = r'assembly = "([^"]*(?:\\.[^"]*)*)"'
    matches = re.findall(pattern, module_str)

    if matches:
        ptx = matches[0]
        # Unescape
        ptx = ptx.replace('\\0A', '\n').replace('\\09', '\t')
        return ptx
    return None


# Usage in NVDSL:
mlir_module = gemm_128_128_64(a, b, d)

with ir.Context():
    compiler = NVDSL(shared_libs=[SUPPORT_LIB])

    # Compile
    compiler.compile_module(mlir_module.module)

    # Extract PTX
    ptx = compile_and_extract_ptx(mlir_module.module)

    # Save or use PTX
    with open("kernel.ptx", "w") as f:
        f.write(ptx)
```

---

## Part 4: PTX Extraction - C++ Approach

### C++ API Overview

**Key Classes**:
- `mlir::ModuleOp` - Top-level module
- `mlir::Operation*` - Generic operation
- `mlir::gpu::BinaryOp` - GPU binary operation (typed)
- `mlir::Attribute` - Attribute base class
- `mlir::StringAttr` - String attribute
- `mlir::ArrayAttr` - Array of attributes
- `mlir::gpu::ObjectAttr` - GPU object attribute

### Example 1: Basic Module Loading (C++)

```cpp
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/Support/SourceMgr.h"
#include <iostream>

int main(int argc, char **argv) {
  // Create context and load dialects
  mlir::MLIRContext context;
  context.loadDialect<mlir::gpu::GPUDialect>();

  // Parse module from file
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile(argv[1]);

  if (!file) {
    llvm::errs() << "Failed to open file\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*file), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (!module) {
    llvm::errs() << "Failed to parse module\n";
    return 1;
  }

  // Print module
  module->print(llvm::outs());

  return 0;
}
```

### Example 2: Walking Operations (C++)

```cpp
#include "mlir/IR/Visitors.h"

// Walk all operations
module->walk([&](mlir::Operation *op) {
  llvm::outs() << "Operation: " << op->getName() << "\n";

  // Access attributes
  for (auto namedAttr : op->getAttrs()) {
    llvm::outs() << "  Attr: " << namedAttr.getName() << "\n";
  }
});
```

### Example 3: PTX Extraction (C++)

```cpp
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <string>
#include <map>

std::map<std::string, std::string> extractPTX(mlir::ModuleOp module) {
  std::map<std::string, std::string> ptxKernels;

  // Walk all operations looking for gpu.binary
  module.walk([&](mlir::gpu::BinaryOp binaryOp) {
    // Get symbol name
    std::string kernelName = binaryOp.getSymName().str();

    // Get objects array
    mlir::ArrayAttr objects = binaryOp.getObjectsAttr();

    for (mlir::Attribute objAttr : objects) {
      // Cast to GPU ObjectAttr
      if (auto gpuObj = llvm::dyn_cast<mlir::gpu::ObjectAttr>(objAttr)) {
        // Check if it's NVVM/PTX
        if (auto targetAttr = llvm::dyn_cast<mlir::gpu::TargetAttr>(
                gpuObj.getTarget())) {
          // Get assembly string
          if (auto assembly = gpuObj.getAssembly()) {
            std::string ptx = assembly->str();
            ptxKernels[kernelName] = ptx;
          }
        }
      }
    }
  });

  return ptxKernels;
}

// Usage
int main(int argc, char **argv) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::gpu::GPUDialect>();

  // ... parse module ...

  auto ptxMap = extractPTX(*module);

  for (const auto &[name, ptx] : ptxMap) {
    llvm::outs() << "Kernel: " << name << "\n";
    llvm::outs() << ptx << "\n";
  }

  return 0;
}
```

### Complete C++ Example

```cpp
// ptx_extractor.cpp
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include <fstream>

using namespace mlir;

// Command line options
static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input mlir file>"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> outputDir(
    "o", llvm::cl::desc("Output directory for PTX files"),
    llvm::cl::value_desc("directory"), llvm::cl::init("."));

std::map<std::string, std::string> extractPTXFromModule(ModuleOp module) {
  std::map<std::string, std::string> ptxKernels;

  module.walk([&](gpu::BinaryOp binaryOp) {
    std::string kernelName = binaryOp.getSymName().str();
    ArrayAttr objects = binaryOp.getObjectsAttr();

    for (Attribute objAttr : objects) {
      if (auto gpuObj = llvm::dyn_cast<gpu::ObjectAttr>(objAttr)) {
        if (auto assembly = gpuObj.getAssembly()) {
          ptxKernels[kernelName] = assembly->str();
        }
      }
    }
  });

  return ptxKernels;
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR PTX Extractor\n");

  // Setup context
  MLIRContext context;
  context.loadDialect<gpu::GPUDialect>();
  context.loadDialect<NVVM::NVVMDialect>();

  // Load module
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile(inputFilename);

  if (!file) {
    llvm::errs() << "Failed to open file: " << inputFilename << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*file), llvm::SMLoc());

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &context);

  if (!module) {
    llvm::errs() << "Failed to parse MLIR module\n";
    return 1;
  }

  // Extract PTX
  auto ptxKernels = extractPTXFromModule(*module);

  // Write to files
  for (const auto &[kernelName, ptx] : ptxKernels) {
    std::string outputPath = outputDir + "/" + kernelName + ".ptx";

    std::ofstream outFile(outputPath);
    if (!outFile) {
      llvm::errs() << "Failed to open output file: " << outputPath << "\n";
      continue;
    }

    outFile << ptx;
    outFile.close();

    llvm::outs() << "Extracted PTX for '" << kernelName << "' to "
                 << outputPath << "\n";
  }

  return 0;
}
```

**CMakeLists.txt**:
```cmake
add_executable(ptx-extractor ptx_extractor.cpp)

target_link_libraries(ptx-extractor PRIVATE
  MLIRParser
  MLIRIR
  MLIRGPUDialect
  MLIRNVVMDialect
  MLIRSupport
)
```

**Build and run**:
```bash
cd llvm-project/build
cmake --build . --target ptx-extractor

./bin/ptx-extractor ../mlir/test/Examples/NVGPU/module_after.mlir -o ./ptx_output/
# Output:
# Extracted PTX for 'gemm_128_128_64_kernel' to ./ptx_output/gemm_128_128_64_kernel.ptx
```

---

## Part 5: Comparison and Recommendations

### Python Approach

**Pros**:
- Simple regex-based extraction works without full parsing
- Easy to integrate into existing Python workflows
- Fast prototyping
- No compilation required

**Cons**:
- Regex can be fragile if MLIR format changes
- String manipulation for unescaping
- Less type-safe

**Best for**:
- Quick scripts
- Integration with Python ML/GPU workflows
- Rapid prototyping

### C++ Approach

**Pros**:
- Type-safe access to GPU dialect operations
- Direct access to structured attributes
- More robust to format changes
- Can be part of MLIR tool ecosystem

**Cons**:
- Requires C++ compilation
- More boilerplate
- Longer development time

**Best for**:
- Production tools
- MLIR compiler passes
- When working in C++ codebase

---

## Part 6: Quick Reference

### Python: Extract PTX in 3 Lines

```python
import re
ptx = re.search(r'assembly = "([^"]*(?:\\.[^"]*)*)"', module_str).group(1)
ptx = ptx.replace('\\0A', '\n').replace('\\09', '\t')
print(ptx)
```

### C++: Extract PTX in ~5 Lines

```cpp
module->walk([&](gpu::BinaryOp op) {
  for (auto obj : op.getObjectsAttr()) {
    if (auto gpuObj = llvm::dyn_cast<gpu::ObjectAttr>(obj)) {
      if (auto asm = gpuObj.getAssembly()) {
        llvm::outs() << asm->str() << "\n";
      }
    }
  }
});
```

---

## Summary

**To extract PTX from your MLIR module**:

1. **Python (Recommended for quick extraction)**:
   - Use regex to find `assembly = "..."` in `gpu.binary` operations
   - Unescape `\0A` → newline, `\09` → tab
   - Save to `.ptx` file

2. **C++ (Recommended for robust tooling)**:
   - Walk module for `gpu::BinaryOp`
   - Access `getObjectsAttr()` → `gpu::ObjectAttr` → `getAssembly()`
   - Write string to file

The PTX is stored as a string attribute in the MLIR module after the `gpu-lower-to-nvvm-pipeline` pass completes. Both approaches provide direct access to this compiled GPU code.
