# PTX Extraction Quickstart Guide

## TL;DR - Extract PTX Now

```bash
# Simple version (no dependencies except Python 3)
python claude/docs/nvdsl/extract_ptx.py mlir/test/Examples/NVGPU/module_after.mlir --print

# Save to file
python claude/docs/nvdsl/extract_ptx.py module_after.mlir -o ptx_output/

# MLIR Python bindings version (more robust)
python claude/docs/nvdsl/extract_ptx_mlir.py module_after.mlir -o ptx_output/ -v
```

---

## What's the PTX?

The PTX (Parallel Thread Execution) is NVIDIA's intermediate assembly language for GPUs. After your MLIR code goes through the `gpu-lower-to-nvvm-pipeline`, the final PTX is embedded in your MLIR module as a string attribute in `gpu.binary` operations.

**Location in your MLIR**:
```mlir
gpu.binary @gemm_128_128_64_kernel [
  #gpu.object<
    #nvvm.target<chip = "sm_90a", features = "+ptx80">,
    properties = {LLVMIRToISATimeInMs = 15 : i64, O = 2 : i32},
    assembly = "...PTX CODE HERE..."  <-- This is what we extract!
  >
]
```

---

## Available Tools

### 1. Simple Regex-Based Extractor (`extract_ptx.py`)

**Pros**:
- ✅ No dependencies (just Python 3 + regex)
- ✅ Fast
- ✅ Works on any MLIR file

**Cons**:
- ⚠️ Uses regex parsing (less robust)

**Usage**:
```bash
# Print to stdout
python claude/docs/nvdsl/extract_ptx.py module_after.mlir --print

# Save to directory
python claude/docs/nvdsl/extract_ptx.py module_after.mlir -o ./ptx_kernels/

# Extract specific kernel
python claude/docs/nvdsl/extract_ptx.py module_after.mlir --kernel my_kernel --print

# Verbose mode
python claude/docs/nvdsl/extract_ptx.py module_after.mlir -o ./ptx/ -v
```

### 2. MLIR Python Bindings Extractor (`extract_ptx_mlir.py`)

**Pros**:
- ✅ Uses official MLIR Python bindings
- ✅ Proper MLIR parsing
- ✅ More robust to format changes

**Cons**:
- ⚠️ Requires MLIR Python bindings in PYTHONPATH

**Setup**:
```bash
# Set PYTHONPATH to MLIR Python bindings
export PYTHONPATH=/home/jeromeku/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH

# Or if using build directory directly:
export PYTHONPATH=/home/jeromeku/llvm-project/build/python:$PYTHONPATH
```

**Usage**:
```bash
# Same interface as extract_ptx.py
python claude/docs/nvdsl/extract_ptx_mlir.py module_after.mlir --print
python claude/docs/nvdsl/extract_ptx_mlir.py module_after.mlir -o ./ptx/ -v
```

---

## Integration with Your NVDSL Pipeline

### Option 1: Add to Your Compilation Script

```python
from mlir import ir
from mlir.passmanager import PassManager
import subprocess

# Your existing NVDSL compilation
mlir_module = gemm_128_128_64(a, b, d)

with ir.Context():
    compiler = NVDSL(shared_libs=[SUPPORT_LIB])

    # Save module before compilation
    with open("module_before.mlir", "w") as f:
        mlir_module.module.operation.print(file=f)

    # Compile
    compiler.compile_module(mlir_module.module)

    # Save compiled module
    with open("module_after.mlir", "w") as f:
        mlir_module.module.operation.print(file=f)

    # Extract PTX
    subprocess.run([
        "python", "claude/docs/nvdsl/extract_ptx.py",
        "module_after.mlir",
        "-o", "./ptx_output/"
    ])
```

### Option 2: Direct Extraction in Python

```python
import re

def extract_ptx_from_module(module: ir.Module) -> dict:
    """Extract PTX directly from MLIR module object."""
    module_str = str(module)

    # Find gpu.binary operations
    pattern = r'gpu\.binary\s+@(\w+)\s+\[.*?assembly\s*=\s*"([^"]*(?:\\.[^"]*)*)"'
    matches = re.findall(pattern, module_str, re.DOTALL)

    ptx_kernels = {}
    for kernel_name, ptx_escaped in matches:
        ptx = ptx_escaped.replace('\\0A', '\n').replace('\\09', '\t')
        ptx_kernels[kernel_name] = ptx

    return ptx_kernels

# Usage in your pipeline:
compiler.compile_module(mlir_module.module)

# Extract PTX
ptx_map = extract_ptx_from_module(mlir_module.module)

for kernel_name, ptx in ptx_map.items():
    with open(f"{kernel_name}.ptx", "w") as f:
        f.write(ptx)
    print(f"Saved PTX for {kernel_name}")
```

---

## What Can You Do With PTX?

### 1. Inspect Generated Code

```bash
# View the PTX
python claude/docs/nvdsl/extract_ptx.py module_after.mlir --print | less
```

**Look for**:
- Register usage (`.reg .b32 %r<145>` means 145 registers!)
- Shared memory usage (`.shared .align 16 .b8 __dynamic_shmem__0`)
- Inline assembly instructions (`wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16`)
- Thread configuration (`.maxntid 128, 1, 1`)

### 2. Compile PTX to SASS (Native Assembly)

```bash
# Compile PTX to CUBIN
ptxas -arch=sm_90a -o kernel.cubin gemm_128_128_64_kernel.ptx

# Disassemble to SASS
cuobjdump -sass kernel.cubin
```

### 3. Profile with NVIDIA Tools

```bash
# Compile to executable
nvcc -arch=sm_90a kernel.ptx driver.cu -o program

# Profile
ncu --set full ./program
nsys profile ./program
```

### 4. Analyze Resource Usage

```bash
# Get resource information
ptxas -arch=sm_90a --resource-usage gemm_128_128_64_kernel.ptx
```

**Example output**:
```
ptxinfo : Compiling entry function 'gemm_128_128_64_kernel' for 'sm_90a'
ptxinfo : Used 145 registers, 32768 bytes smem, 576 bytes cmem[0]
```

---

## Understanding Your PTX Output

Your `gemm_128_128_64_kernel.ptx` contains:

```ptx
.version 8.0              // PTX ISA version
.target sm_90a            // Target GPU architecture
.address_size 64          // 64-bit addressing

.visible .entry gemm_128_128_64_kernel(...)  // Kernel entry point
.maxntid 128, 1, 1        // Max threads per block

{
  .reg .pred  %p<2>;      // Predicate registers
  .reg .b32   %r<145>;    // 145 general-purpose registers
  .reg .b64   %rd<26>;    // 26 64-bit registers

  .shared .align 16 .b8 __dynamic_shmem__0;  // Shared memory
  .shared .align 8 .b8 __mbarrier[8];        // Barrier memory

  // ... actual instructions ...
  wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {...};  // WGMMA instruction
  // ... stores, loads, etc ...
}
```

**Key metrics for your kernel**:
- **Registers**: 145 (this is high! May limit occupancy)
- **Shared memory**: 32768 bytes (32KB - matches your dynamic allocation)
- **Thread block**: 128 threads (1D layout)
- **WGMMA instructions**: Using tensor core operations

---

## Troubleshooting

### "No PTX found in module"

**Cause**: The module hasn't been through the compilation pipeline yet.

**Solution**: Make sure you run `gpu-lower-to-nvvm-pipeline` first:
```python
pm = PassManager.parse("builtin.module(gpu-lower-to-nvvm-pipeline{...})")
pm.run(module.operation)
```

### "MLIR Python bindings not found"

**Cause**: PYTHONPATH doesn't include MLIR bindings.

**Solution**: Use the simple `extract_ptx.py` script (no MLIR dependencies), or set PYTHONPATH:
```bash
export PYTHONPATH=/path/to/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH
```

### "Regex doesn't match"

**Cause**: MLIR format might have changed or be malformed.

**Solution**: Check your MLIR file manually:
```bash
grep -A 5 "gpu.binary" module_after.mlir
```

You should see the `assembly = "..."` attribute.

---

## Advanced: C++ PTX Extractor

For building MLIR tools, see the full C++ implementation in [mlir-module-inspection-and-ptx-extraction.md](mlir-module-inspection-and-ptx-extraction.md).

**Build**:
```cmake
# CMakeLists.txt
add_executable(ptx-extractor ptx_extractor.cpp)
target_link_libraries(ptx-extractor PRIVATE
  MLIRParser MLIRIR MLIRGPUDialect MLIRNVVMDialect MLIRSupport
)
```

**Usage**:
```bash
./bin/ptx-extractor module_after.mlir -o ./ptx_output/
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Extract and print | `python extract_ptx.py module.mlir --print` |
| Save to directory | `python extract_ptx.py module.mlir -o ptx/` |
| Extract one kernel | `python extract_ptx.py module.mlir --kernel foo --print` |
| Verbose output | `python extract_ptx.py module.mlir -o ptx/ -v` |
| View PTX details | `ptxas --resource-usage kernel.ptx` |
| Compile to SASS | `ptxas -arch=sm_90a -o kernel.cubin kernel.ptx` |
| Disassemble | `cuobjdump -sass kernel.cubin` |

---

## Summary

✅ **Two extraction tools provided**: Simple regex version and MLIR bindings version

✅ **Both work on your `module_after.mlir`**: Extracts `gemm_128_128_64_kernel.ptx`

✅ **Easy integration**: Can be called from command line or integrated into your Python pipeline

✅ **Full documentation**: See [mlir-module-inspection-and-ptx-extraction.md](mlir-module-inspection-and-ptx-extraction.md) for complete API reference

🚀 **Start now**: `python claude/docs/nvdsl/extract_ptx.py module_after.mlir --print`
