# Compilation Pipeline: MLIR IR → Executable GPU Code

## Introduction

This document traces the complete compilation flow from MLIR IR to executable CUDA binary.

---

## Pipeline Overview

```
Python User Code
    ↓ [@NVDSL.mlir_func decorator]
MLIR IR (GPU Dialect)
    ↓ [NvgpuCompiler.compile()]
MLIR IR (NVVM Dialect)
    ↓ [ExecutionEngine + LLVM]
LLVM IR
    ↓ [LLVM Optimizer]
PTX Assembly
    ↓ [NVIDIA ptxas]
CUDA Binary (.cubin)
    ↓ [GPU Runtime]
Executable GPU Code
```

---

## Stage 1: IR Construction (Python)

**File**: [nvdsl.py:411-428](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L411-L428)

```python
with ir.Context(), ir.Location.unknown():
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        fop = func.FuncOp(function_name, (types, []))
        with ir.InsertionPoint(fop.add_entry_block()):
            result = funcBody(*fargs, **kwargs)  # Builds IR!
            func.ReturnOp([])
```

**Output**: MLIR Module with GPU dialect operations

**Example IR** (simplified):
```mlir
module {
  func.func @gemm_128_128_64(%arg0: memref<128x64xf16>,
                              %arg1: memref<64x128xf16>,
                              %arg2: memref<128x128xf32>) {
    %token = gpu.async.token
    %a_dev, %t1 = gpu.alloc async [%token] () : memref<128x64xf16>
    %b_dev, %t2 = gpu.alloc async [%t1] () : memref<64x128xf16>
    %d_dev, %t3 = gpu.alloc async [%t2] () : memref<128x128xf32>

    %t4 = gpu.memcpy async [%t3] %a_dev, %arg0 : memref<128x64xf16>, memref<128x64xf16>
    %t5 = gpu.memcpy async [%t4] %b_dev, %arg1 : memref<64x128xf16>, memref<64x128xf16>

    %tma_desc_a = nvgpu.tma.create.descriptor %a_dev box[%c128, %c64] : ...
    %tma_desc_b = nvgpu.tma.create.descriptor %b_dev box[%c64, %c64] : ...

    gpu.launch blocks(%bx, %by, %bz) in (%c1, %c1, %c1)
               threads(%tx, %ty, %tz) in (%c128, %c1, %c1)
               dynamic_shared_memory_size %c16384 {
      %tidx = gpu.thread_id x : index
      %mbar_group = nvgpu.mbarrier.create : !nvgpu.mbarrier.group<...>

      %smem_a = gpu.dynamic_shared_memory : memref<?xi8, 3>
      %a_view = memref.view %smem_a[%c0][] : memref<?xi8, 3> to memref<128x64xf16, 3>

      nvgpu.tma.async.load %a_view, %mbar_group[%c0], %tma_desc_a
        coordinates [%c0, %c0] : ...

      nvgpu.mbarrier.try_wait.parity %mbar_group[%c0], %false, %c10000000

      %desc_a = nvgpu.warpgroup.generate.descriptor %a_view, %tma_desc_a : ...
      %desc_b = nvgpu.warpgroup.generate.descriptor %b_view, %tma_desc_b : ...

      %acc_init = nvgpu.warpgroup.mma.init.accumulator : !nvgpu.warpgroup.accumulator<...>
      %acc = nvgpu.warpgroup.mma %desc_a, %desc_b, %acc_init {transposeB = true} : ...

      nvgpu.warpgroup.mma.store %acc, %d_smem : ...

      gpu.terminator
    }

    %t6 = gpu.memcpy async [%t5] %arg2, %d_dev : memref<128x128xf32>, memref<128x128xf32>
    gpu.wait [%t6]
    return
  }
}
```

---

## Stage 2: Pass Pipeline Execution

**File**: [nvgpucompiler.py:32-34](../../../mlir/test/Examples/NVGPU/tools/nvgpucompiler.py#L32-L34)

```python
def compile(self, module: ir.Module):
    """Compiles the module by invoking the nvgpu pipeline."""
    passmanager.PassManager.parse(self.pipeline).run(module.operation)
```

**Pipeline String**: [nvgpucompiler.py:23](../../../mlir/test/Examples/NVGPU/tools/nvgpucompiler.py#L23)
```python
pipeline = f"builtin.module(gpu-lower-to-nvvm-pipeline{{{options}}})"
```

**Full pipeline**: `builtin.module(gpu-lower-to-nvvm-pipeline{cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3})`

### The gpu-lower-to-nvvm-pipeline

**Location**: [mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp](../../../mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp)

**What it does**: Transforms GPU dialect → NVVM dialect → LLVM dialect

**Passes included** (partial list):

1. **convert-nvgpu-to-nvvm**: Converts NVGPU ops to NVVM intrinsics
   - `nvgpu.tma.async.load` → `nvvm.cp.async.bulk.tensor.shared.cluster.global`
   - `nvgpu.warpgroup.mma` → `nvvm.wgmma.mma_async`
   - `nvgpu.mbarrier.*` → `nvvm.mbarrier.*`

2. **gpu-kernel-outlining**: Extracts kernels into separate modules

3. **convert-gpu-to-nvvm**: Converts GPU ops to NVVM
   - `gpu.thread_id` → `nvvm.read.ptx.sreg.tid.x`
   - `gpu.block_id` → `nvvm.read.ptx.sreg.ctaid.x`
   - `gpu.barrier` → `nvvm.barrier0`

4. **gpu-to-llvm**: Converts GPU host operations to LLVM
   - `gpu.alloc` → runtime calls
   - `gpu.memcpy` → runtime calls

5. **convert-nvvm-to-llvm**: Lowers NVVM to LLVM+inline asm

6. **gpu.module(convert-gpu-to-nvvm)**: Lowers GPU module ops

7. **gpu-to-cubin**: Assembles NVVM/LLVM to CUDA binary
   - Translates to LLVM IR
   - Compiles to PTX
   - Assembles to .cubin

**Example transformation**:

**Before** (GPU/NVGPU dialect):
```mlir
%tidx = gpu.thread_id x : index
nvgpu.warpgroup.mma %desc_a, %desc_b, %acc_init : ...
```

**After** (NVVM dialect):
```mlir
%tidx = nvvm.read.ptx.sreg.tid.x : i32
%result = nvvm.wgmma.mma_async.m64n128k16.f32.f16.f16
            %desc_a, %desc_b, %acc_init {transposeB} : ...
```

**After** (LLVM+PTX inline asm):
```mlir
%tidx = llvm.inline_asm "mov.u32 $0, %tid.x;", "=r" : () -> i32
%result = llvm.inline_asm
  "wgmma.mma_async.m64n128k16.f32.f16.f16 {...}",
  "..." : (...) -> !llvm.struct<...>
```

---

## Stage 3: JIT Compilation & Execution

**File**: [nvgpucompiler.py:36-44](../../../mlir/test/Examples/NVGPU/tools/nvgpucompiler.py#L36-L44)

```python
def jit(self, module: ir.Module) -> execution_engine.ExecutionEngine:
    """Wraps the module in a JIT execution engine."""
    return execution_engine.ExecutionEngine(
        module, opt_level=self.opt_level, shared_libs=self.shared_libs
    )

def compile_and_jit(self, module: ir.Module) -> execution_engine.ExecutionEngine:
    """Compiles and jits the module."""
    self.compile(module)
    return self.jit(module)
```

**Usage**: [nvdsl.py:443-450](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L443-L450)
```python
compiler = nvgpucompiler.NvgpuCompiler(
    options, opt_level=3, shared_libs=[support_lib]
)
engine = compiler.compile_and_jit(module)
engine.initialize()  # Run static constructors
```

### ExecutionEngine

**What it does**:
1. Converts MLIR → LLVM IR
2. Runs LLVM optimization passes (opt-level=3)
3. JIT compiles to machine code
4. Loads GPU binaries
5. Provides function invocation interface

**Call Path**:
```
execution_engine.ExecutionEngine(module)
    ↓
Python: mlir.execution_engine.ExecutionEngine(...)
    ↓
C++: ExecutionEngine::create(module, options)
    ↓
LLVM JIT: LLVMOrcCreateLLJIT()
    ↓
Compiles LLVM IR to machine code
```

### Function Invocation

**File**: [nvdsl.py:453-456](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py#L453-L456)

```python
newArgs = get_mlir_func_obj_ty(args)  # Convert NumPy → memref descriptors
engine.invoke(function_name, *newArgs)
```

**What happens**:
```
engine.invoke("gemm_128_128_64", memref_desc_a, memref_desc_b, memref_desc_c)
    ↓
C++: ExecutionEngine::invoke(fn_name, args...)
    ↓
Look up JIT-compiled function pointer
    ↓
Cast to function signature:
  void (*)(MemRefDescriptor*, MemRefDescriptor*, MemRefDescriptor*)
    ↓
Call function pointer with arguments
    ↓
Compiled host code executes:
  - Calls cuMemAlloc() via gpu runtime
  - Calls cuMemcpy() for transfers
  - Calls cuLaunchKernel() for GPU kernel
  - GPU kernel executes!
  - Copies results back
```

---

## Stage 4: GPU Runtime Integration

**Shared Library**: `$SUPPORT_LIB` environment variable

**Location**: Usually `libmlir_cuda_runtime.so` or `libmlir_runner_utils.so`

**What it provides**:
- CUDA API wrappers
- GPU memory management
- Kernel launching
- Module loading

**Functions called** (examples):
```c
void* mgpuMemAlloc(uint64_t size);
void mgpuMemcpy(void* dst, void* src, uint64_t size);
void mgpuLaunchKernel(void* module, void* function,
                      dim3 grid, dim3 block,
                      void** args, size_t smem);
```

**Mapping** GPU operations:

| MLIR Operation | Runtime Call | CUDA API |
|----------------|--------------|----------|
| `gpu.alloc` | `mgpuMemAlloc()` | `cuMemAlloc()` |
| `gpu.memcpy` | `mgpuMemcpy()` | `cuMemcpy()` |
| `gpu.launch` | `mgpuLaunchKernel()` | `cuLaunchKernel()` |
| `gpu.wait` | `mgpuStreamSynchronize()` | `cuStreamSynchronize()` |

---

## Stage 5: PTX & CUBIN Generation

### PTX Generation

When NVVM dialect is lowered, it generates PTX (Parallel Thread Execution) assembly:

**Example PTX** (conceptual):
```ptx
.version 8.0
.target sm_90a
.address_size 64

.visible .entry gemm_kernel(
    .param .u64 param_0,  // Matrix A
    .param .u64 param_1,  // Matrix B
    .param .u64 param_2   // Matrix D
) {
    .reg .pred %p<10>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<10>;
    .reg .f16 %h<256>;
    .reg .f32 %f<512>;

    .shared .align 128 .b8 shared_mem[16384];

    // Get thread ID
    mov.u32 %r0, %tid.x;

    // TMA load
    cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
        [shared_mem], [%rd0, {%r1, %r2, %r3}], [%rd1];

    // Barrier wait
    mbarrier.try_wait.parity.shared.b64 [%rd2], %r3, 0x989680;

    // WGMMA operation
    wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
        {%f0, %f1, ..., %f127},  // Accumulator (128 registers)
        %rd3,                     // Descriptor A
        %rd4,                     // Descriptor B
        1, 1, 0, 1, 0;           // Modifiers

    // Wait for WGMMA
    wgmma.wait_group.sync.aligned 0;

    // Store results
    st.shared.v4.f32 [shared_mem + %r5], {%f0, %f1, %f2, %f3};

    ret;
}
```

### CUBIN Assembly

PTX is assembled to CUBIN (CUDA Binary):

**Command** (conceptual):
```bash
ptxas --gpu-name=sm_90a --opt-level=3 kernel.ptx -o kernel.cubin
```

**CUBIN format**:
- ELF binary format
- Contains machine code for specific GPU architecture
- Includes symbol tables, relocation info
- Can be disassembled with `cuobjdump`

---

## Complete Data Flow

### NumPy Array → GPU Memory

```
Python:
a = np.random.randn(128, 64).astype(np.float16)
    ↓
Memory layout:
  - Contiguous buffer: float16[128][64]
  - Total size: 128 * 64 * 2 = 16384 bytes
  - Base pointer: 0x7fff12340000
    ↓
NVDSL: get_mlir_ty(a)
    ↓
MLIR: memref<128x64xf16>
    ↓
Runtime: rt.get_ranked_memref_descriptor(a)
    ↓
MemRefDescriptor {
  allocated: 0x7fff12340000
  aligned: 0x7fff12340000
  offset: 0
  sizes: [128, 64]
  strides: [64, 1]
}
    ↓
JIT Call: engine.invoke("kernel", &descriptor)
    ↓
GPU Runtime: mgpuMemAlloc(16384)
    ↓
CUDA: cuMemAlloc(&dev_ptr, 16384)
    ↓
GPU Memory: dev_ptr = 0xdeadbeef0000
    ↓
GPU Runtime: mgpuMemcpy(dev_ptr, host_ptr, 16384)
    ↓
CUDA: cuMemcpy(dev_ptr, 0x7fff12340000, 16384, H2D)
    ↓
GPU Global Memory: [128][64] f16 data
    ↓
TMA Load: Transfers to shared memory
    ↓
Shared Memory: [128][64] f16 data at smem[0]
    ↓
WGMMA: Reads from shared memory
    ↓
Accumulator Registers: Distributed across 128 threads
    ↓
Store: Registers → Shared Memory → Global Memory
    ↓
GPU Runtime: mgpuMemcpy(host_ptr, dev_ptr, 65536)
    ↓
NumPy Array: d = results written back
```

---

## Performance Considerations

### Optimization Levels

**opt-level=3** enables:
- Loop unrolling
- Instruction scheduling
- Register allocation optimization
- Dead code elimination
- Constant propagation

### GPU-Specific Optimizations

1. **Shared Memory Swizzling**: `SWIZZLE_128B`
   - Reduces bank conflicts
   - Improves memory throughput

2. **L2 Cache Promotion**: `L2PROMO_*`
   - Hints for cache residency
   - Reduces global memory latency

3. **Register Allocation**: `nvvm.setmaxregister`
   - Controls occupancy
   - Balances resource usage

4. **Asynchronous Operations**:
   - TMA: Overlaps copy with compute
   - WGMMA: Async matrix multiply
   - Barriers: Precise synchronization

---

## Debugging & Inspection

### View Generated IR

```python
# In nvdsl.py, enable IR saving:
with open("nvdsl.mlir", "w") as f:
    print(module, file=f)
```

### View PTX

```bash
# After compilation
cuobjdump --dump-ptx kernel.cubin
```

### View SASS (machine code)

```bash
cuobjdump --dump-sass kernel.cubin
```

### Profile Execution

```bash
nsys profile python Ch3.py
ncu --set full python Ch3.py
```

---

## Summary

**Complete Pipeline**:

```
User Python Code
    ↓ [NVDSL decorators + MLIR bindings]
MLIR IR (GPU/NVGPU dialects)
    ↓ [gpu-lower-to-nvvm-pipeline]
MLIR IR (NVVM dialect)
    ↓ [LLVM translation]
LLVM IR
    ↓ [LLVM opt]
Optimized LLVM IR
    ↓ [NVPTX backend]
PTX Assembly
    ↓ [ptxas]
CUDA Binary (.cubin)
    ↓ [cuModuleLoad]
GPU Code in Memory
    ↓ [cuLaunchKernel]
Executing on GPU!
```

**Key Files**:
- [nvdsl.py](../../../mlir/test/Examples/NVGPU/tools/nvdsl.py): DSL + compilation orchestration
- [nvgpucompiler.py](../../../mlir/test/Examples/NVGPU/tools/nvgpucompiler.py): Pass pipeline wrapper
- [GPUToNVVMPipeline.cpp](../../../mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp): Pass pipeline implementation
- [NVGPUToNVVM.cpp](../../../mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp): NVGPU→NVVM conversion

**Next**: [README.md](README.md) - Documentation index
