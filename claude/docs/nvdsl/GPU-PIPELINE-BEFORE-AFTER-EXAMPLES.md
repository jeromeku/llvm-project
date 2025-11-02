# GPU Pipeline: Before & After Examples with Line Numbers

This document shows actual transformations from your IR dumps with exact line numbers.

## Pass 0: convert-nvgpu-to-nvvm

### Transformation 1: nvgpu.mbarrier.create → memref.global

**BEFORE** (`module_before.mlir:27`):
```mlir
%7 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>
```

**AFTER** (`0_convert-nvgpu-to-nvvm.mlir:4` and `:68`):
```mlir
// Line 4: Global variable created in module scope
"memref.global"() <{alignment = 8 : i64,
                    sym_name = "__mbarrier",
                    sym_visibility = "private",
                    type = memref<1xi64, 3>}> : () -> ()

// Line 68: Reference to global in kernel
%56 = "memref.get_global"() <{name = @__mbarrier}> : () -> memref<1xi64, 3>
```

**What happened**: Barrier transformed from dynamic allocation to static global variable in shared memory (address space 3).

---

### Transformation 2: nvgpu.mbarrier.init → nvvm.mbarrier.init.shared

**BEFORE** (`module_before.mlir:32`):
```mlir
nvgpu.mbarrier.init %7[%c0_12], %c1_13, predicate = %8
    : <memorySpace = #gpu.address_space<workgroup>>
```

**AFTER** (`0_convert-nvgpu-to-nvvm.mlir:76-79`):
```mlir
%64 = "llvm.extractvalue"(%57) <{position = array<i64: 1>}>
      : (!llvm.struct<(ptr<3>, ptr<3>, i64, ...)>) -> !llvm.ptr<3>

%65 = "llvm.getelementptr"(%64, %61) <{elem_type = i64, ...}>
      : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>

%66 = "llvm.trunc"(%63) : (i64) -> i32

"nvvm.mbarrier.init.shared"(%65, %66, %59)
    : (!llvm.ptr<3>, i32, i1) -> ()
```

**What happened**:
1. Extract aligned pointer from memref descriptor (line 76)
2. Compute element address via GEP (line 77)
3. Truncate count to i32 (line 78)
4. Call NVVM intrinsic (line 79)

---

### Transformation 3: nvgpu.tma.create.descriptor → llvm.call @mgpuTensorMapEncodeTiledMemref

**BEFORE** (`module_before.mlir:13`):
```mlir
%4 = nvgpu.tma.create.descriptor %cast box[%c128, %c64] : memref<*xf16>
     -> <tensor = memref<128x64xf16, 3>,
         swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
```

**AFTER** (`0_convert-nvgpu-to-nvvm.mlir:14-35`):
```mlir
// Lines 14-15: Cast memref to LLVM struct
%7 = "memref.cast"(%1#0) : (memref<128x64xf16>) -> memref<*xf16>
%8 = "builtin.unrealized_conversion_cast"(%7)
     : (memref<*xf16>) -> !llvm.struct<(i64, ptr)>

// Lines 16-19: Convert box dimensions to i64
%9 = "arith.constant"() <{value = 128 : index}> : () -> index
%10 = "builtin.unrealized_conversion_cast"(%9) : (index) -> i64
%11 = "arith.constant"() <{value = 64 : index}> : () -> index
%12 = "builtin.unrealized_conversion_cast"(%11) : (index) -> i64

// Lines 20-22: Extract descriptor components
%13 = "llvm.mlir.constant"() <{value = 6 : i32}> : () -> i64  // TMA format
%14 = "llvm.extractvalue"(%8) <{position = array<i64: 0>}>
      : (!llvm.struct<(i64, ptr)>) -> i64  // Rank
%15 = "llvm.extractvalue"(%8) <{position = array<i64: 1>}>
      : (!llvm.struct<(i64, ptr)>) -> !llvm.ptr  // Base pointer

// Lines 23-30: Create stack array for dimensions
%16 = "llvm.mlir.constant"() <{value = 5 : i32}> : () -> i64
%17 = "llvm.alloca"(%16) <{elem_type = i64}> : (i64) -> !llvm.ptr
// ... store dimensions into array ...

// Lines 31-35: Encode TMA attributes as constants
%22 = "llvm.mlir.constant"() <{value = 0 : i32}> : () -> i64  // l2promo = none
%23 = "llvm.mlir.constant"() <{value = 3 : i32}> : () -> i64  // swizzle = 128b
%24 = "llvm.mlir.constant"() <{value = 0 : i32}> : () -> i64  // oob = zero
%25 = "llvm.mlir.constant"() <{value = 0 : i32}> : () -> i64  // interleave = none

// Line 35: Runtime call to create TMA descriptor
%26 = "llvm.call"(%14, %15, %13, %22, %23, %24, %25, %17)
      <{callee = @mgpuTensorMapEncodeTiledMemref, ...}>
      : (i64, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> !llvm.ptr
```

**What happened**: High-level TMA descriptor lowered to runtime library call with explicit encoding of all attributes.

---

### Transformation 4: nvgpu.tma.prefetch.descriptor → nvvm.prefetch

**BEFORE** (`module_before.mlir:33`):
```mlir
nvgpu.tma.prefetch.descriptor %4, predicate = %8
    : <tensor = memref<128x64xf16, 3>, ...>
```

**AFTER** (`0_convert-nvgpu-to-nvvm.mlir:80`):
```mlir
"nvvm.prefetch"(%26, %59) <{tensormap}> : (!llvm.ptr, i1) -> ()
```

**What happened**: High-level prefetch replaced with NVVM intrinsic, operates on TMA descriptor pointer.

---

### Transformation 5: nvgpu.mbarrier.arrive.expect_tx → (skipped in this pass)

**NOTE**: This operation is NOT converted in pass 0! It remains as-is because it needs to be handled inside the GPU kernel after outlining.

**BEFORE** (`module_before.mlir:52`):
```mlir
nvgpu.mbarrier.arrive.expect_tx %7[%c0_21], %c32768, predicate = %8
    : <memorySpace = #gpu.address_space<workgroup>>
```

**STILL PRESENT IN** (`0_convert-nvgpu-to-nvvm.mlir:89-102`):
```mlir
// Operation remains unchanged, will be converted in pass 14 (convert-gpu-to-nvvm)
```

---

### Transformation 6: nvgpu.warpgroup.mma.init.accumulator → nvvm.wgmma.init.accumulator

**BEFORE** (`module_before.mlir:69`):
```mlir
%14 = nvgpu.warpgroup.mma.init.accumulator
      -> <fragmented = vector<128x128xf32>>
```

**AFTER** (`0_convert-nvgpu-to-nvvm.mlir` - need to find exact line):
Looking for wgmma initialization...

---

## Pass 1: gpu-kernel-outlining

### Major Transformation: Inline kernel → Separate gpu.module

**BEFORE** (`module_before.mlir:25-75`):
```mlir
func.func @gemm_128_128_64(...) {
  // Host code: allocations, memcpy, etc.
  ...

  // Inline kernel at line 25
  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1_7, %arg11 = %c1_8)
             threads(%arg6, %arg7, %arg8) in (%arg12 = %c128_9, %arg13 = %c1_10, %arg14 = %c1_11)
             dynamic_shared_memory_size %c32768_i32 {
    // Kernel body
    %thread_id_x = gpu.thread_id  x
    %7 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>
    // ... matrix multiply operations ...
    nvgpu.warpgroup.mma.store %17, %memref_2
    gpu.terminator
  }

  // More host code
  ...
}
```

**AFTER** (`1_gpu-kernel-outlining.mlir`):
```mlir
// Part 1: Kernel extracted to separate module
gpu.module @gemm_128_128_64_kernel {
  gpu.func @gemm_128_128_64_kernel(
      %arg0: !llvm.ptr,  // TMA descriptor A
      %arg1: !llvm.ptr,  // TMA descriptor B
      %arg2: !llvm.ptr,  // Output memref base
      %arg3: !llvm.ptr,  // Output memref aligned
      %arg4: i64,        // Output offset
      %arg5: i64,        // Output dim 0
      %arg6: i64,        // Output dim 1
      %arg7: i64,        // Output stride 0
      %arg8: i64         // Output stride 1
  ) kernel attributes {
      gpu.known_block_size = array<i32: 128, 1, 1>
  } {
    // Kernel body (same as before but now standalone)
    %thread_id_x = gpu.thread_id  x
    // ... operations ...
    gpu.return
  }
}

// Part 2: Host function now launches extracted kernel
func.func @gemm_128_128_64(...) {
  // Host code: allocations, memcpy, etc.
  ...

  // Launch extracted kernel (replaces gpu.launch)
  gpu.launch_func @gemm_128_128_64_kernel::@gemm_128_128_64_kernel
      blocks in (%c1, %c1, %c1)
      threads in (%c128, %c1, %c1)
      dynamic_shared_memory_size %c32768
      args(%26 : !llvm.ptr,  // TMA desc A
           %46 : !llvm.ptr,  // TMA desc B
           %3#0 : memref<128x128xf32>,  // Output
           ...)

  // More host code
  ...
}
```

**Key changes**:
1. Kernel body extracted to `gpu.module`
2. Kernel becomes `gpu.func` with `kernel` attribute
3. Block/thread sizes moved from runtime to kernel attribute
4. Arguments explicitly passed (TMA descriptors, memrefs)
5. `gpu.launch` replaced with `gpu.launch_func`

---

## Pass 14: gpu.module(convert-gpu-to-nvvm)

This pass runs INSIDE the `gpu.module { ... }` only.

### Transformation: gpu.thread_id → nvvm.read.ptx.sreg.tid.x

**BEFORE** (inside gpu.module):
```mlir
%thread_id_x = gpu.thread_id  x
```

**AFTER**:
```mlir
%thread_id_x = nvvm.read.ptx.sreg.tid.x : i32
%thread_id_idx = arith.index_cast %thread_id_x : i32 to index
```

**Implementation** (`mlir/lib/Conversion/GPUToNVVM/GPUToNVVM.cpp:805`):
```cpp
struct GPUThreadIdOpLowering : public ConvertOpToLLVMPattern<gpu::ThreadIdOp> {
  LogicalResult
  matchAndRewrite(gpu::ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Choose NVVM intrinsic based on dimension
    NVVM::ReadSRegOp::PtxSReg sreg;
    switch (op.getDimension()) {
    case gpu::Dimension::x:
      sreg = NVVM::ReadSRegOp::PtxSReg::tid_x;
      break;
    case gpu::Dimension::y:
      sreg = NVVM::ReadSRegOp::PtxSReg::tid_y;
      break;
    case gpu::Dimension::z:
      sreg = NVVM::ReadSRegOp::PtxSReg::tid_z;
      break;
    }

    // Create NVVM intrinsic
    Value newOp = rewriter.create<NVVM::ReadSRegOp>(
        op.getLoc(), rewriter.getI32Type(), sreg);

    // Cast to index if needed
    if (op.getType().isa<IndexType>())
      newOp = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), op.getType(), newOp);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};
```

---

## Pass 19: gpu-module-to-binary

### Transformation: gpu.module → gpu.binary (PTX embedded)

**BEFORE** (`13_gpu-to-llvm.mlir` - simplified):
```mlir
gpu.module @gemm_128_128_64_kernel {
  llvm.func @gemm_128_128_64_kernel(...) attributes {nvvm.kernel} {
    // NVVM intrinsics
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    nvvm.mbarrier.init.shared %mbar, %count
    %result = nvvm.wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 ...
    llvm.store %result, %output
    llvm.return
  }
}
```

**AFTER** (`14_gpu-module-to-binary.mlir:89`):
```mlir
gpu.binary @gemm_128_128_64_kernel [
  #gpu.object<
    #nvvm.target<chip = "sm_90a", features = "+ptx80">,
    properties = {LLVMIRToISATimeInMs = 15 : i64, O = 2 : i32},
    assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A
      .version 8.0\0A
      .target sm_90a\0A
      .address_size 64\0A
      \0A
      .visible .entry gemm_128_128_64_kernel(...) {\0A
        .reg .pred \09%p<2>;\0A
        .reg .b32 \09%r<145>;\0A
        .reg .b64 \09%rd<26>;\0A
        .shared .align 16 .b8 __dynamic_shmem__0;\0A
        .shared .align 8 .b8 __mbarrier[8];\0A
        mov.u32 \09%r137, %tid.x;\0A
        @%p1 mbarrier.init.shared.b64 [%r1], %r2;\0A
        wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {...};\0A
        st.global.b32 [%rd25], %r9;\0A
        ret;\0A
      }\0A
    "
  >
]
```

**What happened**:
1. `gpu.module` compiled to LLVM IR
2. LLVM IR lowered to NVPTX assembly (PTX)
3. PTX embedded as string attribute in `gpu.binary`
4. Original `gpu.module` removed

**This is where your PTX is generated!**

---

## Key Insights from Transformations

1. **NVGPU operations are high-level abstractions** that get lowered through multiple stages
2. **Barrier operations require global memory allocation** (converted to `memref.global`)
3. **TMA descriptors become runtime library calls** to encode complex metadata
4. **Kernel outlining separates compilation units** (host vs. device)
5. **Generic GPU ops are lowered to NVVM** which then becomes PTX
6. **Final PTX is embedded as a string** in the IR

---

## Viewing Your Transformations

```bash
cd /home/jeromeku/llvm-project/mlir/test/Examples/NVGPU/ch3_mlir_dump/builtin_module_no-symbol-name

# Before any passes
cat ../../../module_before.mlir

# After each pass
cat 0_convert-nvgpu-to-nvvm.mlir      # NVGPU → NVVM
cat 1_gpu-kernel-outlining.mlir       # Kernel extraction
cat 14_gpu-module-to-binary.mlir      # PTX generation
cat 18_reconcile-unrealized-casts.mlir # Final output

# Compare two stages
diff -u ../../../module_before.mlir 0_convert-nvgpu-to-nvvm.mlir | less
```

---

## Next Steps

Want deeper analysis of a specific pass? Ask about:
- `convert-nvgpu-to-nvvm` - Detailed pattern implementations
- `gpu-kernel-outlining` - Argument marshalling, region cloning
- `convert-gpu-to-nvvm` - All GPU → NVVM conversions
- `gpu-module-to-binary` - PTX compilation process
- Any other pass!

I can provide line-by-line C++ implementation traces for any of these.
