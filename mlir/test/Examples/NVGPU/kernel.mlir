// -----// IR Dump Before ConvertNVGPUToNVVMPass (convert-nvgpu-to-nvvm) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module {
  func.func @main(%arg0: index loc(unknown)) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index loc(#loc)
    %c1_0 = arith.constant 1 : index loc(#loc)
    %c1_1 = arith.constant 1 : index loc(#loc)
    %c4 = arith.constant 4 : index loc(#loc)
    %c1_2 = arith.constant 1 : index loc(#loc)
    %c1_3 = arith.constant 1 : index loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    gpu.launch blocks(%arg1, %arg2, %arg3) in (%arg7 = %c1, %arg8 = %c1_0, %arg9 = %c1_1) threads(%arg4, %arg5, %arg6) in (%arg10 = %c4, %arg11 = %c1_2, %arg12 = %c1_3) dynamic_shared_memory_size %c0_i32 {
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %0 = arith.addi %arg0, %thread_id_x : index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %0 : index, index loc(#loc)
      gpu.terminator loc(#loc)
    } loc(#loc)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before GpuKernelOutliningPass (gpu-kernel-outlining) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module {
  func.func @main(%arg0: index loc(unknown)) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index loc(#loc)
    %c1_0 = arith.constant 1 : index loc(#loc)
    %c1_1 = arith.constant 1 : index loc(#loc)
    %c4 = arith.constant 4 : index loc(#loc)
    %c1_2 = arith.constant 1 : index loc(#loc)
    %c1_3 = arith.constant 1 : index loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    gpu.launch blocks(%arg1, %arg2, %arg3) in (%arg7 = %c1, %arg8 = %c1_0, %arg9 = %c1_1) threads(%arg4, %arg5, %arg6) in (%arg10 = %c4, %arg11 = %c1_2, %arg12 = %c1_3) dynamic_shared_memory_size %c0_i32 {
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %0 = arith.addi %arg0, %thread_id_x : index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %0 : index, index loc(#loc)
      gpu.terminator loc(#loc)
    } loc(#loc)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before ConvertVectorToSCF (convert-vector-to-scf) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  func.func @main(%arg0: index loc(unknown)) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index loc(#loc)
    %c1_0 = arith.constant 1 : index loc(#loc)
    %c1_1 = arith.constant 1 : index loc(#loc)
    %c4 = arith.constant 4 : index loc(#loc)
    %c1_2 = arith.constant 1 : index loc(#loc)
    %c1_3 = arith.constant 1 : index loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c1, %c1_0, %c1_1) threads in (%c4, %c1_2, %c1_3)  dynamic_shared_memory_size %c0_i32 args(%arg0 : index) loc(#loc)
    return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: index loc(unknown)) kernel attributes {known_block_size = array<i32: 4, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id  x loc(#loc)
      %block_id_y = gpu.block_id  y loc(#loc)
      %block_id_z = gpu.block_id  z loc(#loc)
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %thread_id_y = gpu.thread_id  y loc(#loc)
      %thread_id_z = gpu.thread_id  z loc(#loc)
      %grid_dim_x = gpu.grid_dim  x loc(#loc)
      %grid_dim_y = gpu.grid_dim  y loc(#loc)
      %grid_dim_z = gpu.grid_dim  z loc(#loc)
      %block_dim_x = gpu.block_dim  x loc(#loc)
      %block_dim_y = gpu.block_dim  y loc(#loc)
      %block_dim_z = gpu.block_dim  z loc(#loc)
      %thread_id_x_0 = gpu.thread_id  x loc(#loc)
      %0 = arith.addi %arg0, %thread_id_x_0 : index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x_0, %0 : index, index loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before SCFToControlFlowPass (convert-scf-to-cf) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  func.func @main(%arg0: index loc(unknown)) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index loc(#loc)
    %c4 = arith.constant 4 : index loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c4, %c1, %c1)  dynamic_shared_memory_size %c0_i32 args(%arg0 : index) loc(#loc)
    return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: index loc(unknown)) kernel attributes {known_block_size = array<i32: 4, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %0 = arith.addi %arg0, %thread_id_x : index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %0 : index, index loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before ConvertNVVMToLLVMPass (convert-nvvm-to-llvm) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  func.func @main(%arg0: index loc(unknown)) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index loc(#loc)
    %c4 = arith.constant 4 : index loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c4, %c1, %c1)  dynamic_shared_memory_size %c0_i32 args(%arg0 : index) loc(#loc)
    return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: index loc(unknown)) kernel attributes {known_block_size = array<i32: 4, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %0 = arith.addi %arg0, %thread_id_x : index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %0 : index, index loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before ConvertFuncToLLVMPass (convert-func-to-llvm) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  func.func @main(%arg0: index loc(unknown)) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index loc(#loc)
    %c4 = arith.constant 4 : index loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c4, %c1, %c1)  dynamic_shared_memory_size %c0_i32 args(%arg0 : index) loc(#loc)
    return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: index loc(unknown)) kernel attributes {known_block_size = array<i32: 4, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %0 = arith.addi %arg0, %thread_id_x : index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %0 : index, index loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before ExpandStridedMetadataPass (expand-strided-metadata) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
    %c1 = arith.constant 1 : index loc(#loc)
    %c4 = arith.constant 4 : index loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c4, %c1, %c1)  dynamic_shared_memory_size %c0_i32 args(%0 : index) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: index loc(unknown)) kernel attributes {known_block_size = array<i32: 4, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %0 = arith.addi %arg0, %thread_id_x : index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %0 : index, index loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before GpuNVVMAttachTarget (nvvm-attach-target) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c4 = arith.constant 4 : index loc(#loc)
    %c1 = arith.constant 1 : index loc(#loc)
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c4, %c1, %c1)  dynamic_shared_memory_size %c0_i32 args(%0 : index) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: index loc(unknown)) kernel attributes {known_block_size = array<i32: 4, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %0 = arith.addi %arg0, %thread_id_x : index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %0 : index, index loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before LowerAffinePass (lower-affine) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c4 = arith.constant 4 : index loc(#loc)
    %c1 = arith.constant 1 : index loc(#loc)
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c4, %c1, %c1)  dynamic_shared_memory_size %c0_i32 args(%0 : index) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">] {
    gpu.func @main_kernel(%arg0: index loc(unknown)) kernel attributes {known_block_size = array<i32: 4, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %0 = arith.addi %arg0, %thread_id_x : index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %0 : index, index loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before ArithToLLVMConversionPass (convert-arith-to-llvm) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c4 = arith.constant 4 : index loc(#loc)
    %c1 = arith.constant 1 : index loc(#loc)
    %0 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c4, %c1, %c1)  dynamic_shared_memory_size %c0_i32 args(%0 : index) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">] {
    gpu.func @main_kernel(%arg0: index loc(unknown)) kernel attributes {known_block_size = array<i32: 4, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %0 = arith.addi %arg0, %thread_id_x : index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %0 : index, index loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before ConvertIndexToLLVMPass (convert-index-to-llvm) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %1 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    %2 = builtin.unrealized_conversion_cast %1 : i64 to index loc(#loc)
    %3 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %4 = builtin.unrealized_conversion_cast %3 : i64 to index loc(#loc)
    %5 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%4, %4, %4) threads in (%2, %4, %4)  dynamic_shared_memory_size %0 args(%5 : index) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">] {
    gpu.func @main_kernel(%arg0: index loc(unknown)) kernel attributes {known_block_size = array<i32: 4, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %0 = builtin.unrealized_conversion_cast %arg0 : index to i64 loc(#loc)
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %1 = builtin.unrealized_conversion_cast %thread_id_x : index to i64 loc(#loc)
      %2 = llvm.add %0, %1 : i64 loc(#loc)
      %3 = builtin.unrealized_conversion_cast %2 : i64 to index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %3 : index, index loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %1 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    %2 = builtin.unrealized_conversion_cast %1 : i64 to index loc(#loc)
    %3 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %4 = builtin.unrealized_conversion_cast %3 : i64 to index loc(#loc)
    %5 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%4, %4, %4) threads in (%2, %4, %4)  dynamic_shared_memory_size %0 args(%5 : index) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">] {
    gpu.func @main_kernel(%arg0: index loc(unknown)) kernel attributes {known_block_size = array<i32: 4, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %0 = builtin.unrealized_conversion_cast %arg0 : index to i64 loc(#loc)
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %1 = builtin.unrealized_conversion_cast %thread_id_x : index to i64 loc(#loc)
      %2 = llvm.add %0, %1 : i64 loc(#loc)
      %3 = builtin.unrealized_conversion_cast %2 : i64 to index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %3 : index, index loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before CSE (cse) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %1 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %2 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index loc(#loc)
    %4 = builtin.unrealized_conversion_cast %0 : i64 to index loc(#loc)
    %5 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%4, %4, %4) threads in (%3, %4, %4)  dynamic_shared_memory_size %1 args(%5 : index) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">] {
    gpu.func @main_kernel(%arg0: index loc(unknown)) kernel attributes {known_block_size = array<i32: 4, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %0 = builtin.unrealized_conversion_cast %arg0 : index to i64 loc(#loc)
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %1 = builtin.unrealized_conversion_cast %thread_id_x : index to i64 loc(#loc)
      %2 = llvm.add %0, %1 : i64 loc(#loc)
      %3 = builtin.unrealized_conversion_cast %2 : i64 to index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %3 : index, index loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before ConvertGpuOpsToNVVMOps (convert-gpu-to-nvvm) ('gpu.module' operation: @main_kernel) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %1 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %2 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index loc(#loc)
    %4 = builtin.unrealized_conversion_cast %0 : i64 to index loc(#loc)
    %5 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%4, %4, %4) threads in (%3, %4, %4)  dynamic_shared_memory_size %1 args(%5 : index) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">] {
    gpu.func @main_kernel(%arg0: index loc(unknown)) kernel attributes {known_block_size = array<i32: 4, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %0 = builtin.unrealized_conversion_cast %arg0 : index to i64 loc(#loc)
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %1 = builtin.unrealized_conversion_cast %thread_id_x : index to i64 loc(#loc)
      %2 = llvm.add %0, %1 : i64 loc(#loc)
      %3 = builtin.unrealized_conversion_cast %2 : i64 to index loc(#loc)
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %3 : index, index loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before Canonicalizer (canonicalize) ('gpu.module' operation: @main_kernel) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %1 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %2 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index loc(#loc)
    %4 = builtin.unrealized_conversion_cast %0 : i64 to index loc(#loc)
    %5 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%4, %4, %4) threads in (%3, %4, %4)  dynamic_shared_memory_size %1 args(%5 : index) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">] {
    llvm.mlir.global internal constant @printfFormat_0("GPU thread %llu has %llu\0A\00") {addr_space = 0 : i32} loc(#loc)
    llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32 loc(#loc)
    llvm.func @main_kernel(%arg0: i64 loc(unknown)) attributes {gpu.kernel, gpu.known_block_size = array<i32: 4, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 4, 1, 1>} {
      %0 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
      %1 = builtin.unrealized_conversion_cast %0 : index to i64 loc(#loc)
      %2 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 4> : i32 loc(#loc)
      %3 = llvm.sext %2 : i32 to i64 loc(#loc)
      %4 = builtin.unrealized_conversion_cast %3 : i64 to index loc(#loc)
      %5 = builtin.unrealized_conversion_cast %4 : index to i64 loc(#loc)
      %6 = llvm.add %1, %5 : i64 loc(#loc)
      %7 = llvm.mlir.addressof @printfFormat_0 : !llvm.ptr loc(#loc)
      %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<26 x i8> loc(#loc)
      %9 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
      %10 = llvm.alloca %9 x !llvm.struct<(i64, i64)> : (i64) -> !llvm.ptr loc(#loc)
      %11 = llvm.getelementptr %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)> loc(#loc)
      llvm.store %3, %11 : i64, !llvm.ptr loc(#loc)
      %12 = llvm.getelementptr %10[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)> loc(#loc)
      llvm.store %6, %12 : i64, !llvm.ptr loc(#loc)
      %13 = llvm.call @vprintf(%8, %10) : (!llvm.ptr, !llvm.ptr) -> i32 loc(#loc)
      llvm.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before CSE (cse) ('gpu.module' operation: @main_kernel) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %1 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %2 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index loc(#loc)
    %4 = builtin.unrealized_conversion_cast %0 : i64 to index loc(#loc)
    %5 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%4, %4, %4) threads in (%3, %4, %4)  dynamic_shared_memory_size %1 args(%5 : index) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">] {
    llvm.mlir.global internal constant @printfFormat_0("GPU thread %llu has %llu\0A\00") {addr_space = 0 : i32} loc(#loc)
    llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32 loc(#loc)
    llvm.func @main_kernel(%arg0: i64 loc(unknown)) attributes {gpu.kernel, gpu.known_block_size = array<i32: 4, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 4, 1, 1>} {
      %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
      %1 = llvm.mlir.addressof @printfFormat_0 : !llvm.ptr loc(#loc)
      %2 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 4> : i32 loc(#loc)
      %3 = llvm.sext %2 : i32 to i64 loc(#loc)
      %4 = llvm.add %arg0, %3 : i64 loc(#loc)
      %5 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<26 x i8> loc(#loc)
      %6 = llvm.alloca %0 x !llvm.struct<(i64, i64)> : (i64) -> !llvm.ptr loc(#loc)
      %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)> loc(#loc)
      llvm.store %3, %7 : i64, !llvm.ptr loc(#loc)
      %8 = llvm.getelementptr %6[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)> loc(#loc)
      llvm.store %4, %8 : i64, !llvm.ptr loc(#loc)
      %9 = llvm.call @vprintf(%5, %6) : (!llvm.ptr, !llvm.ptr) -> i32 loc(#loc)
      llvm.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before ReconcileUnrealizedCastsPass (reconcile-unrealized-casts) ('gpu.module' operation: @main_kernel) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %1 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %2 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index loc(#loc)
    %4 = builtin.unrealized_conversion_cast %0 : i64 to index loc(#loc)
    %5 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%4, %4, %4) threads in (%3, %4, %4)  dynamic_shared_memory_size %1 args(%5 : index) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">] {
    llvm.mlir.global internal constant @printfFormat_0("GPU thread %llu has %llu\0A\00") {addr_space = 0 : i32} loc(#loc)
    llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32 loc(#loc)
    llvm.func @main_kernel(%arg0: i64 loc(unknown)) attributes {gpu.kernel, gpu.known_block_size = array<i32: 4, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 4, 1, 1>} {
      %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
      %1 = llvm.mlir.addressof @printfFormat_0 : !llvm.ptr loc(#loc)
      %2 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 4> : i32 loc(#loc)
      %3 = llvm.sext %2 : i32 to i64 loc(#loc)
      %4 = llvm.add %arg0, %3 : i64 loc(#loc)
      %5 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<26 x i8> loc(#loc)
      %6 = llvm.alloca %0 x !llvm.struct<(i64, i64)> : (i64) -> !llvm.ptr loc(#loc)
      %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)> loc(#loc)
      llvm.store %3, %7 : i64, !llvm.ptr loc(#loc)
      %8 = llvm.getelementptr %6[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)> loc(#loc)
      llvm.store %4, %8 : i64, !llvm.ptr loc(#loc)
      %9 = llvm.call @vprintf(%5, %6) : (!llvm.ptr, !llvm.ptr) -> i32 loc(#loc)
      llvm.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before GpuToLLVMConversionPass (gpu-to-llvm) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %1 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %2 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index loc(#loc)
    %4 = builtin.unrealized_conversion_cast %0 : i64 to index loc(#loc)
    %5 = builtin.unrealized_conversion_cast %arg0 : i64 to index loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%4, %4, %4) threads in (%3, %4, %4)  dynamic_shared_memory_size %1 args(%5 : index) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">] {
    llvm.mlir.global internal constant @printfFormat_0("GPU thread %llu has %llu\0A\00") {addr_space = 0 : i32} loc(#loc)
    llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32 loc(#loc)
    llvm.func @main_kernel(%arg0: i64 loc(unknown)) attributes {gpu.kernel, gpu.known_block_size = array<i32: 4, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 4, 1, 1>} {
      %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
      %1 = llvm.mlir.addressof @printfFormat_0 : !llvm.ptr loc(#loc)
      %2 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 4> : i32 loc(#loc)
      %3 = llvm.sext %2 : i32 to i64 loc(#loc)
      %4 = llvm.add %arg0, %3 : i64 loc(#loc)
      %5 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<26 x i8> loc(#loc)
      %6 = llvm.alloca %0 x !llvm.struct<(i64, i64)> : (i64) -> !llvm.ptr loc(#loc)
      %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)> loc(#loc)
      llvm.store %3, %7 : i64, !llvm.ptr loc(#loc)
      %8 = llvm.getelementptr %6[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)> loc(#loc)
      llvm.store %4, %8 : i64, !llvm.ptr loc(#loc)
      %9 = llvm.call @vprintf(%5, %6) : (!llvm.ptr, !llvm.ptr) -> i32 loc(#loc)
      llvm.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before GpuModuleToBinaryPass (gpu-module-to-binary) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %1 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %2 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%0, %0, %0) threads in (%2, %0, %0) : i64 dynamic_shared_memory_size %1 args(%arg0 : i64) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">] {
    llvm.mlir.global internal constant @printfFormat_0("GPU thread %llu has %llu\0A\00") {addr_space = 0 : i32} loc(#loc)
    llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32 loc(#loc)
    llvm.func @main_kernel(%arg0: i64 loc(unknown)) attributes {gpu.kernel, gpu.known_block_size = array<i32: 4, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 4, 1, 1>} {
      %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
      %1 = llvm.mlir.addressof @printfFormat_0 : !llvm.ptr loc(#loc)
      %2 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 4> : i32 loc(#loc)
      %3 = llvm.sext %2 : i32 to i64 loc(#loc)
      %4 = llvm.add %arg0, %3 : i64 loc(#loc)
      %5 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<26 x i8> loc(#loc)
      %6 = llvm.alloca %0 x !llvm.struct<(i64, i64)> : (i64) -> !llvm.ptr loc(#loc)
      %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)> loc(#loc)
      llvm.store %3, %7 : i64, !llvm.ptr loc(#loc)
      %8 = llvm.getelementptr %6[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)> loc(#loc)
      llvm.store %4, %8 : i64, !llvm.ptr loc(#loc)
      %9 = llvm.call @vprintf(%5, %6) : (!llvm.ptr, !llvm.ptr) -> i32 loc(#loc)
      llvm.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----// IR Dump Before ConvertMathToLLVMPass (convert-math-to-llvm) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %1 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %2 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%0, %0, %0) threads in (%2, %0, %0) : i64 dynamic_shared_memory_size %1 args(%arg0 : i64) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.binary @main_kernel  [#gpu.object<#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">, properties = {ISAToBinaryTimeInMs = 18 : i64, LLVMIRToISATimeInMs = 2 : i64}, "P\EDU\BA\01\00\10\00\D8\12\00\00\00\00\00\00\02\00\01\01@\00\00\00\C8\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\07\00\01\00Z\00\00\00\00\00\00\00\00\00\00\00\11\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00\80\00\00\00\00\00\00\00\00\00\00\00x\0E\00\00\00\00\00\00\B8\0A\00\00\00\00\00\00Z\0DZ\00@\008\00\06\00@\00\0F\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.shared.reserved.0\00.nv.constant4\00\00\00\00.nv.global.init\00.rel.text.main_kernel\00.rela.text.main_kernel\00.debug_frame\00.rel.nv.constant.pic\00.rela.nv.constant4\00\00\00\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.constant0.main_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.reservedSmem.offset0\00.nv.shared.reserved.0\00__nv_reservedSMEM_offset_0_alias\00.nv.constant4\00\00\00\00.nv.global.init\00printfFormat_0\00.rel.text.main_kernel\00.rela.text.main_kernel\00.debug_frame\00.rel.nv.constant.pic\00.rela.nv.constant.pic\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00main_kernel\00vprintf\00.nv.constant0.main_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00p\00\00\00!\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\9F\00\00\00 \A0\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C0\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D1\00\00\00\03\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E1\00\00\00\01\00\0C\00\00\00\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\1D\01\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\94\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\A0\01\00\00\12\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\01\00\00\03\00\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\04\10\00\00\00\0C\81\80\80(\10\04@\00\00\00\00\00\00\00\04/\08\00\09\00\00\00\18\00\00\00\04\11\08\00\09\00\00\00\10\00\00\00\04\12\08\00\09\00\00\00\10\00\00\00\047\04\00\80\00\00\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03P\00\00\03\1B\FF\00\04\0F\04\00\0A\00\00\00\04F\04\000\01\00\00\04\1C\04\00@\01\00\00\04\05\0C\00\04\00\00\00\01\00\00\00\01\00\00\00\03\19\08\00\04\0A\08\00\0B\00\00\00\10\02\08\00\046\04\00\08\00\00\00\00\00\00\00\FF\FF\FF\FF\09\00\00\00\0A\00\00\00\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\0A\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00\00\02\00\00\00\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\82{\01\FF\00\0A\00\00\00\08\00\00\00\22\0E\00\19y\02\00\00\00\00\00\00!\00\00\00b\0E\00\B9z\04\00\00\84\00\00\00\0A\00\00\00\E2\0F\006x\01\01\F0\FF\FF\FF\00\00\00\00\00\E2\1F\00\02x\00\00\00\00\00\00\00\0F\00\00\00\E2\0F\00$r\03\FF\FF\00\00\00\FF\00\8E\07\00\E2\0F\00\B9z\06\00\00\02\00\01\00\0A\00\00\00\E4\0F\00\82{\08\00\00\00\00\01\00\0A\00\00\00\A2\00\00\02|\04\00\06\00\00\00\00\0F\00\08\00\E2\0F\00$~\05\FF\07\00\00\00\FF\00\8E\0F\00\E2\0F\00\10|\0A\02\04\00\00\00\FF\E0\F1\0F\00\E2/\00\87s\00\01\02\00\00\00\00\0A\10\00\00\E2\01\00\B9z\04\00\00\08\00\00\00\08\00\00\00\C6\0F\00$~\0B\FF\05\00\00\00\FF\06\0E\08\00\E2\0F\00\10|\06\01\04\00\00\00\FF\E0\F1\0F\00\E2\0F\00\B9z\04\00\00\09\00\00\00\08\00\00\00\C6\0F\00\87s\00\01\0A\08\00\00\00\0A\10\00\00\E2\01\00\10|\07\FF\04\00\00\00\FF\E4\7F\08\00\D0\0F\00Ny\14\10\00\00\00\00\00\00\00\00\00\CE\0F\00Cs\00\08\00\00\00\00\00\00\C0\03\00\EA_\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\FC\00\FC\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00GPU thread %llu has %llu\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00e\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\01\00\00\00\00\00\00\C2\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\98\03\00\00\00\00\00\00 \01\00\00\00\00\00\00\02\00\00\00\09\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\D4\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B8\04\00\00\00\00\00\00h\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00 \05\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\05\00\00\00\00\00\00`\00\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00/\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A4\05\00\00\00\00\00\00(\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\F6\00\00\00\04\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\05\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\0A\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\1D\01\00\00\04\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\00\00\00\18\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\86\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\06\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\00\02\00\00\00\00\00\00\03\00\00\00\09\00\00\00\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\97\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\08\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00p\00\00\00\08\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9A\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00K\01\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9C\08\00\00\00\00\00\00\18\02\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\04\00\00\00x\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00P\01\00\00\00\00\00\00P\01\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00x\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00P\01\00\00\00\00\00\00P\01\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00\18\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\80\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\00\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\80\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00\9C\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\02\00\00\00\00\00\00\18\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\01\01P\00\00\00\80\02\00\00\00\00\00\00\80\02\00\00@\00\00\00\07\00\08\00Z\00\00\00\00\00\00\00\00\00\00\00\11 \10\00\00\00\00\00\00\00\00\00\00\00\00\00\04\04\00\00\00\00\00\00H\00\00\00\00\00\00\00\00\00\00\00\00\00\00\000\0A//\03\00\F0\1F\0A.version 8.7\0A.target sm_90a\0A.address_size 64\0A2\00\F0\0C.extern .func (.param .b32 \12\00\F5\05_retval0) vprintf\0A(\0A$\00$64\16\00\11_\13\00?_0,\1D\00\08\F2\0C1\0A)\0A;\0A.global .align 1 .b8 (\00\F0\19Format_0[26] = {71, 80, 85, 32, 116, 104\0A\00\00\05\00`01, 97\09\00\120!\00\113\0D\00\148\05\00#178\00#04)\00/15)\00\06@10};?\01\F6\0Aisible .entry main_kernel\F8\008u64\19\00\04\FC\00\C0\0A)\0A.maxntid \9F\00\A6, 1\0A{\0A.loc\EF\00\118\EF\00!__\15\00\F2\02_depot0[16];\0A.reg6\01;%SP\0F\00\15L\10\00\8932 %r<2>!\00\B5rd<8>;\0A\0Amov2\00\1B,e\00b;\0Acvta\8D\00\22.u%\00\13,\\\00\22ld\DA\00\04N\00O1, [\E0\00\00!];b\00\11u\84\00\911, %tid.xY\00\00R\00\03\19\000d2,\1F\00r;\0Aadd.sQ\00#3,W\00\00\1F\00\02\1A\00\01\83\00Brd4,\89\00\190\16\00#5,\C4\00S0;\0Ast\B5\00\00\9D\00\10[\1D\00\14]H\00\0E\1B\00\22+8\1D\00y3;\0A{ //\8D\02\01\0B\00.0;\14\0081;\0A\F6\02\03\F1\02\01_\00\06\17\01\11[\0B\00\121_\00&4;l\01Krd6,\C0\02\03l\01\02\E9\02\04\D4\00\117:\00\1F6W\00\02\120W\00\D47;\0Acall.uni (x\03\14,F\032, (-\00\13,\B7\00@);\0A}\E5\00\90ret;\0A\0A}\0A\00">] loc(#loc)
} loc(#loc)


// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %1 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %2 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%0, %0, %0) threads in (%2, %0, %0) : i64 dynamic_shared_memory_size %1 args(%arg0 : i64) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.binary @main_kernel  [#gpu.object<#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">, properties = {ISAToBinaryTimeInMs = 18 : i64, LLVMIRToISATimeInMs = 2 : i64}, "P\EDU\BA\01\00\10\00\D8\12\00\00\00\00\00\00\02\00\01\01@\00\00\00\C8\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\07\00\01\00Z\00\00\00\00\00\00\00\00\00\00\00\11\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00\80\00\00\00\00\00\00\00\00\00\00\00x\0E\00\00\00\00\00\00\B8\0A\00\00\00\00\00\00Z\0DZ\00@\008\00\06\00@\00\0F\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.shared.reserved.0\00.nv.constant4\00\00\00\00.nv.global.init\00.rel.text.main_kernel\00.rela.text.main_kernel\00.debug_frame\00.rel.nv.constant.pic\00.rela.nv.constant4\00\00\00\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.constant0.main_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.reservedSmem.offset0\00.nv.shared.reserved.0\00__nv_reservedSMEM_offset_0_alias\00.nv.constant4\00\00\00\00.nv.global.init\00printfFormat_0\00.rel.text.main_kernel\00.rela.text.main_kernel\00.debug_frame\00.rel.nv.constant.pic\00.rela.nv.constant.pic\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00main_kernel\00vprintf\00.nv.constant0.main_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00p\00\00\00!\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\9F\00\00\00 \A0\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C0\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D1\00\00\00\03\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E1\00\00\00\01\00\0C\00\00\00\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\1D\01\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\94\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\A0\01\00\00\12\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\01\00\00\03\00\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\04\10\00\00\00\0C\81\80\80(\10\04@\00\00\00\00\00\00\00\04/\08\00\09\00\00\00\18\00\00\00\04\11\08\00\09\00\00\00\10\00\00\00\04\12\08\00\09\00\00\00\10\00\00\00\047\04\00\80\00\00\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03P\00\00\03\1B\FF\00\04\0F\04\00\0A\00\00\00\04F\04\000\01\00\00\04\1C\04\00@\01\00\00\04\05\0C\00\04\00\00\00\01\00\00\00\01\00\00\00\03\19\08\00\04\0A\08\00\0B\00\00\00\10\02\08\00\046\04\00\08\00\00\00\00\00\00\00\FF\FF\FF\FF\09\00\00\00\0A\00\00\00\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\0A\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00\00\02\00\00\00\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\82{\01\FF\00\0A\00\00\00\08\00\00\00\22\0E\00\19y\02\00\00\00\00\00\00!\00\00\00b\0E\00\B9z\04\00\00\84\00\00\00\0A\00\00\00\E2\0F\006x\01\01\F0\FF\FF\FF\00\00\00\00\00\E2\1F\00\02x\00\00\00\00\00\00\00\0F\00\00\00\E2\0F\00$r\03\FF\FF\00\00\00\FF\00\8E\07\00\E2\0F\00\B9z\06\00\00\02\00\01\00\0A\00\00\00\E4\0F\00\82{\08\00\00\00\00\01\00\0A\00\00\00\A2\00\00\02|\04\00\06\00\00\00\00\0F\00\08\00\E2\0F\00$~\05\FF\07\00\00\00\FF\00\8E\0F\00\E2\0F\00\10|\0A\02\04\00\00\00\FF\E0\F1\0F\00\E2/\00\87s\00\01\02\00\00\00\00\0A\10\00\00\E2\01\00\B9z\04\00\00\08\00\00\00\08\00\00\00\C6\0F\00$~\0B\FF\05\00\00\00\FF\06\0E\08\00\E2\0F\00\10|\06\01\04\00\00\00\FF\E0\F1\0F\00\E2\0F\00\B9z\04\00\00\09\00\00\00\08\00\00\00\C6\0F\00\87s\00\01\0A\08\00\00\00\0A\10\00\00\E2\01\00\10|\07\FF\04\00\00\00\FF\E4\7F\08\00\D0\0F\00Ny\14\10\00\00\00\00\00\00\00\00\00\CE\0F\00Cs\00\08\00\00\00\00\00\00\C0\03\00\EA_\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\FC\00\FC\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00GPU thread %llu has %llu\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00e\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\01\00\00\00\00\00\00\C2\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\98\03\00\00\00\00\00\00 \01\00\00\00\00\00\00\02\00\00\00\09\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\D4\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B8\04\00\00\00\00\00\00h\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00 \05\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\05\00\00\00\00\00\00`\00\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00/\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A4\05\00\00\00\00\00\00(\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\F6\00\00\00\04\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\05\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\0A\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\1D\01\00\00\04\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\00\00\00\18\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\86\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\06\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\00\02\00\00\00\00\00\00\03\00\00\00\09\00\00\00\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\97\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\08\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00p\00\00\00\08\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9A\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00K\01\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9C\08\00\00\00\00\00\00\18\02\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\04\00\00\00x\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00P\01\00\00\00\00\00\00P\01\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00x\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00P\01\00\00\00\00\00\00P\01\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00\18\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\80\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\00\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\80\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00\9C\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\02\00\00\00\00\00\00\18\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\01\01P\00\00\00\80\02\00\00\00\00\00\00\80\02\00\00@\00\00\00\07\00\08\00Z\00\00\00\00\00\00\00\00\00\00\00\11 \10\00\00\00\00\00\00\00\00\00\00\00\00\00\04\04\00\00\00\00\00\00H\00\00\00\00\00\00\00\00\00\00\00\00\00\00\000\0A//\03\00\F0\1F\0A.version 8.7\0A.target sm_90a\0A.address_size 64\0A2\00\F0\0C.extern .func (.param .b32 \12\00\F5\05_retval0) vprintf\0A(\0A$\00$64\16\00\11_\13\00?_0,\1D\00\08\F2\0C1\0A)\0A;\0A.global .align 1 .b8 (\00\F0\19Format_0[26] = {71, 80, 85, 32, 116, 104\0A\00\00\05\00`01, 97\09\00\120!\00\113\0D\00\148\05\00#178\00#04)\00/15)\00\06@10};?\01\F6\0Aisible .entry main_kernel\F8\008u64\19\00\04\FC\00\C0\0A)\0A.maxntid \9F\00\A6, 1\0A{\0A.loc\EF\00\118\EF\00!__\15\00\F2\02_depot0[16];\0A.reg6\01;%SP\0F\00\15L\10\00\8932 %r<2>!\00\B5rd<8>;\0A\0Amov2\00\1B,e\00b;\0Acvta\8D\00\22.u%\00\13,\\\00\22ld\DA\00\04N\00O1, [\E0\00\00!];b\00\11u\84\00\911, %tid.xY\00\00R\00\03\19\000d2,\1F\00r;\0Aadd.sQ\00#3,W\00\00\1F\00\02\1A\00\01\83\00Brd4,\89\00\190\16\00#5,\C4\00S0;\0Ast\B5\00\00\9D\00\10[\1D\00\14]H\00\0E\1B\00\22+8\1D\00y3;\0A{ //\8D\02\01\0B\00.0;\14\0081;\0A\F6\02\03\F1\02\01_\00\06\17\01\11[\0B\00\121_\00&4;l\01Krd6,\C0\02\03l\01\02\E9\02\04\D4\00\117:\00\1F6W\00\02\120W\00\D47;\0Acall.uni (x\03\14,F\032, (-\00\13,\B7\00@);\0A}\E5\00\90ret;\0A\0A}\0A\00">] loc(#loc)
} loc(#loc)


// -----// IR Dump Before CSE (cse) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %1 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %2 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%0, %0, %0) threads in (%2, %0, %0) : i64 dynamic_shared_memory_size %1 args(%arg0 : i64) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.binary @main_kernel  [#gpu.object<#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">, properties = {ISAToBinaryTimeInMs = 18 : i64, LLVMIRToISATimeInMs = 2 : i64}, "P\EDU\BA\01\00\10\00\D8\12\00\00\00\00\00\00\02\00\01\01@\00\00\00\C8\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\07\00\01\00Z\00\00\00\00\00\00\00\00\00\00\00\11\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00\80\00\00\00\00\00\00\00\00\00\00\00x\0E\00\00\00\00\00\00\B8\0A\00\00\00\00\00\00Z\0DZ\00@\008\00\06\00@\00\0F\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.shared.reserved.0\00.nv.constant4\00\00\00\00.nv.global.init\00.rel.text.main_kernel\00.rela.text.main_kernel\00.debug_frame\00.rel.nv.constant.pic\00.rela.nv.constant4\00\00\00\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.constant0.main_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.reservedSmem.offset0\00.nv.shared.reserved.0\00__nv_reservedSMEM_offset_0_alias\00.nv.constant4\00\00\00\00.nv.global.init\00printfFormat_0\00.rel.text.main_kernel\00.rela.text.main_kernel\00.debug_frame\00.rel.nv.constant.pic\00.rela.nv.constant.pic\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00main_kernel\00vprintf\00.nv.constant0.main_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00p\00\00\00!\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\9F\00\00\00 \A0\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C0\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D1\00\00\00\03\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E1\00\00\00\01\00\0C\00\00\00\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\1D\01\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\94\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\A0\01\00\00\12\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\01\00\00\03\00\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\04\10\00\00\00\0C\81\80\80(\10\04@\00\00\00\00\00\00\00\04/\08\00\09\00\00\00\18\00\00\00\04\11\08\00\09\00\00\00\10\00\00\00\04\12\08\00\09\00\00\00\10\00\00\00\047\04\00\80\00\00\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03P\00\00\03\1B\FF\00\04\0F\04\00\0A\00\00\00\04F\04\000\01\00\00\04\1C\04\00@\01\00\00\04\05\0C\00\04\00\00\00\01\00\00\00\01\00\00\00\03\19\08\00\04\0A\08\00\0B\00\00\00\10\02\08\00\046\04\00\08\00\00\00\00\00\00\00\FF\FF\FF\FF\09\00\00\00\0A\00\00\00\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\0A\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00\00\02\00\00\00\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\82{\01\FF\00\0A\00\00\00\08\00\00\00\22\0E\00\19y\02\00\00\00\00\00\00!\00\00\00b\0E\00\B9z\04\00\00\84\00\00\00\0A\00\00\00\E2\0F\006x\01\01\F0\FF\FF\FF\00\00\00\00\00\E2\1F\00\02x\00\00\00\00\00\00\00\0F\00\00\00\E2\0F\00$r\03\FF\FF\00\00\00\FF\00\8E\07\00\E2\0F\00\B9z\06\00\00\02\00\01\00\0A\00\00\00\E4\0F\00\82{\08\00\00\00\00\01\00\0A\00\00\00\A2\00\00\02|\04\00\06\00\00\00\00\0F\00\08\00\E2\0F\00$~\05\FF\07\00\00\00\FF\00\8E\0F\00\E2\0F\00\10|\0A\02\04\00\00\00\FF\E0\F1\0F\00\E2/\00\87s\00\01\02\00\00\00\00\0A\10\00\00\E2\01\00\B9z\04\00\00\08\00\00\00\08\00\00\00\C6\0F\00$~\0B\FF\05\00\00\00\FF\06\0E\08\00\E2\0F\00\10|\06\01\04\00\00\00\FF\E0\F1\0F\00\E2\0F\00\B9z\04\00\00\09\00\00\00\08\00\00\00\C6\0F\00\87s\00\01\0A\08\00\00\00\0A\10\00\00\E2\01\00\10|\07\FF\04\00\00\00\FF\E4\7F\08\00\D0\0F\00Ny\14\10\00\00\00\00\00\00\00\00\00\CE\0F\00Cs\00\08\00\00\00\00\00\00\C0\03\00\EA_\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\FC\00\FC\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00GPU thread %llu has %llu\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00e\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\01\00\00\00\00\00\00\C2\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\98\03\00\00\00\00\00\00 \01\00\00\00\00\00\00\02\00\00\00\09\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\D4\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B8\04\00\00\00\00\00\00h\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00 \05\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\05\00\00\00\00\00\00`\00\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00/\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A4\05\00\00\00\00\00\00(\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\F6\00\00\00\04\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\05\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\0A\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\1D\01\00\00\04\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\00\00\00\18\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\86\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\06\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\00\02\00\00\00\00\00\00\03\00\00\00\09\00\00\00\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\97\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\08\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00p\00\00\00\08\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9A\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00K\01\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9C\08\00\00\00\00\00\00\18\02\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\04\00\00\00x\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00P\01\00\00\00\00\00\00P\01\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00x\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00P\01\00\00\00\00\00\00P\01\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00\18\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\80\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\00\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\80\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00\9C\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\02\00\00\00\00\00\00\18\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\01\01P\00\00\00\80\02\00\00\00\00\00\00\80\02\00\00@\00\00\00\07\00\08\00Z\00\00\00\00\00\00\00\00\00\00\00\11 \10\00\00\00\00\00\00\00\00\00\00\00\00\00\04\04\00\00\00\00\00\00H\00\00\00\00\00\00\00\00\00\00\00\00\00\00\000\0A//\03\00\F0\1F\0A.version 8.7\0A.target sm_90a\0A.address_size 64\0A2\00\F0\0C.extern .func (.param .b32 \12\00\F5\05_retval0) vprintf\0A(\0A$\00$64\16\00\11_\13\00?_0,\1D\00\08\F2\0C1\0A)\0A;\0A.global .align 1 .b8 (\00\F0\19Format_0[26] = {71, 80, 85, 32, 116, 104\0A\00\00\05\00`01, 97\09\00\120!\00\113\0D\00\148\05\00#178\00#04)\00/15)\00\06@10};?\01\F6\0Aisible .entry main_kernel\F8\008u64\19\00\04\FC\00\C0\0A)\0A.maxntid \9F\00\A6, 1\0A{\0A.loc\EF\00\118\EF\00!__\15\00\F2\02_depot0[16];\0A.reg6\01;%SP\0F\00\15L\10\00\8932 %r<2>!\00\B5rd<8>;\0A\0Amov2\00\1B,e\00b;\0Acvta\8D\00\22.u%\00\13,\\\00\22ld\DA\00\04N\00O1, [\E0\00\00!];b\00\11u\84\00\911, %tid.xY\00\00R\00\03\19\000d2,\1F\00r;\0Aadd.sQ\00#3,W\00\00\1F\00\02\1A\00\01\83\00Brd4,\89\00\190\16\00#5,\C4\00S0;\0Ast\B5\00\00\9D\00\10[\1D\00\14]H\00\0E\1B\00\22+8\1D\00y3;\0A{ //\8D\02\01\0B\00.0;\14\0081;\0A\F6\02\03\F1\02\01_\00\06\17\01\11[\0B\00\121_\00&4;l\01Krd6,\C0\02\03l\01\02\E9\02\04\D4\00\117:\00\1F6W\00\02\120W\00\D47;\0Acall.uni (x\03\14,F\032, (-\00\13,\B7\00@);\0A}\E5\00\90ret;\0A\0A}\0A\00">] loc(#loc)
} loc(#loc)


// -----// IR Dump Before ReconcileUnrealizedCastsPass (reconcile-unrealized-casts) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  llvm.func @main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64 loc(#loc)
    %1 = llvm.mlir.constant(0 : i32) : i32 loc(#loc)
    %2 = llvm.mlir.constant(4 : index) : i64 loc(#loc)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%0, %0, %0) threads in (%2, %0, %0) : i64 dynamic_shared_memory_size %1 args(%arg0 : i64) loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  llvm.func @_mlir_ciface_main(%arg0: i64 loc(unknown)) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i64) -> () loc(#loc)
    llvm.return loc(#loc)
  } loc(#loc)
  gpu.binary @main_kernel  [#gpu.object<#nvvm.target<O = 3, chip = "sm_90a", features = "+ptx87">, properties = {ISAToBinaryTimeInMs = 18 : i64, LLVMIRToISATimeInMs = 2 : i64}, "P\EDU\BA\01\00\10\00\D8\12\00\00\00\00\00\00\02\00\01\01@\00\00\00\C8\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\07\00\01\00Z\00\00\00\00\00\00\00\00\00\00\00\11\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00\80\00\00\00\00\00\00\00\00\00\00\00x\0E\00\00\00\00\00\00\B8\0A\00\00\00\00\00\00Z\0DZ\00@\008\00\06\00@\00\0F\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.shared.reserved.0\00.nv.constant4\00\00\00\00.nv.global.init\00.rel.text.main_kernel\00.rela.text.main_kernel\00.debug_frame\00.rel.nv.constant.pic\00.rela.nv.constant4\00\00\00\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.constant0.main_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.reservedSmem.offset0\00.nv.shared.reserved.0\00__nv_reservedSMEM_offset_0_alias\00.nv.constant4\00\00\00\00.nv.global.init\00printfFormat_0\00.rel.text.main_kernel\00.rela.text.main_kernel\00.debug_frame\00.rel.nv.constant.pic\00.rela.nv.constant.pic\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00main_kernel\00vprintf\00.nv.constant0.main_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00p\00\00\00!\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\9F\00\00\00 \A0\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C0\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D1\00\00\00\03\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E1\00\00\00\01\00\0C\00\00\00\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\1D\01\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\94\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\A0\01\00\00\12\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\01\00\00\03\00\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\04\10\00\00\00\0C\81\80\80(\10\04@\00\00\00\00\00\00\00\04/\08\00\09\00\00\00\18\00\00\00\04\11\08\00\09\00\00\00\10\00\00\00\04\12\08\00\09\00\00\00\10\00\00\00\047\04\00\80\00\00\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03P\00\00\03\1B\FF\00\04\0F\04\00\0A\00\00\00\04F\04\000\01\00\00\04\1C\04\00@\01\00\00\04\05\0C\00\04\00\00\00\01\00\00\00\01\00\00\00\03\19\08\00\04\0A\08\00\0B\00\00\00\10\02\08\00\046\04\00\08\00\00\00\00\00\00\00\FF\FF\FF\FF\09\00\00\00\0A\00\00\00\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00\08\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\0A\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00\00\02\00\00\00\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\82{\01\FF\00\0A\00\00\00\08\00\00\00\22\0E\00\19y\02\00\00\00\00\00\00!\00\00\00b\0E\00\B9z\04\00\00\84\00\00\00\0A\00\00\00\E2\0F\006x\01\01\F0\FF\FF\FF\00\00\00\00\00\E2\1F\00\02x\00\00\00\00\00\00\00\0F\00\00\00\E2\0F\00$r\03\FF\FF\00\00\00\FF\00\8E\07\00\E2\0F\00\B9z\06\00\00\02\00\01\00\0A\00\00\00\E4\0F\00\82{\08\00\00\00\00\01\00\0A\00\00\00\A2\00\00\02|\04\00\06\00\00\00\00\0F\00\08\00\E2\0F\00$~\05\FF\07\00\00\00\FF\00\8E\0F\00\E2\0F\00\10|\0A\02\04\00\00\00\FF\E0\F1\0F\00\E2/\00\87s\00\01\02\00\00\00\00\0A\10\00\00\E2\01\00\B9z\04\00\00\08\00\00\00\08\00\00\00\C6\0F\00$~\0B\FF\05\00\00\00\FF\06\0E\08\00\E2\0F\00\10|\06\01\04\00\00\00\FF\E0\F1\0F\00\E2\0F\00\B9z\04\00\00\09\00\00\00\08\00\00\00\C6\0F\00\87s\00\01\0A\08\00\00\00\0A\10\00\00\E2\01\00\10|\07\FF\04\00\00\00\FF\E4\7F\08\00\D0\0F\00Ny\14\10\00\00\00\00\00\00\00\00\00\CE\0F\00Cs\00\08\00\00\00\00\00\00\C0\03\00\EA_\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\FC\00\FC\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00GPU thread %llu has %llu\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00e\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\01\00\00\00\00\00\00\C2\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\98\03\00\00\00\00\00\00 \01\00\00\00\00\00\00\02\00\00\00\09\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\D4\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B8\04\00\00\00\00\00\00h\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00 \05\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\05\00\00\00\00\00\00`\00\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00/\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A4\05\00\00\00\00\00\00(\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\F6\00\00\00\04\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\05\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\0A\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\1D\01\00\00\04\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\00\00\00\18\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\86\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\06\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\00\02\00\00\00\00\00\00\03\00\00\00\09\00\00\00\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\97\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\08\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00p\00\00\00\08\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9A\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00K\01\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9C\08\00\00\00\00\00\00\18\02\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\04\00\00\00x\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00P\01\00\00\00\00\00\00P\01\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00x\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00P\01\00\00\00\00\00\00P\01\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00\18\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\80\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\00\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\80\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00\9C\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\02\00\00\00\00\00\00\18\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\01\01P\00\00\00\80\02\00\00\00\00\00\00\80\02\00\00@\00\00\00\07\00\08\00Z\00\00\00\00\00\00\00\00\00\00\00\11 \10\00\00\00\00\00\00\00\00\00\00\00\00\00\04\04\00\00\00\00\00\00H\00\00\00\00\00\00\00\00\00\00\00\00\00\00\000\0A//\03\00\F0\1F\0A.version 8.7\0A.target sm_90a\0A.address_size 64\0A2\00\F0\0C.extern .func (.param .b32 \12\00\F5\05_retval0) vprintf\0A(\0A$\00$64\16\00\11_\13\00?_0,\1D\00\08\F2\0C1\0A)\0A;\0A.global .align 1 .b8 (\00\F0\19Format_0[26] = {71, 80, 85, 32, 116, 104\0A\00\00\05\00`01, 97\09\00\120!\00\113\0D\00\148\05\00#178\00#04)\00/15)\00\06@10};?\01\F6\0Aisible .entry main_kernel\F8\008u64\19\00\04\FC\00\C0\0A)\0A.maxntid \9F\00\A6, 1\0A{\0A.loc\EF\00\118\EF\00!__\15\00\F2\02_depot0[16];\0A.reg6\01;%SP\0F\00\15L\10\00\8932 %r<2>!\00\B5rd<8>;\0A\0Amov2\00\1B,e\00b;\0Acvta\8D\00\22.u%\00\13,\\\00\22ld\DA\00\04N\00O1, [\E0\00\00!];b\00\11u\84\00\911, %tid.xY\00\00R\00\03\19\000d2,\1F\00r;\0Aadd.sQ\00#3,W\00\00\1F\00\02\1A\00\01\83\00Brd4,\89\00\190\16\00#5,\C4\00S0;\0Ast\B5\00\00\9D\00\10[\1D\00\14]H\00\0E\1B\00\22+8\1D\00y3;\0A{ //\8D\02\01\0B\00.0;\14\0081;\0A\F6\02\03\F1\02\01_\00\06\17\01\11[\0B\00\121_\00&4;l\01Krd6,\C0\02\03l\01\02\E9\02\04\D4\00\117:\00\1F6W\00\02\120W\00\D47;\0Acall.uni (x\03\14,F\032, (-\00\13,\B7\00@);\0A}\E5\00\90ret;\0A\0A}\0A\00">] loc(#loc)
} loc(#loc)


Compiler pipeline: builtin.module(gpu-lower-to-nvvm-pipeline{cubin-chip=sm_90a cubin-features=+ptx87 opt-level=3})
