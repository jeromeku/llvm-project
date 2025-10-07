// -----// IR Dump Before GpuKernelOutliningPass (gpu-kernel-outlining) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module {
  func.func @saxpy(%arg0: memref<256x32xf32> loc(unknown), %arg1: memref<256x32xf32> loc(unknown), %arg2: f32 loc(unknown)) attributes {llvm.emit_c_interface} {
    %0 = gpu.wait async loc(#loc)
    %memref, %asyncToken = gpu.alloc async [%0] () : memref<256x32xf32> loc(#loc)
    %memref_0, %asyncToken_1 = gpu.alloc async [%asyncToken] () : memref<256x32xf32> loc(#loc)
    %1 = gpu.memcpy async [%asyncToken_1] %memref, %arg0 : memref<256x32xf32>, memref<256x32xf32> loc(#loc)
    %2 = gpu.memcpy async [%1] %memref_0, %arg1 : memref<256x32xf32>, memref<256x32xf32> loc(#loc)
    %3 = gpu.wait async [%2] loc(#loc)
    %c256 = arith.constant 256 : index loc(#loc)
    %c1 = arith.constant 1 : index loc(#loc)
    %c1_2 = arith.constant 1 : index loc(#loc)
    %c32 = arith.constant 32 : index loc(#loc)
    %c1_3 = arith.constant 1 : index loc(#loc)
    %c1_4 = arith.constant 1 : index loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c256, %arg10 = %c1, %arg11 = %c1_2) threads(%arg6, %arg7, %arg8) in (%arg12 = %c32, %arg13 = %c1_3, %arg14 = %c1_4) dynamic_shared_memory_size %c0_i32 {
      %block_id_x = gpu.block_id  x loc(#loc)
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %6 = memref.load %memref[%block_id_x, %thread_id_x] : memref<256x32xf32> loc(#loc)
      %7 = memref.load %memref_0[%block_id_x, %thread_id_x] : memref<256x32xf32> loc(#loc)
      %8 = arith.mulf %6, %arg2 : f32 loc(#loc)
      %9 = arith.addf %7, %8 : f32 loc(#loc)
      memref.store %9, %memref_0[%block_id_x, %thread_id_x] : memref<256x32xf32> loc(#loc)
      gpu.terminator loc(#loc)
    } loc(#loc)
    %4 = gpu.memcpy async [%3] %arg1, %memref_0 : memref<256x32xf32>, memref<256x32xf32> loc(#loc)
    %5 = gpu.wait async [%4] loc(#loc)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)


