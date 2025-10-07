// -----// IR Dump Before ConvertNVVMToLLVMPass (convert-nvvm-to-llvm) ('builtin.module' operation) //----- //
#loc = loc(unknown)
module attributes {gpu.container_module} {
  func.func @saxpy(%arg0: memref<256x32xf32> loc(unknown), %arg1: memref<256x32xf32> loc(unknown), %arg2: f32 loc(unknown)) attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c32 = arith.constant 32 : index loc(#loc)
    %c1 = arith.constant 1 : index loc(#loc)
    %c256 = arith.constant 256 : index loc(#loc)
    %0 = gpu.wait async loc(#loc)
    %memref, %asyncToken = gpu.alloc async [%0] () : memref<256x32xf32> loc(#loc)
    %memref_0, %asyncToken_1 = gpu.alloc async [%asyncToken] () : memref<256x32xf32> loc(#loc)
    %1 = gpu.memcpy async [%asyncToken_1] %memref, %arg0 : memref<256x32xf32>, memref<256x32xf32> loc(#loc)
    %2 = gpu.memcpy async [%1] %memref_0, %arg1 : memref<256x32xf32>, memref<256x32xf32> loc(#loc)
    %3 = gpu.wait async [%2] loc(#loc)
    gpu.launch_func  @saxpy_kernel::@saxpy_kernel blocks in (%c256, %c1, %c1) threads in (%c32, %c1, %c1)  dynamic_shared_memory_size %c0_i32 args(%memref : memref<256x32xf32>, %memref_0 : memref<256x32xf32>, %arg2 : f32) loc(#loc)
    %4 = gpu.memcpy async [%3] %arg1, %memref_0 : memref<256x32xf32>, memref<256x32xf32> loc(#loc)
    %5 = gpu.wait async [%4] loc(#loc)
    return loc(#loc)
  } loc(#loc)
  gpu.module @saxpy_kernel {
    gpu.func @saxpy_kernel(%arg0: memref<256x32xf32> loc(unknown), %arg1: memref<256x32xf32> loc(unknown), %arg2: f32 loc(unknown)) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 256, 1, 1>} {
      %block_id_x = gpu.block_id  x loc(#loc)
      %thread_id_x = gpu.thread_id  x loc(#loc)
      %0 = memref.load %arg0[%block_id_x, %thread_id_x] : memref<256x32xf32> loc(#loc)
      %1 = memref.load %arg1[%block_id_x, %thread_id_x] : memref<256x32xf32> loc(#loc)
      %2 = arith.mulf %0, %arg2 : f32 loc(#loc)
      %3 = arith.addf %1, %2 : f32 loc(#loc)
      memref.store %3, %arg1[%block_id_x, %thread_id_x] : memref<256x32xf32> loc(#loc)
      gpu.return loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)


