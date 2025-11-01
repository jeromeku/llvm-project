module {
  func.func @gemm_128_128_64(%arg0: memref<128x64xf16>, %arg1: memref<64x128xf16>, %arg2: memref<128x128xf32>) attributes {llvm.emit_c_interface} {
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] () : memref<128x64xf16>
    %memref_0, %asyncToken_1 = gpu.alloc async [%asyncToken] () : memref<64x128xf16>
    %memref_2, %asyncToken_3 = gpu.alloc async [%asyncToken_1] () : memref<128x128xf32>
    %1 = gpu.memcpy async [%asyncToken_3] %memref, %arg0 : memref<128x64xf16>, memref<128x64xf16>
    %2 = gpu.memcpy async [%1] %memref_0, %arg1 : memref<64x128xf16>, memref<64x128xf16>
    %3 = gpu.wait async [%2]
    %cast = memref.cast %memref : memref<128x64xf16> to memref<*xf16>
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %4 = nvgpu.tma.create.descriptor %cast box[%c128, %c64] : memref<*xf16> -> <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
    %cast_4 = memref.cast %memref_0 : memref<64x128xf16> to memref<*xf16>
    %c64_5 = arith.constant 64 : index
    %c64_6 = arith.constant 64 : index
    %5 = nvgpu.tma.create.descriptor %cast_4 box[%c64_5, %c64_6] : memref<*xf16> -> <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
    %c1 = arith.constant 1 : index
    %c1_7 = arith.constant 1 : index
    %c1_8 = arith.constant 1 : index
    %c128_9 = arith.constant 128 : index
    %c1_10 = arith.constant 1 : index
    %c1_11 = arith.constant 1 : index
    %c32768_i32 = arith.constant 32768 : i32
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1_7, %arg11 = %c1_8) threads(%arg6, %arg7, %arg8) in (%arg12 = %c128_9, %arg13 = %c1_10, %arg14 = %c1_11) dynamic_shared_memory_size %c32768_i32 {
      %thread_id_x = gpu.thread_id  x
      %7 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>
      %c0 = arith.constant 0 : index
      %8 = arith.cmpi eq, %thread_id_x, %c0 : index
      %c0_12 = arith.constant 0 : index
      %c1_13 = arith.constant 1 : index
      nvgpu.mbarrier.init %7[%c0_12], %c1_13, predicate = %8 : <memorySpace = #gpu.address_space<workgroup>>
      nvgpu.tma.prefetch.descriptor %4, predicate = %8 : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
      nvgpu.tma.prefetch.descriptor %5, predicate = %8 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
      %9 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
      %c0_14 = arith.constant 0 : index
      %view = memref.view %9[%c0_14][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
      %10 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
      %c16384 = arith.constant 16384 : index
      %view_15 = memref.view %10[%c16384][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x128xf16, #gpu.address_space<workgroup>>
      %11 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
      %c0_16 = arith.constant 0 : index
      %view_17 = memref.view %11[%c0_16][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
      %12 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
      %c16384_18 = arith.constant 16384 : index
      %view_19 = memref.view %12[%c16384_18][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
      %13 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
      %c24576 = arith.constant 24576 : index
      %view_20 = memref.view %13[%c24576][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
      %c0_21 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      nvgpu.mbarrier.arrive.expect_tx %7[%c0_21], %c32768, predicate = %8 : <memorySpace = #gpu.address_space<workgroup>>
      %c0_22 = arith.constant 0 : index
      %c0_23 = arith.constant 0 : index
      %c0_24 = arith.constant 0 : index
      nvgpu.tma.async.load %4[%c0_23, %c0_24], %7[%c0_22] to %view_17, predicate = %8 : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<128x64xf16, #gpu.address_space<workgroup>>
      %c0_25 = arith.constant 0 : index
      %c0_26 = arith.constant 0 : index
      %c0_27 = arith.constant 0 : index
      nvgpu.tma.async.load %5[%c0_26, %c0_27], %7[%c0_25] to %view_19, predicate = %8 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<64x64xf16, #gpu.address_space<workgroup>>
      %c0_28 = arith.constant 0 : index
      %c64_29 = arith.constant 64 : index
      %c0_30 = arith.constant 0 : index
      nvgpu.tma.async.load %5[%c64_29, %c0_30], %7[%c0_28] to %view_20, predicate = %8 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<64x64xf16, #gpu.address_space<workgroup>>
      %c0_31 = arith.constant 0 : index
      %c10000000 = arith.constant 10000000 : index
      %false = arith.constant false
      nvgpu.mbarrier.try_wait.parity %7[%c0_31], %false, %c10000000 : <memorySpace = #gpu.address_space<workgroup>>
      %14 = nvgpu.warpgroup.mma.init.accumulator -> <fragmented = vector<128x128xf32>>
      %15 = nvgpu.warpgroup.generate.descriptor %view, %4 : memref<128x64xf16, #gpu.address_space<workgroup>>, <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>
      %16 = nvgpu.warpgroup.generate.descriptor %view_15, %5 : memref<64x128xf16, #gpu.address_space<workgroup>>, <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>
      %17 = nvgpu.warpgroup.mma %15, %16, %14 {transposeB} : <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>, <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>, <fragmented = vector<128x128xf32>> -> <fragmented = vector<128x128xf32>>
      nvgpu.warpgroup.mma.store %17, %memref_2 : <fragmented = vector<128x128xf32>> to memref<128x128xf32>
      gpu.terminator
    }
    %6 = gpu.memcpy async [%3] %arg2, %memref_2 : memref<128x128xf32>, memref<128x128xf32>
    gpu.wait [%6]
    return
  }
}
