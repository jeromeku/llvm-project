// -----// IR Dump After ConvertLinalgToLoopsPass (convert-linalg-to-loops) //----- //
module {
  func.func @main() -> tensor<256x1024xf32> {
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
      }
    }
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        scf.for %arg2 = %c0 to %c512 step %c1 {
          %1 = memref.load %alloc[%arg0, %arg2] : memref<256x512xf32>
          %2 = memref.load %alloc_0[%arg2, %arg1] : memref<512x1024xf32>
          %3 = memref.load %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
          %4 = arith.mulf %1, %2 : f32
          %5 = arith.addf %3, %4 : f32
          memref.store %5, %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = memref.load %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
        %2 = arith.cmpf ugt, %1, %cst : f32
        %3 = arith.select %2, %1, %cst : f32
        memref.store %3, %alloc_2[%arg0, %arg1] : memref<256x1024xf32>
      }
    }
    %0 = bufferization.to_tensor %alloc_2 : memref<256x1024xf32> to tensor<256x1024xf32>
    return %0 : tensor<256x1024xf32>
  }
}


