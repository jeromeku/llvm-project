// -----// IR Dump After OneShotBufferizePass (one-shot-bufferize) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main() -> tensor<256x1024xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    linalg.map outs(%alloc_1 : memref<256x1024xf32>)
      () {
        linalg.yield %cst : f32
      }
    linalg.matmul ins(%alloc, %alloc_0 : memref<256x512xf32>, memref<512x1024xf32>) outs(%alloc_1 : memref<256x1024xf32>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_1 : memref<256x1024xf32>) outs(%alloc_2 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_3 = arith.constant 0.000000e+00 : f32
      %1 = arith.cmpf ugt, %in, %cst_3 : f32
      %2 = arith.select %1, %in, %cst_3 : f32
      linalg.yield %2 : f32
    }
    %0 = bufferization.to_tensor %alloc_2 : memref<256x1024xf32> to tensor<256x1024xf32>
    return %0 : tensor<256x1024xf32>
  }
}


