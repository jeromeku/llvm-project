// -----// IR Dump After ConvertControlFlowToLLVMPass (convert-cf-to-llvm) //----- //
module {
  func.func @main() -> tensor<256x1024xf32> {
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %c0 : index to i64
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    llvm.br ^bb1(%0 : i64)
  ^bb1(%1: i64):  // 2 preds: ^bb0, ^bb5
    %2 = builtin.unrealized_conversion_cast %1 : i64 to index
    %3 = arith.cmpi slt, %2, %c256 : index
    llvm.cond_br %3, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%0 : i64)
  ^bb3(%4: i64):  // 2 preds: ^bb2, ^bb4
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = arith.cmpi slt, %5, %c1024 : index
    llvm.cond_br %6, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    memref.store %cst, %alloc_1[%2, %5] : memref<256x1024xf32>
    %7 = arith.addi %5, %c1 : index
    %8 = builtin.unrealized_conversion_cast %7 : index to i64
    llvm.br ^bb3(%8 : i64)
  ^bb5:  // pred: ^bb3
    %9 = arith.addi %2, %c1 : index
    %10 = builtin.unrealized_conversion_cast %9 : index to i64
    llvm.br ^bb1(%10 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%0 : i64)
  ^bb7(%11: i64):  // 2 preds: ^bb6, ^bb14
    %12 = builtin.unrealized_conversion_cast %11 : i64 to index
    %13 = arith.cmpi slt, %12, %c256 : index
    llvm.cond_br %13, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%0 : i64)
  ^bb9(%14: i64):  // 2 preds: ^bb8, ^bb13
    %15 = builtin.unrealized_conversion_cast %14 : i64 to index
    %16 = arith.cmpi slt, %15, %c1024 : index
    llvm.cond_br %16, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%0 : i64)
  ^bb11(%17: i64):  // 2 preds: ^bb10, ^bb12
    %18 = builtin.unrealized_conversion_cast %17 : i64 to index
    %19 = arith.cmpi slt, %18, %c512 : index
    llvm.cond_br %19, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %20 = memref.load %alloc[%12, %18] : memref<256x512xf32>
    %21 = memref.load %alloc_0[%18, %15] : memref<512x1024xf32>
    %22 = memref.load %alloc_1[%12, %15] : memref<256x1024xf32>
    %23 = arith.mulf %20, %21 : f32
    %24 = arith.addf %22, %23 : f32
    memref.store %24, %alloc_1[%12, %15] : memref<256x1024xf32>
    %25 = arith.addi %18, %c1 : index
    %26 = builtin.unrealized_conversion_cast %25 : index to i64
    llvm.br ^bb11(%26 : i64)
  ^bb13:  // pred: ^bb11
    %27 = arith.addi %15, %c1 : index
    %28 = builtin.unrealized_conversion_cast %27 : index to i64
    llvm.br ^bb9(%28 : i64)
  ^bb14:  // pred: ^bb9
    %29 = arith.addi %12, %c1 : index
    %30 = builtin.unrealized_conversion_cast %29 : index to i64
    llvm.br ^bb7(%30 : i64)
  ^bb15:  // pred: ^bb7
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    llvm.br ^bb16(%0 : i64)
  ^bb16(%31: i64):  // 2 preds: ^bb15, ^bb20
    %32 = builtin.unrealized_conversion_cast %31 : i64 to index
    %33 = arith.cmpi slt, %32, %c256 : index
    llvm.cond_br %33, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18(%0 : i64)
  ^bb18(%34: i64):  // 2 preds: ^bb17, ^bb19
    %35 = builtin.unrealized_conversion_cast %34 : i64 to index
    %36 = arith.cmpi slt, %35, %c1024 : index
    llvm.cond_br %36, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %37 = memref.load %alloc_1[%32, %35] : memref<256x1024xf32>
    %38 = arith.cmpf ugt, %37, %cst : f32
    %39 = arith.select %38, %37, %cst : f32
    memref.store %39, %alloc_2[%32, %35] : memref<256x1024xf32>
    %40 = arith.addi %35, %c1 : index
    %41 = builtin.unrealized_conversion_cast %40 : index to i64
    llvm.br ^bb18(%41 : i64)
  ^bb20:  // pred: ^bb18
    %42 = arith.addi %32, %c1 : index
    %43 = builtin.unrealized_conversion_cast %42 : index to i64
    llvm.br ^bb16(%43 : i64)
  ^bb21:  // pred: ^bb16
    %44 = bufferization.to_tensor %alloc_2 : memref<256x1024xf32> to tensor<256x1024xf32>
    return %44 : tensor<256x1024xf32>
  }
}


