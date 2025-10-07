// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  func.func @main() -> tensor<256x1024xf32> {
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %c0 : index to i64
    %cst = arith.constant 0.000000e+00 : f32
    %1 = llvm.mlir.constant(256 : index) : i64
    %2 = llvm.mlir.constant(512 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(131072 : index) : i64
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.getelementptr %5[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %7 = llvm.ptrtoint %6 : !llvm.ptr to i64
    %8 = llvm.mlir.constant(64 : index) : i64
    %9 = llvm.add %7, %8 : i64
    %10 = llvm.call @malloc(%9) : (i64) -> !llvm.ptr
    %11 = llvm.ptrtoint %10 : !llvm.ptr to i64
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.sub %8, %12 : i64
    %14 = llvm.add %11, %13 : i64
    %15 = llvm.urem %14, %8 : i64
    %16 = llvm.sub %14, %15 : i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    %18 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.insertvalue %10, %18[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %17, %19[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.mlir.constant(0 : index) : i64
    %22 = llvm.insertvalue %21, %20[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %1, %22[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %2, %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %2, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %3, %25[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.mlir.constant(512 : index) : i64
    %28 = llvm.mlir.constant(1024 : index) : i64
    %29 = llvm.mlir.constant(1 : index) : i64
    %30 = llvm.mlir.constant(524288 : index) : i64
    %31 = llvm.mlir.zero : !llvm.ptr
    %32 = llvm.getelementptr %31[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.mlir.constant(64 : index) : i64
    %35 = llvm.add %33, %34 : i64
    %36 = llvm.call @malloc(%35) : (i64) -> !llvm.ptr
    %37 = llvm.ptrtoint %36 : !llvm.ptr to i64
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.sub %34, %38 : i64
    %40 = llvm.add %37, %39 : i64
    %41 = llvm.urem %40, %34 : i64
    %42 = llvm.sub %40, %41 : i64
    %43 = llvm.inttoptr %42 : i64 to !llvm.ptr
    %44 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.insertvalue %36, %44[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.insertvalue %43, %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.mlir.constant(0 : index) : i64
    %48 = llvm.insertvalue %47, %46[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %49 = llvm.insertvalue %27, %48[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.insertvalue %28, %49[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.insertvalue %28, %50[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.insertvalue %29, %51[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.mlir.constant(256 : index) : i64
    %54 = llvm.mlir.constant(1024 : index) : i64
    %55 = llvm.mlir.constant(1 : index) : i64
    %56 = llvm.mlir.constant(262144 : index) : i64
    %57 = llvm.mlir.zero : !llvm.ptr
    %58 = llvm.getelementptr %57[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.mlir.constant(64 : index) : i64
    %61 = llvm.add %59, %60 : i64
    %62 = llvm.call @malloc(%61) : (i64) -> !llvm.ptr
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.mlir.constant(1 : index) : i64
    %65 = llvm.sub %60, %64 : i64
    %66 = llvm.add %63, %65 : i64
    %67 = llvm.urem %66, %60 : i64
    %68 = llvm.sub %66, %67 : i64
    %69 = llvm.inttoptr %68 : i64 to !llvm.ptr
    %70 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %71 = llvm.insertvalue %62, %70[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.insertvalue %69, %71[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.mlir.constant(0 : index) : i64
    %74 = llvm.insertvalue %73, %72[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.insertvalue %53, %74[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.insertvalue %54, %75[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.insertvalue %54, %76[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.insertvalue %55, %77[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%0 : i64)
  ^bb1(%79: i64):  // 2 preds: ^bb0, ^bb5
    %80 = builtin.unrealized_conversion_cast %79 : i64 to index
    %81 = arith.cmpi slt, %80, %c256 : index
    llvm.cond_br %81, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%0 : i64)
  ^bb3(%82: i64):  // 2 preds: ^bb2, ^bb4
    %83 = builtin.unrealized_conversion_cast %82 : i64 to index
    %84 = arith.cmpi slt, %83, %c1024 : index
    llvm.cond_br %84, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %85 = llvm.extractvalue %78[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.mlir.constant(1024 : index) : i64
    %87 = llvm.mul %79, %86 overflow<nsw, nuw> : i64
    %88 = llvm.add %87, %82 overflow<nsw, nuw> : i64
    %89 = llvm.getelementptr inbounds|nuw %85[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %cst, %89 : f32, !llvm.ptr
    %90 = arith.addi %83, %c1 : index
    %91 = builtin.unrealized_conversion_cast %90 : index to i64
    llvm.br ^bb3(%91 : i64)
  ^bb5:  // pred: ^bb3
    %92 = arith.addi %80, %c1 : index
    %93 = builtin.unrealized_conversion_cast %92 : index to i64
    llvm.br ^bb1(%93 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%0 : i64)
  ^bb7(%94: i64):  // 2 preds: ^bb6, ^bb14
    %95 = builtin.unrealized_conversion_cast %94 : i64 to index
    %96 = arith.cmpi slt, %95, %c256 : index
    llvm.cond_br %96, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%0 : i64)
  ^bb9(%97: i64):  // 2 preds: ^bb8, ^bb13
    %98 = builtin.unrealized_conversion_cast %97 : i64 to index
    %99 = arith.cmpi slt, %98, %c1024 : index
    llvm.cond_br %99, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%0 : i64)
  ^bb11(%100: i64):  // 2 preds: ^bb10, ^bb12
    %101 = builtin.unrealized_conversion_cast %100 : i64 to index
    %102 = arith.cmpi slt, %101, %c512 : index
    llvm.cond_br %102, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %103 = llvm.extractvalue %26[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.mlir.constant(512 : index) : i64
    %105 = llvm.mul %94, %104 overflow<nsw, nuw> : i64
    %106 = llvm.add %105, %100 overflow<nsw, nuw> : i64
    %107 = llvm.getelementptr inbounds|nuw %103[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %108 = llvm.load %107 : !llvm.ptr -> f32
    %109 = llvm.extractvalue %52[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.mlir.constant(1024 : index) : i64
    %111 = llvm.mul %100, %110 overflow<nsw, nuw> : i64
    %112 = llvm.add %111, %97 overflow<nsw, nuw> : i64
    %113 = llvm.getelementptr inbounds|nuw %109[%112] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %114 = llvm.load %113 : !llvm.ptr -> f32
    %115 = llvm.extractvalue %78[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.mlir.constant(1024 : index) : i64
    %117 = llvm.mul %94, %116 overflow<nsw, nuw> : i64
    %118 = llvm.add %117, %97 overflow<nsw, nuw> : i64
    %119 = llvm.getelementptr inbounds|nuw %115[%118] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %120 = llvm.load %119 : !llvm.ptr -> f32
    %121 = arith.mulf %108, %114 : f32
    %122 = arith.addf %120, %121 : f32
    %123 = llvm.extractvalue %78[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %124 = llvm.mlir.constant(1024 : index) : i64
    %125 = llvm.mul %94, %124 overflow<nsw, nuw> : i64
    %126 = llvm.add %125, %97 overflow<nsw, nuw> : i64
    %127 = llvm.getelementptr inbounds|nuw %123[%126] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %122, %127 : f32, !llvm.ptr
    %128 = arith.addi %101, %c1 : index
    %129 = builtin.unrealized_conversion_cast %128 : index to i64
    llvm.br ^bb11(%129 : i64)
  ^bb13:  // pred: ^bb11
    %130 = arith.addi %98, %c1 : index
    %131 = builtin.unrealized_conversion_cast %130 : index to i64
    llvm.br ^bb9(%131 : i64)
  ^bb14:  // pred: ^bb9
    %132 = arith.addi %95, %c1 : index
    %133 = builtin.unrealized_conversion_cast %132 : index to i64
    llvm.br ^bb7(%133 : i64)
  ^bb15:  // pred: ^bb7
    %134 = llvm.mlir.constant(256 : index) : i64
    %135 = llvm.mlir.constant(1024 : index) : i64
    %136 = llvm.mlir.constant(1 : index) : i64
    %137 = llvm.mlir.constant(262144 : index) : i64
    %138 = llvm.mlir.zero : !llvm.ptr
    %139 = llvm.getelementptr %138[%137] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %140 = llvm.ptrtoint %139 : !llvm.ptr to i64
    %141 = llvm.mlir.constant(64 : index) : i64
    %142 = llvm.add %140, %141 : i64
    %143 = llvm.call @malloc(%142) : (i64) -> !llvm.ptr
    %144 = llvm.ptrtoint %143 : !llvm.ptr to i64
    %145 = llvm.mlir.constant(1 : index) : i64
    %146 = llvm.sub %141, %145 : i64
    %147 = llvm.add %144, %146 : i64
    %148 = llvm.urem %147, %141 : i64
    %149 = llvm.sub %147, %148 : i64
    %150 = llvm.inttoptr %149 : i64 to !llvm.ptr
    %151 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %152 = llvm.insertvalue %143, %151[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %153 = llvm.insertvalue %150, %152[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %154 = llvm.mlir.constant(0 : index) : i64
    %155 = llvm.insertvalue %154, %153[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %156 = llvm.insertvalue %134, %155[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %157 = llvm.insertvalue %135, %156[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %158 = llvm.insertvalue %135, %157[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.insertvalue %136, %158[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = builtin.unrealized_conversion_cast %159 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<256x1024xf32>
    llvm.br ^bb16(%0 : i64)
  ^bb16(%161: i64):  // 2 preds: ^bb15, ^bb20
    %162 = builtin.unrealized_conversion_cast %161 : i64 to index
    %163 = arith.cmpi slt, %162, %c256 : index
    llvm.cond_br %163, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18(%0 : i64)
  ^bb18(%164: i64):  // 2 preds: ^bb17, ^bb19
    %165 = builtin.unrealized_conversion_cast %164 : i64 to index
    %166 = arith.cmpi slt, %165, %c1024 : index
    llvm.cond_br %166, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %167 = llvm.extractvalue %78[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = llvm.mlir.constant(1024 : index) : i64
    %169 = llvm.mul %161, %168 overflow<nsw, nuw> : i64
    %170 = llvm.add %169, %164 overflow<nsw, nuw> : i64
    %171 = llvm.getelementptr inbounds|nuw %167[%170] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %172 = llvm.load %171 : !llvm.ptr -> f32
    %173 = arith.cmpf ugt, %172, %cst : f32
    %174 = arith.select %173, %172, %cst : f32
    %175 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.mlir.constant(1024 : index) : i64
    %177 = llvm.mul %161, %176 overflow<nsw, nuw> : i64
    %178 = llvm.add %177, %164 overflow<nsw, nuw> : i64
    %179 = llvm.getelementptr inbounds|nuw %175[%178] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %174, %179 : f32, !llvm.ptr
    %180 = arith.addi %165, %c1 : index
    %181 = builtin.unrealized_conversion_cast %180 : index to i64
    llvm.br ^bb18(%181 : i64)
  ^bb20:  // pred: ^bb18
    %182 = arith.addi %162, %c1 : index
    %183 = builtin.unrealized_conversion_cast %182 : index to i64
    llvm.br ^bb16(%183 : i64)
  ^bb21:  // pred: ^bb16
    %184 = bufferization.to_tensor %160 : memref<256x1024xf32> to tensor<256x1024xf32>
    return %184 : tensor<256x1024xf32>
  }
}


