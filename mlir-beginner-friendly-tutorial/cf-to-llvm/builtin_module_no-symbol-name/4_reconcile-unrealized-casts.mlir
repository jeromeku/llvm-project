// -----// IR Dump After ReconcileUnrealizedCastsPass (reconcile-unrealized-casts) //----- //
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  func.func @main() -> tensor<256x1024xf32> {
    %0 = llvm.mlir.constant(512 : index) : i64
    %1 = llvm.mlir.constant(1024 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(256 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(256 : index) : i64
    %7 = llvm.mlir.constant(512 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(131072 : index) : i64
    %10 = llvm.mlir.zero : !llvm.ptr
    %11 = llvm.getelementptr %10[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.mlir.constant(64 : index) : i64
    %14 = llvm.add %12, %13 : i64
    %15 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.sub %13, %17 : i64
    %19 = llvm.add %16, %18 : i64
    %20 = llvm.urem %19, %13 : i64
    %21 = llvm.sub %19, %20 : i64
    %22 = llvm.inttoptr %21 : i64 to !llvm.ptr
    %23 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.insertvalue %15, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.mlir.constant(0 : index) : i64
    %27 = llvm.insertvalue %26, %25[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %6, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %7, %28[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %7, %29[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %8, %30[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.mlir.constant(512 : index) : i64
    %33 = llvm.mlir.constant(1024 : index) : i64
    %34 = llvm.mlir.constant(1 : index) : i64
    %35 = llvm.mlir.constant(524288 : index) : i64
    %36 = llvm.mlir.zero : !llvm.ptr
    %37 = llvm.getelementptr %36[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.ptrtoint %37 : !llvm.ptr to i64
    %39 = llvm.mlir.constant(64 : index) : i64
    %40 = llvm.add %38, %39 : i64
    %41 = llvm.call @malloc(%40) : (i64) -> !llvm.ptr
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.mlir.constant(1 : index) : i64
    %44 = llvm.sub %39, %43 : i64
    %45 = llvm.add %42, %44 : i64
    %46 = llvm.urem %45, %39 : i64
    %47 = llvm.sub %45, %46 : i64
    %48 = llvm.inttoptr %47 : i64 to !llvm.ptr
    %49 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %50 = llvm.insertvalue %41, %49[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.insertvalue %48, %50[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.mlir.constant(0 : index) : i64
    %53 = llvm.insertvalue %52, %51[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.insertvalue %32, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %33, %54[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %33, %55[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.insertvalue %34, %56[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.mlir.constant(256 : index) : i64
    %59 = llvm.mlir.constant(1024 : index) : i64
    %60 = llvm.mlir.constant(1 : index) : i64
    %61 = llvm.mlir.constant(262144 : index) : i64
    %62 = llvm.mlir.zero : !llvm.ptr
    %63 = llvm.getelementptr %62[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.ptrtoint %63 : !llvm.ptr to i64
    %65 = llvm.mlir.constant(64 : index) : i64
    %66 = llvm.add %64, %65 : i64
    %67 = llvm.call @malloc(%66) : (i64) -> !llvm.ptr
    %68 = llvm.ptrtoint %67 : !llvm.ptr to i64
    %69 = llvm.mlir.constant(1 : index) : i64
    %70 = llvm.sub %65, %69 : i64
    %71 = llvm.add %68, %70 : i64
    %72 = llvm.urem %71, %65 : i64
    %73 = llvm.sub %71, %72 : i64
    %74 = llvm.inttoptr %73 : i64 to !llvm.ptr
    %75 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %76 = llvm.insertvalue %67, %75[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.insertvalue %74, %76[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.mlir.constant(0 : index) : i64
    %79 = llvm.insertvalue %78, %77[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.insertvalue %58, %79[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.insertvalue %59, %80[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.insertvalue %59, %81[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.insertvalue %60, %82[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%4 : i64)
  ^bb1(%84: i64):  // 2 preds: ^bb0, ^bb5
    %85 = llvm.icmp "slt" %84, %3 : i64
    llvm.cond_br %85, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%4 : i64)
  ^bb3(%86: i64):  // 2 preds: ^bb2, ^bb4
    %87 = llvm.icmp "slt" %86, %1 : i64
    llvm.cond_br %87, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %88 = llvm.extractvalue %83[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.mlir.constant(1024 : index) : i64
    %90 = llvm.mul %84, %89 overflow<nsw, nuw> : i64
    %91 = llvm.add %90, %86 overflow<nsw, nuw> : i64
    %92 = llvm.getelementptr inbounds|nuw %88[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %5, %92 : f32, !llvm.ptr
    %93 = llvm.add %86, %2 : i64
    llvm.br ^bb3(%93 : i64)
  ^bb5:  // pred: ^bb3
    %94 = llvm.add %84, %2 : i64
    llvm.br ^bb1(%94 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%4 : i64)
  ^bb7(%95: i64):  // 2 preds: ^bb6, ^bb14
    %96 = llvm.icmp "slt" %95, %3 : i64
    llvm.cond_br %96, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%4 : i64)
  ^bb9(%97: i64):  // 2 preds: ^bb8, ^bb13
    %98 = llvm.icmp "slt" %97, %1 : i64
    llvm.cond_br %98, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%4 : i64)
  ^bb11(%99: i64):  // 2 preds: ^bb10, ^bb12
    %100 = llvm.icmp "slt" %99, %0 : i64
    llvm.cond_br %100, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %101 = llvm.extractvalue %31[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.mlir.constant(512 : index) : i64
    %103 = llvm.mul %95, %102 overflow<nsw, nuw> : i64
    %104 = llvm.add %103, %99 overflow<nsw, nuw> : i64
    %105 = llvm.getelementptr inbounds|nuw %101[%104] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %106 = llvm.load %105 : !llvm.ptr -> f32
    %107 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.mlir.constant(1024 : index) : i64
    %109 = llvm.mul %99, %108 overflow<nsw, nuw> : i64
    %110 = llvm.add %109, %97 overflow<nsw, nuw> : i64
    %111 = llvm.getelementptr inbounds|nuw %107[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %112 = llvm.load %111 : !llvm.ptr -> f32
    %113 = llvm.extractvalue %83[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.mlir.constant(1024 : index) : i64
    %115 = llvm.mul %95, %114 overflow<nsw, nuw> : i64
    %116 = llvm.add %115, %97 overflow<nsw, nuw> : i64
    %117 = llvm.getelementptr inbounds|nuw %113[%116] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %118 = llvm.load %117 : !llvm.ptr -> f32
    %119 = llvm.fmul %106, %112 : f32
    %120 = llvm.fadd %118, %119 : f32
    %121 = llvm.extractvalue %83[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %122 = llvm.mlir.constant(1024 : index) : i64
    %123 = llvm.mul %95, %122 overflow<nsw, nuw> : i64
    %124 = llvm.add %123, %97 overflow<nsw, nuw> : i64
    %125 = llvm.getelementptr inbounds|nuw %121[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %120, %125 : f32, !llvm.ptr
    %126 = llvm.add %99, %2 : i64
    llvm.br ^bb11(%126 : i64)
  ^bb13:  // pred: ^bb11
    %127 = llvm.add %97, %2 : i64
    llvm.br ^bb9(%127 : i64)
  ^bb14:  // pred: ^bb9
    %128 = llvm.add %95, %2 : i64
    llvm.br ^bb7(%128 : i64)
  ^bb15:  // pred: ^bb7
    %129 = llvm.mlir.constant(256 : index) : i64
    %130 = llvm.mlir.constant(1024 : index) : i64
    %131 = llvm.mlir.constant(1 : index) : i64
    %132 = llvm.mlir.constant(262144 : index) : i64
    %133 = llvm.mlir.zero : !llvm.ptr
    %134 = llvm.getelementptr %133[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %135 = llvm.ptrtoint %134 : !llvm.ptr to i64
    %136 = llvm.mlir.constant(64 : index) : i64
    %137 = llvm.add %135, %136 : i64
    %138 = llvm.call @malloc(%137) : (i64) -> !llvm.ptr
    %139 = llvm.ptrtoint %138 : !llvm.ptr to i64
    %140 = llvm.mlir.constant(1 : index) : i64
    %141 = llvm.sub %136, %140 : i64
    %142 = llvm.add %139, %141 : i64
    %143 = llvm.urem %142, %136 : i64
    %144 = llvm.sub %142, %143 : i64
    %145 = llvm.inttoptr %144 : i64 to !llvm.ptr
    %146 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %147 = llvm.insertvalue %138, %146[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.insertvalue %145, %147[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %149 = llvm.mlir.constant(0 : index) : i64
    %150 = llvm.insertvalue %149, %148[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.insertvalue %129, %150[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %152 = llvm.insertvalue %130, %151[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %153 = llvm.insertvalue %130, %152[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %154 = llvm.insertvalue %131, %153[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %155 = builtin.unrealized_conversion_cast %154 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<256x1024xf32>
    llvm.br ^bb16(%4 : i64)
  ^bb16(%156: i64):  // 2 preds: ^bb15, ^bb20
    %157 = llvm.icmp "slt" %156, %3 : i64
    llvm.cond_br %157, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18(%4 : i64)
  ^bb18(%158: i64):  // 2 preds: ^bb17, ^bb19
    %159 = llvm.icmp "slt" %158, %1 : i64
    llvm.cond_br %159, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %160 = llvm.extractvalue %83[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.mlir.constant(1024 : index) : i64
    %162 = llvm.mul %156, %161 overflow<nsw, nuw> : i64
    %163 = llvm.add %162, %158 overflow<nsw, nuw> : i64
    %164 = llvm.getelementptr inbounds|nuw %160[%163] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %165 = llvm.load %164 : !llvm.ptr -> f32
    %166 = llvm.fcmp "ugt" %165, %5 : f32
    %167 = llvm.select %166, %165, %5 : i1, f32
    %168 = llvm.extractvalue %154[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %169 = llvm.mlir.constant(1024 : index) : i64
    %170 = llvm.mul %156, %169 overflow<nsw, nuw> : i64
    %171 = llvm.add %170, %158 overflow<nsw, nuw> : i64
    %172 = llvm.getelementptr inbounds|nuw %168[%171] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %167, %172 : f32, !llvm.ptr
    %173 = llvm.add %158, %2 : i64
    llvm.br ^bb18(%173 : i64)
  ^bb20:  // pred: ^bb18
    %174 = llvm.add %156, %2 : i64
    llvm.br ^bb16(%174 : i64)
  ^bb21:  // pred: ^bb16
    %175 = bufferization.to_tensor %155 : memref<256x1024xf32> to tensor<256x1024xf32>
    return %175 : tensor<256x1024xf32>
  }
}


