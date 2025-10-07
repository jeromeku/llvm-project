// -----// IR Dump After ArithToLLVMConversionPass (convert-arith-to-llvm) //----- //
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  func.func @main() -> tensor<256x1024xf32> {
    %0 = llvm.mlir.constant(512 : index) : i64
    %1 = llvm.mlir.constant(1024 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(256 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = builtin.unrealized_conversion_cast %5 : index to i64
    %7 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(256 : index) : i64
    %9 = llvm.mlir.constant(512 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(131072 : index) : i64
    %12 = llvm.mlir.zero : !llvm.ptr
    %13 = llvm.getelementptr %12[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %14 = llvm.ptrtoint %13 : !llvm.ptr to i64
    %15 = llvm.mlir.constant(64 : index) : i64
    %16 = llvm.add %14, %15 : i64
    %17 = llvm.call @malloc(%16) : (i64) -> !llvm.ptr
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.sub %15, %19 : i64
    %21 = llvm.add %18, %20 : i64
    %22 = llvm.urem %21, %15 : i64
    %23 = llvm.sub %21, %22 : i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    %25 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %26 = llvm.insertvalue %17, %25[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %24, %26[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.mlir.constant(0 : index) : i64
    %29 = llvm.insertvalue %28, %27[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %8, %29[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %9, %30[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %9, %31[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %10, %32[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(512 : index) : i64
    %35 = llvm.mlir.constant(1024 : index) : i64
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.mlir.constant(524288 : index) : i64
    %38 = llvm.mlir.zero : !llvm.ptr
    %39 = llvm.getelementptr %38[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.mlir.constant(64 : index) : i64
    %42 = llvm.add %40, %41 : i64
    %43 = llvm.call @malloc(%42) : (i64) -> !llvm.ptr
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.mlir.constant(1 : index) : i64
    %46 = llvm.sub %41, %45 : i64
    %47 = llvm.add %44, %46 : i64
    %48 = llvm.urem %47, %41 : i64
    %49 = llvm.sub %47, %48 : i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    %51 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %52 = llvm.insertvalue %43, %51[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.insertvalue %50, %52[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.mlir.constant(0 : index) : i64
    %55 = llvm.insertvalue %54, %53[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %34, %55[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.insertvalue %35, %56[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.insertvalue %35, %57[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %36, %58[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.mlir.constant(256 : index) : i64
    %61 = llvm.mlir.constant(1024 : index) : i64
    %62 = llvm.mlir.constant(1 : index) : i64
    %63 = llvm.mlir.constant(262144 : index) : i64
    %64 = llvm.mlir.zero : !llvm.ptr
    %65 = llvm.getelementptr %64[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %66 = llvm.ptrtoint %65 : !llvm.ptr to i64
    %67 = llvm.mlir.constant(64 : index) : i64
    %68 = llvm.add %66, %67 : i64
    %69 = llvm.call @malloc(%68) : (i64) -> !llvm.ptr
    %70 = llvm.ptrtoint %69 : !llvm.ptr to i64
    %71 = llvm.mlir.constant(1 : index) : i64
    %72 = llvm.sub %67, %71 : i64
    %73 = llvm.add %70, %72 : i64
    %74 = llvm.urem %73, %67 : i64
    %75 = llvm.sub %73, %74 : i64
    %76 = llvm.inttoptr %75 : i64 to !llvm.ptr
    %77 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %78 = llvm.insertvalue %69, %77[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.insertvalue %76, %78[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mlir.constant(0 : index) : i64
    %81 = llvm.insertvalue %80, %79[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.insertvalue %60, %81[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.insertvalue %61, %82[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.insertvalue %61, %83[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.insertvalue %62, %84[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%6 : i64)
  ^bb1(%86: i64):  // 2 preds: ^bb0, ^bb5
    %87 = llvm.icmp "slt" %86, %3 : i64
    llvm.cond_br %87, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%6 : i64)
  ^bb3(%88: i64):  // 2 preds: ^bb2, ^bb4
    %89 = llvm.icmp "slt" %88, %1 : i64
    llvm.cond_br %89, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %90 = llvm.extractvalue %85[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.mlir.constant(1024 : index) : i64
    %92 = llvm.mul %86, %91 overflow<nsw, nuw> : i64
    %93 = llvm.add %92, %88 overflow<nsw, nuw> : i64
    %94 = llvm.getelementptr inbounds|nuw %90[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %7, %94 : f32, !llvm.ptr
    %95 = llvm.add %88, %2 : i64
    %96 = builtin.unrealized_conversion_cast %95 : i64 to index
    %97 = builtin.unrealized_conversion_cast %96 : index to i64
    llvm.br ^bb3(%97 : i64)
  ^bb5:  // pred: ^bb3
    %98 = llvm.add %86, %2 : i64
    %99 = builtin.unrealized_conversion_cast %98 : i64 to index
    %100 = builtin.unrealized_conversion_cast %99 : index to i64
    llvm.br ^bb1(%100 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%6 : i64)
  ^bb7(%101: i64):  // 2 preds: ^bb6, ^bb14
    %102 = llvm.icmp "slt" %101, %3 : i64
    llvm.cond_br %102, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%6 : i64)
  ^bb9(%103: i64):  // 2 preds: ^bb8, ^bb13
    %104 = llvm.icmp "slt" %103, %1 : i64
    llvm.cond_br %104, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%6 : i64)
  ^bb11(%105: i64):  // 2 preds: ^bb10, ^bb12
    %106 = llvm.icmp "slt" %105, %0 : i64
    llvm.cond_br %106, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %107 = llvm.extractvalue %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.mlir.constant(512 : index) : i64
    %109 = llvm.mul %101, %108 overflow<nsw, nuw> : i64
    %110 = llvm.add %109, %105 overflow<nsw, nuw> : i64
    %111 = llvm.getelementptr inbounds|nuw %107[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %112 = llvm.load %111 : !llvm.ptr -> f32
    %113 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.mlir.constant(1024 : index) : i64
    %115 = llvm.mul %105, %114 overflow<nsw, nuw> : i64
    %116 = llvm.add %115, %103 overflow<nsw, nuw> : i64
    %117 = llvm.getelementptr inbounds|nuw %113[%116] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %118 = llvm.load %117 : !llvm.ptr -> f32
    %119 = llvm.extractvalue %85[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %120 = llvm.mlir.constant(1024 : index) : i64
    %121 = llvm.mul %101, %120 overflow<nsw, nuw> : i64
    %122 = llvm.add %121, %103 overflow<nsw, nuw> : i64
    %123 = llvm.getelementptr inbounds|nuw %119[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %124 = llvm.load %123 : !llvm.ptr -> f32
    %125 = llvm.fmul %112, %118 : f32
    %126 = llvm.fadd %124, %125 : f32
    %127 = llvm.extractvalue %85[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.mlir.constant(1024 : index) : i64
    %129 = llvm.mul %101, %128 overflow<nsw, nuw> : i64
    %130 = llvm.add %129, %103 overflow<nsw, nuw> : i64
    %131 = llvm.getelementptr inbounds|nuw %127[%130] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %126, %131 : f32, !llvm.ptr
    %132 = llvm.add %105, %2 : i64
    %133 = builtin.unrealized_conversion_cast %132 : i64 to index
    %134 = builtin.unrealized_conversion_cast %133 : index to i64
    llvm.br ^bb11(%134 : i64)
  ^bb13:  // pred: ^bb11
    %135 = llvm.add %103, %2 : i64
    %136 = builtin.unrealized_conversion_cast %135 : i64 to index
    %137 = builtin.unrealized_conversion_cast %136 : index to i64
    llvm.br ^bb9(%137 : i64)
  ^bb14:  // pred: ^bb9
    %138 = llvm.add %101, %2 : i64
    %139 = builtin.unrealized_conversion_cast %138 : i64 to index
    %140 = builtin.unrealized_conversion_cast %139 : index to i64
    llvm.br ^bb7(%140 : i64)
  ^bb15:  // pred: ^bb7
    %141 = llvm.mlir.constant(256 : index) : i64
    %142 = llvm.mlir.constant(1024 : index) : i64
    %143 = llvm.mlir.constant(1 : index) : i64
    %144 = llvm.mlir.constant(262144 : index) : i64
    %145 = llvm.mlir.zero : !llvm.ptr
    %146 = llvm.getelementptr %145[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %147 = llvm.ptrtoint %146 : !llvm.ptr to i64
    %148 = llvm.mlir.constant(64 : index) : i64
    %149 = llvm.add %147, %148 : i64
    %150 = llvm.call @malloc(%149) : (i64) -> !llvm.ptr
    %151 = llvm.ptrtoint %150 : !llvm.ptr to i64
    %152 = llvm.mlir.constant(1 : index) : i64
    %153 = llvm.sub %148, %152 : i64
    %154 = llvm.add %151, %153 : i64
    %155 = llvm.urem %154, %148 : i64
    %156 = llvm.sub %154, %155 : i64
    %157 = llvm.inttoptr %156 : i64 to !llvm.ptr
    %158 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %159 = llvm.insertvalue %150, %158[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.insertvalue %157, %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.mlir.constant(0 : index) : i64
    %162 = llvm.insertvalue %161, %160[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.insertvalue %141, %162[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %142, %163[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.insertvalue %142, %164[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = llvm.insertvalue %143, %165[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %167 = builtin.unrealized_conversion_cast %166 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<256x1024xf32>
    llvm.br ^bb16(%6 : i64)
  ^bb16(%168: i64):  // 2 preds: ^bb15, ^bb20
    %169 = llvm.icmp "slt" %168, %3 : i64
    llvm.cond_br %169, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18(%6 : i64)
  ^bb18(%170: i64):  // 2 preds: ^bb17, ^bb19
    %171 = llvm.icmp "slt" %170, %1 : i64
    llvm.cond_br %171, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %172 = llvm.extractvalue %85[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %173 = llvm.mlir.constant(1024 : index) : i64
    %174 = llvm.mul %168, %173 overflow<nsw, nuw> : i64
    %175 = llvm.add %174, %170 overflow<nsw, nuw> : i64
    %176 = llvm.getelementptr inbounds|nuw %172[%175] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %177 = llvm.load %176 : !llvm.ptr -> f32
    %178 = llvm.fcmp "ugt" %177, %7 : f32
    %179 = llvm.select %178, %177, %7 : i1, f32
    %180 = llvm.extractvalue %166[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.mlir.constant(1024 : index) : i64
    %182 = llvm.mul %168, %181 overflow<nsw, nuw> : i64
    %183 = llvm.add %182, %170 overflow<nsw, nuw> : i64
    %184 = llvm.getelementptr inbounds|nuw %180[%183] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %179, %184 : f32, !llvm.ptr
    %185 = llvm.add %170, %2 : i64
    %186 = builtin.unrealized_conversion_cast %185 : i64 to index
    %187 = builtin.unrealized_conversion_cast %186 : index to i64
    llvm.br ^bb18(%187 : i64)
  ^bb20:  // pred: ^bb18
    %188 = llvm.add %168, %2 : i64
    %189 = builtin.unrealized_conversion_cast %188 : i64 to index
    %190 = builtin.unrealized_conversion_cast %189 : index to i64
    llvm.br ^bb16(%190 : i64)
  ^bb21:  // pred: ^bb16
    %191 = bufferization.to_tensor %167 : memref<256x1024xf32> to tensor<256x1024xf32>
    return %191 : tensor<256x1024xf32>
  }
}


