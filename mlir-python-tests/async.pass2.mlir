module attributes {gpu.container_module} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private @__mbarrier() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<1 x i64>
  llvm.mlir.global private @bufferLhsGlobal() {addr_space = 3 : i32} : !llvm.array<64 x array<8 x f32>>
  llvm.mlir.global private @bufferRhsGlobal() {addr_space = 3 : i32} : !llvm.array<8 x array<128 x f32>>
  llvm.func @main() {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(128 : index) : i64
    %5 = llvm.mlir.constant(8 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(64 : index) : i64
    %9 = llvm.mlir.constant(7 : index) : i64
    %10 = llvm.mlir.constant(0 : i32) : i64
    %11 = llvm.mlir.constant(5 : i32) : i64
    %12 = llvm.mlir.constant(7 : i32) : i64
    %13 = llvm.mlir.constant(45 : index) : i64
    %14 = builtin.unrealized_conversion_cast %13 : i64 to index
    %15 = builtin.unrealized_conversion_cast %9 : i64 to index
    %16 = builtin.unrealized_conversion_cast %7 : i64 to index
    %17 = builtin.unrealized_conversion_cast %6 : i64 to index
    %18 = builtin.unrealized_conversion_cast %4 : i64 to index
    %19 = llvm.getelementptr %2[512] : (!llvm.ptr) -> !llvm.ptr, f32
    %20 = llvm.ptrtoint %19 : !llvm.ptr to i64
    %21 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
    %22 = llvm.insertvalue %21, %1[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %21, %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %6, %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %8, %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %5, %25[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %5, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %7, %27[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = builtin.unrealized_conversion_cast %28 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<64x8xf32>
    %30 = llvm.getelementptr %2[1024] : (!llvm.ptr) -> !llvm.ptr, f32
    %31 = llvm.ptrtoint %30 : !llvm.ptr to i64
    %32 = llvm.call @malloc(%31) : (i64) -> !llvm.ptr
    %33 = llvm.insertvalue %32, %1[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %32, %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %6, %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %5, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %4, %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %4, %37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %7, %38[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = builtin.unrealized_conversion_cast %39 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<8x128xf32>
    cf.br ^bb1(%17 : index)
  ^bb1(%41: index):  // 2 preds: ^bb0, ^bb4
    %42 = builtin.unrealized_conversion_cast %41 : index to i64
    %43 = builtin.unrealized_conversion_cast %41 : index to i64
    %44 = llvm.icmp "slt" %43, %5 : i64
    cf.cond_br %44, ^bb2(%17 : index), ^bb5(%17 : index)
  ^bb2(%45: index):  // 2 preds: ^bb1, ^bb3
    %46 = builtin.unrealized_conversion_cast %45 : index to i64
    %47 = builtin.unrealized_conversion_cast %45 : index to i64
    %48 = llvm.icmp "slt" %47, %4 : i64
    cf.cond_br %48, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %49 = llvm.mul %42, %4 overflow<nsw, nuw> : i64
    %50 = llvm.add %49, %46 overflow<nsw, nuw> : i64
    %51 = llvm.getelementptr inbounds|nuw %32[%50] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %3, %51 : f32, !llvm.ptr
    %52 = llvm.add %47, %7 : i64
    %53 = builtin.unrealized_conversion_cast %52 : i64 to index
    cf.br ^bb2(%53 : index)
  ^bb4:  // pred: ^bb2
    %54 = llvm.add %43, %7 : i64
    %55 = builtin.unrealized_conversion_cast %54 : i64 to index
    cf.br ^bb1(%55 : index)
  ^bb5(%56: index):  // 2 preds: ^bb1, ^bb8
    %57 = builtin.unrealized_conversion_cast %56 : index to i64
    %58 = builtin.unrealized_conversion_cast %56 : index to i64
    %59 = llvm.icmp "slt" %58, %8 : i64
    cf.cond_br %59, ^bb6(%17 : index), ^bb9
  ^bb6(%60: index):  // 2 preds: ^bb5, ^bb7
    %61 = builtin.unrealized_conversion_cast %60 : index to i64
    %62 = builtin.unrealized_conversion_cast %60 : index to i64
    %63 = llvm.icmp "slt" %62, %5 : i64
    cf.cond_br %63, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %64 = llvm.uitofp %62 : i64 to f32
    %65 = llvm.mul %57, %5 overflow<nsw, nuw> : i64
    %66 = llvm.add %65, %61 overflow<nsw, nuw> : i64
    %67 = llvm.getelementptr inbounds|nuw %21[%66] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %64, %67 : f32, !llvm.ptr
    %68 = llvm.add %62, %7 : i64
    %69 = builtin.unrealized_conversion_cast %68 : i64 to index
    cf.br ^bb6(%69 : index)
  ^bb8:  // pred: ^bb6
    %70 = llvm.add %58, %7 : i64
    %71 = builtin.unrealized_conversion_cast %70 : i64 to index
    cf.br ^bb5(%71 : index)
  ^bb9:  // pred: ^bb5
    %72 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%72] () : memref<64x8xf32>
    %73 = builtin.unrealized_conversion_cast %memref : memref<64x8xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %memref_0, %asyncToken_1 = gpu.alloc async [%72] () : memref<8x128xf32>
    %74 = builtin.unrealized_conversion_cast %memref_0 : memref<8x128xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %75 = gpu.memcpy async [%72] %memref, %29 : memref<64x8xf32>, memref<64x8xf32>
    %76 = gpu.memcpy async [%72] %memref_0, %40 : memref<8x128xf32>, memref<8x128xf32>
    %77 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %73, %77 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %78 = llvm.alloca %11 x i64 : (i64) -> !llvm.ptr
    llvm.store %8, %78 : i64, !llvm.ptr
    %79 = llvm.getelementptr %78[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %5, %79 : i64, !llvm.ptr
    %80 = llvm.call @mgpuTensorMapEncodeTiledMemref(%0, %77, %12, %10, %10, %10, %10, %78) : (i64, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> !llvm.ptr
    %81 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %74, %81 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %82 = llvm.alloca %11 x i64 : (i64) -> !llvm.ptr
    llvm.store %5, %82 : i64, !llvm.ptr
    %83 = llvm.getelementptr %82[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %4, %83 : i64, !llvm.ptr
    %84 = llvm.call @mgpuTensorMapEncodeTiledMemref(%0, %81, %12, %10, %10, %10, %10, %82) : (i64, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> !llvm.ptr
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%16, %16, %16) threads in (%18, %16, %16)  args(%80 : !llvm.ptr, %84 : !llvm.ptr, %17 : index, %14 : index, %15 : index)
    llvm.return
  }
  gpu.module @main_kernel [#nvvm.target<O = 3, chip = "sm_90", features = "+ptx80">] {
    llvm.mlir.global internal constant @printfFormat_1("[GPU] TMA LOADED rhs[7][0] %f\0A\00") {addr_space = 0 : i32}
    llvm.mlir.global internal constant @printfFormat_0("[GPU] TMA LOADED lhs[45][7] %f\0A\00") {addr_space = 0 : i32}
    llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32
    llvm.func @main_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.addressof @printfFormat_1 : !llvm.ptr
      %1 = llvm.mlir.constant(1 : index) : i64
      %2 = llvm.mlir.addressof @printfFormat_0 : !llvm.ptr
      %3 = llvm.mlir.constant(0 : i32) : i32
      %4 = llvm.mlir.constant(10000000 : index) : i64
      %5 = llvm.mlir.constant(6144 : index) : i64
      %6 = llvm.mlir.constant(128 : index) : i64
      %7 = llvm.mlir.constant(0 : index) : i64
      %8 = llvm.mlir.constant(8 : index) : i64
      %9 = llvm.mlir.addressof @bufferLhsGlobal : !llvm.ptr<3>
      %10 = llvm.mlir.addressof @bufferRhsGlobal : !llvm.ptr<3>
      %11 = llvm.mlir.addressof @__mbarrier : !llvm.ptr<3>
      %12 = llvm.getelementptr %9[0, 0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.array<64 x array<8 x f32>>
      %13 = llvm.getelementptr %10[0, 0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.array<8 x array<128 x f32>>
      %14 = llvm.getelementptr %11[0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.array<1 x i64>
      %15 = llvm.trunc %6 : i64 to i32
      nvvm.mbarrier.init.shared %14, %15 : !llvm.ptr<3>, i32
      nvvm.barrier0
      %16 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %17 = llvm.sext %16 : i32 to i64
      %18 = llvm.icmp "eq" %17, %7 : i64
      llvm.cond_br %18, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %19 = llvm.trunc %7 : i64 to i32
      llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2,$3} ], [$4];", "r,l,r,r,r" %12, %arg0, %19, %19, %14 : (!llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> ()
      llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2,$3} ], [$4];", "r,l,r,r,r" %13, %arg1, %19, %19, %14 : (!llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> ()
      %20 = llvm.trunc %5 : i64 to i32
      llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.shared.b64 _, [$0], $1;", "r,r" %14, %20 : (!llvm.ptr<3>, i32) -> ()
      llvm.br ^bb3
    ^bb2:  // pred: ^bb0
      %21 = llvm.trunc %7 : i64 to i32
      llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.shared.b64 _, [$0], $1;", "r,r" %14, %21 : (!llvm.ptr<3>, i32) -> ()
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      %22 = llvm.trunc %4 : i64 to i32
      llvm.inline_asm has_side_effects asm_dialect = att "{\0A\09.reg .pred       P1; \0A\09LAB_WAIT: \0A\09mbarrier.try_wait.parity.shared.b64 P1, [$0], $1, $2; \0A\09@P1 bra.uni DONE; \0A\09bra.uni     LAB_WAIT; \0A\09DONE: \0A\09}", "r,r,r" %14, %3, %22 : (!llvm.ptr<3>, i32, i32) -> ()
      %23 = llvm.icmp "eq" %17, %arg2 : i64
      llvm.cond_br %23, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %24 = llvm.mul %arg3, %8 overflow<nsw, nuw> : i64
      %25 = llvm.add %24, %arg4 overflow<nsw, nuw> : i64
      %26 = llvm.getelementptr inbounds|nuw %12[%25] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      %27 = llvm.load %26 : !llvm.ptr<3> -> f32
      %28 = llvm.mul %arg4, %6 overflow<nsw, nuw> : i64
      %29 = llvm.add %28, %arg2 overflow<nsw, nuw> : i64
      %30 = llvm.getelementptr inbounds|nuw %13[%29] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      %31 = llvm.load %30 : !llvm.ptr<3> -> f32
      %32 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x i8>
      %33 = llvm.fpext %27 : f32 to f64
      %34 = llvm.alloca %1 x !llvm.struct<(f64)> : (i64) -> !llvm.ptr
      %35 = llvm.getelementptr %34[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f64)>
      llvm.store %33, %35 : f64, !llvm.ptr
      %36 = llvm.call @vprintf(%32, %34) : (!llvm.ptr, !llvm.ptr) -> i32
      %37 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<31 x i8>
      %38 = llvm.fpext %31 : f32 to f64
      %39 = llvm.alloca %1 x !llvm.struct<(f64)> : (i64) -> !llvm.ptr
      %40 = llvm.getelementptr %39[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f64)>
      llvm.store %38, %40 : f64, !llvm.ptr
      %41 = llvm.call @vprintf(%37, %39) : (!llvm.ptr, !llvm.ptr) -> i32
      llvm.br ^bb5
    ^bb5:  // 2 preds: ^bb3, ^bb4
      llvm.return
    }
    llvm.mlir.global private @bufferLhsGlobal() {addr_space = 3 : i32} : !llvm.array<64 x array<8 x f32>>
    llvm.mlir.global private @bufferRhsGlobal() {addr_space = 3 : i32} : !llvm.array<8 x array<128 x f32>>
    llvm.mlir.global private @__mbarrier() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<1 x i64>
  }
  module attributes {transform.with_named_sequence} {
  }
  llvm.func @mgpuTensorMapEncodeTiledMemref(i64, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> !llvm.ptr
}

