module attributes {gpu.container_module} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private @__mbarrier() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<1 x i64>
  llvm.mlir.global private @bufferLhsGlobal() {addr_space = 3 : i32} : !llvm.array<64 x array<8 x f32>>
  llvm.mlir.global private @bufferRhsGlobal() {addr_space = 3 : i32} : !llvm.array<8 x array<128 x f32>>
  llvm.func @main() {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.constant(2 : index) : i64
    %2 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(128 : index) : i64
    %6 = llvm.mlir.constant(8 : index) : i64
    %7 = llvm.mlir.constant(0 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(64 : index) : i64
    %10 = llvm.mlir.constant(7 : index) : i64
    %11 = llvm.mlir.constant(0 : i32) : i64
    %12 = llvm.mlir.constant(5 : i32) : i64
    %13 = llvm.mlir.constant(7 : i32) : i64
    %14 = llvm.mlir.constant(45 : index) : i64
    %15 = llvm.getelementptr %3[512] : (!llvm.ptr) -> !llvm.ptr, f32
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.call @malloc(%16) : (i64) -> !llvm.ptr
    %18 = llvm.getelementptr %3[1024] : (!llvm.ptr) -> !llvm.ptr, f32
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.call @malloc(%19) : (i64) -> !llvm.ptr
    llvm.br ^bb1(%7 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb4
    %22 = llvm.icmp "slt" %21, %6 : i64
    llvm.cond_br %22, ^bb2(%7 : i64), ^bb5(%7 : i64)
  ^bb2(%23: i64):  // 2 preds: ^bb1, ^bb3
    %24 = llvm.icmp "slt" %23, %5 : i64
    llvm.cond_br %24, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %25 = llvm.mul %21, %5 overflow<nsw, nuw> : i64
    %26 = llvm.add %25, %23 overflow<nsw, nuw> : i64
    %27 = llvm.getelementptr inbounds|nuw %20[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %4, %27 : f32, !llvm.ptr
    %28 = llvm.add %23, %8 : i64
    llvm.br ^bb2(%28 : i64)
  ^bb4:  // pred: ^bb2
    %29 = llvm.add %21, %8 : i64
    llvm.br ^bb1(%29 : i64)
  ^bb5(%30: i64):  // 2 preds: ^bb1, ^bb8
    %31 = llvm.icmp "slt" %30, %9 : i64
    llvm.cond_br %31, ^bb6(%7 : i64), ^bb9
  ^bb6(%32: i64):  // 2 preds: ^bb5, ^bb7
    %33 = llvm.icmp "slt" %32, %6 : i64
    llvm.cond_br %33, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %34 = llvm.uitofp %32 : i64 to f32
    %35 = llvm.mul %30, %6 overflow<nsw, nuw> : i64
    %36 = llvm.add %35, %32 overflow<nsw, nuw> : i64
    %37 = llvm.getelementptr inbounds|nuw %17[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %34, %37 : f32, !llvm.ptr
    %38 = llvm.add %32, %8 : i64
    llvm.br ^bb6(%38 : i64)
  ^bb8:  // pred: ^bb6
    %39 = llvm.add %30, %8 : i64
    llvm.br ^bb5(%39 : i64)
  ^bb9:  // pred: ^bb5
    %40 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    %41 = llvm.call @mgpuMemAlloc(%16, %40, %0) : (i64, !llvm.ptr, i8) -> !llvm.ptr
    %42 = llvm.insertvalue %41, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.insertvalue %41, %42[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %7, %43[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.insertvalue %9, %44[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.insertvalue %6, %45[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.insertvalue %6, %46[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.insertvalue %8, %47[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %49 = llvm.call @mgpuMemAlloc(%19, %40, %0) : (i64, !llvm.ptr, i8) -> !llvm.ptr
    %50 = llvm.insertvalue %49, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.insertvalue %49, %50[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.insertvalue %7, %51[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.insertvalue %6, %52[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.insertvalue %5, %53[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %5, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %8, %55[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @mgpuMemcpy(%41, %17, %16, %40) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
    llvm.call @mgpuMemcpy(%49, %20, %19, %40) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
    %57 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %48, %57 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %58 = llvm.alloca %12 x i64 : (i64) -> !llvm.ptr
    llvm.store %9, %58 : i64, !llvm.ptr
    %59 = llvm.getelementptr %58[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %6, %59 : i64, !llvm.ptr
    %60 = llvm.call @mgpuTensorMapEncodeTiledMemref(%1, %57, %13, %11, %11, %11, %11, %58) : (i64, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> !llvm.ptr
    %61 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %56, %61 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %62 = llvm.alloca %12 x i64 : (i64) -> !llvm.ptr
    llvm.store %6, %62 : i64, !llvm.ptr
    %63 = llvm.getelementptr %62[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %5, %63 : i64, !llvm.ptr
    %64 = llvm.call @mgpuTensorMapEncodeTiledMemref(%1, %61, %13, %11, %11, %11, %11, %62) : (i64, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> !llvm.ptr
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%8, %8, %8) threads in (%5, %8, %8) : i64 args(%60 : !llvm.ptr, %64 : !llvm.ptr, %7 : i64, %14 : i64, %10 : i64)
    llvm.return
  }
  gpu.binary @main_kernel  [#gpu.object<#nvvm.target<O = 3, chip = "sm_90", features = "+ptx80">, properties = {LLVMIRToISATimeInMs = 2 : i64, O = 3 : i32}, assembly = "//\0A// Generated by LLVM NVPTX Back-End\0A//\0A\0A.version 8.0\0A.target sm_90\0A.address_size 64\0A\0A\09// .globl\09main_kernel\0A.extern .func  (.param .b32 func_retval0) vprintf\0A(\0A\09.param .b64 vprintf_param_0,\0A\09.param .b64 vprintf_param_1\0A)\0A;\0A.global .align 1 .b8 printfFormat_1[31] = {91, 71, 80, 85, 93, 32, 84, 77, 65, 32, 76, 79, 65, 68, 69, 68, 32, 114, 104, 115, 91, 55, 93, 91, 48, 93, 32, 37, 102, 10};\0A.global .align 1 .b8 printfFormat_0[32] = {91, 71, 80, 85, 93, 32, 84, 77, 65, 32, 76, 79, 65, 68, 69, 68, 32, 108, 104, 115, 91, 52, 53, 93, 91, 55, 93, 32, 37, 102, 10};\0A// bufferLhsGlobal has been demoted\0A// bufferRhsGlobal has been demoted\0A// __mbarrier has been demoted\0A\0A.visible .entry main_kernel(\0A\09.param .u64 .ptr .align 1 main_kernel_param_0,\0A\09.param .u64 .ptr .align 1 main_kernel_param_1,\0A\09.param .u64 main_kernel_param_2,\0A\09.param .u64 main_kernel_param_3,\0A\09.param .u64 main_kernel_param_4\0A)\0A.maxntid 128, 1, 1\0A{\0A\09.local .align 8 .b8 \09__local_depot0[16];\0A\09.reg .b64 \09%SP;\0A\09.reg .b64 \09%SPL;\0A\09.reg .pred \09%p<3>;\0A\09.reg .b32 \09%r<15>;\0A\09.reg .b64 \09%rd<30>;\0A\09// demoted variable\0A\09.shared .align 4 .b8 bufferLhsGlobal[2048];\0A\09// demoted variable\0A\09.shared .align 4 .b8 bufferRhsGlobal[4096];\0A\09// demoted variable\0A\09.shared .align 8 .b8 __mbarrier[8];\0A\09mov.b64 \09%SPL, __local_depot0;\0A\09cvta.local.u64 \09%SP, %SPL;\0A\09ld.param.b64 \09%rd2, [main_kernel_param_2];\0A\09mov.b32 \09%r1, 128;\0A\09mbarrier.init.shared.b64 \09[__mbarrier], %r1;\0A\09bar.sync \090;\0A\09mov.u32 \09%r2, %tid.x;\0A\09cvt.u64.u32 \09%rd1, %r2;\0A\09setp.ne.b32 \09%p1, %r2, 0;\0A\09mov.b64 \09%rd29, __mbarrier;\0A\09@%p1 bra \09$L__BB0_2;\0A\09bra.uni \09$L__BB0_1;\0A$L__BB0_2:\0A\09cvt.u32.u64 \09%r3, %rd29;\0A\09mov.b32 \09%r4, 0;\0A\09// begin inline asm\0A\09mbarrier.arrive.expect_tx.shared.b64 _, [%r3], %r4;\0A\09// end inline asm\0A\09bra.uni \09$L__BB0_3;\0A$L__BB0_1:\0A\09ld.param.b64 \09%rd6, [main_kernel_param_1];\0A\09ld.param.b64 \09%rd5, [main_kernel_param_0];\0A\09mov.b64 \09%rd7, bufferLhsGlobal;\0A\09cvt.u32.u64 \09%r5, %rd7;\0A\09cvt.u32.u64 \09%r7, %rd29;\0A\09mov.b32 \09%r6, 0;\0A\09// begin inline asm\0A\09cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r5], [%rd5, {%r6,%r6} ], [%r7];\0A\09// end inline asm\0A\09mov.b64 \09%rd8, bufferRhsGlobal;\0A\09cvt.u32.u64 \09%r8, %rd8;\0A\09// begin inline asm\0A\09cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r8], [%rd6, {%r6,%r6} ], [%r7];\0A\09// end inline asm\0A\09mov.b32 \09%r9, 6144;\0A\09// begin inline asm\0A\09mbarrier.arrive.expect_tx.shared.b64 _, [%r7], %r9;\0A\09// end inline asm\0A$L__BB0_3:\0A\09cvt.u32.u64 \09%r10, %rd29;\0A\09mov.b32 \09%r11, 0;\0A\09mov.b32 \09%r12, 10000000;\0A\09// begin inline asm\0A\09{\0A\09.reg .pred       P1; \0A\09LAB_WAIT: \0A\09mbarrier.try_wait.parity.shared.b64 P1, [%r10], %r11, %r12; \0A\09@P1 bra.uni DONE; \0A\09bra.uni     LAB_WAIT; \0A\09DONE: \0A\09}\0A\09// end inline asm\0A\09setp.ne.b64 \09%p2, %rd2, %rd1;\0A\09@%p2 bra \09$L__BB0_5;\0A\09ld.param.b64 \09%rd4, [main_kernel_param_4];\0A\09ld.param.b64 \09%rd3, [main_kernel_param_3];\0A\09shl.b64 \09%rd9, %rd3, 5;\0A\09mov.b64 \09%rd10, bufferLhsGlobal;\0A\09add.s64 \09%rd11, %rd10, %rd9;\0A\09shl.b64 \09%rd12, %rd4, 2;\0A\09add.s64 \09%rd13, %rd11, %rd12;\0A\09ld.shared.b32 \09%r13, [%rd13];\0A\09shl.b64 \09%rd14, %rd4, 9;\0A\09mov.b64 \09%rd15, bufferRhsGlobal;\0A\09add.s64 \09%rd16, %rd15, %rd14;\0A\09shl.b64 \09%rd17, %rd2, 2;\0A\09add.s64 \09%rd18, %rd16, %rd17;\0A\09ld.shared.b32 \09%r14, [%rd18];\0A\09cvt.f64.f32 \09%rd19, %r13;\0A\09add.u64 \09%rd20, %SP, 0;\0A\09add.u64 \09%rd21, %SPL, 0;\0A\09st.local.b64 \09[%rd21], %rd19;\0A\09{ // callseq 0, 0\0A\09.param .b64 \09param0;\0A\09.param .b64 \09param1;\0A\09.param .b32 \09retval0;\0A\09st.param.b64 \09[param1], %rd20;\0A\09mov.b64 \09%rd22, printfFormat_0;\0A\09cvta.global.u64 \09%rd23, %rd22;\0A\09st.param.b64 \09[param0], %rd23;\0A\09call.uni (retval0), vprintf, (param0, param1);\0A\09} // callseq 0\0A\09cvt.f64.f32 \09%rd24, %r14;\0A\09add.u64 \09%rd25, %SP, 8;\0A\09add.u64 \09%rd26, %SPL, 8;\0A\09st.local.b64 \09[%rd26], %rd24;\0A\09{ // callseq 1, 0\0A\09.param .b64 \09param0;\0A\09.param .b64 \09param1;\0A\09.param .b32 \09retval0;\0A\09st.param.b64 \09[param1], %rd25;\0A\09mov.b64 \09%rd27, printfFormat_1;\0A\09cvta.global.u64 \09%rd28, %rd27;\0A\09st.param.b64 \09[param0], %rd28;\0A\09call.uni (retval0), vprintf, (param0, param1);\0A\09} // callseq 1\0A$L__BB0_5:\0A\09ret;\0A\0A}\0A">]
  module attributes {transform.with_named_sequence} {
  }
  llvm.func @mgpuTensorMapEncodeTiledMemref(i64, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> !llvm.ptr
  llvm.func @mgpuStreamCreate() -> !llvm.ptr
  llvm.func @mgpuMemAlloc(i64, !llvm.ptr, i8) -> !llvm.ptr
  llvm.func @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
}

