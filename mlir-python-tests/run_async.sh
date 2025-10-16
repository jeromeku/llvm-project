 mlir-opt async.mlir \
     -transform-interpreter \
     -test-transform-dialect-erase-schedule \
     -convert-nvgpu-to-nvvm -gpu-kernel-outlining \
     -convert-scf-to-cf -convert-nvvm-to-llvm \
     -convert-vector-to-llvm \
     -convert-math-to-llvm \
     -expand-strided-metadata \
     -lower-affine \
     -convert-index-to-llvm=index-bitwidth=32 \
     -convert-arith-to-llvm \
     -finalize-memref-to-llvm \
     -convert-func-to-llvm \
     -canonicalize \
     -expand-strided-metadata --nvvm-attach-target="module=main_kernel features=+ptx80 chip=sm_90 O=3" > async.pass1.mlir
#strip-debuginfo,
mlir-opt async.pass1.mlir \
-pass-pipeline='builtin.module(gpu.module(convert-gpu-to-nvvm,convert-index-to-llvm{index-bitwidth=32},canonicalize,cse))' > async.pass2.mlir 


mlir-opt async.pass2.mlir --gpu-to-llvm --gpu-module-to-binary=format=isa -canonicalize -cse -reconcile-unrealized-casts -debug-only=serialize-to-isa 2> async.ptx 1> async.pass3.mlir
