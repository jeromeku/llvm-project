module {
  func.func @main(%arg0: index) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    gpu.launch blocks(%arg1, %arg2, %arg3) in (%arg7 = %c1, %arg8 = %c1_0, %arg9 = %c1_1) threads(%arg4, %arg5, %arg6) in (%arg10 = %c4, %arg11 = %c1_2, %arg12 = %c1_3) dynamic_shared_memory_size %c0_i32 {
      %thread_id_x = gpu.thread_id  x
      %0 = arith.addi %arg0, %thread_id_x : index
      gpu.printf "GPU thread %llu has %llu\0A", %thread_id_x, %0 : index, index
      gpu.terminator
    }
    return
  }
}

