func.func @example () -> i32 {
    %a = arith.constant 5 : i32
    %a1 = arith.constant 4: i32
    %b = arith.addi %a, %a1 : i32
    %c = arith.addi %a, %a : i32
    %d = arith.addi %b, %c : i32
    return %d : i32
}
