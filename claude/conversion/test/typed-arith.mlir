// RUN: typed-opt %s -convert-typed-to-std | FileCheck %s

// This test demonstrates type conversion with the TypedArith dialect.
// Focus on how custom types (FixedPoint) are converted to standard types.

// CHECK-LABEL: func @test_fixed_add
func.func @test_fixed_add(%arg0: !typed.fixed<16, 8>, %arg1: !typed.fixed<16, 8>) -> !typed.fixed<16, 8> {
  // Fixed-point addition converts to integer addition
  // The scale (8) doesn't affect addition, so it's just:
  //   result = lhs + rhs

  // CHECK: %[[ADD:.*]] = arith.addi %arg0, %arg1 : i16
  %0 = typed.fixed_add %arg0, %arg1 : !typed.fixed<16, 8>

  // CHECK: return %[[ADD]] : i16
  return %0 : !typed.fixed<16, 8>
}

// CHECK-LABEL: func @test_fixed_mul
func.func @test_fixed_mul(%arg0: !typed.fixed<16, 8>, %arg1: !typed.fixed<16, 8>) -> !typed.fixed<16, 8> {
  // Fixed-point multiplication requires scale adjustment:
  //   result = (lhs * rhs) >> scale

  // CHECK: %[[MUL:.*]] = arith.muli %arg0, %arg1 : i16
  // CHECK: %[[SCALE:.*]] = arith.constant 8 : i16
  // CHECK: %[[RESULT:.*]] = arith.shrsi %[[MUL]], %[[SCALE]] : i16
  %0 = typed.fixed_mul %arg0, %arg1 : !typed.fixed<16, 8>

  // CHECK: return %[[RESULT]] : i16
  return %0 : !typed.fixed<16, 8>
}

// CHECK-LABEL: func @test_fixed_constant
func.func @test_fixed_constant() -> !typed.fixed<16, 8> {
  // Constant 896 in Q8.8 format represents 3.5
  // (896 / 256 = 3.5)

  // CHECK: %[[C:.*]] = arith.constant 896 : i16
  %0 = typed.fixed_constant 896 : !typed.fixed<16, 8>

  // CHECK: return %[[C]] : i16
  return %0 : !typed.fixed<16, 8>
}

// CHECK-LABEL: func @test_int_to_fixed
func.func @test_int_to_fixed(%arg0: i32) -> !typed.fixed<32, 16> {
  // Convert integer to fixed-point: left shift by scale
  // Example: 7 in Q16.16 = 7 << 16 = 458752

  // CHECK: %[[SCALE:.*]] = arith.constant 16 : i32
  // CHECK: %[[RESULT:.*]] = arith.shli %arg0, %[[SCALE]] : i32
  %0 = typed.int_to_fixed %arg0 : i32 -> !typed.fixed<32, 16>

  // CHECK: return %[[RESULT]] : i32
  return %0 : !typed.fixed<32, 16>
}

// CHECK-LABEL: func @test_fixed_to_int
func.func @test_fixed_to_int(%arg0: !typed.fixed<32, 16>) -> i32 {
  // Convert fixed-point to integer: right shift by scale
  // Example: 458752 (=7.0 in Q16.16) >> 16 = 7

  // CHECK: %[[SCALE:.*]] = arith.constant 16 : i32
  // CHECK: %[[RESULT:.*]] = arith.shrsi %arg0, %[[SCALE]] : i32
  %0 = typed.fixed_to_int %arg0 : !typed.fixed<32, 16> -> i32

  // CHECK: return %[[RESULT]] : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_complex_fixed_expression
func.func @test_complex_fixed_expression(%a: !typed.fixed<16, 8>, %b: !typed.fixed<16, 8>) -> !typed.fixed<16, 8> {
  // Compute: (a + b) * (a - a)
  // This will result in: (a + b) * 0 = 0

  // Constants for zero
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i16

  %0 = typed.fixed_add %a, %b : !typed.fixed<16, 8>
  // CHECK: %[[ADD:.*]] = arith.addi %arg0, %arg1 : i16

  // Note: a - a would typically be optimized away, but we'll show the conversion
  %1 = typed.fixed_add %a, %a : !typed.fixed<16, 8>
  // CHECK: %[[ADD2:.*]] = arith.addi %arg0, %arg0 : i16

  %2 = typed.fixed_mul %0, %1 : !typed.fixed<16, 8>
  // CHECK: %[[MUL:.*]] = arith.muli %[[ADD]], %[[ADD2]] : i16
  // CHECK: %[[SCALE:.*]] = arith.constant 8 : i16
  // CHECK: %[[RESULT:.*]] = arith.shrsi %[[MUL]], %[[SCALE]] : i16

  return %2 : !typed.fixed<16, 8>
}

// CHECK-LABEL: func @test_mixed_int_fixed
func.func @test_mixed_int_fixed(%int_val: i32) -> !typed.fixed<32, 16> {
  // Convert integer to fixed, multiply, convert back

  %fp1 = typed.int_to_fixed %int_val : i32 -> !typed.fixed<32, 16>
  // CHECK: %[[SCALE1:.*]] = arith.constant 16 : i32
  // CHECK: %[[FP1:.*]] = arith.shli %arg0, %[[SCALE1]] : i32

  %fp2 = typed.fixed_constant 65536 : !typed.fixed<32, 16>  // 1.0 in Q16.16
  // CHECK: %[[C1:.*]] = arith.constant 65536 : i32

  %result = typed.fixed_mul %fp1, %fp2 : !typed.fixed<32, 16>
  // CHECK: %[[MUL:.*]] = arith.muli %[[FP1]], %[[C1]] : i32
  // CHECK: %[[SCALE2:.*]] = arith.constant 16 : i32
  // CHECK: %[[RESULT:.*]] = arith.shrsi %[[MUL]], %[[SCALE2]] : i32

  return %result : !typed.fixed<32, 16>
}

// Demonstrate signature conversion
// The function signature itself will be converted: FixedPoint → Integer

// CHECK-LABEL: func @signature_conversion(%arg0: i16, %arg1: i16) -> i16
func.func @signature_conversion(%arg0: !typed.fixed<16, 8>, %arg1: !typed.fixed<16, 8>) -> !typed.fixed<16, 8> {
  // Inside the function body, operations work with converted types

  // CHECK: %[[ADD:.*]] = arith.addi %arg0, %arg1 : i16
  %0 = typed.fixed_add %arg0, %arg1 : !typed.fixed<16, 8>

  // CHECK: return %[[ADD]] : i16
  return %0 : !typed.fixed<16, 8>
}
