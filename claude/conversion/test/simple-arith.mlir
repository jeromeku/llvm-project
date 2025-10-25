// RUN: simple-opt %s -convert-simple-to-arith | FileCheck %s

// This test demonstrates the conversion of simple arithmetic operations
// to the standard arith dialect.

// CHECK-LABEL: func @test_integer_ops
func.func @test_integer_ops(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[ADD:.*]] = arith.addi %arg0, %arg1 : i32
  %0 = simple.add %arg0, %arg1 : i32

  // CHECK: %[[SUB:.*]] = arith.subi %[[ADD]], %arg1 : i32
  %1 = simple.sub %0, %arg1 : i32

  // CHECK: %[[MUL:.*]] = arith.muli %[[SUB]], %arg0 : i32
  %2 = simple.mul %1, %arg0 : i32

  // CHECK: %[[DIV:.*]] = arith.divsi %[[MUL]], %arg1 : i32
  %3 = simple.div %2, %arg1 : i32

  // CHECK: return %[[DIV]] : i32
  return %3 : i32
}

// CHECK-LABEL: func @test_float_ops
func.func @test_float_ops(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: %[[ADD:.*]] = arith.addf %arg0, %arg1 : f32
  %0 = simple.add %arg0, %arg1 : f32

  // CHECK: %[[SUB:.*]] = arith.subf %[[ADD]], %arg1 : f32
  %1 = simple.sub %0, %arg1 : f32

  // CHECK: %[[MUL:.*]] = arith.mulf %[[SUB]], %arg0 : f32
  %2 = simple.mul %1, %arg0 : f32

  // CHECK: %[[DIV:.*]] = arith.divf %[[MUL]], %arg1 : f32
  %3 = simple.div %2, %arg1 : f32

  // CHECK: return %[[DIV]] : f32
  return %3 : f32
}

// CHECK-LABEL: func @test_negation
func.func @test_negation(%arg0: i32, %arg1: f32) -> (i32, f32) {
  // Integer negation: -x becomes (0 - x)
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[NEG_INT:.*]] = arith.subi %[[C0]], %arg0 : i32
  %0 = simple.neg %arg0 : i32

  // Float negation: -x becomes arith.negf
  // CHECK: %[[NEG_FLOAT:.*]] = arith.negf %arg1 : f32
  %1 = simple.neg %arg1 : f32

  // CHECK: return %[[NEG_INT]], %[[NEG_FLOAT]] : i32, f32
  return %0, %1 : i32, f32
}

// CHECK-LABEL: func @test_constants
func.func @test_constants() -> (i32, f64) {
  // CHECK: %[[C42:.*]] = arith.constant 42 : i32
  %0 = simple.constant 42 : i32

  // CHECK: %[[C_PI:.*]] = arith.constant 3.140000e+00 : f64
  %1 = simple.constant 3.14 : f64

  // CHECK: return %[[C42]], %[[C_PI]] : i32, f64
  return %0, %1 : i32, f64
}

// CHECK-LABEL: func @test_complex_expression
func.func @test_complex_expression(%a: i64, %b: i64, %c: i64) -> i64 {
  // Expression: -(a + b) * (c - a)

  // CHECK: %[[ADD:.*]] = arith.addi %arg0, %arg1 : i64
  %0 = simple.add %a, %b : i64

  // CHECK: %[[C0:.*]] = arith.constant 0 : i64
  // CHECK: %[[NEG:.*]] = arith.subi %[[C0]], %[[ADD]] : i64
  %1 = simple.neg %0 : i64

  // CHECK: %[[SUB:.*]] = arith.subi %arg2, %arg0 : i64
  %2 = simple.sub %c, %a : i64

  // CHECK: %[[MUL:.*]] = arith.muli %[[NEG]], %[[SUB]] : i64
  %3 = simple.mul %1, %2 : i64

  // CHECK: return %[[MUL]] : i64
  return %3 : i64
}
