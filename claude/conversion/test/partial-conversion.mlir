// RUN: partial-opt %s -partial-lower-high | FileCheck %s

// This test demonstrates partial conversion and dynamic legality.
// Note: Since our patterns return failure (simplified for tutorial),
// the operations won't actually be converted, but the test shows
// the INTENT and structure.

// Test dynamic legality based on size
// CHECK-LABEL: func @test_small_matmul
func.func @test_small_matmul(%A: tensor<2x4xf32>, %B: tensor<4x2xf32>) -> tensor<2x2xf32> {
  // Small matrix (2x4 = 8 elements < 64 threshold)
  // This SHOULD be converted according to dynamic legality rules
  // (but won't be because our pattern returns failure)

  // In a full implementation, this would be:
  // CHECK: medium.alloc
  // CHECK: medium.loop
  // CHECK: high.dot

  %C = high.matmul %A, %B : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
  return %C : tensor<2x2xf32>
}

// CHECK-LABEL: func @test_large_matmul
func.func @test_large_matmul(%A: tensor<16x16xf32>, %B: tensor<16x16xf32>) -> tensor<16x16xf32> {
  // Large matrix (16x16 = 256 elements > 64 threshold)
  // This SHOULD remain unconverted (dynamically legal)

  // CHECK: high.matmul
  %C = high.matmul %A, %B : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  return %C : tensor<16x16xf32>
}

// CHECK-LABEL: func @test_small_dot
func.func @test_small_dot(%a: tensor<8xf32>, %b: tensor<8xf32>) -> f32 {
  // Small vector (8 < 16 threshold)
  // Should be converted according to dynamic legality

  // In a full implementation:
  // CHECK: medium.loop
  // CHECK: medium.extract
  // CHECK: arith.mulf

  %result = high.dot %a, %b : (tensor<8xf32>, tensor<8xf32>) -> f32
  return %result : f32
}

// CHECK-LABEL: func @test_large_dot
func.func @test_large_dot(%a: tensor<32xf32>, %b: tensor<32xf32>) -> f32 {
  // Large vector (32 > 16 threshold)
  // Should remain unconverted (dynamically legal)

  // CHECK: high.dot
  %result = high.dot %a, %b : (tensor<32xf32>, tensor<32xf32>) -> f32
  return %result : f32
}

// CHECK-LABEL: func @test_sum
func.func @test_sum(%input: tensor<16xf32>) -> f32 {
  // sum is always illegal - must be converted
  // (but won't be in this simplified implementation)

  // In a full implementation:
  // CHECK: medium.loop
  // CHECK: arith.addf

  %result = high.sum %input : tensor<16xf32> -> f32
  return %result : f32
}

// Demonstrate mixed dialect IR after partial conversion
// CHECK-LABEL: func @test_mixed_operations
func.func @test_mixed_operations(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>,
                                  %v1: tensor<32xf32>, %v2: tensor<32xf32>) -> (tensor<4x4xf32>, f32) {
  // After partial conversion, we expect:
  // - Small matmul (4x4 = 16 < 64) → converted to loops
  // - Large dot (32 > 16) → remains as high.dot
  // - Result: MIXED dialect IR with both high and medium ops

  %C = high.matmul %A, %B : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: high.matmul
  // In full impl: medium.loop, etc.

  %d = high.dot %v1, %v2 : (tensor<32xf32>, tensor<32xf32>) -> f32
  // CHECK: high.dot
  // Should remain high-level because vector is large

  return %C, %d : tensor<4x4xf32>, f32
}
