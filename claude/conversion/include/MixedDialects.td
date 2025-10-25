//===- MixedDialects.td - Mixed Dialect Operations ---------*- tablegen -*-===//
//
// Multiple dialect levels to demonstrate partial conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MIXED_DIALECTS
#define MIXED_DIALECTS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// High-Level Dialect
//===----------------------------------------------------------------------===//

def High_Dialect : Dialect {
  let name = "high";
  let summary = "High-level operations for partial conversion demo";
  let description = [{
    This dialect represents high-level operations that should be
    progressively lowered through medium-level to low-level representations.

    Demonstrates:
    - Partial conversion (not all ops need to be legal immediately)
    - Dynamic legality (conditional based on operand properties)
    - Multi-stage lowering pipelines
  }];
  let cppNamespace = "::mlir::high";
}

class High_Op<string mnemonic, list<Trait> traits = []> :
    Op<High_Dialect, mnemonic, traits>;

def MatMulOp : High_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication";
  let description = [{
    High-level matrix multiplication operation.

    Example:
    ```mlir
    %C = high.matmul %A, %B : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    ```

    Lowering strategy:
    - High: matmul
    - Medium: loops with dot products
    - Low: individual multiply-add operations
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def DotOp : High_Op<"dot", [Pure]> {
  let summary = "Dot product of vectors";
  let description = [{
    Computes dot product of two vectors.

    Example:
    ```mlir
    %result = high.dot %a, %b : (tensor<8xf32>, tensor<8xf32>) -> f32
    ```

    Lowering: Expand to sum of element-wise multiplications
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyFloat:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def SumOp : High_Op<"sum", [Pure]> {
  let summary = "Sum all elements of a tensor";
  let description = [{
    Reduces a tensor to a scalar by summing all elements.

    Example:
    ```mlir
    %result = high.sum %input : tensor<8xf32> -> f32
    ```

    Lowering: Expand to sequential additions
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyFloat:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $result)";
}

//===----------------------------------------------------------------------===//
// Medium-Level Dialect
//===----------------------------------------------------------------------===//

def Medium_Dialect : Dialect {
  let name = "medium";
  let summary = "Medium-level operations (intermediate representation)";
  let description = [{
    This dialect represents a middle layer between high-level operations
    and low-level scalar operations. Think of this as similar to "affine"
    or "scf" dialects in MLIR.
  }];
  let cppNamespace = "::mlir::medium";
}

class Medium_Op<string mnemonic, list<Trait> traits = []> :
    Op<Medium_Dialect, mnemonic, traits>;

def LoopOp : Medium_Op<"loop", []> {
  let summary = "Simple loop operation";
  let description = [{
    A simplified loop construct for demonstration.

    Example:
    ```mlir
    medium.loop %i = 0 to 10 step 1 {
      // body
      medium.yield
    }
    ```

    Lowering: Convert to control flow (branches)
  }];

  let arguments = (ins Index:$lowerBound, Index:$upperBound, Index:$step);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$lowerBound `to` $upperBound `step` $step $body attr-dict";
}

def YieldOp : Medium_Op<"yield", [Pure, Terminator]> {
  let summary = "Yield operation for loop termination";
  let assemblyFormat = "attr-dict";
}

def ExtractOp : Medium_Op<"extract", [Pure]> {
  let summary = "Extract element from tensor";
  let description = [{
    Extract a single element from a tensor at the given indices.

    Example:
    ```mlir
    %elem = medium.extract %tensor[%i, %j] : tensor<4x8xf32>
    ```
  }];

  let arguments = (ins AnyTensor:$tensor, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$tensor `[` $indices `]` attr-dict `:` type($tensor)";
}

def InsertOp : Medium_Op<"insert", [Pure]> {
  let summary = "Insert element into tensor";
  let description = [{
    Insert a value into a tensor at the given indices, returning a new tensor.

    Example:
    ```mlir
    %new_tensor = medium.insert %value into %tensor[%i, %j] : tensor<4x8xf32>
    ```
  }];

  let arguments = (ins AnyType:$value, AnyTensor:$tensor, Variadic<Index>:$indices);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$value `into` $tensor `[` $indices `]` attr-dict `:` type($tensor)";
}

def AllocOp : Medium_Op<"alloc", []> {
  let summary = "Allocate a tensor";
  let description = [{
    Allocate storage for a tensor.

    Example:
    ```mlir
    %tensor = medium.alloc : tensor<4x8xf32>
    ```
  }];

  let results = (outs AnyTensor:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

#endif // MIXED_DIALECTS
