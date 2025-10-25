//===- SimpleArith.td - Simple Arithmetic Dialect ---------*- tablegen -*-===//
//
// Simple arithmetic dialect for demonstrating basic dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef SIMPLE_ARITH_DIALECT
#define SIMPLE_ARITH_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//

def SimpleArith_Dialect : Dialect {
  let name = "simple";
  let summary = "A simple arithmetic dialect for tutorial purposes";
  let description = [{
    This dialect provides basic arithmetic operations to demonstrate
    MLIR's dialect conversion infrastructure. Operations are intentionally
    simple to focus on conversion mechanics rather than complex semantics.
  }];
  let cppNamespace = "::mlir::simple";
}

//===----------------------------------------------------------------------===//
// Base Operation Definition
//===----------------------------------------------------------------------===//

class SimpleArith_Op<string mnemonic, list<Trait> traits = []> :
    Op<SimpleArith_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

// Base class for binary arithmetic operations
class BinaryArithOp<string mnemonic, list<Trait> traits = []> :
    SimpleArith_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def AddOp : BinaryArithOp<"add"> {
  let summary = "addition operation";
  let description = [{
    The `simple.add` operation adds two values.

    Example:
    ```mlir
    %result = simple.add %lhs, %rhs : i32
    ```

    This operation will be converted to `arith.addi` or `arith.addf`
    depending on the operand types.
  }];
}

def SubOp : BinaryArithOp<"sub"> {
  let summary = "subtraction operation";
  let description = [{
    The `simple.sub` operation subtracts the second operand from the first.

    Example:
    ```mlir
    %result = simple.sub %lhs, %rhs : f32
    ```

    This operation will be converted to `arith.subi` or `arith.subf`
    depending on the operand types.
  }];
}

def MulOp : BinaryArithOp<"mul"> {
  let summary = "multiplication operation";
  let description = [{
    The `simple.mul` operation multiplies two values.

    Example:
    ```mlir
    %result = simple.mul %lhs, %rhs : i64
    ```

    This operation will be converted to `arith.muli` or `arith.mulf`
    depending on the operand types.
  }];
}

def DivOp : BinaryArithOp<"div"> {
  let summary = "division operation";
  let description = [{
    The `simple.div` operation divides the first operand by the second.

    Example:
    ```mlir
    %result = simple.div %lhs, %rhs : f64
    ```

    For integers, this converts to `arith.divsi` (signed division).
    For floats, this converts to `arith.divf`.
  }];
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

def NegOp : SimpleArith_Op<"neg", [Pure, SameOperandsAndResultType]> {
  let summary = "negation operation";
  let description = [{
    The `simple.neg` operation negates a value (computes -x).

    Example:
    ```mlir
    %result = simple.neg %operand : i32
    ```

    This is lowered to `arith.subi` (0 - operand) for integers
    or `arith.negf` for floats.
  }];

  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Constant Operation
//===----------------------------------------------------------------------===//

def ConstantOp : SimpleArith_Op<"constant", [Pure, ConstantLike]> {
  let summary = "constant value operation";
  let description = [{
    The `simple.constant` operation produces a constant value.

    Example:
    ```mlir
    %c = simple.constant 42 : i32
    %f = simple.constant 3.14 : f32
    ```

    This is converted to `arith.constant`.
  }];
 let arguments = (ins
    TypedAttrInterface:$value
  );

  // Result can be any signless integer or float scalar.
  let results = (outs AnyTypeOf<[AnySignlessInteger, AnyFloat]>:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

#endif // SIMPLE_ARITH_DIALECT
