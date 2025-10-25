//===- TypedArith.td - Typed Arithmetic Dialect -----------*- tablegen -*-===//
//
// Arithmetic dialect with custom types to demonstrate type conversion.
//
//===----------------------------------------------------------------------===//

#ifndef TYPED_ARITH_DIALECT
#define TYPED_ARITH_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CastInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//

def TypedArith_Dialect : Dialect {
  let name = "typed";
  let summary = "Arithmetic dialect with custom types for conversion demo";
  let description = [{
    This dialect introduces custom numeric types that must be converted
    to standard MLIR types. It demonstrates:
    - Custom type definitions
    - TypeConverter usage
    - Type materialization
    - Signature conversion

    Custom types:
    - FixedPoint<width, scale>: Fixed-point numbers
    - Complex<element>: Complex numbers
  }];
  let cppNamespace = "::mlir::typed";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

class TypedArith_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TypedArith_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Fixed-point type: represents numbers with fixed decimal places
// Example: FixedPoint<16, 8> means 16-bit value with 8 fractional bits
// This will be converted to standard integer types during lowering
def FixedPointType : TypedArith_Type<"FixedPoint", "fixed"> {
  let summary = "Fixed-point numeric type";
  let description = [{
    A fixed-point type with specified width and scale.
    - width: Total number of bits
    - scale: Number of fractional bits

    Example: `!typed.fixed<16, 8>` represents Q8.8 fixed-point
    (8 integer bits, 8 fractional bits)

    Conversion: FixedPoint<width, scale> → iWidth
  }];

  let parameters = (ins "unsigned":$width, "unsigned":$scale);

  let assemblyFormat = "`<` $width `,` $scale `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "unsigned":$width, "unsigned":$scale), [{
      return $_get(nullptr, width, scale);
    }]>
  ];

  // Add verification to ensure scale <= width
  let genVerifyDecl = 1;
}

// Complex number type: represents complex numbers (real + imaginary)
// This will be converted to a struct-like representation or two separate values
def ComplexType : TypedArith_Type<"Complex", "complex"> {
  let summary = "Complex number type";
  let description = [{
    A complex number type with real and imaginary components of the given element type.

    Example: `!typed.complex<f32>` represents complex numbers with f32 components

    Conversion: Complex<T> → Two separate values of type T
    (or a struct containing two T values, depending on lowering strategy)
  }];

  let parameters = (ins "Type":$elementType);

  let assemblyFormat = "`<` $elementType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Base Operation Definition
//===----------------------------------------------------------------------===//

class TypedArith_Op<string mnemonic, list<Trait> traits = []> :
    Op<TypedArith_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FixedPoint Operations
//===----------------------------------------------------------------------===//

def FixedAddOp : TypedArith_Op<"fixed_add", [Pure, SameOperandsAndResultType]> {
  let summary = "Fixed-point addition";
  let description = [{
    Adds two fixed-point numbers.

    Example:
    ```mlir
    %result = typed.fixed_add %lhs, %rhs : !typed.fixed<16, 8>
    ```

    Lowering: Convert to integer addition (operands already aligned by scale)
  }];

  let arguments = (ins FixedPointType:$lhs, FixedPointType:$rhs);
  let results = (outs FixedPointType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def FixedMulOp : TypedArith_Op<"fixed_mul", [Pure]> {
  let summary = "Fixed-point multiplication";
  let description = [{
    Multiplies two fixed-point numbers.

    Example:
    ```mlir
    %result = typed.fixed_mul %lhs, %rhs : !typed.fixed<16, 8>
    ```

    Lowering: Multiply integers, then right-shift by scale to normalize
    If a * b where both have scale S, result is (a * b) >> S
  }];

  let arguments = (ins FixedPointType:$lhs, FixedPointType:$rhs);
  let results = (outs FixedPointType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def FixedConstantOp : TypedArith_Op<"fixed_constant", [Pure, ConstantLike]> {
  let summary = "Fixed-point constant";
  let description = [{
    Creates a fixed-point constant.

    Example:
    ```mlir
    // Represents 3.5 in Q8.8 format (3.5 * 256 = 896)
    %c = typed.fixed_constant 896 : !typed.fixed<16, 8>
    ```

    Lowering: Convert to integer constant
  }];

  let arguments = (ins I64Attr:$value);
  let results = (outs FixedPointType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Complex Number Operations
//===----------------------------------------------------------------------===//

def ComplexCreateOp : TypedArith_Op<"complex_create", [Pure]> {
  let summary = "Create complex number from real and imaginary parts";
  let description = [{
    Creates a complex number from real and imaginary components.

    Example:
    ```mlir
    %c = typed.complex_create %real, %imag : f32 -> !typed.complex<f32>
    ```

    Lowering: Package two values into complex representation
  }];

  let arguments = (ins AnyType:$real, AnyType:$imag);
  let results = (outs ComplexType:$result);
  let assemblyFormat = "$real `,` $imag attr-dict `:` type($real) `->` type($result)";

  let hasVerifier = 1;
}

def ComplexRealOp : TypedArith_Op<"complex_real", [Pure]> {
  let summary = "Extract real part of complex number";
  let description = [{
    Extracts the real component of a complex number.

    Example:
    ```mlir
    %real = typed.complex_real %c : !typed.complex<f32> -> f32
    ```

    Lowering: Extract first element from complex representation
  }];

  let arguments = (ins ComplexType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def ComplexImagOp : TypedArith_Op<"complex_imag", [Pure]> {
  let summary = "Extract imaginary part of complex number";
  let description = [{
    Extracts the imaginary component of a complex number.

    Example:
    ```mlir
    %imag = typed.complex_imag %c : !typed.complex<f32> -> f32
    ```

    Lowering: Extract second element from complex representation
  }];

  let arguments = (ins ComplexType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def ComplexAddOp : TypedArith_Op<"complex_add", [Pure, SameOperandsAndResultType]> {
  let summary = "Complex number addition";
  let description = [{
    Adds two complex numbers: (a+bi) + (c+di) = (a+c) + (b+d)i

    Example:
    ```mlir
    %result = typed.complex_add %lhs, %rhs : !typed.complex<f32>
    ```

    Lowering: Extract components, add separately, reconstruct
  }];

  let arguments = (ins ComplexType:$lhs, ComplexType:$rhs);
  let results = (outs ComplexType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def ComplexMulOp : TypedArith_Op<"complex_mul", [Pure, SameOperandsAndResultType]> {
  let summary = "Complex number multiplication";
  let description = [{
    Multiplies two complex numbers: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i

    Example:
    ```mlir
    %result = typed.complex_mul %lhs, %rhs : !typed.complex<f32>
    ```

    Lowering: Extract components, compute real and imaginary parts, reconstruct
  }];

  let arguments = (ins ComplexType:$lhs, ComplexType:$rhs);
  let results = (outs ComplexType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Conversion/Cast Operations
//===----------------------------------------------------------------------===//

def IntToFixedOp : TypedArith_Op<"int_to_fixed",
    [Pure, DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Convert integer to fixed-point";
  let description = [{
    Converts an integer to fixed-point representation by scaling.

    Example:
    ```mlir
    // Convert integer 7 to fixed-point (7 << 8 = 1792 for Q8.8)
    %fp = typed.int_to_fixed %int : i32 -> !typed.fixed<16, 8>
    ```
  }];

  let arguments = (ins AnyInteger:$input);
  let results = (outs FixedPointType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def FixedToIntOp : TypedArith_Op<"fixed_to_int",
    [Pure, DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Convert fixed-point to integer";
  let description = [{
    Converts a fixed-point number to integer by descaling (rounding down).

    Example:
    ```mlir
    // Convert Q8.8 value to integer (right shift by 8)
    %int = typed.fixed_to_int %fp : !typed.fixed<16, 8> -> i32
    ```
  }];

  let arguments = (ins FixedPointType:$input);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

#endif // TYPED_ARITH_DIALECT
