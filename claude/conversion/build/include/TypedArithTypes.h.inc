/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace typed {
/// Fixed-point numeric type
/// A fixed-point type with specified width and scale.
///     - width: Total number of bits
///     - scale: Number of fractional bits
/// 
///     Example: `!typed.fixed<16, 8>` represents Q8.8 fixed-point
///     (8 integer bits, 8 fractional bits)
/// 
///     Conversion: FixedPoint<width, scale> → iWidth
class FixedPointType;
/// Complex number type
/// A complex number type with real and imaginary components of the given element type.
/// 
///     Example: `!typed.complex<f32>` represents complex numbers with f32 components
/// 
///     Conversion: Complex<T> → Two separate values of type T
///     (or a struct containing two T values, depending on lowering strategy)
class ComplexType;
namespace detail {
struct FixedPointTypeStorage;
} // namespace detail
class FixedPointType : public ::mlir::Type::TypeBase<FixedPointType, ::mlir::Type, detail::FixedPointTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "typed.fixed";
  static constexpr ::llvm::StringLiteral dialectName = "typed";
  using Base::getChecked;
  static FixedPointType get(::mlir::MLIRContext *context, unsigned width, unsigned scale);
  static FixedPointType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned width, unsigned scale);
  static FixedPointType get(unsigned width, unsigned scale);
  static FixedPointType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned width, unsigned scale);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned width, unsigned scale);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned width, unsigned scale);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"fixed"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getWidth() const;
  unsigned getScale() const;
};
namespace detail {
struct ComplexTypeStorage;
} // namespace detail
class ComplexType : public ::mlir::Type::TypeBase<ComplexType, ::mlir::Type, detail::ComplexTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "typed.complex";
  static constexpr ::llvm::StringLiteral dialectName = "typed";
  static ComplexType get(::mlir::MLIRContext *context, Type elementType);
  static ComplexType get(Type elementType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"complex"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getElementType() const;
};
} // namespace typed
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::typed::FixedPointType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::typed::ComplexType)

#endif  // GET_TYPEDEF_CLASSES

