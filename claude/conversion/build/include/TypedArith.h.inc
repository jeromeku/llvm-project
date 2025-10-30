/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TypedArith.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace typed {

/// Arithmetic dialect with custom types for conversion demo
/// This dialect introduces custom numeric types that must be converted
///     to standard MLIR types. It demonstrates:
///     - Custom type definitions
///     - TypeConverter usage
///     - Type materialization
///     - Signature conversion
/// 
///     Custom types:
///     - FixedPoint<width, scale>: Fixed-point numbers
///     - Complex<element>: Complex numbers
class TypedArithDialect : public ::mlir::Dialect {
  explicit TypedArithDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~TypedArithDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("typed");
  }

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;
};
} // namespace typed
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::typed::TypedArithDialect)
