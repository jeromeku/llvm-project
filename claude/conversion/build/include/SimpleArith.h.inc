/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: SimpleArith.td                                                       *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace simple {

/// A simple arithmetic dialect for tutorial purposes
/// This dialect provides basic arithmetic operations to demonstrate
///     MLIR's dialect conversion infrastructure. Operations are intentionally
///     simple to focus on conversion mechanics rather than complex semantics.
class SimpleArithDialect : public ::mlir::Dialect {
  explicit SimpleArithDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~SimpleArithDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("simple");
  }
};
} // namespace simple
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::simple::SimpleArithDialect)
