/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TypedArith.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::typed::TypedArithDialect)
namespace mlir {
namespace typed {

TypedArithDialect::TypedArithDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<TypedArithDialect>())
    
     {
  
  initialize();
}

TypedArithDialect::~TypedArithDialect() = default;

} // namespace typed
} // namespace mlir
