/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: SimpleArith.td                                                       *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::simple::SimpleArithDialect)
namespace mlir {
namespace simple {

SimpleArithDialect::SimpleArithDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<SimpleArithDialect>())
    
     {
  
  initialize();
}

SimpleArithDialect::~SimpleArithDialect() = default;

} // namespace simple
} // namespace mlir
