//===- TypedArith.h - Typed Arithmetic Dialect -----------------*- C++ -*-===//
//
// Typed arithmetic dialect declaration.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TYPED_ARITH_H
#define MLIR_TYPED_ARITH_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TypedArith.h.inc"

#define GET_TYPEDEF_CLASSES
#include "TypedArithTypes.h.inc"

#define GET_OP_CLASSES
#include "TypedArithOps.h.inc"

namespace mlir {
namespace typed {

std::unique_ptr<Pass> createConvertTypedArithToStdPass();
void registerConvertTypedArithToStdPass();

} // namespace typed
} // namespace mlir

#endif // MLIR_TYPED_ARITH_H
