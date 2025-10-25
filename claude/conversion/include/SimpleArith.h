//===- SimpleArith.h - Simple Arithmetic Dialect ---------------*- C++ -*-===//
//
// Simple arithmetic dialect declaration.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SIMPLE_ARITH_H
#define MLIR_SIMPLE_ARITH_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the generated dialect declaration
#include "SimpleArith.h.inc"

// Declare the generated operation classes
#define GET_OP_CLASSES
#include "SimpleArithOps.h.inc"

namespace mlir {
namespace simple {

// Forward declare pass creation function
std::unique_ptr<Pass> createConvertSimpleArithToArithPass();

// Pass registration function
void registerConvertSimpleArithToArithPass();

} // namespace simple
} // namespace mlir

#endif // MLIR_SIMPLE_ARITH_H
