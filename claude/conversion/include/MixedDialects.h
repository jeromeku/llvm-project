//===- MixedDialects.h - Mixed Dialect Declarations -------------*- C++ -*-===//
//
// Mixed high/medium dialect declarations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MIXED_DIALECTS_H
#define MLIR_MIXED_DIALECTS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "MixedDialects.h.inc"

#define GET_OP_CLASSES
#include "MixedDialectsOps.h.inc"

namespace mlir {
namespace high {

std::unique_ptr<Pass> createPartialHighToMediumPass();
std::unique_ptr<Pass> createCompleteMediumToLowPass();
std::unique_ptr<Pass> createAnalyzeLoweringFeasibilityPass();
void registerMixedDialectPasses();

} // namespace high
} // namespace mlir

#endif // MLIR_MIXED_DIALECTS_H
