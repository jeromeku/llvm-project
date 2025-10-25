//===- partial-opt.cpp - Mixed dialects optimizer ------------------------===//
//
// Tool for testing partial conversion with mixed dialects.
//
//===----------------------------------------------------------------------===//

#include "MixedDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::registerAllPasses();
  mlir::high::registerMixedDialectPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::high::HighDialect, mlir::medium::MediumDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Partial conversion optimizer\n", registry));
}
