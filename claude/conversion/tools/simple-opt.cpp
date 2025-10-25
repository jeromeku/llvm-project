//===- simple-opt.cpp - SimpleArith dialect optimizer --------------------===//
//
// Tool for testing SimpleArith dialect and conversion passes.
//
//===----------------------------------------------------------------------===//

#include "SimpleArith.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  // Initialize LLVM infrastructure
  llvm::InitLLVM y(argc, argv);

  // Register all standard MLIR passes
  // This includes common transformation and analysis passes
  mlir::registerAllPasses();

  // Register our custom SimpleArith conversion pass
  mlir::simple::registerConvertSimpleArithToArithPass();

  // Set up dialect registry with all dialects we might need
  mlir::DialectRegistry registry;

  // Register standard MLIR dialects (func, arith, builtin, etc.)
  mlir::registerAllDialects(registry);

  // Register our custom SimpleArith dialect
  registry.insert<mlir::simple::SimpleArithDialect>();

  // mlirOptMain handles:
  // - Command-line parsing
  // - File I/O
  // - Pass pipeline parsing and execution
  // - IR verification
  //
  // Usage:
  //   simple-opt input.mlir -convert-simple-to-arith -o output.mlir
  //   simple-opt input.mlir -convert-simple-to-arith --mlir-print-ir-after-all
  //
  // Source: mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h:29
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SimpleArith optimizer\n", registry));
}
