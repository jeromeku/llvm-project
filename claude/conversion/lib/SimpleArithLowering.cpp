//===- SimpleArithLowering.cpp - Lower SimpleArith to Arith --------------===//
//
// Demonstrates basic dialect conversion patterns and infrastructure.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// Include generated dialect definitions
#include "SimpleArith.h.inc"

#define GET_OP_CLASSES
#include "SimpleArithOps.cpp.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// SimpleArith Dialect Definition
//===----------------------------------------------------------------------===//

namespace mlir {
namespace simple {

void SimpleArithDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SimpleArith.cpp.inc"
      >();
}

} // namespace simple
} // namespace mlir

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

// -----------------------------------------------------------------------------
// Pattern 1: Convert simple.add → arith.addi/addf
// -----------------------------------------------------------------------------
//
// This pattern demonstrates:
// - Basic OpConversionPattern usage
// - Type-based lowering decisions (integer vs float)
// - Simple one-to-one operation replacement
//
// Implementation notes:
// - OpConversionPattern<T> provides type-safe access to the operation
// - OpAdaptor provides access to converted operands (after type conversion)
// - matchAndRewrite() returns success() or failure() to indicate outcome
//
// Source reference:
// - OpConversionPattern: mlir/include/mlir/Transforms/DialectConversion.h:1051
// - Pattern matching: mlir/lib/Transforms/Utils/DialectConversion.cpp:2509
//
struct ConvertAddOp : public OpConversionPattern<simple::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(simple::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the result type - used to determine integer vs float lowering
    Type resultType = op.getResult().getType();

    Location loc = op.getLoc();

    // Get converted operands from the adaptor
    // IMPORTANT: These are the TYPE-CONVERTED versions of the original operands
    // If a TypeConverter was provided, these may have different types than
    // the original op.getLhs() and op.getRhs()
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value result;

    // Choose the appropriate arith operation based on type
    if (resultType.isIntOrIndex()) {
      // Integer addition: simple.add → arith.addi
      // arith.addi: mlir/include/mlir/Dialect/Arith/IR/ArithOps.td:114
      result = rewriter.create<arith::AddIOp>(loc, lhs, rhs);
    } else if (resultType.isa<FloatType>()) {
      // Float addition: simple.add → arith.addf
      // arith.addf: mlir/include/mlir/Dialect/Arith/IR/ArithOps.td:143
      result = rewriter.create<arith::AddFOp>(loc, lhs, rhs);
    } else {
      // Unsupported type - conversion fails
      return failure();
    }

    // Replace the original operation with the new one
    // CRITICAL: This updates the IR and the value mapping maintained by
    // the conversion framework. The mapping is used to provide converted
    // operands to subsequent patterns.
    //
    // Implementation: mlir/lib/Transforms/Utils/DialectConversion.cpp:1643
    rewriter.replaceOp(op, result);

    return success();
  }
};

// -----------------------------------------------------------------------------
// Pattern 2: Convert simple.sub → arith.subi/subf
// -----------------------------------------------------------------------------
//
// Similar structure to AddOp conversion, demonstrates pattern consistency.
//
struct ConvertSubOp : public OpConversionPattern<simple::SubOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(simple::SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value result;
    if (resultType.isIntOrIndex()) {
      result = rewriter.create<arith::SubIOp>(loc, lhs, rhs);
    } else if (resultType.isa<FloatType>()) {
      result = rewriter.create<arith::SubFOp>(loc, lhs, rhs);
    } else {
      return failure();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// -----------------------------------------------------------------------------
// Pattern 3: Convert simple.mul → arith.muli/mulf
// -----------------------------------------------------------------------------
//
struct ConvertMulOp : public OpConversionPattern<simple::MulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(simple::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value result;
    if (resultType.isIntOrIndex()) {
      result = rewriter.create<arith::MulIOp>(loc, lhs, rhs);
    } else if (resultType.isa<FloatType>()) {
      result = rewriter.create<arith::MulFOp>(loc, lhs, rhs);
    } else {
      return failure();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// -----------------------------------------------------------------------------
// Pattern 4: Convert simple.div → arith.divsi/divf
// -----------------------------------------------------------------------------
//
// Note: For integers, we use signed division (divsi). A more sophisticated
// conversion might inspect attributes or types to choose signed vs unsigned.
//
struct ConvertDivOp : public OpConversionPattern<simple::DivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(simple::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value result;
    if (resultType.isIntOrIndex()) {
      // Using signed division for integers
      result = rewriter.create<arith::DivSIOp>(loc, lhs, rhs);
    } else if (resultType.isa<FloatType>()) {
      result = rewriter.create<arith::DivFOp>(loc, lhs, rhs);
    } else {
      return failure();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// -----------------------------------------------------------------------------
// Pattern 5: Convert simple.neg → arith.subi(0, x) or arith.negf
// -----------------------------------------------------------------------------
//
// This pattern demonstrates:
// - Multi-operation lowering (negation → subtraction for integers)
// - Creating intermediate values during conversion
// - Different lowering strategies based on type
//
// For integers: -x becomes (0 - x)
// For floats: -x becomes arith.negf
//
struct ConvertNegOp : public OpConversionPattern<simple::NegOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(simple::NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();
    Location loc = op.getLoc();
    Value operand = adaptor.getOperand();

    Value result;
    if (resultType.isIntOrIndex()) {
      // Integer negation: simple.neg %x → arith.subi 0, %x
      //
      // Step 1: Create constant zero
      // The IntegerAttr::get() creates a constant attribute
      // IntegerAttr: mlir/include/mlir/IR/BuiltinAttributes.h:215
      IntegerAttr zeroAttr = rewriter.getIntegerAttr(resultType, 0);
      Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

      // Step 2: Subtract operand from zero
      result = rewriter.create<arith::SubIOp>(loc, zero, operand);

    } else if (resultType.isa<FloatType>()) {
      // Float negation: simple.neg %x → arith.negf %x
      // arith.negf: mlir/include/mlir/Dialect/Arith/IR/ArithOps.td:729
      result = rewriter.create<arith::NegFOp>(loc, operand);

    } else {
      return failure();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// -----------------------------------------------------------------------------
// Pattern 6: Convert simple.constant → arith.constant
// -----------------------------------------------------------------------------
//
// This pattern demonstrates:
// - Attribute forwarding during conversion
// - Direct one-to-one operation mapping with attributes
//
struct ConvertConstantOp : public OpConversionPattern<simple::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(simple::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Forward the constant attribute directly to arith.constant
    // The attribute contains the actual constant value (integer, float, etc.)
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, op.getResult().getType(), adaptor.getValueAttr());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// ConvertSimpleArithToArithPass
// -----------------------------------------------------------------------------
//
// This pass demonstrates:
// - Setting up a ConversionTarget
// - Registering conversion patterns
// - Using applyFullConversion to ensure complete lowering
//
// The conversion framework flow:
// 1. ConversionTarget determines what operations are legal/illegal
// 2. Patterns define how to rewrite illegal operations
// 3. applyFullConversion applies patterns until all operations are legal
//
// Source references:
// - Pass infrastructure: mlir/include/mlir/Pass/Pass.h:273
// - applyFullConversion: mlir/lib/Transforms/Utils/DialectConversion.cpp:2876
//
struct ConvertSimpleArithToArithPass
    : public PassWrapper<ConvertSimpleArithToArithPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSimpleArithToArithPass)

  StringRef getArgument() const final { return "convert-simple-to-arith"; }

  StringRef getDescription() const final {
    return "Convert SimpleArith dialect operations to Arith dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // Register all dialects that might be used during or after conversion
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    // Get the function operation to convert
    func::FuncOp funcOp = getOperation();

    // -------------------------------------------------------------------------
    // Step 1: Define the conversion target
    // -------------------------------------------------------------------------
    //
    // ConversionTarget specifies which operations are legal in the output IR.
    // The conversion framework will only succeed if all operations can be
    // made legal through pattern application.
    //
    // Source: mlir/include/mlir/Transforms/DialectConversion.h:301
    //
    ConversionTarget target(getContext());

    // Mark the arith and func dialects as legal
    // This means any operation from these dialects is acceptable in the output
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();

    // Mark the entire simple dialect as illegal
    // This means ALL operations from simple dialect must be converted
    // If any remain after pattern application, the conversion fails
    target.addIllegalDialect<simple::SimpleArithDialect>();

    // -------------------------------------------------------------------------
    // Step 2: Register conversion patterns
    // -------------------------------------------------------------------------
    //
    // RewritePatternSet holds all the conversion patterns.
    // Each pattern describes how to convert one or more operations.
    //
    // Source: mlir/include/mlir/IR/PatternMatch.h:862
    //
    RewritePatternSet patterns(&getContext());

    // Register all our conversion patterns
    // The patterns will be tried in order of benefit (higher first),
    // then registration order as a tiebreaker.
    //
    // Note: No TypeConverter needed for this simple example since we're
    // not changing any types - we only convert operations.
    patterns.add<ConvertAddOp,
                 ConvertSubOp,
                 ConvertMulOp,
                 ConvertDivOp,
                 ConvertNegOp,
                 ConvertConstantOp>(&getContext());

    // -------------------------------------------------------------------------
    // Step 3: Apply the conversion
    // -------------------------------------------------------------------------
    //
    // applyFullConversion applies patterns until:
    // - All operations are legal (SUCCESS), or
    // - No more patterns can be applied but illegal ops remain (FAILURE)
    //
    // Implementation details:
    // 1. Creates an OperationConverter with the target and patterns
    // 2. Iteratively tries to apply patterns to illegal operations
    // 3. Maintains value mappings for converted operands
    // 4. Rolls back changes if conversion fails
    //
    // Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:3116
    //
    if (failed(applyFullConversion(funcOp, target, std::move(patterns)))) {
      // Conversion failed - some operations couldn't be made legal
      // This signals a pass failure, which will be reported to the user
      signalPassFailure();
    }

    // If we reach here, all simple.* operations have been successfully
    // converted to arith.* operations!
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace simple {

// Factory function to create the pass
// This is called by the pass manager infrastructure
std::unique_ptr<Pass> createConvertSimpleArithToArithPass() {
  return std::make_unique<ConvertSimpleArithToArithPass>();
}

// Register the pass with MLIR's pass infrastructure
// This allows it to be used via command-line with -convert-simple-to-arith
void registerConvertSimpleArithToArithPass() {
  PassRegistration<ConvertSimpleArithToArithPass>();
}

} // namespace simple
} // namespace mlir
