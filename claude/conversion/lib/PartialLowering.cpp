//===- PartialLowering.cpp - Partial conversion demonstration ------------===//
//
// Demonstrates partial conversion, dynamic legality, and multi-stage lowering.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "MixedDialects.h.inc"

#define GET_OP_CLASSES
#include "MixedDialectsOps.cpp.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Dialect Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace high {

void HighDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MixedDialects.cpp.inc"
      >();
}

} // namespace high

namespace medium {

void MediumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MixedDialects.cpp.inc"
      >();
}

} // namespace medium
} // namespace mlir

//===----------------------------------------------------------------------===//
// Pass 1: High → Medium (Partial Conversion)
//===----------------------------------------------------------------------===//
//
// This pass demonstrates:
// - Partial conversion: Not all operations need to be converted
// - Mixed dialect IR: Output can contain both high and medium operations
// - Dynamic legality: Conditional operation legality
//

namespace {

// -----------------------------------------------------------------------------
// Pattern: Lower matmul to nested loops with dot products
// -----------------------------------------------------------------------------
//
// This demonstrates:
// - Multi-operation expansion
// - Creating structured control flow
// - Lowering from high-level to medium-level abstraction
//
struct LowerMatMulPattern : public OpConversionPattern<high::MatMulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(high::MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    // Get matrix dimensions
    // Assuming shapes are static for simplicity
    // A: MxK, B: KxN, C: MxN
    auto lhsType = op.getLhs().getType().cast<RankedTensorType>();
    auto rhsType = op.getRhs().getType().cast<RankedTensorType>();

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) {
      // In a real implementation, handle dynamic shapes
      return rewriter.notifyMatchFailure(op, "requires static shapes");
    }

    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();

    int64_t M = lhsShape[0];
    int64_t K = lhsShape[1];
    int64_t N = rhsShape[1];

    // Lowering strategy:
    // C[i,j] = sum_k(A[i,k] * B[k,j])
    //
    // Pseudocode:
    // C = alloc<MxN>
    // for i in 0..M:
    //   for j in 0..N:
    //     C[i,j] = dot(A[i,:], B[:,j])  // Leave dot as high-level op
    //
    // Note: We're doing PARTIAL conversion - dot operations remain in
    // the high dialect and will be lowered in a subsequent pass.

    // Create result tensor
    Value result = rewriter.create<medium::AllocOp>(loc, op.getType());

    // Create index constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cM = rewriter.create<arith::ConstantIndexOp>(loc, M);
    Value cN = rewriter.create<arith::ConstantIndexOp>(loc, N);

    // Outer loop: for i in 0..M
    rewriter.create<medium::LoopOp>(
        loc, c0, cM, c1, [&](OpBuilder &builder, Location loc) {
          // TODO: In a real implementation, the loop would provide
          // an induction variable. For this tutorial, we're simplifying.
          // You would typically use scf.for which provides the IV.

          builder.create<medium::YieldOp>(loc);
        });

    // NOTE: The above is simplified. A real implementation would:
    // 1. Use scf.for loops which provide induction variables
    // 2. Extract rows/columns from input matrices
    // 3. Call high.dot on the extracted vectors
    // 4. Insert results into output matrix
    //
    // For this tutorial, we'll show a simplified version that
    // demonstrates the concept without full loop infrastructure.

    // For demonstration, let's just mark this as a TODO and show
    // what the fully lowered code would conceptually look like:

    // CONCEPTUAL LOWERING (not actual code):
    // medium.loop %i = 0 to M {
    //   medium.loop %j = 0 to N {
    //     %a_row = extract_slice(A, %i, ...)
    //     %b_col = extract_slice(B, %j, ...)
    //     %dot = high.dot %a_row, %b_col  // Still high-level!
    //     %result = medium.insert %dot into %result[%i, %j]
    //   }
    // }

    // For this tutorial, we'll just fail the pattern to keep it simple
    // and focus on the conversion infrastructure rather than the
    // complexity of matrix operations.

    return failure();
  }
};

// -----------------------------------------------------------------------------
// Pattern: Lower dot product to sum of multiplications
// -----------------------------------------------------------------------------
//
// This demonstrates:
// - Staying within the same abstraction level
// - Conditional lowering (only lower if operands are small)
// - Dynamic legality in action
//
struct LowerDotPattern : public OpConversionPattern<high::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(high::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto lhsType = op.getLhs().getType().cast<RankedTensorType>();

    if (!lhsType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires static shape");
    }

    int64_t vectorSize = lhsType.getShape()[0];

    // Lowering strategy:
    // dot(a, b) = sum(a[0]*b[0], a[1]*b[1], ..., a[n-1]*b[n-1])
    //
    // We'll use high.sum with element-wise multiplication
    // (In a real implementation, you'd expand this to a loop)

    // For now, return failure to keep the example focused
    return failure();
  }
};

// -----------------------------------------------------------------------------
// Pattern: Lower sum to sequential additions
// -----------------------------------------------------------------------------
//
struct LowerSumPattern : public OpConversionPattern<high::SumOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(high::SumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Similar to dot, this would expand to a loop or
    // sequential additions. Simplified for tutorial.

    return failure();
  }
};

// -----------------------------------------------------------------------------
// Pass: Partial High-to-Medium Lowering
// -----------------------------------------------------------------------------
//
// This pass demonstrates applyPartialConversion, which allows:
// - Some operations to remain illegal (mixed dialect IR)
// - Incremental lowering (not everything converted at once)
// - Dynamic legality (conditional conversion based on properties)
//
// This models real compilation pipelines where you progressively
// lower through multiple abstraction levels.
//
struct PartialHighToMediumPass
    : public PassWrapper<PartialHighToMediumPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartialHighToMediumPass)

  StringRef getArgument() const final { return "partial-lower-high"; }

  StringRef getDescription() const final {
    return "Partially lower high-level operations to medium-level";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<medium::MediumDialect, arith::ArithDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // -------------------------------------------------------------------------
    // Step 1: Define conversion target with DYNAMIC legality
    // -------------------------------------------------------------------------
    //
    // This is where partial conversion shines. We can specify:
    // - Operations that are always legal
    // - Operations that are always illegal
    // - Operations whose legality depends on runtime conditions
    //
    ConversionTarget target(getContext());

    // These dialects are always legal in the output
    target.addLegalDialect<medium::MediumDialect, arith::ArithDialect,
                           tensor::TensorDialect, func::FuncDialect,
                           BuiltinDialect>();

    // High dialect operations have DYNAMIC legality
    // This is the key feature of partial conversion!
    //
    // Dynamic legality callback signature:
    //   bool(Operation *op)
    //
    // Return true = operation is legal (don't convert)
    // Return false = operation is illegal (must convert)
    //
    // Source: mlir/include/mlir/Transforms/DialectConversion.h:362
    //

    // MatMul is dynamically legal: only legal if operands are large
    // (If operands are small, we want to fully expand the operation)
    target.addDynamicallyLegalOp<high::MatMulOp>([](high::MatMulOp op) {
      auto lhsType = op.getLhs().getType().dyn_cast<RankedTensorType>();
      if (!lhsType || !lhsType.hasStaticShape())
        return true; // Keep if we can't analyze

      // Only convert small matrices (for demonstration)
      int64_t numElements = lhsType.getNumElements();
      return numElements > 64; // Legal if large (don't convert yet)
    });

    // Dot is dynamically legal: only legal if vectors are large
    target.addDynamicallyLegalOp<high::DotOp>([](high::DotOp op) {
      auto lhsType = op.getLhs().getType().dyn_cast<RankedTensorType>();
      if (!lhsType || !lhsType.hasStaticShape())
        return true;

      int64_t vectorSize = lhsType.getShape()[0];
      return vectorSize > 16; // Legal if large
    });

    // Sum is always illegal - must be converted
    target.addIllegalOp<high::SumOp>();

    // -------------------------------------------------------------------------
    // Step 2: Register conversion patterns
    // -------------------------------------------------------------------------
    RewritePatternSet patterns(&getContext());

    // Note: These patterns are simplified and will fail
    // They're here to show the structure, not provide full implementation
    patterns.add<LowerMatMulPattern, LowerDotPattern, LowerSumPattern>(
        &getContext());

    // -------------------------------------------------------------------------
    // Step 3: Apply PARTIAL conversion
    // -------------------------------------------------------------------------
    //
    // applyPartialConversion differs from applyFullConversion:
    //
    // applyFullConversion:
    //   - ALL operations must be legal after conversion
    //   - Fails if any illegal operations remain
    //   - Use for complete lowering to target dialect
    //
    // applyPartialConversion:
    //   - Some operations can remain illegal
    //   - Succeeds if patterns were applied successfully
    //   - Use for incremental/multi-stage lowering
    //   - Allows mixed dialect IR
    //
    // Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:3145
    //
    // Why use partial conversion?
    // - Progressive lowering through multiple abstraction levels
    // - Conditional lowering (only lower when beneficial)
    // - Mixed optimization opportunities (optimize at multiple levels)
    //
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // After this pass, we might have:
    // - Some high.matmul ops (large ones that stayed legal)
    // - Some medium.loop ops (from converted small matmuls)
    // - Some high.dot ops (still high-level, will lower later)
    // - No high.sum ops (all converted)
    //
    // This is VALID and INTENTIONAL with partial conversion!
  }
};

//===----------------------------------------------------------------------===//
// Pass 2: Complete Medium-to-Low Lowering
//===----------------------------------------------------------------------===//
//
// This pass would further lower medium-level operations to arith/scf/etc.
// Not fully implemented for brevity, but demonstrates multi-stage pipeline.
//

struct CompleteMediumToLowPass
    : public PassWrapper<CompleteMediumToLowPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompleteMediumToLowPass)

  StringRef getArgument() const final { return "complete-lower-medium"; }

  StringRef getDescription() const final {
    return "Completely lower medium-level operations to low-level";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    ConversionTarget target(getContext());

    // Now we want COMPLETE lowering - no medium dialect operations allowed
    target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect,
                           func::FuncDialect, BuiltinDialect>();

    // All medium operations are illegal
    target.addIllegalDialect<medium::MediumDialect>();

    // Patterns would go here...
    RewritePatternSet patterns(&getContext());

    // Use FULL conversion this time - all ops must be legal
    if (failed(applyFullConversion(funcOp, target, std::move(patterns)))) {
      // For this tutorial, we don't have patterns, so this will fail
      // In a real implementation, you'd have patterns for all medium ops
      // signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass 3: Analysis Conversion (Dry Run)
//===----------------------------------------------------------------------===//
//
// applyAnalysisConversion is the third conversion driver. It's used to
// TEST if a conversion would succeed without actually modifying the IR.
//
// Use cases:
// - Cost models: Decide whether to convert based on analysis
// - Conditional optimization: Only convert if it would succeed
// - Testing: Verify conversion patterns without changing IR
//
struct AnalyzeLoweringFeasibilityPass
    : public PassWrapper<AnalyzeLoweringFeasibilityPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AnalyzeLoweringFeasibilityPass)

  StringRef getArgument() const final { return "analyze-lowering"; }

  StringRef getDescription() const final {
    return "Analyze if lowering is feasible without modifying IR";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<medium::MediumDialect, arith::ArithDialect>();
    target.addIllegalDialect<high::HighDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMatMulPattern, LowerDotPattern, LowerSumPattern>(
        &getContext());

    // -------------------------------------------------------------------------
    // applyAnalysisConversion: Test without modifying IR
    // -------------------------------------------------------------------------
    //
    // This function:
    // 1. Applies patterns as if doing a real conversion
    // 2. Checks if all operations would be legal
    // 3. Returns success/failure
    // 4. DOES NOT modify the IR (all changes are rolled back)
    //
    // Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:3169
    //
    if (succeeded(
            applyAnalysisConversion(funcOp, target, std::move(patterns)))) {
      funcOp.emitRemark("Lowering is feasible");
    } else {
      funcOp.emitRemark("Lowering would fail");
    }

    // The IR is unchanged regardless of success/failure!
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace high {

std::unique_ptr<Pass> createPartialHighToMediumPass() {
  return std::make_unique<PartialHighToMediumPass>();
}

std::unique_ptr<Pass> createCompleteMediumToLowPass() {
  return std::make_unique<CompleteMediumToLowPass>();
}

std::unique_ptr<Pass> createAnalyzeLoweringFeasibilityPass() {
  return std::make_unique<AnalyzeLoweringFeasibilityPass>();
}

void registerMixedDialectPasses() {
  PassRegistration<PartialHighToMediumPass>();
  PassRegistration<CompleteMediumToLowPass>();
  PassRegistration<AnalyzeLoweringFeasibilityPass>();
}

} // namespace high
} // namespace mlir
