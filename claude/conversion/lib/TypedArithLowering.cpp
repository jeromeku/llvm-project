//===- TypedArithLowering.cpp - Lower TypedArith with type conversion ----===//
//
// Demonstrates type conversion, materialization, and signature conversion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "TypedArith.h.inc"

#define GET_TYPEDEF_CLASSES
#include "TypedArithTypes.cpp.inc"

#define GET_OP_CLASSES
#include "TypedArithOps.cpp.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TypedArith Dialect Definition
//===----------------------------------------------------------------------===//

namespace mlir {
namespace typed {

void TypedArithDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TypedArith.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "TypedArith.cpp.inc"
      >();
}

// Verify FixedPointType parameters
LogicalResult FixedPointType::verify(
    function_ref<InPlaceInferenceSignature> emitError,
    unsigned width, unsigned scale) {
  if (scale > width) {
    return emitError() << "scale (" << scale
                      << ") cannot exceed width (" << width << ")";
  }
  return success();
}

// Verify operations
LogicalResult FixedMulOp::verify() {
  // Ensure both operands have same fixed-point format
  auto lhsType = getLhs().getType().cast<FixedPointType>();
  auto rhsType = getRhs().getType().cast<FixedPointType>();
  auto resultType = getResult().getType().cast<FixedPointType>();

  if (lhsType != rhsType || lhsType != resultType) {
    return emitOpError("operands and result must have same fixed-point type");
  }
  return success();
}

LogicalResult ComplexCreateOp::verify() {
  auto complexType = getResult().getType().cast<ComplexType>();
  if (getReal().getType() != complexType.getElementType() ||
      getImag().getType() != complexType.getElementType()) {
    return emitOpError("component types must match complex element type");
  }
  return success();
}

// CastOpInterface implementations
bool IntToFixedOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return inputs[0].isa<IntegerType>() && outputs[0].isa<FixedPointType>();
}

bool FixedToIntOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return inputs[0].isa<FixedPointType>() && outputs[0].isa<IntegerType>();
}

} // namespace typed
} // namespace mlir

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

namespace {

// -----------------------------------------------------------------------------
// TypedArithTypeConverter
// -----------------------------------------------------------------------------
//
// TypeConverter is the heart of type-aware dialect conversion.
// It defines:
// 1. How types are converted (e.g., FixedPoint → Integer)
// 2. How to materialize conversions when there are type mismatches
// 3. How to convert function signatures
//
// Key concepts:
// - Type conversion: Maps source types to target types
// - Target materialization: Source → Target when target IR needs target types
// - Source materialization: Target → Source when source IR needs source types
// - Argument materialization: Special handling for block/function arguments
//
// Source: mlir/include/mlir/Transforms/DialectConversion.h:555
//
class TypedArithTypeConverter : public TypeConverter {
public:
  TypedArithTypeConverter() {
    // -------------------------------------------------------------------------
    // Register type conversions
    // -------------------------------------------------------------------------

    // Keep all standard types as-is (integers, floats, etc.)
    // This is important so we don't try to convert arith types
    addConversion([](Type type) { return type; });

    // Convert FixedPoint<width, scale> → iWidth (integer of same width)
    //
    // Example: FixedPoint<16, 8> → i16
    //
    // The scale information is lost in conversion - we're treating
    // fixed-point values as their underlying integer representation.
    addConversion([](typed::FixedPointType type) -> Type {
      return IntegerType::get(type.getContext(), type.getWidth());
    });

    // Convert Complex<T> → Tuple<T, T> (represented as two separate values)
    //
    // Example: Complex<f32> → (f32, f32)
    //
    // NOTE: MLIR doesn't have a built-in tuple type that works well for SSA,
    // so we'll handle this specially in our patterns by expanding complex
    // operations into operations on separate real/imaginary values.
    //
    // In a production system, you might:
    // 1. Use a struct type from a target dialect
    // 2. Lower to separate values (our approach)
    // 3. Use LLVM's struct types if lowering to LLVM
    addConversion([](typed::ComplexType type) -> Type {
      // For now, just keep as complex - patterns will handle decomposition
      // In a full implementation, this might return a struct type
      return type;
    });

    // -------------------------------------------------------------------------
    // Register materialization callbacks
    // -------------------------------------------------------------------------
    //
    // Materialization creates "glue" operations when there's a type mismatch
    // between what we have and what we need.

    // Target materialization: Insert casts from source types to target types
    //
    // When is this used?
    // - When a converted operation produces a target type, but an unconverted
    //   operation still expects the source type.
    //
    // Example scenario:
    //   %fp = some_unconverted_op : !typed.fixed<16,8>  // Still in source dialect
    //   %int = converted_op %fp : i16                    // Needs integer!
    //
    // Solution: Insert typed.fixed_to_int to bridge the gap
    //
    // Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:1157
    addTargetMaterialization(
        [](OpBuilder &builder, Type targetType, ValueRange inputs,
           Location loc) -> Value {
          // We should have exactly one input value
          if (inputs.size() != 1)
            return nullptr;

          Value input = inputs[0];

          // FixedPoint → Integer: insert explicit cast
          if (targetType.isa<IntegerType>() &&
              input.getType().isa<typed::FixedPointType>()) {
            return builder.create<typed::FixedToIntOp>(loc, targetType, input);
          }

          // Integer → FixedPoint: insert explicit cast
          if (targetType.isa<typed::FixedPointType>() &&
              input.getType().isa<IntegerType>()) {
            return builder.create<typed::IntToFixedOp>(loc, targetType, input);
          }

          // If no conversion needed, return null to signal failure
          // The framework will report an error
          return nullptr;
        });

    // Source materialization: Insert casts from target types to source types
    //
    // When is this used?
    // - Opposite of target materialization
    // - When a converted operation produces a target type, but an unconverted
    //   operation expects a source type
    //
    // Example scenario:
    //   %int = converted_op : i16                      // Produces integer
    //   %result = unconverted_op %int : !typed.fixed<16,8>  // Needs fixed!
    //
    // Solution: Insert typed.int_to_fixed to bridge the gap
    addSourceMaterialization(
        [](OpBuilder &builder, Type sourceType, ValueRange inputs,
           Location loc) -> Value {
          if (inputs.size() != 1)
            return nullptr;

          Value input = inputs[0];

          // Integer → FixedPoint: insert explicit cast
          if (sourceType.isa<typed::FixedPointType>() &&
              input.getType().isa<IntegerType>()) {
            return builder.create<typed::IntToFixedOp>(loc, sourceType, input);
          }

          // FixedPoint → Integer: insert explicit cast
          if (sourceType.isa<IntegerType>() &&
              input.getType().isa<typed::FixedPointType>()) {
            return builder.create<typed::FixedToIntOp>(loc, sourceType, input);
          }

          return nullptr;
        });

    // Argument materialization: Handle function/block argument conversions
    //
    // When is this used?
    // - During signature conversion (convertSignatureArgs)
    // - When function argument types need to be converted
    //
    // Example:
    //   func @foo(%arg: !typed.fixed<16,8>) {  // Before
    //   func @foo(%arg: i16) {                  // After
    //     %fp = typed.int_to_fixed %arg         // Materialization
    //
    // This allows function bodies to continue using the original types
    // while the function signature uses converted types.
    addArgumentMaterialization(
        [](OpBuilder &builder, Type resultType, ValueRange inputs,
           Location loc) -> Value {
          if (inputs.size() != 1)
            return nullptr;

          Value input = inputs[0];

          // Integer argument → FixedPoint usage
          if (resultType.isa<typed::FixedPointType>() &&
              input.getType().isa<IntegerType>()) {
            return builder.create<typed::IntToFixedOp>(loc, resultType, input);
          }

          // FixedPoint argument → Integer usage
          if (resultType.isa<IntegerType>() &&
              input.getType().isa<typed::FixedPointType>()) {
            return builder.create<typed::FixedToIntOp>(loc, resultType, input);
          }

          return nullptr;
        });
  }
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// FixedPoint Operations
// -----------------------------------------------------------------------------

// Convert fixed_add: addition just works on the integer representation
struct ConvertFixedAddOp : public OpConversionPattern<typed::FixedAddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(typed::FixedAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get converted operands (now integers instead of fixed-point)
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Fixed-point addition is just integer addition!
    // If we have Q8.8 format:
    //   (3.5 = 896) + (2.25 = 576) = (5.75 = 1472)
    //   896 + 576 = 1472 ✓
    //
    // The scale is implicit and doesn't affect addition.
    Value result = rewriter.create<arith::AddIOp>(op.getLoc(), lhs, rhs);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Convert fixed_mul: multiplication requires scale adjustment
struct ConvertFixedMulOp : public OpConversionPattern<typed::FixedMulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(typed::FixedMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the fixed-point type info before conversion
    auto fixedType = op.getLhs().getType().cast<typed::FixedPointType>();
    unsigned scale = fixedType.getScale();

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Fixed-point multiplication:
    //   If a = A * 2^-S and b = B * 2^-S (where A, B are integers, S is scale)
    //   Then a * b = (A * B) * 2^-2S
    //   We need result = (A * B) * 2^-S
    //   So: result = (A * B) >> S
    //
    // Example with Q8.8:
    //   3.5 = 896, 2.0 = 512
    //   896 * 512 = 458752
    //   458752 >> 8 = 1792 = 7.0 ✓

    // Step 1: Multiply the integer representations
    Value product = rewriter.create<arith::MulIOp>(loc, lhs, rhs);

    // Step 2: Right shift by scale to normalize
    Value scaleValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(lhs.getType(), scale));

    Value result = rewriter.create<arith::ShRSIOp>(loc, product, scaleValue);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Convert fixed_constant: just forward as integer constant
struct ConvertFixedConstantOp
    : public OpConversionPattern<typed::FixedConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(typed::FixedConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the converted result type (should be integer)
    Type resultType = getTypeConverter()->convertType(op.getType());

    // Create integer constant with same value
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, resultType, rewriter.getIntegerAttr(resultType, op.getValue()));

    return success();
  }
};

// Convert int_to_fixed: left shift by scale
struct ConvertIntToFixedOp : public OpConversionPattern<typed::IntToFixedOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(typed::IntToFixedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto fixedType = op.getType().cast<typed::FixedPointType>();
    unsigned scale = fixedType.getScale();

    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    // Convert integer to fixed-point representation: i << scale
    // Example: 7 in Q8.8 = 7 << 8 = 1792

    // Get the target integer type
    Type resultType = getTypeConverter()->convertType(op.getType());

    // If input is narrower than result, extend it
    if (input.getType().getIntOrFloatBitWidth() <
        resultType.getIntOrFloatBitWidth()) {
      input = rewriter.create<arith::ExtSIOp>(loc, resultType, input);
    }

    Value scaleValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(resultType, scale));

    Value result = rewriter.create<arith::ShLIOp>(loc, input, scaleValue);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Convert fixed_to_int: right shift by scale
struct ConvertFixedToIntOp : public OpConversionPattern<typed::FixedToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(typed::FixedToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto fixedType = op.getInput().getType().cast<typed::FixedPointType>();
    unsigned scale = fixedType.getScale();

    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    // Convert fixed-point to integer: fp >> scale
    // Example: 1792 (=7.0 in Q8.8) >> 8 = 7

    Type inputType = input.getType();
    Value scaleValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(inputType, scale));

    Value shifted = rewriter.create<arith::ShRSIOp>(loc, input, scaleValue);

    // Truncate to result width if needed
    Type resultType = op.getResult().getType();
    if (shifted.getType() != resultType) {
      shifted = rewriter.create<arith::TruncIOp>(loc, resultType, shifted);
    }

    rewriter.replaceOp(op, shifted);
    return success();
  }
};

// -----------------------------------------------------------------------------
// Complex Number Operations
// -----------------------------------------------------------------------------
//
// Complex numbers are trickier because we're decomposing one value into two.
// We'll demonstrate a decomposition strategy where complex operations are
// expanded into operations on separate real/imaginary components.

// Convert complex_create: becomes a no-op (values already separate)
//
// In a real implementation, this might pack values into a struct or tuple.
// For this tutorial, we track real and imaginary components separately.
struct ConvertComplexCreateOp
    : public OpConversionPattern<typed::ComplexCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(typed::ComplexCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // NOTE: This is a simplified implementation.
    // In practice, you'd need to track real/imaginary pairs somehow.
    // Options include:
    // 1. Use a custom struct/tuple type
    // 2. Return both values (MLIR supports multiple results)
    // 3. Use analysis to track which values are real/imaginary

    // For this tutorial, we'll just fail conversion and demonstrate
    // the concept in the test file.
    return failure();

    // A full implementation might look like:
    // auto structType = /* create or get struct type */;
    // Value packed = /* pack real and imag into struct */;
    // rewriter.replaceOp(op, packed);
  }
};

// Similar simplification for other complex ops - the real implementation
// would need infrastructure to track complex number decomposition

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

struct ConvertTypedArithToStdPass
    : public PassWrapper<ConvertTypedArithToStdPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTypedArithToStdPass)

  StringRef getArgument() const final { return "convert-typed-to-std"; }

  StringRef getDescription() const final {
    return "Convert TypedArith dialect with custom types to standard dialects";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // -------------------------------------------------------------------------
    // Step 1: Create TypeConverter
    // -------------------------------------------------------------------------
    //
    // The TypeConverter defines how types are converted. This is essential
    // when lowering dialects with custom types.
    TypedArithTypeConverter typeConverter;

    // -------------------------------------------------------------------------
    // Step 2: Define conversion target
    // -------------------------------------------------------------------------
    ConversionTarget target(getContext());

    // Standard dialects are legal
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           BuiltinDialect>();

    // TypedArith dialect is illegal EXCEPT for cast operations
    // We keep casts legal because they're used for materialization
    target.addIllegalDialect<typed::TypedArithDialect>();
    target.addLegalOp<typed::IntToFixedOp, typed::FixedToIntOp>();

    // Function operations are legal only if their types are converted
    //
    // This is a powerful pattern: we allow func.func, but only if
    // all argument and result types have been converted.
    //
    // Source: mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h:22
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      // Legal if function type is fully converted
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    // -------------------------------------------------------------------------
    // Step 3: Register conversion patterns
    // -------------------------------------------------------------------------
    RewritePatternSet patterns(&getContext());

    // Add our custom patterns
    patterns.add<ConvertFixedAddOp, ConvertFixedMulOp, ConvertFixedConstantOp,
                 ConvertIntToFixedOp, ConvertFixedToIntOp>(
        typeConverter, &getContext());

    // Add function signature conversion patterns
    //
    // populateFunctionOpInterfaceTypeConversionPattern adds patterns to
    // convert function signatures (arguments and results) according to
    // the TypeConverter.
    //
    // Source: mlir/lib/Dialect/Func/Transforms/FuncConversions.cpp:59
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    // -------------------------------------------------------------------------
    // Step 4: Apply conversion
    // -------------------------------------------------------------------------
    //
    // We use applyPartialConversion instead of applyFullConversion because:
    // 1. We're keeping cast operations legal (for materialization)
    // 2. Some operations might not need conversion
    //
    // Source: mlir/lib/Transforms/Utils/DialectConversion.cpp:3145
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace typed {

std::unique_ptr<Pass> createConvertTypedArithToStdPass() {
  return std::make_unique<ConvertTypedArithToStdPass>();
}

void registerConvertTypedArithToStdPass() {
  PassRegistration<ConvertTypedArithToStdPass>();
}

} // namespace typed
} // namespace mlir
