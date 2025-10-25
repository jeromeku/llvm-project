# Build SimpleArith lowering library
add_mlir_library(MLIRSimpleArithLowering
  SimpleArithLowering.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRSimpleArithIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)

# Build TypedArith lowering library
add_mlir_library(MLIRTypedArithLowering
  TypedArithLowering.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRTypedArithIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)

# Build PartialLowering library
add_mlir_library(MLIRPartialLowering
  PartialLowering.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRMixedDialectsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)
