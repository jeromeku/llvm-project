# simple-opt tool
add_llvm_executable(simple-opt
  simple-opt.cpp

  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(simple-opt PRIVATE
  MLIRSimpleArithLowering
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIROptLib
)

llvm_update_compile_flags(simple-opt)

# typed-opt tool
add_llvm_executable(typed-opt
  typed-opt.cpp

  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(typed-opt PRIVATE
  MLIRTypedArithLowering
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIROptLib
)

llvm_update_compile_flags(typed-opt)

# partial-opt tool
add_llvm_executable(partial-opt
  partial-opt.cpp

  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(partial-opt PRIVATE
  MLIRPartialLowering
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIROptLib
)

llvm_update_compile_flags(partial-opt)
