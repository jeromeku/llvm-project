# Generate dialect and operation definitions from TableGen

# SimpleArith dialect
set(LLVM_TARGET_DEFINITIONS SimpleArith.td)
mlir_tablegen(SimpleArith.h.inc -gen-dialect-decls -dialect=simple)
mlir_tablegen(SimpleArith.cpp.inc -gen-dialect-defs -dialect=simple)
mlir_tablegen(SimpleArithOps.h.inc -gen-op-decls)
mlir_tablegen(SimpleArithOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRSimpleArithIncGen)

# TypedArith dialect
set(LLVM_TARGET_DEFINITIONS TypedArith.td)
mlir_tablegen(TypedArith.h.inc -gen-dialect-decls -dialect=typed)
mlir_tablegen(TypedArith.cpp.inc -gen-dialect-defs -dialect=typed)
mlir_tablegen(TypedArithTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TypedArithTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(TypedArithOps.h.inc -gen-op-decls)
mlir_tablegen(TypedArithOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTypedArithIncGen)

# MixedDialects (High and Medium)
set(LLVM_TARGET_DEFINITIONS MixedDialects.td)
mlir_tablegen(MixedDialects.h.inc -gen-dialect-decls)
mlir_tablegen(MixedDialects.cpp.inc -gen-dialect-defs)
mlir_tablegen(MixedDialectsOps.h.inc -gen-op-decls)
mlir_tablegen(MixedDialectsOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRMixedDialectsIncGen)
