cmake_minimum_required(VERSION 3.20)
project(mlir_dialect_names_shim LANGUAGES CXX)

# Configure MLIR from your local LLVM build.
# Example: -DMLIR_DIR=/home/jeromeku/llvm-project/build/lib/cmake/mlir
find_package(MLIR REQUIRED CONFIG)

add_library(mlir_dialect_names_shim SHARED dialect_names_shim.cpp)

target_include_directories(mlir_dialect_names_shim PRIVATE
  ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(mlir_dialect_names_shim PRIVATE ${MLIR_DEFINITIONS})
target_compile_features(mlir_dialect_names_shim PRIVATE cxx_std_17)

target_link_libraries(mlir_dialect_names_shim PRIVATE
  MLIRIR
  MLIRRegisterAllDialects
)

set_target_properties(mlir_dialect_names_shim PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN NO
)

