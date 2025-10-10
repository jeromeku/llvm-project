LLVM_HOME=/home/jeromeku/llvm-project
export SUPPORT_LIB="${LLVM_HOME}/build/lib/libmlir_cuda_runtime.so"
export PYTHONPATH="${LLVM_HOME}/build/tools/mlir/python_packages/mlir_core"

python Ch2.py