
LLVM_HOME="/home/jeromeku/mlir/llvm-project"
RUNNER=${LLVM_HOME}/build/bin/mlir-runner
SUPPORT_LIB="${LLVM_HOME}/build/lib/libmlir_cuda_runtime.so"
RUNNER_UTILS="${LLVM_HOME}/build/lib/libmlir_runner_utils.so"
RUNNER_UTILS_C="${LLVM_HOME}/build/lib/libmlir_c_runner_utils.so"

export CUDA_ROOT="/home/jeromeku/cuda-toolkit"
export MLIR_CUDA_DEBUG=1
export PATH="/home/jeromeku/cuda-toolkit/bin:$PATH"

$RUNNER kernel_only.mlir \
  -O3 \
  -e run -entry-point-result=void \
  -shared-libs=${SUPPORT_LIB},${RUNNER_UTILS},${RUNNER_UTILS_C}