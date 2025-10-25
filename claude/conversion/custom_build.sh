#!/bin/bash
set -euo pipefail

# Usage: ./build.sh [configure|build] [target]
# - Mode defaults to 'configure' if omitted.
# - In 'build' mode, optional [target] builds a specific target.

MODE=${1:-configure}
TARGET=${2:-}

LLVM_ROOT=$(dirname "$(realpath -L ..)")
echo "LLVM_ROOT=${LLVM_ROOT}"
LLVM_BUILD_DIR="${LLVM_ROOT}/build"

# Allow overrides via environment; default to clang-22 toolchain
CC=${CC:-/usr/bin/clang-22}
CXX=${CXX:-/usr/bin/clang++-22}
BUILD_DIR=${BUILD_DIR:-build}
BUILD_TYPE=${BUILD_TYPE:-Debug}

case "${MODE}" in
  configure)
    cmake -S . -B "${BUILD_DIR}" -G "Ninja"               \
        -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm"     \
        -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"     \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON                   \
        -DPython3_FIND_VIRTUALENV=FIRST                    \
        -DLLVM_ENABLE_LLD=On                               \
        -Wno-dev                                           \
        -DCMAKE_C_COMPILER="${CC}"                         \
        -DCMAKE_CXX_COMPILER="${CXX}"                     \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON                 \
        -DCMAKE_VERBOSE_MAKEFILE=ON                        \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"                 \
        --trace --trace-expand --trace-source="/home/jeromeku/llvm-project/build/lib/cmake/mlir/AddMLIR.cmake" \
        --trace-redirect="_cmake.trace.txt" \
        2>&1 | tee _cmake.config.log
    ;;

  build)
    if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
      echo "Error: Build directory '${BUILD_DIR}' is not configured. Run: $0 configure" >&2
      exit 1
    fi
    if [[ -n "${TARGET}" ]]; then
      cmake --build "${BUILD_DIR}" --target "${TARGET}" -- -v 2>&1 | tee _cmake.build.log
    else
      cmake --build "${BUILD_DIR}" -- -v 2>&1 | tee _cmake.build.log
    fi
    ;;

  *)
    echo "Usage: $0 [configure|build] [target]" >&2
    exit 1
    ;;
esac

# Example linker flags if needed for lld:
# -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld -Wl,--ld-path=$(command -v ld.lld-22)" \
# -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld -Wl,--ld-path=$(command -v ld.lld-22)" \
# -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld -Wl,--ld-path=$(command -v ld.lld-22)"
        # -DLLVM_BUILD_LLVM_DYLIB=ON                         \
        # -DLLVM_LINK_LLVM_DYLIB=ON                          \