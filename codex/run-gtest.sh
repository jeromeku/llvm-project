#!/usr/bin/env bash
set -euo pipefail

# Run an MLIR gtest binary directly (no lit).
# Usage:
#   bash codex/run-gtest.sh IR/MLIRIRTests [-- --gtest_filter=OperationTest.*]
# or with absolute/relative path to a specific binary.

here=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$here/.." && pwd)
build_dir="$repo_root/build"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <binary | subpath under build/tools/mlir/unittests> [-- <extra gtest args>]" >&2
  exit 2
fi

bin_arg="$1"; shift || true

if [[ -x "$bin_arg" ]]; then
  bin="$bin_arg"
elif [[ -x "$build_dir/tools/mlir/unittests/$bin_arg" ]]; then
  bin="$build_dir/tools/mlir/unittests/$bin_arg"
else
  # Try to find by name under unittests
  match=$(find "$build_dir/tools/mlir/unittests" -type f -perm -111 -name "$(basename "$bin_arg")" | head -n1 || true)
  if [[ -n "$match" ]]; then
    bin="$match"
  else
    echo "error: cannot find test binary: $bin_arg" >&2
    exit 2
  fi
fi

echo "Running: $bin $*"
"$bin" "$@"

