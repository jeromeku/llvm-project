# export PATH=/home/jeromeku/cuda-toolkit/bin:$PATH
export LD_LIBRARY_PATH=$(cd build && pwd)/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$(cd build && pwd)/tools/mlir/python_packages/mlir_core

#MLIR_PY=$HOME/.venv/bin/python
python - <<'PY'
from mlir import ir
p = "/home/jeromeku/llvm-project/mlir/test/python/dialects/gpu/module-to-binary-nvvm.py"
with ir.Context() as ctx:
    # key line — set, don't call
    ctx.emit_error_diagnostics = True
    exec(compile(open(p).read(), str(p), "exec"), {"__name__": "__main__"})
PY
