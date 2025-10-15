#!/usr/bin/env python3
"""
Run a single lit RUN line from an MLIR/LLVM test file without invoking lit.

Features:
- Parses RUN lines (supports multi-line continuations with trailing '\\').
- Expands common lit substitutions using your current repo/build:
  %s, %S, %PATH%, %shlibext, %llvm_src_root, %mlir_src_root, %PYTHON,
  and runtime libs like %mlir_runner_utils, %mlir_c_runner_utils, etc.
- Sets PATH, FILECHECK_OPTS, PYTHONPATH (for MLIR Python bindings) automatically.

Usage:
  python codex/run-litless.py <test-file> [--list] [--run N] [--dry-run]

Examples:
  # List parsed RUN commands
  python codex/run-litless.py mlir/test/IR/attribute.mlir --list

  # Execute the first RUN command
  python codex/run-litless.py mlir/test/IR/attribute.mlir

  # Execute the Nth RUN command (1-based)
  python codex/run-litless.py mlir/test/Integration/Dialect/Linalg/CPU/test-elementwise.mlir --run 1
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path


RUN_RE = re.compile(r"^\s*(?://|#|;)\s*RUN:\s*(.*)$")


def repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    # codex/ is one level below repo root
    return here.parent.parent


def parse_site_cfg(build_dir: Path) -> dict[str, str]:
    cfg = {
        "python_executable": "",
        "llvm_shlib_ext": ".so" if platform.system() != "Darwin" else ".dylib",
        "host_cc": "",
        "host_cxx": "",
    }
    site = build_dir / "tools/mlir/test/lit.site.cfg.py"
    if not site.is_file():
        return cfg
    text = site.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"config\.python_executable\s*=\s*\"([^\"]+)\"", text)
    if m:
        cfg["python_executable"] = m.group(1)
    m = re.search(r"config\.llvm_shlib_ext\s*=\s*\"([^\"]+)\"", text)
    if m:
        cfg["llvm_shlib_ext"] = m.group(1)
    m = re.search(r"config\.host_cc\s*=\s*\"([^\"]+)\"", text)
    if m:
        cfg["host_cc"] = m.group(1).strip()
    m = re.search(r"config\.host_cxx\s*=\s*\"([^\"]+)\"", text)
    if m:
        cfg["host_cxx"] = m.group(1).strip()
    return cfg


def parse_run_lines(test_path: Path) -> list[str]:
    runs: list[str] = []
    cur: list[str] = []
    with test_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RUN_RE.match(line)
            if not m:
                continue
            body = m.group(1).rstrip()
            # Handle line continuations with trailing '\\'
            if body.endswith("\\"):
                body = body[:-1].rstrip()
                cur.append(body)
                continue
            else:
                cur.append(body)
                cmd = " ".join(cur).strip()
                if cmd:
                    runs.append(cmd)
                cur = []
    # If file ended while in continuation, flush what we have.
    if cur:
        cmd = " ".join(cur).strip()
        if cmd:
            runs.append(cmd)
    return runs


def detect_python(repo_root: Path, site_cfg: dict[str, str]) -> str:
    if site_cfg.get("python_executable"):
        return site_cfg["python_executable"]
    venv_py = repo_root / ".venv/bin/python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable or "python3"


def build_substitutions(
    repo_root: Path, build_dir: Path, test_file: Path, site_cfg: dict[str, str]
) -> dict[str, str]:
    bindir = build_dir / "bin"
    libdir = build_dir / "lib"
    llvm_src_root = repo_root / "llvm"
    mlir_src_root = repo_root / "mlir"
    shlibext = site_cfg.get("llvm_shlib_ext", ".so")
    python_exe = detect_python(repo_root, site_cfg)

    # Runtime libs commonly used by tests
    def lib(name: str) -> str:
        cand = libdir / f"lib{name}{shlibext}"
        return str(cand)

    subs = {
        "%s": str(test_file),
        "%S": str(test_file.parent),
        "%PATH%": os.environ.get("PATH", ""),
        "%shlibext": shlibext,
        "%llvm_src_root": str(llvm_src_root),
        "%mlir_src_root": str(mlir_src_root),
        "%mlir_src_dir": str(mlir_src_root),
        "%PYTHON": python_exe,
        # Tool path helpers (rarely used as substitutions in MLIR, but safe to include)
        "%mlir_lib_dir": str(libdir),
        "%mlir_tools_dir": str(bindir),
        "%llvm_tools_dir": str(bindir),
    }
    if site_cfg.get("host_cc"):
        subs["%host_cc"] = site_cfg["host_cc"]
    if site_cfg.get("host_cxx"):
        subs["%host_cxx"] = site_cfg["host_cxx"]

    # Add known runtime lib substitutions (expand to full paths)
    for token in [
        "mlir_runner_utils",
        "mlir_c_runner_utils",
        "mlir_async_runtime",
        "mlir_float16_utils",
        "mlir_vulkan_runtime",
        "mlir_rocm_runtime",
        "mlir_cuda_runtime",
        "mlir_sycl_runtime",
        "mlir_levelzero_runtime",
        "mlir_arm_runner_utils",
        "mlir_spirv_cpu_runtime",
    ]:
        subs[f"%{token}"] = lib(token)

    return subs


def apply_substitutions(cmd: str, subs: dict[str, str]) -> str:
    out = cmd
    # Apply longer tokens first to avoid partial overlaps
    for k in sorted(subs.keys(), key=len, reverse=True):
        out = out.replace(k, subs[k])
    return out


def ensure_env_for_command(cmd: str, repo_root: Path, build_dir: Path, site_cfg: dict[str, str]):
    # PATH: ensure build/bin first so mlir-opt, FileCheck, etc. resolve
    bindir = build_dir / "bin"
    os.environ["PATH"] = f"{bindir}:{os.environ.get('PATH','')}"
    # Default FileCheck opts MLIR sets in its lit config
    os.environ.setdefault(
        "FILECHECK_OPTS", "-enable-var-scope --allow-unused-prefixes=false"
    )
    # Python env for MLIR Python tests: detect if command uses %PYTHON or ends with FileCheck test.py
    if "%PYTHON" in cmd or cmd.strip().startswith("python") or cmd.strip().startswith("/usr/bin/python"):
        core = build_dir / "tools/mlir/python_packages/mlir_core"
        test = build_dir / "tools/mlir/python_packages/mlir_test"
        pp = os.environ.get("PYTHONPATH", "")
        parts = [str(core)]
        if test.is_dir():
            parts.append(str(test))
        if pp:
            parts.append(pp)
        os.environ["PYTHONPATH"] = ":".join(parts)
        # Native libs for Python extensions
        libdir = build_dir / "lib"
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{libdir}:{ld}" if ld else str(libdir)


def main():
    ap = argparse.ArgumentParser(description="Run a lit RUN command without lit")
    ap.add_argument("test_file", help="Path to test file (.mlir, .py, etc.)")
    ap.add_argument("--list", action="store_true", help="List parsed RUN commands and exit")
    ap.add_argument("--run", type=int, default=1, help="1-based index of RUN command to execute")
    ap.add_argument("--dry-run", action="store_true", help="Print the expanded command and exit")
    args = ap.parse_args()

    repo_root = repo_root_from_here()
    build_dir = repo_root / "build"
    if not build_dir.exists():
        print(f"error: build dir not found: {build_dir}", file=sys.stderr)
        sys.exit(2)

    test_path = (repo_root / args.test_file).resolve() if not os.path.isabs(args.test_file) else Path(args.test_file)
    if not test_path.exists():
        print(f"error: test file not found: {test_path}", file=sys.stderr)
        sys.exit(2)

    runs = parse_run_lines(test_path)
    if not runs:
        print("No RUN lines found.")
        sys.exit(1)

    if args.list:
        print(f"Found {len(runs)} RUN command(s):\n")
        for i, r in enumerate(runs, 1):
            print(f"[{i}] {r}")
        return

    if args.run < 1 or args.run > len(runs):
        print(f"error: --run must be between 1 and {len(runs)}", file=sys.stderr)
        sys.exit(2)

    raw_cmd = runs[args.run - 1]
    site_cfg = parse_site_cfg(build_dir)
    subs = build_substitutions(repo_root, build_dir, test_path, site_cfg)
    expanded = apply_substitutions(raw_cmd, subs)

    ensure_env_for_command(expanded, repo_root, build_dir, site_cfg)

    if args.dry_run:
        print(expanded)
        return

    # Run with bash -lc to get pipefail and consistent behavior
    bash_cmd = f"set -euo pipefail; {expanded}"
    try:
        subprocess.run(["bash", "-lc", bash_cmd], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
