#!/usr/bin/env python3
"""
Extract PTX assembly from MLIR gpu.binary operations.

Usage:
    python extract_ptx.py module_after.mlir -o ptx_output/
    python extract_ptx.py module_after.mlir --print  # Print to stdout
"""

import re
import sys
import argparse
from pathlib import Path


def unescape_mlir_string(escaped_str):
    """Unescape MLIR string literals."""
    # MLIR uses C-style escape sequences
    unescaped = escaped_str.replace('\\0A', '\n')  # Newline
    unescaped = unescaped.replace('\\09', '\t')    # Tab
    unescaped = unescaped.replace('\\"', '"')      # Quote
    unescaped = unescaped.replace('\\\\', '\\')    # Backslash
    return unescaped


def extract_ptx_from_mlir(mlir_text):
    """
    Extract PTX assembly from MLIR module text.

    Args:
        mlir_text: String containing MLIR module

    Returns:
        dict: Mapping of kernel names to PTX assembly strings
    """
    ptx_kernels = {}

    # Pattern to match gpu.binary operations
    # gpu.binary @kernel_name [...assembly = "...PTX..."]
    pattern = r'gpu\.binary\s+@(\w+)\s+\[.*?assembly\s*=\s*"([^"]*(?:\\.[^"]*)*)"'

    matches = re.findall(pattern, mlir_text, re.DOTALL)

    for kernel_name, ptx_escaped in matches:
        ptx = unescape_mlir_string(ptx_escaped)
        ptx_kernels[kernel_name] = ptx

    return ptx_kernels


def extract_ptx_from_file(mlir_file):
    """Extract PTX from MLIR file."""
    with open(mlir_file, 'r') as f:
        mlir_text = f.read()

    return extract_ptx_from_mlir(mlir_text)


def main():
    parser = argparse.ArgumentParser(
        description='Extract PTX assembly from MLIR gpu.binary operations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract to directory
  %(prog)s module_after.mlir -o ptx_output/

  # Print to stdout
  %(prog)s module_after.mlir --print

  # Extract specific kernel
  %(prog)s module_after.mlir --kernel gemm_kernel --print
        """
    )

    parser.add_argument('mlir_file', help='Input MLIR file')
    parser.add_argument('-o', '--output-dir', default='.',
                        help='Output directory for PTX files (default: current directory)')
    parser.add_argument('--print', action='store_true',
                        help='Print PTX to stdout instead of saving to files')
    parser.add_argument('--kernel', help='Extract only specific kernel by name')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Extract PTX
    try:
        ptx_kernels = extract_ptx_from_file(args.mlir_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.mlir_file}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    if not ptx_kernels:
        print("No PTX found in module", file=sys.stderr)
        return 1

    # Filter by kernel name if specified
    if args.kernel:
        if args.kernel not in ptx_kernels:
            print(f"Error: Kernel '{args.kernel}' not found", file=sys.stderr)
            print(f"Available kernels: {', '.join(ptx_kernels.keys())}", file=sys.stderr)
            return 1
        ptx_kernels = {args.kernel: ptx_kernels[args.kernel]}

    # Print or save
    if args.print:
        for kernel_name, ptx in ptx_kernels.items():
            if len(ptx_kernels) > 1:
                print(f"{'='*70}")
                print(f"Kernel: {kernel_name}")
                print('='*70)
            print(ptx)
    else:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each kernel to a file
        for kernel_name, ptx in ptx_kernels.items():
            output_file = output_dir / f"{kernel_name}.ptx"

            with open(output_file, 'w') as f:
                f.write(ptx)

            print(f"Extracted PTX for '{kernel_name}' to {output_file}")

            if args.verbose:
                lines = ptx.splitlines()
                print(f"  Lines: {len(lines)}")
                print(f"  Bytes: {len(ptx)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
