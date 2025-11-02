#!/usr/bin/env python3
"""
Extract PTX assembly from MLIR gpu.binary operations using MLIR Python bindings.

This version uses proper MLIR parsing for more robust extraction.

Usage:
    python extract_ptx_mlir.py module_after.mlir -o ptx_output/
    python extract_ptx_mlir.py module_after.mlir --print
"""

import sys
import argparse
from pathlib import Path
import re

try:
    from mlir import ir
except ImportError:
    print("Error: MLIR Python bindings not found", file=sys.stderr)
    print("Make sure MLIR_DIR/python is in your PYTHONPATH", file=sys.stderr)
    sys.exit(1)


def unescape_mlir_string(escaped_str):
    """Unescape MLIR string literals."""
    unescaped = escaped_str.replace('\\0A', '\n')
    unescaped = unescaped.replace('\\09', '\t')
    unescaped = unescaped.replace('\\"', '"')
    unescaped = unescaped.replace('\\\\', '\\')
    return unescaped


def find_gpu_binary_ops(op):
    """Recursively find all gpu.binary operations."""
    gpu_binaries = []

    if op.name == "gpu.binary":
        gpu_binaries.append(op)

    # Recursively search regions
    for region in op.regions:
        for block in region.blocks:
            for child_op in block.operations:
                gpu_binaries.extend(find_gpu_binary_ops(child_op))

    return gpu_binaries


def extract_ptx_from_operation(op):
    """
    Extract PTX from a gpu.binary operation.

    Args:
        op: An ir.Operation of type gpu.binary

    Returns:
        tuple: (kernel_name, ptx_string) or (None, None) if extraction fails
    """
    try:
        # Get symbol name
        kernel_name = None
        if "sym_name" in op.attributes:
            sym_name_attr = op.attributes["sym_name"]
            kernel_name = str(sym_name_attr).strip('"')

        # The PTX is embedded in the "objects" array attribute
        # We need to parse it from the string representation
        # because the GPU dialect attributes aren't fully exposed in Python bindings

        # Convert operation to string and extract assembly field
        op_str = str(op)

        # Pattern to find assembly = "..."
        match = re.search(r'assembly\s*=\s*"([^"]*(?:\\.[^"]*)*)"', op_str)

        if match:
            ptx_escaped = match.group(1)
            ptx = unescape_mlir_string(ptx_escaped)
            return (kernel_name or "unknown", ptx)

    except Exception as e:
        print(f"Warning: Failed to extract PTX from operation: {e}", file=sys.stderr)

    return (None, None)


def extract_ptx_from_module(module_text):
    """
    Extract PTX from MLIR module using Python bindings.

    Args:
        module_text: String containing MLIR module

    Returns:
        dict: Mapping of kernel names to PTX assembly strings
    """
    ptx_kernels = {}

    with ir.Context() as ctx:
        # Load all dialects to ensure proper parsing
        ctx.load_all_available_dialects()

        try:
            # Parse module
            module = ir.Module.parse(module_text)
        except Exception as e:
            print(f"Error: Failed to parse MLIR module: {e}", file=sys.stderr)
            return {}

        # Find all gpu.binary operations
        gpu_binaries = find_gpu_binary_ops(module.operation)

        print(f"Found {len(gpu_binaries)} gpu.binary operation(s)", file=sys.stderr)

        # Extract PTX from each
        for binary_op in gpu_binaries:
            kernel_name, ptx = extract_ptx_from_operation(binary_op)

            if ptx:
                ptx_kernels[kernel_name] = ptx
            else:
                print(f"Warning: Could not extract PTX from operation", file=sys.stderr)

    return ptx_kernels


def extract_ptx_from_file(mlir_file):
    """Extract PTX from MLIR file."""
    with open(mlir_file, 'r') as f:
        mlir_text = f.read()

    return extract_ptx_from_module(mlir_text)


def main():
    parser = argparse.ArgumentParser(
        description='Extract PTX assembly from MLIR gpu.binary operations (MLIR Python bindings version)',
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
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress stderr messages')

    args = parser.parse_args()

    # Redirect stderr if quiet mode
    if args.quiet:
        sys.stderr = open('/dev/null', 'w')

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
                # Count instructions
                instructions = [line.strip() for line in lines
                                if line.strip() and not line.strip().startswith('//')]
                print(f"  Instructions: ~{len(instructions)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
