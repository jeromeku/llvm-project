#!/bin/bash

# Test script for MLIR Dialect Conversion Tutorial
# Runs the example conversions and shows the results.

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$SCRIPT_DIR/build"
TEST_DIR="$SCRIPT_DIR/test"

echo -e "${GREEN}MLIR Dialect Conversion Tutorial - Test Runner${NC}"
echo "================================================"
echo ""

# Check if build exists
if [ ! -d "$BUILD_DIR/bin" ]; then
    echo -e "${RED}Error: Build directory not found${NC}"
    echo "Please run ./build.sh first"
    exit 1
fi

# Add build directory to PATH
export PATH="$BUILD_DIR/bin:$PATH"

echo -e "${BLUE}=== Example 1: Simple Arithmetic Conversion ===${NC}"
echo ""
echo "Input IR (SimpleArith dialect):"
echo "--------------------------------"
cat "$TEST_DIR/simple-arith.mlir" | grep -A 3 "@test_integer_ops"
echo ""
echo "Running: simple-opt test/simple-arith.mlir -convert-simple-to-arith"
echo ""
echo "Output IR (Arith dialect):"
echo "--------------------------------"
simple-opt "$TEST_DIR/simple-arith.mlir" -convert-simple-to-arith | grep -A 10 "@test_integer_ops" || true
echo ""
echo -e "${GREEN}✓ Example 1 completed${NC}"
echo ""
echo "---"
echo ""

echo -e "${BLUE}=== Example 2: Type Conversion ===${NC}"
echo ""
echo "Input IR (TypedArith with custom types):"
echo "--------------------------------"
cat "$TEST_DIR/typed-arith.mlir" | grep -A 4 "@test_fixed_mul"
echo ""
echo "Running: typed-opt test/typed-arith.mlir -convert-typed-to-std"
echo ""
echo "Output IR (Standard types with fixed-point math):"
echo "--------------------------------"
typed-opt "$TEST_DIR/typed-arith.mlir" -convert-typed-to-std | grep -A 8 "@test_fixed_mul" || true
echo ""
echo -e "${GREEN}✓ Example 2 completed${NC}"
echo ""
echo "Notice how:"
echo "  - FixedPoint<16,8> types → i16"
echo "  - fixed_mul expanded to: mul + right-shift by scale"
echo ""
echo "---"
echo ""

echo -e "${BLUE}=== Example 3: Partial Conversion ===${NC}"
echo ""
echo "Input IR (High-level operations):"
echo "--------------------------------"
cat "$TEST_DIR/partial-conversion.mlir" | grep -A 4 "@test_large_matmul"
echo ""
echo "Running: partial-opt test/partial-conversion.mlir -partial-lower-high"
echo ""
echo "Output IR (Mixed high/medium dialects):"
echo "--------------------------------"
partial-opt "$TEST_DIR/partial-conversion.mlir" -partial-lower-high | grep -A 4 "@test_large_matmul" || true
echo ""
echo -e "${GREEN}✓ Example 3 completed${NC}"
echo ""
echo "Note: Large matmul remains as high.matmul (dynamically legal)"
echo "      Small matmul would be converted to loops (if patterns were complete)"
echo ""
echo "---"
echo ""

echo -e "${GREEN}All examples completed!${NC}"
echo ""
echo "To explore further:"
echo "  1. Modify test/*.mlir files and re-run"
echo "  2. Add --mlir-print-ir-after-all to see IR after each transformation"
echo "  3. Use -debug-only=dialect-conversion for detailed conversion info"
echo ""
echo "Examples:"
echo "  simple-opt test/simple-arith.mlir -convert-simple-to-arith --mlir-print-ir-after-all"
echo "  MLIR_ENABLE_DUMP=1 simple-opt test/simple-arith.mlir -convert-simple-to-arith -debug-only=dialect-conversion"
