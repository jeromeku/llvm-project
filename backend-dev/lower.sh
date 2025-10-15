#/bin/bash

set -euo pipefail
alias clang=clang-22
alias clang++=clang++-22
SOURCE_STEM="simple"

OPT_LEVEL="2"
clang-22 -S -O${OPT_LEVEL} -emit-llvm ${SOURCE_STEM}.c -o ${SOURCE_STEM}.ll
cat simple.ll

llc-22 -mtriple=x86_64 -stop-after=instruction-select ${SOURCE_STEM}.ll -o ${SOURCE_STEM}.mir
llc-22 -mtriple=x86_64 -filetype=asm ${SOURCE_STEM}.ll -o ${SOURCE_STEM}.s
