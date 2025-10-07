command="clang++ $filename -fno-exceptions -fno-rtti \
  -o /tmp/playing-with-mlir \
  -I $LLVM_PREFIX/include -L$LLVM_PREFIX/lib \
  -lMLIR -lLLVM -Wl,-rpath=$LLVM_PREFIX/lib && \
  echo 'Compiled successfully! Running now...' && echo && \
  /tmp/playing-with-mlir $input $quoted_args"
