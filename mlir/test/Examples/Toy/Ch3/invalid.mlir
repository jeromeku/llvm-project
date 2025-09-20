// RUN: not toyc-ch3 %s -emit=mlir 2>&1

// The following IR is not "valid":
// - toy.print should not return a value.
// - toy.print should take an argument.
// - There should be a block terminator.
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
  "toy.print"(%0)  : (tensor<2x3xf64>) -> ()
  "toy.return" () : () -> ()
}
