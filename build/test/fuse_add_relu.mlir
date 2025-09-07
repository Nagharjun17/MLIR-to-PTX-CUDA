// test/fuse_add_relu.mlir
module {
  func.func @scalar(%a: f32, %b: f32) -> f32 {
    %s = arith.addf %a, %b : f32
    %z = arith.constant 0.0 : f32
    %r = arith.maximumf %s, %z : f32
    return %r : f32
  }
  func.func @tensor_rhs_zero(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> attributes { llvm.emit_c_interface } {
    %z = arith.constant dense<0.0> : tensor<4xf32>
    %add = arith.addf %a, %b : tensor<4xf32>
    %r = arith.maximumf %add, %z : tensor<4xf32>
    return %r : tensor<4xf32>
  }
  func.func @tensor_lhs_zero(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> attributes { llvm.emit_c_interface } {
    %z = arith.constant dense<0.0> : tensor<4xf32>
    %add = arith.addf %a, %b : tensor<4xf32>
    %r = arith.maximumf %z, %add : tensor<4xf32>
    return %r : tensor<4xf32>
  }
}