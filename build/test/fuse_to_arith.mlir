module {
  func.func @scalar(%arg0: f32, %arg1: f32) -> f32 {
    %0 = arith.addf %arg0, %arg1 : f32
    %cst = arith.constant 0.000000e+00 : f32
    %1 = arith.maximumf %0, %cst : f32
    return %1 : f32
  }
  func.func @tensor_rhs_zero(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    %1 = arith.maximumf %0, %cst : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
  func.func @tensor_lhs_zero(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    %1 = arith.maximumf %cst, %0 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}

