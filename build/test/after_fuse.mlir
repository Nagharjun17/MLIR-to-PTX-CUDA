module {
  func.func @scalar(%arg0: f32, %arg1: f32) -> f32 {
    %0 = mcomp.fuse_add_relu %arg0, %arg1 : f32
    return %0 : f32
  }
  func.func @tensor_rhs_zero(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {llvm.emit_c_interface} {
    %0 = mcomp.fuse_add_relu %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func @tensor_lhs_zero(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {llvm.emit_c_interface} {
    %0 = mcomp.fuse_add_relu %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

