// mcomp.tools/mcomp-opt.cpp
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "MCompDialect.h.inc"
#define GET_OP_CLASSES
#include "MCompOps.h.inc"

using namespace mlir;

namespace mlir::mcomp {
  std::unique_ptr<Pass> createFuseAddReluPass();
  std::unique_ptr<Pass> createConvertMCompToStdPass();
}

int main(int argc, char **argv) {
  mlir::registerTransformsPasses();

  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect,
                  LLVM::LLVMDialect, mcomp::MCompDialect>();

  [[maybe_unused]] auto _force1 = &mlir::mcomp::createFuseAddReluPass;
  [[maybe_unused]] auto _force2 = &mlir::mcomp::createConvertMCompToStdPass;

  return failed(MlirOptMain(argc, argv, "mcomp-opt driver", registry));
}
