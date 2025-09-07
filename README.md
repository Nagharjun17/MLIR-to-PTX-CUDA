# MLIR to PTX
This repository contains custom dialects and fusion passes built on top of MLIR and LLVM and lowering to PTX GPU code for learning and exploring compiler engineering for machine learning.

### First, I add the dialect I want to

Defining a new dialect called MComp and a new operation called FuseAddReluOp which fuses the addition and max(addition, 0) code.
```tablegen
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MComp_Dialect : Dialect {
    let name = "mcomp";
    let cppNamespace = "mlir::mcomp";
}

class MComp_Op<string mnemonic, list<Trait> traits = []> :
    Op<MComp_Dialect, mnemonic, traits>;

def FuseAddReluOp : MComp_Op<"fuse_add_relu", [Pure, AllTypesMatch<["lhs","rhs","result"]>]> {
    let summary = "computes the sum and applies relu activation";

    let arguments = (ins AnyTypeOf<[F32, RankedTensorOf<[F32]>]>:$lhs, AnyTypeOf<[F32, RankedTensorOf<[F32]>]>:$rhs);
    let results = (outs AnyTypeOf<[F32, RankedTensorOf<[F32]>]>:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

Writing `MCompDialect.cpp`
```cpp
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "MCompDialect.h.inc"

#define GET_OP_CLASSES
#include "MCompOps.h.inc"

namespace mlir::mcomp {
void MCompDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "MCompOps.cpp.inc"
  >();
}
}

#include "MCompDialect.cpp.inc"
```


Writing `FuseEltwise.cpp` where I detect the add + max operations and fuse them together by rewriting patterns.
```cpp
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"


#include "MCompDialect.h.inc"
#define GET_OP_CLASSES
#include "MCompOps.h.inc"

using namespace mlir;
using namespace mlir::mcomp;

static bool isZeroFloatConstant(Value val){
    auto op = dyn_cast<arith::ConstantOp>(val.getDefiningOp());
    if(!op) return false;

    auto attr = dyn_cast<FloatAttr>(op.getValue());
    if(attr) return attr.getValue().isZero();

    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if(denseAttr){
        auto elemTy = llvm::dyn_cast<FloatType>(denseAttr.getElementType());
        if (!elemTy) return false;
        if (!denseAttr.isSplat()) return false;
        return denseAttr.getSplatValue<APFloat>().isZero();
    }
    return false;
};

struct FuseAddReluPattern : OpRewritePattern<arith::MaximumFOp>{
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(arith::MaximumFOp maxOp, PatternRewriter &rewriter) const override{
        auto lhs = maxOp.getLhs();
        auto rhs = maxOp.getRhs();
        
        Value other;
        if(isZeroFloatConstant(rhs)){
            other = lhs;
        }
        else if(isZeroFloatConstant(lhs)){
            other = rhs;
        }
        else{
            return failure();
        }

        auto addOp = dyn_cast<arith::AddFOp>(other.getDefiningOp());
        if(!addOp) return failure();

        if(!addOp.getResult().hasOneUse()) return failure();

        Type t = maxOp.getType();
        if(t != addOp.getResult().getType()) return failure();

        rewriter.replaceOpWithNewOp<mcomp::FuseAddReluOp>(maxOp, t, addOp.getLhs(), addOp.getRhs());
        return success();
    }
};

struct FuseAddReluPass : PassWrapper<FuseAddReluPass, OperationPass<func::FuncOp>>{
    StringRef getArgument() const final { return "mcomp-fuse-add-relu"; }
    StringRef getDescription() const final {
        return "Fuse arith.addf + arith.maxf(..., 0.0) into mcomp.fuse_add_relu";
    }
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mcomp::MCompDialect>();
    }
    void runOnOperation() override{
        auto func = getOperation();
        RewritePatternSet patterns(&getContext());
        patterns.add<FuseAddReluPattern>(&getContext());
        (void)applyPatternsGreedily(func, std::move(patterns));
    }
};

namespace mlir::mcomp{
std::unique_ptr<Pass> createFuseAddReluPass(){
    return std::make_unique<FuseAddReluPass>();
}
}

static PassRegistration<FuseAddReluPass> FuseAddReluReg;
```



Writing `MCompToStd.cpp` where I detect the new MComp dialect's fuseaddrelu operation and convert them back to the arith add and max dialect operations.
```cpp
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "MCompDialect.h.inc"
#define GET_OP_CLASSES
#include "MCompOps.h.inc"

using namespace mlir;
using namespace mlir::mcomp;

struct FuseAddReluLowering : OpConversionPattern<mcomp::FuseAddReluOp>{
    using OpConversionPattern<mcomp::FuseAddReluOp>::OpConversionPattern;
    using OpAdaptor = typename OpConversionPattern<mcomp::FuseAddReluOp>::OpAdaptor;

    LogicalResult matchAndRewrite(mcomp::FuseAddReluOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final{
        Location loc = op.getLoc();
        Type resultType = op.getResult().getType();
        Value lhs = adaptor.getLhs();
        Value rhs = adaptor.getRhs();
        if(!lhs || !rhs) return failure();

        auto add = rewriter.create<arith::AddFOp>(loc, resultType, lhs, rhs);

        Value zeroConst;
        if(auto currentType = dyn_cast<FloatType>(resultType)){
            auto zeroAttr = rewriter.getFloatAttr(currentType, 0.0);
            zeroConst = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
        }
        else if(auto currentType = dyn_cast<RankedTensorType>(resultType)){
            if(!currentType.hasStaticShape()) return failure();
            auto elementType = dyn_cast<FloatType>(currentType.getElementType());
            if(!elementType) return failure();
            auto zeroAttr = rewriter.getFloatAttr(elementType, 0.0);
            auto splat = SplatElementsAttr::get(currentType, zeroAttr);
            zeroConst = rewriter.create<arith::ConstantOp>(loc, splat);
        }
        else{
            return failure();
        }

        auto relu = rewriter.create<arith::MaximumFOp>(loc, resultType, add.getResult(), zeroConst);
        rewriter.replaceOp(op, relu.getResult());

        return success();
    }
};

struct ConvertMCompToStdPass : PassWrapper<ConvertMCompToStdPass, OperationPass<func::FuncOp>>{
    StringRef getArgument() const final { return "convert-mcomp-to-std"; }
    StringRef getDescription() const final {
        return "Lower mcomp.fuse_add_relu to arith.addf + arith.maxf(…, 0.0)";
    }
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mcomp::MCompDialect, arith::ArithDialect, func::FuncDialect>();
    }
    void runOnOperation() override{
        MLIRContext &context = getContext();
        
        TypeConverter typeConverter;
        typeConverter.addConversion([](Type type) { return type; });

        ConversionTarget target(context);
        target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();
        target.addIllegalOp<mcomp::FuseAddReluOp>();

        RewritePatternSet patterns(&context);
        patterns.add<FuseAddReluLowering>(typeConverter, &context);

        if(failed(applyPartialConversion(getOperation(), target, std::move(patterns)))){
            signalPassFailure();
        }
    }
};

namespace mlir::mcomp {
std::unique_ptr<Pass> createConvertMCompToStdPass() {
  return std::make_unique<ConvertMCompToStdPass>();
}
}

static PassRegistration<ConvertMCompToStdPass> ConvertReg;
```


Writing `mcomp-opt.cpp` where I register the new passes and run them.
```cpp
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

```

Build now
```bash
cmake -GNinja -DMLIR_DIR=/root/src/llvm-project/build/lib/cmake/mlir -DLLVM_DIR=/root/src/llvm-project/build/lib/cmake/llvm -DCMAKE_PREFIX_PATH="/root/src/llvm-project/build" ..

ninja mlir-opt
```

Testing with `fuse_add_relu.mlir`
```mlir
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
```

Run the fusing
```bash
./tools/mcomp-opt test/fuse_add_relu.mlir --mcomp-fuse-add-relu -o test/after_fuse.mlir
```

After the fuse `after_fuse.mlir`
```mlir
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
```

Running the lowering from the fused mlir back to the arith dialect
```bash
./tools/mcomp-opt test/fuse_add_relu.mlir --convert-mcomp-to-std -o test/fuse_to_arith.mlir
```

After the fuse->arith conversion `fuse_to_arith.mlir`
```mlir
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
```

Running conversion to LLVM Dialect from Arith
```bash
./tools/mcomp-opt test/fuse_add_relu.mlir \
  --mcomp-fuse-add-relu \
  --convert-mcomp-to-std \
| /root/src/llvm-project/build/bin/mlir-opt \
  --canonicalize --cse \
  --convert-elementwise-to-linalg \
  --one-shot-bufferize="bufferize-function-boundaries=1" \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-to-llvm \
  --reconcile-unrealized-casts \
  -o /tmp/llvm_tensor.mlir
```

Converted LLVM Dialect code `llvm_dialect.mlir`
```mlir
module {
  llvm.mlir.global private constant @__constant_4xf32(dense<0.000000e+00> : tensor<4xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<4 x f32>
  llvm.func @scalar(%arg0: f32, %arg1: f32) -> f32 {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %1 = llvm.fadd %arg0, %arg1 : f32
    %2 = llvm.intr.maximum(%1, %0) : (f32, f32) -> f32
    llvm.return %2 : f32
  }
  llvm.func @tensor_rhs_zero(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg5, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg6, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg7, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    ...................
```

Running conversion to LLVM IR
```bash
/root/src/llvm-project/build/bin/mlir-translate --mlir-to-llvmir test/llvm_dialect.mlir -o test/module.ll
```

Converted LLVM IR `module.ll`
```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_4xf32 = private constant [4 x float] zeroinitializer, align 64

define float @scalar(float %0, float %1) {
  %3 = fadd float %0, %1
  %4 = call float @llvm.maximum.f32(float %3, float 0.000000e+00)
  ret float %4
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @tensor_rhs_zero(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9) {
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %5, 0
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, ptr %6, 1
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, i64 %7, 2
  ...................
```

Generating the object file to run further
```bash
clang -O2 -c test/module.ll -o test/module.o
```

Writing a simple driver to test `driver.c`
```C
#include <stdio.h>
#include <stdint.h>

typedef struct {
  float   *allocated;
  float   *aligned;
  int64_t  offset;
  int64_t  sizes[1];
  int64_t  strides[1];
} StridedMemRefType_f32_1d;

extern float scalar(float, float);
extern void _mlir_ciface_tensor_rhs_zero(StridedMemRefType_f32_1d *out,
                                         StridedMemRefType_f32_1d *a,
                                         StridedMemRefType_f32_1d *b);
extern void _mlir_ciface_tensor_lhs_zero(StridedMemRefType_f32_1d *out,
                                         StridedMemRefType_f32_1d *a,
                                         StridedMemRefType_f32_1d *b);

static StridedMemRefType_f32_1d make_memref(float *data, int64_t n) {
  StridedMemRefType_f32_1d m;
  m.allocated = m.aligned = data;
  m.offset = 0; m.sizes[0] = n; m.strides[0] = 1;
  return m;
}
static void print_memref(const char *tag, StridedMemRefType_f32_1d *m) {
  float *p = m->aligned + m->offset;
  printf("%s: [%f, %f, %f, %f]\n", tag, p[0], p[1], p[2], p[3]);
}

int main(void) {
  printf("scalar(2.0,-1.5) = %f\n", scalar(2.0f, -1.5f));

  const float A0[4] = {1, -2, 3, -4};
  const float B0[4] = {5,  6, -7,  8};

  float a1[4], b1[4], out1[4] = {0};
  for (int i = 0; i < 4; ++i) { a1[i] = A0[i]; b1[i] = B0[i]; }
  StridedMemRefType_f32_1d A = make_memref(a1, 4);
  StridedMemRefType_f32_1d B = make_memref(b1, 4);
  StridedMemRefType_f32_1d OUT = make_memref(out1, 4);
  _mlir_ciface_tensor_rhs_zero(&OUT, &A, &B);
  print_memref("tensor_rhs_zero", &OUT);

  float a2[4], b2[4], out2[4] = {0};
  for (int i = 0; i < 4; ++i) { a2[i] = A0[i]; b2[i] = B0[i]; }
  A = make_memref(a2, 4);
  B = make_memref(b2, 4);
  OUT = make_memref(out2, 4);
  _mlir_ciface_tensor_lhs_zero(&OUT, &A, &B);
  print_memref("tensor_lhs_zero", &OUT);
  return 0;
}
```

Testing it out
```bash
root@mlir-dev:~/mlir-learning/mcomp/build# clang -O2 test/driver.c test/module.o -o test/run_scalar
root@mlir-dev:~/mlir-learning/mcomp/build# ./test/run_scalar
scalar(2.0,-1.5) = 0.500000
tensor_rhs_zero: [6.000000, 4.000000, 0.000000, 4.000000]
tensor_lhs_zero: [6.000000, 4.000000, 0.000000, 4.000000]
```

IT WORKS!


Now moving on to generating PTX.

Just checking if NVPTX can compile the LLVM code
```bash
 /root/src/llvm-project/build/bin/llc \
   -mtriple=nvptx64-nvidia-cuda \
   -mcpu=sm_86 \
   -filetype=asm \
   test/module.ll -o test/out.ptx
```

Results `out.ptx`
```ptx
//
// Generated by LLVM NVPTX Back-End
//

.version 7.1
.target sm_86
.address_size 64

	// .globl	scalar                  // -- Begin function scalar
.global .align 64 .b8 __constant_4xf32[16];
                                        // @scalar
.visible .func  (.param .b32 func_retval0) scalar(
	.param .b32 scalar_param_0,
	.param .b32 scalar_param_1
)
{
	.reg .b32 	%r<5>;

// %bb.0:
	ld.param.b32 	%r1, [scalar_param_0];
	ld.param.b32 	%r2, [scalar_param_1];
```

Now I am focusing on MLIR GPU -> MLIR NVVM -> LLVM IR -> PTX

Running conversion from GPU MLIR to NVVM MLIR
```mlir
./tools/mcomp-opt test/fuse_add_relu.mlir \
  --mcomp-fuse-add-relu \
  --convert-mcomp-to-std \
| /root/src/llvm-project/build/bin/mlir-opt \
  --canonicalize --cse \
  --convert-elementwise-to-linalg \
  --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" \
  --convert-linalg-to-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  --convert-scf-to-cf \
  --convert-gpu-to-nvvm \
  --convert-math-to-llvm \
  --convert-to-llvm \
  --reconcile-unrealized-casts \
  -o test/nvvm.mlir
```

Converted NVVM MLIR `nvvm.mlir`
```mlir
module {
  llvm.mlir.global private constant @__constant_4xf32(dense<0.000000e+00> : tensor<4xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<4 x f32>
  llvm.func @scalar(%arg0: f32, %arg1: f32) -> f32 {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %1 = llvm.fadd %arg0, %arg1 : f32
    %2 = llvm.intr.maximum(%1, %0) : (f32, f32) -> f32
    llvm.return %2 : f32
  }
  llvm.func @tensor_rhs_zero(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg5, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg6, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg7, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
```

Running conversion from NVVM MLIR to LLVM IR
```bash
/root/src/llvm-project/build/bin/mlir-translate --mlir-to-llvmir test/nvvm.mlir -o test/nvvm.ll
``` 

Converted LLVM IR from NVVM `nvvm.ll`
```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_4xf32 = private constant [4 x float] zeroinitializer, align 64

define float @scalar(float %0, float %1) {
  %3 = fadd float %0, %1
  %4 = call float @llvm.maximum.f32(float %3, float 0.000000e+00)
  ret float %4
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @tensor_rhs_zero(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9) {
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %5, 0
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, ptr %6, 1
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, i64 %7, 2
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, i64 %8, 3, 0
```

Running conversion from LLVM IR to PTX
```bash
/root/src/llvm-project/build/bin/llc \
  -mtriple=nvptx64-nvidia-cuda -mcpu=sm_86 -filetype=asm \
  test/nvvm.ll -o test/kernel.ptx
```

Converted PTX from LLVM IR `kernel.ptx`
```ptx
//
// Generated by LLVM NVPTX Back-End
//

.version 7.1
.target sm_86
.address_size 64

	// .globl	scalar                  // -- Begin function scalar
.global .align 64 .b8 __constant_4xf32[16];
                                        // @scalar
.visible .func  (.param .b32 func_retval0) scalar(
	.param .b32 scalar_param_0,
	.param .b32 scalar_param_1
)
{
	.reg .b32 	%r<5>;

// %bb.0:
	ld.param.b32 	%r1, [scalar_param_0];
	ld.param.b32 	%r2, [scalar_param_1];
```