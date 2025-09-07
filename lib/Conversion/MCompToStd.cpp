// lib.Conversion/MCompToStd.cpp

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