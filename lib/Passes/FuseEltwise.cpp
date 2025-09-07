// lib.Passes/FuseEltwise.cpp

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