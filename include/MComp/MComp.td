// include.MComp/MComp.td

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