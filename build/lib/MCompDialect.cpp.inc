/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MComp.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::mcomp::MCompDialect)
namespace mlir {
namespace mcomp {

MCompDialect::MCompDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<MCompDialect>())
    
     {
  
  initialize();
}

MCompDialect::~MCompDialect() = default;

} // namespace mcomp
} // namespace mlir
