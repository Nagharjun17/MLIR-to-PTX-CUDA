/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MComp.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace mcomp {


class MCompDialect : public ::mlir::Dialect {
  explicit MCompDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~MCompDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("mcomp");
  }
};
} // namespace mcomp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::mcomp::MCompDialect)
