// lib/MComp/MCompDialect.cpp

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
