// lib.MComp/MCompOps.cpp

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Support/TypeID.h"

#define GET_OP_CLASSES
#include "MCompOps.h.inc"

#define GET_OP_CLASSES
#include "MCompOps.cpp.inc"


