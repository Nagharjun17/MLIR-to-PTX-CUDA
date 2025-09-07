# mcomp.lib/CMakeLists.txt

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/MComp/MComp.td)

mlir_tablegen(MCompDialect.h.inc -gen-dialect-decls -dialect=mcomp)
mlir_tablegen(MCompDialect.cpp.inc -gen-dialect-defs  -dialect=mcomp)
mlir_tablegen(MCompOps.h.inc     -gen-op-decls)
mlir_tablegen(MCompOps.cpp.inc   -gen-op-defs)
add_public_tablegen_target(MCompTableGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_dialect_library(MCompDialect
  MComp/MCompDialect.cpp
  MComp/MCompOps.cpp

  DEPENDS
  MCompTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRSideEffectInterfaces
)

target_include_directories(MCompDialect PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

add_mlir_library(MCompPasses
  Passes/FuseEltwise.cpp

  DEPENDS
  MCompTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSideEffectInterfaces
  MCompDialect
)

target_include_directories(MCompPasses PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

add_mlir_conversion_library(MCompToStd
  Conversion/MCompToStd.cpp

  DEPENDS
  MCompTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MCompDialect
)

target_include_directories(MCompToStd PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)