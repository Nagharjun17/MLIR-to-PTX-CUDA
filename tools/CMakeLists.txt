# mcomp.tools/CMakeLists.txt

add_executable(mcomp-opt mcomp-opt.cpp)

target_include_directories(mcomp-opt PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/lib   # for generated *.inc from TableGen
)

# Link against our libs + core MLIR stuff
target_link_libraries(mcomp-opt PRIVATE
  MCompDialect
  MCompPasses
  MCompToStd

  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRAsmParser
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
)

target_link_options(mcomp-opt PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:LINKER:--whole-archive>
)
target_link_libraries(mcomp-opt PRIVATE
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRControlFlowToLLVM
)
target_link_options(mcomp-opt PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:LINKER:--no-whole-archive>
)