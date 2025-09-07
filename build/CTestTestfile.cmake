# CMake generated Testfile for 
# Source directory: /root/mlir-learning/mcomp
# Build directory: /root/mlir-learning/mcomp/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[mcomp-opt-help]=] "/root/mlir-learning/mcomp/build/tools/mcomp-opt" "--help")
set_tests_properties([=[mcomp-opt-help]=] PROPERTIES  _BACKTRACE_TRIPLES "/root/mlir-learning/mcomp/CMakeLists.txt;33;add_test;/root/mlir-learning/mcomp/CMakeLists.txt;0;")
subdirs("lib")
subdirs("tools")
