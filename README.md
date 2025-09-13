# MLIR → PTX (CUDA)

Tiny playground for a custom MLIR dialect (`mcomp`) with:
- a fuse pass: `arith.addf` + `max(..., 0.0)` → `mcomp.fuse_add_relu`
- a lowering pass back to standard/arith
- a pipeline down to LLVM IR and PTX

## Requirements

- A local **LLVM/MLIR build**
- CMake ≥ 3.20, Ninja, Clang/LLD
- CUDA toolkit / NVIDIA GPU to test PTX

`MLIR_DIR` and `LLVM_DIR` should be pointing at LLVM build’s CMake packages.

## Build

```bash
git clone https://github.com/Nagharjun17/MLIR-to-PTX-CUDA.git
cd MLIR-to-PTX-CUDA
mkdir -p build && cd build

export LLVM_ROOT=~/src/llvm-project/build

cmake -GNinja   -DMLIR_DIR=$LLVM_ROOT/lib/cmake/mlir   -DLLVM_DIR=$LLVM_ROOT/lib/cmake/llvm   -DCMAKE_PREFIX_PATH="$LLVM_ROOT"   ..

ninja
```

Builds `tools/mcomp-opt` (driver with passes registered).

## Quick test: fuse → arith

Input: `test/fuse_add_relu.mlir`

```bash
# Fuse add+relu into custom op
./tools/mcomp-opt test/fuse_add_relu.mlir   --mcomp-fuse-add-relu   -o test/after_fuse.mlir

# Lower custom op back to arith
./tools/mcomp-opt /tmp/after_fuse.mlir   --convert-mcomp-to-std   -o test/back_to_arith.mlir
```

Open the outputs to see `mcomp.fuse_add_relu` in the first, and `arith.addf` + `arith.maximumf` in the second.

## Lower to LLVM IR (CPU path)

```bash
./tools/mcomp-opt test/fuse_add_relu.mlir   --mcomp-fuse-add-relu   --convert-mcomp-to-std | $LLVM_ROOT/bin/mlir-opt   --canonicalize --cse   --convert-elementwise-to-linalg   --one-shot-bufferize="bufferize-function-boundaries=1"   --convert-linalg-to-loops   --convert-scf-to-cf   --convert-to-llvm   --reconcile-unrealized-casts   -o test/llvm_dialect.mlir

$LLVM_ROOT/bin/mlir-translate   --mlir-to-llvmir test/llvm_dialect.mlir -o /test/module.ll
```

We can then `clang -O2 -c test/module.ll -o test/module.o` and link with `test/driver.c` for a CPU run.

## Lower to PTX (GPU path)

Goes through GPU → NVVM → LLVM → PTX.

```bash
./tools/mcomp-opt test/fuse_add_relu.mlir   --mcomp-fuse-add-relu   --convert-mcomp-to-std | $LLVM_ROOT/bin/mlir-opt   --canonicalize --cse   --convert-elementwise-to-linalg   --one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map"   --convert-linalg-to-parallel-loops   --convert-parallel-loops-to-gpu   --gpu-kernel-outlining   --convert-scf-to-cf   --convert-gpu-to-nvvm   --convert-math-to-llvm   --convert-to-llvm   --reconcile-unrealized-casts   -o test/nvvm.mlir

$LLVM_ROOT/bin/mlir-translate   --mlir-to-llvmir test/nvvm.mlir -o test/nvvm.ll

$LLVM_ROOT/bin/llc   -mtriple=nvptx64-nvidia-cuda   -mcpu=sm_86   -filetype=asm   test/nvvm.ll -o test/kernel.ptx
```

Open `test/kernel.ptx` — you should see the NVPTX header.

Change `-mcpu=sm_86` to match your GPU (e.g., `sm_75`, `sm_90`).
