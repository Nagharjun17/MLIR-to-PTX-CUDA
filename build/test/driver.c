#include <stdio.h>
#include <stdint.h>

typedef struct {
  float   *allocated;
  float   *aligned;
  int64_t  offset;
  int64_t  sizes[1];
  int64_t  strides[1];
} StridedMemRefType_f32_1d;

extern float scalar(float, float);
extern void _mlir_ciface_tensor_rhs_zero(StridedMemRefType_f32_1d *out,
                                         StridedMemRefType_f32_1d *a,
                                         StridedMemRefType_f32_1d *b);
extern void _mlir_ciface_tensor_lhs_zero(StridedMemRefType_f32_1d *out,
                                         StridedMemRefType_f32_1d *a,
                                         StridedMemRefType_f32_1d *b);

static StridedMemRefType_f32_1d make_memref(float *data, int64_t n) {
  StridedMemRefType_f32_1d m;
  m.allocated = m.aligned = data;
  m.offset = 0; m.sizes[0] = n; m.strides[0] = 1;
  return m;
}
static void print_memref(const char *tag, StridedMemRefType_f32_1d *m) {
  float *p = m->aligned + m->offset;
  printf("%s: [%f, %f, %f, %f]\n", tag, p[0], p[1], p[2], p[3]);
}

int main(void) {
  printf("scalar(2.0,-1.5) = %f\n", scalar(2.0f, -1.5f));

  const float A0[4] = {1, -2, 3, -4};
  const float B0[4] = {5,  6, -7,  8};

  float a1[4], b1[4], out1[4] = {0};
  for (int i = 0; i < 4; ++i) { a1[i] = A0[i]; b1[i] = B0[i]; }
  StridedMemRefType_f32_1d A = make_memref(a1, 4);
  StridedMemRefType_f32_1d B = make_memref(b1, 4);
  StridedMemRefType_f32_1d OUT = make_memref(out1, 4);
  _mlir_ciface_tensor_rhs_zero(&OUT, &A, &B);
  print_memref("tensor_rhs_zero", &OUT);

  float a2[4], b2[4], out2[4] = {0};
  for (int i = 0; i < 4; ++i) { a2[i] = A0[i]; b2[i] = B0[i]; }
  A = make_memref(a2, 4);
  B = make_memref(b2, 4);
  OUT = make_memref(out2, 4);
  _mlir_ciface_tensor_lhs_zero(&OUT, &A, &B);
  print_memref("tensor_lhs_zero", &OUT);
  return 0;
}
