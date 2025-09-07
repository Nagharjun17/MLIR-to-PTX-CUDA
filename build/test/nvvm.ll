; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_4xf32 = private constant [4 x float] zeroinitializer, align 64

define float @scalar(float %0, float %1) {
  %3 = fadd float %0, %1
  %4 = call float @llvm.maximum.f32(float %3, float 0.000000e+00)
  ret float %4
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @tensor_rhs_zero(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9) {
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %5, 0
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, ptr %6, 1
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, i64 %7, 2
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, i64 %8, 3, 0
  %15 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, i64 %9, 4, 0
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %1, 1
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0
  br label %21

21:                                               ; preds = %24, %10
  %22 = phi i64 [ %34, %24 ], [ 0, %10 ]
  %23 = icmp slt i64 %22, 4
  br i1 %23, label %24, label %35

24:                                               ; preds = %21
  %25 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %26 = getelementptr inbounds nuw float, ptr %25, i64 %22
  %27 = load float, ptr %26, align 4
  %28 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 1
  %29 = getelementptr inbounds nuw float, ptr %28, i64 %22
  %30 = load float, ptr %29, align 4
  %31 = fadd float %27, %30
  %32 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %33 = getelementptr inbounds nuw float, ptr %32, i64 %22
  store float %31, ptr %33, align 4
  %34 = add i64 %22, 1
  br label %21

35:                                               ; preds = %21
  br label %36

36:                                               ; preds = %39, %35
  %37 = phi i64 [ %48, %39 ], [ 0, %35 ]
  %38 = icmp slt i64 %37, 4
  br i1 %38, label %39, label %49

39:                                               ; preds = %36
  %40 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %41 = getelementptr inbounds nuw float, ptr %40, i64 %37
  %42 = load float, ptr %41, align 4
  %43 = getelementptr inbounds nuw float, ptr @__constant_4xf32, i64 %37
  %44 = load float, ptr %43, align 4
  %45 = call float @llvm.maximum.f32(float %42, float %44)
  %46 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %47 = getelementptr inbounds nuw float, ptr %46, i64 %37
  store float %45, ptr %47, align 4
  %48 = add i64 %37, 1
  br label %36

49:                                               ; preds = %36
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %20
}

define void @_mlir_ciface_tensor_rhs_zero(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %5 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 0
  %6 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 1
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 2
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 3, 0
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 4, 0
  %10 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 2
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 3, 0
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 4, 0
  %16 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @tensor_rhs_zero(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15)
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %0, align 8
  ret void
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @tensor_lhs_zero(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9) {
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %5, 0
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, ptr %6, 1
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, i64 %7, 2
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, i64 %8, 3, 0
  %15 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, i64 %9, 4, 0
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %1, 1
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0
  br label %21

21:                                               ; preds = %24, %10
  %22 = phi i64 [ %34, %24 ], [ 0, %10 ]
  %23 = icmp slt i64 %22, 4
  br i1 %23, label %24, label %35

24:                                               ; preds = %21
  %25 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %26 = getelementptr inbounds nuw float, ptr %25, i64 %22
  %27 = load float, ptr %26, align 4
  %28 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 1
  %29 = getelementptr inbounds nuw float, ptr %28, i64 %22
  %30 = load float, ptr %29, align 4
  %31 = fadd float %27, %30
  %32 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %33 = getelementptr inbounds nuw float, ptr %32, i64 %22
  store float %31, ptr %33, align 4
  %34 = add i64 %22, 1
  br label %21

35:                                               ; preds = %21
  br label %36

36:                                               ; preds = %39, %35
  %37 = phi i64 [ %48, %39 ], [ 0, %35 ]
  %38 = icmp slt i64 %37, 4
  br i1 %38, label %39, label %49

39:                                               ; preds = %36
  %40 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %41 = getelementptr inbounds nuw float, ptr %40, i64 %37
  %42 = load float, ptr %41, align 4
  %43 = getelementptr inbounds nuw float, ptr @__constant_4xf32, i64 %37
  %44 = load float, ptr %43, align 4
  %45 = call float @llvm.maximum.f32(float %42, float %44)
  %46 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %47 = getelementptr inbounds nuw float, ptr %46, i64 %37
  store float %45, ptr %47, align 4
  %48 = add i64 %37, 1
  br label %36

49:                                               ; preds = %36
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %20
}

define void @_mlir_ciface_tensor_lhs_zero(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %5 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 0
  %6 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 1
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 2
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 3, 0
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 4, 0
  %10 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 2
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 3, 0
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 4, 0
  %16 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @tensor_lhs_zero(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15)
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
