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
  %22 = phi i64 [ %46, %24 ], [ 0, %10 ]
  %23 = icmp slt i64 %22, 4
  br i1 %23, label %24, label %47

24:                                               ; preds = %21
  %25 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %26 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %27 = getelementptr float, ptr %25, i64 %26
  %28 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %29 = mul nuw nsw i64 %22, %28
  %30 = getelementptr inbounds nuw float, ptr %27, i64 %29
  %31 = load float, ptr %30, align 4
  %32 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 1
  %33 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 2
  %34 = getelementptr float, ptr %32, i64 %33
  %35 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 4, 0
  %36 = mul nuw nsw i64 %22, %35
  %37 = getelementptr inbounds nuw float, ptr %34, i64 %36
  %38 = load float, ptr %37, align 4
  %39 = fadd float %31, %38
  %40 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %41 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %42 = getelementptr float, ptr %40, i64 %41
  %43 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %44 = mul nuw nsw i64 %22, %43
  %45 = getelementptr inbounds nuw float, ptr %42, i64 %44
  store float %39, ptr %45, align 4
  %46 = add i64 %22, 1
  br label %21

47:                                               ; preds = %21
  br label %48

48:                                               ; preds = %51, %47
  %49 = phi i64 [ %68, %51 ], [ 0, %47 ]
  %50 = icmp slt i64 %49, 4
  br i1 %50, label %51, label %69

51:                                               ; preds = %48
  %52 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %53 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %54 = getelementptr float, ptr %52, i64 %53
  %55 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %56 = mul nuw nsw i64 %49, %55
  %57 = getelementptr inbounds nuw float, ptr %54, i64 %56
  %58 = load float, ptr %57, align 4
  %59 = getelementptr inbounds nuw float, ptr @__constant_4xf32, i64 %49
  %60 = load float, ptr %59, align 4
  %61 = call float @llvm.maximum.f32(float %58, float %60)
  %62 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %63 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %64 = getelementptr float, ptr %62, i64 %63
  %65 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %66 = mul nuw nsw i64 %49, %65
  %67 = getelementptr inbounds nuw float, ptr %64, i64 %66
  store float %61, ptr %67, align 4
  %68 = add i64 %49, 1
  br label %48

69:                                               ; preds = %48
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
  %22 = phi i64 [ %46, %24 ], [ 0, %10 ]
  %23 = icmp slt i64 %22, 4
  br i1 %23, label %24, label %47

24:                                               ; preds = %21
  %25 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %26 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %27 = getelementptr float, ptr %25, i64 %26
  %28 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %29 = mul nuw nsw i64 %22, %28
  %30 = getelementptr inbounds nuw float, ptr %27, i64 %29
  %31 = load float, ptr %30, align 4
  %32 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 1
  %33 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 2
  %34 = getelementptr float, ptr %32, i64 %33
  %35 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 4, 0
  %36 = mul nuw nsw i64 %22, %35
  %37 = getelementptr inbounds nuw float, ptr %34, i64 %36
  %38 = load float, ptr %37, align 4
  %39 = fadd float %31, %38
  %40 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %41 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %42 = getelementptr float, ptr %40, i64 %41
  %43 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %44 = mul nuw nsw i64 %22, %43
  %45 = getelementptr inbounds nuw float, ptr %42, i64 %44
  store float %39, ptr %45, align 4
  %46 = add i64 %22, 1
  br label %21

47:                                               ; preds = %21
  br label %48

48:                                               ; preds = %51, %47
  %49 = phi i64 [ %68, %51 ], [ 0, %47 ]
  %50 = icmp slt i64 %49, 4
  br i1 %50, label %51, label %69

51:                                               ; preds = %48
  %52 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %53 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %54 = getelementptr float, ptr %52, i64 %53
  %55 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %56 = mul nuw nsw i64 %49, %55
  %57 = getelementptr inbounds nuw float, ptr %54, i64 %56
  %58 = load float, ptr %57, align 4
  %59 = getelementptr inbounds nuw float, ptr @__constant_4xf32, i64 %49
  %60 = load float, ptr %59, align 4
  %61 = call float @llvm.maximum.f32(float %58, float %60)
  %62 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %63 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %64 = getelementptr float, ptr %62, i64 %63
  %65 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %66 = mul nuw nsw i64 %49, %65
  %67 = getelementptr inbounds nuw float, ptr %64, i64 %66
  store float %61, ptr %67, align 4
  %68 = add i64 %49, 1
  br label %48

69:                                               ; preds = %48
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
