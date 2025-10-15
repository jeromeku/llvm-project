; ModuleID = 'simple.c'
source_filename = "simple.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define dso_local i32 @foo(ptr noundef readonly captures(none) %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp sgt i32 %1, 0
  br i1 %3, label %4, label %30

4:                                                ; preds = %2
  %5 = zext nneg i32 %1 to i64
  %6 = icmp ult i32 %1, 8
  br i1 %6, label %27, label %7

7:                                                ; preds = %4
  %8 = and i64 %5, 2147483640
  br label %9

9:                                                ; preds = %9, %7
  %10 = phi i64 [ 0, %7 ], [ %21, %9 ]
  %11 = phi <4 x i32> [ zeroinitializer, %7 ], [ %19, %9 ]
  %12 = phi <4 x i32> [ zeroinitializer, %7 ], [ %20, %9 ]
  %13 = getelementptr inbounds nuw i32, ptr %0, i64 %10
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 16
  %15 = load <4 x i32>, ptr %13, align 4, !tbaa !5
  %16 = load <4 x i32>, ptr %14, align 4, !tbaa !5
  %17 = mul nsw <4 x i32> %15, splat (i32 3)
  %18 = mul nsw <4 x i32> %16, splat (i32 3)
  %19 = add <4 x i32> %17, %11
  %20 = add <4 x i32> %18, %12
  %21 = add nuw i64 %10, 8
  %22 = icmp eq i64 %21, %8
  br i1 %22, label %23, label %9, !llvm.loop !9

23:                                               ; preds = %9
  %24 = add <4 x i32> %20, %19
  %25 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %24)
  %26 = icmp eq i64 %8, %5
  br i1 %26, label %30, label %27

27:                                               ; preds = %4, %23
  %28 = phi i64 [ 0, %4 ], [ %8, %23 ]
  %29 = phi i32 [ 0, %4 ], [ %25, %23 ]
  br label %32

30:                                               ; preds = %32, %23, %2
  %31 = phi i32 [ 0, %2 ], [ %25, %23 ], [ %38, %32 ]
  ret i32 %31

32:                                               ; preds = %27, %32
  %33 = phi i64 [ %39, %32 ], [ %28, %27 ]
  %34 = phi i32 [ %38, %32 ], [ %29, %27 ]
  %35 = getelementptr inbounds nuw i32, ptr %0, i64 %33
  %36 = load i32, ptr %35, align 4, !tbaa !5
  %37 = mul nsw i32 %36, 3
  %38 = add nsw i32 %37, %34
  %39 = add nuw nsw i64 %33, 1
  %40 = icmp eq i64 %39, %5
  br i1 %40, label %30, label %32, !llvm.loop !13
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) #1

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 22.0.0 (++20251007042053+9cbcc87f5b22-1~exp1~20251007042222.2713)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11, !12}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.isvectorized", i32 1}
!12 = !{!"llvm.loop.unroll.runtime.disable"}
!13 = distinct !{!13, !10, !12, !11}
