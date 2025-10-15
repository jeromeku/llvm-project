	.file	"simple.c"
	.text
	.globl	foo                             # -- Begin function foo
	.p2align	4
	.type	foo,@function
foo:                                    # @foo
	.cfi_startproc
# %bb.0:
	testl	%esi, %esi
	jle	.LBB0_1
# %bb.2:
	movl	%esi, %ecx
	cmpl	$8, %esi
	jae	.LBB0_4
# %bb.3:
	xorl	%edx, %edx
	xorl	%eax, %eax
	jmp	.LBB0_7
.LBB0_1:
	xorl	%eax, %eax
	retq
.LBB0_4:
	movl	%ecx, %edx
	andl	$2147483640, %edx               # imm = 0x7FFFFFF8
	movl	%ecx, %eax
	shrl	$3, %eax
	andl	$268435455, %eax                # imm = 0xFFFFFFF
	shlq	$5, %rax
	pxor	%xmm0, %xmm0
	xorl	%esi, %esi
	pxor	%xmm1, %xmm1
	.p2align	4
.LBB0_5:                                # =>This Inner Loop Header: Depth=1
	movdqa	%xmm0, %xmm2
	movdqu	(%rdi,%rsi), %xmm0
	movdqa	%xmm1, %xmm3
	movdqu	16(%rdi,%rsi), %xmm1
	paddd	%xmm0, %xmm2
	paddd	%xmm0, %xmm0
	paddd	%xmm2, %xmm0
	paddd	%xmm1, %xmm3
	paddd	%xmm1, %xmm1
	paddd	%xmm3, %xmm1
	addq	$32, %rsi
	cmpq	%rsi, %rax
	jne	.LBB0_5
# %bb.6:
	paddd	%xmm0, %xmm1
	pshufd	$238, %xmm1, %xmm0              # xmm0 = xmm1[2,3,2,3]
	paddd	%xmm1, %xmm0
	pshufd	$85, %xmm0, %xmm1               # xmm1 = xmm0[1,1,1,1]
	paddd	%xmm0, %xmm1
	movd	%xmm1, %eax
	cmpl	%ecx, %edx
	je	.LBB0_8
	.p2align	4
.LBB0_7:                                # =>This Inner Loop Header: Depth=1
	movl	(%rdi,%rdx,4), %esi
	leal	(%rsi,%rsi,2), %esi
	addl	%esi, %eax
	incq	%rdx
	cmpq	%rdx, %rcx
	jne	.LBB0_7
.LBB0_8:
	retq
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 22.0.0 (++20251007042053+9cbcc87f5b22-1~exp1~20251007042222.2713)"
	.section	".note.GNU-stack","",@progbits
