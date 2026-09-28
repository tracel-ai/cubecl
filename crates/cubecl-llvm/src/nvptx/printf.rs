//! CUDA device printing.

use llvm_sys::{
    core::*,
    prelude::{LLVMModuleRef, LLVMTypeRef, LLVMValueRef},
};

const VPRINTF: &str = "vprintf";

/// # Safety
/// `module` must be a live LLVM module.
pub unsafe fn lower_printf_to_vprintf(module: LLVMModuleRef) -> Result<(), String> {
    unsafe {
        let printf = LLVMGetNamedFunction(module, c"printf".as_ptr());
        if printf.is_null() {
            return Ok(());
        }

        let mut calls = Vec::new();
        let mut use_ = LLVMGetFirstUse(printf);
        while !use_.is_null() {
            let user = LLVMGetUser(use_);
            use_ = LLVMGetNextUse(use_);
            if !LLVMIsACallInst(user).is_null() {
                calls.push(user);
            }
        }
        if calls.is_empty() {
            return Ok(());
        }

        let ctx = LLVMGetModuleContext(module);
        let i32_ty = LLVMInt32TypeInContext(ctx);
        let ptr_ty = LLVMPointerTypeInContext(ctx, 0);
        let mut vprintf_params = [ptr_ty, ptr_ty];
        let vprintf_ty = LLVMFunctionType(i32_ty, vprintf_params.as_mut_ptr(), 2, 0);
        let vprintf = declare(module, VPRINTF, vprintf_ty)?;

        for call in calls {
            rewrite(ctx, call, vprintf, vprintf_ty, ptr_ty);
        }

        LLVMDeleteFunction(printf);
        Ok(())
    }
}

/// # Safety
/// `call` must be a live call to a variadic `printf`.
unsafe fn rewrite(
    ctx: llvm_sys::prelude::LLVMContextRef,
    call: LLVMValueRef,
    vprintf: LLVMValueRef,
    vprintf_ty: LLVMTypeRef,
    ptr_ty: LLVMTypeRef,
) {
    unsafe {
        let builder = LLVMCreateBuilderInContext(ctx);
        LLVMPositionBuilderBefore(builder, call);

        let format = LLVMGetOperand(call, 0);
        let arg_count = LLVMGetNumArgOperands(call);
        let args: Vec<LLVMValueRef> = (1..arg_count).map(|i| LLVMGetOperand(call, i)).collect();

        let buffer = if args.is_empty() {
            LLVMConstPointerNull(ptr_ty)
        } else {
            let mut field_tys: Vec<LLVMTypeRef> = args.iter().map(|&a| LLVMTypeOf(a)).collect();
            // `vprintf` requires natural alignment for each argument.
            let buffer_ty =
                LLVMStructTypeInContext(ctx, field_tys.as_mut_ptr(), field_tys.len() as u32, 0);
            let buffer = LLVMBuildAlloca(builder, buffer_ty, c"printf_args".as_ptr());
            for (index, &arg) in args.iter().enumerate() {
                let field =
                    LLVMBuildStructGEP2(builder, buffer_ty, buffer, index as u32, c"".as_ptr());
                LLVMBuildStore(builder, arg, field);
            }
            buffer
        };

        let mut vprintf_args = [format, buffer];
        let replacement = LLVMBuildCall2(
            builder,
            vprintf_ty,
            vprintf,
            vprintf_args.as_mut_ptr(),
            2,
            c"".as_ptr(),
        );

        LLVMReplaceAllUsesWith(call, replacement);
        LLVMInstructionEraseFromParent(call);
        LLVMDisposeBuilder(builder);
    }
}

/// # Safety
/// `module` must be a live LLVM module.
unsafe fn declare(
    module: LLVMModuleRef,
    name: &str,
    fn_ty: LLVMTypeRef,
) -> Result<LLVMValueRef, String> {
    unsafe {
        let c_name =
            std::ffi::CString::new(name).map_err(|_| format!("name '{name}' contains a NUL"))?;
        let existing = LLVMGetNamedFunction(module, c_name.as_ptr());
        if !existing.is_null() {
            return Ok(existing);
        }
        Ok(LLVMAddFunction(module, c_name.as_ptr(), fn_ty))
    }
}
