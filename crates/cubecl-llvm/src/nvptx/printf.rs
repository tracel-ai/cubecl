//! Lowering of `printf` to CUDA's `vprintf`.
//!
//! PTX has no variadic calls, so the variadic `printf` the shared lowering emits reaches the
//! driver as a `.extern .func` it refuses — a `CUDA_ERROR_INVALID_PTX` at load time rather than
//! anything the compiler catches. What the device actually has is
//! `int vprintf(const char *format, void *args)`, taking the arguments packed into a buffer,
//! and rewriting into that is what NVCC does with a `printf` in device code too.
//!
//! The arguments arrive already widened by the C default argument promotions
//! ([`shared::to_llvm::general`](crate::shared::to_llvm::general)), which is the same
//! promotion `vprintf`'s buffer is read back with, so nothing further is converted here.

use llvm_sys::core::*;
use llvm_sys::prelude::{LLVMModuleRef, LLVMTypeRef, LLVMValueRef};

/// CUDA's device-side print entry point.
const VPRINTF: &str = "vprintf";

/// Rewrites every `printf` call in `module` into a `vprintf` call over a packed argument
/// buffer.
///
/// # Safety
/// `module` must be a live LLVM module.
pub unsafe fn lower_printf_to_vprintf(module: LLVMModuleRef) -> Result<(), String> {
    unsafe {
        let printf = LLVMGetNamedFunction(module, c"printf".as_ptr());
        if printf.is_null() {
            return Ok(());
        }

        // Collected first: each rewrite erases a call out of the use list being walked.
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

        // Nothing calls it any more, and a variadic declaration left behind is exactly what the
        // driver rejects.
        LLVMDeleteFunction(printf);
        Ok(())
    }
}

/// Replaces one `printf` call with the `vprintf` equivalent.
///
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

        // Operand 0 is the format; the rest are the promoted arguments. (The callee sits past
        // them in the operand list, which is why the count is one more than the arguments.)
        let format = LLVMGetOperand(call, 0);
        let arg_count = LLVMGetNumArgOperands(call);
        let args: Vec<LLVMValueRef> = (1..arg_count).map(|i| LLVMGetOperand(call, i)).collect();

        // A print with nothing to substitute passes a null buffer, which is what `vprintf` is
        // specified to take when the format has no conversions.
        let buffer = if args.is_empty() {
            LLVMConstPointerNull(ptr_ty)
        } else {
            let mut field_tys: Vec<LLVMTypeRef> = args.iter().map(|&a| LLVMTypeOf(a)).collect();
            // Unpacked, so every field lands on its natural alignment -- which is the layout
            // `vprintf` reads the buffer back with.
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

/// The function `name` in `module`, declared with `fn_ty` if it is not there yet.
///
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
