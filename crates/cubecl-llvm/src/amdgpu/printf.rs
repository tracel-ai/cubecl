//! AMDGPU device printing.

use llvm_sys::core::*;
use llvm_sys::prelude::{LLVMModuleRef, LLVMValueRef};

unsafe extern "C" {
    /// Consumes the call. The caller must not use it afterwards.
    fn cubecl_emit_amdgpu_printf(call: LLVMValueRef);
}

/// # Safety
/// `module` must be a live LLVM module.
pub unsafe fn lower_printf_to_hostcall(module: LLVMModuleRef) -> bool {
    unsafe {
        let printf = LLVMGetNamedFunction(module, c"printf".as_ptr());
        if printf.is_null() {
            return false;
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

        let lowered = !calls.is_empty();
        for call in calls {
            cubecl_emit_amdgpu_printf(call);
        }

        if lowered && LLVMGetFirstUse(printf).is_null() {
            LLVMDeleteFunction(printf);
        }
        lowered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};

    unsafe fn parse(ir: &str) -> (llvm_sys::prelude::LLVMContextRef, LLVMModuleRef) {
        unsafe {
            let ctx = LLVMContextCreate();
            let text = CString::new(ir).unwrap();
            let buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(
                text.as_ptr(),
                ir.len(),
                c"test".as_ptr(),
            );
            let mut module = std::ptr::null_mut();
            let mut err = std::ptr::null_mut();
            assert_eq!(
                llvm_sys::ir_reader::LLVMParseIRInContext2(ctx, buffer, &mut module, &mut err),
                0,
                "{}",
                CStr::from_ptr(err).to_string_lossy()
            );
            (ctx, module)
        }
    }

    unsafe fn print(module: LLVMModuleRef) -> String {
        unsafe {
            let c = LLVMPrintModuleToString(module);
            let s = CStr::from_ptr(c).to_string_lossy().into_owned();
            LLVMDisposeMessage(c);
            s
        }
    }

    const WITH_PRINTF: &str = r#"
target triple = "amdgcn-amd-amdhsa"
@fmt = private unnamed_addr constant [16 x i8] c"Test value: %f\0A\00"
declare i32 @printf(ptr, ...)
define void @k(double %d) {
  %r = call i32 (ptr, ...) @printf(ptr @fmt, double %d)
  ret void
}
"#;

    #[test]
    fn printf_becomes_the_hostcall_sequence() {
        unsafe {
            let (ctx, module) = parse(WITH_PRINTF);
            assert!(lower_printf_to_hostcall(module));

            let ir = print(module);
            for expected in [
                "__ockl_printf_begin",
                "__ockl_printf_append_string_n",
                "__ockl_printf_append_args",
            ] {
                assert!(ir.contains(expected), "missing {expected} in:\n{ir}");
            }
            assert!(
                !ir.contains("@printf"),
                "the libc declaration should be gone:\n{ir}"
            );

            LLVMDisposeModule(module);
            LLVMContextDispose(ctx);
        }
    }

    #[test]
    fn a_module_without_printf_needs_nothing() {
        unsafe {
            let (ctx, module) = parse("define void @k() { ret void }");
            assert!(!lower_printf_to_hostcall(module));
            LLVMDisposeModule(module);
            LLVMContextDispose(ctx);
        }
    }

    #[test]
    fn a_declaration_without_a_call_needs_nothing() {
        unsafe {
            let (ctx, module) = parse("declare i32 @printf(ptr, ...)");
            assert!(!lower_printf_to_hostcall(module));
            LLVMDisposeModule(module);
            LLVMContextDispose(ctx);
        }
    }
}
