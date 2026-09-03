//! Lowering of `printf` to the AMDGPU hostcall sequence.
//!
//! A device talks to the host rather than calling libc: a handle from `__ockl_printf_begin`, the
//! format string, then the arguments widened to 64 bits. `cpp_shims/printf.cpp` reaches LLVM's own
//! emitter for it.

use llvm_sys::core::*;
use llvm_sys::prelude::{LLVMModuleRef, LLVMValueRef};

unsafe extern "C" {
    /// See `cpp_shims/printf.cpp`. Consumes the call, which must not be used afterwards.
    fn cubecl_emit_amdgpu_printf(call: LLVMValueRef);
}

/// Rewrites every `printf` call in `module` into the hostcall sequence.
///
/// Returns whether anything was rewritten, i.e. whether OCKL has to be linked in behind it. A
/// kernel that never prints asks for nothing.
///
/// # Safety
/// `module` must be a live LLVM module.
pub unsafe fn lower_printf_to_hostcall(module: LLVMModuleRef) -> bool {
    unsafe {
        let printf = LLVMGetNamedFunction(module, c"printf".as_ptr());
        if printf.is_null() {
            return false;
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

        let lowered = !calls.is_empty();
        for call in calls {
            cubecl_emit_amdgpu_printf(call);
        }

        // The declaration is left with no uses; a GPU links no libc.
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

    /// Parses `ir` into a fresh context the caller must dispose.
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

    /// The module's textual form, for asserting on what the rewrite produced.
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

    /// The call becomes the hostcall conversation, and the `printf` that a GPU has no answer
    /// for is gone from the module entirely.
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

    /// A kernel that never prints neither runs the rewrite nor drags OCKL in behind it.
    #[test]
    fn a_module_without_printf_needs_nothing() {
        unsafe {
            let (ctx, module) = parse("define void @k() { ret void }");
            assert!(!lower_printf_to_hostcall(module));
            LLVMDisposeModule(module);
            LLVMContextDispose(ctx);
        }
    }

    /// A `printf` that is declared but never called is not a reason to link anything.
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
