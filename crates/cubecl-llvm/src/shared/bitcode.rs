use llvm_sys::prelude::LLVMModuleRef;
use std::ffi::{CStr, c_char};

/// Link into `module` the definitions it calls from the bitcode library `bitcode`, and no others.
///
/// # Safety
/// `module` must be a live LLVM module.
pub(crate) unsafe fn link_bitcode(module: LLVMModuleRef, bitcode: &[u8]) -> Result<(), String> {
    // SAFETY: `module` is live, and the shim only reads `bitcode` for the length given.
    let err = unsafe { cubecl_link_device_bitcode(module, bitcode.as_ptr().cast(), bitcode.len()) };
    if err.is_null() {
        return Ok(());
    }
    // SAFETY: the shim returns a NUL-terminated `malloc`'d string we now own, freed once here.
    unsafe {
        let message = CStr::from_ptr(err).to_string_lossy().into_owned();
        cubecl_free_message(err);
        Err(message)
    }
}

unsafe extern "C" {
    /// Returns null on success or an owned error message.
    fn cubecl_link_device_bitcode(
        dest: LLVMModuleRef,
        data: *const c_char,
        len: usize,
    ) -> *mut c_char;

    fn cubecl_free_message(message: *mut c_char);
}
