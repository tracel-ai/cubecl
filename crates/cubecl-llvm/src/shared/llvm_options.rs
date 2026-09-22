use std::ffi::{CStr, c_char};

/// Sets the registered LLVM option `name` to `value`, as `-name=value` would. Returns `false`
/// when this LLVM has no such option or the option refuses the value, rather than exiting the
/// process the way `LLVMParseCommandLineOptions` does.
pub(crate) fn set_llvm_option(name: &CStr, value: &CStr) -> bool {
    // SAFETY: both strings are NUL-terminated and outlive the call; the shim only reads them.
    unsafe { cubecl_set_llvm_option(name.as_ptr(), value.as_ptr()) }
}

unsafe extern "C" {
    fn cubecl_set_llvm_option(name: *const c_char, value: *const c_char) -> bool;
}
