use std::{
    ffi::{CStr, c_char},
    sync::{PoisonError, RwLock},
};

/// The options this crate set, which LLVM keeps in global state. LLD sets every option back to
/// its default on each link, so a link holds the lock exclusively ([`reset_by`]) and a
/// compile that reads an option holds it shared ([`read_llvm_options`]).
static OPTIONS: RwLock<Vec<(&CStr, &CStr)>> = RwLock::new(Vec::new());

/// Sets the registered LLVM option `name` to `value`, as `-name=value` would. Returns `false`
/// when this LLVM has no such option or the option refuses the value, rather than exiting the
/// process the way `LLVMParseCommandLineOptions` does.
#[cfg(feature = "nvptx")]
pub(crate) fn set_llvm_option(name: &'static CStr, value: &'static CStr) -> bool {
    let mut options = OPTIONS.write().unwrap_or_else(PoisonError::into_inner);
    let set = set_raw(name, value);
    if set {
        options.push((name, value));
    }
    set
}

/// Keeps the options this crate set while the guard lives.
#[cfg(feature = "nvptx")]
pub(crate) fn read_llvm_options() -> impl Sized {
    OPTIONS.read().unwrap_or_else(PoisonError::into_inner)
}

/// Runs `reset`, which sets LLVM's options back to their defaults, then sets this crate's again.
#[cfg(feature = "amdgpu")]
pub(crate) fn reset_by<R>(reset: impl FnOnce() -> R) -> R {
    let options = OPTIONS.write().unwrap_or_else(PoisonError::into_inner);
    let result = reset();
    for (name, value) in options.iter() {
        set_raw(name, value);
    }
    result
}

fn set_raw(name: &CStr, value: &CStr) -> bool {
    // SAFETY: both strings are NUL-terminated and outlive the call; the shim only reads them.
    unsafe { cubecl_set_llvm_option(name.as_ptr(), value.as_ptr()) }
}

unsafe extern "C" {
    fn cubecl_set_llvm_option(name: *const c_char, value: *const c_char) -> bool;
}
