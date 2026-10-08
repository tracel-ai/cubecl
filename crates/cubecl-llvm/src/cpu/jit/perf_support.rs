//! The perf support plugin of ORC, through the C++ shim `cpu/cpp_shims/perf_support.cpp`.
//!
//! The LLVM C API has no binding for the plugin, and `pliron-llvm` does not add one. This module
//! is the only user of the shim.

use crate::shared::llvm_module::error_message;
use llvm_sys::{error::LLVMErrorRef, orc2::lljit::LLVMOrcLLJITRef};
use std::{process::Command, sync::OnceLock};

/// Adds the perf support plugin to `jit`. The plugin writes `jit-<pid>.dump` in
/// `$JITDUMPDIR/.debug/jit`, with the line table (`debug_info`) and the unwind data
/// (`unwind_info`) of each symbol. The shim fixes the records for perf, see
/// [`perf_adds_header_offset`].
///
/// # Errors
/// The message LLVM gives, when the JIT does not link with `JITLink`, the object format is not ELF,
/// or the jitdump cannot be opened (for example, not on Linux).
///
/// # Safety
/// `jit` must be a live LLJIT. Call it before the JIT links an object.
pub(crate) unsafe fn enable_perf_support(
    jit: LLVMOrcLLJITRef,
    debug_info: bool,
    unwind_info: bool,
) -> Result<(), String> {
    // SAFETY: the caller's contract. The shim returns an error that `error_message` takes.
    error_message(unsafe {
        cubecl_orc_lljit_enable_perf_support(
            jit,
            debug_info,
            unwind_info,
            perf_adds_header_offset(),
        )
    })
}

/// Whether `perf inject` adds the 0x40 bytes of its ELF header to the line addresses itself. The
/// perf support plugin also adds them, so the shim removes them again. perf adds them since Linux
/// 5.8. Before, only the plugin's offset puts the lines in the right place.
///
/// The check runs `perf --version` one time for each process, and only when the jitdump is asked
/// for. It takes approximately 10 ms. Without `perf` on the `PATH`, perf is assumed to be 5.8 or
/// newer. The check reads the perf of the process host. If a different perf reads the jitdump,
/// the lines can be 0x40 bytes wrong.
pub(super) fn perf_adds_header_offset() -> bool {
    static ADDS: OnceLock<bool> = OnceLock::new();
    *ADDS.get_or_init(|| {
        Command::new("perf")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| perf_version(&String::from_utf8_lossy(&output.stdout)))
            .is_none_or(|version| version >= (5, 8))
    })
}

/// The major and minor version in the output of `perf --version`, such as
/// `perf version 6.8.0-45-generic`.
fn perf_version(output: &str) -> Option<(u32, u32)> {
    let version = output.trim().strip_prefix("perf version ")?;
    let mut numbers = version.split(|c: char| !c.is_ascii_digit());
    let major = numbers.next()?.parse().ok()?;
    let minor = numbers.next()?.parse().ok()?;
    Some((major, minor))
}

unsafe extern "C" {
    /// Returns null on success, or an error that the caller owns.
    fn cubecl_orc_lljit_enable_perf_support(
        jit: LLVMOrcLLJITRef,
        emit_debug_info: bool,
        emit_unwind_info: bool,
        remove_offset: bool,
    ) -> LLVMErrorRef;
}

#[cfg(test)]
mod tests {
    use super::perf_version;

    #[test]
    fn the_perf_version_is_read() {
        assert_eq!(
            perf_version("perf version 7.2.8-200.fc44.x86_64\n"),
            Some((7, 2))
        );
        assert_eq!(perf_version("perf version 5.4.0"), Some((5, 4)));
        assert_eq!(perf_version("perf version 6.8.0-45-generic"), Some((6, 8)));
        assert_eq!(perf_version("perf version 6"), None);
        assert_eq!(perf_version("WARNING: perf not found for kernel 6.8"), None);
    }
}
