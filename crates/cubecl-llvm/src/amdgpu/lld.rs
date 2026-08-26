//! Linking AMDGPU objects with LLD.
//!
//! LLD exposes no C API, so `lld_shim.cpp` wraps `lld::elf::link` in an
//! `extern "C"` entry point that `build.rs` compiles and links in. Binding the
//! C++ symbol directly would mean hard-coding an Itanium-mangled name and
//! reproducing `llvm::ArrayRef`'s layout, and would not survive a change of
//! host compiler.

use std::ffi::{CString, c_char};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

unsafe extern "C" {
    /// See `lld_shim.cpp`. Returns `true` when the link succeeded.
    fn cubecl_lld_elf_link(argv: *const *const c_char, argc: usize) -> bool;
}

/// LLD keeps global linker context, so concurrent calls corrupt each other.
static LLD_LOCK: Mutex<()> = Mutex::new(());

/// Keeps temp directories unique. `(pid, name)` alone collides between threads
/// sharing a `name`, letting one call's cleanup delete a sibling's directory.
static CALL_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Links a relocatable AMDGPU ELF into a loadable `ET_DYN` code object.
///
/// LLD writes to a path rather than a buffer, so files are staged in a temp dir.
///
/// On failure the error says only "see stderr above": LLD's diagnostics go to the
/// process's real stderr, not into the returned `String`.
pub fn link_relocatable(object: &[u8], name: &str) -> Result<Vec<u8>, String> {
    let unique = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("cubecl-lld-{}-{unique}-{name}", std::process::id()));
    let result = link_in(&dir, object, name);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// Factored out so [`link_relocatable`] always runs `remove_dir_all`, including
/// on the `?` early returns.
fn link_in(dir: &Path, object: &[u8], name: &str) -> Result<Vec<u8>, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("temp dir: {e}"))?;
    let obj_path = dir.join("kernel.o");
    let out_path = dir.join("kernel.hsaco");
    std::fs::write(&obj_path, object).map_err(|e| format!("write object: {e}"))?;

    let args = [
        CString::new("ld.lld").unwrap(),
        CString::new("--shared").unwrap(),
        CString::new("-o").unwrap(),
        CString::new(out_path.to_string_lossy().as_ref()).unwrap(),
        CString::new(obj_path.to_string_lossy().as_ref()).unwrap(),
    ];
    let argv: Vec<*const c_char> = args.iter().map(|a| a.as_ptr()).collect();

    let ok = {
        let _guard = LLD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: `args` owns the strings and outlives `argv`, which outlives the call.
        unsafe { cubecl_lld_elf_link(argv.as_ptr(), argv.len()) }
    };

    if ok {
        std::fs::read(&out_path).map_err(|e| format!("read code object: {e}"))
    } else {
        Err(format!("lld failed to link '{name}'; see stderr above"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A malformed object must come back as an error rather than killing the
    /// process, which is what `exitEarly=false` in the shim buys. Reaching the
    /// assertion at all also proves the shim and the lld archives linked.
    #[test]
    fn rejects_garbage_object() {
        let err = link_relocatable(b"this is not an ELF", "garbage").unwrap_err();
        assert!(!err.is_empty());
    }

    /// Regression test for a temp-directory race: `(pid, name)` was not a unique
    /// key, so concurrent calls sharing a `name` stomped each other's files.
    ///
    /// No valid relocatable ELF exists at this layer, so this cannot assert that
    /// linked *bytes* were never swapped. It asserts the weaker property that
    /// still catches the bug: every call gets an isolated directory, so the only
    /// error is LLD's own — never a filesystem error from a vanished directory.
    #[test]
    fn concurrent_same_name_does_not_clobber() {
        let name = "same-name";
        let expected = format!("lld failed to link '{name}'; see stderr above");
        let handles: Vec<_> = (0..16)
            .map(|_| std::thread::spawn(move || link_relocatable(b"this is not an ELF", name)))
            .collect();
        for handle in handles {
            let err = handle.join().expect("worker thread panicked").unwrap_err();
            assert_eq!(err, expected);
        }
    }
}
