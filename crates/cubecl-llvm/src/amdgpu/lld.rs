//! Linking AMDGPU objects with LLD, called straight from Rust.
//!
//! LLD publishes no C API, so this declares its C++ entry point by mangled
//! name. That is only tractable because of one detail: `lld::elf::link` takes
//! two `llvm::raw_ostream &` arguments, which cannot be constructed from Rust
//! — but `llvm::nulls()` and `llvm::errs()` return references to process-wide
//! ones, so we borrow rather than build.
//!
//! The mangled names are Itanium ABI, so the real implementation below is
//! `#[cfg(unix)]`-only; MSVC mangles differently. Non-unix targets get a
//! [`link_relocatable`] that always returns `Err` rather than failing to
//! compile, since `cubecl-llvm` is a dependency of the cross-platform
//! `cubecl-cpu`.

#[cfg(unix)]
mod unix_impl {
    use std::ffi::{CString, c_char, c_void};
    use std::path::Path;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// `llvm::ArrayRef<const char *>` — a POD {pointer, length} pair. The `SysV`
    /// classifier passes it in two integer registers, which `#[repr(C)]`
    /// reproduces exactly.
    #[repr(C)]
    struct ArrayRefCStr {
        data: *const *const c_char,
        length: usize,
    }

    unsafe extern "C" {
        /// `llvm::raw_ostream &llvm::nulls()`
        #[link_name = "_ZN4llvm5nullsEv"]
        fn llvm_nulls() -> *mut c_void;

        /// `llvm::raw_ostream &llvm::errs()`
        #[link_name = "_ZN4llvm4errsEv"]
        fn llvm_errs() -> *mut c_void;

        /// `bool lld::elf::link(llvm::ArrayRef<const char *>, llvm::raw_ostream &,
        ///                      llvm::raw_ostream &, bool exitEarly,
        ///                      bool disableOutput)`
        #[link_name = "_ZN3lld3elf4linkEN4llvm8ArrayRefIPKcEERNS1_11raw_ostreamES7_bb"]
        fn lld_elf_link(
            args: ArrayRefCStr,
            stdout_os: *mut c_void,
            stderr_os: *mut c_void,
            exit_early: bool,
            disable_output: bool,
        ) -> bool;
    }

    /// LLD keeps global linker context, so concurrent calls corrupt each other.
    static LLD_LOCK: Mutex<()> = Mutex::new(());

    /// Disambiguates temp directories for concurrent calls that share a `name`.
    /// `(pid, name)` alone collides across threads in the same process: one
    /// call's `remove_dir_all` at the end could delete a still-running
    /// sibling's directory out from under it, or the two could overwrite each
    /// other's `kernel.o`.
    static CALL_COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// Links a relocatable AMDGPU ELF into a loadable `ET_DYN` code object.
    ///
    /// LLD's ELF driver writes to a path rather than a buffer, so both files
    /// are staged in a temp directory and the result is read back.
    ///
    /// On failure the message only says "see stderr above": LLD's diagnostics
    /// go to the process's real stderr via `llvm::errs()`, not into the
    /// returned `String`. Capturing them would need a custom C++
    /// `raw_ostream` subclass, which is exactly the shim this module avoids.
    /// A caller with captured/redirected stderr gets an error with no detail.
    pub fn link_relocatable(object: &[u8], name: &str) -> Result<Vec<u8>, String> {
        let unique = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("cubecl-lld-{}-{unique}-{name}", std::process::id()));
        let result = link_in(&dir, object, name);
        let _ = std::fs::remove_dir_all(&dir);
        result
    }

    /// The actual work, factored out so [`link_relocatable`] can run
    /// `remove_dir_all` unconditionally afterwards — including when this
    /// returns early via `?`, e.g. if the write fails after the directory was
    /// already created.
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
            // SAFETY: `argv` outlives the call; the two stream pointers are
            // process-wide LLVM statics. `exit_early = false` is what keeps a
            // link error from calling `exit()` and taking this process down.
            unsafe {
                lld_elf_link(
                    ArrayRefCStr {
                        data: argv.as_ptr(),
                        length: argv.len(),
                    },
                    llvm_nulls(),
                    llvm_errs(),
                    false,
                    false,
                )
            }
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

        /// Proves the LLD and LLVM support archives actually made it onto the
        /// link line. If `build.rs` gets the library order wrong this fails
        /// to link, and if the mangled names are wrong it fails here.
        #[test]
        fn llvm_ostreams_resolve() {
            // SAFETY: both return references to process-wide streams.
            unsafe {
                assert!(!llvm_nulls().is_null());
                assert!(!llvm_errs().is_null());
            }
        }

        /// A malformed object must come back as an error. LLD calls `exit()`
        /// on link failure unless `exitEarly` is false, so this test also
        /// proves we passed that flag correctly — if we did not, the test
        /// binary dies here instead of failing.
        #[test]
        fn rejects_garbage_object() {
            let err = link_relocatable(b"this is not an ELF", "garbage").unwrap_err();
            assert!(!err.is_empty());
        }

        /// Regression test for a temp-directory race: `(pid, name)` alone is
        /// not a unique directory key, so two concurrent calls sharing a
        /// `name` could interleave — one thread's cleanup or write stomping
        /// on another's files.
        ///
        /// There is no valid relocatable ELF available at this layer (Task 6
        /// produces those), so this cannot assert that linked *bytes* were
        /// never swapped between callers. What it can assert: every
        /// concurrent call gets an isolated directory, so the only error any
        /// of them ever produces is LLD's own "failed to link" — never a
        /// filesystem error from a directory that vanished or was rewritten
        /// underneath a sibling call (`temp dir: ...`, `write object: ...`,
        /// `read code object: ...`). Before the fix, sharing one directory
        /// per `name` made those spurious IO errors reachable under
        /// concurrency.
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
}

#[cfg(unix)]
pub use unix_impl::link_relocatable;

/// Non-unix stand-in: the real implementation depends on Itanium-mangled
/// symbol names that only exist under that ABI. This keeps `cubecl-llvm` —
/// and therefore the cross-platform `cubecl-cpu` that depends on it —
/// compiling on non-unix targets, and fails loudly at the one call site that
/// actually needs AMDGPU codegen rather than at link time for the whole crate.
#[cfg(not(unix))]
pub fn link_relocatable(_object: &[u8], _name: &str) -> Result<Vec<u8>, String> {
    Err(
        "AMDGPU linking requires LLD's Itanium-mangled symbols and is only supported on unix \
         targets"
            .to_string(),
    )
}
