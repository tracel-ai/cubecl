//! Linking AMDGPU objects with LLD.
//!
//! LLD exposes no C API, so `cpp_shims/lld.cpp` wraps `lld::elf::link` in an
//! `extern "C"` entry point that `build.rs` compiles and links in.

use std::ffi::{CString, c_char};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

unsafe extern "C" {
    /// See `cpp_shims/lld.cpp`. Returns `true` when the link succeeded.
    fn cubecl_lld_elf_link(argv: *const *const c_char, argc: usize) -> bool;
}

/// LLD keeps global linker context, so concurrent calls corrupt each other.
static LLD_LOCK: Mutex<()> = Mutex::new(());

/// Keeps temp directories unique. The pid alone collides between threads, letting one
/// call's cleanup delete a sibling's directory.
static CALL_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Links a relocatable AMDGPU ELF into a loadable `ET_DYN` code object.
///
/// LLD writes to a path rather than a buffer, so files are staged in a temp dir. The kernel
/// name is not part of that path: it comes from a user's own types and carries whatever
/// `::`, `<` and `>` those spell. The pid and counter already make the directory unique, and
/// the name is what the error says rather than what the filesystem sees.
pub fn link_relocatable(object: &[u8], name: &str) -> Result<Vec<u8>, String> {
    let unique = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("cubecl-lld-{}-{unique}", std::process::id()));
    let result = link_in(&dir, object, name);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

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
