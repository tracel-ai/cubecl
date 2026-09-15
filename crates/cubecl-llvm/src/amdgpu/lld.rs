//! AMDGPU code object linking.

use std::{
    ffi::{CString, c_char},
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

unsafe extern "C" {
    fn cubecl_lld_elf_link(argv: *const *const c_char, argc: usize) -> bool;
}

/// LLD uses global state and requires serialized access.
static LLD_LOCK: Mutex<()> = Mutex::new(());

/// Unique identifiers for temporary directories.
static CALL_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Links an AMDGPU object into a loadable code object.
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
