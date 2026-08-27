//! Linking `ROCm`'s device libraries into a kernel.
//!
//! The AMDGPU backend has hardware behind only some of the float intrinsics. Linking happens before the optimization pipeline, and takes only the definitions the
//! kernel calls. `OCML`'s own functions are `linkonce_odr hidden`, so once inlined the
//! pipeline strips every one of them back out again.

use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use llvm_sys::prelude::LLVMModuleRef;

unsafe extern "C" {
    /// See `device_libs_shim.cpp`. Returns null on success, else an owned message.
    fn cubecl_link_device_bitcode(
        dest: LLVMModuleRef,
        data: *const c_char,
        len: usize,
    ) -> *mut c_char;

    /// Frees what `cubecl_link_device_bitcode` returned.
    fn cubecl_free_message(message: *mut c_char);
}

/// Where to look for `amdgcn/bitcode`, in the order a `ROCm` install makes them true.
///
/// `CUBECL_ROCM_DEVICE_LIB_PATH` names the directory itself and is the escape hatch for an
/// install none of the rest find. `HIP_DEVICE_LIB_PATH` is what hipcc itself reads.
const DEVICE_LIB_PATH_VARS: [&str; 2] = ["CUBECL_ROCM_DEVICE_LIB_PATH", "HIP_DEVICE_LIB_PATH"];
const ROCM_ROOT_VARS: [&str; 2] = ["ROCM_PATH", "HIP_PATH"];
const DEFAULT_ROCM_ROOTS: [&str; 2] = ["/opt/rocm", "/usr"];

/// The `amdgcn/bitcode` directory of the `ROCm` install, found once per process.
fn bitcode_dir() -> Result<&'static PathBuf, String> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

    DIR.get_or_init(|| {
        let direct = DEVICE_LIB_PATH_VARS.iter().filter_map(std::env::var_os).map(PathBuf::from);
        let roots = (ROCM_ROOT_VARS.iter())
            .filter_map(std::env::var_os)
            .map(PathBuf::from)
            .chain(DEFAULT_ROCM_ROOTS.iter().map(PathBuf::from))
            .map(|root| root.join("amdgcn").join("bitcode"));

        direct.chain(roots).find(|dir| dir.join("ocml.bc").is_file())
    })
    .as_ref()
    .ok_or_else(|| {
        format!(
            "no ROCm device libraries found: looked for amdgcn/bitcode/ocml.bc via {} and under {}. \
             Set CUBECL_ROCM_DEVICE_LIB_PATH to the directory holding ocml.bc",
            DEVICE_LIB_PATH_VARS.join(", "),
            DEFAULT_ROCM_ROOTS.join(", "),
        )
    })
}

/// The bitcode of `name`, read once and kept: every kernel compiled for a given device links
/// the same few hundred kilobytes.
fn device_lib(name: &str) -> Result<&'static [u8], String> {
    static CACHE: OnceLock<Mutex<HashMap<String, &'static [u8]>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Mutex::default);

    let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(bitcode) = cache.get(name) {
        return Ok(bitcode);
    }

    let path = bitcode_dir()?.join(name);
    let bitcode = std::fs::read(&path).map_err(|err| format!("{}: {err}", path.display()))?;
    let bitcode: &'static [u8] = Vec::leak(bitcode);
    cache.insert(name.to_string(), bitcode);
    Ok(bitcode)
}

/// The device libraries a kernel for `arch` links against.
fn device_libs_for(arch: &str) -> [String; 4] {
    let isa = arch.strip_prefix("gfx").unwrap_or(arch);
    [
        "ocml.bc".to_string(),
        "oclc_finite_only_off.bc".to_string(),
        "oclc_unsafe_math_off.bc".to_string(),
        format!("oclc_isa_version_{isa}.bc"),
    ]
}

/// Links what `module` needs out of the `ROCm` device libraries for `arch`.
///
/// # Safety
/// `module` must be a live LLVM module, already stamped with the AMDGPU triple and layout.
pub unsafe fn link_device_libs(module: LLVMModuleRef, arch: &str) -> Result<(), String> {
    for name in device_libs_for(arch) {
        let bitcode = device_lib(&name)?;

        // SAFETY: `bitcode` lives for the process, and the shim only reads it.
        let err = unsafe {
            cubecl_link_device_bitcode(module, bitcode.as_ptr() as *const c_char, bitcode.len())
        };
        if !err.is_null() {
            // SAFETY: the shim returns a NUL-terminated `malloc`'d string we now own.
            let message = unsafe { CStr::from_ptr(err).to_string_lossy().into_owned() };
            unsafe { cubecl_free_message(err) };
            return Err(format!("{name}: {message}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ISA control library is named by the bare architecture number.
    #[test]
    fn the_isa_library_follows_the_architecture() {
        assert_eq!(device_libs_for("gfx1201")[3], "oclc_isa_version_1201.bc");
        assert_eq!(device_libs_for("gfx90a")[3], "oclc_isa_version_90a.bc");
        assert_eq!(
            device_libs_for("gfx12-generic")[3],
            "oclc_isa_version_12-generic.bc"
        );
    }
}
