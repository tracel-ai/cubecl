//! Linking `ROCm`'s device libraries into a kernel.
//!
//! Only the definitions the kernel calls are taken. They arrive `linkonce_odr hidden`, so the
//! optimization pipeline inlines them and strips the rest back out.

use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use llvm_sys::prelude::LLVMModuleRef;

use cubecl_core::ir::amd::GfxArch;

unsafe extern "C" {
    /// See `cpp_shims/device_libs.cpp`. Returns null on success, else an owned message.
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
        let direct = DEVICE_LIB_PATH_VARS
            .iter()
            .filter_map(std::env::var_os)
            .map(PathBuf::from);
        let roots = (ROCM_ROOT_VARS.iter())
            .filter_map(std::env::var_os)
            .map(PathBuf::from)
            .chain(DEFAULT_ROCM_ROOTS.iter().map(PathBuf::from))
            .map(|root| root.join("amdgcn").join("bitcode"));

        direct
            .chain(roots)
            .find(|dir| dir.join("ocml.bc").is_file())
    })
    .as_ref()
    .ok_or_else(|| {
        format!(
            "no ROCm device libraries found: looked for ocml.bc via {}, and for \
             amdgcn/bitcode/ocml.bc under {} and {}. \
             Set CUBECL_ROCM_DEVICE_LIB_PATH to the directory holding ocml.bc",
            DEVICE_LIB_PATH_VARS.join(", "),
            ROCM_ROOT_VARS.join(", "),
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

/// What a module needs out of the device libraries. A kernel that needs neither links nothing,
/// and so still compiles on a machine with no `ROCm` installed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceLibs {
    /// `OCML`, for the float intrinsics the hardware has no correct answer for. See
    /// [`ocml`](super::ocml).
    pub math: bool,
    /// `OCKL`, for the `__printf_*` buffer a lowered `printf` writes into. See
    /// [`printf`](super::printf).
    pub printf: bool,
}

impl DeviceLibs {
    pub fn any(&self) -> bool {
        self.math || self.printf
    }
}

/// The device libraries a kernel for `arch` links against, in link order.
///
/// Each library leaves control globals undefined, one bitcode file per global. The math options
/// take the conservative side: operands are not assumed finite and reassociation is not assumed
/// safe. The other three follow the device and the code object.
fn device_libs_for(arch: &GfxArch, needs: DeviceLibs, code_object_version: u32) -> Vec<String> {
    let mut libs = Vec::new();

    if needs.math {
        libs.push("ocml.bc".to_string());
        libs.push("oclc_finite_only_off.bc".to_string());
        libs.push("oclc_unsafe_math_off.bc".to_string());
    }
    if needs.printf {
        libs.push("ockl.bc".to_string());
        libs.push(format!("oclc_abi_version_{code_object_version}.bc"));
        let wave = if arch.plane_dim() == Some(64) {
            "on"
        } else {
            "off"
        };
        libs.push(format!("oclc_wavefrontsize64_{wave}.bc"));
    }
    if needs.any() {
        // Wanted by both, so appended once.
        libs.push(format!("oclc_isa_version_{}.bc", arch.isa_version()));
    }

    libs
}

/// Links what `module` needs out of the `ROCm` device libraries for `arch`.
///
/// # Safety
/// `module` must be a live LLVM module, already stamped with the AMDGPU triple and layout.
pub unsafe fn link_device_libs(
    module: LLVMModuleRef,
    arch: &GfxArch,
    needs: DeviceLibs,
    code_object_version: u32,
) -> Result<(), String> {
    for name in device_libs_for(arch, needs, code_object_version) {
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

    const MATH: DeviceLibs = DeviceLibs {
        math: true,
        printf: false,
    };
    const PRINTF: DeviceLibs = DeviceLibs {
        math: false,
        printf: true,
    };

    /// The ISA control library is named by the bare architecture number, and comes along
    /// whichever library asked.
    #[test]
    fn the_isa_library_follows_the_architecture() {
        for needs in [MATH, PRINTF] {
            assert_eq!(
                device_libs_for(&GfxArch::parse("gfx1201"), needs, 500)
                    .last()
                    .unwrap(),
                "oclc_isa_version_1201.bc"
            );
            assert_eq!(
                device_libs_for(&GfxArch::parse("gfx90a"), needs, 500)
                    .last()
                    .unwrap(),
                "oclc_isa_version_90a.bc"
            );
            assert_eq!(
                device_libs_for(&GfxArch::parse("gfx12-generic"), needs, 500)
                    .last()
                    .unwrap(),
                "oclc_isa_version_12-generic.bc"
            );
        }
    }

    /// A kernel that needs nothing links nothing, so `ROCm` is only required by the kernels
    /// that actually reach into it.
    #[test]
    fn needing_nothing_links_nothing() {
        assert!(device_libs_for(&GfxArch::parse("gfx1201"), DeviceLibs::default(), 500).is_empty());
    }

    /// Printing pulls in OCKL, and with it the two control globals OCML never wanted: the
    /// code object's ABI version and the wavefront width of the device.
    #[test]
    fn printing_pulls_in_ockl_and_its_controls() {
        let libs = device_libs_for(&GfxArch::parse("gfx1201"), PRINTF, 500);
        assert!(libs.contains(&"ockl.bc".to_string()), "{libs:?}");
        assert!(
            libs.contains(&"oclc_abi_version_500.bc".to_string()),
            "{libs:?}"
        );
        // gfx1201 is RDNA, so wave32.
        assert!(
            libs.contains(&"oclc_wavefrontsize64_off.bc".to_string()),
            "{libs:?}"
        );
        // gfx90a is CDNA, so wave64.
        let cdna = device_libs_for(&GfxArch::parse("gfx90a"), PRINTF, 500);
        assert!(
            cdna.contains(&"oclc_wavefrontsize64_on.bc".to_string()),
            "{cdna:?}"
        );
    }

    /// The ISA library is not linked twice when both halves want it.
    #[test]
    fn the_shared_control_library_is_listed_once() {
        let libs = device_libs_for(
            &GfxArch::parse("gfx1201"),
            DeviceLibs {
                math: true,
                printf: true,
            },
            500,
        );
        let isa = libs.iter().filter(|l| l.starts_with("oclc_isa_")).count();
        assert_eq!(isa, 1, "{libs:?}");
    }
}
