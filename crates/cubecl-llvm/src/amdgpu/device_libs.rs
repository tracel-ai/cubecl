//! `ROCm` device libraries.

use crate::shared::bitcode::link_bitcode;
use cubecl_core::ir::amd::GfxArch;
use llvm_sys::prelude::LLVMModuleRef;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

/// `CUBECL_ROCM_DEVICE_LIB_PATH` and `HIP_DEVICE_LIB_PATH` override the search paths.
const DEVICE_LIB_PATH_VARS: [&str; 2] = ["CUBECL_ROCM_DEVICE_LIB_PATH", "HIP_DEVICE_LIB_PATH"];
const ROCM_ROOT_VARS: [&str; 2] = ["ROCM_PATH", "HIP_PATH"];
const DEFAULT_ROCM_ROOTS: [&str; 2] = ["/opt/rocm", "/usr"];

/// `LLVM` installs under a `ROCm` root whose clang resource directory holds the bitcode:
/// upstream `ROCm` (`lib/llvm`, `llvm`) and distribution packages (Fedora's `lib64/rocm/llvm`).
const LLVM_SUBDIRS: [&str; 4] = ["lib/llvm", "llvm", "lib64/rocm/llvm", "lib/rocm/llvm"];

/// Bitcode directories under a `ROCm` root, the legacy `amdgcn/bitcode` first, then each `LLVM`
/// install's `lib/clang/<version>/lib/amdgcn/bitcode`, newest clang first.
fn bitcode_candidates(root: PathBuf) -> Vec<PathBuf> {
    let mut candidates = vec![root.join("amdgcn").join("bitcode")];

    for llvm in LLVM_SUBDIRS {
        let Ok(entries) = std::fs::read_dir(root.join(llvm).join("lib").join("clang")) else {
            continue;
        };
        let mut versions: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        versions.sort_by_key(|dir| clang_version(dir));
        candidates.extend(
            versions
                .into_iter()
                .rev()
                .map(|dir| dir.join("lib").join("amdgcn").join("bitcode")),
        );
    }

    candidates
}

/// The numeric components of a clang resource directory's name (`20`, `17.0.0`).
fn clang_version(dir: &std::path::Path) -> Vec<u64> {
    dir.file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            name.split('.')
                .map(|part| part.parse().unwrap_or(0))
                .collect()
        })
        .unwrap_or_default()
}

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
            .flat_map(bitcode_candidates);

        direct
            .chain(roots)
            .find(|dir| dir.join("ocml.bc").is_file())
    })
    .as_ref()
    .ok_or_else(|| {
        format!(
            "no ROCm device libraries found: looked for ocml.bc via {}, and for \
             amdgcn/bitcode/ocml.bc and {{{}}}/lib/clang/*/lib/amdgcn/bitcode/ocml.bc \
             under {} and {}. \
             Set CUBECL_ROCM_DEVICE_LIB_PATH to the directory holding ocml.bc",
            DEVICE_LIB_PATH_VARS.join(", "),
            LLVM_SUBDIRS.join(","),
            ROCM_ROOT_VARS.join(", "),
            DEFAULT_ROCM_ROOTS.join(", "),
        )
    })
}

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

/// Device libraries required by a kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceLibs {
    /// Math functions.
    pub math: bool,
    /// Device printing.
    pub printf: bool,
}

impl DeviceLibs {
    pub fn any(&self) -> bool {
        self.math || self.printf
    }
}

/// Device libraries in link order.
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
        libs.push(format!("oclc_isa_version_{}.bc", arch.isa_version()));
    }

    libs
}

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
        // SAFETY: the caller keeps `module` live.
        unsafe { link_bitcode(module, bitcode) }.map_err(|message| format!("{name}: {message}"))?;
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

    #[test]
    fn needing_nothing_links_nothing() {
        assert!(device_libs_for(&GfxArch::parse("gfx1201"), DeviceLibs::default(), 500).is_empty());
    }

    #[test]
    fn printing_pulls_in_ockl_and_its_controls() {
        let libs = device_libs_for(&GfxArch::parse("gfx1201"), PRINTF, 500);
        assert!(libs.contains(&"ockl.bc".to_string()), "{libs:?}");
        assert!(
            libs.contains(&"oclc_abi_version_500.bc".to_string()),
            "{libs:?}"
        );
        assert!(
            libs.contains(&"oclc_wavefrontsize64_off.bc".to_string()),
            "{libs:?}"
        );
        let cdna = device_libs_for(&GfxArch::parse("gfx90a"), PRINTF, 500);
        assert!(
            cdna.contains(&"oclc_wavefrontsize64_on.bc".to_string()),
            "{cdna:?}"
        );
    }

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
