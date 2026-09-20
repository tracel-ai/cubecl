//! CUDA math library support.

use crate::shared::{
    bitcode::link_bitcode,
    math_library::{FloatWidth, MathLibrary},
};
use llvm_sys::prelude::LLVMModuleRef;
use std::path::{Path, PathBuf};

/// Math intrinsics provided by libdevice.
const NO_LIBCALL: [&str; 19] = [
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "exp2",
    "exp10", "log", "log2", "log10", "pow", "cbrt", "erf",
];

pub struct Libdevice;

impl MathLibrary for Libdevice {
    fn needs_redirect(&self, base: &str, _width: FloatWidth) -> bool {
        NO_LIBCALL.contains(&base)
    }

    /// Libdevice supports f32 and f64; f16 calls use f32.
    fn symbol(&self, base: &str, width: FloatWidth) -> Option<(String, FloatWidth)> {
        match width {
            FloatWidth::F16 | FloatWidth::F32 => Some((format!("__nv_{base}f"), FloatWidth::F32)),
            FloatWidth::F64 => Some((format!("__nv_{base}"), FloatWidth::F64)),
        }
    }
}

/// Bitcode path within the CUDA toolkit.
fn libdevice_path() -> Option<PathBuf> {
    let root = cuda_root()?;
    let path = root.join("nvvm").join("libdevice").join("libdevice.10.bc");
    path.is_file().then_some(path)
}

/// Fallback CUDA toolkit locations.
const CUDA_ROOTS: [&str; 4] = ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda", "/usr"];

fn cuda_root() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("CUDA_PATH") {
        return Some(PathBuf::from(path));
    }
    CUDA_ROOTS
        .into_iter()
        .map(Path::new)
        .find(|candidate| candidate.join("nvvm").join("libdevice").is_dir())
        .map(Path::to_path_buf)
}

/// # Safety
/// `module` must be a live LLVM module.
pub unsafe fn link_libdevice(module: LLVMModuleRef) -> Result<(), String> {
    let path = libdevice_path().ok_or_else(|| {
        "libdevice.10.bc was not found: the kernel calls a math function NVPTX has no \
         instruction for, and the CUDA toolkit is where the implementation lives. Set CUDA_PATH \
         to a toolkit containing nvvm/libdevice/libdevice.10.bc."
            .to_string()
    })?;
    let bitcode =
        std::fs::read(&path).map_err(|err| format!("reading {}: {err}", path.display()))?;
    // SAFETY: the caller keeps `module` live.
    unsafe { link_bitcode(module, &bitcode) }
        .map_err(|message| format!("{}: {message}", path.display()))
}
