//! CUDA's `libdevice`, as [`MathLibrary`] describes it, and linking it in.
//!
//! `libdevice.10.bc` ships with the CUDA toolkit and is what NVCC and NVRTC themselves link a
//! kernel's math against. It matters more here than OCML does on the AMD side: where the
//! AMDGPU backend lowers the single-precision transcendentals itself and only wants the
//! library for double precision, NVPTX has no libcall at all behind any of them. An intrinsic
//! left un-redirected does not compile to something slow, it reaches `ISel` with nothing behind
//! it and takes the process down with `LLVM ERROR: Cannot select`.

use std::ffi::{CStr, c_char};
use std::path::{Path, PathBuf};

use llvm_sys::prelude::LLVMModuleRef;

use crate::shared::math_library::{FloatWidth, MathLibrary};

/// The intrinsics NVPTX has nothing behind.
///
/// Everything transcendental, in short. What is deliberately *not* here is what the backend
/// does select: `sqrt`, `fma`, `fabs`, `copysign`, `minnum`/`maxnum`, and the rounding family
/// (`floor`, `ceil`, `trunc`, `rint`, `nearbyint`), each of which is a PTX instruction.
/// Redirecting one of those would give up an instruction for a call.
const NO_LIBCALL: [&str; 19] = [
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "exp2",
    "exp10", "log", "log2", "log10", "pow", "cbrt", "erf",
];

pub struct Libdevice;

impl MathLibrary for Libdevice {
    fn needs_redirect(&self, base: &str, _width: FloatWidth) -> bool {
        NO_LIBCALL.contains(&base)
    }

    /// `libdevice` is named the way CUDA's own headers are: `__nv_sinf` at single precision and
    /// `__nv_sin` at double.
    ///
    /// It has no half entry points at all, so a half intrinsic is answered at single precision
    /// with conversions around it. That is what CUDA's own `__hsin` and friends do, and the
    /// caller builds the wrapper.
    fn symbol(&self, base: &str, width: FloatWidth) -> Option<(String, FloatWidth)> {
        match width {
            FloatWidth::F16 | FloatWidth::F32 => Some((format!("__nv_{base}f"), FloatWidth::F32)),
            FloatWidth::F64 => Some((format!("__nv_{base}"), FloatWidth::F64)),
        }
    }
}

/// Where the toolkit keeps `libdevice.10.bc`.
///
/// One file for every architecture since CUDA 9; the older per-`compute_XX` files are long
/// gone, so there is nothing to select between.
fn libdevice_path() -> Option<PathBuf> {
    let root = cuda_root()?;
    let path = root.join("nvvm").join("libdevice").join("libdevice.10.bc");
    path.is_file().then_some(path)
}

/// Roots to look under, after `CUDA_PATH`, in the order an install makes them true.
///
/// `/usr/lib/cuda` is where Debian and Ubuntu's `nvidia-cuda-toolkit` package puts the
/// toolkit; the runtime's own `install::cuda_path` answers `/usr` for that layout, which is
/// right for the headers it wants and wrong for this, since there is no `/usr/nvvm`.
const CUDA_ROOTS: [&str; 4] = ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda", "/usr"];

/// The CUDA toolkit, by a search of its own rather than the runtime's: this crate does not
/// depend on `cubecl-cuda` -- the dependency runs the other way -- and what is wanted here is
/// the directory holding `nvvm`, which is not always the one holding `include`.
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

/// Links `libdevice` into `module`, so the calls the redirect just planted resolve.
///
/// Only the functions the module actually reached for come along: the shim links with
/// `LinkOnlyNeeded`, which is what keeps a kernel using one `sinf` from carrying every
/// transcendental CUDA has.
///
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

    // SAFETY: `module` is live, and the shim only reads `bitcode` for the length given. It
    // returns an owned message on failure, freed below.
    let err = unsafe {
        cubecl_link_device_bitcode(module, bitcode.as_ptr() as *const c_char, bitcode.len())
    };
    if !err.is_null() {
        // SAFETY: the shim returns a NUL-terminated `malloc`'d string we now own.
        let message = unsafe { CStr::from_ptr(err).to_string_lossy().into_owned() };
        unsafe { cubecl_free_message(err) };
        return Err(format!("{}: {message}", path.display()));
    }
    Ok(())
}

unsafe extern "C" {
    /// See `amdgpu/cpp_shims/device_libs.cpp`. Named for where it was first needed, but the
    /// shim is a plain `LinkOnlyNeeded` over a bitcode buffer and has nothing AMD about it.
    /// Returns null on success, else an owned message.
    fn cubecl_link_device_bitcode(
        dest: LLVMModuleRef,
        data: *const c_char,
        len: usize,
    ) -> *mut c_char;

    /// Frees what `cubecl_link_device_bitcode` returned.
    fn cubecl_free_message(message: *mut c_char);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Half goes through single precision, because the library has no half entry points; the
    /// caller reads the returned width to know it must convert.
    #[test]
    fn half_is_answered_at_single_precision() {
        assert_eq!(
            Libdevice.symbol("sin", FloatWidth::F16),
            Some(("__nv_sinf".to_string(), FloatWidth::F32))
        );
        assert_eq!(
            Libdevice.symbol("sin", FloatWidth::F32),
            Some(("__nv_sinf".to_string(), FloatWidth::F32))
        );
        assert_eq!(
            Libdevice.symbol("sin", FloatWidth::F64),
            Some(("__nv_sin".to_string(), FloatWidth::F64))
        );
    }

    /// What the backend selects itself must not be redirected: each of these is a PTX
    /// instruction, and a call would be strictly worse.
    #[test]
    fn the_instructions_the_backend_has_are_left_alone() {
        for base in ["sqrt", "fma", "fabs", "floor", "ceil", "trunc", "rint"] {
            assert!(
                !Libdevice.needs_redirect(base, FloatWidth::F32),
                "{base} is a PTX instruction"
            );
        }
        // Unlike AMD, single precision needs the library just as much as double.
        for width in [FloatWidth::F32, FloatWidth::F64] {
            assert!(Libdevice.needs_redirect("sin", width));
            assert!(Libdevice.needs_redirect("atan2", width));
        }
    }
}
