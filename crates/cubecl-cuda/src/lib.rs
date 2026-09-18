#[macro_use]
extern crate derive_new;
extern crate alloc;

pub mod compiler;
mod compute;
mod device;
mod runtime;

pub use compiler::{CudaCompilationOptions, CudaCompiler, CudaRepresentation};
pub use device::*;
pub use runtime::*;

pub mod install {
    use std::path::PathBuf;

    pub fn include_path() -> PathBuf {
        let mut path = cuda_path().expect("
        CUDA installation not found.
        Please ensure that CUDA is installed and the CUDA_PATH environment variable is set correctly.
        Note: Default paths are used for Linux (/usr/local/cuda) and Windows (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/), which may not be correct.
    ");
        path.push("include");
        path
    }

    pub fn cccl_include_path() -> PathBuf {
        let mut path = include_path();
        path.push("cccl");
        path
    }

    pub fn cuda_path() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("CUDA_PATH") {
            return Some(PathBuf::from(path));
        }

        #[cfg(target_os = "linux")]
        {
            // If it is installed as part of the distribution
            return if std::fs::exists("/usr/local/cuda").is_ok_and(|exists| exists) {
                Some(PathBuf::from("/usr/local/cuda"))
            } else if std::fs::exists("/opt/cuda").is_ok_and(|exists| exists) {
                Some(PathBuf::from("/opt/cuda"))
            } else if std::fs::exists("/usr/bin/nvcc").is_ok_and(|exists| exists) {
                // Maybe the compiler was installed within the user path.
                Some(PathBuf::from("/usr"))
            } else {
                None
            };
        }

        #[cfg(target_os = "windows")]
        {
            return Some(PathBuf::from(
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/",
            ));
        }

        #[allow(unreachable_code)]
        None
    }
}

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::CudaRuntime;

    #[cfg(feature = "cpp")]
    pub use half::bf16;
    pub use half::f16;

    // `bf16` and the complex types are the C++ backend's: the LLVM backend has no type to
    // lower either through, so it does not advertise them (see `restrict_to_llvm_backend`)
    // and the tests for them are not generated. `testgen_complex_validation` runs either way
    // -- it asserts that an unadvertised complex operation is refused, which is what the LLVM
    // backend must do and what the C++ backend never reaches.
    #[cfg(feature = "cpp")]
    cubecl_core::testgen_all!(f32: [f16, bf16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    #[cfg(not(feature = "cpp"))]
    cubecl_core::testgen_all!(f32: [f16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);

    cubecl_core::testgen_launch_dynamic_count!();
    cubecl_std::testgen!();

    #[cfg(feature = "cpp")]
    cubecl_std::testgen_tensor_identity!([f16, bf16, f32, u32]);
    #[cfg(not(feature = "cpp"))]
    cubecl_std::testgen_tensor_identity!([f16, f32, u32]);

    cubecl_std::testgen_quantized_view!(f16);

    #[cfg(feature = "cpp")]
    cubecl_core::testgen_complex_core!();
    #[cfg(feature = "cpp")]
    cubecl_core::testgen_complex_compare!();
    #[cfg(feature = "cpp")]
    cubecl_core::testgen_complex_math!();
    cubecl_core::testgen_complex_validation!();
    cubecl_core::testgen_profiling!();
    cubecl_core::testgen_profiling_nested!();
}
