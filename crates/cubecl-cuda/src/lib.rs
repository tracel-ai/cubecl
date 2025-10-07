#[macro_use]
extern crate derive_new;
extern crate alloc;

mod compute;
mod device;
mod runtime;

pub use device::*;
pub use runtime::*;

#[cfg(feature = "ptx-wmma")]
pub(crate) type WmmaCompiler = cubecl_cpp::cuda::mma::PtxWmmaCompiler;

#[cfg(not(feature = "ptx-wmma"))]
pub(crate) type WmmaCompiler = cubecl_cpp::cuda::mma::CudaWmmaCompiler;

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

    pub use half::{bf16, f16};

    cubecl_core::testgen_all!(f32: [f16, bf16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);

    cubecl_std::testgen!();

    cubecl_matmul::testgen_matmul_plane_accelerated!();
    cubecl_matmul::testgen_matmul_plane_mma!();
    cubecl_matmul::testgen_matmul_plane_vecmat!();
    cubecl_matmul::testgen_matmul_unit!();
    cubecl_matmul::testgen_matmul_tma!();
    cubecl_quant::testgen_quant!();

    // TODO: re-instate matmul quantized tests
    cubecl_matmul::testgen_matmul_simple!([f16, bf16, f32]);
    cubecl_std::testgen_tensor_identity!([f16, bf16, f32, u32]);
    cubecl_convolution::testgen_conv2d_accelerated!([f16: f16, bf16: bf16, f32: tf32]);
    cubecl_reduce::testgen_reduce!([f16, bf16, f32, f64]);
    cubecl_random::testgen_random!();
    cubecl_attention::testgen_attention!();
    cubecl_reduce::testgen_shared_sum!([f16, bf16, f32, f64]);
}
