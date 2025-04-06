#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;
extern crate alloc;

#[cfg(target_os = "linux")]
pub mod compute;
#[cfg(target_os = "linux")]
pub mod device;
#[cfg(target_os = "linux")]
pub mod runtime;
#[cfg(target_os = "linux")]
pub use device::*;
#[cfg(target_os = "linux")]
pub use runtime::HipRuntime;
#[cfg(target_os = "linux")]
#[cfg(feature = "wmma-intrinsics")]
pub(crate) type HipWmmaCompiler = cubecl_cpp::hip::mma::WmmaIntrinsicCompiler;
#[cfg(target_os = "linux")]
#[cfg(not(feature = "wmma-intrinsics"))]
pub(crate) type HipWmmaCompiler = cubecl_cpp::hip::mma::RocWmmaCompiler;

#[cfg(target_os = "linux")]
#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    pub type TestRuntime = crate::HipRuntime;

    cubecl_core::testgen_all!();
    cubecl_std::testgen!();
    cubecl_linalg::testgen_matmul_plane!([f32]);
    cubecl_linalg::testgen_matmul_accelerated!([f32]);
    cubecl_linalg::testgen_tensor_identity!([f32, u32]);
    cubecl_reduce::testgen_reduce!([f16, bf16, f32, f64]);
}
