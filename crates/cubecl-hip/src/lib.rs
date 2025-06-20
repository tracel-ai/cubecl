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

#[cfg(not(feature = "rocwmma"))]
pub(crate) type HipWmmaCompiler = cubecl_cpp::hip::mma::WmmaIntrinsicCompiler;

#[cfg(feature = "rocwmma")]
pub(crate) type HipWmmaCompiler = cubecl_cpp::hip::mma::RocWmmaCompiler;

#[cfg(target_os = "linux")]
#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    pub type TestRuntime = crate::HipRuntime;

    cubecl_std::testgen!();
    cubecl_core::testgen_all!(f32: [f16, f32], i32: [i16, i32], u32: [u16, u32]);

    #[cfg(feature = "matmul_tests")]
    cubecl_matmul::testgen_matmul_plane_accelerated!();
    #[cfg(feature = "matmul_tests")]
    cubecl_matmul::testgen_matmul_simple!([f16, f32]);

    cubecl_reduce::testgen_reduce!([f16, bf16, f32, f64]);
    cubecl_reduce::testgen_shared_sum!([f32]);
}
