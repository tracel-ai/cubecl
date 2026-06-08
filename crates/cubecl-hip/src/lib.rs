#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;
extern crate alloc;

pub mod compute;
pub mod device;
pub mod runtime;
pub use device::*;
pub use runtime::HipRuntime;

#[cfg(not(feature = "rocwmma"))]
pub(crate) type HipWmmaCompiler = cubecl_cpp::hip::mma::WmmaIntrinsicCompiler;

#[cfg(feature = "rocwmma")]
pub(crate) type HipWmmaCompiler = cubecl_cpp::hip::mma::RocWmmaCompiler;

#[cfg(test)]
mod tests {
    use half::f16;
    pub type TestRuntime = crate::HipRuntime;

    cubecl_std::testgen!();
    cubecl_core::testgen_all!(f32: [f16, f32], i32: [i16, i32], u32: [u16, u32]);
    cubecl_core::testgen_launch_dynamic_count!();

    #[test]
    fn test_device_utilization_hip() {
        use cubecl_core::Runtime;

        let client = TestRuntime::client(&Default::default());

        // ROCm SMI may be unavailable (e.g. CI without the library), in which case we only require
        // that the query degrades gracefully to `None` rather than panicking.
        if let Some(utilization) = cubecl_common::future::block_on(client.device_utilization()) {
            assert!(
                (0.0..=100.0).contains(&utilization.compute_percentage),
                "compute_percentage out of range: {}",
                utilization.compute_percentage
            );
        }
    }
}
