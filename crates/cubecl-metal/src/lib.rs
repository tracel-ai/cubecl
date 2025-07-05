mod compute;
mod device;
mod runtime;

pub use device::*;
pub use runtime::{MetalRuntime, RuntimeOptions};

pub(crate) const METAL_MEMORY_ALIGNMENT: usize = 256;
pub(crate) const METAL_DISPATCH_LIMIT: usize = 32;

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    use crate::runtime::MetalRuntime;
    use half::f16;

    pub type TestRuntime = MetalRuntime;

    cubecl_core::testgen_all!(f32: [f16, f32], i32: [i16, i32], u32: [u16, u32]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, flex32, f32, u32]);
    cubecl_convolution::testgen_conv2d_accelerated!([f16: f16]);
    cubecl_matmul::testgen_matmul_simple!([f16, f32]);
    cubecl_matmul::testgen_matmul_plane_accelerated!();
    cubecl_matmul::testgen_matmul_unit!();
    cubecl_reduce::testgen_reduce!();
    cubecl_random::testgen_random!();
    cubecl_reduce::testgen_shared_sum!([f32]);
}
