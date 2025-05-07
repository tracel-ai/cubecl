#[macro_use]
extern crate derive_new;

extern crate alloc;

mod backend;
mod compiler;
mod compute;
mod device;
mod element;
mod graphics;
mod runtime;

pub use compiler::base::*;
pub use compiler::wgsl::WgslCompiler;
pub use compute::*;
pub use device::*;
pub use element::*;
pub use graphics::*;
pub use runtime::*;

#[cfg(feature = "spirv")]
pub use backend::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
pub use backend::metal;

#[cfg(all(test, not(feature = "spirv"), not(feature = "msl")))]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::WgpuRuntime;

    cubecl_core::testgen_all!();
    cubecl_std::testgen!();
    cubecl_linalg::testgen_matmul_tiling2d!([flex32, f32]);
    cubecl_linalg::testgen_matmul_simple!([flex32, f32]);
    cubecl_linalg::testgen_tensor_identity!([flex32, f32, u32]);
    cubecl_reduce::testgen_reduce!();
    cubecl_random::testgen_random!();
    cubecl_reduce::testgen_shared_sum!([f32]);
}

#[cfg(all(test, feature = "spirv"))]
#[allow(unexpected_cfgs)]
mod tests_spirv {
    pub type TestRuntime = crate::WgpuRuntime;
    use cubecl_core::flex32;
    use half::f16;

    cubecl_core::testgen_all!(f32: [f16, flex32, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_std::testgen!();
    cubecl_linalg::testgen_matmul_tiling2d!([f16, f32, f64]);
    cubecl_linalg::testgen_matmul_simple!([f32]);
    cubecl_linalg::testgen_matmul_accelerated!([f16]);
    cubecl_reduce::testgen_reduce!();
    cubecl_random::testgen_random!();
    cubecl_reduce::testgen_shared_sum!([f32]);
}

#[cfg(all(test, feature = "msl"))]
#[allow(unexpected_cfgs)]
mod tests_msl {
    pub type TestRuntime = crate::WgpuRuntime;
    use half::f16;

    cubecl_core::testgen_all!(f32: [f16, f32], i32: [i16, i32], u32: [u16, u32]);
    cubecl_std::testgen!();
    cubecl_linalg::testgen_matmul_tiling2d!([f16, f32]);
    cubecl_linalg::testgen_conv2d_accelerated!([f16: f16]);
    cubecl_linalg::testgen_matmul_simple!([f16, f32]);
    cubecl_linalg::testgen_matmul_accelerated!([f16]);
    cubecl_reduce::testgen_reduce!();
    cubecl_random::testgen_random!();
    cubecl_reduce::testgen_shared_sum!([f32]);
}
