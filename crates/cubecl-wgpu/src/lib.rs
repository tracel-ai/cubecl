#[macro_use]
extern crate derive_new;

extern crate alloc;

mod compiler;
mod compute;
mod device;
mod element;
mod graphics;
mod runtime;

pub use compiler::wgsl::WgslCompiler;
pub use compute::*;
pub use device::*;
pub use element::*;
pub use graphics::*;
pub use runtime::*;

#[cfg(feature = "spirv")]
pub use compiler::spirv;

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::WgpuRuntime<crate::WgslCompiler>;

    cubecl_core::testgen_all!();
    cubecl_linalg::testgen_matmul_plane!([f32]);
    // cubecl_linalg::testgen_matmul_accelerated!([f32]);
    cubecl_linalg::testgen_matmul_tiling2d!([flex32, f32]);
    cubecl_linalg::testgen_matmul_simple!([flex32, f32]);
    cubecl_linalg::testgen_tensor_identity!([flex32, f32, u32]);
    cubecl_reduce::testgen_reduce!();
}

#[cfg(all(test, feature = "spirv"))]
#[allow(unexpected_cfgs)]
mod tests_spirv {
    pub type TestRuntime = crate::WgpuRuntime<crate::spirv::VkSpirvCompiler>;
    use cubecl_core::flex32;
    use half::f16;

    cubecl_core::testgen_all!(f32: [f16, flex32, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_linalg::testgen_matmul_plane!([f16, f32]);
    cubecl_linalg::testgen_matmul_tiling2d!([f16, f32, f64]);
    cubecl_linalg::testgen_matmul_simple!([f32]);
    cubecl_linalg::testgen_matmul_accelerated!([f16]);
    cubecl_reduce::testgen_reduce!();
}
