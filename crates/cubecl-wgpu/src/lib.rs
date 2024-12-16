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
mod tests {
    pub type TestRuntime = crate::WgpuRuntime<crate::WgslCompiler>;

    cubecl_core::testgen_all!();
    cubecl_linalg::testgen_plane_mma_matmul!([flex32, f32], f32);
    cubecl_linalg::testgen_tiling2d_matmul!([flex32, f32]);
    cubecl_linalg::testgen_simple_matmul!([flex32, f32]);
    cubecl_reduce::testgen_reduce!();
}

#[cfg(all(test, feature = "spirv"))]
mod tests_spirv {
    pub type TestRuntime = crate::WgpuRuntime<crate::spirv::VkSpirvCompiler>;
    use cubecl_core::flex32;
    use half::f16;

    cubecl_core::testgen_all!(f32: [f16, flex32, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_linalg::testgen_plane_mma_matmul!([f16, flex32, f32], f32);
    cubecl_linalg::testgen_tiling2d_matmul!([f16, flex32, f32, f64]);
    cubecl_linalg::testgen_simple_matmul!([flex32, f32]);
    cubecl_linalg::testgen_cmma_matmul!([f16]);
}
