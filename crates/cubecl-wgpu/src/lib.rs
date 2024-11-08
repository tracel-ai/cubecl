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

/// Platform dependent reference counting. Uses [`alloc::sync::Arc`] on all platforms except
/// `wasm32` when the feature `atomics` is enabled. Uses [`alloc::rc::Rc`] instead when on
/// `wasm32` and with the `atomics` feature enabled.
#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
type Pdrc<T> = alloc::sync::Arc<T>;

/// Platform dependent reference counting. Uses [`alloc::sync::Arc`] on all platforms except
/// `wasm32` when the feature `atomics` is enabled. Uses [`alloc::rc::Rc`] instead when on
/// `wasm32` and with the `atomics` feature enabled.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
type Pdrc<T> = alloc::rc::Rc<T>;

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::WgpuRuntime<crate::WgslCompiler>;

    cubecl_core::testgen_all!();
    cubecl_linalg::testgen_plane_mma!([flex32, f32], f32);
    cubecl_linalg::testgen_tiling2d!([flex32, f32]);
}

#[cfg(all(test, feature = "spirv"))]
mod tests_spirv {
    pub type TestRuntime = crate::WgpuRuntime<crate::spirv::VkSpirvCompiler>;
    use cubecl_core::flex32;
    use half::f16;

    cubecl_core::testgen_all!(f32: [f16, flex32, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_linalg::testgen_plane_mma!([f16, flex32, f32, f64], f32);
    cubecl_linalg::testgen_tiling2d!([f16, flex32, f32, f64]);
}
