#[macro_use]
extern crate alloc;

pub mod compute;
pub mod device;
pub mod memory;
pub mod runtime;

pub use device::{MetalDevice, register_device};
pub use runtime::MetalRuntime;

pub(crate) type MetalCompiler = cubecl_cpp::shared::CppCompiler<cubecl_cpp::metal::MslDialect>;

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::MetalRuntime;

    use half::f16;

    // bf16 disabled: simd_shuffle lacks bfloat support.
    cubecl_core::testgen_all!(f32: [f16, f32], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, f32, u32]);
    cubecl_std::testgen_quantized_view!(f32);
}
