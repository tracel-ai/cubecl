#[macro_use]
extern crate derive_new;

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::CpuRuntime;

    pub use half::f16;

    cubecl_core::testgen_all!(f32: [f16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, f32, u32]);
    cubecl_std::testgen_quantized_view!(f32);
}

pub mod compiler;
pub mod compute;
pub mod device;
pub mod frontend;
pub mod runtime;

pub use device::CpuDevice;
pub use runtime::*;
