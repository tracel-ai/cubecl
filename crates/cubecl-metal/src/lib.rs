//! Native Metal backend for `CubeCL`
//!
//! This crate provides a Metal backend that directly interfaces with Apple's Metal API,
//! enabling BF16 support, vec8 vectorization, and direct access to simdgroup operations.

#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate alloc;

pub mod compiler;
pub mod compute;
pub mod device;
pub mod memory;
pub mod runtime;

pub use compiler::MetalCompiler;
pub use device::{register_device, MetalDevice};
pub use runtime::MetalRuntime;

/// Re-export objc2-metal for advanced users
pub use objc2_metal as metal;

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::MetalRuntime;

    use half::f16;

    // NOTE: bf16 is disabled because Metal's simd_shuffle doesn't support bfloat,
    // causing plane operation tests to fail.
    cubecl_core::testgen_all!(f32: [f16, f32], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, f32, u32]);
    cubecl_std::testgen_quantized_view!(f32);
}
