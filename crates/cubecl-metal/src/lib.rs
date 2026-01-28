//! Native Metal backend for CubeCL
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
pub use device::MetalDevice;
pub use runtime::MetalRuntime;

/// Re-export objc2-metal for advanced users
pub use objc2_metal as metal;

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::MetalRuntime;

    #[allow(unused_imports)]
    use half::{bf16, f16};

    // MSL supports: f16, bf16, f32 (no f64), i8, i16, i32, i64, u8, u16, u32, u64
    //
    // NOTE: bf16 plane tests are expected to fail because Metal's simd_shuffle
    // doesn't support bfloat. bf16 works for compute, just not plane operations.
    cubecl_core::testgen_all!(f32: [f16, bf16, f32], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, bf16, f32, u32]);
    cubecl_std::testgen_quantized_view!(f32);
}
