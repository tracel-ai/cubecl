//! Cubecl standard library.
use core::f32;

extern crate alloc;

mod reinterpret_slice;
pub use reinterpret_slice::*;
mod fast_math;
pub use fast_math::*;

mod option;
pub use option::*;

pub mod tensor;

use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cfg(feature = "export_tests")]
pub mod tests;

#[cube]
#[allow(clippy::manual_div_ceil)]
pub fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

#[cube]
pub fn to_degrees<F: Float>(val: F) -> F {
    val * F::new(180.0 / f32::consts::PI)
}

#[cube]
pub fn to_radians<F: Float>(val: F) -> F {
    val * F::new(f32::consts::PI / 180.0)
}
