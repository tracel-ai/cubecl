//! Cubecl standard library.

extern crate alloc;

mod reinterpret_slice;
pub use reinterpret_slice::*;
mod fast_math;
pub use fast_math::*;

mod quantization;
pub use quantization::*;

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
