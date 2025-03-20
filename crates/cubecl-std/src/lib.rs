//! Cubecl standard library.

mod quantization;
pub use quantization::*;

mod option;
pub use option::*;

pub mod tensor;

use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube]
#[allow(clippy::manual_div_ceil)]
pub fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
