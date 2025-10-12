//! Cubecl standard library.
extern crate alloc;

mod reinterpret_slice;
pub use reinterpret_slice::*;
mod fast_math;
pub use fast_math::*;

mod option;
pub use option::*;

/// Quantization functionality required in views
pub mod quant;
pub mod tensor;

#[cfg(feature = "export_tests")]
pub mod tests;
