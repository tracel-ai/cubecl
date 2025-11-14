//! Cubecl standard library.
extern crate alloc;

mod fast_math;
mod reinterpret_slice;
mod swizzle;

pub use fast_math::*;
pub use reinterpret_slice::*;
pub use swizzle::*;

mod trigonometry;
pub use trigonometry::*;

mod option;
pub use option::*;

/// Quantization functionality required in views
pub mod quant;
pub mod tensor;

#[cfg(feature = "export_tests")]
pub mod tests;
