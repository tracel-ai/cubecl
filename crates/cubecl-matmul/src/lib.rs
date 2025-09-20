#![allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]

mod base;
/// Components for matrix multiplication
pub mod components;
/// Contains matmul kernels
pub mod kernels;
/// Tests for matmul kernels
#[cfg(feature = "export_tests")]
pub mod tests;

pub use base::*;

/// Autotune key for matmul.
pub mod tune_key;
