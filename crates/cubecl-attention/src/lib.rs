#![allow(clippy::explicit_counter_loop)]
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
