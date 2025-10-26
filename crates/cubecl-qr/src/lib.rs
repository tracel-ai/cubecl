#![doc=include_str!("../README.md")]
mod base;
/// Contains QR kernels
pub(crate) mod kernels;
/// Tests for QR kernels
#[cfg(feature = "export_tests")]
pub mod tests;

pub use base::*;
