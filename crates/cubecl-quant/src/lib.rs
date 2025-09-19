#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
#![allow(clippy::manual_div_ceil, clippy::manual_is_multiple_of)]

extern crate alloc;

#[cfg(feature = "kernels")]
pub mod dequantize;

#[cfg(feature = "kernels")]
pub mod quantize;

#[cfg(feature = "kernels")]
pub mod qparams;

pub mod scheme;

#[cfg(feature = "export_tests")]
pub mod tests;

#[cfg(feature = "kernels")]
pub(crate) mod utils {
    use crate::scheme::{QuantLevel, QuantScheme};

    pub(crate) fn check_block_size_compat(scheme: &QuantScheme, div: usize) {
        // Validate block size compatibility
        if let QuantScheme {
            level: QuantLevel::Block(block_size),
            ..
        } = scheme
        {
            assert!(
                *block_size % div == 0,
                "Block size must be divisible by {div}, got block_size={block_size}"
            );
        }
    }
}
