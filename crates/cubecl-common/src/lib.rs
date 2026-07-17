#![no_std]
#![warn(missing_docs)]

//! # `CubeCL` Common Library
//!
//! This library contains common types used by other crates that must be shared.
//!
//! Environment shims (sync primitives, futures, streams, config, persistence)
//! live in the `cubecl-environment` crate.

#[cfg(feature = "std")]
extern crate std;

#[macro_use]
extern crate derive_new;

/// A circular, allocation-free arena for reusable memory blocks.
#[cfg(feature = "std")]
pub mod arena;

/// Device module.
pub mod device;

/// Device handle module.
pub mod device_handle {
    pub use super::device::handle::{CallError, CallResultExt, DeviceHandle};
}

/// Utilities module to manipulate bytes.
#[cfg(feature = "serde")]
pub mod bytes;

/// Module for benchmark timings
pub mod benchmark;

/// Module for profiling any executable part
pub mod profile;

/// Quantization primitives required outside of `cubecl-quant`
pub mod quant;

/// Format utilities.
pub mod format;

/// Various utilities to create ID's.
extern crate alloc;

/// Hashing helper for stable, collision resistant hashes
#[cfg(feature = "hash")]
pub mod hash;

/// Custom float implementations
mod float;

pub use float::*;
