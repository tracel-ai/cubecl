#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! # CubeCL Common Library
//!
//! This library contains common types used by other crates that must be shared.

#[macro_use]
extern crate derive_new;

/// Rand module contains types for random number generation for non-std environments and for
/// std environments.
pub mod rand;

/// Device module.
pub mod device;

/// Map utilities and implementations.
pub mod map;

/// Utilities module to manipulate bytes.
#[cfg(feature = "serde")]
pub mod bytes;

/// Stub module contains types for stubs for non-std environments and for std environments.
pub mod stub;

/// Stream id related utilities.
pub mod stream_id;

/// Cache module for an efficient in-memory and persistent database.
#[cfg(feature = "cache")]
pub mod cache;

#[cfg(feature = "cache")]
pub(crate) mod cache_file;

/// Module for benchmark timings
pub mod benchmark;

/// Module for profiling any executable part
pub mod profile;

/// Useful when you need to read async data without having to decorate each function with async
/// notation.
pub mod reader;

/// Future utils with a compatible API for native, non-std and wasm environments.
pub mod future;

/// Quantization primitives required outside of `cubecl-quant`
pub mod quant;

/// Various utilities to create ID's.
extern crate alloc;

/// Custom float implementations
mod float;
/// Common kernel types
mod kernel;

pub use float::*;
pub use kernel::*;
