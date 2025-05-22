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

/// Stub module contains types for stubs for non-std environments and for std environments.
pub mod stub;

/// Stream id related utilities.
pub mod stream_id;

/// Cache module for an efficient in-memory and persistent database.
#[cfg(feature = "cache")]
pub mod cache;

#[cfg(feature = "cache")]
pub(crate) mod cache_file;

/// Module for benchmarking any executable part
pub mod benchmark;

/// Useful when you need to read async data without having to decorate each function with async
/// notation.
pub mod reader;

/// Future utils with a compatible API for native, non-std and wasm environments.
pub mod future;

/// Various utilities to create ID's.
extern crate alloc;

/// Custom float implementations
mod float;
/// Common kernel types
mod kernel;

pub use float::*;
pub use kernel::*;
