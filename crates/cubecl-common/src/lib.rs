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

/// Module for benchmarking any executable part
pub mod benchmark;

/// Useful when you need to read async data without having to decorate each function with async
/// notation.
pub mod reader;

/// Operators used by macro and IR
pub mod operator;
/// Synchronization type module, used both by ComputeServer and Backends.
pub mod sync_type;

extern crate alloc;
