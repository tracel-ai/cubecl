#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! CubeCL runtime crate that helps creating high performance async runtimes.

extern crate alloc;

#[macro_use]
extern crate derive_new;

/// Various identifier types used in CubeCL.
pub mod id;

/// Kernel related traits.
pub mod kernel;

/// Compute channel module.
pub mod channel;
/// Compute client module.
pub mod client;

/// Autotune module
pub mod tune;

/// Memory management module.
pub mod memory_management;
/// Compute server module.
pub mod server;
/// Compute Storage module.
pub mod storage;

/// CubeCL config module.
pub mod config;

mod feature_set;

mod base;
pub use base::*;
pub use cubecl_common::benchmark;

pub use feature_set::*;
/// Logging utilities to be used by a compute server.
pub mod logging;

/// TMA-related runtime types
pub mod tma;

/// Simple system profiling using timestamps.
#[cfg(feature = "std")]
pub mod timestamp_profiler;
