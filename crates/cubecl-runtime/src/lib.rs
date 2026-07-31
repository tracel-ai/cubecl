#![no_std]
#![warn(missing_docs)]

//! `CubeCL` runtime crate that helps creating high performance async runtimes.

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

#[macro_use]
extern crate derive_new;

/// Various identifier types used in `CubeCL`.
pub mod id;

/// Kernel related traits.
pub mod kernel;

/// Stream related utilities.
pub mod stream;

/// Throughput related utilities.
pub mod throughput;

/// Compute client module.
pub mod client;

/// Autotune module
pub mod tune;

/// Memory management module.
pub mod memory_management;
/// Cache of per-launch kernel metadata info buffers.
pub mod metadata_cache;
/// Compute server module.
pub mod server;
/// Compute Storage module.
pub mod storage;

/// `CubeCL` config module.
pub mod config;

pub use cubecl_common::benchmark;

/// Logging utilities to be used by a compute server.
pub mod logging;

/// TMA-related runtime types
pub mod tma;

/// Compiler trait and related types
pub mod compiler;
/// Whether a launch reaches the device, or is only compiled.
pub mod dispatch;
/// Runtime trait and related types
pub mod runtime;
/// Simple system profiling using timestamps.
pub mod timestamp_profiler;

/// Validation utils for shared properties
pub mod validation;

/// Allocators moddule.
pub mod allocator;
