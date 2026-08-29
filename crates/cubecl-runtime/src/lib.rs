#![no_std]
#![warn(missing_docs)]

//! `CubeCL` runtime crate that helps creating high performance async runtimes.
//!
//! # Error handling
//!
//! A failure belongs to the memory that work left unwritten and travels the
//! dataflow from there: nothing here holds error state beside the buffers.
//! [`memory_management::Taint`], [`memory_management::ErrorGraph`],
//! [`stream::Failures`] and [`stream::ExecuteScope`] are the four pieces, and
//! `ERROR_HANDLING.md` at the repository root is the argument for why they are
//! shaped that way — read it before changing any of them.

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

#[cfg(multi_threading)]
pub mod command;

pub mod driver;

/// Compiler trait and related types
pub mod compiler;
/// Running a workload for the compilation and tuning it provokes, without
/// running the workload itself.
pub mod dry_run;
/// Runtime trait and related types
pub mod runtime;
/// Simple system profiling using timestamps.
pub mod timestamp_profiler;

/// Validation utils for shared properties
pub mod validation;

/// Allocators moddule.
pub mod allocator;
