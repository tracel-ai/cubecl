#![no_std]
#![warn(missing_docs)]

//! The toolkit for implementing a `CubeCL` runtime.
//!
//! A runtime depends on this crate, not on `cubecl-runtime` directly: every
//! module of the runtime API is re-exported here under the same name, so an
//! implementation sees one tree — the contract it implements and the pieces
//! it implements it with.
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

// The runtime API, re-exported so `crate::server::Handle` and
// `cubecl_server::server::Handle` both name the one type.
pub use cubecl_runtime::benchmark;
pub use cubecl_runtime::{
    client, config, device, dry_run, id, logging, runtime, throughput, tma, tune,
};
pub use cubecl_runtime::{local_tuner, storage_id_type};

/// Kernel related traits, and the compiled kernel a launch produces.
pub mod kernel;

/// Stream related utilities.
pub mod stream;

/// Memory management module.
pub mod memory_management;
/// Cache of per-launch kernel metadata info buffers.
pub mod metadata_cache;
/// Compute server module.
pub mod server;
/// Compute Storage module.
pub mod storage;

#[cfg(multi_threading)]
pub mod command;

pub mod driver;

pub mod device_events;

/// Compiler trait, and the compilation caches in front of one.
pub mod compiler;
/// Simple system profiling using timestamps.
pub mod timestamp_profiler;

/// Validation utils for shared properties
pub mod validation;

/// Allocators moddule.
pub mod allocator;
