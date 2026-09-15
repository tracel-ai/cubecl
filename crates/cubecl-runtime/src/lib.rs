#![no_std]
#![warn(missing_docs)]

//! The runtime API of `CubeCL`: what a kernel author compiles against.
//!
//! A [`Client`](client::Client) reaches a device, a [`Handle`](server::Handle)
//! names memory on it, and a [`CubeKernel`](kernel::CubeKernel) is what a
//! launch hands over. The [`Server`](server::Server) trait is the contract a
//! runtime implements, and everything an implementation needs beyond the
//! contract — memory pools, streams, drivers, the compilation pipeline — lives
//! in `cubecl-server`, so that user code never depends on it.
//!
//! # Error handling
//!
//! A failure belongs to the memory that work left unwritten and travels the
//! dataflow from there: nothing here holds error state beside the buffers.
//! `Taint`, `ErrorGraph`, `Failures` and `ExecuteScope` in `cubecl-server` are
//! the four pieces, and `ERROR_HANDLING.md` at the repository root is the
//! argument for why they are shaped that way — read it before changing any of
//! them.

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

#[macro_use]
extern crate derive_new;

/// Various identifier types used in `CubeCL`.
pub mod id;

/// Kernel related traits.
pub mod kernel;

/// Throughput related utilities.
pub mod throughput;

/// Compute client module.
pub mod client;

/// The device type of each runtime.
pub mod device;

/// Autotune module
pub mod tune;

/// Memory management value types: reports, configuration and the managed
/// handle. The pools behind them are `cubecl-server`'s.
pub mod memory_management;
/// Compute server module.
pub mod server;
/// Compute Storage contract.
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
/// Running a workload for the compilation and tuning it provokes, without
/// running the workload itself.
pub mod dry_run;
/// Runtime trait and related types
pub mod runtime;
