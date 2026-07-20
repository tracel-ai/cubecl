#![no_std]
#![warn(missing_docs)]

//! # `CubeCL` Environment Library
//!
//! A cohesive API over the environment the code runs on: sync/std/thread,
//! async/tokio, async/wasm and no-std.
//!
//! The crate is organized in pillars:
//! - Compatibility shims mirroring `std` module names: [`sync`], [`collections`],
//!   [`time`], [`thread`], [`future`], plus [`rand`] and [`backtrace`].
//! - [`stream`]: stream identity and policies, including tokio task-stable streams.
//! - [`config`]: global configuration loading (`cubecl.toml` and friends).
//! - [`persistence`]: key-value caches kept up to date in memory and synced to the
//!   file system (std), browser storage (wasm) or nothing (no-std).
//! - [`bundle`]: named environment bundles that ship pre-warmed caches.

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

/// Synchronization primitives working across std, no-std and wasm environments.
pub mod sync;

/// Byte buffers with allocation properties, zero-copy views and optional
/// memory-mapped backing. The type to use for byte payloads, never `Vec<u8>`.
pub mod bytes;

/// Hash map and set types with the environment's default hasher.
pub mod collections;

/// Time types working across std, wasm and embedded environments.
pub mod time;

/// Thread utilities with a compatible API for native and wasm environments.
pub mod thread;

/// Future utils with a compatible API for native, non-std and wasm environments.
pub mod future;

/// Rand module contains types for random number generation for non-std environments and for
/// std environments.
pub mod rand;

/// Backtrace module to build error reports.
pub mod backtrace;

/// Stream identity, policies and manual stream management.
pub mod stream;

/// Runtime configuration trait shared across crates.
pub mod config;

/// Key-value persistence: in-memory stores synced to the file system, browser
/// storage or kept memory-only depending on the environment.
pub mod persistence;

/// Named environment bundles: ship pre-warmed autotune and compilation caches.
pub mod bundle;
