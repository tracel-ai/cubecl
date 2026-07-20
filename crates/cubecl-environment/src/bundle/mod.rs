//! Named environment bundles.
//!
//! A bundle packages pre-warmed caches (autotune results, compiled kernels)
//! under a human-chosen name such as "H100 Linux" so applications can ship
//! them and skip cold-start tuning and compilation.
//!
//! # Format
//!
//! [`Bundle`] is the read side, and it is deliberately format-agnostic: a
//! bundle answers lookups by namespace and key bytes, exactly like the local
//! cache, so nothing about a layout on disk is load-bearing. [`SqliteBundle`]
//! is the native format, a single `SQLite` file holding the entries to ship
//! plus a manifest row. Other formats can be added for targets that have no
//! file system, without touching the read path.
//!
//! Writing is native-only on purpose. A bundle for any target is produced on a
//! development machine by [`export`], and only consumed elsewhere.
//!
//! # Correctness
//!
//! Bundles are never trusted for correctness: autotune entries seeded from a
//! bundle go through the same checksum validation as local ones, and every
//! store name carries the cubecl version and the device fingerprint, so a
//! mismatched machine never looks them up. A wrong bundle costs load time,
//! nothing else.
//!
//! Today only the `SQLite` format exists, so wasm and no-std targets keep
//! cold starts.

mod base;
#[cfg(any(feature = "cache", browser_cache))]
mod registry;

pub use base::*;
#[cfg(any(feature = "cache", browser_cache))]
pub use registry::*;

#[cfg(feature = "cache")]
mod export;
#[cfg(feature = "cache")]
mod manifest;
#[cfg(feature = "cache")]
mod sqlite;

#[cfg(feature = "cache")]
pub use export::*;
#[cfg(feature = "cache")]
pub use manifest::*;
#[cfg(feature = "cache")]
pub use sqlite::*;
