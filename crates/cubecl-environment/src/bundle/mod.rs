//! Named environment bundles.
//!
//! A bundle packages pre-warmed caches (autotune results, compiled kernels)
//! under a human-chosen name such as "H100 Linux" so applications can ship
//! them and skip cold-start tuning and compilation.
//!
//! # Format
//!
//! A bundle is a single file: the same `SQLite` database as a local cache root,
//! holding the entries to ship plus a manifest row. Exporting is therefore a
//! query over a warm cache root (see [`export`]) rather than a file copy, and
//! seeding a store is a lookup by the same store name and key bytes the local
//! cache uses. Nothing about the layout on disk is load-bearing.
//!
//! # Correctness
//!
//! Bundles are never trusted for correctness: autotune entries seeded from a
//! bundle go through the same checksum validation as local ones, and every
//! store name carries the cubecl version and the device fingerprint, so a
//! mismatched machine never looks them up. A wrong bundle costs load time,
//! nothing else.
//!
//! Bundles are a native feature: wasm and no-std targets keep cold starts.

#[cfg(any(feature = "cache", browser_cache))]
mod registry;
#[cfg(any(feature = "cache", browser_cache))]
mod seed;

#[cfg(any(feature = "cache", browser_cache))]
pub use registry::*;
#[cfg(any(feature = "cache", browser_cache))]
pub use seed::*;

#[cfg(feature = "cache")]
mod export;
#[cfg(feature = "cache")]
mod manifest;

#[cfg(feature = "cache")]
pub use export::*;
#[cfg(feature = "cache")]
pub use manifest::*;
