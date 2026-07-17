//! Named environment bundles.
//!
//! A bundle packages pre-warmed caches (autotune results, compiled kernels)
//! under a human-chosen name such as "H100 Linux" so applications can ship
//! them and skip cold-start tuning and compilation.
//!
//! # Format
//!
//! A bundle is a directory with a `bundle.toml` manifest and a `store/`
//! subdirectory mirroring the cache root layout exactly:
//!
//! ```text
//! <bundle>/bundle.toml
//! <bundle>/store/autotune/<version>/<device_id>/<name>.json.log
//! <bundle>/store/cuda/<version>/ptx_sm90/{toc.json.log, chunk0.cbor}
//! ```
//!
//! Exporting is therefore a plain file copy of a warm cache root (see
//! [`export`]), and loading needs no path rewriting.
//!
//! # Correctness
//!
//! Bundles are never trusted for correctness: autotune entries seeded from a
//! bundle go through the same checksum validation as local ones, and
//! compilation entries live under version- and architecture-fingerprinted
//! paths that a mismatched machine never looks up. A wrong bundle costs load
//! time, nothing else.

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
