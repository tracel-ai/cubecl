//! Key-value persistence.
//!
//! A [`KvStore`] is an in-memory map that stays authoritative for reads and
//! syncs its content to a [`KvBackend`]: the file system on std targets,
//! browser storage on wasm (feature `browser-cache`), or nothing at all.

mod backend;

pub use backend::*;

#[cfg(any(feature = "cache", browser_cache))]
mod store;

#[cfg(any(feature = "cache", browser_cache))]
pub use store::*;

#[cfg(feature = "cache")]
pub(crate) mod file;

/// Browser storage backend (IndexedDB).
#[cfg(browser_cache)]
pub(crate) mod browser;

/// Chunked cache optimized for compilation artifacts.
#[cfg(feature = "compilation-cache")]
pub mod compilation;

/// Cache root location selection.
#[cfg(feature = "cache")]
mod root;

#[cfg(feature = "cache")]
pub use root::CacheConfig;
