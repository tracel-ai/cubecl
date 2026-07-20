//! Key-value persistence.
//!
//! A [`KvStore`] is an in-memory map that stays authoritative for reads and
//! syncs its content to a [`KvBackend`]: an embedded `SQLite` database on std
//! targets, browser storage on wasm (feature `browser-cache`), or nothing at
//! all.
//!
//! On std targets every store of a cache root lives in a single database file
//! (`cubecl.db`), addressed by a `store` column rather than by a directory
//! tree. Entries are therefore looked up per key, several processes can share
//! a root safely through WAL, and shipping a subset of them is a query away
//! (see [`crate::bundle`]).

mod backend;

pub use backend::*;

#[cfg(any(feature = "cache", browser_cache))]
mod store;

#[cfg(any(feature = "cache", browser_cache))]
pub use store::*;

/// `SQLite` persistence: the database file shared by every store of a cache
/// root.
#[cfg(feature = "cache")]
pub mod sqlite;

#[cfg(feature = "cache")]
pub use sqlite::{DB_FILE_NAME, Database, SqliteBackend, StoreSummary};

/// Browser storage backend (IndexedDB).
#[cfg(browser_cache)]
pub(crate) mod browser;

/// Cache optimized for large values, loaded on demand rather than eagerly.
#[cfg(feature = "compilation-cache")]
pub mod compilation;

/// Cache root location selection.
#[cfg(feature = "cache")]
mod root;

#[cfg(feature = "cache")]
pub use root::CacheConfig;

/// The backend serving `store` in the cache root at `root`, degrading to
/// memory-only when the database can't be opened.
#[cfg(feature = "cache")]
pub(crate) fn open_backend(
    root: &std::path::Path,
    store: &str,
) -> alloc::boxed::Box<dyn KvBackend> {
    use alloc::{boxed::Box, string::ToString};

    match Database::open_root(root) {
        Some(database) => Box::new(SqliteBackend::new(database, store.to_string())),
        None => Box::new(MemoryBackend),
    }
}
