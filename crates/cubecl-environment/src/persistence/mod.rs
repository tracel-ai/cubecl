//! Key-value persistence.
//!
//! A [`KvStore`] is an in-memory map that stays authoritative for reads and
//! syncs its content to a [`Storage`]: an embedded `SQLite` database on std
//! targets, browser storage on wasm (feature `browser-cache`), or nothing at
//! all.
//!
//! Every cache is identified by a namespace, a `/`-separated string such as
//! `autotune/0.11.0/cuda-0/matmul`. On std targets all the namespaces of a
//! cache root share one database file (`cubecl.db`) and are told apart by a
//! column rather than by a directory tree. Entries are therefore looked up per
//! key, several processes can share a root safely through WAL, and shipping a
//! subset of them is a query away (see [`crate::bundle`]).

mod storage;

pub use storage::*;

mod store;

pub use store::*;

/// `SQLite` persistence: the database file shared by every namespace of a
/// cache root.
#[cfg(feature = "cache")]
pub mod sqlite;

#[cfg(feature = "cache")]
pub use sqlite::{DB_FILE_NAME, Database, SqliteStorage};

/// Browser storage (IndexedDB).
#[cfg(browser_cache)]
pub(crate) mod browser;

/// Store optimized for large values, loaded on demand rather than eagerly.
#[cfg(feature = "blob-store")]
pub mod blob;

/// Cache root location selection.
#[cfg(feature = "cache")]
mod root;

#[cfg(feature = "cache")]
pub use root::CacheConfig;

/// The storage serving `namespace` in the cache root at `root`, degrading to
/// memory-only when the database can't be opened.
#[cfg(feature = "cache")]
pub(crate) fn open_storage(
    root: &std::path::Path,
    namespace: &str,
) -> alloc::boxed::Box<dyn Storage> {
    use alloc::{boxed::Box, string::ToString};

    match Database::open_root(root) {
        Some(database) => Box::new(SqliteStorage::new(database, namespace.to_string())),
        None => Box::new(MemoryStorage),
    }
}
