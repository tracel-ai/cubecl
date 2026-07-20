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

/// The writable half of persistence: where a namespace's entries live.
pub mod storage;

pub use storage::*;

mod store;

pub use store::*;

/// `SQLite` persistence: the database file shared by every namespace of a
/// cache root.
#[cfg(native_cache)]
pub mod sqlite;

#[cfg(native_cache)]
pub use sqlite::{Database, SqliteStorage, db_file_name};

/// Browser storage (IndexedDB).
#[cfg(browser_cache)]
pub(crate) mod browser;

/// A storage behind an in-memory cache, filled on demand. [`KvStore`] is this
/// with everything loaded up front.
pub mod blob;

/// Cache root location selection.
///
/// Available wherever there is a file system, not only when the `SQLite`
/// backend is compiled in: the root is what names an environment on disk, and
/// [`crate::environment`] exposes it independently of how entries are stored.
#[cfg(std_io)]
mod root;

#[cfg(std_io)]
pub use root::CacheConfig;

/// The database-backed storage serving `namespace` in the active
/// environment, degrading to process-wide memory when the database can't be
/// opened.
#[cfg(native_cache)]
pub(crate) fn open_database_storage(namespace: &str) -> alloc::boxed::Box<dyn Storage> {
    use alloc::{boxed::Box, string::ToString};

    match Database::open_active() {
        Some(database) => Box::new(SqliteStorage::new(database, namespace.to_string())),
        None => Box::new(MemoryStorage::new(namespace)),
    }
}
