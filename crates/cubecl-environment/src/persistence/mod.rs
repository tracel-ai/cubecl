//! Key-value persistence.
//!
//! A [`Store`] is a typed in-memory map that syncs its content to an optional
//! [`Storage`]: an embedded `SQLite` database on std targets, browser storage
//! on wasm (feature `browser-cache`), or nothing at all. [`CacheOption`]
//! decides whether the whole namespace is ingested at open or entries are
//! faulted in one key at a time.
//!
//! Every cache is identified by a [`Namespace`], a `/`-separated string such
//! as `autotune/0.11.0/cuda-0/matmul`. On std targets all the namespaces of a
//! cache root share one database file (`cubecl.db`) and are told apart by a
//! column rather than by a directory tree. Entries are therefore looked up per
//! key, several processes can share a root safely through WAL, and shipping a
//! subset of them is a query away (see [`crate::bundle`]).

/// The writable half of persistence: where a namespace's entries live.
pub mod storage;

pub use storage::*;

mod namespace;
mod store;

pub use namespace::Namespace;
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
        // Isolate the memory fallback per environment, so a switch after the
        // database failed to open doesn't serve the previous environment's
        // entries.
        None => Box::new(MemoryStorage::in_environment(namespace)),
    }
}
