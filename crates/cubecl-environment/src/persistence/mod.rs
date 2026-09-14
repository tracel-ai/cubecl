//! Key-value persistence.
//!
//! A [`Store`] is a typed in-memory map that syncs its content to an optional
//! [`Storage`]: an embedded Turso database (feature `persistence`) on a file
//! system natively and on OPFS in the browser, or nothing at all.
//! [`CacheOption`] decides whether the whole namespace is ingested at open or
//! entries are faulted in one key at a time.
//!
//! Every cache is identified by a [`Namespace`], a `/`-separated string such
//! as `autotune/0.11.0/cuda-0/matmul`. All the namespaces of an environment
//! share one database file and are told apart by a column rather than by a
//! directory tree. Entries are therefore looked up per key, several processes
//! can share a root safely through WAL, and shipping a subset of them is a
//! query away (see [`crate::bundle`]).

/// The writable half of persistence: where a namespace's entries live.
pub mod storage;

pub use storage::*;

mod namespace;
mod store;

pub use namespace::Namespace;
pub use store::*;

/// Turso persistence: the database file shared by every namespace of an
/// environment.
#[cfg(any(native_cache, browser_cache))]
mod turso;

#[cfg(browser_cache)]
mod turso_browser;

/// Cache root location selection.
///
/// Available wherever there is a file system, not only when the Turso
/// backend is compiled in: the root is what names an environment on disk, and
/// [`crate::environment`] exposes it independently of how entries are stored.
#[cfg(std_io)]
mod root;

#[cfg(std_io)]
pub use root::CacheConfig;
