//! Named environment bundles.
//!
//! A bundle packages pre-warmed caches (autotune results, compiled kernels)
//! under a human-chosen name such as "H100 Linux" so applications can ship
//! them and skip cold-start tuning and compilation.
//!
//! # Import, not a runtime layer
//!
//! A bundle is only ever used to *fill* the local storage, through
//! [`import`]. Once imported, its entries are ordinary rows and the file can
//! be deleted: runtime lookups go to
//! [`Storage`](crate::persistence::Storage) and nothing else. There is no
//! read path in which a bundle participates, so no cache hit ever depends on
//! a file staying installed.
//!
//! Entries land with [`Origin::Imported`](crate::persistence::Origin), which
//! lets a locally computed value replace one that turns out to be stale.
//!
//! # Formats
//!
//! [`Bundle`] is deliberately format-agnostic: a bundle answers by namespace
//! and key bytes, exactly like the local storage, so nothing about a layout on
//! disk is load-bearing. [`SqliteBundle`] is the native format, a single
//! `SQLite` file. [`EmbeddedBundle`] is one flat blob for wasm and no-std
//! targets, which have no file system to open.
//!
//! Writing is native-only on purpose. A bundle for any target is produced on a
//! development machine by [`export`], and only consumed elsewhere.
//!
//! # Correctness
//!
//! Bundles are never trusted for correctness: imported autotune entries go
//! through the same checksum validation as local ones, and every namespace
//! carries the cubecl version and the device fingerprint, so a mismatched
//! machine never looks them up. A wrong bundle costs load time, nothing
//! else.

mod base;
mod embedded;
mod import;
// The manifest is the description of a bundle, not a way of storing one, so it
// is available wherever a bundle can be read: the flat format exists for the
// targets `cache` can't reach, and they need the same schema guards.
mod manifest;

pub use base::*;
pub use embedded::*;
pub use import::*;
pub use manifest::*;

#[cfg(native_cache)]
mod export;
#[cfg(native_cache)]
mod flat;
#[cfg(native_cache)]
mod open;
#[cfg(native_cache)]
mod sqlite;

#[cfg(native_cache)]
pub use export::*;
#[cfg(native_cache)]
pub use open::*;
#[cfg(native_cache)]
pub use sqlite::*;
