use alloc::string::String;

use crate::bytes::Bytes;

/// A read-only set of pre-computed cache entries shipped with an application.
///
/// Lookups use the same two coordinates as the local cache: the namespace the
/// persistence layer computes (`<name>/<version>/<segments>`) and the
/// serialized key bytes. A bundle therefore needs no path rewriting, and
/// several bundles in different formats can be installed side by side.
///
/// This is the read-only half of persistence. The writable half is
/// [`crate::persistence::Storage`].
///
/// Reads hand back [`Bytes`], which a format can serve as a zero-copy window
/// into its own storage: an embedded bundle returns a view of its blob rather
/// than a copy.
///
/// All methods degrade silently: a miss on any failure.
pub trait Bundle: Send + Sync + core::fmt::Debug {
    /// The value stored under `key` in `namespace`.
    fn get(&self, namespace: &str, key: &[u8]) -> Option<Bytes>;

    /// Visits every entry of `namespace`.
    fn scan(&self, namespace: &str, visit: &mut dyn FnMut(&[u8], &[u8]));

    /// Human-readable origin for log messages.
    fn describe(&self) -> String;
}
