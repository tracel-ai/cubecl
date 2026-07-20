use alloc::string::String;
use alloc::vec::Vec;

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
/// All methods degrade silently: a miss on any failure.
pub trait Bundle: Send + Sync + core::fmt::Debug {
    /// The value stored under `key` in `namespace`.
    fn get(&self, namespace: &str, key: &[u8]) -> Option<Vec<u8>>;

    /// Visits every entry of `namespace`.
    fn scan(&self, namespace: &str, visit: &mut dyn FnMut(Vec<u8>, Vec<u8>));

    /// Human-readable origin for log messages.
    fn describe(&self) -> String;
}
