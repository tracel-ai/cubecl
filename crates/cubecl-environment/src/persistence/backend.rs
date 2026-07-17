use alloc::string::String;
use alloc::vec::Vec;

/// Append-only persistence backend for a key-value store.
///
/// # Contract
///
/// - [`lock`](KvBackend::lock) grants exclusive access (multi-process on the
///   file system) and returns the raw bytes of entries appended by other
///   parties since the last call — `None` means nothing new, or a degraded
///   backend. On asynchronous backends (browser), this is where hydration
///   results are drained.
/// - [`append`](KvBackend::append) durably (best effort) appends one
///   serialized entry. It must only be called between
///   [`lock`](KvBackend::lock) and [`unlock`](KvBackend::unlock).
/// - Any I/O failure silently degrades the backend to a no-op (`lock` returns
///   `None`, `append` is ignored). Backends log failures but never panic on
///   I/O errors.
pub trait KvBackend: Send + core::fmt::Debug {
    /// Grants exclusive access and returns entry bytes appended by other
    /// parties since the last call.
    fn lock(&mut self) -> Option<Vec<u8>>;

    /// Releases the access granted by [`lock`](KvBackend::lock).
    fn unlock(&mut self);

    /// Appends one serialized entry.
    ///
    /// `dedup_key` uniquely identifies the entry for backends that store
    /// entries individually (browser storage); append-log backends ignore it.
    fn append(&mut self, dedup_key: &str, bytes: &[u8]);

    /// Whether the backend is still loading its initial content
    /// asynchronously. Entries become visible through
    /// [`lock`](KvBackend::lock) once hydration completes.
    fn hydrating(&self) -> bool {
        false
    }

    /// Whether asynchronously delivered content may still be waiting to be
    /// ingested — hydration in flight, or delivered bytes not yet drained.
    ///
    /// `false` for synchronous backends: their content is fully ingested at
    /// open, so callers can skip speculative re-syncs on the hot path.
    fn has_pending(&self) -> bool {
        false
    }

    /// Human-readable location for log messages.
    fn describe(&self) -> String;
}

/// A backend that persists nothing: the store stays memory-only.
///
/// Used on environments without any persistence support (no-std) and as an
/// explicit opt-out.
#[derive(Debug, Default, Clone, Copy)]
pub struct MemoryBackend;

impl KvBackend for MemoryBackend {
    fn lock(&mut self) -> Option<Vec<u8>> {
        None
    }

    fn unlock(&mut self) {}

    fn append(&mut self, _dedup_key: &str, _bytes: &[u8]) {}

    fn describe(&self) -> String {
        String::from("memory (no persistence)")
    }
}
