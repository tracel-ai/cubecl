use alloc::string::String;

use crate::bytes::Bytes;

/// Where the entries of a single namespace are kept, and written to.
///
/// A storage is bound to one namespace at construction (a `/`-separated name
/// such as `autotune/0.11.0/cuda-0/matmul`) and addresses entries by their
/// serialized key bytes.
///
/// This is the writable half of persistence. The read-only half, the entries
/// shipped alongside an application, is [`crate::bundle::Bundle`].
///
/// # Contract
///
/// - [`insert`](Storage::insert) is insert-only: it returns the value
///   already stored under `key`, if any, and leaves it untouched. Returning
///   `None` means the entry was written. The check and the write must be
///   atomic with respect to other processes.
/// - Any I/O failure degrades silently: reads report a miss and writes are
///   dropped. Implementations log failures but never panic on them.
///
/// Methods take `&self` because reads happen behind shared references on the
/// hot path; implementations use interior mutability.
pub trait Storage: Send + core::fmt::Debug {
    /// The value stored under `key`, if any.
    fn get(&self, key: &[u8]) -> Option<Bytes>;

    /// Stores `value` under `key` unless the key is already present, in which
    /// case the existing value is returned and nothing is written.
    fn insert(&self, key: &[u8], value: &[u8]) -> Option<Bytes>;

    /// Visits every entry of the namespace.
    fn scan(&self, visit: &mut dyn FnMut(&[u8], &[u8]));

    /// Whether the storage is still loading its content asynchronously.
    /// Entries become visible through [`get`](Storage::get) and
    /// [`scan`](Storage::scan) once the load completes.
    fn loading(&self) -> bool {
        false
    }

    /// Human-readable location for log messages.
    fn describe(&self) -> String;
}

/// A storage that persists nothing: the store stays memory-only.
///
/// Used on environments without any persistence support (no-std) and as an
/// explicit opt-out.
#[derive(Debug, Default, Clone, Copy)]
pub struct MemoryStorage;

impl Storage for MemoryStorage {
    fn get(&self, _key: &[u8]) -> Option<Bytes> {
        None
    }

    fn insert(&self, _key: &[u8], _value: &[u8]) -> Option<Bytes> {
        None
    }

    fn scan(&self, _visit: &mut dyn FnMut(&[u8], &[u8])) {}

    fn describe(&self) -> String {
        String::from("memory (no persistence)")
    }
}

/// One namespace's contribution to a storage or a bundle, for reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceSummary {
    /// The namespace.
    pub namespace: String,
    /// Number of entries.
    pub entries: u64,
    /// Total size of the keys and values, in bytes.
    pub bytes: u64,
}
