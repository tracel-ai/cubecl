use alloc::string::String;
use alloc::vec::Vec;

/// Persistence backend for a single key-value store.
///
/// A backend is bound to one logical store at construction (a `/`-separated
/// name such as `autotune/0.11.0/cuda-0/matmul`) and addresses entries by
/// their serialized key bytes.
///
/// # Contract
///
/// - [`insert`](KvBackend::insert) is insert-only: it returns the value
///   already stored under `key`, if any, and leaves it untouched. Returning
///   `None` means the entry was written. The check and the write must be
///   atomic with respect to other processes.
/// - Any I/O failure silently degrades the backend: reads report a miss and
///   writes are dropped. Backends log failures but never panic on them.
///
/// Methods take `&self` because reads happen behind shared references on the
/// hot path; backends use interior mutability.
pub trait KvBackend: Send + core::fmt::Debug {
    /// The value stored under `key`, if any.
    fn get(&self, key: &[u8]) -> Option<Vec<u8>>;

    /// Stores `value` under `key` unless the key is already present, in which
    /// case the existing value is returned and nothing is written.
    fn insert(&self, key: &[u8], value: &[u8]) -> Option<Vec<u8>>;

    /// Visits every entry of the store.
    fn scan(&self, visit: &mut dyn FnMut(Vec<u8>, Vec<u8>));

    /// Whether the backend is still loading its content asynchronously.
    /// Entries become visible through [`get`](KvBackend::get) and
    /// [`scan`](KvBackend::scan) once hydration completes.
    fn hydrating(&self) -> bool {
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
    fn get(&self, _key: &[u8]) -> Option<Vec<u8>> {
        None
    }

    fn insert(&self, _key: &[u8], _value: &[u8]) -> Option<Vec<u8>> {
        None
    }

    fn scan(&self, _visit: &mut dyn FnMut(Vec<u8>, Vec<u8>)) {}

    fn describe(&self) -> String {
        String::from("memory (no persistence)")
    }
}
