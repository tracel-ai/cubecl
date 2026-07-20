use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::bytes::Bytes;
use crate::sync::{Arc, Lazy, Mutex};

/// Where an entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Computed on this machine.
    Local,
    /// Copied in from a bundle by [`crate::bundle::import`].
    Imported,
}

/// Where the entries of a single namespace are kept, and written to.
///
/// A storage is bound to one namespace at construction (a `/`-separated name
/// such as `autotune/0.11.0/cuda-0/matmul`) and addresses entries by their
/// serialized key bytes.
///
/// This is the only thing read at runtime. Bundles are an import format: they
/// fill a storage once through [`crate::bundle::import`] and are never
/// consulted again.
///
/// # Contract
///
/// - [`insert`](Storage::insert) is insert-only between two [`Origin::Local`]
///   values: it returns the value already stored under `key` and leaves it
///   untouched. Returning `None` means the entry was written.
/// - A [`Origin::Local`] value *replaces* an [`Origin::Imported`] one, so a
///   stale bundle entry can never wedge the application that imported it.
///   An [`Origin::Imported`] value never replaces anything.
/// - The check and the write must be atomic with respect to other processes.
/// - Any I/O failure degrades silently: reads report a miss and writes are
///   dropped. Implementations log failures but never panic on them.
///
/// Methods take `&self` because reads happen behind shared references on the
/// hot path; implementations use interior mutability.
pub trait Storage: Send + core::fmt::Debug {
    /// The value stored under `key`, if any.
    fn get(&self, key: &[u8]) -> Option<Bytes>;

    /// Stores `value` under `key`, returning the existing value when the write
    /// was declined. See the trait contract for when that happens.
    fn insert(&self, key: &[u8], value: &[u8], origin: Origin) -> Option<Bytes>;

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

/// The entries of every memory-backed namespace in this process.
///
/// Shared globally so that a namespace opened twice, or imported and then
/// read, sees the same entries. Without a file system that is the only way an
/// import can outlive the call that performed it.
static MEMORY: Lazy<Mutex<HashMap<String, Arc<Mutex<Namespace>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

type Namespace = HashMap<Vec<u8>, (Bytes, Origin)>;

/// A storage that keeps entries in memory for the lifetime of the process.
///
/// Used where nothing durable is available (no-std, or a cache root that can't
/// be opened). Entries do not survive a restart, so an imported bundle has to
/// be imported again on the next run, which costs nothing but time.
#[derive(Debug, Clone)]
pub struct MemoryStorage {
    namespace: String,
    entries: Arc<Mutex<Namespace>>,
}

impl MemoryStorage {
    /// The in-memory storage for `namespace`, shared process-wide.
    pub fn new(namespace: &str) -> Self {
        let mut memory = MEMORY.lock().expect("Lock recovers from poisoning");
        let entries = match memory.get(namespace) {
            Some(entries) => entries.clone(),
            None => {
                let entries = Arc::new(Mutex::new(HashMap::new()));
                memory.insert(namespace.to_string(), entries.clone());
                entries
            }
        };

        Self {
            namespace: namespace.to_string(),
            entries,
        }
    }

    /// Every namespace held in memory by this process.
    pub fn namespaces() -> Vec<NamespaceSummary> {
        let memory = MEMORY.lock().expect("Lock recovers from poisoning");

        memory
            .iter()
            .map(|(namespace, entries)| {
                let entries = entries.lock().expect("Lock recovers from poisoning");
                NamespaceSummary {
                    namespace: namespace.clone(),
                    entries: entries.len() as u64,
                    bytes: entries
                        .iter()
                        .map(|(key, (value, _))| (key.len() + value.len()) as u64)
                        .sum(),
                }
            })
            .collect()
    }

    /// Forgets every in-memory namespace. Mostly useful in tests.
    pub fn clear() {
        MEMORY.lock().expect("Lock recovers from poisoning").clear();
    }
}

impl Storage for MemoryStorage {
    fn get(&self, key: &[u8]) -> Option<Bytes> {
        let entries = self.entries.lock().expect("Lock recovers from poisoning");
        entries.get(key).map(|(value, _)| value.clone())
    }

    fn insert(&self, key: &[u8], value: &[u8], origin: Origin) -> Option<Bytes> {
        let mut entries = self.entries.lock().expect("Lock recovers from poisoning");

        if let Some((existing, existing_origin)) = entries.get(key)
            && !replaces(origin, *existing_origin)
        {
            return Some(existing.clone());
        }

        entries.insert(
            key.to_vec(),
            (Bytes::from_bytes_vec(value.to_vec()), origin),
        );

        None
    }

    fn scan(&self, visit: &mut dyn FnMut(&[u8], &[u8])) {
        let entries = self.entries.lock().expect("Lock recovers from poisoning");
        for (key, (value, _)) in entries.iter() {
            visit(key, value);
        }
    }

    fn describe(&self) -> String {
        alloc::format!("memory ({})", self.namespace)
    }
}

/// Whether a write of `incoming` may overwrite an entry of `existing`.
///
/// Only one case overwrites: a locally computed value replacing an imported
/// one. That is what keeps a stale bundle entry from wedging the application,
/// now that imported entries live in the storage like any other.
pub(crate) fn replaces(incoming: Origin, existing: Origin) -> bool {
    matches!((incoming, existing), (Origin::Local, Origin::Imported))
}

/// A storage that persists nothing and remembers nothing.
///
/// The explicit opt-out, distinct from [`MemoryStorage`], which does remember.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoStorage;

impl Storage for NoStorage {
    fn get(&self, _key: &[u8]) -> Option<Bytes> {
        None
    }

    fn insert(&self, _key: &[u8], _value: &[u8], _origin: Origin) -> Option<Bytes> {
        None
    }

    fn scan(&self, _visit: &mut dyn FnMut(&[u8], &[u8])) {}

    fn describe(&self) -> String {
        String::from("none (no persistence)")
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

/// Boxes the storage serving `namespace` on this target.
pub(crate) fn open(
    #[cfg_attr(not(feature = "cache"), allow(unused_variables))] root: Option<&str>,
    namespace: &str,
) -> Box<dyn Storage> {
    cfg_if::cfg_if! {
        if #[cfg(all(feature = "cache", not(target_family = "wasm")))] {
            match root {
                Some(root) => super::open_database_storage(std::path::Path::new(root), namespace),
                None => Box::new(MemoryStorage::new(namespace)),
            }
        } else if #[cfg(browser_cache)] {
            super::browser::open_storage(namespace)
        } else {
            Box::new(MemoryStorage::new(namespace))
        }
    }
}
