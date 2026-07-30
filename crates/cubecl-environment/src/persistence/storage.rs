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

/// What a [`Storage::insert`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Insertion {
    /// The storage now holds the value that was passed in.
    Stored,
    /// The storage declined the write and kept a different value, which is
    /// returned. Someone else — another process, or an earlier run — got
    /// there first.
    Conflict(Bytes),
    /// The backend refused the write: a full disk, a lock held past the busy
    /// timeout, a revoked permission. Nothing was stored, and the message
    /// says why.
    Failed(String),
}

/// What an [`insert_many`](Storage::insert_many) did, counted per outcome.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct InsertSummary {
    /// Entries written.
    pub stored: usize,
    /// Entries the storage already held under a different value.
    pub conflict: usize,
    /// Entries the backend refused.
    pub failed: usize,
}

impl InsertSummary {
    /// Counts one outcome.
    pub fn record(&mut self, insertion: &Insertion) {
        match insertion {
            Insertion::Stored => self.stored += 1,
            Insertion::Conflict(_) => self.conflict += 1,
            Insertion::Failed(_) => self.failed += 1,
        }
    }
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
///   values: it leaves the stored value untouched and reports
///   [`Insertion::Conflict`] with it.
/// - A [`Origin::Local`] value *replaces* an [`Origin::Imported`] one, so a
///   stale bundle entry can never wedge the application that imported it.
///   An [`Origin::Imported`] value never replaces anything.
/// - [`replace`](Storage::replace) ignores both rules. It exists for one
///   caller: repairing a row whose bytes no longer decode, which no `insert`
///   could ever agree with.
/// - [`purge`](Storage::purge) and [`purge_key`](Storage::purge_key) are the
///   deletions: the whole namespace, or one entry. Everything else is
///   insert-only.
/// - The check and the write must be atomic with respect to other processes.
/// - A read that fails reports a miss: a cache entry we can't read is one we
///   recompute. A *write* that fails reports [`Insertion::Failed`] rather
///   than passing for success, so a caller can tell "someone else got there
///   first" from "the write did not happen". Implementations log failures but
///   never panic on them.
/// - [`scan`](Storage::scan) may run the visitor while holding the backend's
///   lock, and several namespaces of one environment share that lock. The
///   visitor must therefore not touch any other store of the same
///   environment: doing so deadlocks.
///
/// Methods take `&self` because reads happen behind shared references on the
/// hot path; implementations use interior mutability.
pub trait Storage: Send + core::fmt::Debug {
    /// The value stored under `key`, if any.
    fn get(&self, key: &[u8]) -> Option<Bytes>;

    /// Stores `value` under `key`. See the trait contract for when the write
    /// is declined.
    fn insert(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion;

    /// Stores `value` under `key`, overwriting whatever is there.
    ///
    /// Bypasses the insert-only rule, so it never reports a conflict. Only for
    /// repairing an entry that can't be decoded; ordinary writes go through
    /// [`insert`](Storage::insert).
    fn replace(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion;

    /// Stores many entries under the same rules as
    /// [`insert`](Storage::insert).
    ///
    /// Backends that can commit a batch atomically override this; the default
    /// is one `insert` per entry.
    fn insert_many(
        &self,
        entries: &mut dyn Iterator<Item = (Bytes, Bytes)>,
        origin: Origin,
    ) -> InsertSummary {
        let mut summary = InsertSummary::default();
        for (key, value) in entries {
            summary.record(&self.insert(&key, value, origin));
        }
        summary
    }

    /// Visits every entry of the namespace.
    ///
    /// The visitor must not read or write another store of the same
    /// environment; see the trait contract.
    fn scan(&self, visit: &mut dyn FnMut(&[u8], &[u8]));

    /// Deletes every entry of the namespace, durably.
    ///
    /// A failed delete is logged, not reported: the entries were expendable
    /// cache content either way, and whatever survives is arbitrated like any
    /// other pre-existing entry.
    fn purge(&self);

    /// Deletes the entry under `key`, durably. Same failure contract as
    /// [`purge`](Storage::purge).
    fn purge_key(&self, key: &[u8]);

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
static MEMORY: Lazy<Mutex<HashMap<String, Arc<Mutex<Entries>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub(crate) type Entries = HashMap<Vec<u8>, (Bytes, Origin)>;

/// The [`Storage`] contract applied to an in-memory namespace.
///
/// Every backend that keeps entries in memory shares these, so the insert
/// arbitration exists once and the backends cannot drift on the contract
/// documented on [`Storage`]. They take the map rather than owning it because
/// the backends disagree on the lock around it and on what they do after a
/// write lands.
pub(crate) mod entries {
    use super::{Bytes, Entries, Insertion, Origin, replaces};

    pub(crate) fn get(entries: &Entries, key: &[u8]) -> Option<Bytes> {
        entries.get(key).map(|(value, _)| value.clone())
    }

    pub(crate) fn insert(
        entries: &mut Entries,
        key: &[u8],
        value: Bytes,
        origin: Origin,
    ) -> Insertion {
        if let Some((existing, existing_origin)) = entries.get(key)
            && !replaces(origin, *existing_origin)
        {
            return Insertion::Conflict(existing.clone());
        }

        entries.insert(key.to_vec(), (value, origin));

        Insertion::Stored
    }

    pub(crate) fn replace(
        entries: &mut Entries,
        key: &[u8],
        value: Bytes,
        origin: Origin,
    ) -> Insertion {
        entries.insert(key.to_vec(), (value, origin));

        Insertion::Stored
    }

    pub(crate) fn scan(entries: &Entries, visit: &mut dyn FnMut(&[u8], &[u8])) {
        for (key, (value, _)) in entries.iter() {
            visit(key, value);
        }
    }
}

/// A storage that keeps entries in memory for the lifetime of the process.
///
/// Used where nothing durable is available (no-std, or a cache root that can't
/// be opened). Entries do not survive a restart, so an imported bundle has to
/// be imported again on the next run, which costs nothing but time.
#[derive(Debug, Clone)]
pub struct MemoryStorage {
    namespace: String,
    entries: Arc<Mutex<Entries>>,
}

impl MemoryStorage {
    /// The in-memory storage for `namespace`, shared process-wide across every
    /// environment.
    ///
    /// The environment-bound path uses [`in_environment`](Self::in_environment)
    /// instead, which isolates the entries per environment so a switch doesn't
    /// serve the previous one's data. This unscoped constructor is for explicit
    /// storages that aren't tied to an environment (tests, benches).
    pub fn new(namespace: &str) -> Self {
        Self::with_key(namespace.to_string(), namespace)
    }

    /// Like [`new`](Self::new) but isolated per active environment.
    ///
    /// The database backend keys its entries by the environment's file path, so
    /// a switch reopens a different file; the memory fallback has no file, so it
    /// folds the environment [`scope`](crate::environment::scope) into its
    /// global key to get the same isolation. Without this, a bound store that
    /// resets after a switch would reopen the memory storage and immediately
    /// re-ingest the previous environment's entries.
    pub(crate) fn in_environment(namespace: &str) -> Self {
        // `\u{1f}` (unit separator) can appear in neither a `/`-separated
        // namespace nor a file-system path, so the split back out is
        // unambiguous.
        let key = alloc::format!("{}\u{1f}{namespace}", crate::environment::scope());
        Self::with_key(key, namespace)
    }

    fn with_key(key: String, namespace: &str) -> Self {
        let mut memory = MEMORY.lock();
        let entries = match memory.get(&key) {
            Some(entries) => entries.clone(),
            None => {
                let entries = Arc::new(Mutex::new(HashMap::new()));
                memory.insert(key, entries.clone());
                entries
            }
        };

        Self {
            namespace: namespace.to_string(),
            entries,
        }
    }

    /// Every namespace the active environment holds in memory.
    ///
    /// Only the active environment's entries are reported: the process-wide map
    /// also holds other environments' entries and unscoped explicit storages,
    /// but a summary is always about the environment in effect right now.
    pub fn namespaces() -> Vec<NamespaceSummary> {
        let prefix = alloc::format!("{}\u{1f}", crate::environment::scope());
        let memory = MEMORY.lock();

        memory
            .iter()
            .filter_map(|(key, entries)| {
                let namespace = key.strip_prefix(&prefix)?;
                let entries = entries.lock();
                Some(NamespaceSummary {
                    namespace: namespace.to_string(),
                    entries: entries.len() as u64,
                    bytes: entries
                        .iter()
                        .map(|(key, (value, _))| (key.len() + value.len()) as u64)
                        .sum(),
                })
            })
            .collect()
    }
}

impl Storage for MemoryStorage {
    fn get(&self, key: &[u8]) -> Option<Bytes> {
        entries::get(&self.entries.lock(), key)
    }

    fn insert(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion {
        entries::insert(&mut self.entries.lock(), key, value, origin)
    }

    fn replace(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion {
        entries::replace(&mut self.entries.lock(), key, value, origin)
    }

    fn scan(&self, visit: &mut dyn FnMut(&[u8], &[u8])) {
        entries::scan(&self.entries.lock(), visit)
    }

    fn purge(&self) {
        self.entries.lock().clear();
    }

    fn purge_key(&self, key: &[u8]) {
        self.entries.lock().remove(key);
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

/// The storage serving `namespace` in the active environment.
///
/// The location is not a parameter: an environment is the store, so a cache
/// can't be opened somewhere else without making "a single active
/// environment" false. See [`crate::environment`].
pub fn open(namespace: &str) -> Box<dyn Storage> {
    cfg_if::cfg_if! {
        if #[cfg(native_cache)] {
            super::open_database_storage(namespace)
        } else if #[cfg(browser_cache)] {
            super::browser::open_storage(namespace)
        } else {
            Box::new(MemoryStorage::in_environment(namespace))
        }
    }
}
