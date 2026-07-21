use core::{fmt::Display, hash::Hash};

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::{HashMap, HashSet};
use serde::{Serialize, de::DeserializeOwned};

use super::namespace::Namespace;
use super::storage::{Insertion, Origin, Storage};
use crate::bytes::Bytes;

/// Error related to a [`Store`].
#[derive(Debug)]
pub enum StoreError<K, V> {
    /// This process already stored a different value under that key: the same
    /// function was computed twice with disagreeing results.
    #[allow(missing_docs)]
    DuplicatedKey {
        key: K,
        value_previous: V,
        value_updated: V,
    },
    /// The durable entry was written by someone else — another process sharing
    /// the cache root, or a bundle import — before this insert reached it.
    /// Benign: the two values are equally valid and the stored one stays.
    #[allow(missing_docs)]
    KeyOutOfSync {
        key: K,
        value_previous: V,
        value_updated: V,
    },
    /// The storage backend refused the write, so the entry is not durable and
    /// will be recomputed on the next run.
    #[allow(missing_docs)]
    Backend { key: K, error: String },
}

impl<K, V> StoreError<K, V> {
    /// Why the write failed, without the key or the value.
    ///
    /// [`Display`] renders both, which is right for small entries and wrong
    /// for a compiled kernel: those values are megabytes of binary, and a log
    /// line must not carry one. Callers storing large values report this
    /// instead.
    pub fn reason(&self) -> &str {
        match self {
            Self::DuplicatedKey { .. } => "the key was already stored with a different value",
            Self::KeyOutOfSync { .. } => "another process stored the key first",
            Self::Backend { error, .. } => error,
        }
    }
}

impl<K: core::fmt::Debug, V: core::fmt::Debug> Display for StoreError<K, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DuplicatedKey {
                key,
                value_previous,
                value_updated,
            } => write!(
                f,
                "key {key:?} was already stored with a different value: \
                 kept {value_previous:?}, dropped {value_updated:?}"
            ),
            Self::KeyOutOfSync {
                key,
                value_previous,
                value_updated,
            } => write!(
                f,
                "key {key:?} was stored concurrently: kept {value_previous:?}, \
                 dropped {value_updated:?}"
            ),
            Self::Backend { key, error } => {
                write!(f, "storing key {key:?} failed: {error}")
            }
        }
    }
}

impl<K: core::fmt::Debug, V: core::fmt::Debug> core::error::Error for StoreError<K, V> {}

/// Trait to be implemented for store keys.
pub trait StoreKey: Serialize + DeserializeOwned + PartialEq + Eq + Hash + Clone {}
/// Trait to be implemented for store values.
pub trait StoreValue: Serialize + DeserializeOwned + PartialEq + Eq + Clone {}

impl<T: Serialize + DeserializeOwned + PartialEq + Eq + Clone + Hash> StoreKey for T {}
impl<T: Serialize + DeserializeOwned + PartialEq + Eq + Clone> StoreValue for T {}

/// How a [`Store`] populates its in-memory map from its storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheOption {
    /// Scan everything at open. Reads are complete: a miss is a miss and
    /// never queries the storage.
    #[default]
    Eager,
    /// Fault entries in from the storage on first access. Right when the
    /// values are large enough that loading all of them eagerly is wasteful,
    /// typically compilation artifacts.
    Lazy,
}

/// Where a [`Store`] persists its entries.
#[derive(Debug, Default)]
enum StorageOption {
    /// Nowhere: the in-memory map is all there is.
    #[default]
    InMemory,
    /// The active environment's storage for this namespace.
    Environment(Namespace),
    /// An explicitly provided storage, addressing this namespace.
    Explicit(Box<dyn Storage>, Namespace),
}

/// Defines how to create a [`Store`].
#[derive(Debug, Default)]
pub struct StoreOptions {
    storage: StorageOption,
    cache: CacheOption,
}

impl StoreOptions {
    /// Options for a store with no persistence and everything in memory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Persist entries under `namespace` in the active environment.
    pub fn storage<N: Into<Namespace>>(mut self, namespace: N) -> Self {
        self.storage = StorageOption::Environment(namespace.into());
        self
    }

    /// Persist entries to an explicit storage addressing `namespace`,
    /// bypassing the active environment. Mostly for tests and benches.
    pub fn storage_with<N: Into<Namespace>>(
        mut self,
        storage: Box<dyn Storage>,
        namespace: N,
    ) -> Self {
        self.storage = StorageOption::Explicit(storage, namespace.into());
        self
    }

    /// How the in-memory map is populated from the storage. Meaningless
    /// without one: an in-memory store keeps everything regardless.
    pub fn cache(mut self, cache: CacheOption) -> Self {
        self.cache = cache;
        self
    }
}

/// A typed key-value store over an optional persistence [`Storage`]: an
/// embedded database on native targets, browser storage on wasm (feature
/// `browser-cache`), or nothing at all.
///
/// Reads follow `HashMap`'s shape and mutation rules, with no interior
/// mutability: [`get`](Store::get) serves shared references from memory,
/// while everything that may change the store — [`insert`](Store::insert),
/// [`get_mut`](Store::get_mut), [`remove`](Store::remove),
/// [`sync`](Store::sync) — requires `&mut self`. Share a store by wrapping it
/// in a lock, not by cloning it.
///
/// [`CacheOption`] decides how the map is populated: [`Eager`
/// ](CacheOption::Eager) ingests the whole namespace at open and serves every
/// read from memory, [`Lazy`](CacheOption::Lazy) reads one key at a time on
/// demand.
///
/// # No Edits
///
/// A stored value never changes: there is no update, and reinserting a key
/// with a different value is an error. The one exception is imported entries:
/// a locally computed value replaces a value that came from a bundle, because
/// a shipped bundle must never be able to wedge the application that imported
/// it. See [`Store::insert`]. Mutating a value through
/// [`get_mut`](Store::get_mut) changes only the in-memory copy, never the
/// storage.
///
/// On an asynchronous storage (browser) a read can miss until the background
/// load finishes, which costs a recompute and nothing else; any `&mut`
/// operation ingests newly delivered content first.
///
/// # Environment switches
///
/// A store opened on the active environment stays bound to *the environment*,
/// not to the storage it opened: when [`crate::environment`] switches
/// ([`activate`](crate::environment::activate),
/// [`load`](crate::environment::load), ...), the store detects it and resets —
/// reads miss instead of serving the old environment's entries, and the next
/// `&mut` operation drops the in-memory state and reopens the storage.
/// Detection is one relaxed atomic load, so it costs nothing while the
/// environment stays put. Stores on an explicit or absent storage are not
/// bound and never reset.
pub struct Store<K, V> {
    entries: HashMap<K, V>,
    /// Keys this process interacted with without the map holding their value:
    /// [`CacheOption::Lazy`] inserts (which drop the value rather than retain
    /// megabytes of compiled artifacts never read again) and
    /// [`remove`](Store::remove)d entries. What tells a reinsert collision
    /// [`StoreError::DuplicatedKey`] apart from [`StoreError::KeyOutOfSync`].
    known: HashSet<K>,
    storage: Option<Box<dyn Storage>>,
    namespace: Option<Namespace>,
    cache: CacheOption,
    /// `false` while an asynchronous storage may still deliver entries that
    /// the eager map has not ingested.
    loaded: bool,
    /// The environment generation the state belongs to, for stores bound to
    /// the active environment; `None` for unbound ones (explicit storage, or
    /// none). A mismatch with the current generation means everything here
    /// describes an environment that is no longer active.
    generation: Option<u32>,
}

impl<K: StoreKey, V: StoreValue> Store<K, V> {
    /// Create a new store from the options.
    ///
    /// With an [`Eager`](CacheOption::Eager) cache over a storage, everything
    /// the storage holds is ingested before returning. On asynchronous
    /// storages (browser) the store returns with the load in flight: existing
    /// entries become visible to a later `&mut` operation or
    /// [`sync`](Store::sync).
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip_all, fields(options = ?options))
    )]
    pub fn new(options: StoreOptions) -> Self {
        let (storage, namespace, generation) = match options.storage {
            StorageOption::InMemory => (None, None, None),
            StorageOption::Environment(namespace) => {
                // Sampled before the storage opens: a switch landing in
                // between leaves a stale generation, which reads as "reset",
                // never as "this storage belongs to the new environment".
                let generation = crate::environment::generation();
                (
                    Some(super::storage::open(namespace.as_str())),
                    Some(namespace),
                    Some(generation),
                )
            }
            StorageOption::Explicit(storage, namespace) => (Some(storage), Some(namespace), None),
        };

        let mut store = Self {
            entries: HashMap::new(),
            known: HashSet::new(),
            storage,
            namespace,
            cache: options.cache,
            loaded: false,
            generation,
        };

        match (store.cache, &store.storage) {
            (CacheOption::Eager, Some(_)) => store.sync(),
            // Nothing to ingest eagerly: lazy reads consult the storage per
            // key, and an in-memory map is always complete.
            _ => store.loaded = true,
        }

        store
    }

    /// The namespace this store addresses, if it persists anywhere.
    pub fn namespace(&self) -> Option<&Namespace> {
        self.namespace.as_ref()
    }

    /// Fetch an item from memory.
    ///
    /// Never touches the storage: on an eager store the map is complete, so a
    /// miss is a miss. On a lazy store this only serves entries a previous
    /// [`get_mut`](Store::get_mut) faulted in; use `get_mut` to read through.
    ///
    /// After an environment switch everything in memory belongs to the old
    /// environment, so this misses rather than serve it — a miss costs a
    /// recompute, a stale hit would be wrong.
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.stale() {
            return None;
        }

        self.entries.get(key)
    }

    /// Fetch an item, reading it from the storage on the first lookup of a
    /// lazy store and memoizing it afterwards.
    ///
    /// Mutating the value changes only the in-memory copy, never the storage.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.reset_if_stale();
        self.refresh_if_pending();

        if matches!(self.cache, CacheOption::Lazy)
            && !self.entries.contains_key(key)
            && let Some(value) = self.fetch(key)
        {
            self.entries.insert(key.clone(), value);
        }

        self.entries.get_mut(key)
    }

    /// Take an item out of memory, reading through to the storage on a lazy
    /// store, and hand it out owned.
    ///
    /// The storage is untouched: entries are never deleted from it, and this
    /// key stays [`known`](StoreError::DuplicatedKey) to the store. This is
    /// the read for a value consumed once per process — a compiled kernel
    /// about to be loaded — because nothing is cloned and nothing stays
    /// memoized.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.reset_if_stale();
        self.refresh_if_pending();

        let value = match self.entries.remove(key) {
            Some(value) => Some(value),
            None => match self.cache {
                CacheOption::Eager => None,
                CacheOption::Lazy => self.fetch(key),
            },
        };

        // Only when durable: for a purely in-memory store the entry is simply
        // gone, and reinserting the key is a fresh insert.
        if value.is_some() && self.storage.is_some() {
            self.known.insert(key.clone());
        }

        value
    }

    /// Insert a new item into the store.
    ///
    /// - Key absent: the entry is written to the storage.
    /// - Present with the same value: `Ok`, nothing written.
    /// - Present with a different value: an error, and the stored value is
    ///   left untouched — [`StoreError::DuplicatedKey`] when this process
    ///   wrote or read it, [`StoreError::KeyOutOfSync`] when another process
    ///   sharing the environment got there first. Both are routine, not bugs.
    ///
    /// The exception is an entry that came from a bundle: the storage lets a
    /// locally computed value replace it, so a stale bundle can never wedge
    /// the application that imported it.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), StoreError<K, V>> {
        self.reset_if_stale();
        self.refresh_if_pending();

        let known = match self.entries.get(&key) {
            Some(existing) if existing == &value => return Ok(()),
            existing => existing.is_some() || self.known.contains(&key),
        };

        let Some(storage) = self.storage.as_deref() else {
            return match self.entries.get(&key) {
                Some(existing) => Err(StoreError::DuplicatedKey {
                    value_previous: existing.clone(),
                    value_updated: value,
                    key,
                }),
                None => {
                    self.entries.insert(key, value);
                    Ok(())
                }
            };
        };

        // Only memory is consulted above: `write_through` asks the storage
        // atomically, so reading it first would cost a second round trip and
        // still not know whether the existing entry is imported.
        match write_through(storage, &key, &value) {
            Written::Stored => {
                self.record(key, value);
                Ok(())
            }
            Written::Failed(error) => Err(StoreError::Backend { key, error }),
            Written::Conflict(existing) => {
                // A later read must serve the durable value, so the eager map
                // memoizes it; the clone only happens on this cold path.
                if matches!(self.cache, CacheOption::Eager) {
                    self.entries.insert(key.clone(), existing.clone());
                } else {
                    self.known.insert(key.clone());
                }

                // `known` is what tells the two conflicts apart: this process
                // saw the key before, or someone else stored it first. The
                // second is a routine multi-process race, not a bug.
                let (value_previous, value_updated) = (existing, value);
                Err(if known {
                    StoreError::DuplicatedKey {
                        key,
                        value_previous,
                        value_updated,
                    }
                } else {
                    StoreError::KeyOutOfSync {
                        key,
                        value_previous,
                        value_updated,
                    }
                })
            }
        }
    }

    /// Ingest everything the storage holds into memory.
    ///
    /// This is what makes an eager store complete, and [`new`](Store::new)
    /// performs it; call it again to ingest content an asynchronous storage
    /// delivered since, or content another store wrote to a shared storage.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip_all, fields(namespace = ?self.namespace))
    )]
    pub fn sync(&mut self) {
        self.reset_if_stale();

        let Some(storage) = self.storage.as_deref() else {
            self.loaded = true;
            return;
        };

        // Sampled before the scan: a load completing halfway through would
        // otherwise mark a partial snapshot as fully ingested.
        let loading = storage.loading();
        let entries = &mut self.entries;

        storage.scan(&mut |key, value| {
            if let Some((key, value)) = decode_entry::<K, V>(key, value) {
                entries.insert(key, value);
            }
        });

        self.loaded = !loading;
    }

    /// Whether asynchronously delivered content may still be waiting to be
    /// ingested. `false` for synchronous storages (database, memory), whose
    /// content is fully ingested at open. Also `true` right after an
    /// environment switch, whose content is pending until the reset.
    pub fn pending_load(&self) -> bool {
        !self.loaded || self.stale()
    }

    /// Iterate over all in-memory entries of the store.
    pub fn for_each<F: FnMut(&K, &V)>(&self, mut func: F) {
        if self.stale() {
            return;
        }

        for (key, value) in self.entries.iter() {
            func(key, value);
        }
    }

    /// How many entries are in memory.
    pub fn len(&self) -> usize {
        if self.stale() {
            return 0;
        }

        self.entries.len()
    }

    /// If nothing is in memory.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One entry read straight from the storage.
    fn fetch(&self, key: &K) -> Option<V> {
        let bytes = self.storage.as_deref()?.get(&encode(key))?;
        decode::<V>(&bytes)
    }

    /// Records a value the storage accepted, according to the cache option.
    fn record(&mut self, key: K, value: V) {
        match self.cache {
            CacheOption::Eager => {
                self.entries.insert(key, value);
            }
            // The value is dropped rather than memoized: freshly compiled
            // artifacts are typically never read again by this process.
            CacheOption::Lazy => {
                self.known.insert(key);
            }
        }
    }

    /// Ingests content an asynchronous storage delivered since the last scan.
    /// A one-bool check once the load completed.
    fn refresh_if_pending(&mut self) {
        if !self.loaded {
            self.sync();
        }
    }

    /// Whether the in-memory state belongs to an environment that is no
    /// longer active. One relaxed atomic load for bound stores; unbound ones
    /// are never stale.
    fn stale(&self) -> bool {
        match self.generation {
            Some(generation) => generation != crate::environment::generation(),
            None => false,
        }
    }

    /// Drops everything belonging to the previous environment and reopens the
    /// storage against the active one.
    ///
    /// The eager rescan is not performed here: `loaded` is left `false`, so
    /// the caller's ordinary refresh ingests the new environment in the same
    /// operation.
    fn reset_if_stale(&mut self) {
        if !self.stale() {
            return;
        }

        // `stale` implies `generation` and an environment-bound namespace.
        let (Some(namespace), Some(_)) = (&self.namespace, self.generation) else {
            return;
        };

        log::debug!("Environment switched, resetting the store for {namespace}");

        // Generation first, storage second, mirroring `new`: a switch landing
        // in between reads as stale again, never as up to date.
        self.generation = Some(crate::environment::generation());
        self.storage = Some(super::storage::open(namespace.as_str()));
        self.entries.clear();
        self.known.clear();
        self.loaded = matches!(self.cache, CacheOption::Lazy);
    }
}

impl<K, V> core::fmt::Debug for Store<K, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Store")
            .field("namespace", &self.namespace)
            .field("cache", &self.cache)
            .field("entries", &self.entries.len())
            .field("known", &self.known.len())
            .field("storage", &self.storage)
            .field("loaded", &self.loaded)
            .finish()
    }
}

impl<K: StoreKey, V: StoreValue> Display for Store<K, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match (&self.namespace, &self.storage) {
            (Some(namespace), Some(storage)) => write!(
                f,
                "{namespace} ({} entries in {})",
                self.len(),
                storage.describe()
            ),
            _ => write!(f, "in-memory ({} entries)", self.len()),
        }
    }
}

/// The outcome of writing an entry through to the storage.
pub(crate) enum Written<V> {
    /// The storage now holds this value, or already held an identical one.
    Stored,
    /// The storage kept a different value, which is the durable one.
    Conflict(V),
    /// The backend refused the write; nothing is durable.
    Failed(String),
}

/// Writes `value` through to `storage` and lets it arbitrate.
///
/// The storage decides what happens on a collision: it lets a local value
/// replace an imported one, and otherwise refuses to overwrite. Both stores go
/// through here, so the rule is identical for eager and lazy caches rather
/// than reimplemented per store.
pub(crate) fn write_through<K: StoreKey, V: StoreValue>(
    storage: &dyn Storage,
    key: &K,
    value: &V,
) -> Written<V> {
    let key_bytes = encode(key);

    match storage.insert(&key_bytes, encode(value), Origin::Local) {
        Insertion::Stored => Written::Stored,
        Insertion::Failed(error) => Written::Failed(error),
        Insertion::Conflict(existing) => match decode::<V>(&existing) {
            Some(existing) if &existing != value => Written::Conflict(existing),
            Some(_) => Written::Stored,
            // Bytes that don't decode are bytes no later insert could ever
            // agree with, so leaving them in place would refuse every write
            // for this key forever — a permanent recompile for a lazily read
            // key. Repair the row instead.
            None => match storage.replace(&key_bytes, encode(value), Origin::Local) {
                Insertion::Failed(error) => Written::Failed(error),
                _ => Written::Stored,
            },
        },
    }
}

/// Serializes a key or a value to its stored representation.
pub(crate) fn encode<T: Serialize>(value: &T) -> Bytes {
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(value, &mut bytes).expect("Can serialize data");
    Bytes::from_bytes_vec(bytes)
}

/// Deserializes a key or a value, reporting corrupted content instead of
/// failing: a cache entry we can't read is one we recompute.
pub(crate) fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    match ciborium::de::from_reader(bytes) {
        Ok(value) => Some(value),
        Err(err) => {
            log::warn!("Corrupted cache entry, ignoring it: {err}");
            None
        }
    }
}

fn decode_entry<K: StoreKey, V: StoreValue>(key: &[u8], value: &[u8]) -> Option<(K, V)> {
    Some((decode::<K>(key)?, decode::<V>(value)?))
}

#[cfg(all(test, feature = "cache"))]
mod tests {
    use std::string::ToString;
    use std::vec;

    use super::*;

    fn eager(path: &str) -> StoreOptions {
        StoreOptions::new().storage(Namespace::new(path))
    }

    fn lazy(path: &str) -> StoreOptions {
        eager(path).cache(CacheOption::Lazy)
    }

    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn test_cache_simple() {
        let dir = tempfile::tempdir().unwrap();
        crate::environment::set_root(dir.path());

        let key1 = || "key1".to_string();
        let key2 = || "key2".to_string();

        let value1 = || "value1".to_string();
        let value2 = || "value2".to_string();

        let mut cache = Store::<String, String>::new(eager("test"));
        cache.insert(key1(), value1()).unwrap();
        cache.insert(key2(), value2()).unwrap();

        let result = cache.insert(key1(), value2());
        assert!(
            result.is_err(),
            "Can't reinsert the same key with a different value."
        );

        assert_eq!(cache.len(), 2);

        let value1_actual = cache.get(&key1()).unwrap();
        assert_eq!(value1_actual, &value1());

        let value2_actual = cache.get(&key2()).unwrap();
        assert_eq!(value2_actual, &value2());
    }

    /// Guards the on-disk contract: the database file location and the exact
    /// namespace a given set of options resolves to. Breaking either
    /// invalidates every existing cache on users' machines.
    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn test_on_disk_format_is_stable() {
        use super::super::sqlite::{Database, db_file_name};

        let dir = tempfile::tempdir().unwrap();
        crate::environment::set_root(dir.path());
        let namespace = Namespace::scoped("golden", "device0/matmul");

        let mut cache = Store::<String, u32>::new(StoreOptions::new().storage(namespace));
        cache.insert("shape=2x2".to_string(), 42).unwrap();

        let expected_namespace =
            std::format!("golden/{}/device0/matmul", env!("CARGO_PKG_VERSION"));
        assert_eq!(cache.namespace().unwrap().as_str(), expected_namespace);

        let path = dir.path().join(db_file_name(&crate::environment::active()));
        assert!(path.exists(), "Database missing at {path:?}");

        // Read it back through a fresh connection: the entry must be
        // addressable by namespace and encoded key alone.
        let database = Database::open(&path, true).unwrap();
        let stored = database
            .get(&expected_namespace, &encode(&"shape=2x2".to_string()))
            .expect("Entry should be stored");
        assert_eq!(decode::<u32>(&stored), Some(42));
    }

    /// A store reopened over the same root must see what the previous one
    /// wrote, without any bundle involved.
    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn test_entries_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        crate::environment::set_root(dir.path());

        let mut cache = Store::<String, u32>::new(eager("reopen"));
        cache.insert("key".to_string(), 7).unwrap();
        drop(cache);

        let cache = Store::<String, u32>::new(eager("reopen"));
        assert_eq!(cache.get(&"key".to_string()), Some(&7));
    }

    /// Two namespaces in the same root must not see each other's entries.
    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn test_stores_are_isolated() {
        let dir = tempfile::tempdir().unwrap();
        crate::environment::set_root(dir.path());

        let mut first = Store::<String, u32>::new(eager("device0/matmul"));
        first.insert("key".to_string(), 1).unwrap();

        let mut second = Store::<String, u32>::new(eager("device1/matmul"));
        assert_eq!(second.get(&"key".to_string()), None);
        second.insert("key".to_string(), 2).unwrap();

        assert_eq!(first.get(&"key".to_string()), Some(&1));
        assert_eq!(second.get(&"key".to_string()), Some(&2));
    }

    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn lazy_values_survive_reopen_and_load_lazily() {
        let dir = tempfile::tempdir().unwrap();
        crate::environment::set_root(dir.path());

        let mut cache = Store::<String, Bytes>::new(lazy("ptx_sm90"));
        cache
            .insert(
                "kernel_a".to_string(),
                Bytes::from_bytes_vec(std::vec![1, 2, 3]),
            )
            .unwrap();
        cache
            .insert(
                "kernel_b".to_string(),
                Bytes::from_bytes_vec(std::vec![4, 5]),
            )
            .unwrap();
        // A lazy insert records the key but never retains the artifact.
        assert!(cache.is_empty());
        drop(cache);

        let mut cache = Store::<String, Bytes>::new(lazy("ptx_sm90"));
        // Nothing is read until a key is asked for.
        assert!(cache.is_empty());

        assert_eq!(
            cache.get_mut(&"kernel_a".to_string()).map(|v| v.to_vec()),
            Some(std::vec![1, 2, 3])
        );
        assert_eq!(cache.len(), 1, "get_mut memoizes");
        assert_eq!(
            cache.remove(&"kernel_b".to_string()).map(|v| v.to_vec()),
            Some(std::vec![4, 5])
        );
        assert_eq!(cache.len(), 1, "remove reads through without memoizing");
        assert_eq!(cache.get_mut(&"missing".to_string()), None);
    }

    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn lazy_reinserting_a_different_value_errors() {
        let dir = tempfile::tempdir().unwrap();
        crate::environment::set_root(dir.path());

        let mut cache = Store::<String, Bytes>::new(lazy("ptx_sm90"));
        let kernel = |byte: u8| Bytes::from_bytes_vec(std::vec![byte]);
        cache.insert("kernel".to_string(), kernel(1)).unwrap();

        assert!(cache.insert("kernel".to_string(), kernel(1)).is_ok());
        let error = cache.insert("kernel".to_string(), kernel(2));
        assert!(matches!(error, Err(StoreError::DuplicatedKey { .. })));

        // Taking the value out doesn't forget the key: the entry is still
        // durable, so a disagreeing reinsert stays a duplicate.
        assert!(cache.remove(&"kernel".to_string()).is_some());
        let error = cache.insert("kernel".to_string(), kernel(2));
        assert!(matches!(error, Err(StoreError::DuplicatedKey { .. })));
    }

    /// A bound store follows the environment: a switch makes reads miss
    /// instead of serving the old environment, and the next `&mut` access
    /// reopens against the new one.
    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn switching_environments_resets_bound_stores() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();

        crate::environment::set_root(first.path());
        let mut store = Store::<String, u32>::new(eager("reset"));
        store.insert("key".to_string(), 1).unwrap();
        assert_eq!(store.get(&"key".to_string()), Some(&1));

        // The old environment's entries are never served after the switch.
        crate::environment::set_root(second.path());
        assert_eq!(store.get(&"key".to_string()), None);
        assert_eq!(store.len(), 0);
        assert!(store.pending_load());

        // The next write lands in the new environment, with no conflict
        // against the value the old one holds.
        store.insert("key".to_string(), 2).unwrap();
        assert_eq!(store.get(&"key".to_string()), Some(&2));

        // Switching back serves the first environment's value again.
        crate::environment::set_root(first.path());
        store.sync();
        assert_eq!(store.get(&"key".to_string()), Some(&1));
    }

    /// Stores on an explicit or absent storage are not bound to the
    /// environment and must not reset on a switch.
    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn unbound_stores_survive_environment_switches() {
        let root = tempfile::tempdir().unwrap();

        let mut store = Store::<String, u32>::new(StoreOptions::new());
        store.insert("key".to_string(), 1).unwrap();

        crate::environment::set_root(root.path());
        assert_eq!(store.get(&"key".to_string()), Some(&1));
    }

    #[test]
    fn in_memory_store_needs_no_storage() {
        let mut store = Store::<String, u32>::new(StoreOptions::new());

        store.insert("key".to_string(), 1).unwrap();
        assert_eq!(store.get(&"key".to_string()), Some(&1));
        assert!(store.insert("key".to_string(), 1).is_ok());
        assert!(matches!(
            store.insert("key".to_string(), 2),
            Err(StoreError::DuplicatedKey { .. })
        ));

        // Nothing durable behind the map: a removed entry is simply gone and
        // the key is free again.
        assert_eq!(store.remove(&"key".to_string()), Some(1));
        store.insert("key".to_string(), 2).unwrap();
        assert_eq!(store.get(&"key".to_string()), Some(&2));
    }
}
