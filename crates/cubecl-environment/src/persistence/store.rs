use core::{cell::Cell, fmt::Display, hash::Hash};

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::{Serialize, de::DeserializeOwned};

use super::blob::BlobStore;
use super::storage::{Insertion, Origin, Storage};
use crate::bytes::Bytes;

#[derive(Debug)]
/// An in-memory key-value store that is automatically synced to a persistence
/// storage: an embedded database on native targets, browser storage on wasm
/// (feature `browser-cache`), or nothing at all.
///
/// All entries of a store are loaded into memory when it is opened, and reads
/// are served from there. Use [`crate::persistence::blob::BlobStore`]
/// instead when the values are large enough that loading all of them eagerly
/// is wasteful.
///
/// # Warning
///
/// ## No Edits
///
/// The biggest constraint for the store is that values should never change for
/// a given key. There is no update possible; if a value is reinserted a second
/// time with a different value but the same key, an error will arise.
///
/// The one exception is imported entries: a locally computed value replaces a
/// value that came from a bundle, because a shipped bundle must never be able
/// to wedge the application that imported it. See [`KvStore::insert`].
pub struct KvStore<K, V> {
    /// A store *is* a storage behind an in-memory cache, which is exactly what
    /// [`BlobStore`] provides. The only thing added here is loading
    /// everything up front, so reads never fall through to the storage.
    blob: BlobStore<K, V>,
    /// `false` while an asynchronous storage may still deliver entries that
    /// have not been ingested. A `Cell` so [`KvStore::get`] can ingest them
    /// itself instead of every caller having to remember to sync first.
    loaded: Cell<bool>,
}

/// Define the options to create a [`KvStore`].
#[derive(Default, Debug, Clone)]
pub struct KvStoreOptions {
    name: Option<String>,
}

/// Error related to a store, shared by [`KvStore`] and
/// [`BlobStore`](super::blob::BlobStore).
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

impl KvStoreOptions {
    /// The name of the store, the first segment of its namespace.
    pub fn name<R: Into<String>>(mut self, name: R) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The namespace for `path`: `<name>/<version>/<path>`.
    ///
    /// The version is this build's, never a caller's choice: it is what makes
    /// entries written by one cubecl invisible to another, so a bundle built
    /// elsewhere can't be read as if it matched.
    pub(crate) fn resolve_namespace(&mut self, path: &str) -> String {
        let version = env!("CARGO_PKG_VERSION");
        let name = self.name.take().unwrap_or_else(|| "cubecl".to_string());

        let path = path.trim_matches('/');
        alloc::format!("{name}/{version}/{path}")
    }
}

/// Trait to be implemented for store keys.
pub trait StoreKey: Serialize + DeserializeOwned + PartialEq + Eq + Hash + Clone {}
/// Trait to be implemented for store values.
pub trait StoreValue: Serialize + DeserializeOwned + PartialEq + Eq + Clone {}

impl<T: Serialize + DeserializeOwned + PartialEq + Eq + Clone + Hash> StoreKey for T {}
impl<T: Serialize + DeserializeOwned + PartialEq + Eq + Clone> StoreValue for T {}

impl<K: StoreKey, V: StoreValue> KvStore<K, V> {
    /// Create a new store and load everything the storage holds.
    ///
    /// `path` is a `/`-separated logical location, namespaced under the store
    /// name and version from the options.
    ///
    /// On asynchronous storages (browser storage), the store returns
    /// immediately with the load in flight: existing entries become visible
    /// after a later [`for_each`](KvStore::for_each) call.
    #[cfg_attr(feature="tracing", tracing::instrument(
        level = "trace",
        skip(path),
        fields(path = ?path.as_ref())))]
    pub fn open<P: AsRef<str>>(path: P, option: KvStoreOptions) -> Self {
        let this = Self {
            blob: BlobStore::new(path, option),
            loaded: Cell::new(false),
        };
        this.sync();

        this
    }

    /// Create a new store on an explicit storage, addressing `namespace`.
    ///
    /// No initial load is performed; call [`sync`](KvStore::sync) to ingest
    /// existing content.
    pub fn with_storage<S: Into<String>>(storage: Box<dyn Storage>, namespace: S) -> Self {
        Self {
            blob: BlobStore::with_storage(storage, namespace),
            loaded: Cell::new(false),
        }
    }

    /// The namespace this instance addresses.
    pub fn namespace(&self) -> &str {
        self.blob.namespace()
    }

    /// Ingest everything the storage holds into memory.
    pub fn sync(&self) {
        // Sampled before the scan: a load completing halfway through would
        // otherwise mark a partial snapshot as fully ingested.
        let loading = self.blob.storage().loading();
        let blob = &self.blob;

        blob.storage().scan(&mut |key, value| {
            if let Some((key, value)) = decode_entry::<K, V>(key, value) {
                blob.memoize(key, value);
            }
        });

        self.loaded.set(!loading);
    }

    /// Whether asynchronously delivered content may still be waiting to be
    /// ingested. `false` for synchronous storages (database, memory), whose
    /// content is fully ingested at open.
    pub fn pending_load(&self) -> bool {
        !self.loaded.get()
    }

    /// Iterate over all values of the store.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, func))
    )]
    pub fn for_each<F: FnMut(&K, &V)>(&self, func: F) {
        if self.pending_load() {
            self.sync();
        }

        self.blob.for_each_memoized(func);
    }

    /// Fetch an item from the store.
    ///
    /// Served from memory: everything a synchronous storage holds was ingested
    /// at open, so a miss there is a miss and not a reason to query. An
    /// asynchronous storage (browser hydration) is ingested here on the first
    /// lookup that follows the load, so no caller has to sync by hand.
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.pending_load() {
            self.sync();
        }

        self.blob.get_memoized(key)
    }

    /// The size of the store.
    pub fn len(&self) -> usize {
        self.blob.memoized_len()
    }

    /// If the store is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert a new item into the store.
    ///
    /// - Key absent: the entry is written to the storage.
    /// - Present with the same value: `Ok`, nothing written.
    /// - Present with a different value: an error,
    ///   [`StoreError::DuplicatedKey`], and the stored value is left
    ///   untouched. This holds whether this process wrote it or another one
    ///   did concurrently.
    ///
    /// The exception is an entry that came from a bundle: the storage lets a
    /// locally computed value replace it, so a stale bundle can never wedge
    /// the application that imported it.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), StoreError<K, V>> {
        self.blob.insert(key, value)
    }
}

impl<K: StoreKey, V: StoreValue> Display for KvStore<K, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} ({} entries in {})",
            self.namespace(),
            self.len(),
            self.blob.storage().describe()
        )
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
            // for this key forever — a permanent recompile for a `BlobStore`
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
    use std::vec;

    use super::*;

    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn test_cache_simple() {
        let dir = tempfile::tempdir().unwrap();
        crate::environment::set_root(dir.path());
        let option = KvStoreOptions::default();

        let key1 = || "key1".to_string();
        let key2 = || "key2".to_string();

        let value1 = || "value1".to_string();
        let value2 = || "value2".to_string();

        let mut cache = KvStore::<String, String>::open("test", option);
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
        let option = KvStoreOptions::default().name("golden");

        let mut cache = KvStore::<String, u32>::open("device0/matmul", option);
        cache.insert("shape=2x2".to_string(), 42).unwrap();

        let expected_namespace =
            std::format!("golden/{}/device0/matmul", env!("CARGO_PKG_VERSION"));
        assert_eq!(cache.namespace(), expected_namespace);

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
        let options = || KvStoreOptions::default();

        let mut cache = KvStore::<String, u32>::open("reopen", options());
        cache.insert("key".to_string(), 7).unwrap();
        drop(cache);

        let cache = KvStore::<String, u32>::open("reopen", options());
        assert_eq!(cache.get(&"key".to_string()), Some(&7));
    }

    /// Two namespaces in the same root must not see each other's entries.
    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn test_stores_are_isolated() {
        let dir = tempfile::tempdir().unwrap();
        crate::environment::set_root(dir.path());
        let options = || KvStoreOptions::default();

        let mut first = KvStore::<String, u32>::open("device0/matmul", options());
        first.insert("key".to_string(), 1).unwrap();

        let mut second = KvStore::<String, u32>::open("device1/matmul", options());
        assert_eq!(second.get(&"key".to_string()), None);
        second.insert("key".to_string(), 2).unwrap();

        assert_eq!(first.get(&"key".to_string()), Some(&1));
        assert_eq!(second.get(&"key".to_string()), Some(&2));
    }
}
