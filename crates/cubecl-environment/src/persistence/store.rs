use core::{fmt::Display, hash::Hash};

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use hashbrown::HashMap;
use serde::{Serialize, de::DeserializeOwned};

#[cfg(feature = "cache")]
use std::path::PathBuf;

use super::storage::{Origin, Storage};
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
    in_memory_cache: HashMap<K, V>,
    storage: Box<dyn Storage>,
    /// The namespace this instance addresses, `/`-separated:
    /// `<name>/<version>/<segments>`.
    namespace: String,
    /// `false` while an asynchronous storage may still deliver entries that
    /// have not been ingested into [`Self::in_memory_cache`].
    loaded: bool,
}

/// Define the options to create a [`KvStore`].
#[derive(Default, Debug, Clone)]
pub struct KvStoreOptions {
    version: Option<String>,
    name: Option<String>,
    #[cfg(feature = "cache")]
    root: Option<PathBuf>,
}

/// Error related to the store.
#[derive(Debug)]
pub enum KvStoreError<K: Serialize, V: Serialize> {
    /// Can't insert an entry with the same key, but different value. The
    /// conflicting entry may have been inserted by this process or, on the
    /// file system storage, concurrently by another one.
    #[allow(missing_docs)]
    DuplicatedKey {
        key: K,
        value_previous: V,
        value_updated: V,
    },
}

impl KvStoreOptions {
    /// The version used for the store.
    pub fn version<V: Into<String>>(mut self, version: V) -> Self {
        self.version = Some(version.into());
        self
    }

    /// The name of the store, the first segment of its namespace.
    pub fn name<R: Into<String>>(mut self, name: R) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The cache root holding the database file.
    #[cfg(feature = "cache")]
    pub fn root<R: Into<PathBuf>>(mut self, path: R) -> Self {
        self.root = Some(path.into());
        self
    }

    /// The cache root, defaulting to the user cache directory.
    #[cfg(feature = "cache")]
    pub(crate) fn resolve_root(&mut self) -> PathBuf {
        self.root.take().unwrap_or_else(|| {
            etcetera::home_dir()
                .expect("An home directory should exist")
                .join(".cache")
        })
    }

    /// The namespace for `path`: `<name>/<version>/<path>`.
    pub(crate) fn resolve_namespace(&mut self, path: &str) -> String {
        let version = self
            .version
            .take()
            .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());
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
    /// Create a new store, picking the persistence storage for the current
    /// environment, and load existing data if any.
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
        let mut option = option;

        #[cfg(feature = "cache")]
        let root = option.resolve_root();
        #[cfg(feature = "cache")]
        let root = root.to_str();
        #[cfg(not(feature = "cache"))]
        let root: Option<&str> = None;

        let namespace = option.resolve_namespace(path.as_ref());
        let storage = super::storage::open(root, &namespace);

        let mut this = Self::with_storage(storage, namespace);
        this.sync();

        this
    }

    /// Create a new store on an explicit storage, addressing `namespace`.
    ///
    /// No initial synchronization is performed; call [`sync`](KvStore::sync)
    /// to ingest existing content.
    pub fn with_storage<S: Into<String>>(storage: Box<dyn Storage>, namespace: S) -> Self {
        Self {
            in_memory_cache: HashMap::new(),
            storage,
            namespace: namespace.into(),
            loaded: false,
        }
    }

    /// The namespace this instance addresses.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Ingest the storage's entries into the in-memory map.
    pub fn sync(&mut self) {
        // Sampled before the scan: a load completing halfway through would
        // otherwise mark a partial snapshot as fully ingested.
        let loading = self.storage.loading();
        let entries = &mut self.in_memory_cache;

        self.storage.scan(&mut |key, value| {
            if let Some(entry) = decode_entry::<K, V>(key, value) {
                entries.insert(entry.0, entry.1);
            }
        });

        self.loaded = !loading;
    }

    /// Whether the storage is still loading its initial content
    /// asynchronously.
    pub fn loading(&self) -> bool {
        self.storage.loading()
    }

    /// Whether asynchronously delivered content may still be waiting to be
    /// ingested. `false` for synchronous storages (database, memory), whose
    /// content is fully ingested at open.
    pub fn pending_load(&self) -> bool {
        !self.loaded
    }

    /// Iterate over all values of the store.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, func))
    )]
    pub fn for_each<F: FnMut(&K, &V)>(&mut self, mut func: F) {
        if self.pending_load() {
            self.sync();
        }

        for (key, value) in self.in_memory_cache.iter() {
            func(key, value);
        }
    }

    /// Fetch an item from the store.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.in_memory_cache.get(key)
    }

    /// Return all values in the store.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.in_memory_cache.values()
    }

    /// The size of the store.
    pub fn len(&self) -> usize {
        self.in_memory_cache.len()
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
    ///   [`KvStoreError::DuplicatedKey`], and the stored value is left
    ///   untouched. This holds whether this process wrote it or another one
    ///   did concurrently.
    ///
    /// The exception is an entry that came from a bundle: the storage lets a
    /// locally computed value replace it, so a stale bundle can never wedge
    /// the application that imported it.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), KvStoreError<K, V>> {
        if let Some(existing) = self.in_memory_cache.get(&key) {
            // An imported entry is indistinguishable here, so the storage
            // arbitrates: it returns `None` when it accepted the replacement.
            if existing == &value {
                return Ok(());
            }
        }

        let key_bytes = encode(&key);
        let value_bytes = encode(&value);

        // The storage performs the presence check and the write atomically, so
        // a concurrent writer in another process can't be clobbered.
        if let Some(existing) = self.storage.insert(&key_bytes, &value_bytes, Origin::Local)
            && let Some(existing) = decode::<V>(&existing)
        {
            let conflicting = existing != value;
            // Whatever the storage kept is the durable value, so adopt it
            // rather than pretending ours was stored.
            self.in_memory_cache.insert(key.clone(), existing.clone());

            if conflicting {
                return Err(KvStoreError::DuplicatedKey {
                    key,
                    value_previous: existing,
                    value_updated: value,
                });
            }

            return Ok(());
        }

        self.in_memory_cache.insert(key, value);

        Ok(())
    }
}

impl<K: StoreKey, V: StoreValue> Display for KvStore<K, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} ({} entries in {})",
            self.namespace,
            self.in_memory_cache.len(),
            self.storage.describe()
        )
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
    use super::*;

    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn test_cache_simple() {
        let dir = tempfile::tempdir().unwrap();
        let option = KvStoreOptions::default().root(dir.path());

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
    #[cfg_attr(miri, ignore)]
    fn test_on_disk_format_is_stable() {
        use super::super::sqlite::{Database, db_file_name};

        let dir = tempfile::tempdir().unwrap();
        let option = KvStoreOptions::default().root(dir.path()).name("golden");

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
    #[cfg_attr(miri, ignore)]
    fn test_entries_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let options = || KvStoreOptions::default().root(dir.path());

        let mut cache = KvStore::<String, u32>::open("reopen", options());
        cache.insert("key".to_string(), 7).unwrap();
        drop(cache);

        let cache = KvStore::<String, u32>::open("reopen", options());
        assert_eq!(cache.get(&"key".to_string()), Some(&7));
    }

    /// Two namespaces in the same root must not see each other's entries.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn test_stores_are_isolated() {
        let dir = tempfile::tempdir().unwrap();
        let options = || KvStoreOptions::default().root(dir.path());

        let mut first = KvStore::<String, u32>::open("device0/matmul", options());
        first.insert("key".to_string(), 1).unwrap();

        let mut second = KvStore::<String, u32>::open("device1/matmul", options());
        assert_eq!(second.get(&"key".to_string()), None);
        second.insert("key".to_string(), 2).unwrap();

        assert_eq!(first.get(&"key".to_string()), Some(&1));
        assert_eq!(second.get(&"key".to_string()), Some(&2));
    }
}
