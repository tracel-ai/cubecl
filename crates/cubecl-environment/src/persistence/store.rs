use core::{fmt::Display, hash::Hash};

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use hashbrown::HashMap;
use serde::{Serialize, de::DeserializeOwned};

#[cfg(feature = "cache")]
use std::path::PathBuf;

use super::storage::Storage;
use crate::bundle::Bundle;
use crate::bytes::Bytes;
use crate::sync::Arc;

/// Where an in-memory entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Served by an installed bundle (index into the store's bundle list).
    Bundle(u16),
    /// Computed on this machine, or loaded from the writable storage.
    Local,
}

/// Which bundles a store consults when opened.
#[derive(Debug, Clone, Default)]
pub enum BundleMode {
    /// Consult the globally installed bundles (see [`crate::bundle::install`]).
    #[default]
    Installed,
    /// Consult exactly these bundles.
    Explicit(Vec<Arc<dyn Bundle>>),
    /// Ignore bundles entirely.
    Disabled,
}

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
/// The one exception is bundle-provided entries: a locally computed value
/// silently *shadows* a stale bundle entry instead of erroring, because a
/// shipped bundle must never be able to break the application that installs
/// it. See [`KvStore::insert`].
pub struct KvStore<K, V> {
    in_memory_cache: HashMap<K, (V, Origin)>,
    storage: Box<dyn Storage>,
    /// The namespace this instance addresses, `/`-separated:
    /// `<name>/<version>/<segments>`.
    namespace: String,
    bundles: Vec<Arc<dyn Bundle>>,
    /// `false` while an asynchronous storage may still deliver entries that
    /// have not been ingested into [`Self::in_memory_cache`].
    loaded: bool,
}

/// Define the options to create a [`KvStore`].
#[derive(Default, Debug, Clone)]
pub struct KvStoreOptions {
    version: Option<String>,
    name: Option<String>,
    bundles: BundleMode,
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

    /// Which bundles the store consults when opened.
    pub fn bundles(mut self, bundles: BundleMode) -> Self {
        self.bundles = bundles;
        self
    }

    /// Takes the configured bundle mode, leaving the default behind.
    pub(crate) fn take_bundles(&mut self) -> BundleMode {
        core::mem::take(&mut self.bundles)
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
        let bundle_mode = option.take_bundles();

        cfg_if::cfg_if! {
            if #[cfg(all(feature = "cache", not(target_family = "wasm")))] {
                let root = option.resolve_root();
                let namespace = option.resolve_namespace(path.as_ref());
                let storage = super::open_storage(&root, &namespace);
            } else if #[cfg(browser_cache)] {
                let namespace = option.resolve_namespace(path.as_ref());
                let storage = super::browser::open_storage(&namespace);
            } else {
                let namespace = option.resolve_namespace(path.as_ref());
                let storage: Box<dyn Storage> = Box::new(super::storage::MemoryStorage);
            }
        }

        let mut this = Self::with_storage(storage, namespace);
        this.bundles = resolve_bundles(bundle_mode);
        this.load_bundles();
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
            bundles: Vec::new(),
            loaded: false,
        }
    }

    /// The namespace this instance addresses.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Loads the entries installed bundles provide for this namespace.
    ///
    /// Runs before the writable storage sync so local entries win on
    /// collision. Between bundles, later installs win.
    fn load_bundles(&mut self) {
        let bundles = core::mem::take(&mut self.bundles);

        for (index, bundle) in bundles.iter().enumerate() {
            let mut count = 0;
            let entries = &mut self.in_memory_cache;

            bundle.scan(&self.namespace, &mut |key, value| {
                if let Some(entry) = decode_entry::<K, V>(key, value) {
                    entries.insert(entry.0, (entry.1, Origin::Bundle(index as u16)));
                    count += 1;
                }
            });

            if count > 0 {
                log::debug!(
                    "Loaded {count} entries into {} from {}",
                    self.namespace,
                    bundle.describe()
                );
            }
        }

        self.bundles = bundles;
    }

    /// Ingest the storage's entries into the in-memory map.
    ///
    /// Entries loaded here win over bundled ones: a value computed on this
    /// machine is always preferred to a shipped one.
    pub fn sync(&mut self) {
        // Sampled before the scan: a load completing halfway through would
        // otherwise mark a partial snapshot as fully ingested.
        let loading = self.storage.loading();
        let entries = &mut self.in_memory_cache;

        self.storage.scan(&mut |key, value| {
            if let Some(entry) = decode_entry::<K, V>(key, value) {
                entries.insert(entry.0, (entry.1, Origin::Local));
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

        for (key, (value, _)) in self.in_memory_cache.iter() {
            func(key, value);
        }
    }

    /// Fetch an item from the store.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.in_memory_cache.get(key).map(|(value, _)| value)
    }

    /// Fetch an item from the store along with where it came from.
    pub fn get_with_origin(&self, key: &K) -> Option<(&V, Origin)> {
        self.in_memory_cache
            .get(key)
            .map(|(value, origin)| (value, *origin))
    }

    /// Return all values in the store.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.in_memory_cache.values().map(|(value, _)| value)
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
    /// Insert-only semantics depend on where the existing entry came from:
    ///
    /// - Key absent: the entry is written to the storage.
    /// - Present with the same value: `Ok`, nothing written (a bundle-served
    ///   entry keeps being served by the bundle).
    /// - Present with a different value, [`Origin::Local`]: an error,
    ///   [`KvStoreError::DuplicatedKey`]. The stored value is left untouched,
    ///   including when another process wrote it concurrently.
    /// - Present with a different value, [`Origin::Bundle`]: the locally
    ///   computed value silently shadows the stale bundle entry and is
    ///   written locally. Bundles are never trusted over local computation.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), KvStoreError<K, V>> {
        match self.in_memory_cache.get(&key) {
            Some((existing, Origin::Local)) => {
                return if existing != &value {
                    Err(KvStoreError::DuplicatedKey {
                        key,
                        value_previous: existing.clone(),
                        value_updated: value,
                    })
                } else {
                    Ok(())
                };
            }
            Some((existing, Origin::Bundle(_))) => {
                if existing == &value {
                    // The bundle keeps serving this entry; not writing it
                    // locally keeps the local database minimal. Removing the
                    // bundle re-colds such entries.
                    return Ok(());
                }
                // A locally computed value shadows a stale bundle entry.
                log::debug!("Shadowing bundle entry with a locally computed value.");
            }
            None => {}
        }

        let key_bytes = encode(&key);
        let value_bytes = encode(&value);

        // The storage performs the presence check and the write atomically, so
        // a concurrent writer in another process can't be clobbered.
        if let Some(existing) = self.storage.insert(&key_bytes, &value_bytes)
            && let Some(existing) = decode::<V>(&existing)
            && existing != value
        {
            // Another process won the race with a different value. It is the
            // durable one, so adopt it rather than pretending ours was stored.
            self.in_memory_cache
                .insert(key.clone(), (existing.clone(), Origin::Local));

            return Err(KvStoreError::DuplicatedKey {
                key,
                value_previous: existing,
                value_updated: value,
            });
        }

        self.in_memory_cache.insert(key, (value, Origin::Local));

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

/// Resolves a bundle mode into the concrete bundles to consult.
pub(crate) fn resolve_bundles(mode: BundleMode) -> Vec<Arc<dyn Bundle>> {
    match mode {
        BundleMode::Installed => crate::bundle::installed(),
        BundleMode::Explicit(bundles) => bundles,
        BundleMode::Disabled => Vec::new(),
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
        let option = KvStoreOptions::default()
            .root(dir.path())
            .bundles(BundleMode::Disabled);

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
        use super::super::sqlite::{DB_FILE_NAME, Database};

        let dir = tempfile::tempdir().unwrap();
        let option = KvStoreOptions::default()
            .root(dir.path())
            .name("golden")
            .bundles(BundleMode::Disabled);

        let mut cache = KvStore::<String, u32>::open("device0/matmul", option);
        cache.insert("shape=2x2".to_string(), 42).unwrap();

        let expected_namespace =
            std::format!("golden/{}/device0/matmul", env!("CARGO_PKG_VERSION"));
        assert_eq!(cache.namespace(), expected_namespace);

        let path = dir.path().join(DB_FILE_NAME);
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
        let options = || {
            KvStoreOptions::default()
                .root(dir.path())
                .bundles(BundleMode::Disabled)
        };

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
        let options = || {
            KvStoreOptions::default()
                .root(dir.path())
                .bundles(BundleMode::Disabled)
        };

        let mut first = KvStore::<String, u32>::open("device0/matmul", options());
        first.insert("key".to_string(), 1).unwrap();

        let mut second = KvStore::<String, u32>::open("device1/matmul", options());
        assert_eq!(second.get(&"key".to_string()), None);
        second.insert("key".to_string(), 2).unwrap();

        assert_eq!(first.get(&"key".to_string()), Some(&1));
        assert_eq!(second.get(&"key".to_string()), Some(&2));
    }
}
