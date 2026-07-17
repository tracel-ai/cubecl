#[cfg(feature = "cache")]
use core::time::Duration;
use core::{fmt::Display, hash::Hash};

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use hashbrown::HashMap;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[cfg(feature = "cache")]
use std::path::{Path, PathBuf};

use super::backend::KvBackend;
use crate::bundle::SeedSource;
use crate::sync::Arc;

/// Where an in-memory entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Seeded from an installed bundle (index into the store's seed list).
    Bundle(u16),
    /// Inserted locally or loaded from the writable backend.
    Local,
}

/// Which bundle seed sources a store consults when opened.
#[derive(Debug, Clone, Default)]
pub enum SeedMode {
    /// Consult the globally installed bundles (see [`crate::bundle::install`]).
    #[default]
    Auto,
    /// Consult exactly these seed sources.
    Explicit(alloc::vec::Vec<Arc<dyn SeedSource>>),
    /// Ignore bundles entirely.
    Disabled,
}

#[derive(Debug)]
/// An in-memory key-value store that is automatically synced to a persistence
/// backend: the file system on native targets, browser storage on wasm
/// (feature `browser-cache`), or nothing at all.
///
/// The goal is simplicity, ease of use, and ease of distribution. On the file
/// system, all data is stored in a single file, which is automatically loaded
/// into memory when using the store.
///
/// # Warning
///
/// ## No Edits
///
/// The biggest constraint for the store is that values should never change for a given key.
/// There is no update possible; if a value is reinserted a second time with a different value but
/// the same key, an error will arise.
///
/// This is important to keep the file format simple: there is no metadata, no headers, just a plain
/// separator between each entry. Therefore, it isn’t possible to edit previously saved content.
///
/// ## No Big Files
///
/// The store isn’t optimized for space; use it for small caches.
pub struct KvStore<K, V> {
    in_memory_cache: HashMap<K, (V, Origin)>,
    backend: Box<dyn KvBackend>,
    separator: Vec<u8>,
    seeds: Vec<Arc<dyn SeedSource>>,
}

/// Backward-compatible name for [`KvStore`].
pub type Cache<K, V> = KvStore<K, V>;

/// Define the options to create a [`KvStore`].
#[derive(Default, Debug, Clone)]
pub struct KvStoreOptions {
    separator: Option<Vec<u8>>,
    version: Option<String>,
    name: Option<String>,
    seeds: SeedMode,
    #[cfg(feature = "cache")]
    root: Option<PathBuf>,
    #[cfg(feature = "cache")]
    lock_max_duration: Option<Duration>,
}

/// Backward-compatible name for [`KvStoreOptions`].
pub type CacheOption = KvStoreOptions;

/// Error related to the store.
#[derive(Debug)]
pub enum KvStoreError<K: Serialize, V: Serialize> {
    /// Can't insert an entry with the same key, but different value.
    #[allow(missing_docs)]
    DuplicatedKey {
        key: K,
        value_previous: V,
        value_updated: V,
    },
    /// Tried to insert an entry with the same key, but a new entry on disk was just synched with
    /// the same key.
    #[allow(missing_docs)]
    KeyOutOfSync {
        key: K,
        value_previous: V,
        value_updated: V,
    },
}

/// Backward-compatible name for [`KvStoreError`].
pub type CacheError<K, V> = KvStoreError<K, V>;

impl KvStoreOptions {
    /// The separator used between each entry in the store.
    ///
    /// It should not be used in both the keys and the values.
    pub fn separator<S: Into<Vec<u8>>>(mut self, separator: S) -> Self {
        self.separator = Some(separator.into());
        self
    }

    /// The version used for the store.
    pub fn version<V: Into<String>>(mut self, version: V) -> Self {
        self.version = Some(version.into());
        self
    }

    /// The name for the store.
    ///
    /// It will appear in the directory "$HOME/.cache/{name}/"
    pub fn name<R: Into<String>>(mut self, name: R) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The root path for the store.
    ///
    /// It will appear in the directory "{path}/{name}/"
    #[cfg(feature = "cache")]
    pub fn root<R: Into<PathBuf>>(mut self, path: R) -> Self {
        self.root = Some(path.into());
        self
    }

    /// Which bundle seed sources the store consults when opened.
    pub fn seeds(mut self, seeds: SeedMode) -> Self {
        self.seeds = seeds;
        self
    }

    fn resolve_base(&mut self) -> (Vec<u8>, String, String) {
        let separator = self.separator.take().unwrap_or_else(|| b"\n".to_vec());
        let version = self
            .version
            .take()
            .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());
        let name = self.name.take().unwrap_or_else(|| "cubecl".to_string());

        (separator, name, version)
    }

    #[cfg(feature = "cache")]
    pub(crate) fn resolve(mut self) -> (Vec<u8>, String, String, PathBuf, Duration) {
        let (separator, name, version) = self.resolve_base();
        let duration = self
            .lock_max_duration
            .unwrap_or_else(|| Duration::from_secs(30));
        let root = match self.root {
            Some(root) => root,
            None => etcetera::home_dir()
                .expect("An home directory should exist")
                .join(".cache"),
        };

        (separator, name, version, root, duration)
    }
}

/// Trait to be implemented for store keys.
pub trait CacheKey: Serialize + DeserializeOwned + PartialEq + Eq + Hash + Clone {}
/// Trait to be implemented for store values.
pub trait CacheValue: Serialize + DeserializeOwned + PartialEq + Eq + Clone {}

impl<T: Serialize + DeserializeOwned + PartialEq + Eq + Clone + Hash> CacheKey for T {}
impl<T: Serialize + DeserializeOwned + PartialEq + Eq + Clone> CacheValue for T {}

impl<K: CacheKey, V: CacheValue> KvStore<K, V> {
    /// Create a new store, picking the persistence backend for the current
    /// environment, and load existing data if any.
    ///
    /// `path` is a `/`-separated logical location, namespaced under the store
    /// name and version from the options.
    ///
    /// On asynchronous backends (browser storage), the store returns
    /// immediately with hydration in flight: existing entries become visible
    /// after a later [`sync`](KvStore::sync), [`for_each`](KvStore::for_each)
    /// or [`insert`](KvStore::insert) call.
    #[cfg_attr(feature="tracing", tracing::instrument(
        level = "trace",
        skip(path),
        fields(path = ?path.as_ref())))]
    pub fn open<P: AsRef<str>>(path: P, option: KvStoreOptions) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "cache", not(target_family = "wasm")))] {
                Self::open_file(path, option)
            } else if #[cfg(browser_cache)] {
                Self::open_browser(path, option)
            } else {
                let _ = path;
                let mut option = option;
                let (separator, _, _) = option.resolve_base();
                Self::with_backend(Box::new(super::backend::MemoryBackend), separator)
            }
        }
    }

    /// Create a new store persisted with the file system backend.
    #[cfg(feature = "cache")]
    pub(crate) fn open_file<P: AsRef<str>>(path: P, mut option: KvStoreOptions) -> Self {
        let seed_mode = core::mem::take(&mut option.seeds);
        let (separator, name, version, root, lock_max_duration) = option.resolve();
        let rel = relative_store_path(path.as_ref(), &name, &version);
        let path = root.join(&rel);

        let backend = super::file::CacheFile::new(&path, lock_max_duration);
        let mut this = Self::with_backend(Box::new(backend), separator);
        this.seeds = resolve_seeds(seed_mode);
        this.hydrate_seeds(&normalized_rel(&rel));
        this.sync();

        this
    }

    /// Create a new store persisted with the browser storage backend.
    #[cfg(browser_cache)]
    fn open_browser<P: AsRef<str>>(path: P, mut option: KvStoreOptions) -> Self {
        let seed_mode = core::mem::take(&mut option.seeds);
        let (separator, name, version) = option.resolve_base();
        let prefix = browser_prefix(path.as_ref(), &name, &version);
        let rel = browser_rel_path(path.as_ref(), &name, &version);

        let backend = super::browser::BrowserBackend::new(prefix);
        let mut this = Self::with_backend(Box::new(backend), separator);
        this.seeds = resolve_seeds(seed_mode);
        this.hydrate_seeds(&rel);

        this
    }

    /// Create a new store on an explicit backend.
    ///
    /// No initial synchronization is performed; call
    /// [`sync`](KvStore::sync) to ingest existing content.
    pub fn with_backend(backend: Box<dyn KvBackend>, separator: Vec<u8>) -> Self {
        Self {
            in_memory_cache: HashMap::new(),
            backend,
            separator,
            seeds: Vec::new(),
        }
    }

    /// Loads bundle-seeded entries for this store's relative location.
    ///
    /// Runs before the writable backend sync so local entries win on
    /// collision. Between bundles, later installs win.
    fn hydrate_seeds(&mut self, rel: &str) {
        let seeds = core::mem::take(&mut self.seeds);
        for (index, seed) in seeds.iter().enumerate() {
            if let Some(bytes) = seed.kv_bytes(rel) {
                log::debug!("Seeding store {rel} from {}", seed.describe());
                self.seed_content(&bytes, Origin::Bundle(index as u16));
            }
        }
        self.seeds = seeds;
    }

    /// Ingest entries from a bundle seed layer.
    fn seed_content(&mut self, bytes: &[u8], origin: Origin) {
        let mut start = 0;

        while let Some(pos) = bytes[start..]
            .windows(self.separator.len())
            .position(|w| w == self.separator)
        {
            match serde_json::from_slice::<Entry<K, V>>(&bytes[start..start + pos]) {
                Ok(entry) => {
                    self.in_memory_cache
                        .insert(entry.key, (entry.value, origin));
                }
                Err(err) => {
                    log::warn!(
                        "Corrupted bundle entry ({}..{}) : {err}",
                        start,
                        start + pos,
                    );
                }
            };
            start += pos + self.separator.len();
        }
    }

    /// The seed source at the given index, as recorded in
    /// [`Origin::Bundle`] entries.
    pub fn seed(&self, index: u16) -> Option<&Arc<dyn SeedSource>> {
        self.seeds.get(index as usize)
    }

    /// Ingest entries newly delivered by the backend into the in-memory map.
    pub fn sync(&mut self) {
        if let Some(buffer) = self.backend.lock() {
            self.sync_content(&buffer, None).ok();
        }
        self.backend.unlock();
    }

    /// Whether the backend is still loading its initial content
    /// asynchronously.
    pub fn hydrating(&self) -> bool {
        self.backend.hydrating()
    }

    /// Iterate over all values of the store.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, func))
    )]
    pub fn for_each<F: FnMut(&K, &V)>(&mut self, mut func: F) {
        if let Some(buffer) = self.backend.lock() {
            self.sync_content(&buffer, None).ok();
        }

        for (key, (value, _)) in self.in_memory_cache.iter() {
            func(key, value);
        }

        self.backend.unlock();
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
    pub fn insert(&mut self, key: K, value: V) -> Result<(), KvStoreError<K, V>> {
        if let Some(buffer) = self.backend.lock()
            && let Err(err) = self.sync_content(&buffer, Some((&key, &value)))
        {
            self.backend.unlock();
            return Err(err);
        }

        match self.in_memory_cache.get(&key) {
            Some((existing, Origin::Local)) => {
                let result = if existing != &value {
                    Err(KvStoreError::DuplicatedKey {
                        key,
                        value_previous: existing.clone(),
                        value_updated: value,
                    })
                } else {
                    Ok(())
                };
                self.backend.unlock();
                return result;
            }
            Some((existing, Origin::Bundle(_))) => {
                if existing == &value {
                    // The bundle keeps serving this entry; not writing it
                    // locally keeps the local log minimal. Removing the
                    // bundle re-colds such entries.
                    self.backend.unlock();
                    return Ok(());
                }
                // A locally computed value shadows a stale bundle entry.
                log::debug!("Shadowing bundle entry with a locally computed value.");
            }
            None => {}
        }

        self.insert_unchecked(key, value);

        self.backend.unlock();
        Ok(())
    }

    fn sync_content(
        &mut self,
        bytes: &[u8],
        new_insert: Option<(&K, &V)>,
    ) -> Result<(), KvStoreError<K, V>> {
        let mut start = 0;
        let mut result = Ok(());

        while let Some(pos) = bytes[start..]
            .windows(self.separator.len())
            .position(|w| w == self.separator)
        {
            match serde_json::from_slice::<Entry<K, V>>(&bytes[start..start + pos]) {
                Ok(entry) => {
                    if let Some(insert) = &new_insert
                        && result.is_ok()
                        && insert.0 == &entry.key
                        && insert.1 != &entry.value
                    {
                        result = Err(KvStoreError::KeyOutOfSync {
                            key: entry.key.clone(),
                            value_previous: entry.value.clone(),
                            value_updated: insert.1.clone(),
                        })
                    }
                    self.in_memory_cache
                        .insert(entry.key, (entry.value, Origin::Local));
                }
                Err(err) => {
                    log::warn!(
                        "Corrupted store {}, ignoring entry ({}..{}) : {err}",
                        self.backend.describe(),
                        start,
                        start + pos,
                    );
                }
            };
            start += pos + self.separator.len();
        }

        result
    }

    fn insert_unchecked(&mut self, key: K, value: V) {
        let entry = Entry { key, value };
        let mut bytes = serde_json::to_vec(&entry).expect("Can serialize data");

        for b in self.separator.iter() {
            bytes.push(*b);
        }

        let dedup_key = serde_json::to_string(&entry.key).expect("Can serialize key");
        self.backend.append(&dedup_key, &bytes);
        self.in_memory_cache
            .insert(entry.key, (entry.value, Origin::Local));
    }
}

impl<K: CacheKey, V: CacheValue> Display for KvStore<K, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "{}", self.backend.describe())?;

        for (key, (value, _)) in self.in_memory_cache.iter() {
            let key = serde_json::to_string_pretty(key).unwrap();
            let value = serde_json::to_string_pretty(value).unwrap();

            writeln!(f, "  [{key}] => {value}")?;
        }

        Ok(())
    }
}

/// Resolves a seed mode into the concrete seed sources to consult.
fn resolve_seeds(mode: SeedMode) -> Vec<Arc<dyn SeedSource>> {
    match mode {
        SeedMode::Auto => crate::bundle::seeds(),
        SeedMode::Explicit(seeds) => seeds,
        SeedMode::Disabled => Vec::new(),
    }
}

/// The store location relative to the cache root:
/// `<name>/<version>/<segments>.json.log`.
#[cfg(feature = "cache")]
pub(crate) fn relative_store_path(path_partial: &str, name: &str, version: &str) -> PathBuf {
    let path_partial: &Path = Path::new(path_partial);
    let add_extension = !path_partial.ends_with("json.log");

    let mut path = PathBuf::from(sanitize_path_segment(name)).join(sanitize_path_segment(version));

    for segment in path_partial.iter() {
        // Skip the name directory since it resets the previous path segments.
        if segment == "/" {
            continue;
        }
        path = path.join(sanitize_path_segment(segment.to_str().unwrap()));
    }

    if add_extension {
        path.set_extension("json.log");
    }

    path
}

/// The `/`-separated form of a relative store path, used as the bundle seed
/// lookup key on every platform.
#[cfg(feature = "cache")]
pub(crate) fn normalized_rel(rel: &Path) -> String {
    rel.to_string_lossy().replace('\\', "/")
}

/// The seed lookup key on browser targets, mirroring
/// [`relative_store_path`]'s naming with string operations only.
#[cfg(browser_cache)]
pub(crate) fn browser_rel_path(path_partial: &str, name: &str, version: &str) -> String {
    let mut rel = String::new();
    rel.push_str(&sanitize_path_segment(name));
    rel.push('/');
    rel.push_str(&sanitize_path_segment(version));

    let mut ends_with_extension = false;
    for segment in path_partial.split('/').filter(|s| !s.is_empty()) {
        rel.push('/');
        rel.push_str(&sanitize_path_segment(segment));
        ends_with_extension = segment.ends_with("json.log");
    }

    if !ends_with_extension {
        // Mirror `Path::set_extension`: replace the last component's
        // extension when it has one, append otherwise.
        let component_start = rel.rfind('/').map(|pos| pos + 1).unwrap_or(0);
        if let Some(dot) = rel[component_start..].rfind('.') {
            rel.truncate(component_start + dot);
        }
        rel.push_str(".json.log");
    }

    rel
}

/// The browser record key prefix, mirroring the file system layout
/// `<name>/<version>/<segments>`.
#[cfg(browser_cache)]
pub(crate) fn browser_prefix(path_partial: &str, name: &str, version: &str) -> String {
    let mut prefix = String::new();
    prefix.push_str(&sanitize_path_segment(name));
    prefix.push('/');
    prefix.push_str(&sanitize_path_segment(version));

    for segment in path_partial.split('/') {
        if segment.is_empty() {
            continue;
        }
        prefix.push('/');
        prefix.push_str(&sanitize_path_segment(segment));
    }

    prefix.push('/');
    prefix
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Entry<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
}

impl<K: Serialize, V: Serialize> core::fmt::Debug for Entry<K, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let formatted = serde_json::to_string_pretty(self).unwrap();
        write!(f, "{formatted}")
    }
}

#[cfg(feature = "cache")]
pub(crate) fn sanitize_path_segment(segment: &str) -> String {
    sanitize_filename::sanitize_with_options(
        segment,
        sanitize_filename::Options {
            replacement: "_",
            ..Default::default()
        },
    )
}

/// A minimal sanitizer for browser record keys: no file system involved, only
/// path traversal characters matter.
#[cfg(all(browser_cache, not(feature = "cache")))]
pub(crate) fn sanitize_path_segment(segment: &str) -> String {
    segment.replace(['/', '\\'], "_")
}

#[cfg(test)]
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

    /// Guards the on-disk contract: the exact file location (including the
    /// workspace version segment) and the exact serialized line format.
    /// Breaking either invalidates every existing cache on users' machines.
    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn test_on_disk_format_is_stable() {
        let dir = tempfile::tempdir().unwrap();
        let option = KvStoreOptions::default().root(dir.path()).name("golden");

        let mut cache = KvStore::<String, u32>::open("device0/matmul", option);
        cache.insert("shape=2x2".to_string(), 42).unwrap();

        let expected_path = dir
            .path()
            .join("golden")
            .join(env!("CARGO_PKG_VERSION"))
            .join("device0")
            .join("matmul.json.log");
        let content = std::fs::read_to_string(&expected_path)
            .unwrap_or_else(|err| panic!("Golden file missing at {expected_path:?}: {err}"));

        assert_eq!(content, "{\"key\":\"shape=2x2\",\"value\":42}\n");
    }
}
