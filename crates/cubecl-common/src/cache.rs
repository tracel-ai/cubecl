use core::time::Duration;
use core::{fmt::Display, hash::Hash};
use std::{
    io::Read,
    path::{Path, PathBuf},
};

use alloc::vec::Vec;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::cache_file::CacheFile;

#[derive(Debug)]
/// An in-memory key-value cache that is automatically synced to disk.
///
/// The goal is simplicity, ease of use, and ease of distribution. All data is stored in a single
/// file, which is automatically loaded into memory when using the cache.
///
/// # Warning
///
/// ## No Edits
///
/// The biggest constraint for the cache is that values should never change for a given key.
/// There is no update possible; if a value is reinserted a second time with a different value but
/// the same key, an error will arise.
///
/// This is important to keep the file format simple: there is no metadata, no headers, just a plain
/// separator between each cache entry. Therefore, it isn’t possible to edit previously saved content.
///
/// ## No Big Files
///
/// The cache isn’t optimized for space; use it for small caches.
pub struct Cache<K, V> {
    in_memory_cache: HashMap<K, V>,
    file: CacheFile,
    separator: Vec<u8>,
}

/// Define the option to create a cache.
#[derive(Default)]
pub struct CacheOption {
    separator: Option<Vec<u8>>,
    version: Option<String>,
    name: Option<String>,
    root: Option<PathBuf>,
    lock_max_duration: Option<Duration>,
}

/// Error related to caching.
#[derive(Debug)]
pub enum CacheError<K: Serialize, V: Serialize> {
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

impl CacheOption {
    /// The separator used between each entry in the cache.
    ///
    /// It should not be used in both the keys and the values.
    pub fn separator<S: Into<Vec<u8>>>(mut self, separator: S) -> Self {
        self.separator = Some(separator.into());
        self
    }

    /// The version used for the cache.
    pub fn version<V: Into<String>>(mut self, version: V) -> Self {
        self.version = Some(version.into());
        self
    }

    /// The name for the cache.
    ///
    /// It will appear in the directory "$HOME/.cache/{name}/"
    pub fn name<R: Into<String>>(mut self, name: R) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The root path for the cache.
    ///
    /// It will appear in the directory "{path}/{name}/"
    pub fn root<R: Into<PathBuf>>(mut self, path: R) -> Self {
        self.root = Some(path.into());
        self
    }

    fn resolve(self) -> (Vec<u8>, String, String, PathBuf, Duration) {
        let separator = self.separator.unwrap_or_else(|| b"\n".to_vec());
        let version = self
            .version
            .unwrap_or_else(|| std::env!("CARGO_PKG_VERSION").to_string());
        let name = self.name.unwrap_or_else(|| "cubecl".to_string());
        let duration = self
            .lock_max_duration
            .unwrap_or_else(|| Duration::from_secs(30));
        let root = match self.root {
            Some(root) => root,
            None => dirs::home_dir()
                .expect("An home directory should exist")
                .join(".cache"),
        };

        (separator, name, version, root, duration)
    }
}

/// Trait to be implemented for cache keys.
pub trait CacheKey: Serialize + DeserializeOwned + PartialEq + Eq + Hash + Clone {}
/// Trait to be implemented for cache value.
pub trait CacheValue: Serialize + DeserializeOwned + PartialEq + Eq + Clone {}

impl<T: Serialize + DeserializeOwned + PartialEq + Eq + Clone + Hash> CacheKey for T {}
impl<T: Serialize + DeserializeOwned + PartialEq + Eq + Clone> CacheValue for T {}

impl<K: CacheKey, V: CacheValue> Cache<K, V> {
    /// Create a new cache and load the data from the provided path if it exists.
    pub fn new<P: AsRef<Path>>(path: P, option: CacheOption) -> Self {
        let (separator, name, version, root, lock_max_duration) = option.resolve();
        let path = get_persistent_cache_file_path(path, root, name, version);

        let mut this = Self {
            in_memory_cache: HashMap::new(),
            file: CacheFile::new(&path, lock_max_duration),
            separator,
        };

        if let Some(mut reader) = this.file.lock() {
            let mut buffer = Vec::new();
            reader
                .read_to_end(&mut buffer)
                .expect("Can read the cache content");
            this.sync_content(&buffer, None).ok();
        }

        this.file.unlock();

        this
    }

    /// Iterate over all values of the cache.
    pub fn for_each<F: FnMut(&K, &V)>(&mut self, mut func: F) {
        if let Some(mut reader) = self.file.lock() {
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer).unwrap();
            self.sync_content(&buffer, None).ok();
        }

        for (key, value) in self.in_memory_cache.iter() {
            func(key, value)
        }

        self.file.unlock();
    }

    /// Fetch an item from the cache.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.in_memory_cache.get(key)
    }

    /// The size of the cache.
    pub fn len(&self) -> usize {
        self.in_memory_cache.len()
    }

    /// If the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert a new item to the cache.
    ///
    /// Panic if an item with a different value exists in the cache.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), CacheError<K, V>> {
        if let Some(mut reader) = self.file.lock() {
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer).unwrap();

            if let Err(err) = self.sync_content(&buffer, Some((&key, &value))) {
                self.file.unlock();
                return Err(err);
            }
        }

        if let Some(existing) = self.in_memory_cache.get(&key) {
            if existing != &value {
                self.file.unlock();

                return Err(CacheError::DuplicatedKey {
                    key,
                    value_previous: existing.clone(),
                    value_updated: value,
                });
            } else {
                self.file.unlock();
                return Ok(());
            }
        }

        self.insert_unchecked(key, value);

        self.file.unlock();
        Ok(())
    }

    fn sync_content(
        &mut self,
        bytes: &[u8],
        new_insert: Option<(&K, &V)>,
    ) -> Result<(), CacheError<K, V>> {
        let mut start = 0;
        let mut result = Ok(());

        while let Some(pos) = bytes[start..]
            .windows(self.separator.len())
            .position(|w| w == self.separator)
        {
            match serde_json::from_slice::<Entry<K, V>>(&bytes[start..start + pos]) {
                Ok(entry) => {
                    if let Some(insert) = &new_insert {
                        if result.is_ok() && insert.0 == &entry.key && insert.1 != &entry.value {
                            result = Err(CacheError::KeyOutOfSync {
                                key: entry.key.clone(),
                                value_previous: entry.value.clone(),
                                value_updated: insert.1.clone(),
                            })
                        }
                    }
                    self.in_memory_cache.insert(entry.key, entry.value);
                }
                Err(err) => {
                    log::warn!(
                        "Corrupted cache file {}, ignoring entry ({}..{}) : {err}",
                        self.file,
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

        self.file.write(&bytes);
        self.in_memory_cache.insert(entry.key, entry.value);
    }
}

impl<K: CacheKey, V: CacheValue> Display for Cache<K, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.file)?;

        for (key, value) in self.in_memory_cache.iter() {
            let key = serde_json::to_string_pretty(key).unwrap();
            let value = serde_json::to_string_pretty(value).unwrap();

            writeln!(f, "  [{key}] => {value}")?;
        }

        Ok(())
    }
}

fn get_persistent_cache_file_path<P: AsRef<Path>>(
    path_partial: P,
    root: PathBuf,
    name: String,
    version: String,
) -> PathBuf {
    let path_partial: &Path = path_partial.as_ref();
    let add_extension = !path_partial.ends_with("json.log");

    let mut path = root
        .join(sanitize_path_segment(&name))
        .join(sanitize_path_segment(&version));

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

#[derive(Serialize, Deserialize)]
struct Entry<K, V> {
    key: K,
    value: V,
}

impl<K: Serialize, V: Serialize> core::fmt::Debug for Entry<K, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let formatted = serde_json::to_string_pretty(self).unwrap();
        write!(f, "{formatted}")
    }
}

pub(crate) fn sanitize_path_segment(segment: &str) -> String {
    sanitize_filename::sanitize_with_options(
        segment,
        sanitize_filename::Options {
            replacement: "_",
            ..Default::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_simple() {
        let key1 = || "key1".to_string();
        let key2 = || "key2".to_string();

        let value1 = || "value1".to_string();
        let value2 = || "value2".to_string();

        let mut cache = Cache::<String, String>::new("test", CacheOption::default());
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
}
