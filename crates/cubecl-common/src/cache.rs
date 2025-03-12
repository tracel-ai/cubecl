use core::{fmt::Display, hash::Hash};
use std::{
    io::Read,
    path::{Path, PathBuf},
};

use alloc::vec::Vec;
use hashbrown::HashMap;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::cache_file::CacheFile;

#[derive(Debug)]
/// An in-memory key-value cache that is automatically synced on disk.
///
/// The goal is simplicity, ease of use and ease of distribution. All data are stored in a single
/// file, which is automatically loaded in memory when creating the cache.
///
/// # Warning
///
/// The bigguest constraint for the cache is that values should never change for the given key.
/// The is no update possible, if a value is reinserted a second time with a different value but
/// the same key, a panic will arise.
///
/// This is important to keep the file format simple, there is no metadata, no headers, just plain
/// separator between each cache entry. It isn't possible to edit previously saved content.
///
/// The cubecl version is taken for versionning, but you can customize this using
/// [cache option](CacheOption).
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
    root: Option<String>,
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

    /// The root directory for the cache.
    ///
    /// It will appear in the directory "$HOME/.cache/{root}/"
    pub fn root<R: Into<String>>(mut self, root: R) -> Self {
        self.root = Some(root.into());
        self
    }

    fn resolve(self) -> (Vec<u8>, String, String) {
        let separator = self.separator.unwrap_or_else(|| b"\n".to_vec());
        let version = self
            .version
            .unwrap_or_else(|| std::env!("CARGO_PKG_VERSION").to_string());
        let root = self.root.unwrap_or_else(|| "cubecl".to_string());

        (separator, root, version)
    }
}

impl<K, V> Cache<K, V>
where
    K: Serialize + DeserializeOwned + Hash + PartialEq + Eq,
    V: Serialize + DeserializeOwned + PartialEq,
{
    /// Create a new cache and load the data from the provided path if it exist.
    pub fn new<P: AsRef<Path>>(path: P, option: CacheOption) -> Self {
        let (separator, root, version) = option.resolve();
        let path = get_persistent_cache_file_path(path, root, version);

        let mut this = Self {
            in_memory_cache: HashMap::new(),
            file: CacheFile::new(path),
            separator: separator,
        };

        if let Some(mut reader) = this.file.lock() {
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer).unwrap();
            this.sync_content(&buffer);
        }

        this.file.unlock();

        this
    }

    /// Iterate over all values of the cache.
    pub fn for_each<F: FnMut(&K, &V)>(&mut self, mut func: F) {
        if let Some(mut reader) = self.file.lock() {
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer).unwrap();
            self.sync_content(&buffer);
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

    /// Insert a new item to the cache.
    ///
    /// Panic is an item with a different value exist for the cache.
    pub fn insert(&mut self, key: K, value: V) {
        if let Some(mut reader) = self.file.lock() {
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer).unwrap();
            self.sync_content(&buffer);
        }

        if let Some(existing) = self.in_memory_cache.get(&key) {
            if existing != &value {
                let entry = Entry { key, value };
                let entry = serde_json::to_string_pretty(&entry).unwrap();

                panic!(
                    r#"
Can't insert a duplicated entry in the cache file.
The cache might be corrupted, cleaning it might resolve the issue.

{entry}
"#
                );
            } else {
                self.file.unlock();
                return;
            }
        }

        self.insert_unchecked(key, value);

        self.file.unlock();
    }

    fn sync_content(&mut self, bytes: &[u8]) {
        let mut start = 0;

        while let Some(pos) = bytes[start..]
            .windows(self.separator.len())
            .position(|w| w == &self.separator)
        {
            let entry: Entry<K, V> = serde_json::from_slice(&bytes[start..start + pos]).unwrap();
            self.in_memory_cache.insert(entry.key, entry.value);
            start += pos + self.separator.len();
        }
    }

    fn insert_unchecked(&mut self, key: K, value: V) {
        let entry = Entry { key, value };
        let mut bytes = serde_json::to_vec(&entry).unwrap();

        for b in self.separator.iter() {
            bytes.push(*b);
        }

        self.file.write(&bytes);
        self.in_memory_cache.insert(entry.key, entry.value);
    }
}

impl<K, V> Display for Cache<K, V>
where
    K: Serialize + DeserializeOwned + Hash + PartialEq + Eq,
    V: Serialize + DeserializeOwned + PartialEq,
{
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
    root: String,
    version: String,
) -> PathBuf {
    let path_partial: &Path = path_partial.as_ref();
    let home_dir = dirs::home_dir().expect("An home directory should exist");
    let add_extension = !path_partial.ends_with("json");

    let mut path = home_dir.join(".cache").join(root).join(version);

    for segment in path_partial {
        // Skip the root directory since it resets the previous path segments.
        //
        // "/path/file" == "path/file" => "$HOME/.cache/tracel-ai/path/file"
        if segment == "/" {
            continue;
        }
        path = path.join(segment);
    }

    if add_extension {
        path.set_extension("json");
    }

    path
}

#[derive(Serialize, Deserialize)]
struct Entry<K, V> {
    key: K,
    value: V,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache() {
        let mut cache = Cache::<String, String>::new("test", CacheOption::default());
        cache.insert("key".to_string(), "value \n valval".to_string());
        cache.insert("key2".to_string(), "Value2".to_string());
    }
}
