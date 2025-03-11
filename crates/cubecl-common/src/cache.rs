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
pub struct Cache<K, V> {
    in_memory_cache: HashMap<K, V>,
    file: CacheFile,
    separator: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct Entry<K, V> {
    key: K,
    value: V,
}

impl<K, V> Cache<K, V>
where
    K: Serialize + DeserializeOwned + Hash + PartialEq + Eq,
    V: Serialize + DeserializeOwned + PartialEq,
{
    /// Create a new cache and load the data from the provided path if it exist.
    pub fn new<P: AsRef<Path>>(path: P, separator: Option<&[u8]>) -> Self {
        let path = get_persistent_cache_file_path(path);

        let mut this = Self {
            in_memory_cache: HashMap::new(),
            file: CacheFile::new(path),
            separator: separator.unwrap_or(b"\x1F").to_vec(),
        };

        if let Some(mut reader) = this.file.lock() {
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer).unwrap();
            this.sync_content(&buffer);
        }

        this.file.unlock();

        this
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
                // Add cache versionning so that upgrading burn/cubecl doesn't cause this error.
                // We should know when it's safe to reset the cache, but it should be done at
                // initialization.
                panic!(
                    r#"
Can't insert a duplicated entry in the cache file.
The cache might be corrupted, cleaning it might resolve the issue.
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
            let entry: Entry<K, V> = rmp_serde::from_slice(&bytes[start..start + pos]).unwrap();
            self.in_memory_cache.insert(entry.key, entry.value);
            start += pos + self.separator.len();
        }
    }

    fn insert_unchecked(&mut self, key: K, value: V) {
        let entry = Entry { key, value };
        let mut bytes = rmp_serde::to_vec(&entry).unwrap();

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

fn get_persistent_cache_file_path<P: AsRef<Path>>(path_partial: P) -> PathBuf {
    let path_partial: &Path = path_partial.as_ref();
    let home_dir = dirs::home_dir().expect("An home directory should exist");
    let mut path = home_dir.join(".cache").join("tracel-ai");

    for segment in path_partial {
        // Skip the root directory since it resets the previous path segments.
        //
        // "/path/file" == "path/file" => "$HOME/.cache/tracel-ai/path/file"
        if segment == "/" {
            continue;
        }
        path = path.join(segment);
    }

    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache() {
        let mut cache = Cache::<String, String>::new("/tmp/allo", Some("--".as_bytes()));
        cache.insert("Allo".to_string(), "Toyo \n Yessir".to_string());
        cache.insert("Alice".to_string(), "AAhh".to_string());
    }
}
