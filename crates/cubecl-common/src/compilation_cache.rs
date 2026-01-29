use std::{
    cell::RefCell,
    fs::{self, File},
    io::{BufRead, BufReader, Cursor, Write},
    path::{Path, PathBuf},
};

use bincode::{config::Configuration, de::read::SliceReader, error::DecodeError, serde::Compat};
use bytes::Buf;
use hashbrown::HashMap;
use serde::Serialize;

use crate::cache::{
    Cache, CacheError, CacheKey, CacheOption, CacheValue, Entry, sanitize_path_segment,
};

type InMemoryCache<K, V> = RefCell<HashMap<K, Box<V>>>;

#[derive(Debug)]
pub struct CompilationCache<K: CacheKey, V: CacheValue> {
    toc: Cache<K, String>,
    // Box ensures values aren't moved when inserting new elements, so we don't need to keep it
    // locked for reads
    in_memory_cache: InMemoryCache<K, V>,
    current_chunk: File,
    current_chunk_path_normalized: String,
    cache_root: PathBuf,
}

/// Error related to caching.
#[derive(Debug)]
pub enum CompilationCacheError<K: Serialize, V: Serialize> {
    /// Can't insert an entry with the same key, but different value.
    #[allow(missing_docs)]
    DuplicatedKey {
        key: K,
        value_previous: V,
        value_updated: V,
    },
    /// The table of contents cache had an error
    #[allow(missing_docs)]
    TocError(CacheError<K, String>),
}

const CONFIG: Configuration = bincode::config::standard();

impl<K: CacheKey, V: CacheValue> CompilationCache<K, V> {
    /// Create a new cache and load the data from the provided path if it exists.
    #[cfg_attr(feature="tracing", tracing::instrument(
        level = "trace",
        skip(path),
        fields(path = ?path.as_ref())))]
    pub fn new<P: AsRef<Path>>(path: P, option: CacheOption) -> Self {
        let (_, name, version, root, _) = option.clone().resolve();
        let path = path.as_ref();
        let toc_path = path.join("toc");
        let chunk_path = Path::new("chunk1.bin"); // Split later

        let cache_root = get_persistent_cache_root(path, root, name, version);
        let chunk_path = get_persistent_chunk_file_path(chunk_path, &cache_root);

        let in_memory_cache = InMemoryCache::default();
        let toc = Cache::new(toc_path, option);

        if fs::exists(&chunk_path).unwrap_or(false) {
            Self::read_chunk(&chunk_path, &in_memory_cache);
        }

        let current_chunk = open_chunk_writable(&chunk_path);

        Self {
            toc,
            in_memory_cache,
            current_chunk,
            current_chunk_path_normalized: normalized_path(
                chunk_path
                    .strip_prefix(&cache_root)
                    .expect("Should contain root"),
            ),
            cache_root,
        }
    }

    /// Fetch an item from the cache.
    pub fn get(&self, key: &K) -> Option<&V> {
        if let Some(value) = self.get_ref_unsafe(key) {
            return Some(value);
        }
        let chunk = self.toc.get(key)?;
        Self::read_chunk(&self.cache_root.join(chunk), &self.in_memory_cache);
        self.get_ref_unsafe(key)
    }

    /// Unsafely construct a reference of lifetime 'self, ignoring the read guard.
    /// # Safety
    ///
    /// This is only safe because we never remove or update elements, so boxes remain valid for the
    /// entire lifetime of `self`
    fn get_ref_unsafe<'a>(&'a self, key: &K) -> Option<&'a V> {
        let cache = self.in_memory_cache.borrow();
        if let Some(value) = cache.get(key) {
            let value = unsafe { core::mem::transmute::<&'_ V, &'a V>(&**value) };
            Some(value)
        } else {
            None
        }
    }

    fn read_chunk(chunk: &PathBuf, cache: &InMemoryCache<K, V>) {
        let data =
            fs::read(chunk).expect("Can't open chunk in table of contents, cache is corrupted!");
        let mut data_reader = SliceReader::new(&data);
        // Collect new entries first so we only need to lock once everything is loaded
        let mut new_entries = Vec::new();
        let mut idx = 0;
        loop {
            match bincode::decode_from_reader::<Compat<Entry<K, V>>, _, _>(&mut data_reader, CONFIG)
            {
                Ok(Compat(entry)) => {
                    new_entries.push((entry.key, Box::new(entry.value)));
                }
                Err(DecodeError::UnexpectedEnd { .. }) => {
                    break;
                }
                Err(err) => {
                    log::warn!("Corrupted cache file {chunk:?}, ignoring entry {idx} : {err}",);
                }
            }
            idx += 1;
        }

        cache.borrow_mut().extend(new_entries);
    }

    /// Insert a new item to the cache.
    ///
    /// Panic if an item with a different value exists in the cache.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), CompilationCacheError<K, V>> {
        if let Some(existing) = self.get_ref_unsafe(&key) {
            if existing != &value {
                return Err(CompilationCacheError::DuplicatedKey {
                    key,
                    value_previous: existing.clone(),
                    value_updated: value,
                });
            } else {
                return Ok(());
            }
        }

        // Insert
        {
            let entry = Entry {
                key: key.clone(),
                value,
            };
            let bytes = bincode::encode_to_vec(Compat(&entry), CONFIG).expect("Can serialize data");
            self.current_chunk
                .write_all(&bytes)
                .expect("Failed to write to chunk");

            let mut cache = self.in_memory_cache.borrow_mut();
            cache.insert(entry.key, Box::new(entry.value));
        }
        self.toc
            .insert(key, self.current_chunk_path_normalized.clone())
            .map_err(CompilationCacheError::TocError)?;

        Ok(())
    }
}

/// This exists in std but is not stabilized
fn has_data(reader: &mut impl BufRead) -> bool {
    match reader.fill_buf() {
        Ok(buf) => !buf.is_empty(),
        Err(_) => false,
    }
}

fn get_persistent_cache_root(
    path_partial: impl AsRef<Path>,
    root: PathBuf,
    name: String,
    version: String,
) -> PathBuf {
    let path_partial = path_partial.as_ref();
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

    std::path::absolute(path).expect("Not empty, so can't fail")
}

fn get_persistent_chunk_file_path<P: AsRef<Path>>(path_partial: P, chunks_root: &Path) -> PathBuf {
    let path_partial: &Path = path_partial.as_ref();

    let mut path = chunks_root.to_path_buf();

    for segment in path_partial.iter() {
        // Skip the name directory since it resets the previous path segments.
        if segment == "/" {
            continue;
        }
        path = path.join(sanitize_path_segment(segment.to_str().unwrap()));
    }

    std::path::absolute(path).expect("Not empty, so can't fail")
}

fn normalized_path(path: &Path) -> String {
    let path = path.to_string_lossy().to_string();
    path.replace("\\", "/")
}

fn open_chunk_writable(path: &Path) -> File {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("Failed to create parent");
    }
    let file = File::options().append(true).create(true).open(path);
    file.expect("Failed to open write chunk")
}
