use std::{
    boxed::Box,
    cell::RefCell,
    format,
    fs::{self, File},
    io::{Cursor, Write},
    path::{Path, PathBuf},
    string::{String, ToString},
    vec::Vec,
};

use ciborium::{de::from_reader, ser::into_writer};
use hashbrown::HashMap;
use serde::Serialize;

use super::store::{
    Cache, CacheError, CacheKey, CacheOption, CacheValue, Entry, Origin, sanitize_path_segment,
};

/// The in-memory cache used by the chunked kernel cache.
/// Box ensures values aren't moved when inserting new elements, so we don't need to keep it
/// locked for reads
type InMemoryCache<K, V> = RefCell<HashMap<K, Box<V>>>;

/// A chunked cache for compilation artifacts. Uses a human readable table of contents, with binary
/// storage for the compiled kernel.
#[derive(Debug)]
pub struct CompilationCache<K: CacheKey, V: CacheValue> {
    toc: Cache<K, String>,
    in_memory_cache: InMemoryCache<K, V>,
    current_chunk: File,
    current_chunk_path_normalized: String,
    cache_root: PathBuf,
    /// The cache root relative to any root directory (`name/version/segments`,
    /// `/`-separated): the base for bundle chunk lookups.
    rel_root: String,
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

impl<K: CacheKey, V: CacheValue> CompilationCache<K, V> {
    /// Create a new cache and load the data from the provided path if it exists.
    #[cfg_attr(feature="tracing", tracing::instrument(
        level = "trace",
        skip(path),
        fields(path = ?path.as_ref())))]
    pub fn new<P: AsRef<str>>(path: P, option: CacheOption) -> Self {
        let (_, name, version, root, _) = option.clone().resolve();
        let path = path.as_ref();
        let toc_path = format!("{path}/toc");
        // `.cbor` suffix (was `.bin` with bincode) invalidates old on-disk chunks.
        let chunk_path = Path::new("chunk0.cbor"); // Split later

        let rel_root = relative_cache_root(Path::new(path), &name, &version);
        let cache_root =
            std::path::absolute(root.join(&rel_root)).expect("Not empty, so can't fail");
        let chunk_path = get_persistent_chunk_file_path(chunk_path, &cache_root);

        let in_memory_cache = InMemoryCache::default();
        let toc = Cache::open_file(toc_path, option);

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
            rel_root: normalized_path(&rel_root),
        }
    }

    /// Fetch an item from the cache.
    pub fn get(&self, key: &K) -> Option<&V> {
        if let Some(value) = self.get_ref_unsafe(key) {
            return Some(value);
        }
        let (chunk, origin) = self.toc.get_with_origin(key)?;
        match origin {
            Origin::Local => {
                Self::read_chunk(&self.cache_root.join(chunk), &self.in_memory_cache);
            }
            Origin::Bundle(index) => {
                let seed = self.toc.seed(index)?;
                let rel = format!("{}/{chunk}", self.rel_root);
                match seed.chunk_bytes(&rel) {
                    Some(bytes) => {
                        Self::read_chunk_bytes(bytes.into_owned(), &self.in_memory_cache, &rel);
                    }
                    None => {
                        // A truncated bundle: recompile instead of panicking.
                        log::warn!(
                            "Bundle chunk '{rel}' referenced by the table of contents is \
                             missing in {}; recompiling.",
                            seed.describe()
                        );
                        return None;
                    }
                }
            }
        }
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
        Self::read_chunk_bytes(data, cache, &chunk.to_string_lossy());
    }

    fn read_chunk_bytes(data: Vec<u8>, cache: &InMemoryCache<K, V>, chunk: &str) {
        let mut cursor = Cursor::new(data);
        // Collect new entries first so we only need to lock once everything is loaded
        let mut new_entries = Vec::new();
        let mut idx = 0;
        loop {
            let pos = cursor.position() as usize;
            let total_len = cursor.get_ref().len();
            if pos >= total_len {
                break;
            }
            match from_reader::<Entry<K, V>, _>(&mut cursor) {
                Ok(entry) => {
                    new_entries.push((entry.key, Box::new(entry.value)));
                }
                Err(err) => {
                    let pos_after = cursor.position() as usize;
                    if pos_after == pos {
                        if pos < total_len {
                            log::warn!(
                                "Corrupted cache file {chunk:?}, stopping at entry {idx} : {err}",
                            );
                        }
                        break;
                    }
                    log::warn!("Corrupted cache file {chunk:?}, ignoring entry {idx} : {err}",);
                }
            }
            idx += 1;
        }

        // Never replace an existing entry: `get_ref_unsafe` hands out
        // references whose validity depends on boxes never being dropped, so
        // a key present in both a local chunk and a bundle chunk must keep
        // its first-loaded value.
        let mut cache = cache.borrow_mut();
        for (key, value) in new_entries {
            cache.entry(key).or_insert(value);
        }
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
            let mut bytes = Vec::new();
            into_writer(&entry, &mut bytes).expect("Can serialize data");
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

/// The cache root relative to any root directory: `name/version/segments`.
fn relative_cache_root(path_partial: &Path, name: &str, version: &str) -> PathBuf {
    let mut path = PathBuf::from(sanitize_path_segment(name)).join(sanitize_path_segment(version));

    for segment in path_partial.iter() {
        // Skip the name directory since it resets the previous path segments.
        if segment == "/" {
            continue;
        }
        path = path.join(sanitize_path_segment(segment.to_str().unwrap()));
    }

    path
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
