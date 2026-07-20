use std::{boxed::Box, cell::RefCell, string::String, vec::Vec};

use hashbrown::HashMap;
use serde::Serialize;

use super::backend::KvBackend;
use super::store::{CacheKey, CacheOption, CacheValue, decode, encode, resolve_seeds};
use crate::bundle::SeedSource;
use crate::sync::Arc;

/// Values already read from the backend.
///
/// Box ensures values aren't moved when inserting new elements, so we don't need to keep it
/// locked for reads
type InMemoryCache<K, V> = RefCell<HashMap<K, Box<V>>>;

/// A cache for large values, typically compilation artifacts.
///
/// Unlike [`KvStore`](super::KvStore), entries are read from the backend one
/// key at a time and memoized, so opening a cache holding hundreds of compiled
/// kernels costs nothing until they are actually looked up.
#[derive(Debug)]
pub struct CompilationCache<K: CacheKey, V: CacheValue> {
    in_memory_cache: InMemoryCache<K, V>,
    backend: Box<dyn KvBackend>,
    store: String,
    seeds: Vec<Arc<dyn SeedSource>>,
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
}

impl<K: CacheKey, V: CacheValue> CompilationCache<K, V> {
    /// Create a new cache addressing `path` under the options' cache root.
    #[cfg_attr(feature="tracing", tracing::instrument(
        level = "trace",
        skip(path),
        fields(path = ?path.as_ref())))]
    pub fn new<P: AsRef<str>>(path: P, option: CacheOption) -> Self {
        let mut option = option;
        let seeds = resolve_seeds(option.take_seeds());
        let root = option.resolve_root();
        let store = option.resolve_store(path.as_ref());
        let backend = super::open_backend(&root, &store);

        Self {
            in_memory_cache: InMemoryCache::default(),
            backend,
            store,
            seeds,
        }
    }

    /// The logical store this cache addresses.
    pub fn store(&self) -> &str {
        &self.store
    }

    /// Fetch an item from the cache, reading it from the backend or from an
    /// installed bundle on the first lookup.
    pub fn get(&self, key: &K) -> Option<&V> {
        if let Some(value) = self.get_ref_unsafe(key) {
            return Some(value);
        }

        let key_bytes = encode(key);
        // Local entries win over bundled ones: a value produced on this
        // machine is always preferred to a shipped one.
        let bytes = self.backend.get(&key_bytes).or_else(|| {
            self.seeds
                .iter()
                .rev()
                .find_map(|seed| seed.get(&self.store, &key_bytes))
        })?;

        let value = decode::<V>(&bytes)?;
        self.in_memory_cache
            .borrow_mut()
            .entry(key.clone())
            .or_insert_with(|| Box::new(value));

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

    /// Insert a new item to the cache.
    ///
    /// Returns an error when a different value is already stored under `key`,
    /// whether it was inserted by this process or concurrently by another one.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), CompilationCacheError<K, V>> {
        if let Some(existing) = self.get(&key) {
            return if existing != &value {
                Err(CompilationCacheError::DuplicatedKey {
                    key: key.clone(),
                    value_previous: existing.clone(),
                    value_updated: value,
                })
            } else {
                Ok(())
            };
        }

        if let Some(existing) = self.backend.insert(&encode(&key), &encode(&value))
            && let Some(existing) = decode::<V>(&existing)
            && existing != value
        {
            // Another process won the race with a different value. It is the
            // durable one, so memoize it rather than ours. `or_insert` and
            // never `insert`: `get_ref_unsafe` hands out references whose
            // validity depends on boxes never being dropped.
            self.in_memory_cache
                .borrow_mut()
                .entry(key.clone())
                .or_insert_with(|| Box::new(existing.clone()));

            return Err(CompilationCacheError::DuplicatedKey {
                key,
                value_previous: existing,
                value_updated: value,
            });
        }

        self.in_memory_cache
            .borrow_mut()
            .entry(key)
            .or_insert_with(|| Box::new(value));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::store::SeedMode;
    use super::*;
    use std::string::ToString;

    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn values_survive_reopen_and_load_lazily() {
        let dir = tempfile::tempdir().unwrap();
        let options = || {
            CacheOption::default()
                .root(dir.path())
                .name("kernels")
                .seeds(SeedMode::Disabled)
        };

        let mut cache = CompilationCache::<String, Vec<u8>>::new("ptx_sm90", options());
        cache
            .insert("kernel_a".to_string(), std::vec![1, 2, 3])
            .unwrap();
        cache
            .insert("kernel_b".to_string(), std::vec![4, 5])
            .unwrap();
        drop(cache);

        let cache = CompilationCache::<String, Vec<u8>>::new("ptx_sm90", options());
        // Nothing is read until a key is asked for.
        assert!(cache.in_memory_cache.borrow().is_empty());

        assert_eq!(
            cache.get(&"kernel_a".to_string()),
            Some(&std::vec![1, 2, 3])
        );
        assert_eq!(cache.in_memory_cache.borrow().len(), 1);
        assert_eq!(cache.get(&"kernel_b".to_string()), Some(&std::vec![4, 5]));
        assert_eq!(cache.get(&"missing".to_string()), None);
    }

    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn reinserting_a_different_value_errors() {
        let dir = tempfile::tempdir().unwrap();
        let option = CacheOption::default()
            .root(dir.path())
            .name("kernels")
            .seeds(SeedMode::Disabled);

        let mut cache = CompilationCache::<String, Vec<u8>>::new("ptx_sm90", option);
        cache.insert("kernel".to_string(), std::vec![1]).unwrap();

        assert!(cache.insert("kernel".to_string(), std::vec![1]).is_ok());
        assert!(cache.insert("kernel".to_string(), std::vec![2]).is_err());
    }
}
