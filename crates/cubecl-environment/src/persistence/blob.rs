use std::{boxed::Box, cell::RefCell, string::String, vec::Vec};

use hashbrown::HashMap;
use serde::Serialize;

use super::storage::Storage;
use super::store::{KvStoreOptions, StoreKey, StoreValue, decode, encode, resolve_bundles};
use crate::bundle::Bundle;
use crate::sync::Arc;

/// Values already read from the storage or a bundle.
///
/// Box ensures values aren't moved when inserting new elements, so we don't need to keep it
/// locked for reads
type InMemoryCache<K, V> = RefCell<HashMap<K, Box<V>>>;

/// A store for large values, typically compilation artifacts.
///
/// Unlike [`KvStore`](super::KvStore), entries are read one key at a time and
/// memoized, so opening a store holding hundreds of compiled kernels costs
/// nothing until they are actually looked up.
#[derive(Debug)]
pub struct BlobStore<K: StoreKey, V: StoreValue> {
    in_memory_cache: InMemoryCache<K, V>,
    storage: Box<dyn Storage>,
    namespace: String,
    bundles: Vec<Arc<dyn Bundle>>,
}

/// Error related to the store.
#[derive(Debug)]
pub enum BlobStoreError<K: Serialize, V: Serialize> {
    /// Can't insert an entry with the same key, but different value.
    #[allow(missing_docs)]
    DuplicatedKey {
        key: K,
        value_previous: V,
        value_updated: V,
    },
}

impl<K: StoreKey, V: StoreValue> BlobStore<K, V> {
    /// Create a new store addressing `path` under the options' cache root.
    #[cfg_attr(feature="tracing", tracing::instrument(
        level = "trace",
        skip(path),
        fields(path = ?path.as_ref())))]
    pub fn new<P: AsRef<str>>(path: P, option: KvStoreOptions) -> Self {
        let mut option = option;
        let bundles = resolve_bundles(option.take_bundles());
        let root = option.resolve_root();
        let namespace = option.resolve_namespace(path.as_ref());
        let storage = super::open_storage(&root, &namespace);

        Self {
            in_memory_cache: InMemoryCache::default(),
            storage,
            namespace,
            bundles,
        }
    }

    /// The namespace this store addresses.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Fetch an item from the store, reading it from the storage or from an
    /// installed bundle on the first lookup.
    pub fn get(&self, key: &K) -> Option<&V> {
        if let Some(value) = self.get_ref_unsafe(key) {
            return Some(value);
        }

        let key_bytes = encode(key);
        // Local entries win over bundled ones: a value produced on this
        // machine is always preferred to a shipped one. A bundle may serve the
        // bytes borrowed, which for a compiled kernel saves copying it whole.
        let bytes = match self.storage.get(&key_bytes) {
            Some(bytes) => bytes,
            None => self
                .bundles
                .iter()
                .rev()
                .find_map(|bundle| bundle.get(&self.namespace, &key_bytes))?,
        };

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

    /// Insert a new item to the store.
    ///
    /// Returns an error when a different value is already stored under `key`,
    /// whether it was inserted by this process or concurrently by another one.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), BlobStoreError<K, V>> {
        if let Some(existing) = self.get(&key) {
            return if existing != &value {
                Err(BlobStoreError::DuplicatedKey {
                    key: key.clone(),
                    value_previous: existing.clone(),
                    value_updated: value,
                })
            } else {
                Ok(())
            };
        }

        if let Some(existing) = self.storage.insert(&encode(&key), &encode(&value))
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

            return Err(BlobStoreError::DuplicatedKey {
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
    use super::super::store::BundleMode;
    use super::*;
    use crate::bytes::Bytes;
    use std::string::ToString;

    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn values_survive_reopen_and_load_lazily() {
        let dir = tempfile::tempdir().unwrap();
        let options = || {
            KvStoreOptions::default()
                .root(dir.path())
                .name("kernels")
                .bundles(BundleMode::Disabled)
        };

        let mut cache = BlobStore::<String, Bytes>::new("ptx_sm90", options());
        cache
            .insert(
                "kernel_a".to_string(),
                Bytes::from_bytes_vec(std::vec![1, 2, 3]),
            )
            .unwrap();
        cache
            .insert(
                "kernel_b".to_string(),
                Bytes::from_bytes_vec(std::vec![4, 5]),
            )
            .unwrap();
        drop(cache);

        let cache = BlobStore::<String, Bytes>::new("ptx_sm90", options());
        // Nothing is read until a key is asked for.
        assert!(cache.in_memory_cache.borrow().is_empty());

        assert_eq!(
            cache.get(&"kernel_a".to_string()).map(|v| v.to_vec()),
            Some(std::vec![1, 2, 3])
        );
        assert_eq!(cache.in_memory_cache.borrow().len(), 1);
        assert_eq!(
            cache.get(&"kernel_b".to_string()).map(|v| v.to_vec()),
            Some(std::vec![4, 5])
        );
        assert_eq!(cache.get(&"missing".to_string()), None);
    }

    #[test_log::test]
    #[cfg_attr(miri, ignore)]
    fn reinserting_a_different_value_errors() {
        let dir = tempfile::tempdir().unwrap();
        let option = KvStoreOptions::default()
            .root(dir.path())
            .name("kernels")
            .bundles(BundleMode::Disabled);

        let mut cache = BlobStore::<String, Bytes>::new("ptx_sm90", option);
        let kernel = |byte: u8| Bytes::from_bytes_vec(std::vec![byte]);
        cache.insert("kernel".to_string(), kernel(1)).unwrap();

        assert!(cache.insert("kernel".to_string(), kernel(1)).is_ok());
        assert!(cache.insert("kernel".to_string(), kernel(2)).is_err());
    }
}
