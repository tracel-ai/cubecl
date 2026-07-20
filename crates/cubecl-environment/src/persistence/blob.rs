use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;

use hashbrown::HashMap;

use super::storage::Storage;
use super::store::{
    KvStoreOptions, StoreError, StoreKey, StoreValue, Written, decode, encode, write_through,
};

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
///
/// It works over whichever [`Storage`] the target has. On an asynchronous one
/// (browser storage) a lookup can miss until the background load finishes,
/// which costs a recompile and nothing else.
#[derive(Debug)]
pub struct BlobStore<K, V> {
    in_memory_cache: InMemoryCache<K, V>,
    /// Values a later write replaced. [`Self::get_ref_unsafe`] may have handed
    /// out a reference into one of them, so they are retired here rather than
    /// dropped: the boxes stay put and the references stay valid.
    superseded: RefCell<Vec<Box<V>>>,
    storage: Box<dyn Storage>,
    namespace: String,
}

impl<K: StoreKey, V: StoreValue> BlobStore<K, V> {
    /// Create a new store addressing `path` under the options' cache root.
    #[cfg_attr(feature="tracing", tracing::instrument(
        level = "trace",
        skip(path),
        fields(path = ?path.as_ref())))]
    pub fn new<P: AsRef<str>>(path: P, option: KvStoreOptions) -> Self {
        let mut option = option;
        let namespace = option.resolve_namespace(path.as_ref());
        let storage = super::storage::open(&namespace);

        Self {
            in_memory_cache: InMemoryCache::default(),
            superseded: RefCell::default(),
            storage,
            namespace,
        }
    }

    /// Create a new store on an explicit storage, addressing `namespace`.
    pub fn with_storage<S: Into<String>>(storage: Box<dyn Storage>, namespace: S) -> Self {
        Self {
            in_memory_cache: InMemoryCache::default(),
            superseded: RefCell::default(),
            storage,
            namespace: namespace.into(),
        }
    }

    /// The namespace this store addresses.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// The storage this store caches.
    pub fn storage(&self) -> &dyn Storage {
        self.storage.as_ref()
    }

    /// Fetch an item from the memo alone, without falling through to the
    /// storage. Used by [`KvStore`](super::KvStore), whose memo is complete
    /// after its initial load, so a miss there is genuinely a miss and must
    /// not cost a query.
    pub(crate) fn get_memoized(&self, key: &K) -> Option<&V> {
        self.get_ref_unsafe(key)
    }

    /// How many entries are memoized.
    pub(crate) fn memoized_len(&self) -> usize {
        self.in_memory_cache.borrow().len()
    }

    /// Visits every memoized entry.
    pub(crate) fn for_each_memoized<F: FnMut(&K, &V)>(&self, mut func: F) {
        for (key, value) in self.in_memory_cache.borrow().iter() {
            func(key, value);
        }
    }

    /// Fetch an item from the store, reading it from the storage on the first
    /// lookup and memoizing it afterwards.
    pub fn get(&self, key: &K) -> Option<&V> {
        if let Some(value) = self.get_ref_unsafe(key) {
            return Some(value);
        }

        let bytes = self.storage.get(&encode(key))?;
        let value = decode::<V>(&bytes)?;
        self.memoize(key.clone(), value);

        self.get_ref_unsafe(key)
    }

    /// Records `value` in the memo, retiring anything it replaces.
    pub(crate) fn memoize(&self, key: K, value: V) {
        if let Some(previous) = self
            .in_memory_cache
            .borrow_mut()
            .insert(key, Box::new(value))
        {
            self.superseded.borrow_mut().push(previous);
        }
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
    /// Insert-only, with the one exception the storage arbitrates: a locally
    /// computed value replaces one that came from a bundle, so a stale
    /// imported kernel can never wedge compilation. Any other collision
    /// returns [`StoreError::DuplicatedKey`] and leaves the stored value
    /// alone.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), StoreError<K, V>> {
        if let Some(existing) = self.get_memoized(&key)
            && existing == &value
        {
            return Ok(());
        }

        // Only the memo is consulted above: `write_through` asks the storage
        // atomically, so reading it first would cost a second round trip and
        // still not know whether the existing entry is imported.
        match write_through(self.storage.as_ref(), &key, &value) {
            Written::Stored => {
                self.memoize(key, value);
                Ok(())
            }
            Written::Conflict(existing) => {
                self.memoize(key.clone(), existing.clone());

                Err(StoreError::DuplicatedKey {
                    key,
                    value_previous: existing,
                    value_updated: value,
                })
            }
        }
    }
}

// The tests reopen the store, which only persists with a real storage.
#[cfg(all(test, feature = "cache"))]
mod tests {
    use std::vec;

    use super::*;
    use crate::bytes::Bytes;
    use std::string::ToString;

    #[test_log::test]
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn values_survive_reopen_and_load_lazily() {
        let dir = tempfile::tempdir().unwrap();
        crate::environment::set_root(dir.path());
        let options = || KvStoreOptions::default().name("kernels");

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
    #[serial_test::serial]
    #[cfg_attr(miri, ignore)]
    fn reinserting_a_different_value_errors() {
        let dir = tempfile::tempdir().unwrap();
        crate::environment::set_root(dir.path());
        let option = KvStoreOptions::default().name("kernels");

        let mut cache = BlobStore::<String, Bytes>::new("ptx_sm90", option);
        let kernel = |byte: u8| Bytes::from_bytes_vec(std::vec![byte]);
        cache.insert("kernel".to_string(), kernel(1)).unwrap();

        assert!(cache.insert("kernel".to_string(), kernel(1)).is_ok());
        assert!(cache.insert("kernel".to_string(), kernel(2)).is_err());
    }
}
