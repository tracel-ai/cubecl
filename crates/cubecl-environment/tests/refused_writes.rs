//! A write the storage refuses costs the next run a recompute, not every
//! lookup of this one.
//!
//! A cache root the process can read but not write, such as a read-only mount
//! or a database another process holds for itself, refuses every write. An
//! eager store is read from memory alone, so forgetting the value made every
//! lookup miss, and a caller that measures on a miss, as the throughput probes
//! do, measured again on each.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cubecl_environment::bytes::Bytes;
use cubecl_environment::collections::HashMap;
use cubecl_environment::persistence::{
    CacheOption, Insertion, Origin, Storage, Store, StoreError, StoreOptions,
};

/// A storage that refuses writes until told otherwise, and counts them.
#[derive(Debug, Clone, Default)]
struct Refusing(Arc<RefusingStorage>);

#[derive(Debug, Default)]
struct RefusingStorage {
    entries: Mutex<HashMap<Vec<u8>, Bytes>>,
    accepts: AtomicBool,
    writes: AtomicUsize,
}

impl Refusing {
    fn writes(&self) -> usize {
        self.0.writes.load(Ordering::Relaxed)
    }

    fn accept(&self) {
        self.0.accepts.store(true, Ordering::Relaxed);
    }

    fn durable(&self) -> usize {
        self.0.entries.lock().unwrap().len()
    }

    fn write(&self, key: &[u8], value: Bytes, replace: bool) -> Insertion {
        self.0.writes.fetch_add(1, Ordering::Relaxed);
        if !self.0.accepts.load(Ordering::Relaxed) {
            return Insertion::Failed(String::from("attempt to write a readonly database"));
        }

        let mut entries = self.0.entries.lock().unwrap();
        if !replace && let Some(existing) = entries.get(key) {
            return Insertion::Conflict(existing.clone());
        }
        entries.insert(key.to_vec(), value);

        Insertion::Stored
    }
}

impl Storage for Refusing {
    fn get(&self, key: &[u8]) -> Option<Bytes> {
        self.0.entries.lock().unwrap().get(key).cloned()
    }

    fn insert(&self, key: &[u8], value: Bytes, _origin: Origin) -> Insertion {
        self.write(key, value, false)
    }

    fn replace(&self, key: &[u8], value: Bytes, _origin: Origin) -> Insertion {
        self.write(key, value, true)
    }

    fn scan(&self, visit: &mut dyn FnMut(&[u8], &[u8])) {
        for (key, value) in self.0.entries.lock().unwrap().iter() {
            visit(key, value);
        }
    }

    fn purge(&self) {
        self.0.entries.lock().unwrap().clear();
    }

    fn purge_key(&self, key: &[u8]) {
        self.0.entries.lock().unwrap().remove(key);
    }

    fn describe(&self) -> String {
        String::from("refusing")
    }
}

fn store(storage: &Refusing, cache: CacheOption) -> Store<String, u32> {
    Store::new(
        StoreOptions::new()
            .storage_with(Box::new(storage.clone()), "refused/ns")
            .cache(cache),
    )
}

fn key() -> String {
    "key".to_string()
}

/// The refusal is reported, and an eager store serves the value from then on.
#[test]
fn an_eager_store_serves_a_refused_value() {
    let storage = Refusing::default();
    let mut store = store(&storage, CacheOption::Eager);

    assert!(matches!(
        store.insert(key(), 1),
        Err(StoreError::Backend { .. })
    ));
    assert_eq!(store.get(&key()), Some(&1));
}

/// Reinserting a refused value retries the write, and reports success only
/// once the storage takes it.
#[test]
fn a_refused_value_is_written_again_on_reinsert() {
    let storage = Refusing::default();
    let mut store = store(&storage, CacheOption::Eager);

    assert!(store.insert(key(), 1).is_err());
    assert!(matches!(
        store.insert(key(), 1),
        Err(StoreError::Backend { .. })
    ));
    assert_eq!(storage.writes(), 2);

    storage.accept();
    store.insert(key(), 1).unwrap();
    assert_eq!(storage.durable(), 1);

    // Durable now, so a reinsert is the usual in-memory no-op.
    store.insert(key(), 1).unwrap();
    assert_eq!(storage.writes(), 3);
}

/// A different value for a key memory holds is a duplicate, whether or not
/// the storage took the first one, and the storage isn't asked.
#[test]
fn a_different_value_after_a_refusal_is_a_duplicate() {
    let storage = Refusing::default();
    let mut store = store(&storage, CacheOption::Eager);

    assert!(store.insert(key(), 1).is_err());
    assert!(matches!(
        store.insert(key(), 2),
        Err(StoreError::DuplicatedKey { .. })
    ));
    assert_eq!(storage.writes(), 1);
    assert_eq!(store.get(&key()), Some(&1));
}

/// Removing a refused value forgets it: nothing durable backs the key, so a
/// later insert is a fresh one.
#[test]
fn a_removed_refused_value_leaves_the_key_fresh() {
    let storage = Refusing::default();
    let mut store = store(&storage, CacheOption::Eager);

    assert!(store.insert(key(), 1).is_err());
    assert_eq!(store.remove(&key()), Some(1));

    storage.accept();
    store.insert(key(), 2).unwrap();
    assert_eq!(store.get(&key()), Some(&2));
}

/// A lazy store drops what it writes, a refused value included: its callers
/// keep their own map in front of it, and retaining compiled artifacts here
/// would hold them twice.
#[test]
fn a_lazy_store_does_not_retain_a_refused_value() {
    let storage = Refusing::default();
    let mut store = store(&storage, CacheOption::Lazy);

    assert!(matches!(
        store.insert(key(), 1),
        Err(StoreError::Backend { .. })
    ));
    assert!(store.is_empty());
    assert_eq!(store.get_mut(&key()), None);
}
