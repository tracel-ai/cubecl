//! A write the storage refuses costs the next run a recompute, never this one.
//!
//! A cache root the process can read but not write — a read-only mount, or a
//! database another process holds for itself — refuses every write. The value
//! was computed all the same, so the store keeps it in memory. Forgetting it
//! made every lookup miss, and a caller that measures on a miss, as the
//! throughput probes do, measured again on each.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cubecl_environment::bytes::Bytes;
use cubecl_environment::persistence::{
    CacheOption, Insertion, Origin, Storage, Store, StoreError, StoreOptions,
};

/// A storage that refuses every write and counts them. It holds nothing.
#[derive(Debug, Clone, Default)]
struct Refusing(Arc<AtomicUsize>);

impl Refusing {
    fn writes(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }

    fn refuse(&self) -> Insertion {
        self.0.fetch_add(1, Ordering::Relaxed);
        Insertion::Failed(String::from("attempt to write a readonly database"))
    }
}

impl Storage for Refusing {
    fn get(&self, _key: &[u8]) -> Option<Bytes> {
        None
    }

    fn insert(&self, _key: &[u8], _value: Bytes, _origin: Origin) -> Insertion {
        self.refuse()
    }

    fn replace(&self, _key: &[u8], _value: Bytes, _origin: Origin) -> Insertion {
        self.refuse()
    }

    fn scan(&self, _visit: &mut dyn FnMut(&[u8], &[u8])) {}

    fn purge(&self) {}

    fn purge_key(&self, _key: &[u8]) {}

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

/// The refusal is reported, and the value is served from then on.
#[test]
fn a_refused_value_is_served_for_the_rest_of_the_run() {
    let storage = Refusing::default();
    let mut store = store(&storage, CacheOption::Eager);

    assert!(matches!(
        store.insert("key".to_string(), 1),
        Err(StoreError::Backend { .. })
    ));
    assert_eq!(store.get(&"key".to_string()), Some(&1));
}

/// Inserting the value again is the in-memory no-op it is for any value the
/// store holds: the storage isn't asked, and refuses nothing, a second time.
#[test]
fn a_refused_value_is_not_written_again() {
    let storage = Refusing::default();
    let mut store = store(&storage, CacheOption::Eager);
    let _ = store.insert("key".to_string(), 1);

    for _ in 0..100 {
        store.insert("key".to_string(), 1).unwrap();
    }
    assert_eq!(storage.writes(), 1);
}

/// A refused write never replaces a value the store already holds: nothing
/// arbitrated it, so the first value stays, as a stored one would.
#[test]
fn a_refused_write_replaces_nothing() {
    let storage = Refusing::default();
    let mut store = store(&storage, CacheOption::Eager);
    let _ = store.insert("key".to_string(), 1);

    assert!(store.insert("key".to_string(), 2).is_err());
    assert_eq!(store.get(&"key".to_string()), Some(&1));
}

/// A lazy store drops what it writes only because the storage can hand it
/// back, so a value the storage refused stays in memory there too. The CUDA
/// and HIP backends store a compiled kernel and take it straight back out
/// with `remove`, expecting it to be there.
#[test]
fn a_lazy_store_reads_a_refused_value_back() {
    let storage = Refusing::default();
    let mut store = store(&storage, CacheOption::Lazy);

    assert!(matches!(
        store.insert("key".to_string(), 1),
        Err(StoreError::Backend { .. })
    ));
    assert_eq!(store.remove(&"key".to_string()), Some(1));
}
