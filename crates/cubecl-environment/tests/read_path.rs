//! Guards the property the design depends on: once a store is open, reads are
//! pure memory. If a lookup ever reaches the storage it takes a mutex and
//! possibly the disk, which on the autotune and kernel-launch paths would be
//! felt immediately.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use cubecl_environment::bytes::Bytes;
use cubecl_environment::collections::HashMap;
use cubecl_environment::persistence::{KvStore, Origin, Storage};

/// Shared so the test can read the counters while the store owns the storage.
#[derive(Debug, Clone, Default)]
struct Counting(std::sync::Arc<CountingStorage>);

/// A storage that records how often it is reached.
#[derive(Debug, Default)]
struct CountingStorage {
    entries: Mutex<HashMap<Vec<u8>, Bytes>>,
    gets: AtomicUsize,
    inserts: AtomicUsize,
    scans: AtomicUsize,
}

impl Counting {
    /// `(gets, inserts, scans)`
    fn counts(&self) -> (usize, usize, usize) {
        (
            self.0.gets.load(Ordering::Relaxed),
            self.0.inserts.load(Ordering::Relaxed),
            self.0.scans.load(Ordering::Relaxed),
        )
    }
}

impl Storage for Counting {
    fn get(&self, key: &[u8]) -> Option<Bytes> {
        self.0.gets.fetch_add(1, Ordering::Relaxed);
        self.0.entries.lock().unwrap().get(key).cloned()
    }

    fn insert(&self, key: &[u8], value: &[u8], _origin: Origin) -> Option<Bytes> {
        self.0.inserts.fetch_add(1, Ordering::Relaxed);
        let mut entries = self.0.entries.lock().unwrap();

        if let Some(existing) = entries.get(key) {
            return Some(existing.clone());
        }
        entries.insert(key.to_vec(), Bytes::from_bytes_vec(value.to_vec()));

        None
    }

    fn scan(&self, visit: &mut dyn FnMut(&[u8], &[u8])) {
        self.0.scans.fetch_add(1, Ordering::Relaxed);
        for (key, value) in self.0.entries.lock().unwrap().iter() {
            visit(key, value);
        }
    }

    fn describe(&self) -> String {
        String::from("counting")
    }
}

#[test]
fn reads_never_reach_the_storage() {
    let storage = Counting::default();

    // Warm it the way an application would.
    let mut store = KvStore::<String, u32>::with_storage(Box::new(storage.clone()), "bench/ns");
    for index in 0..1_000u32 {
        store.insert(format!("key{index}"), index).unwrap();
    }

    let (gets, inserts, _) = storage.counts();
    assert_eq!(gets, 0, "writing must not read back through the storage");
    assert_eq!(inserts, 1_000);

    // Reopen: exactly one scan ingests everything, and nothing else.
    let mut store = KvStore::<String, u32>::with_storage(Box::new(storage.clone()), "bench/ns");
    store.sync();
    let (_, _, scans_after_open) = storage.counts();

    // Now hammer the read path.
    for _ in 0..100 {
        for index in 0..1_000u32 {
            assert_eq!(store.get(&format!("key{index}")), Some(&index));
        }
    }
    // Misses too: a miss must not fall through to the storage either.
    for index in 1_000..2_000u32 {
        assert_eq!(store.get(&format!("key{index}")), None);
    }

    let (gets, _, scans) = storage.counts();
    assert_eq!(
        gets, 0,
        "100_000 hits and 1_000 misses reached the storage {gets} times"
    );
    assert_eq!(
        scans, scans_after_open,
        "reads must not trigger another scan"
    );

    // And iteration stays in memory once the load is ingested.
    let mut seen = 0;
    store.for_each(|_, _| seen += 1);
    assert_eq!(seen, 1_000);
    assert_eq!(storage.counts().2, scans_after_open, "for_each rescanned");
}

/// Re-inserting a value the store already holds must not reach the storage
/// either: autotune does this on every duplicate tune.
#[test]
fn reinserting_a_known_value_stays_in_memory() {
    let storage = Counting::default();
    let mut store = KvStore::<String, u32>::with_storage(Box::new(storage.clone()), "bench/ns");

    store.insert("key".to_string(), 1).unwrap();
    let (_, inserts, _) = storage.counts();

    for _ in 0..1_000 {
        store.insert("key".to_string(), 1).unwrap();
    }

    assert_eq!(
        storage.counts().1,
        inserts,
        "re-inserting an identical value must not write again"
    );
}
