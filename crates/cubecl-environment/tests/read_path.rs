//! Guards the property the design depends on: once a store is open, reads are
//! pure memory. If a lookup ever reaches the storage it takes a mutex and
//! possibly the disk, which on the autotune and kernel-launch paths would be
//! felt immediately.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use cubecl_environment::bytes::Bytes;
use cubecl_environment::collections::HashMap;
use cubecl_environment::persistence::{Insertion, Origin, Storage, Store, StoreOptions};

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

#[async_trait::async_trait]
impl Storage for Counting {
    async fn get(&self, key: &[u8]) -> Option<Bytes> {
        self.0.gets.fetch_add(1, Ordering::Relaxed);
        self.0.entries.lock().unwrap().get(key).cloned()
    }

    async fn insert(&self, key: &[u8], value: Bytes, _origin: Origin) -> Insertion {
        self.0.inserts.fetch_add(1, Ordering::Relaxed);
        let mut entries = self.0.entries.lock().unwrap();

        if let Some(existing) = entries.get(key) {
            return Insertion::Conflict(existing.clone());
        }
        entries.insert(key.to_vec(), value);

        Insertion::Stored
    }

    async fn replace(&self, key: &[u8], value: Bytes, _origin: Origin) -> Insertion {
        self.0.inserts.fetch_add(1, Ordering::Relaxed);
        self.0.entries.lock().unwrap().insert(key.to_vec(), value);

        Insertion::Stored
    }

    async fn scan(&self) -> Vec<(Bytes, Bytes)> {
        self.0.scans.fetch_add(1, Ordering::Relaxed);
        self.0
            .entries
            .lock()
            .unwrap()
            .iter()
            .map(|(key, value)| (Bytes::from_bytes_vec(key.clone()), value.clone()))
            .collect()
    }

    async fn purge(&self) {
        self.0.entries.lock().unwrap().clear();
    }

    async fn purge_key(&self, key: &[u8]) {
        self.0.entries.lock().unwrap().remove(key);
    }

    fn describe(&self) -> String {
        String::from("counting")
    }
}

#[tokio::test]
async fn reads_never_reach_the_storage() {
    let storage = Counting::default();

    // Warm it the way an application would.
    let mut store = Store::<String, u32>::open(
        StoreOptions::new().storage_with(Box::new(storage.clone()), "bench/ns"),
    )
    .await;
    for index in 0..1_000u32 {
        store.insert(format!("key{index}"), index).await.unwrap();
    }

    let (gets, inserts, _) = storage.counts();
    assert_eq!(gets, 0, "writing must not read back through the storage");
    assert_eq!(inserts, 1_000);

    // Reopen: exactly one scan ingests everything, and nothing else.
    let mut store = Store::<String, u32>::open(
        StoreOptions::new().storage_with(Box::new(storage.clone()), "bench/ns"),
    )
    .await;
    let (_, _, scans_after_open) = storage.counts();

    // Now hammer the read path.
    for _ in 0..100 {
        for index in 0..1_000u32 {
            assert_eq!(store.get(&format!("key{index}")).await, Some(&index));
        }
    }
    // Misses too: a miss must not fall through to the storage either.
    for index in 1_000..2_000u32 {
        assert_eq!(store.get(&format!("key{index}")).await, None);
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
#[tokio::test]
async fn reinserting_a_known_value_stays_in_memory() {
    let storage = Counting::default();
    let mut store = Store::<String, u32>::open(
        StoreOptions::new().storage_with(Box::new(storage.clone()), "bench/ns"),
    )
    .await;

    store.insert("key".to_string(), 1).await.unwrap();
    let (_, inserts, _) = storage.counts();

    for _ in 0..1_000 {
        store.insert("key".to_string(), 1).await.unwrap();
    }

    assert_eq!(
        storage.counts().1,
        inserts,
        "re-inserting an identical value must not write again"
    );
}
