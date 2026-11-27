//! Tests of a tracker over a table in the process-wide memory storage, each table its own.

use super::*;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use cubecl_environment::persistence::MemoryStorage;

const KEY: &str = "matmul-64x64";
/// Sorts before [`CURRENT`], so a table read in key order meets it first.
const STALE: &str = "0000-older-list";
const CURRENT: &str = "ffff-current-list";

/// An answer stored under an older candidate list is never served for
/// the current one, even when it sorts first: the lookup asks for the key
/// and the list together, and misses until the current list is answered.
#[test]
fn an_answer_to_an_older_list_is_never_served() {
    let table = "tune-cache-tests/stale-first";
    plant(table, KEY, STALE, 1);
    let mut cache = cache_over(table);

    assert!(matches!(
        cache.track_stored(&KEY.to_string(), CURRENT, 3),
        TuneCacheResult::Miss
    ));

    plant(table, KEY, CURRENT, 2);
    let mut cache = cache_over(table);
    assert!(matches!(
        cache.track_stored(&KEY.to_string(), CURRENT, 3),
        TuneCacheResult::Hit { fastest_index: 2 }
    ));
}

/// Storing an answer to the current list leaves the key's answer to an
/// older list in the table: an environment never loses an answer, and the
/// older one only costs a row, since no lookup for the current list reads
/// it.
#[test]
fn a_stored_answer_leaves_the_answers_to_older_lists() {
    let table = "tune-cache-tests/kept";
    plant(table, KEY, STALE, 1);
    let mut cache = cache_over(table);
    cache.track_stored(&KEY.to_string(), CURRENT, 3);

    assert!(cache.store_answer(KEY.to_string(), CURRENT.into(), answer(2)));

    assert_eq!(stored(table), [pair(KEY, STALE), pair(KEY, CURRENT)]);
}

/// A tracker over the in-memory table `table`.
fn cache_over(table: &str) -> TuneTracker<String> {
    TuneTracker::with_store(
        Some(answers(table)),
        TuneTable {
            namespace: table.into(),
            generation: cubecl_environment::environment::generation(),
        },
    )
}

/// The store of the in-memory table `table`, read lazily as a tuner's is.
fn answers(table: &str) -> Store<PersistentCacheKey<String>, PersistentCacheValue> {
    Store::new(
        StoreOptions::new()
            .storage_with(Box::new(MemoryStorage::new(table)), Namespace::new(table))
            .cache(CacheOption::Lazy),
    )
}

/// Stores in `table`, as a tune of `checksum`'s list would, `fastest_index`
/// as the answer to `key`.
fn plant(table: &str, key: &str, checksum: &str, fastest_index: usize) {
    answers(table)
        .insert(pair(key, checksum), answer(fastest_index))
        .unwrap();
}

fn pair(key: &str, checksum: &str) -> PersistentCacheKey<String> {
    PersistentCacheKey {
        key: key.to_string(),
        checksum: checksum.into(),
    }
}

fn answer(fastest_index: usize) -> PersistentCacheValue {
    PersistentCacheValue {
        decided_index: fastest_index,
        results: Vec::new(),
        bounds: None,
        limit: None,
    }
}

/// Every answer `table` holds, by key and list.
fn stored(table: &str) -> Vec<PersistentCacheKey<String>> {
    let mut stored = Vec::new();
    answers(table).scan(|key, _| stored.push(key));
    stored.sort_by(|a, b| (&a.key, &a.checksum).cmp(&(&b.key, &b.checksum)));
    stored
}
