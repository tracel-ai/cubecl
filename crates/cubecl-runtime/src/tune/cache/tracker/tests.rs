//! Tests of a tracker over a table in the process-wide memory storage, each table its own. Serial
//! with every test that switches the environment, which would reset the tables under them.

use super::*;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use cubecl_environment::persistence::{MemoryStorage, Namespace};

const KEY: &str = "matmul-64x64";
/// Sorts before [`CURRENT`], so a table read in key order meets it first.
const STALE: &str = "0000-older-list";
const CURRENT: &str = "ffff-current-list";

/// An answer stored under an older candidate list is never served for
/// the current one, even when it sorts first: the lookup asks for the key
/// and the list together.
#[test]
#[serial_test::serial]
fn an_answer_to_an_older_list_is_never_served() {
    let table = "tune-cache-tests/stale-first";
    plant(table, KEY, STALE, 1);

    assert!(matches!(
        tracker_over(table).track_stored(&KEY.to_string(), CURRENT, 3),
        TuneAnswer::Miss
    ));
}

#[test]
#[serial_test::serial]
fn an_answer_to_the_current_list_is_served() {
    let table = "tune-cache-tests/current";
    plant(table, KEY, STALE, 1);
    plant(table, KEY, CURRENT, 2);

    assert!(matches!(
        tracker_over(table).track_stored(&KEY.to_string(), CURRENT, 3),
        TuneAnswer::Hit { winner_index: 2 }
    ));
}

#[test]
#[serial_test::serial]
fn a_stored_index_past_the_set_is_a_miss() {
    let table = "tune-cache-tests/past-the-set";
    plant(table, KEY, CURRENT, 3);

    assert!(matches!(
        tracker_over(table).track_stored(&KEY.to_string(), CURRENT, 3),
        TuneAnswer::Miss
    ));
}

/// A row past the set is deleted, so the tune that follows can store its answer.
#[test]
#[serial_test::serial]
fn a_stored_index_past_the_set_is_deleted() {
    let table = "tune-cache-tests/past-the-set-deleted";
    plant(table, KEY, CURRENT, 3);
    tracker_over(table).track_stored(&KEY.to_string(), CURRENT, 3);

    assert_eq!(stored(table), []);
}

#[test]
#[serial_test::serial]
fn a_second_answer_to_a_stored_key_is_not_stored() {
    let table = "tune-cache-tests/second-answer";
    let mut tracker = tracker_over(table);
    tracker.store_answer(KEY.to_string(), CURRENT.into(), answer_at(1));

    assert!(!tracker.store_answer(KEY.to_string(), CURRENT.into(), answer_at(2)));
}

#[test]
#[serial_test::serial]
fn a_switched_environment_misses_every_answer() {
    let mut tracker = tracker_over("tune-cache-tests/switched");
    tracker.track_answer(KEY.to_string(), 1);
    // Far from any generation a concurrent switch of the environment reaches.
    tracker.table.generation = tracker.table.generation.wrapping_add(u32::MAX / 2);

    assert!(matches!(
        tracker.decided(&KEY.to_string()),
        TuneAnswer::Miss
    ));
}

#[test]
#[serial_test::serial]
fn a_reset_drops_the_answers_of_the_switched_environment() {
    let mut tracker = tracker_over("tune-cache-tests/reset");
    tracker.track_answer(KEY.to_string(), 1);
    // Far from any generation a concurrent switch of the environment reaches.
    tracker.table.generation = tracker.table.generation.wrapping_add(u32::MAX / 2);

    tracker.reset_if_environment_switched();

    assert!(tracker.answers.is_empty());
}

/// Opening a table drops the tables an earlier layout wrote, and nothing else. The only test of
/// this binary that opens a table, since the drop happens once per process.
#[cfg(not(target_family = "wasm"))]
#[test]
#[serial_test::serial]
fn opening_a_table_drops_the_earlier_layouts_only() {
    use crate::config::RuntimeConfig;
    use cubecl_environment::bytes::Bytes;
    use cubecl_environment::persistence::{self, NamespaceRoot};

    let root = tempfile::tempdir().unwrap();
    let _ = crate::config::CubeClRuntimeConfig::get();
    cubecl_environment::environment::set_root(root.path());
    let scope = Namespace::scoped(NamespaceRoot::AUTOTUNE, "")
        .as_str()
        .to_string();
    let at = |path: &str| alloc::format!("{scope}{path}");
    let stale = at("layout-v1/cuda_l4/matmul");
    let mut kept = alloc::vec![
        at("layout-v2/cuda_l4/conv"),
        "throughput/0.11.0/probe-v2/cuda_l4".to_string(),
    ];
    for namespace in core::iter::once(&stale).chain(&kept) {
        persistence::open(namespace).insert(
            b"key",
            Bytes::from_bytes_vec(alloc::vec![1]),
            persistence::Origin::Local,
        );
    }

    TuneTracker::<String>::open(TuneTableName {
        hardware: "cuda_l4",
        tuner: "matmul",
        refuses_register_spills: false,
    });

    let mut left = persistence::namespaces();
    left.retain(|namespace| *namespace == stale || kept.contains(namespace));
    left.sort();
    kept.sort();
    assert_eq!(left, kept);
}

#[test]
#[serial_test::serial]
fn a_loaded_answer_is_recorded_as_used() {
    let table = "tune-cache-tests/used-loaded";
    plant(table, KEY, CURRENT, 1);

    tracker_over(table).track_stored(&KEY.to_string(), CURRENT, 3);

    assert_eq!(used(table), [pair(KEY, CURRENT)]);
}

#[test]
#[serial_test::serial]
fn a_missed_answer_is_not_recorded_as_used() {
    let table = "tune-cache-tests/used-missed";

    tracker_over(table).track_stored(&KEY.to_string(), CURRENT, 3);

    assert_eq!(used(table), []);
}

/// The work runs another candidate than one past the set, so that row is no answer it used.
#[test]
#[serial_test::serial]
fn an_answer_past_the_set_is_not_recorded_as_used() {
    let table = "tune-cache-tests/used-past-the-set";
    plant(table, KEY, CURRENT, 3);

    tracker_over(table).track_stored(&KEY.to_string(), CURRENT, 3);

    assert_eq!(used(table), []);
}

#[test]
#[serial_test::serial]
fn a_stored_answer_is_recorded_as_used() {
    let table = "tune-cache-tests/used-stored";

    tracker_over(table).store_answer(KEY.to_string(), CURRENT.into(), answer_at(1));

    assert_eq!(used(table), [pair(KEY, CURRENT)]);
}

/// The table keeps the answer it held, and that answer is the one the next process loads.
#[test]
#[serial_test::serial]
fn the_answer_a_table_kept_is_recorded_as_used() {
    let table = "tune-cache-tests/used-kept";
    plant(table, KEY, CURRENT, 1);

    tracker_over(table).store_answer(KEY.to_string(), CURRENT.into(), answer_at(2));

    assert_eq!(used(table), [pair(KEY, CURRENT)]);
}

/// Storing an answer to the current list leaves the key's answer to an
/// older list in the table: an environment never loses an answer, and the
/// older one only costs a row, since no lookup for the current list reads
/// it.
#[test]
#[serial_test::serial]
fn a_stored_answer_leaves_the_answers_to_older_lists() {
    let table = "tune-cache-tests/kept";
    plant(table, KEY, STALE, 1);
    let mut tracker = tracker_over(table);
    tracker.track_stored(&KEY.to_string(), CURRENT, 3);

    assert!(tracker.store_answer(KEY.to_string(), CURRENT.into(), answer_at(2)));

    assert_eq!(stored(table), [pair(KEY, STALE), pair(KEY, CURRENT)]);
}

/// A tracker over the in-memory table `table`.
fn tracker_over(table: &str) -> TuneTracker<String> {
    TuneTracker::with_store(
        Some(PersistedTable {
            tunes: answers(table),
            uses: WorkUses::from_store(uses(table)),
        }),
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

/// The in-memory record of the entries of `table` the work used.
fn uses(table: &str) -> Store<PersistentCacheKey<String>, ()> {
    let namespace = alloc::format!("used-by-tests/{table}");
    Store::new(
        StoreOptions::new()
            .storage_with(
                Box::new(MemoryStorage::new(&namespace)),
                Namespace::new(&namespace),
            )
            .cache(CacheOption::Lazy),
    )
}

/// Every entry of `table` the work was recorded using, by key and list.
fn used(table: &str) -> Vec<PersistentCacheKey<String>> {
    let mut used = Vec::new();
    uses(table).scan(|key, ()| used.push(key));
    used.sort_by(|a, b| (&a.key, &a.checksum).cmp(&(&b.key, &b.checksum)));
    used
}

/// Stores in `table`, as a tune of `checksum`'s list would, `winner_index`
/// as the answer to `key`.
fn plant(table: &str, key: &str, checksum: &str, winner_index: usize) {
    answers(table)
        .insert(pair(key, checksum), answer_at(winner_index))
        .unwrap();
}

fn pair(key: &str, checksum: &str) -> PersistentCacheKey<String> {
    PersistentCacheKey {
        key: key.to_string(),
        checksum: checksum.into(),
    }
}

fn answer_at(winner_index: usize) -> PersistentCacheValue {
    PersistentCacheValue {
        winner_index,
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
