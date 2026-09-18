#![cfg(all(feature = "persistence", not(target_family = "wasm")))]
//! What several writers on one cache root must not do to each other.
//!
//! A Turso `Connection` is `Sync` but refuses concurrent use at runtime with
//! `Misuse("concurrent use forbidden")`, so each storage holds its own behind
//! a mutex. These tests run enough work in parallel that a connection shared
//! without one would be used twice at once.

use cubecl_environment::persistence::{Namespace, Store, StoreOptions};

/// Entries per thread. Enough to keep several operations in flight at once.
const ENTRIES: u32 = 32;
/// Threads writing at the same time.
const THREADS: u32 = 16;

fn key(thread: u32, entry: u32) -> String {
    alloc::format!("thread{thread}-key{entry}")
}

fn open() -> Store<String, u32> {
    Store::new(StoreOptions::new().storage(Namespace::scoped("probe", "v1")))
}

/// Every writer's entries survive, and none of them faults the engine.
#[test]
#[serial_test::serial]
#[cfg_attr(miri, ignore)]
fn concurrent_writers_do_not_lose_entries() {
    let root = tempfile::tempdir().unwrap();
    cubecl_environment::environment::set_root(root.path());

    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            scope.spawn(move || {
                let mut store = open();
                for entry in 0..ENTRIES {
                    store.insert(key(thread, entry), entry).unwrap();
                }
            });
        }
    });

    // Read back through a store that opens the root cold, so the entries come
    // from the file rather than from a writer's own memory.
    let store = open();
    for thread in 0..THREADS {
        for entry in 0..ENTRIES {
            let key = key(thread, entry);
            assert_eq!(store.get(&key).copied(), Some(entry), "{key}");
        }
    }
}

/// Readers and a writer on one root at the same time: a read never fails and
/// never blocks the writer out, which is what WAL buys.
#[test]
#[serial_test::serial]
#[cfg_attr(miri, ignore)]
fn readers_run_alongside_a_writer() {
    let root = tempfile::tempdir().unwrap();
    cubecl_environment::environment::set_root(root.path());

    open().insert(key(0, 0), 7).unwrap();

    std::thread::scope(|scope| {
        for thread in 1..THREADS {
            scope.spawn(move || {
                let mut store = open();

                // The entry every thread reads was written before any of them
                // started, so it is there however the writes interleave.
                assert_eq!(store.get(&key(0, 0)).copied(), Some(7));

                for entry in 0..ENTRIES {
                    store.insert(key(thread, entry), entry).unwrap();
                    assert_eq!(store.get(&key(thread, entry)).copied(), Some(entry));
                }
            });
        }
    });
}

extern crate alloc;
