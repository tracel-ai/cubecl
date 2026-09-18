#![cfg(all(feature = "persistence", not(target_family = "wasm")))]
//! What several writers on one cache root must not do to each other.
//!
//! A Turso `Connection` is `Sync` but refuses concurrent use at runtime with
//! `Misuse("concurrent use forbidden")`, so the storage opens one per
//! operation rather than sharing a long-lived one. These tests fail if that
//! ever changes: they run enough work in parallel that a shared connection
//! would be used twice at once.

use cubecl_environment::persistence::{Namespace, Store, StoreOptions};

/// Entries per task. Enough to keep several operations in flight at once.
const ENTRIES: u32 = 32;
/// Tasks writing at the same time, over more than one worker thread.
const TASKS: u32 = 16;

fn key(task: u32, entry: u32) -> String {
    alloc::format!("task{task}-key{entry}")
}

/// Every writer's entries survive, and none of them faults the engine: the
/// whole point of a connection per operation.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[serial_test::serial]
#[cfg_attr(miri, ignore)]
async fn concurrent_writers_do_not_lose_entries() {
    let root = tempfile::tempdir().unwrap();
    cubecl_environment::environment::set_root(root.path());

    let mut tasks = Vec::new();
    for task in 0..TASKS {
        tasks.push(tokio::spawn(async move {
            let mut store: Store<String, u32> =
                Store::open(StoreOptions::new().storage(Namespace::scoped("probe", "v1"))).await;

            for entry in 0..ENTRIES {
                store.insert(key(task, entry), entry).await.unwrap();
            }
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }

    // Read back through a store that opens the root cold, so the entries come
    // from the file rather than from a writer's own memory.
    let store: Store<String, u32> =
        Store::open(StoreOptions::new().storage(Namespace::scoped("probe", "v1"))).await;

    for task in 0..TASKS {
        for entry in 0..ENTRIES {
            let key = key(task, entry);
            assert_eq!(store.get(&key).copied(), Some(entry), "{key}");
        }
    }
}

/// Readers and a writer on one root at the same time: a read never fails and
/// never blocks the writer out, which is what WAL buys.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[serial_test::serial]
#[cfg_attr(miri, ignore)]
async fn readers_run_alongside_a_writer() {
    let root = tempfile::tempdir().unwrap();
    cubecl_environment::environment::set_root(root.path());

    {
        let mut store: Store<String, u32> =
            Store::open(StoreOptions::new().storage(Namespace::scoped("probe", "v1"))).await;
        store.insert(key(0, 0), 7).await.unwrap();
    }

    let mut tasks = Vec::new();
    for task in 1..TASKS {
        tasks.push(tokio::spawn(async move {
            let mut store: Store<String, u32> =
                Store::open(StoreOptions::new().storage(Namespace::scoped("probe", "v1"))).await;

            // The entry every task reads was written before any of them
            // started, so it is there however the writes interleave.
            assert_eq!(store.get(&key(0, 0)).copied(), Some(7));

            for entry in 0..ENTRIES {
                store.insert(key(task, entry), entry).await.unwrap();
                assert_eq!(store.get(&key(task, entry)).copied(), Some(entry));
            }
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
}

extern crate alloc;
