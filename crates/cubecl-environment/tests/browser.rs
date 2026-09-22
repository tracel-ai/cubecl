#![cfg(all(target_family = "wasm", feature = "persistence"))]
//! The persistence layer in a browser: the database opens over OPFS from a
//! dedicated worker — the one place a synchronous access handle exists — and
//! every store after that is synchronous, exactly as it is natively.
//!
//! Run with `cargo test -p cubecl-environment --features persistence
//! --target wasm32-unknown-unknown --test browser`, with
//! `wasm-bindgen-test-runner` as the target's runner and `CHROMEDRIVER` set.

use cubecl_environment::bytes::Bytes;
use cubecl_environment::environment;
use cubecl_environment::persistence::{CacheOption, Namespace, Store, StoreOptions};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_dedicated_worker);

fn eager(namespace: &str) -> StoreOptions {
    StoreOptions::new().storage(Namespace::scoped("browser", namespace))
}

fn lazy(namespace: &str) -> StoreOptions {
    eager(namespace).cache(CacheOption::Lazy)
}

/// Every test shares the worker; whichever runs first opens the database,
/// the others find it in the registry.
async fn open() {
    environment::open().await;
}

/// The store is on the database, not on the memory a failed open falls back
/// to — which would pass every test below just the same, from this process's
/// own memory.
fn assert_durable<K, V>(store: &Store<K, V>)
where
    K: cubecl_environment::persistence::StoreKey,
    V: cubecl_environment::persistence::StoreValue,
{
    let description = store.to_string();
    assert!(
        description.contains("Turso cubecl-"),
        "the store is not on the browser's database: {description}"
    );
}

/// The whole point: once the database is open, a store is synchronous. An
/// entry written by one store is read by a store opened after it, from the
/// file, with nothing awaited in between.
#[wasm_bindgen_test]
async fn entries_survive_a_reopen_of_the_store() {
    open().await;

    let mut store = Store::<String, u32>::new(eager("reopen"));
    assert_durable(&store);
    store.insert("key".to_string(), 7).unwrap();
    drop(store);

    let store = Store::<String, u32>::new(eager("reopen"));
    assert_durable(&store);
    assert_eq!(store.get(&"key".to_string()), Some(&7));
}

/// The compile cache's shape: a lazy store never keeps the value, and reads
/// it back through the storage on demand.
#[wasm_bindgen_test]
async fn a_lazy_store_reads_through() {
    open().await;

    let mut store = Store::<String, Bytes>::new(lazy("kernels"));
    assert_durable(&store);
    store
        .insert("kernel".to_string(), Bytes::from_bytes_vec(vec![1, 2, 3]))
        .unwrap();
    assert!(store.is_empty(), "a lazy insert retains nothing");

    let mut reopened = Store::<String, Bytes>::new(lazy("kernels"));
    assert_eq!(
        reopened
            .remove(&"kernel".to_string())
            .map(|bytes| bytes.to_vec()),
        Some(vec![1, 2, 3])
    );
}

/// What the environment reports about itself is read from the file too.
#[wasm_bindgen_test]
async fn the_namespaces_are_listed() {
    open().await;

    let mut store = Store::<String, u32>::new(eager("listed"));
    assert_durable(&store);
    store.insert("key".to_string(), 1).unwrap();

    let namespaces: Vec<String> = environment::namespaces()
        .into_iter()
        .map(|summary| summary.namespace)
        .collect();
    assert!(
        namespaces
            .iter()
            .any(|namespace| namespace.ends_with("/listed")),
        "{namespaces:?}"
    );
}
