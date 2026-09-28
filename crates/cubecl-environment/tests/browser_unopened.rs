#![cfg(all(target_family = "wasm", feature = "persistence"))]
//! A browser store opened before `environment::open()` was awaited: its own
//! module, so that no other test has opened the database for it.

use cubecl_environment::persistence::{Namespace, Store, StoreOptions};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_dedicated_worker);

/// The database can't be opened synchronously in the browser, and a store
/// that needs it before it is open must neither panic nor hang: it serves
/// memory and says so.
#[wasm_bindgen_test]
fn a_store_before_the_open_serves_memory() {
    let mut store = Store::<String, u32>::new(
        StoreOptions::new().storage(Namespace::scoped("browser", "early")),
    );

    let description = store.to_string();
    assert!(description.contains("memory ("), "{description}");

    store.insert("key".to_string(), 1).unwrap();
    assert_eq!(store.get(&"key".to_string()), Some(&1));
}
