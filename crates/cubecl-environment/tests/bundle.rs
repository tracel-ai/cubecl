#![cfg(feature = "cache")]

use std::sync::Arc;

use cubecl_environment::bundle::{Bundle, ExportOptions, SeedSource, export};
use cubecl_environment::persistence::{KvStore, KvStoreOptions, Origin, SeedMode};

#[test]
fn bundle_roundtrip_seeds_a_fresh_store() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();

    // Warm up a store, as an application run would.
    {
        let option = KvStoreOptions::default()
            .root(warm_root.path())
            .name("autotune")
            .seeds(SeedMode::Disabled);
        let mut store = KvStore::<String, u32>::open("device0/matmul", option);
        store.insert("shape=2x2".to_string(), 3).unwrap();
        store.insert("shape=4x4".to_string(), 7).unwrap();
    }

    // Export the warm cache root into a bundle.
    let options = ExportOptions {
        name: "Test GPU Linux".to_string(),
        ..Default::default()
    };
    let manifest = export(&[warm_root.path()], bundle_dir.path(), &options).unwrap();
    assert_eq!(manifest.name, "Test GPU Linux");
    assert_eq!(manifest.environments.len(), 1);

    // Open the bundle on a "fresh machine" (empty cache root).
    let bundle = Bundle::open(bundle_dir.path()).unwrap();
    let seeds: Vec<Arc<dyn SeedSource>> = vec![Arc::new(bundle)];

    let option = KvStoreOptions::default()
        .root(cold_root.path())
        .name("autotune")
        .seeds(SeedMode::Explicit(seeds.clone()));
    let mut store = KvStore::<String, u32>::open("device0/matmul", option);

    // Bundle entries are visible, tagged with their origin.
    assert_eq!(store.get(&"shape=2x2".to_string()), Some(&3));
    assert_eq!(
        store.get_with_origin(&"shape=4x4".to_string()),
        Some((&7, Origin::Bundle(0)))
    );

    // Re-inserting the same value is fine and does not error.
    store.insert("shape=2x2".to_string(), 3).unwrap();

    // A locally computed different value shadows the bundle silently.
    store.insert("shape=4x4".to_string(), 9).unwrap();
    assert_eq!(
        store.get_with_origin(&"shape=4x4".to_string()),
        Some((&9, Origin::Local))
    );

    // The shadowed value must be durable: a fresh store over the same local
    // root and bundle must prefer the local value.
    let option = KvStoreOptions::default()
        .root(cold_root.path())
        .name("autotune")
        .seeds(SeedMode::Explicit(seeds));
    let store = KvStore::<String, u32>::open("device0/matmul", option);
    assert_eq!(
        store.get_with_origin(&"shape=4x4".to_string()),
        Some((&9, Origin::Local))
    );
}

#[test]
fn bad_bundle_is_skipped_without_failing() {
    let missing = std::path::Path::new("/nonexistent/bundle/path");
    assert!(Bundle::open(missing).is_err());

    // The registry path must swallow the error.
    cubecl_environment::bundle::install_from_paths(&[missing]);
    assert!(cubecl_environment::bundle::seeds().is_empty());
}
