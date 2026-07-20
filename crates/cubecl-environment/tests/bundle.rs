#![cfg(feature = "cache")]

use std::sync::Arc;

use cubecl_environment::bundle::{Bundle, ExportOptions, SeedSource, export};
use cubecl_environment::persistence::{KvStore, KvStoreOptions, Origin, SeedMode};

/// Warms `root` with one store's worth of entries, as an application run would.
fn warm(root: &std::path::Path, name: &str, path: &str, entries: &[(&str, u32)]) {
    let option = KvStoreOptions::default()
        .root(root)
        .name(name)
        .seeds(SeedMode::Disabled);
    let mut store = KvStore::<String, u32>::open(path, option);

    for (key, value) in entries {
        store.insert(key.to_string(), *value).unwrap();
    }
}

fn open_with_bundle(
    root: &std::path::Path,
    name: &str,
    path: &str,
    seeds: Vec<Arc<dyn SeedSource>>,
) -> KvStore<String, u32> {
    let option = KvStoreOptions::default()
        .root(root)
        .name(name)
        .seeds(SeedMode::Explicit(seeds));

    KvStore::open(path, option)
}

#[test]
fn bundle_roundtrip_seeds_a_fresh_store() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("test.cubecl");

    warm(
        warm_root.path(),
        "autotune",
        "device0/matmul",
        &[("shape=2x2", 3), ("shape=4x4", 7)],
    );

    // Export the warm cache root into a bundle.
    let options = ExportOptions {
        name: "Test GPU Linux".to_string(),
        ..Default::default()
    };
    let manifest = export(&[warm_root.path()], &bundle_path, &options).unwrap();
    assert_eq!(manifest.name, "Test GPU Linux");
    assert_eq!(manifest.environments.len(), 1);

    // Open the bundle on a "fresh machine" (empty cache root).
    let bundle = Bundle::open(&bundle_path).unwrap();
    let seeds: Vec<Arc<dyn SeedSource>> = vec![Arc::new(bundle)];

    let mut store = open_with_bundle(
        cold_root.path(),
        "autotune",
        "device0/matmul",
        seeds.clone(),
    );

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
    let store = open_with_bundle(cold_root.path(), "autotune", "device0/matmul", seeds);
    assert_eq!(
        store.get_with_origin(&"shape=4x4".to_string()),
        Some((&9, Origin::Local))
    );
}

/// Merging cache roots dedupes on the primary key. The previous file-copy
/// exporter concatenated colliding files, so the same key could land in a
/// bundle twice with two different values.
#[test]
fn exporting_several_roots_dedupes_shared_keys() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("merged.cubecl");

    warm(
        first.path(),
        "autotune",
        "device0/matmul",
        &[("shared", 1), ("only-first", 10)],
    );
    // The same key with a different value, as two machines would produce.
    warm(
        second.path(),
        "autotune",
        "device0/matmul",
        &[("shared", 2), ("only-second", 20)],
    );

    let options = ExportOptions {
        name: "Merged".to_string(),
        ..Default::default()
    };
    export(&[first.path(), second.path()], &bundle_path, &options).unwrap();

    let bundle = Bundle::open(&bundle_path).unwrap();
    let seeds: Vec<Arc<dyn SeedSource>> = vec![Arc::new(bundle)];
    let store = open_with_bundle(cold_root.path(), "autotune", "device0/matmul", seeds);

    assert_eq!(store.len(), 3, "the shared key must appear exactly once");
    // The first root wins on collision.
    assert_eq!(store.get(&"shared".to_string()), Some(&1));
    assert_eq!(store.get(&"only-first".to_string()), Some(&10));
    assert_eq!(store.get(&"only-second".to_string()), Some(&20));
}

#[test]
fn exporting_can_be_restricted_to_some_stores() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("autotune-only.cubecl");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]);
    warm(warm_root.path(), "throughput", "device0/copy", &[("k", 2)]);

    let options = ExportOptions {
        name: "Autotune only".to_string(),
        stores: vec!["autotune".to_string()],
        ..Default::default()
    };
    export(&[warm_root.path()], &bundle_path, &options).unwrap();

    let bundle = Arc::new(Bundle::open(&bundle_path).unwrap());
    let seeds: Vec<Arc<dyn SeedSource>> = vec![bundle];

    let tuned = open_with_bundle(
        cold_root.path(),
        "autotune",
        "device0/matmul",
        seeds.clone(),
    );
    assert_eq!(tuned.get(&"k".to_string()), Some(&1));

    let throughput = open_with_bundle(cold_root.path(), "throughput", "device0/copy", seeds);
    assert_eq!(
        throughput.get(&"k".to_string()),
        None,
        "the throughput store was not exported"
    );
}

/// Re-exporting must replace the bundle, never merge into the stale one.
#[test]
fn exporting_over_an_existing_bundle_replaces_it() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("replaced.cubecl");

    warm(
        warm_root.path(),
        "autotune",
        "device0/matmul",
        &[("old", 1)],
    );
    let options = ExportOptions {
        name: "First".to_string(),
        ..Default::default()
    };
    export(&[warm_root.path()], &bundle_path, &options).unwrap();

    let second_root = tempfile::tempdir().unwrap();
    warm(
        second_root.path(),
        "autotune",
        "device0/matmul",
        &[("new", 2)],
    );
    let options = ExportOptions {
        name: "Second".to_string(),
        ..Default::default()
    };
    let manifest = export(&[second_root.path()], &bundle_path, &options).unwrap();
    assert_eq!(manifest.name, "Second");

    let bundle = Bundle::open(&bundle_path).unwrap();
    let seeds: Vec<Arc<dyn SeedSource>> = vec![Arc::new(bundle)];
    let store = open_with_bundle(cold_root.path(), "autotune", "device0/matmul", seeds);

    assert_eq!(store.get(&"new".to_string()), Some(&2));
    assert_eq!(store.get(&"old".to_string()), None);
}

/// An unrelated file must never be silently destroyed by an export.
#[test]
fn exporting_over_a_foreign_file_fails() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("precious.txt");
    std::fs::write(&target, b"do not delete me").unwrap();

    let options = ExportOptions {
        name: "Nope".to_string(),
        ..Default::default()
    };
    assert!(export(&[dir.path()], &target, &options).is_err());
    assert_eq!(std::fs::read(&target).unwrap(), b"do not delete me");
}

#[test]
fn bad_bundle_is_skipped_without_failing() {
    let missing = std::path::Path::new("/nonexistent/bundle/path");
    assert!(Bundle::open(missing).is_err());

    // The registry path must swallow the error.
    cubecl_environment::bundle::install_from_paths(&[missing]);
    assert!(cubecl_environment::bundle::seeds().is_empty());
}

/// A file that is not a cubecl database must be rejected, not read as an
/// empty bundle.
#[test]
fn a_foreign_file_is_not_a_bundle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("not-a-bundle.cubecl");
    std::fs::write(&path, b"definitely not sqlite").unwrap();

    assert!(Bundle::open(&path).is_err());
}
