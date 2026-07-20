#![cfg(feature = "cache")]

use std::sync::Arc;

use cubecl_environment::bundle::{
    Bundle, BundleFormat, EmbeddedBundle, ExportOptions, SqliteBundle, export,
};
use cubecl_environment::bytes::Bytes;
use cubecl_environment::persistence::{BundleMode, KvStore, KvStoreOptions, Origin};

/// Warms `root` with one store's worth of entries, as an application run would.
fn warm(root: &std::path::Path, name: &str, path: &str, entries: &[(&str, u32)]) {
    let option = KvStoreOptions::default()
        .root(root)
        .name(name)
        .bundles(BundleMode::Disabled);
    let mut store = KvStore::<String, u32>::open(path, option);

    for (key, value) in entries {
        store.insert(key.to_string(), *value).unwrap();
    }
}

fn open_with_bundle(
    root: &std::path::Path,
    name: &str,
    path: &str,
    seeds: Vec<Arc<dyn Bundle>>,
) -> KvStore<String, u32> {
    let option = KvStoreOptions::default()
        .root(root)
        .name(name)
        .bundles(BundleMode::Explicit(seeds));

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
    let bundle = SqliteBundle::open(&bundle_path).unwrap();
    let seeds: Vec<Arc<dyn Bundle>> = vec![Arc::new(bundle)];

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

    let bundle = SqliteBundle::open(&bundle_path).unwrap();
    let seeds: Vec<Arc<dyn Bundle>> = vec![Arc::new(bundle)];
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
        namespaces: vec!["autotune".to_string()],
        ..Default::default()
    };
    export(&[warm_root.path()], &bundle_path, &options).unwrap();

    let bundle = Arc::new(SqliteBundle::open(&bundle_path).unwrap());
    let seeds: Vec<Arc<dyn Bundle>> = vec![bundle];

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

    let bundle = SqliteBundle::open(&bundle_path).unwrap();
    let seeds: Vec<Arc<dyn Bundle>> = vec![Arc::new(bundle)];
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
    assert!(SqliteBundle::open(missing).is_err());

    // The registry path must swallow the error.
    cubecl_environment::bundle::install_from_paths(&[missing]);
    assert!(cubecl_environment::bundle::installed().is_empty());
}

/// A file that is not a cubecl database must be rejected, not read as an
/// empty bundle.
#[test]
fn a_foreign_file_is_not_a_bundle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("not-a-bundle.cubecl");
    std::fs::write(&path, b"definitely not sqlite").unwrap();

    assert!(SqliteBundle::open(&path).is_err());
}

/// The flat format must seed a store exactly like the `SQLite` one. It is the
/// format wasm and no-std targets get, and they have no way to open a
/// database.
#[test]
fn a_flat_bundle_seeds_a_store_like_the_sqlite_one() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("flat.ccb");

    warm(
        warm_root.path(),
        "autotune",
        "device0/matmul",
        &[("shape=2x2", 3), ("shape=4x4", 7)],
    );
    warm(warm_root.path(), "throughput", "device0/copy", &[("k", 11)]);

    let options = ExportOptions {
        name: "Flat GPU".to_string(),
        format: BundleFormat::Flat,
        ..Default::default()
    };
    let manifest = export(&[warm_root.path()], &bundle_path, &options).unwrap();
    assert_eq!(manifest.name, "Flat GPU");

    // Read it back the way an embedded target would: raw bytes, no file system
    // access beyond loading the blob.
    let bytes = Bytes::from_bytes_vec(std::fs::read(&bundle_path).unwrap());
    let bundle = EmbeddedBundle::open(bytes).unwrap();
    assert_eq!(bundle.len(), 3);

    let cold_root = tempfile::tempdir().unwrap();
    let seeds: Vec<Arc<dyn Bundle>> = vec![Arc::new(bundle)];

    let mut store = open_with_bundle(
        cold_root.path(),
        "autotune",
        "device0/matmul",
        seeds.clone(),
    );
    assert_eq!(store.get(&"shape=2x2".to_string()), Some(&3));
    assert_eq!(
        store.get_with_origin(&"shape=4x4".to_string()),
        Some((&7, Origin::Bundle(0)))
    );

    // Shadowing works the same as with a database-backed bundle.
    store.insert("shape=4x4".to_string(), 9).unwrap();
    assert_eq!(
        store.get_with_origin(&"shape=4x4".to_string()),
        Some((&9, Origin::Local))
    );

    // A second namespace in the same bundle stays reachable and separate.
    let throughput = open_with_bundle(cold_root.path(), "throughput", "device0/copy", seeds);
    assert_eq!(throughput.get(&"k".to_string()), Some(&11));
    assert_eq!(throughput.len(), 1);
}

/// Both formats must be byte-for-byte equivalent in what they serve.
#[test]
fn both_formats_serve_the_same_entries() {
    let warm_root = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();

    warm(
        warm_root.path(),
        "autotune",
        "device0/matmul",
        &[("a", 1), ("b", 2), ("c", 3)],
    );

    let sqlite_path = dir.path().join("bundle.cubecl");
    export(
        &[warm_root.path()],
        &sqlite_path,
        &ExportOptions {
            name: "Same".to_string(),
            ..Default::default()
        },
    )
    .unwrap();

    let flat_path = dir.path().join("bundle.ccb");
    export(
        &[warm_root.path()],
        &flat_path,
        &ExportOptions {
            name: "Same".to_string(),
            format: BundleFormat::Flat,
            ..Default::default()
        },
    )
    .unwrap();

    let namespace = format!("autotune/{}/device0/matmul", env!("CARGO_PKG_VERSION"));
    let sqlite = SqliteBundle::open(&sqlite_path).unwrap();
    let flat =
        EmbeddedBundle::open(Bytes::from_bytes_vec(std::fs::read(&flat_path).unwrap())).unwrap();

    let collect = |bundle: &dyn Bundle| {
        let mut entries = Vec::new();
        bundle.scan(&namespace, &mut |key, value| {
            entries.push((key.to_vec(), value.to_vec()));
        });
        entries.sort();
        entries
    };

    let from_sqlite = collect(&sqlite);
    assert_eq!(from_sqlite.len(), 3);
    assert_eq!(from_sqlite, collect(&flat));

    // Point lookups must agree too, including on misses.
    for (key, _) in &from_sqlite {
        assert_eq!(
            flat.get(&namespace, key).map(|v| v.to_vec()),
            sqlite.get(&namespace, key).map(|v| v.to_vec())
        );
    }
    assert_eq!(flat.get(&namespace, b"missing"), None);
    assert_eq!(flat.get("no/such/namespace", b"a"), None);
}

/// The namespace filter must apply to the flat writer as well.
#[test]
fn a_flat_bundle_honours_the_namespace_filter() {
    let warm_root = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let bundle_path = dir.path().join("filtered.ccb");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]);
    warm(warm_root.path(), "throughput", "device0/copy", &[("k", 2)]);

    export(
        &[warm_root.path()],
        &bundle_path,
        &ExportOptions {
            name: "Autotune only".to_string(),
            namespaces: vec!["autotune".to_string()],
            format: BundleFormat::Flat,
            ..Default::default()
        },
    )
    .unwrap();

    let bundle =
        EmbeddedBundle::open(Bytes::from_bytes_vec(std::fs::read(&bundle_path).unwrap())).unwrap();
    assert_eq!(bundle.len(), 1);
    assert_eq!(
        bundle
            .summary()
            .iter()
            .map(|n| n.namespace.as_str())
            .collect::<Vec<_>>(),
        vec![format!(
            "autotune/{}/device0/matmul",
            env!("CARGO_PKG_VERSION")
        )]
    );
}
