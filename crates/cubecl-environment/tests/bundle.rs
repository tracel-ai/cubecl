#![cfg(feature = "cache")]

use cubecl_environment::bundle::{
    Bundle, BundleFormat, EmbeddedBundle, ExportOptions, SqliteBundle, export, import,
};
use cubecl_environment::bytes::Bytes;
use cubecl_environment::persistence::{Database, KvStore, KvStoreOptions};

// Storage resolves through the process-global active environment, so these
// tests are serialized: only one environment is active at a time by design.

/// Pins the environment to `root`, which is process-global; every test here
/// is serialized for that reason.
fn options(root: &std::path::Path, name: &str) -> KvStoreOptions {
    cubecl_environment::environment::set_root(root);
    KvStoreOptions::default().name(name)
}

/// Warms `root` with one namespace's worth of entries, as an application run
/// would.
fn warm(root: &std::path::Path, name: &str, path: &str, entries: &[(&str, u32)]) {
    let mut store = KvStore::<String, u32>::open(path, options(root, name));

    for (key, value) in entries {
        store.insert(key.to_string(), *value).unwrap();
    }
}

fn open(root: &std::path::Path, name: &str, path: &str) -> KvStore<String, u32> {
    KvStore::open(path, options(root, name))
}

fn export_to(root: &std::path::Path, out: &std::path::Path, format: BundleFormat) {
    let options = ExportOptions {
        name: "Test GPU Linux".to_string(),
        format,
        ..Default::default()
    };
    export(&[root], out, &options).unwrap();
}

/// Imports into `root`, which becomes the active environment first: `import`
/// fills whichever environment is active.
fn import_into(
    root: &std::path::Path,
    bundle: &dyn Bundle,
) -> cubecl_environment::bundle::ImportReport {
    cubecl_environment::environment::set_root(root);
    import(bundle)
}

fn open_bundle(path: &std::path::Path, format: BundleFormat) -> Box<dyn Bundle> {
    match format {
        BundleFormat::Sqlite => Box::new(SqliteBundle::open(path).unwrap()),
        BundleFormat::Flat => Box::new(
            EmbeddedBundle::open(Bytes::from_bytes_vec(std::fs::read(path).unwrap())).unwrap(),
        ),
    }
}

/// The core contract: importing fills the local storage, and afterwards the
/// bundle is irrelevant. Deleting it must change nothing.
#[test]
#[serial_test::serial]
fn importing_fills_the_storage_and_the_bundle_becomes_irrelevant() {
    for format in [BundleFormat::Sqlite, BundleFormat::Flat] {
        let warm_root = tempfile::tempdir().unwrap();
        let bundle_dir = tempfile::tempdir().unwrap();
        let cold_root = tempfile::tempdir().unwrap();
        let bundle_path = bundle_dir.path().join("test.bundle");

        warm(
            warm_root.path(),
            "autotune",
            "device0/matmul",
            &[("shape=2x2", 3), ("shape=4x4", 7)],
        );
        export_to(warm_root.path(), &bundle_path, format);

        let bundle = open_bundle(&bundle_path, format);
        let report = import_into(cold_root.path(), bundle.as_ref());
        assert_eq!(report.imported, 2, "{format:?}");
        assert_eq!(report.skipped, 0, "{format:?}");

        // The bundle is gone, yet the entries are there.
        drop(bundle);
        std::fs::remove_file(&bundle_path).unwrap();

        let store = open(cold_root.path(), "autotune", "device0/matmul");
        assert_eq!(store.get(&"shape=2x2".to_string()), Some(&3), "{format:?}");
        assert_eq!(store.get(&"shape=4x4".to_string()), Some(&7), "{format:?}");
    }
}

/// Importing twice must not duplicate or error.
#[test]
#[serial_test::serial]
fn importing_is_idempotent() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("test.bundle");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]);
    export_to(warm_root.path(), &bundle_path, BundleFormat::Sqlite);

    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);

    let first = import_into(cold_root.path(), bundle.as_ref());
    assert_eq!((first.imported, first.skipped), (1, 0));

    let second = import_into(cold_root.path(), bundle.as_ref());
    assert_eq!(
        (second.imported, second.skipped),
        (0, 1),
        "the second import must skip what is already stored"
    );

    let store = open(cold_root.path(), "autotune", "device0/matmul");
    assert_eq!(store.len(), 1);
}

/// A value computed on this machine must never be overwritten by a bundle.
#[test]
#[serial_test::serial]
fn importing_never_overwrites_a_local_value() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let local_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("test.bundle");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]);
    export_to(warm_root.path(), &bundle_path, BundleFormat::Sqlite);

    // This machine already computed a different answer for the same key.
    warm(
        local_root.path(),
        "autotune",
        "device0/matmul",
        &[("k", 42)],
    );

    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);
    let report = import_into(local_root.path(), bundle.as_ref());
    assert_eq!((report.imported, report.skipped), (0, 1));

    let store = open(local_root.path(), "autotune", "device0/matmul");
    assert_eq!(
        store.get(&"k".to_string()),
        Some(&42),
        "the local value wins"
    );
}

/// The mirror image: a stale imported entry must be replaceable by a locally
/// computed one, or a bad bundle would wedge the application forever.
#[test]
#[serial_test::serial]
fn a_local_value_replaces_a_stale_imported_one() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let local_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("test.bundle");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]);
    export_to(warm_root.path(), &bundle_path, BundleFormat::Sqlite);

    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);
    import_into(local_root.path(), bundle.as_ref());

    // The application disagrees with the shipped answer.
    let mut store = open(local_root.path(), "autotune", "device0/matmul");
    assert_eq!(store.get(&"k".to_string()), Some(&1));
    store.insert("k".to_string(), 99).unwrap();

    // It must stick, including across a reopen, and a re-import must not
    // resurrect the stale value.
    let report = import_into(local_root.path(), bundle.as_ref());
    assert_eq!(report.imported, 0);

    let store = open(local_root.path(), "autotune", "device0/matmul");
    assert_eq!(store.get(&"k".to_string()), Some(&99));

    // And now that it is local, a second disagreement is a plain conflict.
    let mut store = store;
    assert!(store.insert("k".to_string(), 7).is_err());
}

/// Import must cover every namespace the bundle holds.
#[test]
#[serial_test::serial]
fn importing_covers_every_namespace() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("test.bundle");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]);
    warm(warm_root.path(), "throughput", "device0/copy", &[("k", 2)]);
    export_to(warm_root.path(), &bundle_path, BundleFormat::Flat);

    let bundle = open_bundle(&bundle_path, BundleFormat::Flat);
    assert_eq!(bundle.namespaces().len(), 2);

    let report = import_into(cold_root.path(), bundle.as_ref());
    assert_eq!(report.imported, 2);
    assert_eq!(report.namespaces.len(), 2);

    assert_eq!(
        open(cold_root.path(), "autotune", "device0/matmul").get(&"k".to_string()),
        Some(&1)
    );
    assert_eq!(
        open(cold_root.path(), "throughput", "device0/copy").get(&"k".to_string()),
        Some(&2)
    );
}

/// A cache root must be able to report what it holds, which is what you
/// consult before bundling.
#[test]
#[serial_test::serial]
fn a_cache_root_lists_its_namespaces() {
    let root = tempfile::tempdir().unwrap();

    warm(root.path(), "autotune", "device0/matmul", &[("k", 1)]);
    warm(
        root.path(),
        "throughput",
        "device0/copy",
        &[("k", 2), ("j", 3)],
    );

    let database = Database::open_active().unwrap();
    let version = env!("CARGO_PKG_VERSION");

    assert_eq!(
        database.namespaces(),
        vec![
            format!("autotune/{version}/device0/matmul"),
            format!("throughput/{version}/device0/copy"),
        ]
    );

    let summary = database.summary();
    assert_eq!(summary[0].entries, 1);
    assert_eq!(summary[1].entries, 2);
}

/// Merging cache roots dedupes on the primary key. The original file-copy
/// exporter concatenated colliding files instead.
#[test]
#[serial_test::serial]
fn exporting_several_roots_dedupes_shared_keys() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("merged.bundle");

    warm(
        first.path(),
        "autotune",
        "device0/matmul",
        &[("shared", 1), ("only-first", 10)],
    );
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

    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);
    let report = import_into(cold_root.path(), bundle.as_ref());
    assert_eq!(report.imported, 3, "the shared key appears exactly once");

    let store = open(cold_root.path(), "autotune", "device0/matmul");
    assert_eq!(
        store.get(&"shared".to_string()),
        Some(&1),
        "first root wins"
    );
    assert_eq!(store.get(&"only-first".to_string()), Some(&10));
    assert_eq!(store.get(&"only-second".to_string()), Some(&20));
}

#[test]
#[serial_test::serial]
fn exporting_can_be_restricted_to_some_namespaces() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("autotune-only.bundle");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]);
    warm(warm_root.path(), "throughput", "device0/copy", &[("k", 2)]);

    let options = ExportOptions {
        name: "Autotune only".to_string(),
        namespaces: vec!["autotune".to_string()],
        ..Default::default()
    };
    export(&[warm_root.path()], &bundle_path, &options).unwrap();

    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);
    import_into(cold_root.path(), bundle.as_ref());

    assert_eq!(
        open(cold_root.path(), "autotune", "device0/matmul").get(&"k".to_string()),
        Some(&1)
    );
    assert_eq!(
        open(cold_root.path(), "throughput", "device0/copy").get(&"k".to_string()),
        None,
        "the throughput namespace was not exported"
    );
}

/// Re-exporting must replace the bundle, never merge into the stale one.
#[test]
#[serial_test::serial]
fn exporting_over_an_existing_bundle_replaces_it() {
    let first_root = tempfile::tempdir().unwrap();
    let second_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("replaced.bundle");

    warm(
        first_root.path(),
        "autotune",
        "device0/matmul",
        &[("old", 1)],
    );
    export_to(first_root.path(), &bundle_path, BundleFormat::Sqlite);

    warm(
        second_root.path(),
        "autotune",
        "device0/matmul",
        &[("new", 2)],
    );
    export_to(second_root.path(), &bundle_path, BundleFormat::Sqlite);

    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);
    import_into(cold_root.path(), bundle.as_ref());

    let store = open(cold_root.path(), "autotune", "device0/matmul");
    assert_eq!(store.get(&"new".to_string()), Some(&2));
    assert_eq!(store.get(&"old".to_string()), None);
}

/// Both formats must ship exactly the same entries.
#[test]
#[serial_test::serial]
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
    let flat_path = dir.path().join("bundle.ccb");
    export_to(warm_root.path(), &sqlite_path, BundleFormat::Sqlite);
    export_to(warm_root.path(), &flat_path, BundleFormat::Flat);

    let namespace = format!("autotune/{}/device0/matmul", env!("CARGO_PKG_VERSION"));
    let sqlite = open_bundle(&sqlite_path, BundleFormat::Sqlite);
    let flat = open_bundle(&flat_path, BundleFormat::Flat);

    let collect = |bundle: &dyn Bundle| {
        let mut entries = Vec::new();
        bundle.scan(&namespace, &mut |key, value| {
            entries.push((key.to_vec(), value.to_vec()));
        });
        entries.sort();
        entries
    };

    let from_sqlite = collect(sqlite.as_ref());
    assert_eq!(from_sqlite.len(), 3);
    assert_eq!(from_sqlite, collect(flat.as_ref()));
    assert_eq!(sqlite.namespaces(), flat.namespaces());

    for (key, _) in &from_sqlite {
        assert_eq!(
            flat.get(&namespace, key).map(|v| v.to_vec()),
            sqlite.get(&namespace, key).map(|v| v.to_vec())
        );
    }
    assert_eq!(flat.get(&namespace, b"missing"), None);
    assert_eq!(flat.get("no/such/namespace", b"a"), None);
}

/// An unrelated file must never be silently destroyed by an export.
#[test]
#[serial_test::serial]
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
#[serial_test::serial]
fn a_foreign_file_is_not_a_bundle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("not-a-bundle.cubecl");
    std::fs::write(&path, b"definitely not sqlite").unwrap();

    assert!(SqliteBundle::open(&path).is_err());
    assert!(EmbeddedBundle::open(Bytes::from_bytes_vec(std::fs::read(&path).unwrap())).is_err());
}

/// Two named environments must be separate stores, and only the active one is
/// ever touched.
#[test]
#[serial_test::serial]
fn environments_are_isolated_and_switchable() {
    use cubecl_environment::environment;

    let root = tempfile::tempdir().unwrap();

    environment::activate("first");
    assert_eq!(environment::active(), "first");
    warm(root.path(), "autotune", "device0/matmul", &[("k", 1)]);

    environment::activate("second");
    warm(root.path(), "autotune", "device0/matmul", &[("k", 2)]);

    // Each environment kept its own answer.
    assert_eq!(
        open(root.path(), "autotune", "device0/matmul").get(&"k".to_string()),
        Some(&2)
    );
    environment::activate("first");
    assert_eq!(
        open(root.path(), "autotune", "device0/matmul").get(&"k".to_string()),
        Some(&1)
    );

    // Both are discoverable, each in its own file.
    assert_eq!(environment::list(), vec!["first", "second"]);

    // And the active one can report what it holds, which is what you consult
    // before bundling.
    let namespaces = environment::namespaces();
    assert_eq!(namespaces.len(), 1);
    assert_eq!(
        namespaces[0].namespace,
        format!("autotune/{}/device0/matmul", env!("CARGO_PKG_VERSION"))
    );

    environment::activate(environment::DEFAULT);
}

/// Importing must land in the active environment, not somewhere fixed.
#[test]
#[serial_test::serial]
fn importing_targets_the_active_environment() {
    use cubecl_environment::environment;

    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("test.bundle");

    environment::activate(environment::DEFAULT);
    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 7)]);
    export_to(warm_root.path(), &bundle_path, BundleFormat::Sqlite);

    environment::activate("target");
    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);
    import_into(cold_root.path(), bundle.as_ref());

    assert_eq!(
        open(cold_root.path(), "autotune", "device0/matmul").get(&"k".to_string()),
        Some(&7)
    );

    // The default environment of the same root saw nothing.
    environment::activate(environment::DEFAULT);
    assert_eq!(
        open(cold_root.path(), "autotune", "device0/matmul").get(&"k".to_string()),
        None
    );
}

/// The same contract as `a_local_value_replaces_a_stale_imported_one`, for the
/// lazy store. It used to short-circuit on its own memo and never let the
/// storage arbitrate, so a stale imported kernel wedged compilation forever.
#[test]
#[serial_test::serial]
fn a_local_blob_replaces_a_stale_imported_one() {
    use cubecl_environment::persistence::blob::BlobStore;

    let source = tempfile::tempdir().unwrap();
    let target = tempfile::tempdir().unwrap();
    let bundle_path = source.path().join("ship.bundle");

    let kernels = |root: &std::path::Path| {
        cubecl_environment::environment::set_root(root);
        BlobStore::<String, Bytes>::new("spirv", KvStoreOptions::default())
    };

    kernels(source.path())
        .insert("kernel".to_string(), Bytes::from_bytes_vec(vec![1u8]))
        .unwrap();
    export_to(source.path(), &bundle_path, BundleFormat::Sqlite);

    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);
    cubecl_environment::environment::set_root(target.path());
    assert_eq!(import_into(target.path(), bundle.as_ref()).imported, 1);

    // The machine compiles a different kernel for the same key.
    let mut store = kernels(target.path());
    store
        .insert("kernel".to_string(), Bytes::from_bytes_vec(vec![2u8]))
        .expect("a local kernel must replace a stale imported one");

    // Durable, and a re-import must not resurrect the shipped one.
    assert_eq!(import_into(target.path(), bundle.as_ref()).imported, 0);
    assert_eq!(
        kernels(target.path())
            .get(&"kernel".to_string())
            .map(|v| v.to_vec()),
        Some(vec![2u8])
    );

    // Now that it is local, a second disagreement is a plain conflict.
    let mut store = kernels(target.path());
    assert!(
        store
            .insert("kernel".to_string(), Bytes::from_bytes_vec(vec![3u8]))
            .is_err()
    );
}
