#![cfg(feature = "persistence")]

use cubecl_environment::bundle::{
    Bundle, BundleError, BundleFormat, BundleManifest, EmbeddedBundle, ExportOptions, SqliteBundle,
    export, import,
};
use cubecl_environment::bytes::Bytes;
use cubecl_environment::persistence::{Namespace, Store, StoreOptions};

// Storage resolves through the process-global active environment, so these
// tests are serialized: only one environment is active at a time by design.

/// Pins the environment to `root`, which is process-global; every test here
/// is serialized for that reason.
fn options(root: &std::path::Path, name: &str, path: &str) -> StoreOptions {
    cubecl_environment::environment::set_root(root);
    StoreOptions::new().storage(Namespace::scoped(name, path))
}

/// Warms `root` with one namespace's worth of entries, as an application run
/// would.
async fn warm(root: &std::path::Path, name: &str, path: &str, entries: &[(&str, u32)]) {
    let mut store = Store::<String, u32>::open(options(root, name, path)).await;

    for (key, value) in entries {
        store.insert(key.to_string(), *value).await.unwrap();
    }
}

async fn open(root: &std::path::Path, name: &str, path: &str) -> Store<String, u32> {
    Store::open(options(root, name, path)).await
}

async fn export_to(root: &std::path::Path, out: &std::path::Path, format: BundleFormat) {
    export_roots("Test GPU Linux", &[root], out, format, &[])
        .await
        .unwrap();
}

/// The general form: several roots, one bundle name, one format, an optional
/// namespace filter. The result is returned so the failure cases can assert on
/// it.
async fn export_roots(
    name: &str,
    roots: &[&std::path::Path],
    out: &std::path::Path,
    format: BundleFormat,
    namespaces: &[&str],
) -> Result<BundleManifest, BundleError> {
    let options = ExportOptions {
        name: name.to_string(),
        format,
        namespaces: namespaces.iter().map(|it| it.to_string()).collect(),
        ..Default::default()
    };
    export(roots, out, &options).await
}

/// Imports into `root`, which becomes the active environment first: `import`
/// fills whichever environment is active.
async fn import_into(
    root: &std::path::Path,
    bundle: &dyn Bundle,
) -> cubecl_environment::bundle::ImportReport {
    cubecl_environment::environment::set_root(root);
    import(bundle).await
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
#[tokio::test]
#[serial_test::serial]
async fn importing_fills_the_storage_and_the_bundle_becomes_irrelevant() {
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
        )
        .await;
        export_to(warm_root.path(), &bundle_path, format).await;

        let bundle = open_bundle(&bundle_path, format);
        let report = import_into(cold_root.path(), bundle.as_ref()).await;
        assert_eq!(report.imported, 2, "{format:?}");
        assert_eq!(report.skipped, 0, "{format:?}");

        // The bundle is gone, yet the entries are there.
        drop(bundle);
        std::fs::remove_file(&bundle_path).unwrap();

        let store = open(cold_root.path(), "autotune", "device0/matmul").await;
        assert_eq!(store.get(&"shape=2x2".to_string()), Some(&3), "{format:?}");
        assert_eq!(store.get(&"shape=4x4".to_string()), Some(&7), "{format:?}");
    }
}

/// Importing twice must not duplicate or error.
#[tokio::test]
#[serial_test::serial]
async fn importing_is_idempotent() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("test.bundle");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]).await;
    export_to(warm_root.path(), &bundle_path, BundleFormat::Sqlite).await;

    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);

    let first = import_into(cold_root.path(), bundle.as_ref()).await;
    assert_eq!((first.imported, first.skipped), (1, 0));

    let second = import_into(cold_root.path(), bundle.as_ref()).await;
    assert_eq!(
        (second.imported, second.skipped),
        (0, 1),
        "the second import must skip what is already stored"
    );

    let store = open(cold_root.path(), "autotune", "device0/matmul").await;
    assert_eq!(store.len(), 1);
}

/// A value computed on this machine must never be overwritten by a bundle.
#[tokio::test]
#[serial_test::serial]
async fn importing_never_overwrites_a_local_value() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let local_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("test.bundle");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]).await;
    export_to(warm_root.path(), &bundle_path, BundleFormat::Sqlite).await;

    // This machine already computed a different answer for the same key.
    warm(
        local_root.path(),
        "autotune",
        "device0/matmul",
        &[("k", 42)],
    )
    .await;

    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);
    let report = import_into(local_root.path(), bundle.as_ref()).await;
    assert_eq!((report.imported, report.skipped), (0, 1));

    let store = open(local_root.path(), "autotune", "device0/matmul").await;
    assert_eq!(
        store.get(&"k".to_string()),
        Some(&42),
        "the local value wins"
    );
}

/// The mirror image: a stale imported entry must be replaceable by a locally
/// computed one, or a bad bundle would wedge the application forever.
#[tokio::test]
#[serial_test::serial]
async fn a_local_value_replaces_a_stale_imported_one() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let local_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("test.bundle");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]).await;
    export_to(warm_root.path(), &bundle_path, BundleFormat::Sqlite).await;

    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);
    import_into(local_root.path(), bundle.as_ref()).await;

    // The application disagrees with the shipped answer.
    let mut store = open(local_root.path(), "autotune", "device0/matmul").await;
    assert_eq!(store.get(&"k".to_string()), Some(&1));
    store.insert("k".to_string(), 99).await.unwrap();

    // It must stick, including across a reopen, and a re-import must not
    // resurrect the stale value.
    let report = import_into(local_root.path(), bundle.as_ref()).await;
    assert_eq!(report.imported, 0);

    let store = open(local_root.path(), "autotune", "device0/matmul").await;
    assert_eq!(store.get(&"k".to_string()), Some(&99));

    // And now that it is local, a second disagreement is a plain conflict.
    let mut store = store;
    assert!(store.insert("k".to_string(), 7).await.is_err());
}

/// Import must cover every namespace the bundle holds.
#[tokio::test]
#[serial_test::serial]
async fn importing_covers_every_namespace() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("test.bundle");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]).await;
    warm(warm_root.path(), "throughput", "device0/copy", &[("k", 2)]).await;
    export_to(warm_root.path(), &bundle_path, BundleFormat::Flat).await;

    let bundle = open_bundle(&bundle_path, BundleFormat::Flat);
    assert_eq!(bundle.namespaces().len(), 2);

    let report = import_into(cold_root.path(), bundle.as_ref()).await;
    assert_eq!(report.imported, 2);
    assert_eq!(report.namespaces.len(), 2);

    assert_eq!(
        open(cold_root.path(), "autotune", "device0/matmul")
            .await
            .get(&"k".to_string()),
        Some(&1)
    );
    assert_eq!(
        open(cold_root.path(), "throughput", "device0/copy")
            .await
            .get(&"k".to_string()),
        Some(&2)
    );
}

/// A cache root must be able to report what it holds, which is what you
/// consult before bundling.
#[tokio::test]
#[serial_test::serial]
async fn a_cache_root_lists_its_namespaces() {
    let root = tempfile::tempdir().unwrap();

    warm(root.path(), "autotune", "device0/matmul", &[("k", 1)]).await;
    warm(
        root.path(),
        "throughput",
        "device0/copy",
        &[("k", 2), ("j", 3)],
    )
    .await;

    let version = env!("CARGO_PKG_VERSION");

    let summary = cubecl_environment::environment::namespaces().await;
    assert_eq!(
        summary
            .iter()
            .map(|entry| entry.namespace.clone())
            .collect::<Vec<_>>(),
        vec![
            format!("autotune/{version}/device0/matmul"),
            format!("throughput/{version}/device0/copy"),
        ]
    );
    assert_eq!(summary[0].entries, 1);
    assert_eq!(summary[1].entries, 2);
}

/// Merging cache roots dedupes on the primary key. The original file-copy
/// exporter concatenated colliding files instead.
///
/// The flat format merges roots in the reader, not in `SQLite`, so dedup is a
/// separate implementation and both formats need the coverage.
#[tokio::test]
#[serial_test::serial]
async fn exporting_several_roots_dedupes_shared_keys() {
    for format in [BundleFormat::Sqlite, BundleFormat::Flat] {
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
        )
        .await;
        warm(
            second.path(),
            "autotune",
            "device0/matmul",
            &[("shared", 2), ("only-second", 20)],
        )
        .await;

        export_roots(
            "Merged",
            &[first.path(), second.path()],
            &bundle_path,
            format,
            &[],
        )
        .await
        .unwrap();

        let bundle = open_bundle(&bundle_path, format);
        let report = import_into(cold_root.path(), bundle.as_ref()).await;
        assert_eq!(
            report.imported, 3,
            "the shared key appears exactly once: {format:?}"
        );

        let store = open(cold_root.path(), "autotune", "device0/matmul").await;
        assert_eq!(
            store.get(&"shared".to_string()),
            Some(&1),
            "first root wins: {format:?}"
        );
        assert_eq!(
            store.get(&"only-first".to_string()),
            Some(&10),
            "{format:?}"
        );
        assert_eq!(
            store.get(&"only-second".to_string()),
            Some(&20),
            "{format:?}"
        );
    }
}

#[tokio::test]
#[serial_test::serial]
async fn exporting_can_be_restricted_to_some_namespaces() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("autotune-only.bundle");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]).await;
    warm(warm_root.path(), "throughput", "device0/copy", &[("k", 2)]).await;

    export_roots(
        "Autotune only",
        &[warm_root.path()],
        &bundle_path,
        BundleFormat::Sqlite,
        &["autotune"],
    )
    .await
    .unwrap();

    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);
    import_into(cold_root.path(), bundle.as_ref()).await;

    assert_eq!(
        open(cold_root.path(), "autotune", "device0/matmul")
            .await
            .get(&"k".to_string()),
        Some(&1)
    );
    assert_eq!(
        open(cold_root.path(), "throughput", "device0/copy")
            .await
            .get(&"k".to_string()),
        None,
        "the throughput namespace was not exported"
    );
}

/// Re-exporting must replace the bundle, never merge into the stale one.
///
/// The flat format has no manifest row to validate an existing file with, so
/// re-exporting over one used to be rejected as a foreign file forever.
#[tokio::test]
#[serial_test::serial]
async fn exporting_over_an_existing_bundle_replaces_it() {
    for format in [BundleFormat::Sqlite, BundleFormat::Flat] {
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
        )
        .await;
        export_to(first_root.path(), &bundle_path, format).await;

        warm(
            second_root.path(),
            "autotune",
            "device0/matmul",
            &[("new", 2)],
        )
        .await;
        export_to(second_root.path(), &bundle_path, format).await;

        let bundle = open_bundle(&bundle_path, format);
        import_into(cold_root.path(), bundle.as_ref()).await;

        let store = open(cold_root.path(), "autotune", "device0/matmul").await;
        assert_eq!(store.get(&"new".to_string()), Some(&2), "{format:?}");
        assert_eq!(store.get(&"old".to_string()), None, "{format:?}");
    }
}

/// A failed export must leave the previous bundle intact, not a stub that
/// wedges every later export.
#[tokio::test]
#[serial_test::serial]
async fn a_failed_export_leaves_the_previous_bundle_alone() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("kept.ccb");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]).await;
    export_to(warm_root.path(), &bundle_path, BundleFormat::Flat).await;
    let exported = std::fs::read(&bundle_path).unwrap();

    // A source that is not a database at all fails the export.
    let broken = bundle_dir.path().join("broken.db");
    std::fs::write(&broken, b"not a database").unwrap();
    assert!(
        export_roots("Doomed", &[&broken], &bundle_path, BundleFormat::Flat, &[])
            .await
            .is_err()
    );

    assert_eq!(std::fs::read(&bundle_path).unwrap(), exported);
    assert!(
        !bundle_dir.path().join("kept.ccb.tmp").exists(),
        "the staged file must not be left behind"
    );

    // And the bundle is still replaceable, which the stub used to prevent.
    export_to(warm_root.path(), &bundle_path, BundleFormat::Flat).await;
}

/// The shipping path for wasm and no-std: a blob embedded with
/// `include_bytes!`.
#[tokio::test]
#[serial_test::serial]
async fn a_static_blob_opens_as_a_bundle() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("static.ccb");

    warm(
        warm_root.path(),
        "autotune",
        "device0/matmul",
        &[("a", 1), ("b", 2)],
    )
    .await;
    export_to(warm_root.path(), &bundle_path, BundleFormat::Flat).await;

    // Stands in for `include_bytes!`, which needs a file at compile time.
    let blob: &'static [u8] = Vec::leak(std::fs::read(&bundle_path).unwrap());
    let bundle = EmbeddedBundle::from_static(blob).unwrap();

    assert_eq!(bundle.len(), 2);
    assert_eq!(bundle.manifest().unwrap().name, "Test GPU Linux");
    assert_eq!(import_into(cold_root.path(), &bundle).await.imported, 2);
    assert_eq!(
        open(cold_root.path(), "autotune", "device0/matmul")
            .await
            .get(&"b".to_string()),
        Some(&2)
    );
}

/// The prefix filter must select the same namespaces in both formats.
#[tokio::test]
#[serial_test::serial]
async fn exporting_to_flat_can_be_restricted_to_some_namespaces() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("autotune-only.ccb");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]).await;
    warm(warm_root.path(), "throughput", "device0/copy", &[("k", 2)]).await;

    let version = env!("CARGO_PKG_VERSION");
    let selected = format!("autotune/{version}/device0/matmul");

    export_roots(
        "Autotune only",
        &[warm_root.path()],
        &bundle_path,
        BundleFormat::Flat,
        &["autotune"],
    )
    .await
    .unwrap();
    assert_eq!(
        open_bundle(&bundle_path, BundleFormat::Flat).namespaces(),
        vec![selected.clone()]
    );

    // An empty prefix is no filter at all, in both formats.
    export_roots(
        "Autotune only",
        &[warm_root.path()],
        &bundle_path,
        BundleFormat::Flat,
        &[""],
    )
    .await
    .unwrap();
    assert_eq!(
        open_bundle(&bundle_path, BundleFormat::Flat)
            .namespaces()
            .len(),
        2
    );

    let sqlite_path = bundle_dir.path().join("autotune-only.cubecl");
    export_roots(
        "Autotune only",
        &[warm_root.path()],
        &sqlite_path,
        BundleFormat::Sqlite,
        &["autotune"],
    )
    .await
    .unwrap();
    assert_eq!(
        open_bundle(&sqlite_path, BundleFormat::Sqlite).namespaces(),
        vec![selected]
    );
}

/// Both formats must ship exactly the same entries.
#[tokio::test]
#[serial_test::serial]
async fn both_formats_serve_the_same_entries() {
    let warm_root = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();

    warm(
        warm_root.path(),
        "autotune",
        "device0/matmul",
        &[("a", 1), ("b", 2), ("c", 3)],
    )
    .await;

    let sqlite_path = dir.path().join("bundle.cubecl");
    let flat_path = dir.path().join("bundle.ccb");
    export_to(warm_root.path(), &sqlite_path, BundleFormat::Sqlite).await;
    export_to(warm_root.path(), &flat_path, BundleFormat::Flat).await;

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
#[tokio::test]
#[serial_test::serial]
async fn exporting_over_a_foreign_file_fails() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("precious.txt");
    std::fs::write(&target, b"do not delete me").unwrap();

    assert!(
        export_roots("Nope", &[dir.path()], &target, BundleFormat::Sqlite, &[])
            .await
            .is_err()
    );
    assert_eq!(std::fs::read(&target).unwrap(), b"do not delete me");
}

#[tokio::test]
#[serial_test::serial]
async fn a_foreign_file_is_not_a_bundle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("not-a-bundle.cubecl");
    std::fs::write(&path, b"definitely not sqlite").unwrap();

    assert!(SqliteBundle::open(&path).is_err());
    assert!(EmbeddedBundle::open(Bytes::from_bytes_vec(std::fs::read(&path).unwrap())).is_err());
}

/// Two named environments must be separate stores, and only the active one is
/// ever touched.
#[tokio::test]
#[serial_test::serial]
async fn environments_are_isolated_and_switchable() {
    use cubecl_environment::environment;

    let root = tempfile::tempdir().unwrap();

    environment::activate("first");
    assert_eq!(&*environment::active(), "first");
    warm(root.path(), "autotune", "device0/matmul", &[("k", 1)]).await;

    environment::activate("second");
    warm(root.path(), "autotune", "device0/matmul", &[("k", 2)]).await;

    // Each environment kept its own answer.
    assert_eq!(
        open(root.path(), "autotune", "device0/matmul")
            .await
            .get(&"k".to_string()),
        Some(&2)
    );
    environment::activate("first");
    assert_eq!(
        open(root.path(), "autotune", "device0/matmul")
            .await
            .get(&"k".to_string()),
        Some(&1)
    );

    // Both are discoverable, each in its own file.
    assert_eq!(environment::list(), vec!["first", "second"]);

    // And the active one can report what it holds, which is what you consult
    // before bundling.
    let namespaces = environment::namespaces().await;
    assert_eq!(namespaces.len(), 1);
    assert_eq!(
        namespaces[0].namespace,
        format!("autotune/{}/device0/matmul", env!("CARGO_PKG_VERSION"))
    );

    environment::activate(environment::DEFAULT);
}

/// The in-place path: `environment::bundle()` captures the active
/// environment, and `environment::load` mounts the saved file as the active
/// environment — entries are served from it directly, nothing is imported,
/// and stores that already exist reset onto it.
#[tokio::test]
#[serial_test::serial]
async fn a_saved_bundle_can_be_loaded_in_place() {
    use cubecl_environment::environment;

    let warm_root = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("shipped.db");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 7)]).await;
    environment::bundle()
        .save(&bundle_path, BundleFormat::Sqlite)
        .await
        .unwrap();

    // A machine with a cold environment, holding a store opened before the
    // bundle arrives.
    let mut store = open(cold_root.path(), "autotune", "device0/matmul").await;
    assert_eq!(store.get(&"k".to_string()), None);

    environment::load(&bundle_path);

    // The existing store resets onto the bundle, and a fresh one sees it too.
    store.sync().await;
    assert_eq!(store.get(&"k".to_string()), Some(&7));
    let fresh: Store<String, u32> = environment::store(
        StoreOptions::new().storage(Namespace::scoped("autotune", "device0/matmul")),
    )
    .await;
    assert_eq!(fresh.get(&"k".to_string()), Some(&7));

    // The bundle was mounted, not imported: the cold environment stays cold.
    environment::set_root(cold_root.path());
    assert_eq!(
        open(cold_root.path(), "autotune", "device0/matmul")
            .await
            .get(&"k".to_string()),
        None
    );
}

/// Importing must land in the active environment, not somewhere fixed.
#[tokio::test]
#[serial_test::serial]
async fn importing_targets_the_active_environment() {
    use cubecl_environment::environment;

    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("test.bundle");

    environment::activate(environment::DEFAULT);
    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 7)]).await;
    export_to(warm_root.path(), &bundle_path, BundleFormat::Sqlite).await;

    environment::activate("target");
    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);
    import_into(cold_root.path(), bundle.as_ref()).await;

    assert_eq!(
        open(cold_root.path(), "autotune", "device0/matmul")
            .await
            .get(&"k".to_string()),
        Some(&7)
    );

    // The default environment of the same root saw nothing.
    environment::activate(environment::DEFAULT);
    assert_eq!(
        open(cold_root.path(), "autotune", "device0/matmul")
            .await
            .get(&"k".to_string()),
        None
    );
}

/// The same contract as `a_local_value_replaces_a_stale_imported_one`, for the
/// lazy store. It used to short-circuit on its own memo and never let the
/// storage arbitrate, so a stale imported kernel wedged compilation forever.
#[tokio::test]
#[serial_test::serial]
async fn a_local_blob_replaces_a_stale_imported_one() {
    use cubecl_environment::persistence::CacheOption;

    let source = tempfile::tempdir().unwrap();
    let target = tempfile::tempdir().unwrap();
    let bundle_path = source.path().join("ship.bundle");

    let kernels = |root: &std::path::Path| {
        cubecl_environment::environment::set_root(root);
        Store::<String, Bytes>::open(
            StoreOptions::new()
                .storage(Namespace::new("spirv"))
                .cache(CacheOption::Lazy),
        )
    };

    kernels(source.path())
        .await
        .insert("kernel".to_string(), Bytes::from_bytes_vec(vec![1u8]))
        .await
        .unwrap();
    export_to(source.path(), &bundle_path, BundleFormat::Sqlite).await;

    let bundle = open_bundle(&bundle_path, BundleFormat::Sqlite);
    cubecl_environment::environment::set_root(target.path());
    assert_eq!(
        import_into(target.path(), bundle.as_ref()).await.imported,
        1
    );

    // The machine compiles a different kernel for the same key.
    let mut store = kernels(target.path()).await;
    store
        .insert("kernel".to_string(), Bytes::from_bytes_vec(vec![2u8]))
        .await
        .expect("a local kernel must replace a stale imported one");

    // Durable, and a re-import must not resurrect the shipped one.
    assert_eq!(
        import_into(target.path(), bundle.as_ref()).await.imported,
        0
    );
    assert_eq!(
        kernels(target.path())
            .await
            .get_mut(&"kernel".to_string())
            .await
            .map(|v| v.to_vec()),
        Some(vec![2u8])
    );

    // Now that it is local, a second disagreement is a plain conflict.
    let mut store = kernels(target.path()).await;
    assert!(
        store
            .insert("kernel".to_string(), Bytes::from_bytes_vec(vec![3u8]))
            .await
            .is_err()
    );
}

/// The shipping scenario: a bundle installed where it cannot be written to.
///
/// A container image layer, a Nix store path and a macOS app bundle are all
/// read-only *directories*. The engine never checkpoints on close, so an
/// export that shipped the file without folding its WAL in would import zero
/// entries here, and report success while doing it.
#[cfg(unix)]
#[tokio::test]
#[serial_test::serial]
async fn a_bundle_installed_read_only_still_imports() {
    use std::os::unix::fs::PermissionsExt;

    let source = tempfile::tempdir().unwrap();
    let installed = tempfile::tempdir().unwrap();
    let target = tempfile::tempdir().unwrap();

    warm(
        source.path(),
        "default",
        "autotune/matmul",
        &[("shape=2x2", 42)],
    )
    .await;

    let bundle_path = installed.path().join("shipped.ccb");
    export_to(source.path(), &bundle_path, BundleFormat::Sqlite).await;

    let original = std::fs::metadata(installed.path()).unwrap().permissions();
    std::fs::set_permissions(installed.path(), std::fs::Permissions::from_mode(0o555)).unwrap();

    // Root ignores the mode bits, so the test would prove nothing there.
    let writable = std::fs::File::create(installed.path().join("probe")).is_ok();

    let imported = if writable {
        None
    } else {
        let bundle = SqliteBundle::open(&bundle_path).expect("a read-only bundle opens");
        Some(import_into(target.path(), &bundle).await.imported)
    };

    std::fs::set_permissions(installed.path(), original).unwrap();

    if imported.is_none() {
        return;
    }

    assert_eq!(imported, Some(1));
    assert_eq!(
        open(target.path(), "default", "autotune/matmul")
            .await
            .get(&"shape=2x2".to_string()),
        Some(&42)
    );
}

/// The same install, mounted rather than imported: `environment::load` on a
/// file nobody may write opens it read-only and serves its entries, instead
/// of degrading to memory because the rebuild-on-open can't write.
#[cfg(unix)]
#[tokio::test]
#[serial_test::serial]
async fn a_bundle_installed_read_only_can_be_loaded_in_place() {
    use cubecl_environment::environment;
    use std::os::unix::fs::PermissionsExt;

    let source = tempfile::tempdir().unwrap();
    let installed = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();

    warm(source.path(), "autotune", "device0/matmul", &[("k", 7)]).await;
    let bundle_path = installed.path().join("shipped.db");
    export_to(source.path(), &bundle_path, BundleFormat::Sqlite).await;

    let original = std::fs::metadata(installed.path()).unwrap().permissions();
    std::fs::set_permissions(&bundle_path, std::fs::Permissions::from_mode(0o444)).unwrap();
    std::fs::set_permissions(installed.path(), std::fs::Permissions::from_mode(0o555)).unwrap();

    // Root ignores the mode bits, so the test would prove nothing there.
    let writable = std::fs::File::create(installed.path().join("probe")).is_ok();

    let served = if writable {
        None
    } else {
        environment::set_root(cold_root.path());
        environment::load(&bundle_path);
        let store: Store<String, u32> = environment::store(
            StoreOptions::new().storage(Namespace::scoped("autotune", "device0/matmul")),
        )
        .await;
        Some(store.get(&"k".to_string()).copied())
    };

    environment::set_root(cold_root.path());
    std::fs::set_permissions(installed.path(), original).unwrap();

    if let Some(served) = served {
        assert_eq!(served, Some(7));
        assert!(
            !installed.path().join("shipped.db-wal").exists(),
            "a read-only mount must not write beside the file"
        );
    }
}

/// A cache root nobody may write — a mounted bundle, a store path — can
/// still be exported from: the sources are read, never opened for writing.
#[cfg(unix)]
#[tokio::test]
#[serial_test::serial]
async fn exporting_from_a_read_only_root_works() {
    use std::os::unix::fs::PermissionsExt;

    let source = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let cold_root = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("from-read-only.bundle");

    warm(source.path(), "autotune", "device0/matmul", &[("k", 7)]).await;
    let database = cubecl_environment::environment::path();

    let original = std::fs::metadata(source.path()).unwrap().permissions();
    std::fs::set_permissions(&database, std::fs::Permissions::from_mode(0o444)).unwrap();
    std::fs::set_permissions(source.path(), std::fs::Permissions::from_mode(0o555)).unwrap();

    // Root ignores the mode bits, so the test would prove nothing there.
    let writable = std::fs::File::create(source.path().join("probe")).is_ok();

    let exported = if writable {
        None
    } else {
        Some(
            export_roots(
                "Read-only",
                &[source.path()],
                &bundle_path,
                BundleFormat::Flat,
                &[],
            )
            .await,
        )
    };

    std::fs::set_permissions(source.path(), original).unwrap();
    let Some(exported) = exported else {
        return;
    };
    exported.expect("a read-only root exports");

    let bundle = open_bundle(&bundle_path, BundleFormat::Flat);
    assert_eq!(
        import_into(cold_root.path(), bundle.as_ref())
            .await
            .imported,
        1
    );
}

/// A bundle written at a database schema this build doesn't read is still a
/// bundle: re-exporting over it replaces it rather than refusing to touch a
/// "foreign" file.
#[tokio::test]
#[serial_test::serial]
async fn exporting_over_an_older_schema_bundle_replaces_it() {
    let warm_root = tempfile::tempdir().unwrap();
    let bundle_dir = tempfile::tempdir().unwrap();
    let bundle_path = bundle_dir.path().join("old.bundle");

    warm(warm_root.path(), "autotune", "device0/matmul", &[("k", 1)]).await;
    export_to(warm_root.path(), &bundle_path, BundleFormat::Sqlite).await;

    // Age the file: rewrite its schema version.
    {
        let database = turso::Builder::new_local(bundle_path.to_str().unwrap())
            .build()
            .await
            .unwrap();
        let connection = database.connect().unwrap();
        connection
            .execute("UPDATE meta SET v = '3' WHERE k = 'schema_version'", ())
            .await
            .unwrap();
    }
    assert!(matches!(
        SqliteBundle::open(&bundle_path),
        Err(BundleError::UnsupportedDatabase(schema)) if schema == "3"
    ));

    export_to(warm_root.path(), &bundle_path, BundleFormat::Sqlite).await;
    assert_eq!(
        open_bundle(&bundle_path, BundleFormat::Sqlite)
            .namespaces()
            .len(),
        1
    );
}
