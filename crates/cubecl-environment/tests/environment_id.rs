#![cfg(all(feature = "persistence", not(target_family = "wasm")))]

use cubecl_environment::bytes::Bytes;
use cubecl_environment::environment::{
    self, EnvironmentBundle, EnvironmentError, EnvironmentId, EnvironmentOrigin,
};
use cubecl_environment::persistence::{CacheOption, Namespace, Store, StoreOptions};

// The active environment is process-global, so these tests are serialized.

fn store() -> Store<String, u32> {
    Store::new(StoreOptions::new().storage(Namespace::scoped("environment-id", "entries")))
}

/// A store that reads the storage on a miss, as the kernel and autotune
/// caches do.
fn lazy_store() -> Store<String, u32> {
    Store::new(
        StoreOptions::new()
            .storage(Namespace::scoped("environment-id", "entries"))
            .cache(CacheOption::Lazy),
    )
}

/// Activates `id` and writes `entries` into it, as a build would.
fn warm(id: &EnvironmentId, entries: &[(&str, u32)]) {
    id.activate().unwrap();
    let mut store = store();
    for (key, value) in entries {
        store.insert(key.to_string(), *value).unwrap();
    }
}

/// Activates `id` and reads `key` from it.
fn read(id: &EnvironmentId, key: &str) -> Option<u32> {
    id.activate().unwrap();
    store().get(&key.to_string()).copied()
}

fn rooted() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    environment::set_root(root.path());
    environment::activate(environment::DEFAULT);
    root
}

/// A saved environment loads elsewhere holding what it held, and replaces
/// what an environment of that name held before.
#[test]
#[serial_test::serial]
fn a_saved_environment_replaces_one_created_from_its_bundle() {
    let _root = rooted();
    let built = EnvironmentId::create("built", EnvironmentOrigin::Empty).unwrap();
    warm(&built, &[("kernel", 1)]);
    let stale = EnvironmentId::create("loaded", EnvironmentOrigin::Empty).unwrap();
    warm(&stale, &[("stale", 2)]);
    environment::activate(environment::DEFAULT);

    let bundle = built.save().unwrap();
    let loaded = EnvironmentId::create("loaded", EnvironmentOrigin::Bundle(bundle)).unwrap();

    assert_eq!(read(&loaded, "kernel"), Some(1));
    assert_eq!(read(&loaded, "stale"), None);
}

/// A fork holds a copy: what is written to the source afterwards stays out of
/// it.
#[test]
#[serial_test::serial]
fn a_fork_copies_what_its_source_holds_now() {
    let _root = rooted();
    let source = EnvironmentId::create("source", EnvironmentOrigin::Empty).unwrap();
    warm(&source, &[("before", 1)]);
    environment::activate(environment::DEFAULT);

    let fork = EnvironmentId::create("fork", EnvironmentOrigin::Fork(source.clone())).unwrap();
    warm(&source, &[("after", 2)]);

    assert_eq!(read(&fork, "before"), Some(1));
    assert_eq!(read(&fork, "after"), None);
}

/// A merge brings in the keys the environment lacks and leaves the ones it
/// holds as they were: it adds what another environment computed, and never
/// changes a value something may already have read.
#[test]
#[serial_test::serial]
fn a_merge_keeps_what_the_environment_held_beside_what_it_brings() {
    let _root = rooted();
    let built = EnvironmentId::create("built", EnvironmentOrigin::Empty).unwrap();
    warm(&built, &[("kernel", 1), ("shared", 10)]);
    let application = EnvironmentId::create("application", EnvironmentOrigin::Empty).unwrap();
    warm(&application, &[("other", 2), ("shared", 20)]);
    environment::activate(environment::DEFAULT);

    application
        .merge(EnvironmentOrigin::Bundle(built.save().unwrap()))
        .unwrap();

    assert_eq!(read(&application, "kernel"), Some(1));
    assert_eq!(read(&application, "other"), Some(2));
    assert_eq!(read(&application, "shared"), Some(20));
}

/// A merge into the active environment does not switch it, and a store
/// opened before the merge reads the merged keys.
///
/// A switch makes every backend drop and recompile its kernels, so a merge
/// that switched would cost a recompilation per merged model.
#[test]
#[serial_test::serial]
fn a_merge_into_the_active_environment_does_not_switch_it() {
    let _root = rooted();
    let source = EnvironmentId::create("source", EnvironmentOrigin::Empty).unwrap();
    warm(&source, &[("kernel", 1)]);
    let active = EnvironmentId::create("active", EnvironmentOrigin::Empty).unwrap();
    warm(&active, &[("own", 2)]);
    let mut opened = lazy_store();
    assert_eq!(opened.get_mut(&"own".to_string()).copied(), Some(2));
    let generation = environment::generation();

    active
        .merge(EnvironmentOrigin::Fork(source.clone()))
        .unwrap();

    assert_eq!(environment::generation(), generation);
    assert_eq!(EnvironmentId::active(), active);
    assert_eq!(opened.get_mut(&"kernel".to_string()).copied(), Some(1));
    assert_eq!(opened.get_mut(&"own".to_string()).copied(), Some(2));
    assert_eq!(read(&source, "own"), None);
}

/// Merging nothing, or an environment into itself, changes nothing and
/// switches nothing.
#[test]
#[serial_test::serial]
fn a_merge_that_brings_nothing_changes_nothing() {
    let _root = rooted();
    let id = EnvironmentId::create("itself", EnvironmentOrigin::Empty).unwrap();
    warm(&id, &[("kernel", 1)]);
    let generation = environment::generation();

    id.merge(EnvironmentOrigin::Fork(id.clone())).unwrap();
    id.merge(EnvironmentOrigin::Empty).unwrap();

    assert_eq!(environment::generation(), generation);
    assert_eq!(store().get(&"kernel".to_string()).copied(), Some(1));
}

/// A merged entry is imported: a value computed here replaces it, so a stale
/// merged environment never wedges the application it was merged into.
#[test]
#[serial_test::serial]
fn a_value_computed_here_replaces_a_merged_one() {
    let _root = rooted();
    let built = EnvironmentId::create("built", EnvironmentOrigin::Empty).unwrap();
    warm(&built, &[("kernel", 1)]);
    let application = EnvironmentId::create("application", EnvironmentOrigin::Empty).unwrap();
    application.activate().unwrap();

    application.merge(EnvironmentOrigin::Fork(built)).unwrap();
    lazy_store().insert("kernel".to_string(), 2).unwrap();

    assert_eq!(read(&application, "kernel"), Some(2));
}

/// Two merges into one environment at once both land: each stages its bundle
/// in a file of its own.
#[test]
#[serial_test::serial]
fn concurrent_merges_into_one_environment_both_land() {
    let _root = rooted();
    let mut bundles = Vec::new();
    for (name, key) in [("first", "first-kernel"), ("second", "second-kernel")] {
        let built = EnvironmentId::create(name, EnvironmentOrigin::Empty).unwrap();
        let entries: Vec<(String, u32)> = (0..200).map(|i| (format!("{key}-{i}"), i)).collect();
        let entries: Vec<(&str, u32)> = entries.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        warm(&built, &entries);
        bundles.push(built.save().unwrap());
    }
    environment::activate(environment::DEFAULT);
    let application = EnvironmentId::create("application", EnvironmentOrigin::Empty).unwrap();

    std::thread::scope(|scope| {
        for bundle in bundles {
            let application = &application;
            scope.spawn(move || {
                application
                    .merge(EnvironmentOrigin::Bundle(bundle))
                    .unwrap()
            });
        }
    });

    assert_eq!(read(&application, "first-kernel-199"), Some(199));
    assert_eq!(read(&application, "second-kernel-199"), Some(199));
}

/// Replacing the active environment would pull entries from under the stores
/// reading it.
#[test]
#[serial_test::serial]
fn the_active_environment_cannot_be_replaced() {
    let _root = rooted();
    let active = EnvironmentId::create("busy", EnvironmentOrigin::Empty).unwrap();
    active.activate().unwrap();

    assert!(matches!(
        EnvironmentId::create("busy", EnvironmentOrigin::Empty),
        Err(EnvironmentError::Active { .. })
    ));
}

/// An environment another process keeps — a server's — is refused before
/// anything is read, written or activated.
#[test]
#[serial_test::serial]
fn an_environment_of_another_storage_is_refused() {
    let _root = rooted();
    let local = EnvironmentId::active();
    let foreign = EnvironmentId {
        storage: local.storage.wrapping_add(1),
        name: "foreign".into(),
        ..local.clone()
    };

    let incompatible = |result: Result<(), EnvironmentError>| {
        matches!(result, Err(EnvironmentError::IncompatibleStorage { .. }))
    };
    assert!(incompatible(foreign.activate().map(|_| ())));
    assert!(incompatible(foreign.save().map(|_| ())));
    assert!(incompatible(
        EnvironmentId::create("forked", EnvironmentOrigin::Fork(foreign)).map(|_| ())
    ));
    assert_eq!(EnvironmentId::active(), local);
}

/// A root nobody wrote to yet is created on the first environment loaded in it.
#[test]
#[serial_test::serial]
fn a_bundle_loads_into_a_fresh_root() {
    let root = rooted();
    let built = EnvironmentId::create("built", EnvironmentOrigin::Empty).unwrap();
    warm(&built, &[("kernel", 1)]);
    environment::activate(environment::DEFAULT);
    let bundle = built.save().unwrap();

    environment::set_root(root.path().join("fresh"));
    let loaded = EnvironmentId::create("loaded", EnvironmentOrigin::Bundle(bundle)).unwrap();

    assert_eq!(read(&loaded, "kernel"), Some(1));
}

/// Forking an environment into itself, under any spelling of its name, is
/// refused before it is emptied.
#[test]
#[serial_test::serial]
fn an_environment_cannot_be_forked_into_itself() {
    let _root = rooted();
    let source = EnvironmentId::create("same", EnvironmentOrigin::Empty).unwrap();
    warm(&source, &[("kept", 1)]);
    environment::activate(environment::DEFAULT);

    let alias = EnvironmentId {
        name: "alias".into(),
        ..source.clone()
    };
    assert!(matches!(
        EnvironmentId::create("same", EnvironmentOrigin::Fork(alias)),
        Err(EnvironmentError::SelfFork { .. })
    ));

    assert_eq!(read(&source, "kept"), Some(1));
}

/// A bundle that fails to read, before or after its manifest, leaves the
/// environment it would replace as it was.
#[test]
#[serial_test::serial]
fn an_unreadable_bundle_leaves_the_environment_as_it_was() {
    let _root = rooted();
    let built = EnvironmentId::create("built", EnvironmentOrigin::Empty).unwrap();
    let entries: Vec<(String, u32)> = (0..2000).map(|i| (format!("kernel-{i:04}"), i)).collect();
    let entries: Vec<(&str, u32)> = entries.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    warm(&built, &entries);
    let kept = EnvironmentId::create("kept", EnvironmentOrigin::Empty).unwrap();
    warm(&kept, &[("kept", 1)]);
    environment::activate(environment::DEFAULT);

    let bytes = built.save().unwrap().bytes;
    for corrupted in [&bytes[..bytes.len() / 2], &b"not a bundle"[..]] {
        let bundle = EnvironmentBundle {
            bytes: Bytes::from_bytes_vec(corrupted.to_vec()),
        };
        assert!(EnvironmentId::create("kept", EnvironmentOrigin::Bundle(bundle)).is_err());
        assert_eq!(read(&kept, "kept"), Some(1));
        environment::activate(environment::DEFAULT);
    }
}

/// An id keeps naming its environment when the root changes, and the id of
/// a mounted file names that file.
#[test]
#[serial_test::serial]
fn an_id_keeps_its_environment() {
    let root = rooted();
    let first = EnvironmentId::create("first", EnvironmentOrigin::Empty).unwrap();
    warm(&first, &[("first", 1)]);
    let bundle = first.save().unwrap();

    environment::set_root(root.path().join("other"));
    assert_eq!(read(&first, "first"), Some(1));

    let file = root.path().join("mounted.db");
    std::fs::write(&file, &*bundle.bytes).unwrap();
    environment::load(&file);
    let mounted = EnvironmentId::active();
    assert_eq!(mounted.name, "mounted");
    environment::activate(environment::DEFAULT);
    assert_eq!(read(&mounted, "first"), Some(1));
}
