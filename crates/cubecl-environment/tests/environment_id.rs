#![cfg(all(feature = "persistence", not(target_family = "wasm")))]

use cubecl_environment::environment::{self, EnvironmentError, EnvironmentId, EnvironmentOrigin};
use cubecl_environment::persistence::{Namespace, Store, StoreOptions};

// The active environment is process-global, so these tests are serialized.

fn store() -> Store<String, u32> {
    Store::new(StoreOptions::new().storage(Namespace::scoped("environment-id", "entries")))
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
