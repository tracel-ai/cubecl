#![cfg(all(feature = "persistence", not(target_family = "wasm")))]

use cubecl_environment::bytes::Bytes;
use cubecl_environment::environment::{
    self, AutotuneAnswer, BuildStamp, EnvironmentBundle, EnvironmentError, EnvironmentId,
    EnvironmentOrigin, EnvironmentSummary, SaveOptions, UsedAnswersName,
};
use cubecl_environment::persistence::{
    CacheOption, Database, Namespace, Store, StoreOptions, TuneTableName,
};

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

    let bundle = built.save(SaveOptions::default()).unwrap();
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
        .merge(EnvironmentOrigin::Bundle(
            built.save(SaveOptions::default()).unwrap(),
        ))
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
        bundles.push(built.save(SaveOptions::default()).unwrap());
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
    assert!(incompatible(
        foreign.save(SaveOptions::default()).map(|_| ())
    ));
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
    let bundle = built.save(SaveOptions::default()).unwrap();

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

    let bytes = built.save(SaveOptions::default()).unwrap().bytes;
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
    let bundle = first.save(SaveOptions::default()).unwrap();

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

/// A saved bundle says what it is without being loaded: the name it was saved
/// under, when and by which executable, and what it holds, whether it is read
/// from the file it was written to or held in memory.
#[test]
#[serial_test::serial]
fn a_saved_bundle_reads_back_its_name_stamp_and_summary() {
    let root = rooted();
    let built = EnvironmentId::create("built", EnvironmentOrigin::Empty).unwrap();
    plant(&built);
    environment::activate(environment::DEFAULT);
    // The manifest keeps whole seconds.
    let before = std::time::SystemTime::now() - std::time::Duration::from_secs(1);

    let saved = built.save(SaveOptions::new().name("H100 Linux")).unwrap();
    let file = root.path().join("saved.cubecl");
    std::fs::write(&file, &*saved.bytes).unwrap();
    let opened = EnvironmentBundle::open(&file).unwrap();

    for bundle in [&saved, &opened] {
        let description = bundle.read_description().unwrap();
        assert_eq!(description.name, "H100 Linux");
        assert_eq!(description.saved_by, Some(BuildStamp::current()));
        assert_eq!(description.summary, PLANTED);
        let saved_at = description.saved_at.unwrap();
        assert!(before <= saved_at && saved_at <= std::time::SystemTime::now());
    }
    assert_eq!(&*opened.bytes, &*saved.bytes);
}

/// A saved bundle gives back the answers the work of its build used, trimmed
/// to their picks: not the ones nothing used, nor those only another build
/// did, nor a use whose answer is gone. Whether it is read from the file it
/// was written to or held in memory.
#[test]
#[serial_test::serial]
fn a_saved_bundle_gives_back_the_answers_its_build_used() {
    let root = rooted();
    let built = EnvironmentId::create("built", EnvironmentOrigin::Empty).unwrap();
    built.activate().unwrap();
    let (this_build, other_build) = builds();
    for (key, fastest_index) in [
        ("loaded", 1),
        ("measured", 2),
        ("idle", 3),
        ("elsewhere", 4),
    ] {
        answers()
            .insert(key.into(), measured_picking(fastest_index))
            .unwrap();
    }
    for (build, key) in [
        (this_build, "loaded"),
        (this_build, "measured"),
        (this_build, "gone"),
        (other_build, "elsewhere"),
    ] {
        uses(build).insert(key.into(), ()).unwrap();
    }
    environment::activate(environment::DEFAULT);

    let saved = built.save(SaveOptions::default()).unwrap();
    let file = root.path().join("saved.cubecl");
    std::fs::write(&file, &*saved.bytes).unwrap();

    for bundle in [saved, EnvironmentBundle::open(&file).unwrap()] {
        let mut answers = bundle.read_used_answers().unwrap();
        answers.sort_by_key(|answer| answer.fastest_index);
        assert_eq!(answers, [trimmed("loaded", 1), trimmed("measured", 2)]);
    }
}

/// An environment created from answers holds them where this version's tuners
/// read them, and nothing an environment of its name held before.
#[test]
#[serial_test::serial]
fn an_environment_created_from_answers_holds_them() {
    let _root = rooted();
    let stale = EnvironmentId::create("tuned", EnvironmentOrigin::Empty).unwrap();
    warm(&stale, &[("stale", 2)]);
    environment::activate(environment::DEFAULT);

    let tuned = EnvironmentId::create(
        "tuned",
        EnvironmentOrigin::Answers(vec![trimmed("first", 1), trimmed("second", 2)]),
    )
    .unwrap();

    assert_eq!(tuned.summary().unwrap().autotune_answers, 2);
    assert_eq!(read(&tuned, "stale"), None);
    assert_eq!(pick("first"), Some(1));
    assert_eq!(pick("second"), Some(2));
}

/// Answers of a table this version's tuners never read are left out: those of
/// another cubecl version, of another layout, and of a namespace that is no
/// table at all.
#[test]
#[serial_test::serial]
fn answers_of_a_table_this_version_does_not_read_are_left_out() {
    let _root = rooted();
    let moved = |table: &str| AutotuneAnswer {
        table: table.into(),
        ..trimmed("moved", 1)
    };
    let version = env!("CARGO_PKG_VERSION");
    let foreign = [
        "autotune/0.1.0/layout-v1/hardware/tuner".to_string(),
        format!("autotune/{version}-other/layout-v1/hardware/tuner"),
        format!("autotune/{version}/layout-v0/hardware/tuner"),
        format!("autotune/{version}/layout-v10/hardware/tuner"),
        format!("autotune/{version}/device/tuner"),
        format!("cuda/{version}/layout-v1/hardware/tuner"),
    ];
    let mut answers: Vec<AutotuneAnswer> = foreign.iter().map(|table| moved(table)).collect();
    answers.push(trimmed("read", 2));

    let tuned = EnvironmentId::create("tuned", EnvironmentOrigin::Answers(answers)).unwrap();

    tuned.activate().unwrap();
    let database = Database::open_active().unwrap();
    assert_eq!(database.namespaces(), [table().as_str()]);
    assert_eq!(pick("read"), Some(2));
}

/// Where two answers are to one key, the first is held, as it is against what
/// a later merge brings.
#[test]
#[serial_test::serial]
fn the_first_answer_to_a_key_is_held() {
    let _root = rooted();
    let file = EnvironmentId::create("file", EnvironmentOrigin::Empty).unwrap();
    file.activate().unwrap();
    for (key, fastest_index) in [("shared", 1), ("own", 2)] {
        answers()
            .insert(key.into(), measured_picking(fastest_index))
            .unwrap();
    }
    environment::activate(environment::DEFAULT);

    let tuned = EnvironmentId::create(
        "tuned",
        EnvironmentOrigin::Answers(vec![trimmed("shared", 3), trimmed("shared", 4)]),
    )
    .unwrap();
    tuned.merge(EnvironmentOrigin::Fork(file)).unwrap();

    tuned.activate().unwrap();
    assert_eq!(pick("shared"), Some(3));
    assert_eq!(pick("own"), Some(2));
}

/// A merge of answers adds those the environment lacks, and a value computed
/// here replaces one it brought.
#[test]
#[serial_test::serial]
fn merged_answers_never_replace_one_the_environment_held() {
    let _root = rooted();
    let tuned = EnvironmentId::create("tuned", EnvironmentOrigin::Empty).unwrap();
    tuned.activate().unwrap();
    answers()
        .insert("held".into(), measured_picking(1))
        .unwrap();

    tuned
        .merge(EnvironmentOrigin::Answers(vec![
            trimmed("held", 2),
            trimmed("lacked", 3),
        ]))
        .unwrap();

    assert_eq!(pick("held"), Some(1));
    assert_eq!(pick("lacked"), Some(3));
    answers()
        .insert("lacked".into(), measured_picking(4))
        .unwrap();
    assert_eq!(pick("lacked"), Some(4));
}

/// A bundle file that is not there fails to open, as every other environment
/// call fails, rather than at its first use.
#[test]
fn a_missing_bundle_file_fails_to_open() {
    let root = tempfile::tempdir().unwrap();

    let opened = EnvironmentBundle::open(root.path().join("missing.cubecl"));

    assert!(matches!(opened, Err(EnvironmentError::Storage { .. })));
}

/// A bundle saved without a name carries its environment's.
#[test]
#[serial_test::serial]
fn a_bundle_saved_without_a_name_carries_its_environments() {
    let _root = rooted();
    let built = EnvironmentId::create("built", EnvironmentOrigin::Empty).unwrap();
    environment::activate(environment::DEFAULT);

    let saved = built.save(SaveOptions::default()).unwrap();

    assert_eq!(saved.read_description().unwrap().name, "built");
}

/// An environment is counted as it stands, active or not, without saving it.
#[test]
#[serial_test::serial]
fn an_environment_is_counted_as_it_stands() {
    let _root = rooted();
    let built = EnvironmentId::create("built", EnvironmentOrigin::Empty).unwrap();
    assert_eq!(built.summary().unwrap(), EnvironmentSummary::default());

    plant(&built);
    assert_eq!(built.summary().unwrap(), PLANTED, "active");
    environment::activate(environment::DEFAULT);
    assert_eq!(built.summary().unwrap(), PLANTED, "inactive");
}

/// Bytes that are no bundle say so, rather than read as an empty one.
#[test]
#[serial_test::serial]
fn bytes_that_are_no_bundle_do_not_read() {
    let bundle = EnvironmentBundle {
        bytes: Bytes::from_bytes_vec(b"not a bundle".to_vec()),
    };

    assert!(matches!(
        bundle.read_description(),
        Err(EnvironmentError::Storage { .. })
    ));
}

/// The table the answers here are stored in.
fn table() -> Namespace {
    TuneTableName {
        hardware: "hardware",
        tuner: "tuner",
    }
    .namespace()
}

/// This build, and another.
fn builds() -> (BuildStamp, BuildStamp) {
    let build = BuildStamp::current();
    let hash = build.hash ^ 1;
    (build, BuildStamp { hash })
}

/// The answers of [`table`] in the active environment, read on a miss as a
/// tuner reads them.
fn answers() -> Store<String, Answer> {
    Store::new(
        StoreOptions::new()
            .storage(table())
            .cache(CacheOption::Lazy),
    )
}

/// The pick [`table`] holds under `key` in the active environment.
fn pick(key: &str) -> Option<usize> {
    let answer = answers().remove(&key.to_string())?;
    Some(answer.fastest_index)
}

/// The answers of [`table`] the work of `build` used.
fn uses(build: BuildStamp) -> Store<String, ()> {
    let table = table();
    let name = UsedAnswersName {
        build,
        table: table.as_str(),
    };
    Store::new(StoreOptions::new().storage(name.namespace()))
}

/// A stored answer with what it was picked from, as a table keeps it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct Answer {
    pub fastest_index: usize,
    #[serde(default)]
    pub results: Vec<String>,
}

fn measured_picking(fastest_index: usize) -> Answer {
    Answer {
        fastest_index,
        results: vec!["measured".into()],
    }
}

/// The answer [`table`] holds under `key`, without what it was picked from.
fn trimmed(key: &str, fastest_index: usize) -> AutotuneAnswer {
    let mut stored = Vec::new();
    ciborium::into_writer(key, &mut stored).unwrap();
    AutotuneAnswer {
        table: table().as_str().into(),
        key: Bytes::from_bytes_vec(stored),
        fastest_index,
    }
}

/// What [`plant`] writes, counted.
const PLANTED: EnvironmentSummary = EnvironmentSummary {
    kernels: 3,
    autotune_answers: 2,
    records: 1,
};

/// Activates `id` and writes an entry or more under every root a summary
/// tells apart: compiled kernels, a compiler's source map, autotune answers,
/// records, the work's launches, and an application's own namespace.
fn plant(id: &EnvironmentId) {
    id.activate().unwrap();
    for (root, entries) in [
        ("cuda", 2),
        ("vulkan", 1),
        ("cuda-second-line", 2),
        ("autotune", 2),
        ("records", 1),
        ("launched", 4),
        ("my-application", 5),
    ] {
        let mut store: Store<String, u32> =
            Store::new(StoreOptions::new().storage(Namespace::scoped(root, "planted")));
        for entry in 0..entries {
            store.insert(format!("{root}-{entry}"), entry).unwrap();
        }
    }
}
