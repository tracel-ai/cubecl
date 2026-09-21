//! The report model over small databases written the way cubecl writes them:
//! no device, no runtime, only the stored bytes.

use cubecl_environment::bundle::{BundleManifest, EnvironmentInfo, MANIFEST_SCHEMA};
use cubecl_environment::persistence::{Database, Origin};
use cubecl_environment::records::{self, MarkRecord, Session, Stamp, Stamped};
use cubecl_inspect::report::{CandidateOutcome, CandidateReport, KeyId, KeyOrder};
use cubecl_inspect::{InspectError, Inspector};
use cubecl_server::benchmark::BenchmarkComputations;
use cubecl_server::compiler::{CompilationOutcome, CompilationRecord, KernelCacheKey};
use cubecl_server::memory_management::{
    MemoryPoolKind, MemoryPoolReport, MemoryRecord, MemoryReport, MemoryUsage,
};
use cubecl_server::tune::{
    AutotuneError, AutotuneOutcome, AutotuneResult, PersistentCacheValue, Trial, TuneRecord,
};
use std::path::{Path, PathBuf};
use std::time::Duration;

const GEMM: &str = "autotune/0.11.0/hip-0/matmul-tune-gemm";
const NORM: &str = "autotune/0.11.0/hip-0/rms-norm";
const MATMUL: u128 = 0xabcd_ef01_2345_6789_abcd_ef01_2345_6789;

/// A tuner's key: what cubecl's `PersistentCacheKey` serializes to.
#[derive(serde::Serialize)]
struct StoredKey<K> {
    key: K,
    checksum: &'static str,
}

#[derive(serde::Serialize, Clone)]
struct GemmKey {
    m: u32,
    elem: Elem,
}

#[derive(serde::Serialize, Clone)]
enum Elem {
    Float(&'static str),
}

fn cbor(value: &impl serde::Serialize) -> Vec<u8> {
    let mut bytes = Vec::new();
    ciborium::into_writer(value, &mut bytes).expect("encodes");
    bytes
}

fn measured(name: &str, index: usize, micros: u64) -> AutotuneResult {
    let micros = Duration::from_micros(micros);
    AutotuneResult {
        outcome: Ok(AutotuneOutcome {
            name: name.to_string(),
            index,
            computation: BenchmarkComputations {
                mean: micros,
                median: micros,
                variance: Duration::ZERO,
                min: micros,
                max: micros,
            },
        }),
    }
}

/// Writes `record` as cubecl does: under its kind, keyed by its stamp, in
/// the fixture's session 7.
fn insert_record<V: serde::Serialize>(
    database: &Database,
    kind: &str,
    seq: u64,
    offset_ms: u64,
    record: V,
) {
    let stamped = Stamped {
        stamp: Stamp {
            session: 7,
            seq,
            offset: Duration::from_millis(offset_ms),
        },
        record,
    };
    database.insert(
        &records::namespace(kind),
        &cbor(&(7u64, seq)),
        &cbor(&stamped),
        Origin::Local,
    );
}

fn failed(error: AutotuneError) -> AutotuneResult {
    AutotuneResult {
        outcome: Err(error),
    }
}

/// A database holding two gemm keys and one norm key, ranked the way the
/// tuner stores them: best score first.
fn fixture(dir: &Path) -> PathBuf {
    let path = dir.join("fixture.cubecl");
    let database = Database::open(&path, false).expect("creates");
    let insert = |namespace: &str, key: Vec<u8>, value: PersistentCacheValue| {
        database.insert(namespace, &key, &cbor(&value), Origin::Local);
    };

    insert(
        GEMM,
        cbor(&StoredKey {
            key: GemmKey {
                m: 64,
                elem: Elem::Float("F16"),
            },
            checksum: "list-a",
        }),
        PersistentCacheValue {
            fastest_index: 1,
            results: vec![
                measured("tiled", 1, 10),
                measured("naive", 0, 40),
                failed(AutotuneError::Unknown {
                    name: "packed".to_string(),
                    err: "not a packed scheme".to_string(),
                }),
                failed(AutotuneError::Skip {
                    name: "floor".to_string(),
                }),
            ],
            bounds: None,
            limit: None,
        },
    );
    insert(
        GEMM,
        cbor(&StoredKey {
            key: GemmKey {
                m: 128,
                elem: Elem::Float("F16"),
            },
            checksum: "list-a",
        }),
        PersistentCacheValue {
            fastest_index: 1,
            results: vec![measured("tiled", 1, 20), measured("naive", 0, 22)],
            bounds: None,
            limit: None,
        },
    );
    insert(
        NORM,
        cbor(&StoredKey {
            key: 8u32,
            checksum: "list-b",
        }),
        PersistentCacheValue {
            fastest_index: 0,
            results: vec![measured("row", 0, 5)],
            bounds: None,
            limit: None,
        },
    );
    // The build that tuned the m=64 key recorded it: the winner first, then
    // the candidate that lost by 4x, then the one that failed.
    let session = Session {
        id: 7,
        started_unix_ms: 1_789_488_000_000,
        cubecl_version: "0.11.0".to_string(),
        label: Some("models build fixture".to_string()),
        process: 1,
        os: "linux".to_string(),
        arch: "x86_64".to_string(),
    };
    database.insert(
        records::SESSIONS,
        &cbor(&session.id),
        &cbor(&session),
        Origin::Local,
    );
    let trial = |name: &str, millis| Trial {
        name: name.to_string(),
        wall: Duration::from_millis(millis),
    };
    let traced = Stamped {
        stamp: Stamp {
            session: 7,
            seq: 3,
            offset: Duration::from_secs(2),
        },
        record: TuneRecord {
            table: GEMM.to_string(),
            key: GemmKey {
                m: 64,
                elem: Elem::Float("F16"),
            },
            checksum: "list-a".to_string(),
            winner: 1,
            trials: vec![
                trial("tiled", 100),
                trial("naive", 300),
                trial("packed", 50),
            ],
            short_circuit: None,
            wall: Duration::from_millis(500),
            dry_run: true,
        },
    };
    database.insert(
        &records::namespace(TuneRecord::<()>::KIND),
        &cbor(&(7u64, 3u64)),
        &cbor(&traced),
        Origin::Local,
    );

    // The same build compiled `Matmul` inside that tune, then loaded it again
    // from the store after it; `Norm` was compiled after, and never stored.
    let compilation = |seq: u64, offset_ms, kernel: &str, id, outcome| {
        let stamped = Stamped {
            stamp: Stamp {
                session: 7,
                seq,
                offset: Duration::from_millis(offset_ms),
            },
            record: CompilationRecord {
                kernel: kernel.to_string(),
                ir: Some(format!("{kernel} ir")),
                key: KernelCacheKey { id, build_id: 1 },
                outcome,
                source: None,
            },
        };
        database.insert(
            &records::namespace(CompilationRecord::KIND),
            &cbor(&(7u64, seq)),
            &cbor(&stamped),
            Origin::Local,
        );
    };
    compilation(
        4,
        2_100,
        "kernels::Matmul<f16>",
        MATMUL,
        CompilationOutcome::Compiled {
            duration: Duration::from_millis(80),
        },
    );
    compilation(
        5,
        3_000,
        "kernels::Matmul<f16>",
        MATMUL,
        CompilationOutcome::Loaded {
            duration: Duration::from_millis(5),
        },
    );
    compilation(
        6,
        3_100,
        "kernels::Norm",
        0x2222,
        CompilationOutcome::Compiled {
            duration: Duration::from_millis(20),
        },
    );
    database.insert(
        "hip/0.11.0/gfx1151",
        &cbor(&KernelCacheKey {
            id: MATMUL,
            build_id: 1,
        }),
        &[0; 64],
        Origin::Local,
    );

    // The build marked its walk, and inside it the turn that ran the tune;
    // after the walk, it snapshot its pools.
    let mark = |label: &str, wall_ms| MarkRecord {
        label: label.to_string(),
        wall: Duration::from_millis(wall_ms),
    };
    insert_record(&database, MarkRecord::KIND, 1, 1_000, mark("walk", 3_000));
    insert_record(&database, MarkRecord::KIND, 2, 1_900, mark("turn 0", 700));
    let pool = MemoryPoolReport {
        kind: MemoryPoolKind::Persistent,
        usage: MemoryUsage {
            number_allocs: 2,
            bytes_in_use: 1000,
            bytes_padding: 100,
            bytes_reserved: 4096,
        },
        pages: 1,
        pages_peak: 2,
        pages_unmapped: 0,
        largest_alloc: 800,
    };
    insert_record(
        &database,
        MemoryRecord::KIND,
        7,
        3_050,
        MemoryRecord {
            label: "walked".to_string(),
            report: MemoryReport {
                dynamic: vec![pool.clone()],
                persistent: pool,
            },
        },
    );

    database.insert(NORM, b"not cbor", b"not cbor", Origin::Local);
    BundleManifest {
        schema: MANIFEST_SCHEMA,
        name: "fixture".to_string(),
        cubecl_version: "0.11.0".to_string(),
        created_unix_secs: Some(1_789_488_372),
        environments: vec![EnvironmentInfo {
            devices: vec!["hip-gfx1151".to_string()],
            ..Default::default()
        }],
    }
    .write(&database)
    .expect("writes the manifest");
    path
}

#[test]
fn the_summary_totals_every_namespace_by_root() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let inspector = Inspector::open(fixture(dir.path())).expect("opens");
    let summary = inspector.summary();

    assert_eq!(summary.entries(), 13);
    assert_eq!(summary.autotune_keys(), 4);
    let roots: Vec<(String, u64)> = summary
        .roots()
        .into_iter()
        .map(|root| (root.namespace, root.entries))
        .collect();
    assert_eq!(
        roots,
        vec![
            ("autotune".to_string(), 4),
            ("hip".to_string(), 1),
            ("records".to_string(), 8)
        ]
    );
    let manifest = summary.manifest.expect("an exported file carries one");
    assert_eq!(manifest.created_unix_secs, Some(1_789_488_372));
}

#[test]
fn every_key_decodes_without_its_type() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let report = Inspector::open(fixture(dir.path()))
        .expect("opens")
        .autotune();

    assert_eq!(report.keys.len(), 3);
    assert_eq!(report.undecoded, 1, "the entry that is not cbor");

    let gemm = report
        .keys
        .iter()
        .find(|key| key.table.tuner == "matmul-tune-gemm" && key.results.len() == 4)
        .expect("the m=64 key");
    assert_eq!(
        cubecl_inspect::view::KeyText(&gemm.key).to_string(),
        "m=64 elem=Float(F16)"
    );
    assert_eq!(gemm.checksum, "list-a");
    assert_eq!(gemm.winner_name(), "tiled");
    assert_eq!(gemm.measured(), 2);
    assert_eq!(gemm.failed(), 1);
    assert_eq!(gemm.margin(), Some(4.0));
    assert!(matches!(
        &gemm.results[2].outcome,
        CandidateOutcome::Failed { reason } if reason == "not a packed scheme"
    ));
    assert!(matches!(gemm.results[3].outcome, CandidateOutcome::Skipped));
}

#[test]
fn a_key_is_found_by_its_id_and_an_unknown_id_says_so() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let inspector = Inspector::open(fixture(dir.path())).expect("opens");
    let id = inspector.autotune().keys[0].id;

    assert_eq!(inspector.autotune_key(id).expect("found").id, id);
    let unknown: KeyId = "00000000".parse().expect("parses");
    assert!(matches!(
        inspector.autotune_key(unknown),
        Err(InspectError::UnknownKey(_))
    ));
}

#[test]
fn keys_sort_by_trials_and_filter_by_tuner() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut report = Inspector::open(fixture(dir.path()))
        .expect("opens")
        .autotune();

    report.sort(KeyOrder::Margin);
    assert_eq!(report.keys[0].margin(), Some(4.0));
    report.sort(KeyOrder::Trials);
    assert_eq!(report.keys[0].measured(), 2);
    report.retain_tuners("norm");
    assert_eq!(report.keys.len(), 1);
}

/// The candidate table: `naive` raced twice and never won, running 2.3×
/// behind (the geometric mean of 4× and 1.1×); the table puts it first.
#[test]
fn candidates_that_never_win_come_first() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let report = Inspector::open(fixture(dir.path()))
        .expect("opens")
        .autotune();
    let candidates = CandidateReport::from(&report).candidates;

    let gemm: Vec<&str> = candidates
        .iter()
        .filter(|row| row.tuner == "matmul-tune-gemm")
        .map(|row| row.candidate.as_str())
        .collect();
    // Among the ones that never win, costliest first: `packed` failed but
    // its trial was recorded, `floor` never ran.
    assert_eq!(gemm, vec!["naive", "packed", "floor", "tiled"]);

    let naive = &candidates[0];
    assert_eq!((naive.measured, naive.won), (2, 0));
    let slowdown = naive.slowdown.expect("measured against the winner");
    assert!(
        (slowdown - (4.0f64 * 1.1).sqrt()).abs() < 1e-3,
        "{slowdown}"
    );

    let tiled = candidates
        .iter()
        .find(|row| row.candidate == "tiled")
        .expect("tiled");
    assert_eq!((tiled.won, tiled.slowdown), (2, Some(1.0)));
}

#[test]
fn a_listing_reads_every_file_and_names_the_unreadable() {
    let dir = tempfile::tempdir().expect("a temp dir");
    fixture(dir.path());
    std::fs::write(dir.path().join("broken.cubecl"), b"not a database").expect("writes");
    std::fs::write(dir.path().join("notes.txt"), b"ignored").expect("writes");

    let listing = Inspector::list(dir.path()).expect("lists");
    assert_eq!(listing.environments.len(), 1);
    assert_eq!(listing.unreadable.len(), 1);
}

#[test]
fn a_missing_file_does_not_open() {
    let dir = tempfile::tempdir().expect("a temp dir");
    assert!(matches!(
        Inspector::open(dir.path().join("missing.cubecl")),
        Err(InspectError::Open { .. })
    ));
}

#[test]
fn a_recorded_tune_joins_its_key_in_the_order_it_ran() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let report = Inspector::open(fixture(dir.path()))
        .expect("opens")
        .autotune();

    let traced: Vec<_> = report
        .keys
        .iter()
        .filter(|key| key.trace.is_some())
        .collect();
    assert_eq!(traced.len(), 1, "only the m=64 key was recorded");
    let key = traced[0];
    assert_eq!(key.results.len(), 4);
    let order: Vec<&str> = key
        .trials()
        .iter()
        .map(|trial| trial.name.as_str())
        .collect();
    assert_eq!(order, vec!["tiled", "naive", "packed"]);
    assert_eq!(key.wall(), Some(Duration::from_millis(500)));
    // `naive` lost by 4x and `packed` failed: both bought nothing.
    assert_eq!(key.wasted(), Some(Duration::from_millis(350)));
}

#[test]
fn keys_sort_slowest_tune_first() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut report = Inspector::open(fixture(dir.path()))
        .expect("opens")
        .autotune();

    report.sort(KeyOrder::Wall);
    assert!(report.keys[0].trace.is_some());
    assert!(report.keys[1..].iter().all(|key| key.trace.is_none()));
}

#[test]
fn candidates_carry_the_wall_they_cost() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let report = Inspector::open(fixture(dir.path()))
        .expect("opens")
        .autotune();
    let candidates = CandidateReport::from(&report).candidates;

    let wall = |name: &str| {
        candidates
            .iter()
            .find(|row| row.candidate == name)
            .and_then(|row| row.wall)
    };
    assert_eq!(wall("naive"), Some(Duration::from_millis(300)));
    assert_eq!(wall("floor"), None, "never ran");
}

#[test]
fn the_summary_folds_each_session_s_tunes() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let summary = Inspector::open(fixture(dir.path()))
        .expect("opens")
        .summary();

    assert_eq!(summary.sessions.len(), 1);
    let session = &summary.sessions[0];
    assert_eq!(
        session.session.label.as_deref(),
        Some("models build fixture")
    );
    assert_eq!(session.tunes, 1);
    assert_eq!(session.tuning, Duration::from_millis(500));
    assert_eq!(session.compilations, 3);
    assert_eq!(session.compiling, Duration::from_millis(105));
    assert_eq!(session.compiling_in_tunes, Duration::from_millis(80));
    // The last record ends at 3.1 s + 20 ms.
    assert_eq!(session.span, Duration::from_millis(3120));
    assert_eq!(session.other(), Duration::from_millis(3120 - 500 - 25));
}

#[test]
fn a_stripped_copy_keeps_the_caches_and_drops_the_records() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let inspector = Inspector::open(fixture(dir.path())).expect("opens");
    let stripped = inspector
        .strip(&dir.path().join("stripped.cubecl"))
        .expect("strips");

    assert!(stripped.sessions.is_empty());
    assert!(
        stripped
            .namespaces
            .iter()
            .all(|row| row.root() != "records")
    );
    assert_eq!(
        stripped.autotune_keys(),
        inspector.summary().autotune_keys()
    );
    assert_eq!(
        stripped.manifest.expect("an export carries one").name,
        "fixture"
    );
}

#[test]
fn a_tune_owns_the_compilations_that_started_inside_it() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let report = Inspector::open(fixture(dir.path()))
        .expect("opens")
        .autotune();

    let traced = report
        .keys
        .iter()
        .find(|key| key.trace.is_some())
        .expect("the recorded key");
    assert_eq!(traced.compiling, Some(Duration::from_millis(80)));
}

#[test]
fn kernels_fold_their_trips_and_carry_their_stored_size() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let inspector = Inspector::open(fixture(dir.path())).expect("opens");
    let report = inspector.kernels();

    assert_eq!(report.kernels.len(), 2);
    let matmul = &report.kernels[0];
    assert_eq!(matmul.short_name(), "Matmul");
    assert_eq!((matmul.compiled, matmul.loaded), (1, 1));
    assert_eq!(matmul.compiling, Duration::from_millis(80));
    assert_eq!(matmul.bytes, Some(64));
    assert_eq!(matmul.ir.as_deref(), Some("kernels::Matmul<f16> ir"));
    assert_eq!(report.kernels[1].bytes, None, "never stored");
    assert_eq!(report.unrecorded, 0);
    assert_eq!(report.families[0].short_name(), "Matmul");

    assert_eq!(inspector.kernel("abcdef").expect("by prefix").id, matmul.id);
    assert!(matches!(
        inspector.kernel("ffff"),
        Err(InspectError::UnknownKernel(_))
    ));
    assert!(matches!(
        inspector.kernel(""),
        Err(InspectError::AmbiguousKernel { count: 2, .. })
    ));
}

#[test]
fn a_timeline_nests_marks_and_counts_what_ran_inside_them() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let timeline = Inspector::open(fixture(dir.path()))
        .expect("opens")
        .timeline();

    assert_eq!(timeline.sessions.len(), 1);
    let spans = &timeline.sessions[0].spans;
    let shape: Vec<(&str, usize)> = spans
        .iter()
        .map(|span| (span.label.as_str(), span.depth))
        .collect();
    assert_eq!(shape, vec![("walk", 0), ("turn 0", 1)]);

    let (walk, turn) = (&spans[0], &spans[1]);
    assert_eq!((walk.tunes, walk.compilations), (1, 3));
    assert_eq!(walk.compiling, Duration::from_millis(105));
    assert_eq!((turn.tunes, turn.compilations), (1, 1));
    assert_eq!(turn.compiling, Duration::from_millis(80));
    let slowest = turn.slowest.as_ref().expect("the tune inside it");
    assert_eq!(slowest.wall, Duration::from_millis(500));
    assert_eq!(slowest.table, GEMM);
}

#[test]
fn memory_snapshots_read_back_in_order() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let memory = Inspector::open(fixture(dir.path()))
        .expect("opens")
        .memory();

    assert_eq!(memory.snapshots.len(), 1);
    let snapshot = &memory.snapshots[0].record;
    assert_eq!(snapshot.label, "walked");
    assert_eq!(snapshot.report.persistent.pages_peak, 2);
}

/// A second build of the fixture: the m=128 key answered by the other
/// candidate, the norm key gone, and a key the first never tuned.
fn rebuilt(dir: &Path) -> PathBuf {
    let path = dir.join("rebuilt");
    std::fs::create_dir_all(&path).expect("a dir");
    let path = fixture(&path);
    let database = Database::open(&path, false).expect("opens");
    let key = |m| {
        cbor(&StoredKey {
            key: GemmKey {
                m,
                elem: Elem::Float("F16"),
            },
            checksum: "list-a",
        })
    };
    database.replace(
        GEMM,
        &key(128),
        &cbor(&PersistentCacheValue {
            fastest_index: 0,
            results: vec![measured("naive", 0, 20), measured("tiled", 1, 21)],
            bounds: None,
            limit: None,
        }),
        Origin::Local,
    );
    database.purge(NORM);
    database.insert(
        GEMM,
        &key(256),
        &cbor(&PersistentCacheValue {
            fastest_index: 1,
            results: vec![measured("tiled", 1, 30)],
            bounds: None,
            limit: None,
        }),
        Origin::Local,
    );
    path
}

#[test]
fn a_diff_names_what_was_added_removed_and_answered_differently() {
    use cubecl_inspect::report::KeyChange;

    let dir = tempfile::tempdir().expect("a temp dir");
    let before = Inspector::open(fixture(dir.path())).expect("opens");
    let after = Inspector::open(rebuilt(dir.path())).expect("opens");
    let diff = before.diff(&after);

    let count = |change| diff.with(change).count();
    assert_eq!(count(KeyChange::Added), 1);
    assert_eq!(count(KeyChange::Removed), 1);
    assert_eq!(count(KeyChange::WinnerChanged), 1);
    assert_eq!(count(KeyChange::Kept), 1);

    let changed = diff.with(KeyChange::WinnerChanged).next().expect("one");
    assert_eq!(
        changed.before.as_ref().map(|a| a.winner.as_str()),
        Some("tiled")
    );
    assert_eq!(
        changed.after.as_ref().map(|a| a.winner.as_str()),
        Some("naive")
    );
    assert!(!changed.candidates_changed(), "the same list: variance");
    assert_eq!(
        cubecl_inspect::view::KeyText(&changed.key).to_string(),
        "m=128 elem=Float(F16)"
    );
}

#[test]
fn pruning_keeps_the_newest_sessions_records() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = fixture(dir.path());
    {
        let database = Database::open(&path, false).expect("opens");
        let later = Session {
            id: 9,
            started_unix_ms: 1_789_489_000_000,
            cubecl_version: "0.11.0".to_string(),
            label: None,
            process: 2,
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
        };
        database.insert(
            records::SESSIONS,
            &cbor(&later.id),
            &cbor(&later),
            Origin::Local,
        );
    }
    let inspector = Inspector::open(&path).expect("opens");
    let summary = inspector.prune(1).expect("prunes");

    let sessions: Vec<u64> = summary.sessions.iter().map(|row| row.session.id).collect();
    assert_eq!(sessions, vec![9]);
    assert!(
        inspector
            .autotune()
            .keys
            .iter()
            .all(|key| key.trace.is_none())
    );
    assert!(inspector.kernels().kernels.is_empty());
    assert_eq!(summary.autotune_keys(), 4, "the caches are untouched");
}

/// A compact copy keeps the kernels the replay launched, their second-line
/// entries with them, the autotune answers and the application's entries —
/// and nothing else.
#[test]
fn compacting_keeps_what_the_replay_launched() {
    use cubecl_environment::collections::HashSet;

    let dir = tempfile::tempdir().expect("a temp dir");
    let path = fixture(dir.path());
    {
        let database = Database::open(&path, false).expect("opens");
        let unused = KernelCacheKey {
            id: 0x9999,
            build_id: 1,
        };
        database.insert(
            "hip/0.11.0/gfx1151",
            &cbor(&unused),
            &[0; 32],
            Origin::Local,
        );
        // A second-line entry of each: a source hash naming the artifact.
        database.insert(
            "hip-second-line/0.11.0/gfx1151",
            &cbor(&1u128),
            &cbor(&KernelCacheKey {
                id: MATMUL,
                build_id: 1,
            }),
            Origin::Local,
        );
        database.insert(
            "hip-second-line/0.11.0/gfx1151",
            &cbor(&2u128),
            &cbor(&unused),
            Origin::Local,
        );
    }
    let inspector = Inspector::open(&path).expect("opens");
    let out = dir.path().join("compact.cubecl");

    // This build's copy of the matmul artifact beside the fixture's, which an
    // older build compiled: only this build's is kept.
    {
        let database = Database::open(&path, false).expect("opens");
        let current = KernelCacheKey {
            id: MATMUL,
            build_id: cubecl_server::compiler::build_id_hash(),
        };
        database.insert(
            "hip/0.11.0/gfx1151",
            &cbor(&current),
            &[0; 48],
            Origin::Local,
        );
        database.insert(
            "hip-second-line/0.11.0/gfx1151",
            &cbor(&3u128),
            &cbor(&current),
            Origin::Local,
        );
    }
    let launched: HashSet<u128> = [MATMUL, 0x7777].into_iter().collect();
    let compaction = inspector.compact(&out, &launched).expect("compacts");

    assert_eq!(
        (compaction.kept_kernels, compaction.dropped_kernels),
        (1, 2)
    );
    assert_eq!(
        compaction.unstored, 1,
        "0x7777 was launched but never stored"
    );
    let summary = &compaction.summary;
    assert!(summary.sessions.is_empty());
    assert!(summary.namespaces.iter().all(|row| row.root() != "records"));
    assert_eq!(summary.autotune_keys(), 4);
    let count = |root: &str| {
        summary
            .namespaces
            .iter()
            .filter(|row| row.root() == root)
            .map(|row| row.entries)
            .sum::<u64>()
    };
    // This build's matmul artifact stays; the older build's and the unused
    // one go, each with its second-line entry.
    assert_eq!(count("hip"), 1);
    assert_eq!(count("hip-second-line"), 1);
}

#[test]
fn compacting_without_a_replay_is_refused() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let inspector = Inspector::open(fixture(dir.path())).expect("opens");
    let launched = Default::default();
    assert!(matches!(
        inspector.compact(&dir.path().join("compact.cubecl"), &launched),
        Err(InspectError::Export { .. })
    ));
}
