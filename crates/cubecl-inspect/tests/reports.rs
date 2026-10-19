//! The report model over small databases written the way cubecl writes them:
//! no device, no runtime, only the stored bytes.

use cubecl_environment::bundle::{BundleManifest, EnvironmentInfo, MANIFEST_SCHEMA};
use cubecl_environment::persistence::{Access, Database, Origin};
use cubecl_environment::records::{self, MarkRecord, Record, Session, SessionId, Stamp, Stamped};
use cubecl_environment::stream::StreamId;
use cubecl_inspect::report::{
    CandidateOutcome, CandidateReport, Fate, KernelSelector, KeyId, KeyOrder,
};
use cubecl_inspect::{EnvironmentDirectory, InspectError, Inspector};
use cubecl_server::benchmark::BenchmarkComputations;
use cubecl_server::compiler::{
    CompilationBatchRecord, CompilationOutcome, CompilationRecord, KernelCacheKey,
};
use cubecl_server::memory_management::{
    MemoryPoolKind, MemoryPoolReport, MemoryPoolsReport, MemoryRecord, MemoryReport, MemoryUsage,
    StreamMemoryReport,
};
use cubecl_server::tune::{
    AutotuneError, AutotuneOutcome, AutotuneResult, PersistentCacheKey, PersistentCacheValue,
    Placement, PlannedCandidate, Trial, TuneRecord,
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
fn insert_record<R: Record + serde::Serialize>(
    database: &Database,
    seq: u64,
    offset_ms: u64,
    record: R,
) {
    let stamped = Stamped {
        stamp: Stamp {
            session: SessionId(7),
            seq,
            offset: Duration::from_millis(offset_ms),
        },
        record,
    };
    database.insert(
        &R::namespace(),
        &cbor(&(7u64, seq)),
        &cbor(&stamped),
        Origin::Local,
    );
}

/// A candidate of a recorded plan, in one group named `gemm` at level 1.
fn planned(name: &str, index: usize, batch: Option<usize>, priority: i8) -> PlannedCandidate {
    PlannedCandidate {
        name: name.to_string(),
        index,
        batch,
        groups: vec![Placement {
            group: "gemm".to_string(),
            ordered: false,
            group_priority: 1,
            priority,
        }],
    }
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
    let database = Database::open_at(&path, Access::ReadWrite).expect("creates");
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
    // The session that tuned the m=64 key recorded it: the winner first, then
    // the candidate that lost by 4x, then the one that failed.
    let session = Session {
        id: SessionId(7),
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
            session: SessionId(7),
            seq: 3,
            offset: Duration::from_secs(2),
        },
        record: TuneRecord {
            table: GEMM.to_string(),
            entry: PersistentCacheKey {
                key: GemmKey {
                    m: 64,
                    elem: Elem::Float("F16"),
                },
                checksum: "list-a".to_string(),
            },
            winner: 1,
            trials: vec![
                trial("tiled", 100),
                trial("naive", 300),
                trial("packed", 50),
            ],
            short_circuit: None,
            wall: Duration::from_millis(500),
            launches_discarded: true,
            stored: true,
            // The three that ran were the first batch; `wide` waited behind
            // them, and the key priced `floor` out.
            plan: vec![
                planned("tiled", 1, Some(0), 2),
                planned("naive", 0, Some(0), 1),
                planned("packed", 2, Some(0), 1),
                planned("wide", 4, Some(1), 0),
                planned("floor", 3, None, -1),
            ],
        },
    };
    database.insert(
        &TuneRecord::<()>::namespace(),
        &cbor(&(7u64, 3u64)),
        &cbor(&traced),
        Origin::Local,
    );

    // The same session compiled `Matmul` inside that tune, then loaded it again
    // from the store after it; `Norm` was compiled after, and never stored.
    let compilation =
        |seq: u64, offset_ms, kernel: &str, id, outcome: (CompilationOutcome, Duration)| {
            let stamped = Stamped {
                stamp: Stamp {
                    session: SessionId(7),
                    seq,
                    offset: Duration::from_millis(offset_ms),
                },
                record: CompilationRecord {
                    kernel: kernel.to_string(),
                    ir: Some(format!("{kernel} ir")),
                    key: KernelCacheKey { id, build_id: 1 },
                    outcome: outcome.0,
                    work: outcome.1,
                    source: None,
                },
            };
            database.insert(
                &CompilationRecord::namespace(),
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
        (CompilationOutcome::Compiled, Duration::from_millis(80)),
    );
    compilation(
        5,
        3_000,
        "kernels::Matmul<f16>",
        MATMUL,
        (CompilationOutcome::Loaded, Duration::from_millis(5)),
    );
    compilation(
        6,
        3_100,
        "kernels::Norm",
        0x2222,
        (CompilationOutcome::Compiled, Duration::from_millis(20)),
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

    // The session marked its walk, and inside it the turn that ran the tune;
    // after the walk, it snapshot its pools.
    let mark = |label: &str, wall_ms| MarkRecord {
        label: label.to_string(),
        wall: Duration::from_millis(wall_ms),
    };
    insert_record(&database, 1, 1_000, mark("walk", 3_000));
    insert_record(&database, 2, 1_900, mark("turn 0", 700));
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
        7,
        3_050,
        MemoryRecord {
            label: "walked".to_string(),
            report: MemoryReport {
                streams: vec![StreamMemoryReport {
                    stream: StreamId { value: 0 },
                    pools: MemoryPoolsReport {
                        dynamic: vec![pool.clone()],
                        persistent: pool.clone(),
                        dedicated: pool,
                    },
                    auxiliary: Vec::new(),
                }],
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
    .write_to(&database)
    .expect("writes the manifest");
    path
}

#[test]
fn a_manifest_names_the_environment_only_when_it_says_more_than_the_file() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = fixture(dir.path());
    let summary = Inspector::open(&path).expect("opens").summary();
    assert_eq!(
        summary.name(),
        None,
        "the fixture's manifest repeats its file name"
    );

    {
        let database = Database::open_at(&path, Access::ReadWrite).expect("opens");
        let mut manifest = BundleManifest::new(&database).expect("reads the manifest");
        manifest.name = "Qwen/Qwen3-0.6B f16 on rocm:0".to_string();
        manifest.write_to(&database).expect("writes the manifest");
    }
    let summary = Inspector::open(&path).expect("opens").summary();
    assert_eq!(summary.stem(), "fixture");
    assert_eq!(summary.name(), Some("Qwen/Qwen3-0.6B f16 on rocm:0"));
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

/// The fixtures' directory, their files marked the way an application's
/// saved environments are rather than the runtime's own.
fn saved(dir: &Path) -> EnvironmentDirectory {
    EnvironmentDirectory {
        path: dir.to_path_buf(),
        extension: "cubecl".to_string(),
    }
}

#[test]
fn a_listing_reads_every_file_and_names_the_unreadable() {
    let dir = tempfile::tempdir().expect("a temp dir");
    fixture(dir.path());
    std::fs::write(dir.path().join("broken.cubecl"), b"not a database").expect("writes");
    std::fs::write(dir.path().join("notes.txt"), b"ignored").expect("writes");

    let listing = Inspector::list(&[saved(dir.path())]);
    assert_eq!(listing.environments.len(), 1);
    assert_eq!(listing.unreadable.len(), 1);
}

/// One listing reads every directory it is given, so a caller with the
/// runtime's root and an application's saved files sees both at once.
#[test]
fn a_listing_spans_directories() {
    let first = tempfile::tempdir().expect("a temp dir");
    let second = tempfile::tempdir().expect("a temp dir");
    fixture(first.path());
    fixture(second.path());

    let listing = Inspector::list(&[saved(first.path()), saved(second.path())]);

    assert_eq!(listing.directories.len(), 2);
    assert_eq!(listing.environments.len(), 2);
}

/// The extension marks a file with or without its dot: `.cubecl` is not a
/// different extension from `cubecl`.
#[test]
fn an_extension_reads_with_or_without_its_dot() {
    let dir = tempfile::tempdir().expect("a temp dir");
    fixture(dir.path());
    let dotted = EnvironmentDirectory {
        path: dir.path().to_path_buf(),
        extension: ".cubecl".to_string(),
    };
    assert_eq!(Inspector::list(&[dotted]).environments.len(), 1);
}

/// A directory that cannot be read is named in the listing, and the others
/// are listed beside it: one bad directory does not hide the rest.
#[test]
fn an_unreadable_directory_is_listed_beside_the_others() {
    let dir = tempfile::tempdir().expect("a temp dir");
    fixture(dir.path());
    // A file where a directory is expected: reading it fails.
    let not_a_directory = dir.path().join("fixture.cubecl");

    let listing = Inspector::list(&[saved(&not_a_directory), saved(dir.path())]);

    assert_eq!(listing.environments.len(), 1);
    assert_eq!(listing.unreadable.len(), 1);
    assert_eq!(listing.unreadable[0].path, not_a_directory);
}

#[test]
fn a_missing_directory_lists_nothing() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let listing = Inspector::list(&[saved(&dir.path().join("never-built"))]);
    assert!(listing.environments.is_empty());
    assert!(listing.unreadable.is_empty());
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
        .filter(|key| key.tune.is_some())
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
fn a_recorded_plan_says_what_became_of_each_candidate() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let report = Inspector::open(fixture(dir.path()))
        .expect("opens")
        .autotune();
    let key = report
        .keys
        .iter()
        .find(|key| key.tune.is_some())
        .expect("the m=64 key was recorded");

    let fates: Vec<(&str, Fate)> = key
        .plan()
        .iter()
        .map(|planned| (planned.name.as_str(), key.fate(planned)))
        .collect();
    assert_eq!(
        fates,
        vec![
            ("tiled", Fate::Ran { order: 0 }),
            ("naive", Fate::Ran { order: 1 }),
            ("packed", Fate::Ran { order: 2 }),
            ("wide", Fate::NotReached),
            ("floor", Fate::Excluded),
        ]
    );
}

#[test]
fn keys_sort_slowest_tune_first() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut report = Inspector::open(fixture(dir.path()))
        .expect("opens")
        .autotune();

    report.sort(KeyOrder::Wall);
    assert!(report.keys[0].tune.is_some());
    assert!(report.keys[1..].iter().all(|key| key.tune.is_none()));
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
    // The walk ends at 1 s + 3 s, after the last compile's 3.1 s + 20 ms.
    assert_eq!(session.span, Duration::from_millis(4000));
    assert_eq!(session.other(), Duration::from_millis(4000 - 500 - 25));
}

/// A session that recorded its batches takes each batch's wall for its
/// compiling, not the work of the kernels in it, and runs at least until the
/// last batch ended.
#[test]
fn batches_count_their_wall_not_their_kernels_work() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = fixture(dir.path());
    let database = Database::open_at(&path, Access::ReadWrite).expect("opens");
    let batch = |kernels, wall_ms| CompilationBatchRecord {
        kernels,
        threads: 16,
        wall: Duration::from_millis(wall_ms),
    };
    insert_record(&database, 20, 2_100, batch(1, 90));
    insert_record(&database, 21, 3_000, batch(1, 5));
    insert_record(&database, 22, 3_100, batch(1, 1_000));
    drop(database);

    let inspector = Inspector::open(path).expect("opens");
    let session = &inspector.summary().sessions[0];
    assert_eq!(session.compilations, 3);
    assert_eq!(session.compiling, Duration::from_millis(1_095));
    assert_eq!(session.compiling_in_tunes, Duration::from_millis(90));
    assert_eq!(session.span, Duration::from_millis(4_100));

    let timeline = inspector.timeline();
    let turn = &timeline.sessions[0].spans[1];
    assert_eq!(turn.compiling, Duration::from_millis(90));
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
            .all(|row| row.root().as_str() != "records")
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
        .find(|key| key.tune.is_some())
        .expect("the recorded key");
    assert_eq!(
        traced.tune.as_ref().map(|tune| tune.compiling),
        Some(Duration::from_millis(80))
    );
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

    assert_eq!(
        inspector.kernel(id("abcdef")).expect("by prefix").id,
        matmul.id
    );
    assert!(matches!(
        inspector.kernel(id("ffff")),
        Err(InspectError::UnknownKernel(_))
    ));
    assert!(matches!(
        inspector.kernel(id("")),
        Err(InspectError::AmbiguousKernel { count: 2, .. })
    ));
}

/// Selects a kernel by a prefix of its id alone.
fn id(prefix: &str) -> KernelSelector<'_> {
    KernelSelector {
        id: prefix,
        build: None,
    }
}

/// A rebuild records the same kernel under a second build: its full id then
/// names both, and the build is what picks one.
#[test]
fn a_kernel_two_builds_recorded_is_picked_by_its_build() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = fixture(dir.path());
    {
        let database = Database::open_at(&path, Access::ReadWrite).expect("opens");
        insert_record(
            &database,
            8,
            4_000,
            CompilationRecord {
                kernel: "kernels::Matmul<f16>".to_string(),
                ir: None,
                key: KernelCacheKey {
                    id: MATMUL,
                    build_id: 2,
                },
                outcome: CompilationOutcome::Compiled,
                work: Duration::from_millis(90),
                source: None,
            },
        );
    }
    let inspector = Inspector::open(&path).expect("opens");
    let full = format!("{MATMUL:032x}");

    assert!(matches!(
        inspector.kernel(id(&full)),
        Err(InspectError::AmbiguousBuild { builds, .. }) if builds.len() == 2
    ));
    let rebuilt = inspector
        .kernel(KernelSelector {
            id: &full,
            build: Some(&format!("{:032x}", 2u128)),
        })
        .expect("the build names one");
    assert_eq!(rebuilt.compiling, Duration::from_millis(90));
    // Three rows, two kernels: the count is of kernels.
    assert!(matches!(
        inspector.kernel(id("")),
        Err(InspectError::AmbiguousKernel { count: 2, .. })
    ));
}

/// A tune the table did not take — unmeasured, or with the cache disabled —
/// says nothing of the answer the table holds for its key.
#[test]
fn a_tune_the_table_did_not_take_is_not_the_keys_trace() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = fixture(dir.path());
    {
        let database = Database::open_at(&path, Access::ReadWrite).expect("opens");
        insert_record(
            &database,
            9,
            5_000,
            TuneRecord {
                table: GEMM.to_string(),
                entry: PersistentCacheKey {
                    key: GemmKey {
                        m: 128,
                        elem: Elem::Float("F16"),
                    },
                    checksum: "list-a".to_string(),
                },
                winner: 0,
                trials: Vec::new(),
                short_circuit: None,
                wall: Duration::from_millis(1),
                launches_discarded: false,
                stored: false,
                plan: Vec::new(),
            },
        );
    }
    let report = Inspector::open(&path).expect("opens").autotune();

    let traced = report.keys.iter().filter(|key| key.tune.is_some()).count();
    assert_eq!(traced, 1, "only the m=64 tune was stored");
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
    assert_eq!(snapshot.report.streams[0].pools.persistent.pages_peak, 2);
}

/// A second environment beside the fixture: the m=128 key answered by the other
/// candidate, the norm key gone, and a key the first never tuned.
fn rebuilt(dir: &Path) -> PathBuf {
    let path = dir.join("rebuilt");
    std::fs::create_dir_all(&path).expect("a dir");
    let path = fixture(&path);
    let database = Database::open_at(&path, Access::ReadWrite).expect("opens");
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

/// A file one cubecl version apart from itself answers a key twice; the
/// diff reads the newest version's answer, where "0.9.0" sorts after
/// "0.11.0" as text.
#[test]
fn a_diff_reads_a_keys_newest_version() {
    use cubecl_inspect::report::KeyChange;

    let dir = tempfile::tempdir().expect("a temp dir");
    let before = Inspector::open(fixture(dir.path())).expect("opens");
    let upgraded = dir.path().join("upgraded");
    std::fs::create_dir(&upgraded).expect("a dir");
    let path = fixture(&upgraded);
    {
        let database = Database::open_at(&path, Access::ReadWrite).expect("opens");
        database.insert(
            "autotune/0.9.0/hip-0/matmul-tune-gemm",
            &cbor(&StoredKey {
                key: GemmKey {
                    m: 64,
                    elem: Elem::Float("F16"),
                },
                checksum: "list-a",
            }),
            &cbor(&PersistentCacheValue {
                fastest_index: 0,
                results: vec![measured("naive", 0, 10), measured("tiled", 1, 40)],
                bounds: None,
                limit: None,
            }),
            Origin::Local,
        );
    }
    let after = Inspector::open(&path).expect("opens");

    let diff = before.diff(&after);
    assert_eq!(diff.with(KeyChange::WinnerChanged).count(), 0);
    assert_eq!(diff.with(KeyChange::Kept).count(), 3);
}

#[test]
fn pruning_keeps_the_newest_sessions_records() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = fixture(dir.path());
    {
        let database = Database::open_at(&path, Access::ReadWrite).expect("opens");
        let later = Session {
            id: SessionId(9),
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
    let summary = Inspector::open(&path)
        .expect("opens")
        .prune(1)
        .expect("prunes");
    let inspector = Inspector::open(&path).expect("reopens");

    let sessions: Vec<SessionId> = summary.sessions.iter().map(|row| row.session.id).collect();
    assert_eq!(sessions, vec![SessionId(9)]);
    assert!(
        inspector
            .autotune()
            .keys
            .iter()
            .all(|key| key.tune.is_none())
    );
    assert!(inspector.kernels().kernels.is_empty());
    assert_eq!(summary.autotune_keys(), 4, "the caches are untouched");
}

/// Pruning a shipped file in place leaves it shipped: out of WAL, so it
/// still opens from a read-only directory.
#[test]
fn pruning_a_shipped_file_keeps_it_out_of_wal() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = fixture(dir.path());
    Database::open_at(&path, Access::ReadWrite)
        .expect("opens")
        .into_standalone()
        .expect("ships");

    Inspector::open(&path)
        .expect("opens")
        .prune(0)
        .expect("prunes");

    assert!(
        Database::open_at(&path, Access::ReadOnly)
            .expect("opens")
            .is_standalone()
    );
}
