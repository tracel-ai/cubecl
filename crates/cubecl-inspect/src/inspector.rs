use crate::report::{
    AutotuneReport, AutotuneTable, CandidateResult, CompilationSpans, EnvironmentDiff, KernelHash,
    KernelReport, KernelRow, KernelSelector, KeyId, Listing, MemorySnapshots, NamespaceRow,
    SessionRow, SessionTimeline, StoredArtifacts, Summary, Timeline, TuneTrace, TunedKey,
    Unreadable,
};
use crate::{EnvironmentDirectory, InspectError};
use cubecl_environment::bundle::{BundleManifest, ExportOptions, export};
use cubecl_environment::collections::HashSet;
use cubecl_environment::persistence::{Access, Database};
use cubecl_environment::records;
use cubecl_environment::records::{MarkRecord, Records, SessionId, Stamped};
use cubecl_server::compiler::{CompilationBatchRecord, CompilationRecord, KernelCacheKey};
use cubecl_server::memory_management::MemoryRecord;
use cubecl_server::tune::{PersistentCacheKey, PersistentCacheValue};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// One environment file, opened read-only: the reports are what it answers.
///
/// Read-only is what makes it safe to point at an environment a running
/// process is writing: the database is shared through WAL, and nothing here
/// writes, migrates or checkpoints it.
pub struct Inspector {
    database: Database,
    path: PathBuf,
}

/// The namespace roots holding measurements and records rather than
/// compiled artifacts.
const MEASUREMENTS: [&str; 3] = [AutotuneTable::ROOT, "throughput", "records"];

/// The build records of one file, read once per report.
struct BuildRecords {
    /// Oldest first, like every list here.
    tunes: Vec<TuneTrace>,
    compilations: Vec<Stamped<CompilationRecord>>,
    batches: Vec<Stamped<CompilationBatchRecord>>,
    marks: Vec<Stamped<MarkRecord>>,
}

impl BuildRecords {
    fn new(database: &Database) -> Self {
        let records = Records::new(database);
        Self {
            tunes: records.read(),
            compilations: records.read(),
            batches: records.read(),
            marks: records.read(),
        }
    }

    fn tunes_of(&self, session: SessionId) -> impl Iterator<Item = &TuneTrace> {
        self.tunes
            .iter()
            .filter(move |trace| trace.stamp.session == session)
    }

    fn compilations_of(
        &self,
        session: SessionId,
    ) -> impl Iterator<Item = &Stamped<CompilationRecord>> {
        self.compilations
            .iter()
            .filter(move |trip| trip.stamp.session == session)
    }

    fn marks_of(&self, session: SessionId) -> impl Iterator<Item = &Stamped<MarkRecord>> {
        self.marks
            .iter()
            .filter(move |mark| mark.stamp.session == session)
    }

    /// The stretches of `session`'s clock its compilations took.
    fn compiling_of(&self, session: SessionId) -> CompilationSpans {
        CompilationSpans::new(
            self.batches
                .iter()
                .filter(|batch| batch.stamp.session == session),
            self.compilations_of(session),
        )
    }
}

impl Inspector {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, InspectError> {
        let path = path.into();
        // A read-only open of a missing file fails too, but with the engine's
        // own message, which does not say whether the file is missing or
        // unreadable.
        if !path.is_file() {
            return Err(InspectError::Open {
                path,
                reason: "no such file".to_string(),
            });
        }
        let refused = |reason: String| InspectError::Open {
            path: path.clone(),
            reason,
        };
        // Refused unless the file is an environment at this build's schema:
        // anything else would read as an empty one.
        let database = Database::open_at(&path, Access::ReadOnly).map_err(refused)?;
        Ok(Self { database, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Every environment file in `directories`, as one list, newest build
    /// first. A directory that does not exist holds none — a machine that
    /// never built has no environments — and one that cannot be read is
    /// listed as unreadable, beside what the others hold.
    pub fn list(directories: &[EnvironmentDirectory]) -> Listing {
        let mut listing = Listing {
            directories: directories
                .iter()
                .map(|directory| directory.path.clone())
                .collect(),
            environments: Vec::new(),
            unreadable: Vec::new(),
        };
        for directory in directories {
            let files = match files(directory) {
                Ok(files) => files,
                Err(err) => {
                    listing.unreadable.push(Unreadable {
                        path: directory.path.clone(),
                        reason: err.to_string(),
                    });
                    continue;
                }
            };
            for path in files {
                match Self::open(&path) {
                    Ok(inspector) => listing.environments.push(inspector.summary()),
                    Err(err) => listing.unreadable.push(Unreadable {
                        path,
                        reason: err.to_string(),
                    }),
                }
            }
        }
        listing.environments.sort_by(|a, b| {
            let created = |summary: &Summary| {
                summary
                    .manifest
                    .as_ref()
                    .and_then(|manifest| manifest.created_unix_secs)
            };
            created(b)
                .cmp(&created(a))
                .then_with(|| a.path.cmp(&b.path))
        });
        listing.unreadable.sort_by(|a, b| a.path.cmp(&b.path));
        listing
    }

    pub fn summary(&self) -> Summary {
        Summary {
            path: self.path.clone(),
            file_bytes: std::fs::metadata(&self.path).map_or(0, |meta| meta.len()),
            manifest: BundleManifest::new(&self.database).ok(),
            namespaces: self
                .database
                .summary()
                .into_iter()
                .map(|summary| NamespaceRow {
                    namespace: summary.namespace,
                    entries: summary.entries,
                    bytes: summary.bytes,
                })
                .collect(),
            sessions: self.sessions(),
        }
    }

    /// Every session, with what it recorded folded in.
    fn sessions(&self) -> Vec<SessionRow> {
        let records = BuildRecords::new(&self.database);
        Records::new(&self.database)
            .sessions()
            .into_iter()
            .map(|session| {
                let mut row = SessionRow {
                    session,
                    tunes: 0,
                    tuning: Duration::ZERO,
                    compilations: 0,
                    compiling: Duration::ZERO,
                    compiling_in_tunes: Duration::ZERO,
                    span: Duration::ZERO,
                };
                let compiling = records.compiling_of(row.session.id);
                for trace in records.tunes_of(row.session.id) {
                    row.tunes += 1;
                    row.tuning += trace.record.wall;
                    row.compiling_in_tunes += compiling.during(trace);
                    row.span = row.span.max(trace.stamp.offset + trace.record.wall);
                }
                row.compilations = records.compilations_of(row.session.id).count() as u64;
                row.compiling = compiling.wall();
                row.span = row.span.max(compiling.end());
                // A phase after the last compile — a walk served from the
                // store — still ran.
                for mark in records.marks_of(row.session.id) {
                    row.span = row.span.max(mark.stamp.offset + mark.record.wall);
                }
                row
            })
            .collect()
    }

    /// Each session's marks, with what was tuned and compiled inside them.
    pub fn timeline(&self) -> Timeline {
        let records = BuildRecords::new(&self.database);
        Timeline {
            sessions: Records::new(&self.database)
                .sessions()
                .into_iter()
                .map(|session| {
                    let compiling = records.compiling_of(session.id);
                    SessionTimeline::new(
                        session,
                        &records.marks,
                        &records.tunes,
                        &records.compilations,
                        &compiling,
                    )
                })
                .collect(),
        }
    }

    /// Every memory snapshot the file's sessions recorded.
    pub fn memory(&self) -> MemorySnapshots {
        MemorySnapshots {
            snapshots: Records::new(&self.database).read::<MemoryRecord>(),
        }
    }

    /// Every kernel the file's builds compiled or loaded.
    pub fn kernels(&self) -> KernelReport {
        let trips: Vec<CompilationRecord> = BuildRecords::new(&self.database)
            .compilations
            .into_iter()
            .map(|trip| trip.record)
            .collect();
        KernelReport::new(&trips, &self.stored_artifacts())
    }

    /// The one kernel instance `selector` names.
    ///
    /// A kernel is recorded once per build that compiled or loaded it, so
    /// after a rebuild its id alone names several: the selector's build
    /// names one of them.
    pub fn kernel(&self, selector: KernelSelector<'_>) -> Result<KernelRow, InspectError> {
        let mut matches: Vec<KernelRow> = self
            .kernels()
            .kernels
            .into_iter()
            .filter(|row| selector.selects(row))
            .collect();
        let kernels: HashSet<KernelHash> = matches.iter().map(|row| row.id).collect();
        match (matches.len(), kernels.len()) {
            (0, _) => Err(InspectError::UnknownKernel(selector.to_string())),
            (1, _) => Ok(matches.remove(0)),
            (_, 1) => Err(InspectError::AmbiguousBuild {
                kernel: selector.to_string(),
                builds: matches.iter().map(|row| row.build.to_string()).collect(),
            }),
            (_, count) => Err(InspectError::AmbiguousKernel {
                prefix: selector.to_string(),
                count,
            }),
        }
    }

    /// The compilation store's artifacts, by the entry naming them: every
    /// namespace that is not a measurement, read as a compilation store.
    fn stored_artifacts(&self) -> StoredArtifacts {
        let mut stored = StoredArtifacts::new();
        for namespace in self.database.namespaces() {
            let root = namespace.split('/').next().unwrap_or_default();
            if MEASUREMENTS.contains(&root) {
                continue;
            }
            self.database.scan(&namespace, &mut |key, value| {
                if let Ok(key) = ciborium::from_reader::<KernelCacheKey, _>(key) {
                    stored.insert(key, value.len() as u64);
                }
            });
        }
        stored
    }

    /// Every tuned key, in the order the file stores them, each with the
    /// record of its last tune when there is one.
    pub fn autotune(&self) -> AutotuneReport {
        let mut report = AutotuneReport {
            keys: Vec::new(),
            undecoded: 0,
        };
        let records = BuildRecords::new(&self.database);
        for namespace in self.database.namespaces() {
            let Some(table) = AutotuneTable::parse(&namespace) else {
                continue;
            };
            self.database
                .scan(
                    &namespace,
                    &mut |key, value| match tuned_key(&namespace, &table, key, value) {
                        Some(tuned) => report.keys.push(tuned),
                        None => report.undecoded += 1,
                    },
                );
        }
        for key in &mut report.keys {
            // Oldest first, so the last match is the key's latest tune. A tune
            // the table did not take answered its process alone, and says
            // nothing of the answer stored.
            key.trace = records
                .tunes
                .iter()
                .rev()
                .find(|trace| {
                    let record = &trace.record;
                    record.stored
                        && record.table == key.table.namespace()
                        && record.entry.checksum == key.checksum
                        && record.entry.key == key.key
                })
                .cloned();
            key.compiling = key
                .trace
                .as_ref()
                .map(|trace| records.compiling_of(trace.stamp.session).during(trace));
        }
        report
    }

    /// This file's autotune answers against `other`'s, this one before.
    pub fn diff(&self, other: &Inspector) -> EnvironmentDiff {
        EnvironmentDiff::new(
            (self.path.clone(), &self.autotune()),
            (other.path.clone(), &other.autotune()),
        )
    }

    /// Drop the records of every session but the newest `keep`, in place,
    /// and summarize what is left.
    ///
    /// The one write this type makes, through a connection of its own. A
    /// read-write open migrates a file of another schema by dropping its
    /// entries; this one cannot, because [`open`](Self::open) refused any
    /// file this build's cubecl did not write.
    ///
    /// Takes the inspector, whose read-only handle has to be closed first: a
    /// file open for reading in this process cannot be written by it.
    pub fn prune(self, keep: usize) -> Result<Summary, InspectError> {
        let Self { database, path } = self;
        let refused = |reason: String| InspectError::Prune {
            path: path.clone(),
            reason,
        };
        // A read-write open switches the file to WAL for good. A file that was
        // not in WAL — shipped, meant for a read-only directory — is shipped
        // again, as `strip` leaves one.
        let shipped = database.is_standalone();
        drop(database);
        let writable = Database::open_at(&path, Access::ReadWrite).map_err(refused)?;
        Records::new(&writable).prune(keep);
        match shipped {
            true => writable.into_standalone().map_err(refused)?,
            false => drop(writable),
        }
        Ok(Self::open(path)?.summary())
    }

    /// Write a copy of the file to `out` without its records — the account
    /// of how it was built, which a distributed environment has no use for —
    /// and summarize the copy.
    pub fn strip(&self, out: &Path) -> Result<Summary, InspectError> {
        let manifest = BundleManifest::new(&self.database).ok();
        let options = ExportOptions {
            name: manifest.as_ref().map_or_else(
                || {
                    self.path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned()
                },
                |manifest| manifest.name.clone(),
            ),
            environments: manifest
                .map(|manifest| manifest.environments)
                .unwrap_or_default(),
            // Every layout of records, not only the one this build reads.
            excluded_namespaces: vec![
                records::ROOT
                    .split('/')
                    .next()
                    .unwrap_or(records::ROOT)
                    .to_string(),
            ],
            ..Default::default()
        };
        export(&[&self.path], out, &options).map_err(|err| InspectError::Export {
            path: out.to_path_buf(),
            reason: err.to_string(),
        })?;
        Ok(Self::open(out)?.summary())
    }

    /// The one key `id` names.
    pub fn autotune_key(&self, id: KeyId) -> Result<TunedKey, InspectError> {
        self.autotune()
            .keys
            .into_iter()
            .find(|key| key.id == id)
            .ok_or(InspectError::UnknownKey(id))
    }
}

/// The files of `directory` its extension marks, read with or without its
/// dot; none when the directory does not exist.
fn files(directory: &EnvironmentDirectory) -> std::io::Result<Vec<PathBuf>> {
    let entries = match std::fs::read_dir(&directory.path) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    let extension = directory.extension.trim_start_matches('.');
    Ok(entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some(extension))
        .collect())
}

/// A stored entry of `table`, decoded; `None` when either half does not
/// decode.
fn tuned_key(namespace: &str, table: &AutotuneTable, key: &[u8], value: &[u8]) -> Option<TunedKey> {
    let stored: PersistentCacheKey<ciborium::Value> = ciborium::from_reader(key).ok()?;
    let value: PersistentCacheValue = ciborium::from_reader(value).ok()?;
    Some(TunedKey {
        id: KeyId::new(namespace, key),
        table: table.clone(),
        key: stored.key,
        checksum: stored.checksum,
        winner: value.fastest_index,
        results: value
            .results
            .into_iter()
            .map(CandidateResult::new)
            .collect(),
        bounds: value.bounds,
        limit: value.limit,
        trace: None,
        compiling: None,
    })
}
