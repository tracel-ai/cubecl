use crate::InspectError;
use crate::report::{
    AutotuneReport, AutotuneTable, CandidateResult, EnvironmentDiff, KernelReport, KernelRow,
    KeyId, Listing, MemorySnapshots, NamespaceRow, SessionRow, SessionTimeline, StoreEntry,
    StoredArtifacts, Summary, Timeline, TuneTrace, TunedKey, Unreadable,
};
use cubecl_environment::bundle::{BundleManifest, ExportOptions, export};
use cubecl_environment::persistence::Database;
use cubecl_environment::persistence::sqlite::SCHEMA_VERSION;
use cubecl_environment::records;
use cubecl_environment::records::{MarkRecord, Stamped};
use cubecl_server::compiler::{CompilationRecord, KernelCacheKey};
use cubecl_server::memory_management::MemoryRecord;
use cubecl_server::tune::{PersistentCacheValue, TuneRecord};
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
struct Records {
    /// Oldest first, like every list here.
    tunes: Vec<TuneTrace>,
    compilations: Vec<Stamped<CompilationRecord>>,
    marks: Vec<Stamped<MarkRecord>>,
}

impl Records {
    fn read(database: &Database) -> Self {
        Self {
            tunes: records::read(database, TuneRecord::<()>::KIND),
            compilations: records::read(database, CompilationRecord::KIND),
            marks: records::read(database, MarkRecord::KIND),
        }
    }

    fn tunes_of(&self, session: u64) -> impl Iterator<Item = &TuneTrace> {
        self.tunes
            .iter()
            .filter(move |trace| trace.stamp.session == session)
    }

    fn compilations_of(&self, session: u64) -> impl Iterator<Item = &Stamped<CompilationRecord>> {
        self.compilations
            .iter()
            .filter(move |trip| trip.stamp.session == session)
    }

    /// What the compilations that started during `tune` cost: the tuner
    /// waits on the server while it compiles, so a compilation stamped inside
    /// the tune's span on the same session clock is the tune's.
    fn compiling_within(&self, tune: &TuneTrace) -> Duration {
        let start = tune.stamp.offset;
        let end = start + tune.record.wall;
        self.compilations_of(tune.stamp.session)
            .filter(|trip| (start..end).contains(&trip.stamp.offset))
            .map(|trip| trip.record.outcome.duration())
            .sum()
    }
}

/// A stored autotune key, as far as a reader needs it.
///
/// cubecl's `PersistentCacheKey` keeps the checksum private, and the checksum
/// is what tells two candidate lists' answers to one key apart, so the key is
/// decoded through this twin of it rather than through the original.
#[derive(serde::Deserialize)]
struct StoredKey {
    key: ciborium::Value,
    checksum: String,
}

impl Inspector {
    /// The file extension a saved environment carries.
    pub const EXTENSION: &str = "cubecl";

    pub fn open(path: impl Into<PathBuf>) -> Result<Self, InspectError> {
        let path = path.into();
        // A read-only open of a missing file fails too, but with SQLite's
        // "unable to open database file", which does not say which half of
        // that is wrong.
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
        let database = Database::open(&path, true).map_err(|err| refused(err.to_string()))?;
        // SQLite opens any file and fails at the first statement, and the
        // database's own readers log that failure and answer empty: a file
        // that is not an environment would read as an empty one.
        database
            .with_connection(|conn| {
                conn.query_row("SELECT count(*) FROM entries", [], |row| {
                    row.get::<_, i64>(0)
                })
            })
            .map_err(|err| refused(err.to_string()))?;
        Ok(Self { database, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Every environment file in `directory`, newest build first.
    pub fn list(directory: &Path) -> Result<Listing, InspectError> {
        let entries = std::fs::read_dir(directory).map_err(|source| InspectError::Directory {
            path: directory.to_path_buf(),
            source,
        })?;
        let mut listing = Listing {
            directory: directory.to_path_buf(),
            environments: Vec::new(),
            unreadable: Vec::new(),
        };
        for path in entries.flatten().map(|entry| entry.path()) {
            if path.extension().and_then(|ext| ext.to_str()) != Some(Self::EXTENSION) {
                continue;
            }
            match Self::open(&path) {
                Ok(inspector) => listing.environments.push(inspector.summary()),
                Err(err) => listing.unreadable.push(Unreadable {
                    path,
                    reason: err.to_string(),
                }),
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
        Ok(listing)
    }

    pub fn summary(&self) -> Summary {
        Summary {
            path: self.path.clone(),
            file_bytes: std::fs::metadata(&self.path).map_or(0, |meta| meta.len()),
            manifest: BundleManifest::read(&self.database).ok(),
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
            description: None,
        }
    }

    /// Every session, with what it recorded folded in.
    fn sessions(&self) -> Vec<SessionRow> {
        let records = Records::read(&self.database);
        records::sessions(&self.database)
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
                for trace in records.tunes_of(row.session.id) {
                    row.tunes += 1;
                    row.tuning += trace.record.wall;
                    row.compiling_in_tunes += records.compiling_within(trace);
                    row.span = row.span.max(trace.stamp.offset + trace.record.wall);
                }
                for trip in records.compilations_of(row.session.id) {
                    row.compilations += 1;
                    row.compiling += trip.record.outcome.duration();
                    row.span = row
                        .span
                        .max(trip.stamp.offset + trip.record.outcome.duration());
                }
                row
            })
            .collect()
    }

    /// Each session's marks, with what was tuned and compiled inside them.
    pub fn timeline(&self) -> Timeline {
        let records = Records::read(&self.database);
        Timeline {
            sessions: records::sessions(&self.database)
                .into_iter()
                .map(|session| {
                    SessionTimeline::new(
                        session,
                        &records.marks,
                        &records.tunes,
                        &records.compilations,
                    )
                })
                .collect(),
        }
    }

    /// Every memory snapshot the file's sessions recorded.
    pub fn memory(&self) -> MemorySnapshots {
        MemorySnapshots {
            snapshots: records::read(&self.database, MemoryRecord::KIND),
        }
    }

    /// Every kernel the file's builds compiled or loaded.
    pub fn kernels(&self) -> KernelReport {
        let trips: Vec<CompilationRecord> = Records::read(&self.database)
            .compilations
            .into_iter()
            .map(|trip| trip.record)
            .collect();
        KernelReport::new(&trips, &self.stored_artifacts())
    }

    /// The one kernel instance whose id starts with `prefix`.
    pub fn kernel(&self, prefix: &str) -> Result<KernelRow, InspectError> {
        let mut matches: Vec<KernelRow> = self
            .kernels()
            .kernels
            .into_iter()
            .filter(|row| row.id.matches(prefix))
            .collect();
        match matches.len() {
            1 => Ok(matches.remove(0)),
            0 => Err(InspectError::UnknownKernel(prefix.to_string())),
            count => Err(InspectError::AmbiguousKernel {
                prefix: prefix.to_string(),
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
                    stored.insert(StoreEntry::from(&key), value.len() as u64);
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
        let records = Records::read(&self.database);
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
            // Oldest first, so the last match is the key's latest tune.
            key.trace = records
                .tunes
                .iter()
                .rev()
                .find(|trace| {
                    let record = &trace.record;
                    record.table == key.table.namespace()
                        && record.checksum == key.checksum
                        && record.key == key.key
                })
                .cloned();
            key.compiling = key
                .trace
                .as_ref()
                .map(|trace| records.compiling_within(trace));
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
    /// The one write this type makes, through a connection of its own: a
    /// read-write open migrates a file of another schema by dropping its
    /// entries, so a file this build's cubecl did not write is refused
    /// rather than opened.
    pub fn prune(&self, keep: usize) -> Result<Summary, InspectError> {
        let refused = |reason: String| InspectError::Prune {
            path: self.path.clone(),
            reason,
        };
        let schema: Option<String> = self
            .database
            .with_connection(|conn| {
                conn.query_row("SELECT v FROM meta WHERE k = 'schema_version'", [], |row| {
                    row.get(0)
                })
            })
            .ok();
        if schema.as_deref() != Some(SCHEMA_VERSION.to_string().as_str()) {
            return Err(refused(format!(
                "its schema is {}, this build writes {SCHEMA_VERSION}",
                schema.as_deref().unwrap_or("unknown")
            )));
        }
        let writable = Database::open(&self.path, false).map_err(|err| refused(err.to_string()))?;
        records::prune(&writable, keep);
        Ok(self.summary())
    }

    /// Write a copy of the file to `out` without its records — the account
    /// of how it was built, which a distributed environment has no use for —
    /// and summarize the copy.
    pub fn strip(&self, out: &Path) -> Result<Summary, InspectError> {
        let manifest = BundleManifest::read(&self.database).ok();
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

/// A stored entry of `table`, decoded; `None` when either half does not
/// decode.
fn tuned_key(namespace: &str, table: &AutotuneTable, key: &[u8], value: &[u8]) -> Option<TunedKey> {
    let stored: StoredKey = ciborium::from_reader(key).ok()?;
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
