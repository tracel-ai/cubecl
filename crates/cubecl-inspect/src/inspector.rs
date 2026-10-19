use crate::report::{
    AutotuneReport, AutotuneTable, Comparison, DiffSide, EnvironmentDiff, EnvironmentRecords,
    KernelMatch, KernelReport, KernelRow, KernelSelector, KeyId, MemorySnapshots, NamespaceRoot,
    SessionRow, SessionTimeline, StoredArtifacts, Summary, Timeline, TuneIndex, TunedKey,
};
use crate::{InspectError, environment};
use cubecl_environment::bundle::{BundleManifest, ExportOptions, export};
use cubecl_environment::persistence::{Access, Database};
use cubecl_environment::records::Records;
use cubecl_server::compiler::KernelCacheKey;
use std::path::{Path, PathBuf};

/// One environment file, opened read-only: the reports are what it answers.
///
/// Read-only is what makes it safe to point at an environment a running
/// process is writing: the database is shared through WAL, and nothing here
/// writes, migrates or checkpoints it.
pub struct Inspector {
    database: Database,
    path: PathBuf,
    /// Read once, on open: every report reads the file as it was then.
    records: EnvironmentRecords,
}

impl Inspector {
    /// Open the file at `path` and read its records.
    ///
    /// # Errors
    ///
    /// [`InspectError::Open`] when the file is missing, or is not an
    /// environment at the schema this build reads.
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
        Ok(Self {
            records: records(&database),
            database,
            path,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
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
                .map(Into::into)
                .collect(),
            sessions: self.records.sessions().map(SessionRow::new).collect(),
        }
    }

    /// Each session's marks, with what was tuned and compiled inside them.
    pub fn timeline(&self) -> Timeline {
        Timeline {
            sessions: self.records.sessions().map(SessionTimeline::new).collect(),
        }
    }

    /// Every memory snapshot the file's sessions recorded.
    pub fn memory(&self) -> MemorySnapshots {
        MemorySnapshots {
            snapshots: self.records.memory.clone(),
        }
    }

    /// Every kernel the file's sessions compiled or loaded.
    pub fn kernels(&self) -> KernelReport {
        KernelReport::new(
            self.records.compilations.iter().map(|trip| &trip.record),
            &self.stored_artifacts(),
        )
    }

    /// The one kernel instance `selector` names.
    ///
    /// # Errors
    ///
    /// When it names none, or several: the error says what the selector
    /// lacks.
    pub fn kernel(&self, selector: KernelSelector<'_>) -> Result<KernelRow, InspectError> {
        match KernelMatch::new(selector, self.kernels().kernels) {
            KernelMatch::One(row) => Ok(row),
            KernelMatch::Unknown => Err(InspectError::UnknownKernel(selector.to_string())),
            KernelMatch::SeveralBuilds(rows) => Err(InspectError::AmbiguousBuild {
                kernel: selector.to_string(),
                builds: rows.iter().map(|row| row.build.to_string()).collect(),
            }),
            KernelMatch::SeveralKernels(count) => Err(InspectError::AmbiguousKernel {
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
            if !NamespaceRoot::new(&namespace).holds_artifacts() {
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
        for namespace in self.database.namespaces() {
            let Some(table) = AutotuneTable::new(&namespace) else {
                continue;
            };
            self.database.scan(
                &namespace,
                &mut |key, value| match TunedKey::new(&table, key, value) {
                    Some(tuned) => report.keys.push(tuned),
                    None => report.undecoded += 1,
                },
            );
        }
        let tunes = TuneIndex::new(&self.records);
        for key in &mut report.keys {
            key.tune = tunes.recorded_tune(key);
        }
        report
    }

    /// This file's autotune answers against `other`'s, this one before.
    pub fn diff(&self, other: &Inspector) -> EnvironmentDiff {
        EnvironmentDiff::new(Comparison {
            before: DiffSide {
                path: self.path.clone(),
                autotune: &self.autotune(),
            },
            after: DiffSide {
                path: other.path.clone(),
                autotune: &other.autotune(),
            },
        })
    }

    /// Drop the records of every session but the newest `keep`, in place,
    /// and summarize what is left.
    ///
    /// The one write this type makes to its own file, through a connection
    /// of its own. A
    /// read-write open migrates a file of another schema by dropping its
    /// entries; this one cannot, because [`open`](Self::open) refused any
    /// file this build's cubecl did not write.
    ///
    /// Takes the inspector, whose read-only handle has to be closed first: a
    /// file open for reading in this process cannot be written by it.
    ///
    /// # Errors
    ///
    /// [`InspectError::Prune`] when the file cannot be opened for writing,
    /// such as one in a read-only directory; [`InspectError::Open`] when the
    /// pruned file cannot be read back.
    pub fn prune(self, keep: usize) -> Result<Summary, InspectError> {
        let Self { database, path, .. } = self;
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
    ///
    /// # Errors
    ///
    /// [`InspectError::Export`] when the copy cannot be written to `out`;
    /// [`InspectError::Open`] when it cannot be read back.
    pub fn strip(&self, out: &Path) -> Result<Summary, InspectError> {
        let manifest = BundleManifest::new(&self.database).ok();
        let options = ExportOptions {
            name: manifest.as_ref().map_or_else(
                || environment::stem(&self.path),
                |manifest| manifest.name.clone(),
            ),
            environments: manifest
                .map(|manifest| manifest.environments)
                .unwrap_or_default(),
            // Every layout of records, not only the one this build reads.
            excluded_namespaces: vec![NamespaceRoot::records().as_str().to_string()],
            ..Default::default()
        };
        export(&[&self.path], out, &options).map_err(|err| InspectError::Export {
            path: out.to_path_buf(),
            reason: err.to_string(),
        })?;
        Ok(Self::open(out)?.summary())
    }

    /// The one key `id` names.
    ///
    /// # Errors
    ///
    /// [`InspectError::UnknownKey`] when no key of the file carries `id`.
    pub fn autotune_key(&self, id: KeyId) -> Result<TunedKey, InspectError> {
        self.autotune()
            .keys
            .into_iter()
            .find(|key| key.id == id)
            .ok_or(InspectError::UnknownKey(id))
    }
}

/// The file's records, read whole.
fn records(database: &Database) -> EnvironmentRecords {
    let records = Records::new(database);
    EnvironmentRecords {
        sessions: records.sessions(),
        tunes: records.read(),
        compilations: records.read(),
        batches: records.read(),
        marks: records.read(),
        memory: records.read(),
    }
}
