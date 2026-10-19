use crate::report::{
    AutotuneReport, AutotuneTable, EnvironmentDiff, EnvironmentRecords, KernelHash, KernelReport,
    KernelRow, KernelSelector, KeyId, Listing, MemorySnapshots, NamespaceRoot, SessionRow,
    SessionTimeline, StoredArtifacts, Summary, Timeline, TunedKey, Unreadable,
};
use crate::{EnvironmentDirectory, InspectError, environment};
use cubecl_environment::bundle::{BundleManifest, ExportOptions, export};
use cubecl_environment::collections::HashSet;
use cubecl_environment::persistence::{Access, Database};
use cubecl_environment::records::Records;
use cubecl_server::compiler::KernelCacheKey;
use cubecl_server::memory_management::MemoryRecord;
use std::path::{Path, PathBuf};

/// One environment file, opened read-only: the reports are what it answers.
///
/// Read-only is what makes it safe to point at an environment a running
/// process is writing: the database is shared through WAL, and nothing here
/// writes, migrates or checkpoints it.
pub struct Inspector {
    database: Database,
    path: PathBuf,
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

    /// Every environment file in `directories`, as one list, newest
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
            let files = match directory.files() {
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
            b.created_unix_secs()
                .cmp(&a.created_unix_secs())
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
                .map(Into::into)
                .collect(),
            sessions: self.records().sessions().map(SessionRow::new).collect(),
        }
    }

    /// Each session's marks, with what was tuned and compiled inside them.
    pub fn timeline(&self) -> Timeline {
        Timeline {
            sessions: self
                .records()
                .sessions()
                .map(SessionTimeline::new)
                .collect(),
        }
    }

    /// Every memory snapshot the file's sessions recorded.
    pub fn memory(&self) -> MemorySnapshots {
        MemorySnapshots {
            snapshots: Records::new(&self.database).read::<MemoryRecord>(),
        }
    }

    /// Every kernel the file's sessions compiled or loaded.
    pub fn kernels(&self) -> KernelReport {
        let records = self.records();
        KernelReport::new(
            records.compilations.iter().map(|trip| &trip.record),
            &self.stored_artifacts(),
        )
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
        let records = self.records();
        for key in &mut report.keys {
            key.tune = records.recorded_tune(key);
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
    pub fn autotune_key(&self, id: KeyId) -> Result<TunedKey, InspectError> {
        self.autotune()
            .keys
            .into_iter()
            .find(|key| key.id == id)
            .ok_or(InspectError::UnknownKey(id))
    }

    /// The file's records, read whole.
    fn records(&self) -> EnvironmentRecords {
        let records = Records::new(&self.database);
        EnvironmentRecords {
            sessions: records.sessions(),
            tunes: records.read(),
            compilations: records.read(),
            batches: records.read(),
            marks: records.read(),
        }
    }
}
