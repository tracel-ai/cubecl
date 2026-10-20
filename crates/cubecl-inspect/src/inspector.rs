use crate::records::{EnvironmentRecords, TuneIndex};
use crate::report::{
    AutotuneReport, AutotuneTable, CandidateReport, Comparison, DiffSide, EnvironmentDiff,
    KernelMatch, KernelReport, KernelRow, KernelSelector, KeyId, KeyOrder, MemorySnapshots,
    NamespaceRoot, Overview, OverviewReport, SessionRow, SessionTimeline, StoredArtifacts, Summary,
    Timeline, TunedKey,
};
use crate::{EnvironmentFile, InspectError};
use cubecl_environment::bundle::BundleManifest;
use cubecl_environment::persistence::Database;
use cubecl_server::compiler::KernelCacheKey;
use std::path::{Path, PathBuf};

/// One [environment file](EnvironmentFile), its records read: the reports
/// are what it answers.
pub struct Inspector {
    database: Database,
    path: PathBuf,
    /// Read once, on open: every report reads the file as it was then.
    records: EnvironmentRecords,
}

impl Inspector {
    /// Open the file at `path` and read its records: an
    /// [`EnvironmentFile`], [inspected](EnvironmentFile::inspect).
    ///
    /// # Errors
    ///
    /// See [`EnvironmentFile::open`].
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, InspectError> {
        EnvironmentFile::open(path).map(EnvironmentFile::inspect)
    }

    pub(crate) fn new(database: Database, path: PathBuf, records: EnvironmentRecords) -> Self {
        Self {
            database,
            path,
            records,
        }
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

    /// The report `overview` names, in its default order.
    pub fn overview(&self, overview: Overview) -> OverviewReport {
        match overview {
            Overview::Summary => OverviewReport::Summary(self.summary()),
            Overview::Timeline => OverviewReport::Timeline(self.timeline()),
            Overview::Autotune => {
                let mut report = self.autotune();
                report.sort(KeyOrder::default());
                OverviewReport::Autotune(report)
            }
            Overview::Candidates => {
                OverviewReport::Candidates(CandidateReport::from(&self.autotune()))
            }
            Overview::Kernels => OverviewReport::Kernels(self.kernels()),
            Overview::Memory => OverviewReport::Memory(self.memory()),
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
