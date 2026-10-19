use super::{AutotuneTable, SessionRecords};
use cubecl_environment::bundle::BundleManifest;
use cubecl_environment::persistence::NamespaceSummary;
use cubecl_environment::records::{self, Session};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

/// What one environment file holds, namespace by namespace.
#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    pub path: PathBuf,
    /// The size of the file on disk, free pages included.
    pub file_bytes: u64,
    /// What an exported environment says it was built for; `None` for a live
    /// one, which carries no manifest.
    pub manifest: Option<BundleManifest>,
    pub namespaces: Vec<NamespaceRow>,
    /// The processes that changed the file, oldest first; a warm-up that
    /// found everything done leaves none.
    pub sessions: Vec<SessionRow>,
}

/// One session, and what its records add up to.
#[derive(Clone, Debug, Serialize)]
pub struct SessionRow {
    pub session: Session,
    /// Keys it tuned.
    pub tunes: u64,
    /// The walls of those tunes, summed.
    pub tuning: Duration,
    /// Kernels it compiled or loaded.
    pub compilations: u64,
    /// The time those took: a batch compiled on several threads takes its
    /// wall, not the sum of its kernels' work.
    pub compiling: Duration,
    /// The part of [`compiling`](Self::compiling) that happened inside a
    /// tune, and so is also part of [`tuning`](Self::tuning).
    pub compiling_in_tunes: Duration,
    /// From the session's start to the end of its last recorded span: a floor
    /// on how long the process ran, since nothing records its exit.
    pub span: Duration,
}

/// One namespace's share of a file. `cubecl_environment`'s own summary row,
/// which does not implement `Serialize`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct NamespaceRow {
    pub namespace: String,
    pub entries: u64,
    /// Keys and values together.
    pub bytes: u64,
}

/// The first segment of a namespace: who wrote it — `autotune`, a backend's
/// compiler, an application.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamespaceRoot<'a>(&'a str);

impl<'a> NamespaceRoot<'a> {
    pub fn new(namespace: &'a str) -> Self {
        Self(namespace.split('/').next().unwrap_or_default())
    }

    /// The root of the records, which every layout of them shares.
    pub fn records() -> Self {
        Self::new(records::ROOT)
    }

    pub fn as_str(self) -> &'a str {
        self.0
    }

    /// Whether the namespace may hold compiled artifacts: everything but the
    /// measurements and the records cubecl keeps beside them.
    pub fn holds_artifacts(self) -> bool {
        ![
            AutotuneTable::ROOT,
            cubecl_runtime::throughput::ROOT,
            Self::records().0,
        ]
        .contains(&self.0)
    }
}

impl NamespaceRow {
    pub fn root(&self) -> NamespaceRoot<'_> {
        NamespaceRoot::new(&self.namespace)
    }
}

impl From<NamespaceSummary> for NamespaceRow {
    fn from(summary: NamespaceSummary) -> Self {
        Self {
            namespace: summary.namespace,
            entries: summary.entries,
            bytes: summary.bytes,
        }
    }
}

impl Summary {
    /// The file's name: what `--env` takes to open it again.
    pub fn stem(&self) -> String {
        crate::environment::stem(&self.path)
    }

    /// When the manifest says the environment was built; `None` for a live
    /// one.
    pub fn created_unix_secs(&self) -> Option<u64> {
        self.manifest.as_ref()?.created_unix_secs
    }

    /// What the manifest calls the environment, when it says more than the
    /// file's name: the build a digest-named file was made for.
    pub fn name(&self) -> Option<&str> {
        let name = self.manifest.as_ref()?.name.as_str();
        (!name.is_empty() && name != self.stem()).then_some(name)
    }

    /// The namespaces folded by [root](NamespaceRow::root), each row named
    /// for the root it totals.
    pub fn roots(&self) -> Vec<NamespaceRow> {
        let mut roots = BTreeMap::<&str, NamespaceRow>::new();
        for namespace in &self.namespaces {
            let root = namespace.root().as_str();
            let root = roots.entry(root).or_insert_with(|| NamespaceRow {
                namespace: root.to_string(),
                entries: 0,
                bytes: 0,
            });
            root.entries += namespace.entries;
            root.bytes += namespace.bytes;
        }
        roots.into_values().collect()
    }

    /// Every entry of the file.
    pub fn entries(&self) -> u64 {
        self.namespaces.iter().map(|row| row.entries).sum()
    }

    /// How many keys the file holds a tuned answer for.
    pub fn autotune_keys(&self) -> u64 {
        self.namespaces
            .iter()
            .filter(|row| row.root().as_str() == AutotuneTable::ROOT)
            .map(|row| row.entries)
            .sum()
    }
}

impl SessionRow {
    /// What one session's `records` add up to.
    pub fn new(records: SessionRecords<'_>) -> Self {
        let compiling = records.compiling();
        let mut row = Self {
            session: records.session().clone(),
            tunes: 0,
            tuning: Duration::ZERO,
            compilations: records.compilations().count() as u64,
            compiling: compiling.wall(),
            compiling_in_tunes: Duration::ZERO,
            span: compiling.end(),
        };
        for trace in records.tunes() {
            row.tunes += 1;
            row.tuning += trace.record.wall;
            row.compiling_in_tunes += compiling.during(trace);
            row.span = row.span.max(trace.stamp.offset + trace.record.wall);
        }
        // A phase after the last compile — a walk served from the store —
        // still ran.
        for mark in records.marks() {
            row.span = row.span.max(mark.stamp.offset + mark.record.wall);
        }
        row
    }

    /// The span less the tunes and the compilations outside them: loading,
    /// running, everything the records do not name.
    pub fn other(&self) -> Duration {
        self.span
            .saturating_sub(self.tuning)
            .saturating_sub(self.compiling.saturating_sub(self.compiling_in_tunes))
    }
}

/// Every environment file of some directories, as one list.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Listing {
    pub directories: Vec<PathBuf>,
    /// Newest first, whichever directory holds it.
    pub environments: Vec<Summary>,
    /// The files that could not be read, and why.
    pub unreadable: Vec<Unreadable>,
}

/// A file a [`Listing`] found but could not open, or a directory it could
/// not read.
#[derive(Clone, Debug, Serialize)]
pub struct Unreadable {
    pub path: PathBuf,
    pub reason: String,
}
