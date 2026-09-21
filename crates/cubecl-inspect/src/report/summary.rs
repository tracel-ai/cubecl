use super::AutotuneTable;
use cubecl_environment::bundle::BundleManifest;
use cubecl_environment::records::Session;
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
    /// The processes that changed the file, oldest first — its builds, a
    /// warm-up that found everything done leaving none.
    pub sessions: Vec<SessionRow>,
    /// What the application that wrote the file says it is for. The reader
    /// knows no application's vocabulary and leaves it empty; one that does
    /// fills it before printing.
    pub description: Option<String>,
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
    /// What those cost, summed.
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

impl NamespaceRow {
    /// The first segment: who wrote the namespace — `autotune`, a backend's
    /// compiler, an application.
    pub fn root(&self) -> &str {
        self.namespace.split('/').next().unwrap_or_default()
    }
}

impl Summary {
    /// The namespaces folded by [root](NamespaceRow::root), each row named
    /// for the root it totals.
    pub fn roots(&self) -> Vec<NamespaceRow> {
        let mut roots = BTreeMap::<&str, NamespaceRow>::new();
        for namespace in &self.namespaces {
            let root = roots
                .entry(namespace.root())
                .or_insert_with(|| NamespaceRow {
                    namespace: namespace.root().to_string(),
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
            .filter(|row| row.root() == AutotuneTable::ROOT)
            .map(|row| row.entries)
            .sum()
    }
}

impl SessionRow {
    /// The span less the tunes and the compilations outside them: loading,
    /// running, everything the records do not name.
    pub fn other(&self) -> Duration {
        self.span
            .saturating_sub(self.tuning)
            .saturating_sub(self.compiling.saturating_sub(self.compiling_in_tunes))
    }
}

/// Every environment file of one directory.
#[derive(Clone, Debug, Serialize)]
pub struct Listing {
    pub directory: PathBuf,
    /// Newest first.
    pub environments: Vec<Summary>,
    /// The files that could not be read, and why.
    pub unreadable: Vec<Unreadable>,
}

/// A file a [`Listing`] found but could not open.
#[derive(Clone, Debug, Serialize)]
pub struct Unreadable {
    pub path: PathBuf,
    pub reason: String,
}

/// What [compacting](crate::Inspector::compact) an environment for a workload
/// kept and dropped.
#[derive(Clone, Debug, Serialize)]
pub struct Compaction {
    /// The compact copy.
    pub summary: Summary,
    /// Compiled kernels the replay launched, kept with their store entries.
    pub kept_kernels: u64,
    pub kept_bytes: u64,
    /// Compiled kernels it never launched, dropped.
    pub dropped_kernels: u64,
    pub dropped_bytes: u64,
    /// Kernels the replay launched that the file stores no artifact for: a
    /// backend that keeps none, or a build that did not cover the workload.
    pub unstored: u64,
}
