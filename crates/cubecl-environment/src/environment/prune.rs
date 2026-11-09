//! What pruning an environment removes, and what it reports.
//!
//! A build fills its environment with more than its work runs: every
//! candidate an autotune race measured compiled its kernels, at the shape it
//! was tuned for. Pruning keeps the kernels this build's work launched and
//! removes the rest of what it compiled, everything another build compiled,
//! and the build records the caller does not keep. The autotune answers
//! stay.
//!
//! These are values: what a caller asks for and what it gets back cross the
//! network unchanged, so a remote environment prunes where it lives. The
//! pruning itself is the runtime's, which knows the entries it stored.

/// What a prune removes beyond the kernels this build's work never launched.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PruneOptions {
    /// Which of the environment's build records stay.
    #[serde(default)]
    pub records: RecordPruning,
}

/// Which of an environment's build records a prune keeps.
///
/// Records are how the environment was built: which candidates each tune ran
/// and what each cost, which kernels compiled and for how long. Nothing reads
/// them to run a workload; a reader reads them to explain a build.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RecordPruning {
    /// None: the environment keeps what running needs and nothing else.
    #[default]
    RemoveAll,
    /// The records of the newest sessions.
    KeepLatest {
        /// How many sessions keep their records.
        sessions: u32,
    },
    /// Every one.
    KeepAll,
}

/// What a prune kept and removed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Pruned {
    /// The compiled kernels: an artifact and the entries that name it count
    /// as one kernel.
    pub kernels: PruneTally,
    /// The build records, each one entry.
    pub records: PruneTally,
}

/// How many entries of one kind a prune kept and removed, and the bytes the
/// removed ones took.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PruneTally {
    /// The entries kept.
    pub kept: u64,
    /// The entries removed.
    pub removed: u64,
    /// The bytes of the removed entries' keys and values.
    pub removed_bytes: u64,
}
