//! What pruning an environment removes, and what it reports.
//!
//! A build fills its environment with more than its work runs: every
//! candidate an autotune race measured compiled its kernels, at the shape it
//! was tuned for. Pruning keeps the kernels this build's work launched and
//! removes the rest of what it compiled, everything another build compiled,
//! and the build records the caller does not keep. The tunes stay
//! unless the caller asks for those the work did not use to go.
//!
//! These are values: what a caller asks for and what it gets back cross the
//! network unchanged, so a remote environment prunes where it lives. The
//! pruning itself is the runtime's, which knows the entries it stored.

/// What a prune removes beyond the kernels this build's work never launched.
///
/// Built with [`new`](Self::new) and a setter per option, so an option added
/// later breaks no caller.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct PruneOptions {
    /// Which of the environment's build records stay.
    #[serde(default)]
    pub records: RecordPruning,
    /// Which of the environment's tunes stay.
    #[serde(default)]
    pub tunes: TunePruning,
}

impl PruneOptions {
    /// The default prune: every build record goes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Keep the build records `records` says.
    pub fn records(mut self, records: RecordPruning) -> Self {
        self.records = records;
        self
    }

    /// Keep the tunes `tunes` says.
    pub fn tunes(mut self, tunes: TunePruning) -> Self {
        self.tunes = tunes;
        self
    }
}

/// Which of an environment's tunes a prune keeps.
///
/// Each tuner records the tunes this build's work loaded or stored, per
/// table. An environment may hold more: tunes brought from elsewhere, and
/// those of a key the work no longer reaches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TunePruning {
    /// Every one.
    #[default]
    KeepAll,
    /// In each table this build's work used, the tunes it used; every
    /// other table whole, since it may be another hardware's, another cubecl
    /// version's or a tuner's this build never ran. An environment in which
    /// this build used no tune keeps them all.
    KeepUsed,
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
    /// The tunes, each one entry: none counted unless the prune
    /// was asked to [keep only the used ones](TunePruning::KeepUsed), and
    /// then only those of the tables this build used.
    #[serde(default)]
    pub tunes: PruneTally,
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

impl PruneTally {
    /// The entries the prune saw: those kept and those removed.
    pub fn total(&self) -> u64 {
        self.kept + self.removed
    }
}

/// Several prunes' tallies of one kind, summed: every environment of a
/// directory, or kernels and records together.
impl core::ops::AddAssign for PruneTally {
    fn add_assign(&mut self, other: Self) {
        self.kept += other.kept;
        self.removed += other.removed;
        self.removed_bytes += other.removed_bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tallies_sum_field_by_field() {
        let mut sum = PruneTally {
            kept: 1,
            removed: 2,
            removed_bytes: 30,
        };
        sum += PruneTally {
            kept: 4,
            removed: 5,
            removed_bytes: 60,
        };
        assert_eq!(
            sum,
            PruneTally {
                kept: 5,
                removed: 7,
                removed_bytes: 90,
            }
        );
        assert_eq!(sum.total(), 12);
    }
}
