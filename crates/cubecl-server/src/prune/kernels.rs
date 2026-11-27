//! A prune of the compiled kernels, by the ones the loader recorded the work
//! launching.

use super::Prune;
use crate::compiler::{KernelCacheKey, NamespaceRoot};
use alloc::vec::Vec;
use cubecl_environment::collections::{HashMap, HashSet};
use cubecl_environment::environment::{
    BuildStamp, EnvironmentError, EnvironmentId, PruneOptions, PruneTally, Pruned, RecordedUses,
    UsedContent,
};
use cubecl_environment::persistence::{Access, Database};
use cubecl_environment::records::Records;

impl Prune for EnvironmentId {
    fn prune(&self, options: PruneOptions) -> Result<Pruned, EnvironmentError> {
        let database = self.database(Access::ReadWrite)?;
        let build = BuildStamp::current();
        let uses = RecordedUses::read(&database, build);
        let kernels = KernelPruner::new(build).prune(&database, &uses);
        uses.purge_other_builds(&database, &UsedContent::Kernels);
        let records = Records::new(&database).retain(options.records);
        Ok(Pruned { kernels, records })
    }
}

/// What an environment's records say of one compiled artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactFate {
    /// Another build compiled it.
    OtherBuild,
    /// This build compiled it, and its work launched other kernels but not
    /// this one: only an autotune race did.
    RaceOnly,
    /// This build compiled it and its work launched it, in some variant.
    WorkLaunched,
    /// This build compiled it, and nothing records what its work launched.
    NoLaunchRecorded,
}

impl ArtifactFate {
    /// The fate of an artifact, from whether this build compiled it, whether
    /// this build recorded any kernel its work launched, and whether that
    /// record holds the artifact's kernel.
    pub fn of(built_by_this_build: bool, recorded: bool, launched: bool) -> Self {
        match (built_by_this_build, recorded, launched) {
            (false, _, _) => Self::OtherBuild,
            (true, false, _) => Self::NoLaunchRecorded,
            (true, true, true) => Self::WorkLaunched,
            (true, true, false) => Self::RaceOnly,
        }
    }

    /// Whether a prune removes the artifact: what no run of this build will
    /// load again. With no launch recorded, nothing says that of any kernel.
    pub fn removes(self) -> bool {
        match self {
            Self::OtherBuild | Self::RaceOnly => true,
            Self::WorkLaunched | Self::NoLaunchRecorded => false,
        }
    }
}

/// Removes from an environment the compiled artifacts whose
/// [fate](ArtifactFate) is to go.
#[derive(Debug)]
pub struct KernelPruner {
    /// The build this program is.
    build: BuildStamp,
    /// The artifacts of the namespace being pruned that go, with the key of
    /// their entry and the bytes it takes: filled while the namespace is
    /// scanned and emptied once it is (Scratch).
    doomed: Vec<(KernelCacheKey, Vec<u8>, u64)>,
}

impl KernelPruner {
    /// A pruner for the artifacts of `build`.
    pub fn new(build: BuildStamp) -> Self {
        Self {
            build,
            doomed: Vec::new(),
        }
    }

    /// Remove from `database` the artifacts whose fate, by what `uses`
    /// records of this build's work, is to go, counting by artifact: an
    /// artifact and the entries that map a source to it are one kernel.
    pub fn prune(&mut self, database: &Database, uses: &RecordedUses) -> PruneTally {
        let recorded = uses.recorded(&UsedContent::Kernels);
        let mut kept: HashSet<KernelCacheKey> = HashSet::default();
        let mut removed: HashMap<KernelCacheKey, u64> = HashMap::default();
        let artifacts = database
            .namespaces()
            .into_iter()
            .filter(|namespace| NamespaceRoot::new(namespace).holds_artifacts());
        for namespace in artifacts {
            self.doomed.clear();
            database.scan(&namespace, &mut |key, value| {
                let Some(artifact) = KernelCacheKey::from_entry(key, value) else {
                    return;
                };
                let fate = ArtifactFate::of(
                    artifact.built_by == self.build,
                    recorded,
                    uses.used(&UsedContent::Kernels, &artifact.id),
                );
                if fate.removes() {
                    self.doomed
                        .push((artifact, key.to_vec(), (key.len() + value.len()) as u64));
                } else {
                    kept.insert(artifact);
                }
            });
            for (artifact, key, bytes) in self.doomed.drain(..) {
                *removed.entry(artifact).or_default() += bytes;
                database.purge_key(&namespace, &key);
            }
        }
        PruneTally {
            kept: kept.len() as u64,
            removed: removed.len() as u64,
            removed_bytes: removed.values().sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn another_builds_artifact_is_another_builds_even_when_launched() {
        assert_eq!(
            ArtifactFate::of(false, true, true),
            ArtifactFate::OtherBuild
        );
    }

    /// An environment in which this build's work launched nothing says
    /// nothing of its kernels.
    #[test]
    fn with_no_launch_recorded_nothing_is_said_of_this_builds_artifact() {
        assert_eq!(
            ArtifactFate::of(true, false, false),
            ArtifactFate::NoLaunchRecorded
        );
    }

    #[test]
    fn an_artifact_of_a_kernel_the_work_launched_is_the_works() {
        assert_eq!(
            ArtifactFate::of(true, true, true),
            ArtifactFate::WorkLaunched
        );
    }

    #[test]
    fn an_artifact_of_a_kernel_the_work_never_launched_is_a_races() {
        assert_eq!(ArtifactFate::of(true, true, false), ArtifactFate::RaceOnly);
    }

    /// Only another build's artifacts and those only a race launched go.
    #[test]
    fn only_other_builds_and_race_only_artifacts_go() {
        let removed = [
            ArtifactFate::OtherBuild,
            ArtifactFate::RaceOnly,
            ArtifactFate::WorkLaunched,
            ArtifactFate::NoLaunchRecorded,
        ]
        .map(ArtifactFate::removes);

        assert_eq!(removed, [true, true, false, false]);
    }
}
