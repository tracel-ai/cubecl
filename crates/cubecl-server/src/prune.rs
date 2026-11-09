//! Removing from an environment what this build's work never launched.
//!
//! A tune races every candidate it measures, at the shape it tunes for, and
//! each candidate compiles kernels for that shape; the work then launches the
//! pick's kernels, for every shape it runs. What decides a kernel's fate is
//! therefore not the tune but the work: the
//! [loader](crate::compiler::KernelLoader) records each kernel the work
//! launches ([`WorkLaunchStore`]), whatever shape or variant it was launched
//! for, and never one only a race launched. A prune keeps what this build
//! recorded and removes the rest of this build's compiled kernels, every
//! kernel another build compiled with the records of what its work launched,
//! and the build records the caller does not keep. The autotune answers stay.
//!
//! "Build" is the compiled program, as [`build_id_hash`] names it: a kernel is
//! stored under the build that compiled it, and another build's is loaded
//! only where a backend reuses an artifact by its source, as CUDA's and HIP's
//! do; such a kernel compiles again after a prune. An environment in which
//! this build's work launched nothing keeps every kernel this build compiled.

use crate::compiler::{KernelCacheKey, NamespaceRoot, WorkLaunchStore, build_id_hash};
use alloc::string::String;
use alloc::vec::Vec;
use cubecl_common::hash::StableHash;
use cubecl_environment::collections::{HashMap, HashSet};
use cubecl_environment::environment::{
    EnvironmentError, EnvironmentId, PruneOptions, PruneTally, Pruned,
};
use cubecl_environment::persistence::{Access, Database};
use cubecl_environment::records::Records;

/// Pruning an environment: removing what this build's work never launched.
///
/// A trait rather than a method, because an [`EnvironmentId`] is declared
/// where the layout of the compilation stores is unknown: this crate reads it.
pub trait Prune {
    /// Remove the kernels this build compiled and its work never launched,
    /// those another build compiled, and the build records `options` does not
    /// keep. The space they took is reclaimed by the next save, which copies
    /// what is left.
    ///
    /// Prune from the build that runs the environment, once its work has run
    /// with its tunes settled: a work launch made while a key is still
    /// untuned runs whichever candidate comes first, and a pruned environment
    /// keeps that candidate rather than the pick a later tune measures.
    ///
    /// # Errors
    ///
    /// The environment lives in another process's storage, or its database
    /// does not open.
    fn prune(&self, options: PruneOptions) -> Result<Pruned, EnvironmentError>;
}

impl Prune for EnvironmentId {
    fn prune(&self, options: PruneOptions) -> Result<Pruned, EnvironmentError> {
        let database = self.database(Access::ReadWrite)?;
        let launches = WorkLaunches::read(&database, build_id_hash());
        let kernels = launches.prune(&database);
        let records = Records::new(&database).retain(options.records);
        Ok(Pruned { kernels, records })
    }
}

/// What an environment's records say of one compiled artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArtifactFate {
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
    /// Whether a prune removes the artifact: what no run of this build will
    /// load again. With no launch recorded, nothing says that of any kernel.
    fn removes(self) -> bool {
        match self {
            Self::OtherBuild | Self::RaceOnly => true,
            Self::WorkLaunched | Self::NoLaunchRecorded => false,
        }
    }
}

/// What an environment recorded of the kernels this build's work launched.
#[derive(Debug)]
struct WorkLaunches {
    /// The build this program is.
    build: StableHash,
    /// The kernels recorded, `None` when this build's work launched none here.
    launched: Option<HashSet<StableHash>>,
    /// The namespaces recording other builds' launches, removed with those
    /// builds' kernels.
    other_builds: Vec<String>,
}

impl WorkLaunches {
    /// What `database` recorded of the work of `build`.
    fn read(database: &Database, build: StableHash) -> Self {
        let this_build = WorkLaunchStore::namespace(build);
        let mut launched = HashSet::default();
        let mut other_builds = Vec::new();
        for namespace in database.namespaces() {
            if NamespaceRoot::new(&namespace).as_str() != WorkLaunchStore::ROOT {
                continue;
            }
            if namespace != this_build.as_str() {
                other_builds.push(namespace);
                continue;
            }
            database.scan(&namespace, &mut |key, _| {
                if let Ok(kernel) = ciborium::from_reader::<StableHash, _>(key) {
                    launched.insert(kernel);
                }
            });
        }
        Self {
            build,
            launched: (!launched.is_empty()).then_some(launched),
            other_builds,
        }
    }

    /// What the records say of `artifact`.
    fn fate(&self, artifact: &KernelCacheKey) -> ArtifactFate {
        if artifact.build_id != self.build {
            return ArtifactFate::OtherBuild;
        }
        match &self.launched {
            Some(launched) if launched.contains(&artifact.id) => ArtifactFate::WorkLaunched,
            Some(_) => ArtifactFate::RaceOnly,
            None => ArtifactFate::NoLaunchRecorded,
        }
    }

    /// Remove from `database` the artifacts whose [fate](Self::fate) is to
    /// go, counting by artifact: an artifact and the entries that map a
    /// source to it are one kernel. Other builds' records of their work go
    /// with their kernels.
    fn prune(&self, database: &Database) -> PruneTally {
        let mut kept: HashSet<KernelCacheKey> = HashSet::default();
        let mut removed: HashMap<KernelCacheKey, u64> = HashMap::default();
        for namespace in database.namespaces() {
            if !NamespaceRoot::new(&namespace).holds_artifacts() {
                continue;
            }
            let mut doomed = Vec::new();
            database.scan(&namespace, &mut |key, value| {
                let Some(artifact) = KernelCacheKey::from_entry(key, value) else {
                    return;
                };
                match self.fate(&artifact).removes() {
                    true => {
                        *removed.entry(artifact).or_default() += (key.len() + value.len()) as u64;
                        doomed.push(key.to_vec());
                    }
                    false => {
                        kept.insert(artifact);
                    }
                }
            });
            for key in doomed {
                database.purge_key(&namespace, &key);
            }
        }
        for namespace in &self.other_builds {
            database.purge(namespace);
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

    const BUILD: StableHash = 7;

    fn launches(launched: Option<&[StableHash]>) -> WorkLaunches {
        WorkLaunches {
            build: BUILD,
            launched: launched.map(|launched| launched.iter().copied().collect()),
            other_builds: Vec::new(),
        }
    }

    fn artifact(id: StableHash) -> KernelCacheKey {
        KernelCacheKey {
            id,
            variant: 0,
            build_id: BUILD,
        }
    }

    #[test]
    fn a_kernel_the_work_launched_is_kept() {
        let fate = launches(Some(&[1])).fate(&artifact(1));
        assert_eq!(fate, ArtifactFate::WorkLaunched);
        assert!(!fate.removes());
    }

    #[test]
    fn a_kernel_only_a_race_launched_goes() {
        let fate = launches(Some(&[1])).fate(&artifact(2));
        assert_eq!(fate, ArtifactFate::RaceOnly);
        assert!(fate.removes());
    }

    /// The work's launch records the kernel, not the variant it was compiled
    /// in: every variant of a kernel the work launched stays.
    #[test]
    fn every_variant_of_a_kernel_the_work_launched_is_kept() {
        let in_variant = artifact(1).in_variant(5);
        let fate = launches(Some(&[1])).fate(&in_variant);
        assert_eq!(fate, ArtifactFate::WorkLaunched);
    }

    #[test]
    fn another_builds_kernel_goes_even_when_the_work_launched_it() {
        let other = KernelCacheKey {
            build_id: BUILD + 1,
            ..artifact(1)
        };
        let fate = launches(Some(&[1])).fate(&other);
        assert_eq!(fate, ArtifactFate::OtherBuild);
        assert!(fate.removes());
    }

    /// An environment in which this build's work launched nothing says
    /// nothing of its kernels: this build's stay.
    #[test]
    fn an_environment_with_no_launch_of_this_build_keeps_its_kernels() {
        let fate = launches(None).fate(&artifact(9));
        assert_eq!(fate, ArtifactFate::NoLaunchRecorded);
        assert!(!fate.removes());
    }
}
