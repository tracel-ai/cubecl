//! Removing from an environment what this executable's work never loaded.
//!
//! A tune races every candidate it measures, at the shape it tunes for, and
//! each candidate compiles kernels for that shape; the work then loads the
//! pick's kernels, for every shape it runs. What decides a kernel's fate is
//! therefore not the tune but the work: the
//! [loader](crate::compiler::KernelLoader) records each kernel the work loads
//! ([`WorkLaunchStore`]), whatever shape or variant it was launched for, and
//! never one only a race loaded. A prune keeps what was recorded and removes
//! the rest of this executable's compiled kernels, with every kernel another
//! executable compiled, and the build records the caller does not keep. The
//! autotune answers stay. Another executable's kernel is loaded by this one
//! only where a backend reuses an artifact by its source, as CUDA's and HIP's
//! do: such a kernel compiles again after a prune, and nothing else does.
//!
//! An environment that recorded nothing — one written before the work's
//! launches were recorded, or one no work ran in — has no kernel known unused,
//! and keeps every kernel this executable compiled.

use crate::compiler::{KernelCacheKey, NamespaceRoot, WorkLaunchStore, build_id_hash};
use alloc::vec::Vec;
use cubecl_common::hash::StableHash;
use cubecl_environment::collections::{HashMap, HashSet};
use cubecl_environment::environment::{
    EnvironmentError, EnvironmentId, PruneOptions, PruneTally, Pruned,
};
use cubecl_environment::persistence::{Access, Database};
use cubecl_environment::records::Records;

/// Pruning an environment: removing what this executable's work never loaded.
///
/// A trait rather than a method, because an [`EnvironmentId`] is declared
/// where the layout of the compilation stores is unknown: this crate reads it.
pub trait Prune {
    /// Remove the kernels this executable compiled and its work never loaded,
    /// those another executable compiled, and the build records `options`
    /// does not keep. The space they took is reclaimed by the next save, which
    /// copies what is left.
    ///
    /// Prune from the executable that runs the environment: the kernels any
    /// other one compiled are removed.
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

/// What a prune does with one compiled artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArtifactFate {
    /// Another executable compiled it: this one loads it only through a
    /// source it shares, and compiles it again otherwise.
    OtherBuild,
    /// This executable compiled it and its work never loaded it: only a race
    /// did.
    NeverLoaded,
    /// The work loaded it, or nothing says it did not.
    Kept,
}

impl ArtifactFate {
    fn removes(self) -> bool {
        match self {
            Self::OtherBuild | Self::NeverLoaded => true,
            Self::Kept => false,
        }
    }
}

/// The kernels this executable's work loaded in an environment.
#[derive(Debug)]
struct WorkLaunches {
    /// The build this executable is.
    build: StableHash,
    /// The kernels recorded, `None` when the environment recorded none.
    loaded: Option<HashSet<StableHash>>,
}

impl WorkLaunches {
    /// What `database` recorded of the work of `build`.
    fn read(database: &Database, build: StableHash) -> Self {
        let mut loaded = HashSet::default();
        for namespace in database.namespaces() {
            if NamespaceRoot::new(&namespace).as_str() != WorkLaunchStore::ROOT {
                continue;
            }
            database.scan(&namespace, &mut |key, _| {
                if let Ok(kernel) = ciborium::from_reader::<StableHash, _>(key) {
                    loaded.insert(kernel);
                }
            });
        }
        Self {
            build,
            loaded: (!loaded.is_empty()).then_some(loaded),
        }
    }

    fn fate(&self, artifact: &KernelCacheKey) -> ArtifactFate {
        if artifact.build_id != self.build {
            return ArtifactFate::OtherBuild;
        }
        match &self.loaded {
            Some(loaded) if !loaded.contains(&artifact.id) => ArtifactFate::NeverLoaded,
            Some(_) | None => ArtifactFate::Kept,
        }
    }

    /// Remove from `database` the artifacts whose [fate](Self::fate) is to
    /// go, counting by artifact: an artifact and the entries that map a
    /// source to it are one kernel.
    fn prune(&self, database: &Database) -> PruneTally {
        let mut kept: HashSet<KernelCacheKey> = HashSet::default();
        let mut removed: HashMap<KernelCacheKey, u64> = HashMap::default();
        for namespace in database.namespaces() {
            if !NamespaceRoot::new(&namespace).holds_artifacts() {
                continue;
            }
            let mut doomed = Vec::new();
            database.scan(&namespace, &mut |key, value| {
                let Some(artifact) = KernelCacheKey::of_entry(key, value) else {
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
    use alloc::vec;

    const BUILD: StableHash = 7;

    fn launches(loaded: Option<&[StableHash]>) -> WorkLaunches {
        WorkLaunches {
            build: BUILD,
            loaded: loaded.map(|loaded| loaded.iter().copied().collect()),
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
    fn a_kernel_the_work_loaded_is_kept() {
        assert_eq!(launches(Some(&[1])).fate(&artifact(1)), ArtifactFate::Kept);
    }

    #[test]
    fn a_kernel_only_a_race_loaded_goes() {
        assert_eq!(
            launches(Some(&[1])).fate(&artifact(2)),
            ArtifactFate::NeverLoaded
        );
    }

    /// The work's launch records the kernel, not the variant it was compiled
    /// in: every variant of a kernel the work loaded stays.
    #[test]
    fn every_variant_of_a_kernel_the_work_loaded_is_kept() {
        let in_variant = KernelCacheKey {
            variant: 5,
            ..artifact(1)
        };
        assert_eq!(launches(Some(&[1])).fate(&in_variant), ArtifactFate::Kept);
    }

    #[test]
    fn another_builds_kernel_goes_even_when_the_work_loaded_it() {
        let other = KernelCacheKey {
            build_id: BUILD + 1,
            ..artifact(1)
        };
        assert_eq!(launches(Some(&[1])).fate(&other), ArtifactFate::OtherBuild);
    }

    /// An environment that recorded no launch says nothing of its kernels:
    /// this build's stay.
    #[test]
    fn an_environment_that_recorded_nothing_keeps_this_builds_kernels() {
        assert_eq!(launches(None).fate(&artifact(9)), ArtifactFate::Kept);
        assert!(
            vec![ArtifactFate::OtherBuild, ArtifactFate::NeverLoaded]
                .into_iter()
                .all(ArtifactFate::removes)
        );
    }
}
