//! Removing from an environment what no workload of this build loads.
//!
//! An autotune race compiles every candidate it measures, and the stored
//! answer names the one that won: from then on only the winner's kernels are
//! launched for that key. Each stored result lists the kernels its candidate
//! launched, so the answers alone tell which compiled kernels only a losing
//! candidate ever used. Those go, with every kernel another build compiled,
//! which this build never loads, and the build records the caller does not
//! keep. Everything else stays: the answers, the winners' kernels, and every
//! kernel no tune raced.
//!
//! A kernel a loser launched that also runs outside any tune goes too, and
//! compiles again on its first launch: the answers cannot see it, and
//! pruning it costs one compile, never a wrong result.

use crate::compiler::{KernelCacheKey, build_id_hash};
use crate::tune::{PersistentCacheValue, ROOT as AUTOTUNE};
use alloc::vec::Vec;
use cubecl_common::hash::StableHash;
use cubecl_environment::collections::{HashMap, HashSet};
use cubecl_environment::environment::{
    EnvironmentError, EnvironmentId, PruneOptions, PruneTally, Pruned, RecordPruning,
};
use cubecl_environment::persistence::{Access, Database, NamespaceSummary};
use cubecl_environment::records::Records;

/// Pruning an environment: removing what no workload of this build loads.
pub trait Prune {
    /// Remove the kernels only a losing autotune candidate launched, those
    /// another build compiled, and the build records `options` does not keep.
    /// The space they took is reclaimed by the next save, which copies what
    /// is left.
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
        let kernels = prune_kernels(&database, build_id_hash());
        let records = prune_records(&database, options.records);
        Ok(Pruned { kernels, records })
    }
}

/// The kernels the autotune answers of an environment name: those a winner
/// launched, and those only a losing candidate did.
#[derive(Debug, Default)]
pub(crate) struct RacedKernels {
    won: HashSet<StableHash>,
    lost: HashSet<StableHash>,
}

impl RacedKernels {
    /// What `answers` name. A result counts as the winner's when it is the
    /// measured outcome of the answer's pick; every other result lost.
    pub(crate) fn new<'a>(answers: impl IntoIterator<Item = &'a PersistentCacheValue>) -> Self {
        let mut raced = Self::default();
        for answer in answers {
            for result in &answer.results {
                let won = result
                    .outcome
                    .as_ref()
                    .is_ok_and(|outcome| outcome.index == answer.fastest_index);
                let kernels = match won {
                    true => &mut raced.won,
                    false => &mut raced.lost,
                };
                kernels.extend(result.kernels.iter().copied());
            }
        }
        raced
    }

    /// Whether this build's workloads never load `artifact`: another build
    /// compiled it, or only losing candidates launched it.
    pub(crate) fn prunes(&self, artifact: &KernelCacheKey, build: StableHash) -> bool {
        artifact.build_id != build
            || (self.lost.contains(&artifact.id) && !self.won.contains(&artifact.id))
    }
}

/// Remove the kernels [`RacedKernels::prunes`] names, counting by artifact: an
/// artifact and the entries that map a source to it are one kernel.
fn prune_kernels(database: &Database, build: StableHash) -> PruneTally {
    let raced = RacedKernels::new(&answers(database));
    let mut kept: HashSet<KernelCacheKey> = HashSet::default();
    let mut removed: HashMap<KernelCacheKey, u64> = HashMap::default();
    for namespace in database.namespaces() {
        if is_autotune(&namespace) || Records::holds(&namespace) {
            continue;
        }
        let mut doomed = Vec::new();
        database.scan(&namespace, &mut |key, value| {
            // An artifact is keyed by its kernel; a second-line entry maps a
            // source hash to that key, and goes with the artifact.
            let Some(artifact) = decode::<KernelCacheKey>(key).or_else(|| decode(value)) else {
                return;
            };
            match raced.prunes(&artifact, build) {
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

/// Remove the build records `pruning` does not keep, counting entries.
fn prune_records(database: &Database, pruning: RecordPruning) -> PruneTally {
    let before = record_entries(database);
    let records = Records::new(database);
    match pruning {
        RecordPruning::RemoveAll => records.remove_all(),
        RecordPruning::KeepLatest(sessions) => {
            records.prune(sessions as usize);
        }
        RecordPruning::KeepAll => {}
    }
    let after = record_entries(database);
    PruneTally {
        kept: after.entries,
        removed: before.entries - after.entries,
        removed_bytes: before.bytes - after.bytes,
    }
}

/// Every autotune answer the environment holds that decodes.
fn answers(database: &Database) -> Vec<PersistentCacheValue> {
    let mut answers = Vec::new();
    for namespace in database.namespaces() {
        if !is_autotune(&namespace) {
            continue;
        }
        database.scan(&namespace, &mut |_, value| {
            if let Some(answer) = decode(value) {
                answers.push(answer);
            }
        });
    }
    answers
}

/// The entries and bytes of the build records, of every layout.
fn record_entries(database: &Database) -> NamespaceSummary {
    let mut total = NamespaceSummary {
        namespace: Default::default(),
        entries: 0,
        bytes: 0,
    };
    for summary in database.summary() {
        if Records::holds(&summary.namespace) {
            total.entries += summary.entries;
            total.bytes += summary.bytes;
        }
    }
    total
}

fn is_autotune(namespace: &str) -> bool {
    namespace.split('/').next() == Some(AUTOTUNE)
}

/// A stored key or value, `None` when it is something else. Entries of every
/// kind are tried, so one that is not of `T` is routine, not corruption.
fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    ciborium::from_reader(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::{AutotuneError, AutotuneOutcome, AutotuneResult};
    use alloc::string::ToString;
    use alloc::vec;
    use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations, TimingMethod};

    const BUILD: StableHash = 7;

    fn measured(index: usize, kernels: &[StableHash]) -> AutotuneResult {
        let durations = BenchmarkDurations::from_durations(
            TimingMethod::Device,
            vec![core::time::Duration::from_micros(10)],
        );
        AutotuneResult {
            outcome: Ok(AutotuneOutcome::new(
                index.to_string(),
                index,
                BenchmarkComputations::new(&durations),
            )),
            kernels: kernels.to_vec(),
        }
    }

    fn answer(fastest_index: usize, results: Vec<AutotuneResult>) -> PersistentCacheValue {
        PersistentCacheValue {
            fastest_index,
            results,
            bounds: None,
            limit: None,
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
    fn a_kernel_only_a_loser_launched_is_pruned() {
        let raced = RacedKernels::new(&[answer(0, vec![measured(0, &[1]), measured(1, &[2])])]);

        assert!(!raced.prunes(&artifact(1), BUILD));
        assert!(raced.prunes(&artifact(2), BUILD));
    }

    #[test]
    fn a_kernel_the_winner_also_launched_stays() {
        let raced =
            RacedKernels::new(&[answer(1, vec![measured(0, &[1, 3]), measured(1, &[2, 3])])]);

        assert!(raced.prunes(&artifact(1), BUILD));
        assert!(!raced.prunes(&artifact(3), BUILD));
    }

    /// A kernel one key's race lost is another key's winner: it stays.
    #[test]
    fn a_kernel_any_answer_picked_stays() {
        let raced = RacedKernels::new(&[
            answer(0, vec![measured(0, &[1]), measured(1, &[2])]),
            answer(1, vec![measured(0, &[1]), measured(1, &[2])]),
        ]);

        assert!(!raced.prunes(&artifact(1), BUILD));
        assert!(!raced.prunes(&artifact(2), BUILD));
    }

    /// A backend that keeps one kernel's artifacts apart by variant still
    /// names the kernel itself: each variant goes with the kernel.
    #[test]
    fn every_variant_of_a_losers_kernel_is_pruned() {
        let raced = RacedKernels::new(&[answer(0, vec![measured(0, &[1]), measured(1, &[2])])]);
        let variant = |id, variant| KernelCacheKey {
            id,
            variant,
            build_id: BUILD,
        };

        assert!(raced.prunes(&variant(2, 5), BUILD));
        assert!(!raced.prunes(&variant(1, 5), BUILD));
    }

    #[test]
    fn a_kernel_no_tune_raced_stays() {
        let raced = RacedKernels::new(&[answer(0, vec![measured(0, &[1]), measured(1, &[2])])]);

        assert!(!raced.prunes(&artifact(9), BUILD));
    }

    /// A candidate that failed lost, whatever it launched before failing.
    #[test]
    fn a_failed_candidates_kernels_are_pruned() {
        let failed = AutotuneResult {
            outcome: Err(AutotuneError::Skip {
                name: "failed".to_string(),
            }),
            kernels: vec![4],
        };
        let raced = RacedKernels::new(&[answer(0, vec![measured(0, &[1]), failed])]);

        assert!(raced.prunes(&artifact(4), BUILD));
    }

    /// An answer stored before results listed their kernels names none, and
    /// nothing is known to be a loser's.
    #[test]
    fn an_answer_naming_no_kernel_prunes_nothing() {
        let raced = RacedKernels::new(&[answer(0, vec![measured(0, &[]), measured(1, &[])])]);

        assert!(!raced.prunes(&artifact(1), BUILD));
        assert!(!raced.prunes(&artifact(2), BUILD));
    }

    #[test]
    fn another_builds_kernel_is_pruned() {
        let raced = RacedKernels::new(&[answer(0, vec![measured(0, &[1])])]);

        assert!(raced.prunes(
            &KernelCacheKey {
                id: 1,
                variant: 0,
                build_id: BUILD + 1,
            },
            BUILD
        ));
    }
}
