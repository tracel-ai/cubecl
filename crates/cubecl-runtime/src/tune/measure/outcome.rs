use alloc::string::String;
use cubecl_common::benchmark::BenchmarkComputations;
#[cfg(serializable)]
use serde::{Deserialize, Serialize};

use crate::tune::AutotuneError;

/// One candidate's measurement, as a tune ranks it.
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
#[derive(new, Debug, Clone, PartialEq, Eq)]
pub struct AutotuneOutcome {
    /// The candidate's name.
    pub name: String,
    /// The candidate's index in its set.
    pub index: usize,
    /// What its samples measured.
    pub computation: BenchmarkComputations,
}

impl core::fmt::Display for AutotuneOutcome {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Autotune[{}] name {} => {:?}",
            self.index, self.name, self.computation
        )
    }
}

/// One candidate's result in a tune.
///
/// # Guarantees
///
/// - Two results compare equal when both measured equally, or when both failed, whatever they
///   failed of.
#[cfg_attr(serializable, derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct AutotuneResult {
    /// What the candidate measured, or why it has no measurement.
    pub outcome: Result<AutotuneOutcome, AutotuneError>,
}

impl AutotuneResult {
    pub(crate) fn from_error(error: AutotuneError) -> Self {
        Self {
            outcome: Err(error),
        }
    }

    pub(crate) fn from_outcome(outcome: AutotuneOutcome) -> Self {
        Self {
            outcome: Ok(outcome),
        }
    }

    /// What the candidate measured, if it measured.
    pub(crate) fn measured(&self) -> Option<&AutotuneOutcome> {
        self.outcome.as_ref().ok()
    }

    /// The score a tune that compares the results ranks this one by, the worst there is where it
    /// failed.
    pub(crate) fn score(&self) -> u64 {
        self.outcome
            .as_ref()
            .map_or(u64::MAX, |outcome| outcome.computation.score())
    }

    /// Whether the candidate executed and only its timing failed.
    pub(crate) fn executed_unmeasured(&self) -> bool {
        self.outcome
            .as_ref()
            .is_err_and(AutotuneError::executed_unmeasured)
    }
}

/// The index of the measured result of `results` that scores best: the first of them among
/// equals.
pub fn best_scored<'a>(results: impl Iterator<Item = &'a AutotuneResult>) -> Option<usize> {
    results
        .filter_map(AutotuneResult::measured)
        .min_by_key(|outcome| outcome.computation.score())
        .map(|outcome| outcome.index)
}

impl Eq for AutotuneResult {}
impl PartialEq for AutotuneResult {
    fn eq(&self, other: &Self) -> bool {
        match (&self.outcome, &other.outcome) {
            (Ok(lhs), Ok(rhs)) => lhs == rhs,
            (Ok(_), Err(_)) => false,
            (Err(_), Ok(_)) => false,
            // We don't have to check the error
            (Err(_), Err(_)) => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failure() -> AutotuneResult {
        AutotuneResult::from_error(AutotuneError::Skip {
            name: "skipped".into(),
        })
    }

    #[test]
    fn a_failed_result_scores_worst() {
        assert_eq!(failure().score(), u64::MAX);
    }

    #[test]
    fn two_failures_are_equal() {
        assert_eq!(failure(), failure());
    }
}
