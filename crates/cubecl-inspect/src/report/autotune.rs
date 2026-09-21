use super::{AutotuneTable, KeyId};
use cubecl_environment::records::Stamped;
use cubecl_server::benchmark::BenchmarkComputations;
use cubecl_server::tune::{AutotuneError, AutotuneResult, Bounds, Trial, TuneRecord};
use serde::Serialize;
use std::time::Duration;

/// How a tune went, as the environment recorded it: the key decoded without
/// its type, like [`TunedKey::key`].
pub type TuneTrace = Stamped<TuneRecord<ciborium::Value>>;

/// A candidate that ran more than this many times the winner's score lost
/// clearly enough that the time it took is counted as
/// [wasted](TunedKey::wasted).
pub const WASTE_THRESHOLD: f64 = 2.0;

/// Every autotune key an environment holds an answer for.
#[derive(Clone, Debug, Serialize)]
pub struct AutotuneReport {
    pub keys: Vec<TunedKey>,
    /// Entries of an autotune namespace that did not decode as a tuned key: a
    /// cubecl whose value layout this build does not know.
    pub undecoded: u64,
}

/// One key's answer, and everything cubecl kept of how it was reached.
#[derive(Clone, Debug, Serialize)]
pub struct TunedKey {
    pub id: KeyId,
    pub table: AutotuneTable,
    /// The tuner's key, decoded without its type: a map of the fields the key
    /// derives `Serialize` for.
    pub key: ciborium::Value,
    /// The checksum of the candidate list the key was tuned under.
    pub checksum: String,
    /// The index of the candidate the key runs.
    pub winner: usize,
    /// Each candidate's result, in the order cubecl stored them: best score
    /// first, not the order they ran in.
    pub results: Vec<CandidateResult>,
    /// The throughput bounds the tune was held to, when the tuner declared
    /// any.
    pub bounds: Option<Bounds>,
    /// The time budget the tune ran under, when it had one.
    pub limit: Option<Duration>,
    /// The last recorded tune of this key: the trials in the order they ran,
    /// with their walls. `None` for an environment built before cubecl
    /// recorded its tunes.
    pub trace: Option<TuneTrace>,
    /// The part of the tune's wall spent compiling: the recorded compilations
    /// of its session that started while it ran. `None` when the tune was not
    /// recorded.
    pub compiling: Option<Duration>,
}

/// One candidate's part in a tune.
#[derive(Clone, Debug, Serialize)]
pub struct CandidateResult {
    /// `None` for a failure that does not name its candidate.
    pub candidate: Option<String>,
    /// The candidate's position in the tuner's list; cubecl keeps it only for
    /// a measured candidate.
    pub index: Option<usize>,
    pub outcome: CandidateOutcome,
}

/// How a candidate's tune ended.
#[derive(Clone, Debug, Serialize)]
pub enum CandidateOutcome {
    Measured(BenchmarkComputations),
    /// The tuner declined to run it for this key.
    Skipped,
    Failed {
        reason: String,
    },
}

impl CandidateResult {
    /// What a stored result says, named and classified.
    pub fn new(result: AutotuneResult) -> Self {
        match result.outcome {
            Ok(outcome) => Self {
                candidate: Some(outcome.name),
                index: Some(outcome.index),
                outcome: CandidateOutcome::Measured(outcome.computation),
            },
            Err(error) => {
                let (candidate, outcome) = failure(error);
                Self {
                    candidate,
                    index: None,
                    outcome,
                }
            }
        }
    }

    /// The tuner's score: what it ranks candidates by, lower is better.
    pub fn score(&self) -> Option<u64> {
        match &self.outcome {
            CandidateOutcome::Measured(computations) => Some(computations.score()),
            CandidateOutcome::Skipped | CandidateOutcome::Failed { .. } => None,
        }
    }
}

/// The candidate an error names, when it names one, and how its tune ended.
/// The reason is the error's own message, without the candidate's name that
/// `AutotuneError`'s `Display` leads with.
fn failure(error: AutotuneError) -> (Option<String>, CandidateOutcome) {
    let failed = |reason: String| CandidateOutcome::Failed { reason };
    match error {
        AutotuneError::Skip { name } => (Some(name), CandidateOutcome::Skipped),
        AutotuneError::Unknown { name, err } => (Some(name), failed(err)),
        AutotuneError::InvalidSamples { name } => {
            (Some(name), failed("all samples are invalid".to_string()))
        }
        AutotuneError::NotMeasured { name } => (
            Some(name),
            failed("a profiled sample carried no measurement".to_string()),
        ),
        AutotuneError::NoValidKernelFound { context } => (None, failed(context)),
        AutotuneError::Launch(error) => (None, failed(error.to_string())),
    }
}

impl TunedKey {
    /// The winner's own result. `None` when the winner was picked with nothing
    /// measured — the tune ran but its samples could not be timed.
    pub fn winning(&self) -> Option<&CandidateResult> {
        self.results
            .iter()
            .find(|result| result.index == Some(self.winner))
    }

    /// The winner's name, or its index when its result does not carry one.
    pub fn winner_name(&self) -> String {
        self.winning()
            .and_then(|result| result.candidate.clone())
            .unwrap_or_else(|| format!("#{}", self.winner))
    }

    /// The runner-up's score over the winner's: how close the race was. Near
    /// 1 means the candidates were interchangeable here; a wide margin means
    /// the answer was never in doubt and the race could have been shorter.
    /// `None` when fewer than two candidates were measured.
    pub fn margin(&self) -> Option<f64> {
        let winner = self.winning()?.score()?;
        let runner_up = self
            .results
            .iter()
            .filter(|result| result.index != Some(self.winner))
            .filter_map(CandidateResult::score)
            .min()?;
        Some(runner_up as f64 / winner.max(1) as f64)
    }

    /// The time the key's work takes at the device's modeled peak, launch
    /// overhead included: the floor no candidate can beat. `None` when the
    /// tuner declared no bounds.
    pub fn roofline(&self) -> Option<Duration> {
        let bounds = self.bounds.as_ref()?;
        let floor = bounds
            .bounds
            .iter()
            .filter_map(|bound| bound.resource.time_at_peak())
            .max()?;
        Some(floor + bounds.launch_overhead)
    }

    /// The roofline over the winner's median: 1 at the roofline, 0.25 when
    /// the answer runs four times slower than the device allows.
    pub fn efficiency(&self) -> Option<f64> {
        let CandidateOutcome::Measured(winner) = &self.winning()?.outcome else {
            return None;
        };
        let median = winner.median.as_secs_f64();
        if median <= 0.0 {
            return None;
        }
        Some(self.roofline()?.as_secs_f64() / median)
    }

    /// Whether the winner came in under the [limit](Self::limit): a tune
    /// that reaches it may stop before measuring every candidate.
    pub fn met_limit(&self) -> Option<bool> {
        let CandidateOutcome::Measured(winner) = &self.winning()?.outcome else {
            return None;
        };
        Some(winner.median <= self.limit?)
    }

    /// From the cache miss to the answer, when the tune was recorded.
    pub fn wall(&self) -> Option<Duration> {
        self.trace.as_ref().map(|trace| trace.record.wall)
    }

    /// The trials in the order they ran, when the tune was recorded.
    pub fn trials(&self) -> &[Trial] {
        self.trace
            .as_ref()
            .map_or(&[], |trace| trace.record.trials.as_slice())
    }

    /// The result of the candidate named `name`.
    pub fn result(&self, name: &str) -> Option<&CandidateResult> {
        self.results
            .iter()
            .find(|result| result.candidate.as_deref() == Some(name))
    }

    /// The candidate's score over the winner's, when both were measured.
    pub fn slowdown(&self, result: &CandidateResult) -> Option<f64> {
        let winner = self.winning()?.score()?;
        Some(result.score()?.max(1) as f64 / winner.max(1) as f64)
    }

    /// The wall of the trials that bought nothing: candidates that failed,
    /// and candidates that ran more than [`WASTE_THRESHOLD`] times the
    /// winner's score. `None` when the tune was not recorded.
    pub fn wasted(&self) -> Option<Duration> {
        self.trace.as_ref()?;
        Some(
            self.trials()
                .iter()
                .filter(|trial| {
                    self.result(&trial.name)
                        .is_some_and(|result| match &result.outcome {
                            CandidateOutcome::Failed { .. } => true,
                            CandidateOutcome::Measured(_) => self
                                .slowdown(result)
                                .is_some_and(|slowdown| slowdown > WASTE_THRESHOLD),
                            CandidateOutcome::Skipped => false,
                        })
                })
                .map(|trial| trial.wall)
                .sum(),
        )
    }

    /// How many candidates the tuner benchmarked for this key.
    pub fn measured(&self) -> usize {
        self.results
            .iter()
            .filter(|result| matches!(result.outcome, CandidateOutcome::Measured(_)))
            .count()
    }

    /// How many candidates failed for this key.
    pub fn failed(&self) -> usize {
        self.results
            .iter()
            .filter(|result| matches!(result.outcome, CandidateOutcome::Failed { .. }))
            .count()
    }
}

/// The orders an [`AutotuneReport`] lists its keys in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum KeyOrder {
    /// Slowest tune first; the keys whose tune was not recorded after them,
    /// most candidates measured first.
    #[default]
    Wall,
    /// Most candidates measured first.
    Trials,
    /// Widest margin first: the races whose answer was never in doubt.
    Margin,
    /// Grouped by tuner, as the file stores them.
    Table,
}

impl AutotuneReport {
    /// Put the keys in `order`.
    pub fn sort(&mut self, order: KeyOrder) {
        match order {
            KeyOrder::Wall => self.keys.sort_by_key(|key| {
                (
                    std::cmp::Reverse(key.wall()),
                    std::cmp::Reverse(key.measured()),
                    key.id,
                )
            }),
            KeyOrder::Trials => self
                .keys
                .sort_by_key(|key| (std::cmp::Reverse(key.measured()), key.id)),
            KeyOrder::Margin => self.keys.sort_by(|a, b| {
                b.margin()
                    .unwrap_or(0.0)
                    .total_cmp(&a.margin().unwrap_or(0.0))
                    .then(a.id.cmp(&b.id))
            }),
            KeyOrder::Table => self.keys.sort_by(|a, b| a.table.cmp(&b.table)),
        }
    }

    /// Keep only the keys of the tuners whose name contains `pattern`.
    pub fn retain_tuners(&mut self, pattern: &str) {
        self.keys.retain(|key| key.table.tuner.contains(pattern));
    }
}
