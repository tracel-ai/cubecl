use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;
use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations};
use cubecl_common::profile::ProfileDuration;

#[cfg(not(target_family = "wasm"))]
use cubecl_common::device::ServiceId;

use crate::tune::cache::tracker::TuneAnswer;
use crate::tune::drive::settler::TuneSettler;
use crate::tune::log::tune_logger::TuneLogger;
use crate::tune::measure::outcome::best_scored;
use crate::tune::measure::sample_time::measured;
use crate::tune::{AutotuneError, AutotuneKeyable, AutotuneOutcome, AutotuneResult};

/// One candidate of a fixed-count batch, launched.
pub enum LaunchedBench {
    /// Its samples are queued.
    Queued(QueuedBench),
    /// It failed to launch, so its result is known already.
    Failed(ResolvedBench),
}

#[cfg(target_family = "wasm")]
impl LaunchedBench {
    /// Whether its samples are queued.
    pub fn is_queued(&self) -> bool {
        matches!(self, Self::Queued(_))
    }

    /// Its result, where it failed to launch.
    pub fn into_failed(self) -> Option<ResolvedBench> {
        match self {
            Self::Queued(_) => None,
            Self::Failed(resolved) => Some(resolved),
        }
    }
}

/// A candidate whose samples are queued and not resolved yet.
pub struct QueuedBench {
    pub index: usize,
    pub name: String,
    /// The candidate's samples, in the order they were queued (Arrival Usage).
    pub profiles: Vec<ProfileDuration>,
    /// Time spent launching, when steps are being logged. The samples are still unresolved at that
    /// point, so the resolution wait is added before it is reported.
    pub launch: Option<Duration>,
}

/// A candidate's result, and the step it logs.
pub struct ResolvedBench {
    pub index: usize,
    /// The candidate's name and what it cost, when steps are being logged.
    pub step: Option<(String, Duration)>,
    pub result: AutotuneResult,
}

/// Commits a tune: resolves the samples still queued, picks the winner, and hands it to the
/// [`TuneSettler`]. Holds no borrow, so the browser event loop can run it.
pub struct TuneCommitter<K: AutotuneKeyable> {
    key: K,
    /// Each candidate's result, by tunable index (Position Usage).
    results: Vec<AutotuneResult>,
    logger: TuneLogger,
    settler: TuneSettler<K>,
}

impl<K: AutotuneKeyable> TuneCommitter<K> {
    /// A committer of the tune of `key`, every one of `names` skipped until its result is
    /// recorded.
    pub fn new<'n>(
        key: K,
        names: impl Iterator<Item = &'n str>,
        logger: TuneLogger,
        settler: TuneSettler<K>,
    ) -> Self {
        Self {
            key,
            results: names
                .map(|name| AutotuneResult::from_error(AutotuneError::Skip { name: name.into() }))
                .collect(),
            logger,
            settler,
        }
    }

    pub fn results(&self) -> &[AutotuneResult] {
        &self.results
    }

    #[cfg(not(target_family = "wasm"))]
    /// Every candidate's result, for a strategy that writes them in place.
    pub fn results_mut(&mut self) -> &mut [AutotuneResult] {
        &mut self.results
    }

    /// [`record_bench`] into the committer's logger and results.
    pub fn record_bench(&mut self, bench: ResolvedBench) {
        record_bench(&mut self.logger, &mut self.results, bench);
    }

    #[cfg(not(target_family = "wasm"))]
    /// [`TuneLogger::log_step`].
    pub fn log_step(&mut self, name: String, duration: Duration) {
        self.logger.log_step(name, duration);
    }

    #[cfg(not(target_family = "wasm"))]
    /// [`TuneLogger::log_short_circuit`].
    pub fn log_short_circuit(&mut self, name: String) {
        self.logger.log_short_circuit(name);
    }

    #[cfg(not(target_family = "wasm"))]
    /// Stop the tune on the lost card `service`, leaving the key to another card.
    pub fn stop_on_lost_card(self, service: ServiceId) -> TuneAnswer {
        self.settler.stop_on_lost_card(service)
    }

    /// Commit the tune, `pending` the benches of its last batch still queued and `decided` the
    /// winner its strategy picked, if it picked one: inline natively, on the browser event loop on
    /// wasm, where it answers [`Pending`](TuneAnswer::Pending) until then.
    pub fn commit(self, pending: Vec<LaunchedBench>, decided: Option<usize>) -> TuneAnswer {
        #[cfg(target_family = "wasm")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                self.commit_resolved(pending, decided).await;
            });
            TuneAnswer::Pending
        }

        #[cfg(not(target_family = "wasm"))]
        {
            cubecl_environment::future::block_on(self.commit_resolved(pending, decided))
        }
    }
}

impl<K: AutotuneKeyable> TuneCommitter<K> {
    async fn commit_resolved(
        mut self,
        pending: Vec<LaunchedBench>,
        decided: Option<usize>,
    ) -> TuneAnswer {
        let resolved = resolve_benches(pending).await;
        record_benches(&mut self.logger, &mut self.results, resolved);

        // A winner whose own outcome is an error is one picked with nothing measured: the tune
        // executed but could not be timed.
        let decided_index = decided
            .or_else(|| best_scored(self.results.iter()))
            .unwrap_or_else(|| unmeasured_pick(&self.key, &self.results));
        let unmeasured = self.results[decided_index].outcome.is_err();
        self.settler.settle(
            self.key,
            self.results,
            decided_index,
            unmeasured,
            &self.logger,
        );

        TuneAnswer::Hit {
            fastest_index: decided_index,
        }
    }
}

/// Resolve every bench of `pending`, together, in launch order.
///
/// Each benchmark is timed on its own rather than timing the loop: the profiles were all queued
/// before any of them was polled, so awaiting them in turn would charge the first benchmark for
/// draining the whole device queue and report the rest as free.
pub async fn resolve_benches(
    pending: impl IntoIterator<Item = LaunchedBench>,
) -> Vec<ResolvedBench> {
    futures_util::future::join_all(pending.into_iter().map(resolve_bench)).await
}

/// What `bench` measured: its samples resolved where they were queued.
pub async fn resolve_bench(bench: LaunchedBench) -> ResolvedBench {
    match bench {
        LaunchedBench::Queued(queued) => resolve_queued(queued).await,
        LaunchedBench::Failed(resolved) => resolved,
    }
}

/// [`record_bench`] every bench of `resolved`, in order.
pub fn record_benches(
    logger: &mut TuneLogger,
    results: &mut [AutotuneResult],
    resolved: Vec<ResolvedBench>,
) {
    for bench in resolved {
        record_bench(logger, results, bench);
    }
}

/// Log the step of `bench`, and keep its result in `results` at its index.
pub fn record_bench(logger: &mut TuneLogger, results: &mut [AutotuneResult], bench: ResolvedBench) {
    if let Some((name, duration)) = bench.step {
        logger.log_step(name, duration);
    }
    results[bench.index] = bench.result;
}

/// The candidate a tune that measured nothing decides: the first one that executed but could not
/// be measured.
///
/// A timing failure — a profiling hiccup, timestamp query sets on a busy stream — leaves a usable
/// kernel: deciding it unmeasured beats panicking the device thread, since a dead tune poisons
/// every stream sharing it, and the decided kernel executes for real right after, where it fails
/// exactly as a measured winner's would if it truly cannot run. A rejection, a panic, a launch
/// failure, a skip or an unheld device says the kernel did not run or runs wrong: deciding it
/// would fail the same way.
///
/// # Panics
///
/// When no candidate of `results` executed: some candidate the plan schedules executes on the device
/// and only its timing fails, or one measures.
pub fn unmeasured_pick<K: core::fmt::Debug>(key: &K, results: &[AutotuneResult]) -> usize {
    let index = results
        .iter()
        .position(AutotuneResult::executed_unmeasured)
        .unwrap_or_else(|| {
            panic!(
                "Can't execute the autotune plan for key: {key:?}: some candidate the plan schedules should execute on the device\n - results: {results:?}"
            )
        });
    log::warn!(
        "Autotune measured no candidate for key {key:?}; deciding candidate {index} unmeasured.\n - results: {results:?}"
    );
    index
}

/// Await every sample of a queued benchmark and fold them into one result, its step charged the
/// resolution wait.
///
/// The samples are resolved concurrently: a profile only submits its readback when first polled,
/// so awaiting them one by one would serialize a device round-trip per sample.
async fn resolve_queued(bench: QueuedBench) -> ResolvedBench {
    let started = cubecl_common::profile::Instant::now();
    let QueuedBench {
        index,
        name,
        profiles,
        launch,
    } = bench;
    let result = resolve_samples(index, name.clone(), profiles).await;
    ResolvedBench {
        index,
        step: launch.map(|launch| (name, launch + started.elapsed())),
        result,
    }
}

/// What the samples `profiles` of the candidate `name`, at `index`, measured.
async fn resolve_samples(
    index: usize,
    name: String,
    profiles: Vec<ProfileDuration>,
) -> AutotuneResult {
    let timing_method = profiles
        .first()
        .expect("a queued bench should hold a sample: a benchmark that queued answers at least one")
        .timing_method();

    // One sample that is no time disqualifies the candidate, the way one failed sample does.
    // Dropping it and averaging the rest would let a candidate whose window went untimed be
    // judged on its remaining runs.
    let Some(durations) =
        futures_util::future::join_all(profiles.into_iter().map(ProfileDuration::resolve))
            .await
            .into_iter()
            .map(measured)
            .collect::<Option<Vec<Duration>>>()
    else {
        return AutotuneResult::from_error(AutotuneError::NotMeasured { name });
    };

    AutotuneResult::from_outcome(AutotuneOutcome::new(
        name,
        index,
        BenchmarkComputations::new(&BenchmarkDurations::from_durations(
            timing_method,
            durations,
        )),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failed(error: AutotuneError) -> AutotuneResult {
        AutotuneResult::from_error(error)
    }

    fn unknown() -> AutotuneError {
        AutotuneError::Unknown {
            name: "untimed".into(),
            err: "no timestamp query".into(),
        }
    }

    #[test]
    fn a_candidate_that_panicked_is_never_decided_unmeasured() {
        let panicked = failed(AutotuneError::Panicked {
            name: "panicked".into(),
            message: "the block divides the vector".into(),
        });

        assert_eq!(unmeasured_pick(&"key", &[panicked, failed(unknown())]), 1);
    }

    #[test]
    #[should_panic(expected = "should execute on the device")]
    fn a_plan_where_nothing_executed_panics() {
        let launch = failed(AutotuneError::HoldFailed {
            name: "unheld".into(),
            reason: "the device thread is gone".into(),
        });

        unmeasured_pick(&"key", &[launch]);
    }

    #[test]
    fn a_bench_that_failed_to_launch_resolves_to_its_result() {
        let failure = ResolvedBench {
            index: 0,
            step: None,
            result: failed(unknown()),
        };

        let resolved =
            cubecl_environment::future::block_on(resolve_bench(LaunchedBench::Failed(failure)));

        assert!(resolved.result.outcome.is_err());
    }
}

#[cfg(test)]
mod tests_with_logger;
