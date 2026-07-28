use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::time::Duration;

use cubecl_common::profile::{Instant, ProfileDuration, TimingMethod};

use crate::client::ComputeClient;
use crate::config::autotune::BenchConfig;
use crate::runtime::Runtime;
use crate::tune::sampler::SampleSet;
use crate::tune::tune_benchmark::{sample_once, warmup_once};
use crate::tune::{
    AutotuneError, AutotuneOutcome, AutotuneOutput, AutotuneResult, TuneFn, TuneInputs, TunePlan,
};

/// Candidates kept alive no matter how far behind they are, so a batch never narrows to a
/// single kernel that was never compared against anything.
const MIN_SURVIVORS: usize = 2;

/// One candidate kernel and the evidence gathered about it so far.
struct Candidate {
    index: usize,
    name: String,
    samples: SampleSet,
    method: Option<TimingMethod>,
    error: Option<AutotuneError>,
    elapsed: Duration,
    live: bool,
}

impl Candidate {
    fn new(index: usize, name: String) -> Self {
        Self {
            index,
            name,
            samples: SampleSet::default(),
            method: None,
            error: None,
            elapsed: Duration::ZERO,
            live: true,
        }
    }

    fn fail(&mut self, error: AutotuneError) {
        self.error = Some(error);
        self.live = false;
    }

    fn into_result(self) -> AutotuneResult {
        // A candidate that failed at any point is disqualified even if earlier samples
        // succeeded, so a kernel that crashes sporadically can never win on its good runs.
        if let Some(error) = self.error {
            return AutotuneResult::error(error);
        }

        if self.samples.is_empty() {
            return AutotuneResult::error(AutotuneError::Skip { name: self.name });
        }

        let computation = self
            .samples
            .computation(self.method.unwrap_or(TimingMethod::System));

        AutotuneResult::success(AutotuneOutcome::new(self.name, self.index, computation))
    }
}

/// The outcome of benchmarking one batch of the [`TunePlan`].
pub(crate) struct BatchOutcome {
    pub(crate) results: Vec<(usize, AutotuneResult)>,
    pub(crate) steps: Vec<(String, Duration)>,
    pub(crate) short_circuit: Option<String>,
    pub(crate) any_success: bool,
}

/// Round robin benchmarking with early elimination.
///
/// A first pass gives each candidate one warmup and one sample, resolved inline so a candidate
/// that already reaches the time limit can end the batch before the rest are compiled. After
/// that, every live candidate gets one sample per round and the whole round is resolved at once:
/// resolving per sample would serialize a device round trip per measurement, which for short
/// kernels costs more than the samples it saves.
///
/// The sample cap bounds the work at `max_samples` per candidate, so a batch can never cost more
/// than a fixed-sample-count pass over the same candidates.
pub(crate) struct Schedule {
    pub(crate) config: BenchConfig,
    pub(crate) limit: Option<Duration>,
    pub(crate) short_circuit: bool,
    pub(crate) track_steps: bool,
}

impl Schedule {
    /// Benchmark one batch of candidates, blocking until the batch is decided.
    ///
    /// Takes exclusive device access for the entire round robin: candidates are interleaved, so
    /// releasing the device between them would let unrelated work land in the middle of a
    /// measurement.
    pub(crate) fn run_batch<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        indices: Vec<usize>,
        autotunables: &[&TuneFn<F, Out>],
        inputs: <F as TuneInputs>::At<'a>,
        client: &ComputeClient<R>,
    ) -> BatchOutcome
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        let fallback = indices.clone();
        let run =
            || cubecl_common::future::block_on(self.drive(indices, autotunables, inputs, client));

        match client.clone().exclusive(run) {
            Ok(outcome) => outcome,
            Err(err) => BatchOutcome {
                results: fallback
                    .into_iter()
                    .map(|index| {
                        let error = AutotuneError::Unknown {
                            name: autotunables[index].name.to_string(),
                            err: err.to_string(),
                        };
                        (index, AutotuneResult::error(error))
                    })
                    .collect(),
                steps: Vec::new(),
                short_circuit: None,
                any_success: false,
            },
        }
    }

    async fn drive<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        indices: Vec<usize>,
        autotunables: &[&TuneFn<F, Out>],
        inputs: <F as TuneInputs>::At<'a>,
        client: &ComputeClient<R>,
    ) -> BatchOutcome
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        let (min_samples, max_samples) = self.config.samples();

        let mut candidates: Vec<Candidate> = indices
            .into_iter()
            .map(|index| Candidate::new(index, autotunables[index].name.to_string()))
            .collect();

        let mut short_circuit = None;

        // First pass: one warmup and one sample each, resolved inline so a candidate that
        // already hits the limit ends the batch before the remaining kernels are ever compiled.
        // This is the one place where paying a device round trip per sample is worth it.
        for slot in 0..candidates.len() {
            let launched = self.track_steps.then(Instant::now);
            let operation = autotunables[candidates[slot].index];
            let hit = self
                .first_pass(operation, &inputs, client, &mut candidates[slot])
                .await;

            if let Some(launched) = launched {
                candidates[slot].elapsed += launched.elapsed();
            }

            if hit {
                short_circuit = Some(candidates[slot].name.clone());
                break;
            }
        }

        // Later rounds resolve the whole round at once: a profile only submits its readback
        // when first polled, so resolving one at a time would serialize a device round trip
        // per measurement, which for short kernels costs more than the samples it saves.
        for round in 1..max_samples {
            if short_circuit.is_some() {
                break;
            }

            let live = candidates.iter().filter(|c| c.live).count();
            if live == 0 || (live == 1 && round >= min_samples) {
                break;
            }

            let mut pending = Vec::with_capacity(live);

            for (slot, candidate) in candidates.iter_mut().enumerate() {
                // A candidate that took confirmation samples in the first pass is ahead of the
                // others, so the ceiling is enforced per candidate rather than by round count.
                if !candidate.live || candidate.samples.len() >= max_samples {
                    continue;
                }

                let launched = self.track_steps.then(Instant::now);

                match sample_once(autotunables[candidate.index], inputs.clone(), client) {
                    Ok(profile) => {
                        candidate.method.get_or_insert(profile.timing_method());
                        pending.push((slot, profile));
                    }
                    Err(err) => candidate.fail(err),
                }

                if let Some(launched) = launched {
                    candidate.elapsed += launched.elapsed();
                }
            }

            if pending.is_empty() {
                break;
            }

            // Each profile is timed individually rather than timing the join: they resolve
            // concurrently, so the round's wall clock cannot be split across candidates, but the
            // wait for one candidate's own readback can.
            let (slots, profiles): (Vec<_>, Vec<_>) = pending.into_iter().unzip();
            let resolved = futures_util::future::join_all(profiles.into_iter().map(
                |profile: ProfileDuration| async {
                    let started = Instant::now();
                    let ticks = profile.resolve().await;
                    (ticks, started.elapsed())
                },
            ))
            .await;

            for (slot, (ticks, waited)) in slots.into_iter().zip(resolved) {
                candidates[slot].samples.push(ticks.duration());
                if self.track_steps {
                    candidates[slot].elapsed += waited;
                }
            }

            if let Some(name) = self.short_circuit_hit(&candidates) {
                short_circuit = Some(name);
                break;
            }

            if round + 1 >= min_samples {
                self.eliminate(&mut candidates, min_samples);

                let mut live = candidates.iter().filter(|c| c.live).peekable();
                if live.peek().is_some() && live.all(|c| c.samples.converged()) {
                    break;
                }
            }
        }

        let mut steps = Vec::new();
        let mut results = Vec::with_capacity(candidates.len());
        let mut any_success = false;

        for candidate in candidates {
            if self.track_steps {
                steps.push((candidate.name.clone(), candidate.elapsed));
            }
            let index = candidate.index;
            let result = candidate.into_result();
            any_success |= result.outcome.is_ok();
            results.push((index, result));
        }

        BatchOutcome {
            results,
            steps,
            short_circuit,
            any_success,
        }
    }

    /// Warm up a candidate and take its first sample, confirming on the spot if it already
    /// looks close enough to peak throughput. Returns whether the batch can stop here.
    async fn first_pass<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        operation: &TuneFn<F, Out>,
        inputs: &<F as TuneInputs>::At<'a>,
        client: &ComputeClient<R>,
        candidate: &mut Candidate,
    ) -> bool
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        if let Err(err) = warmup_once(operation, inputs.clone(), client) {
            candidate.fail(err);
            return false;
        }

        if !self.take_sample(operation, inputs, client, candidate).await {
            return false;
        }

        let Some(limit) = self.limit.filter(|_| self.short_circuit) else {
            return false;
        };

        if !candidate.samples.confirmed_under(limit, 1) {
            return false;
        }

        let required = self.config.short_circuit_samples.max(1);
        while candidate.samples.len() < required {
            if !self.take_sample(operation, inputs, client, candidate).await {
                return false;
            }
        }

        candidate.samples.confirmed_under(limit, required)
    }

    /// Queue one sample and resolve it immediately. Returns whether the candidate survived.
    async fn take_sample<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        operation: &TuneFn<F, Out>,
        inputs: &<F as TuneInputs>::At<'a>,
        client: &ComputeClient<R>,
        candidate: &mut Candidate,
    ) -> bool
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        match sample_once(operation, inputs.clone(), client) {
            Ok(profile) => {
                candidate.method.get_or_insert(profile.timing_method());
                candidate.samples.push(profile.resolve().await.duration());
                true
            }
            Err(err) => {
                candidate.fail(err);
                false
            }
        }
    }

    /// Stop everything if a candidate is provably close enough to peak throughput.
    ///
    /// The decision is persisted and reused on later runs, so it needs more than one sample:
    /// a single lucky measurement would pin a suboptimal kernel permanently.
    fn short_circuit_hit(&self, candidates: &[Candidate]) -> Option<String> {
        let limit = self.limit?;
        if !self.short_circuit {
            return None;
        }

        let required = self.config.short_circuit_samples.max(1);
        let hit = candidates
            .iter()
            .find(|c| c.live && c.samples.confirmed_under(limit, required))?;

        Some(hit.name.clone())
    }

    /// Drop candidates whose best time is past the speed factor.
    ///
    /// Comparing minima is already outlier resistant: a slow sample cannot lower a candidate's
    /// best, so nothing is eliminated on the strength of one bad measurement.
    fn eliminate(&self, candidates: &mut [Candidate], min_samples: usize) {
        let Some(best) = candidates
            .iter()
            .filter(|c| c.live)
            .filter_map(|c| c.samples.best())
            .min()
        else {
            return;
        };

        let threshold = best.mul_f64(self.config.speed_factor.max(1.0));
        let live = candidates.iter().filter(|c| c.live).count();
        let budget = live.saturating_sub(MIN_SURVIVORS);
        if budget == 0 {
            return;
        }

        let mut eliminable: Vec<(usize, Duration)> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.live && c.samples.len() >= min_samples)
            .filter_map(|(slot, c)| c.samples.best().map(|best| (slot, best)))
            .filter(|(_, best)| *best > threshold)
            .collect();

        eliminable.sort_by_key(|(_, best)| core::cmp::Reverse(*best));

        for (slot, _) in eliminable.into_iter().take(budget) {
            candidates[slot].live = false;
        }
    }

    /// Walk the plan batch by batch until one produces a usable measurement.
    pub(crate) fn run_plan<'a, K, R, F, Out>(
        &self,
        key: &K,
        plan: &mut TunePlan,
        autotunables: &[&TuneFn<F, Out>],
        inputs: &<F as TuneInputs>::At<'a>,
        client: &ComputeClient<R>,
        results: &mut [AutotuneResult],
    ) -> (Vec<(String, Duration)>, Option<String>)
    where
        K: core::fmt::Debug,
        R: Runtime,
        F: TuneInputs,
        Out: AutotuneOutput,
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        let mut steps = Vec::new();

        loop {
            let indices = plan.next();

            if indices.is_empty() {
                panic!(
                    "Can't execute the autotune plan for key: {key:?}\n - plan: {plan:?}\n - results: {results:?}"
                );
            }

            let outcome = self.run_batch(indices, autotunables, inputs.clone(), client);

            for (index, result) in outcome.results {
                results[index] = result;
            }
            steps.extend(outcome.steps);

            if outcome.any_success {
                return (steps, outcome.short_circuit);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn schedule(speed_factor: f64) -> Schedule {
        Schedule {
            config: BenchConfig {
                speed_factor,
                ..Default::default()
            },
            limit: None,
            short_circuit: false,
            track_steps: false,
        }
    }

    fn candidate(slot: usize, samples: impl IntoIterator<Item = u64>) -> Candidate {
        let mut candidate = Candidate::new(slot, alloc::format!("k{slot}"));
        for micros in samples {
            candidate.samples.push(Duration::from_micros(micros));
        }
        candidate
    }

    fn live(candidates: &[Candidate]) -> Vec<usize> {
        candidates
            .iter()
            .filter(|c| c.live)
            .map(|c| c.index)
            .collect()
    }

    #[test]
    fn eliminates_a_candidate_that_is_clearly_behind() {
        let mut candidates = vec![
            candidate(0, [100, 100, 100]),
            candidate(1, [105, 105, 105]),
            candidate(2, [900, 900, 900]),
        ];

        schedule(1.5).eliminate(&mut candidates, 3);

        assert_eq!(live(&candidates), vec![0, 1]);
    }

    #[test]
    fn keeps_a_candidate_within_the_speed_factor() {
        let mut candidates = vec![
            candidate(0, [100, 100, 100]),
            candidate(1, [105, 105, 105]),
            candidate(2, [140, 140, 140]),
        ];

        schedule(1.5).eliminate(&mut candidates, 3);

        assert_eq!(live(&candidates), vec![0, 1, 2]);
    }

    #[test]
    fn a_noisy_candidate_is_judged_on_its_best_run() {
        // One slow outlier must not disqualify a kernel: it cannot lower the minimum, which is
        // what the threshold is compared against.
        let mut candidates = vec![
            candidate(0, [100, 100, 100]),
            candidate(1, [100, 100, 100]),
            candidate(2, [110, 900, 120]),
        ];

        schedule(1.5).eliminate(&mut candidates, 3);

        assert_eq!(live(&candidates), vec![0, 1, 2]);
    }

    #[test]
    fn never_drops_below_the_survivor_floor() {
        // Every candidate but the leader deserves elimination on the numbers alone.
        let mut candidates = vec![
            candidate(0, [100, 100, 100]),
            candidate(1, [900, 900, 900]),
            candidate(2, [950, 950, 950]),
        ];

        schedule(1.5).eliminate(&mut candidates, 3);

        assert_eq!(live(&candidates).len(), MIN_SURVIVORS);
        assert!(candidates[0].live);
        // The slowest goes first when the floor forces a choice.
        assert!(candidates[1].live);
        assert!(!candidates[2].live);
    }

    #[test]
    fn spares_candidates_that_lack_the_minimum_samples() {
        let mut candidates = vec![
            candidate(0, [100, 100, 100]),
            candidate(1, [100, 100, 100]),
            candidate(2, [900]),
        ];

        schedule(1.5).eliminate(&mut candidates, 3);

        assert_eq!(live(&candidates), vec![0, 1, 2]);
    }

    #[test]
    fn short_circuit_requires_the_configured_confirmations() {
        let mut schedule = schedule(1.5);
        schedule.limit = Some(Duration::from_micros(200));
        schedule.short_circuit = true;

        let one = vec![candidate(0, [100, 900])];
        assert_eq!(schedule.short_circuit_hit(&one), None);

        let two = vec![candidate(0, [100, 900, 150])];
        assert_eq!(schedule.short_circuit_hit(&two), Some("k0".to_string()));
    }

    #[test]
    fn short_circuit_stays_off_without_a_limit() {
        let candidates = vec![candidate(0, [1, 1, 1])];
        assert_eq!(schedule(1.5).short_circuit_hit(&candidates), None);
    }

    #[test]
    fn unreached_candidates_are_reported_as_skipped() {
        let result = candidate(3, []).into_result();
        assert!(matches!(result.outcome, Err(AutotuneError::Skip { .. })));
    }

    #[test]
    fn a_candidate_that_failed_late_is_disqualified_despite_good_samples() {
        let mut candidate = candidate(1, [10, 10, 10]);
        candidate.fail(AutotuneError::InvalidSamples {
            name: "k1".to_string(),
        });

        let result = candidate.into_result();

        assert!(matches!(
            result.outcome,
            Err(AutotuneError::InvalidSamples { .. })
        ));
    }

    #[test]
    fn a_sampled_candidate_reports_its_measurements() {
        let result = candidate(1, [50, 60, 70]).into_result();
        let outcome = result.outcome.expect("sampled candidate succeeds");
        assert_eq!(outcome.index, 1);
        assert_eq!(outcome.computation.min, Duration::from_micros(60));
    }
}
