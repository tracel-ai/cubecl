use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::time::Duration;

use cubecl_common::profile::{Instant, ProfileDuration, ProfileTicks, TimingMethod};

use crate::config::autotune::BenchConfig;
use crate::server::ServerError;
use crate::tune::adaptive::first_pass::{CompileWindows, FirstPassPlanner};
use crate::tune::adaptive::sample_keeper::SampleKeeper;
use crate::tune::drive::candidate_sampler::{CandidateSampler, hold_failed};
use crate::tune::drive::sample_time::measured;
use crate::tune::hook::binding::{BoundEvictor, BoundStageEnterer, enter_stage};
use crate::tune::plan::planner::{Batch, TunePlanner};
use crate::tune::set::tune_fn::TuneFn;
use crate::tune::{
    AutotuneCheckable, AutotuneError, AutotuneOutcome, AutotuneResult, TuneInputs, TuneStage,
};

/// The outcome of benchmarking one batch of the plan.
#[derive(Debug)]
pub struct BatchOutcome {
    /// Each candidate's result, by tunable index (Arrival Usage).
    pub results: Vec<(usize, AutotuneResult)>,
    /// What each candidate cost, in batch order (Arrival Usage).
    pub steps: Vec<(String, Duration)>,
    pub short_circuit: Option<String>,
    pub any_success: bool,
    /// The index the round robin picked, when it produced one. See [`Scheduler::outcome`] for why
    /// the caller must not re-derive it by comparing the results.
    pub decided: Option<usize>,
    /// The candidates never measured, skipped by their groups' patience or not reached before a
    /// short circuit, in batch order (Arrival Usage).
    pub unmeasured: Vec<usize>,
}

impl BatchOutcome {
    /// The outcome of a batch of `indices` that could not be measured, the device not held.
    fn hold_failed<F: TuneInputs, Out: AutotuneCheckable>(
        indices: &[usize],
        autotunables: &[&TuneFn<F, Out>],
        reason: &ServerError,
    ) -> Self {
        Self {
            results: indices
                .iter()
                .map(|&index| {
                    let error = hold_failed(autotunables[index].name(), reason);
                    (index, AutotuneResult::error(error))
                })
                .collect(),
            steps: Vec::new(),
            short_circuit: None,
            any_success: false,
            decided: None,
            unmeasured: Vec::new(),
        }
    }
}

/// What walking the plan produced: everything the tuner has to report or commit.
#[derive(Debug)]
pub struct PlanOutcome {
    /// What each candidate measured cost, in the order they ran (Arrival Usage).
    pub steps: Vec<(String, Duration)>,
    pub short_circuit: Option<String>,
    /// The index the round robin picked, when it produced one.
    pub decided: Option<usize>,
    /// Whether every batch of the plan ran without a candidate measured.
    pub exhausted: bool,
}

/// How a [`Scheduler`] measures.
#[derive(Debug, Clone)]
pub struct ScheduleOptions {
    pub config: BenchConfig,
    /// The slowest median close enough to the peak, when the set has bounds and the short circuit
    /// is on.
    pub short_circuit_limit: Option<Duration>,
    pub track_steps: bool,
    /// How many kernels compile at once ([`CompilationConfig::parallelism`]): the window a round
    /// that may stop early compiles ahead of measuring.
    ///
    /// [`CompilationConfig::parallelism`]: crate::config::compilation::CompilationConfig::parallelism
    pub compile_parallelism: usize,
}

/// Schedules a tune's candidates: round robin benchmarking with early elimination.
///
/// A first pass measures the candidates one at a time in priority order, each with one warmup
/// and one sample resolved inline, compiling them ahead a window at a time. After each, a group
/// whose members stop improving on its leader skips the rest of them, and once the batch
/// measured its groups' minimum, a candidate confirmed under the time limit ends the batch.
/// After that, every live candidate gets one sample per round and the whole round is resolved
/// at once: resolving per sample would serialize a device round trip per measurement, which for
/// short kernels costs more than the samples it saves.
///
/// The sample cap bounds the work at `max_sample_count` per candidate, so the sampling itself can
/// never cost more than a fixed-count pass. The batch still can: the short circuit exits on the
/// first candidate confirmed under the limit, and eliminating cheap candidates early can push that
/// exit past kernels a fixed pass would have accepted and stopped at. Measured at +13.6% total
/// tuning cost on one card and −20% on another, both dominated by where the short circuit fires
/// rather than by the sample budget.
pub struct Scheduler<'i> {
    options: ScheduleOptions,
    /// What runs before every measured sample, when the set registered one.
    evictor: Option<Box<BoundEvictor<'i>>>,
    stage_enterer: Option<Box<BoundStageEnterer<'i>>>,
    /// The samples a round queued, by slot, resolved together (Scratch Usage).
    pending: Vec<(usize, ProfileDuration)>,
    /// The candidates a round may eliminate, by slot and best time (Scratch Usage).
    eliminable: Vec<(usize, Duration)>,
}

impl<'i> Scheduler<'i> {
    pub fn new(
        options: ScheduleOptions,
        evictor: Option<Box<BoundEvictor<'i>>>,
        stage_enterer: Option<Box<BoundStageEnterer<'i>>>,
    ) -> Self {
        Self {
            options,
            evictor,
            stage_enterer,
            pending: Vec::new(),
            eliminable: Vec::new(),
        }
    }

    /// Walk the plan batch by batch, on `sampler`'s device, until one produces a usable
    /// measurement, each candidate's result written in `results` by tunable index.
    ///
    /// # Assumptions
    ///
    /// - `results` holds a result per candidate of `autotunables`.
    pub fn schedule_plan<'a, F, Out>(
        &mut self,
        plan: &mut TunePlanner,
        autotunables: &[&TuneFn<F, Out>],
        inputs: &<F as TuneInputs>::At<'a>,
        sampler: CandidateSampler<'_>,
        results: &mut [AutotuneResult],
    ) -> PlanOutcome
    where
        F: TuneInputs,
        Out: AutotuneCheckable,
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        let mut steps = Vec::new();
        let mut retry = None;

        loop {
            let batch = retry.take().unwrap_or_else(|| plan.plan_next());
            if batch.is_empty() {
                return PlanOutcome {
                    steps,
                    short_circuit: None,
                    decided: None,
                    exhausted: true,
                };
            }

            let outcome = self.schedule_batch(batch, autotunables, inputs.clone(), sampler);
            for (index, result) in outcome.results {
                results[index] = result;
            }
            steps.extend(outcome.steps);

            if outcome.any_success {
                return PlanOutcome {
                    steps,
                    short_circuit: outcome.short_circuit,
                    decided: outcome.decided,
                    exhausted: false,
                };
            }

            // Every candidate measured failed, so the ones their groups' patience skipped never
            // had their chance: measure them before the plan moves on.
            retry =
                (!outcome.unmeasured.is_empty()).then(|| Batch::unconditional(outcome.unmeasured));
        }
    }
}

impl<'i> Scheduler<'i> {
    /// Benchmark one batch of candidates, blocking until the batch is decided.
    ///
    /// Under [`TuneDeviceAccess::Exclusive`](crate::tune::TuneDeviceAccess::Exclusive), takes
    /// exclusive device access for the entire round robin: candidates are interleaved, so
    /// releasing the device between them would let unrelated work land in the middle of a
    /// measurement.
    fn schedule_batch<'a, F: TuneInputs, Out: AutotuneCheckable>(
        &mut self,
        batch: Batch,
        autotunables: &[&TuneFn<F, Out>],
        inputs: <F as TuneInputs>::At<'a>,
        sampler: CandidateSampler<'_>,
    ) -> BatchOutcome
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        let indices: Vec<usize> = batch.indices().collect();
        sampler
            .hold(|| self.measure(batch, autotunables, inputs, sampler))
            .unwrap_or_else(|err| BatchOutcome::hold_failed(&indices, autotunables, &err))
    }

    /// Measure `batch` in a measuring stage, its stream executing what it launches.
    fn measure<'a, F: TuneInputs, Out: AutotuneCheckable>(
        &mut self,
        batch: Batch,
        autotunables: &[&TuneFn<F, Out>],
        inputs: <F as TuneInputs>::At<'a>,
        sampler: CandidateSampler<'_>,
    ) -> BatchOutcome
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        let measuring = crate::execution::StreamModeOverride::new(
            crate::execution::StreamMode::Execute,
            sampler.client(),
        );
        let stage = enter_stage(self.stage_enterer.as_deref_mut(), TuneStage::Measuring);
        let outcome =
            cubecl_environment::future::block_on(self.drive(batch, autotunables, inputs, sampler));
        core::mem::drop(stage);
        core::mem::drop(measuring);
        outcome
    }

    async fn drive<'a, F: TuneInputs, Out: AutotuneCheckable>(
        &mut self,
        batch: Batch,
        autotunables: &[&TuneFn<F, Out>],
        inputs: <F as TuneInputs>::At<'a>,
        sampler: CandidateSampler<'_>,
    ) -> BatchOutcome
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        let windows = self.compile_windows(batch.entries.len(), !batch.patience.is_empty());
        let mut planner = FirstPassPlanner::new(&batch, windows);
        let mut candidates: Vec<CandidateTracker> = batch
            .entries
            .into_iter()
            .map(|entry| {
                CandidateTracker::new(entry.index, autotunables[entry.index].name().to_string())
            })
            .collect();

        let first_pass = self
            .first_pass_all(
                &mut planner,
                &mut candidates,
                autotunables,
                &inputs,
                sampler,
            )
            .await;

        // A candidate the first pass never measured was skipped by its groups' patience.
        candidates
            .iter_mut()
            .filter(|candidate| candidate.samples.summary().count == 0)
            .for_each(CandidateTracker::skip);

        let confirmed = match first_pass {
            Some(slot) => Some(slot),
            None => {
                self.sample_rounds(&mut candidates, autotunables, &inputs, sampler)
                    .await
            }
        };

        self.outcome(candidates, confirmed)
    }

    /// The first pass: one warmup and one sample each, resolved inline so the batch can stop
    /// before the remaining kernels are ever compiled. This is the one place where paying a
    /// device round trip per sample is worth it. Answers the slot of the candidate a short circuit
    /// confirmed, if one did.
    async fn first_pass_all<'a, F: TuneInputs, Out: AutotuneCheckable>(
        &mut self,
        planner: &mut FirstPassPlanner,
        candidates: &mut [CandidateTracker],
        autotunables: &[&TuneFn<F, Out>],
        inputs: &<F as TuneInputs>::At<'a>,
        sampler: CandidateSampler<'_>,
    ) -> Option<usize>
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        while let Some(step) = planner.plan_next() {
            let slot = step.slot;
            let launched = self.options.track_steps.then(Instant::now);
            self.compile_ahead(&step.compile, candidates, autotunables, inputs, sampler);
            self.first_pass(autotunables, inputs, &mut candidates[slot], sampler)
                .await;
            candidates[slot].charge(launched.map(|launched| launched.elapsed()));

            if let Some(best) = candidates[slot].live_best() {
                planner.record(slot, best);
            }

            if planner.may_short_circuit()
                && let Some(confirmed) = self
                    .confirm_under_limit(autotunables, inputs, &mut candidates[..=slot], sampler)
                    .await
            {
                return Some(confirmed);
            }
        }
        None
    }

    /// Compile the candidates at `slots` together, in a gathering stage.
    fn compile_ahead<'a, F: TuneInputs, Out: AutotuneCheckable>(
        &mut self,
        slots: &[usize],
        candidates: &mut [CandidateTracker],
        autotunables: &[&TuneFn<F, Out>],
        inputs: &<F as TuneInputs>::At<'a>,
        sampler: CandidateSampler<'_>,
    ) where
        <F as TuneInputs>::At<'a>: Clone,
    {
        if slots.is_empty() {
            return;
        }
        let ahead = candidates
            .iter_mut()
            .enumerate()
            .filter(|(slot, _)| slots.binary_search(slot).is_ok())
            .map(|(_, candidate)| candidate);
        let gathering = enter_stage(self.stage_enterer.as_deref_mut(), TuneStage::Gathering);
        compile_together(ahead, autotunables, inputs, sampler);
        core::mem::drop(gathering);
    }

    /// Fold the finished candidates into the batch outcome, naming the winner.
    ///
    /// The winner is picked here rather than left to the caller's comparison of the results,
    /// because by this point the candidates no longer hold comparable evidence: an eliminated one
    /// stopped at `min_sample_count` while a survivor kept sampling, and
    /// `BenchmarkComputations::score` inflates with observed spread, which grows with the sample
    /// count. Scoring them against each other would hand a candidate an advantage for having been
    /// dropped early. Elimination decides who is eligible; the score only ranks the survivors. A
    /// batch that a short circuit ended is decided by the candidate at slot `confirmed`, the one
    /// confirmed under the limit: the others may hold fewer samples than its confirmations.
    fn outcome(&self, candidates: Vec<CandidateTracker>, confirmed: Option<usize>) -> BatchOutcome {
        let short_circuit = confirmed.map(|slot| candidates[slot].name.clone());
        let steps = candidates
            .iter()
            .filter(|_| self.options.track_steps)
            .map(|candidate| (candidate.name.clone(), candidate.elapsed))
            .collect();
        let unmeasured = candidates
            .iter()
            .filter(|candidate| candidate.is_unmeasured())
            .map(|candidate| candidate.index)
            .collect();
        let ranked = |slot: usize, candidate: &CandidateTracker| {
            candidate.live && confirmed.is_none_or(|hit| hit == slot)
        };
        let ranked_results: Vec<(bool, (usize, AutotuneResult))> = candidates
            .into_iter()
            .enumerate()
            .map(|(slot, candidate)| {
                (
                    ranked(slot, &candidate),
                    (candidate.index, candidate.into_result()),
                )
            })
            .collect();
        let decided = ranked_results
            .iter()
            .filter(|(ranked, _)| *ranked)
            .filter_map(|(_, (_, result))| result.outcome.as_ref().ok())
            .min_by_key(|outcome| outcome.computation.score())
            .map(|outcome| outcome.index);
        let results: Vec<(usize, AutotuneResult)> = ranked_results
            .into_iter()
            .map(|(_, result)| result)
            .collect();

        BatchOutcome {
            any_success: results.iter().any(|(_, result)| result.outcome.is_ok()),
            results,
            steps,
            short_circuit,
            decided,
            unmeasured,
        }
    }

    /// How many of a batch of `len` candidates the first pass compiles together when the next
    /// one to measure is not compiled yet. Where the round may stop early, `patient` or bounded by
    /// a time limit, its first window holds as many as compile at once, so a round that stops
    /// there compiles at most one window it never measures. A `patient` batch, whose groups may
    /// skip the rest of their members, keeps to that window, so a pass its patience ends compiles
    /// at most one window it never measures too. Elsewhere the rest of the batch compiles at once:
    /// a bounded round its first window did not end rarely ends on the next, and window after
    /// window would wait on each window's slowest kernel, then on its measurements, with the
    /// compiler idle. Either way the windows decide when kernels compile, never which are measured.
    fn compile_windows(&self, len: usize, patient: bool) -> CompileWindows {
        let bounded = self.options.short_circuit_limit.is_some();
        CompileWindows {
            first_count: self.window_count(len, patient || bounded),
            later_count: self.window_count(len, patient),
        }
    }

    /// The candidates of a batch of `len` one window compiles: as many as compile at once where
    /// the round may stop before measuring the rest, `stoppable`, and the whole batch elsewhere.
    fn window_count(&self, len: usize, stoppable: bool) -> usize {
        if stoppable {
            return self.options.compile_parallelism;
        }
        len
    }

    /// Warm up a candidate and take its first sample. A candidate that rejected the problem while
    /// compiling is out of the round already.
    async fn first_pass<'a, F: TuneInputs, Out: AutotuneCheckable>(
        &mut self,
        autotunables: &[&TuneFn<F, Out>],
        inputs: &<F as TuneInputs>::At<'a>,
        candidate: &mut CandidateTracker,
        sampler: CandidateSampler<'_>,
    ) where
        <F as TuneInputs>::At<'a>: Clone,
    {
        if !candidate.live {
            return;
        }
        let operation = autotunables[candidate.index];
        if let Err(err) = sampler.warmup_once(operation, inputs.clone()) {
            return candidate.fail(err);
        }
        self.take_sample(operation, inputs, candidate, sampler)
            .await;
    }

    /// The slot of the first of `candidates`, in priority order, confirmed under the time limit,
    /// each one that has a sample under it first taking the confirmation samples it lacks. `None`
    /// without a limit or with the short circuit off.
    async fn confirm_under_limit<'a, F: TuneInputs, Out: AutotuneCheckable>(
        &mut self,
        autotunables: &[&TuneFn<F, Out>],
        inputs: &<F as TuneInputs>::At<'a>,
        candidates: &mut [CandidateTracker],
        sampler: CandidateSampler<'_>,
    ) -> Option<usize>
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        let limit = self.options.short_circuit_limit?;
        let required = self.options.config.clamped_short_circuit_sample_count();

        for (slot, candidate) in candidates
            .iter_mut()
            .enumerate()
            .filter(|(_, candidate)| candidate.live && candidate.samples.any_under(limit))
        {
            let operation = autotunables[candidate.index];
            let launched = self.options.track_steps.then(Instant::now);
            while candidate.samples.summary().count < required
                && self
                    .take_sample(operation, inputs, candidate, sampler)
                    .await
            {}
            candidate.charge(launched.map(|launched| launched.elapsed()));
            if candidate.confirmed_under(limit, required) {
                return Some(slot);
            }
        }

        None
    }

    /// Queue one sample and resolve it immediately. Returns whether the candidate survived.
    async fn take_sample<'a, F: TuneInputs, Out: AutotuneCheckable>(
        &mut self,
        operation: &TuneFn<F, Out>,
        inputs: &<F as TuneInputs>::At<'a>,
        candidate: &mut CandidateTracker,
        sampler: CandidateSampler<'_>,
    ) -> bool
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        let sampled = sampler.sample_once(operation, inputs.clone(), self.evictor.as_deref_mut());
        let profile = match sampled.map(|profile| candidate.queued(profile)) {
            Ok(profile) => profile,
            Err(err) => return candidate.disqualify(err),
        };
        candidate.track_ticks(profile.resolve().await);
        candidate.live
    }

    /// Later rounds, resolved a whole round at once: a profile only submits its readback when
    /// first polled, so resolving one at a time would serialize a device round trip per
    /// measurement, which for short kernels costs more than the samples it saves. Answers the slot
    /// of the candidate a short circuit confirmed, if one did.
    async fn sample_rounds<'a, F: TuneInputs, Out: AutotuneCheckable>(
        &mut self,
        candidates: &mut [CandidateTracker],
        autotunables: &[&TuneFn<F, Out>],
        inputs: &<F as TuneInputs>::At<'a>,
        sampler: CandidateSampler<'_>,
    ) -> Option<usize>
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        let (min_sample_count, max_sample_count) = self.options.config.sample_count_range();

        for round in 1..max_sample_count {
            let live = live_count(candidates);
            if live == 0 || (live == 1 && round >= min_sample_count) {
                return None;
            }

            self.queue_round(candidates, autotunables, inputs, sampler, max_sample_count);
            match self.end_round(candidates, round, min_sample_count).await {
                RoundEnd::Continue => {}
                RoundEnd::ShortCircuit(slot) => return Some(slot),
                RoundEnd::Done => return None,
            }
        }

        None
    }

    /// Queue one sample of every live candidate short of `max_sample_count` samples. A candidate
    /// that took confirmation samples in the first pass is ahead of the others, so the ceiling is
    /// enforced per candidate rather than by round count.
    fn queue_round<'a, F: TuneInputs, Out: AutotuneCheckable>(
        &mut self,
        candidates: &mut [CandidateTracker],
        autotunables: &[&TuneFn<F, Out>],
        inputs: &<F as TuneInputs>::At<'a>,
        sampler: CandidateSampler<'_>,
        max_sample_count: usize,
    ) where
        <F as TuneInputs>::At<'a>: Clone,
    {
        for (slot, candidate) in candidates.iter_mut().enumerate().filter(|(_, candidate)| {
            candidate.live && candidate.samples.summary().count < max_sample_count
        }) {
            let launched = self.options.track_steps.then(Instant::now);
            let operation = autotunables[candidate.index];
            match sampler.sample_once(operation, inputs.clone(), self.evictor.as_deref_mut()) {
                Ok(profile) => self.pending.push((slot, candidate.queued(profile))),
                Err(err) => candidate.fail(err),
            }
            candidate.charge(launched.map(|launched| launched.elapsed()));
        }
    }

    /// Resolve the samples the round queued, and say how the round ends.
    async fn end_round(
        &mut self,
        candidates: &mut [CandidateTracker],
        round: usize,
        min_sample_count: usize,
    ) -> RoundEnd {
        if self.pending.is_empty() {
            return RoundEnd::Done;
        }
        self.resolve_round(candidates).await;
        if let Some(confirmed) = self.short_circuit_hit(candidates) {
            return RoundEnd::ShortCircuit(confirmed);
        }
        if round + 1 < min_sample_count {
            return RoundEnd::Continue;
        }
        self.eliminate(candidates, min_sample_count);
        RoundEnd::after_elimination(candidates)
    }

    /// Resolve every sample the round queued, together.
    ///
    /// Each profile is timed individually rather than timing the join: they resolve concurrently,
    /// so the round's wall clock cannot be split across candidates, but the wait for one
    /// candidate's own readback can.
    async fn resolve_round(&mut self, candidates: &mut [CandidateTracker]) {
        let track_steps = self.options.track_steps;
        let resolved = futures_util::future::join_all(self.pending.drain(..).map(
            |(slot, profile)| async move {
                let started = Instant::now();
                let ticks = profile.resolve().await;
                (slot, ticks, started.elapsed())
            },
        ))
        .await;

        for (slot, ticks, waited) in resolved {
            candidates[slot].track_ticks(ticks);
            candidates[slot].charge(track_steps.then_some(waited));
        }
    }

    /// Stop everything if a candidate is provably close enough to peak throughput.
    ///
    /// The decision is persisted and reused on later runs, so it needs more than one sample:
    /// a single lucky measurement would pin a suboptimal kernel permanently.
    /// Returns the slot of the candidate confirmed under the limit.
    fn short_circuit_hit(&self, candidates: &[CandidateTracker]) -> Option<usize> {
        let limit = self.options.short_circuit_limit?;
        let required = self.options.config.clamped_short_circuit_sample_count();
        candidates
            .iter()
            .position(|candidate| candidate.confirmed_under(limit, required))
    }

    /// Drop candidates whose best time is past the speed factor.
    ///
    /// Comparing minima is already outlier resistant: a slow sample cannot lower a candidate's
    /// best, so nothing is eliminated on the strength of one bad measurement.
    fn eliminate(&mut self, candidates: &mut [CandidateTracker], min_sample_count: usize) {
        let Some(best) = candidates
            .iter()
            .filter_map(CandidateTracker::live_best)
            .min()
        else {
            return;
        };

        let threshold = best.mul_f64(self.options.config.clamped_speed_factor());
        let budget = live_count(candidates).saturating_sub(MIN_SURVIVOR_COUNT);
        if budget == 0 {
            return;
        }

        self.eliminable.clear();
        self.eliminable.extend(
            candidates
                .iter()
                .enumerate()
                .filter(|(_, candidate)| candidate.samples.summary().count >= min_sample_count)
                .filter_map(|(slot, candidate)| candidate.live_best().map(|best| (slot, best)))
                .filter(|(_, best)| *best > threshold),
        );
        self.eliminable
            .sort_by_key(|(_, best)| core::cmp::Reverse(*best));

        for (slot, _) in self.eliminable.drain(..).take(budget) {
            candidates[slot].live = false;
        }
    }
}

/// How a later round ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundEnd {
    /// Another round follows.
    Continue,
    /// The candidate at the slot was confirmed under the limit.
    ShortCircuit(usize),
    /// Every live candidate converged, or none queued a sample.
    Done,
}

impl RoundEnd {
    /// How a round that eliminated ends: done once every live candidate, and there is one, has
    /// converged.
    fn after_elimination(candidates: &[CandidateTracker]) -> Self {
        let mut live = candidates
            .iter()
            .filter(|candidate| candidate.live)
            .peekable();
        match live.peek().is_some() && live.all(|candidate| candidate.samples.summary().converged) {
            true => Self::Done,
            false => Self::Continue,
        }
    }
}

/// How many of `candidates` are still in the round.
fn live_count(candidates: &[CandidateTracker]) -> usize {
    candidates.iter().filter(|candidate| candidate.live).count()
}

/// Candidates kept alive no matter how far behind they are, so a batch never narrows to a
/// single kernel that was never compared against anything.
const MIN_SURVIVOR_COUNT: usize = 2;

/// Tracks one candidate kernel and the evidence gathered about it so far.
#[derive(Debug)]
struct CandidateTracker {
    index: usize,
    name: String,
    samples: SampleKeeper,
    method: Option<TimingMethod>,
    error: Option<AutotuneError>,
    elapsed: Duration,
    live: bool,
}

impl CandidateTracker {
    fn new(index: usize, name: String) -> Self {
        Self {
            index,
            name,
            samples: SampleKeeper::default(),
            method: None,
            error: None,
            elapsed: Duration::ZERO,
            live: true,
        }
    }

    /// Keep what a resolved sample measured, or disqualify the candidate where it measured no
    /// time: recorded as a zero, it would be the fastest sample in the round and would short
    /// circuit on it.
    fn track_ticks(&mut self, ticks: Option<ProfileTicks>) {
        match measured(ticks) {
            Some(duration) => self.samples.keep(duration),
            None => self.fail(AutotuneError::NotMeasured {
                name: self.name.clone(),
            }),
        }
    }

    /// `profile`, a sample of the candidate just queued, its timing method kept.
    fn queued(&mut self, profile: ProfileDuration) -> ProfileDuration {
        self.method.get_or_insert(profile.timing_method());
        profile
    }

    /// Add `spent` to what the candidate cost, when it was timed.
    fn charge(&mut self, spent: Option<Duration>) {
        self.elapsed += spent.unwrap_or_default();
    }

    fn fail(&mut self, error: AutotuneError) {
        self.error = Some(error);
        self.live = false;
    }

    /// [Fail](Self::fail) the candidate, answering that it did not survive.
    fn disqualify(&mut self, error: AutotuneError) -> bool {
        self.fail(error);
        false
    }

    /// Leave it out of the batch before its first sample. With no samples and no error, it
    /// comes out as [skipped](AutotuneError::Skip).
    fn skip(&mut self) {
        self.live = false;
    }

    /// Its best time while it is still in the round.
    fn live_best(&self) -> Option<Duration> {
        self.samples.summary().best.filter(|_| self.live)
    }

    /// Whether it is still in the round with `required` samples under `limit`.
    fn confirmed_under(&self, limit: Duration, required: usize) -> bool {
        self.live && self.samples.confirmed_under(limit, required)
    }

    /// Whether it was never measured and never failed: skipped, or not reached.
    fn is_unmeasured(&self) -> bool {
        self.samples.summary().count == 0 && self.error.is_none()
    }

    fn into_result(self) -> AutotuneResult {
        // A candidate that failed at any point is disqualified even if earlier samples
        // succeeded, so a kernel that crashes sporadically can never win on its good runs.
        if let Some(error) = self.error {
            return AutotuneResult::error(error);
        }

        if self.samples.summary().count == 0 {
            return AutotuneResult::error(AutotuneError::Skip { name: self.name });
        }

        let computation = self
            .samples
            .computation(self.method.unwrap_or(TimingMethod::System));

        AutotuneResult::success(AutotuneOutcome::new(self.name, self.index, computation))
    }
}

/// Compile the kernels of `candidates` in one batch, on as many threads as compile at once,
/// rather than each inside its own warmup, one after another.
///
/// Each runs once on this stream with its launches discarded, which queues the kernels they reach
/// ([`LaunchMode::Queue`](crate::execution::LaunchMode::Queue)); the queue then compiles together.
/// Only this stream discards, so another thread's launches still run. A candidate that rejects
/// the problem or panics keeps its error, as its warmup would have reported it: it is never a
/// kernel to fall back on. One whose kernel fails to compile reports it at its warmup.
fn compile_together<'a, 'c, F: TuneInputs, Out: AutotuneCheckable>(
    candidates: impl Iterator<Item = &'c mut CandidateTracker>,
    autotunables: &[&TuneFn<F, Out>],
    inputs: &<F as TuneInputs>::At<'a>,
    sampler: CandidateSampler<'_>,
) where
    <F as TuneInputs>::At<'a>: Clone,
{
    let queuing = crate::execution::StreamModeOverride::new(
        crate::execution::StreamMode::Discard,
        sampler.client(),
    );
    let gathering = crate::tune::drive::gathering::CandidateGatherer::gather();
    for candidate in candidates {
        if let Err(unusable @ (AutotuneError::Rejected { .. } | AutotuneError::Panicked { .. })) =
            sampler.contain_race(autotunables[candidate.index], inputs.clone())
        {
            candidate.fail(unusable);
        }
    }
    core::mem::drop(gathering);
    core::mem::drop(queuing);
    sampler.client().compile_queued();
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn schedule(speed_factor: f64) -> Scheduler<'static> {
        Scheduler::new(options(speed_factor), None, None)
    }

    fn options(speed_factor: f64) -> ScheduleOptions {
        ScheduleOptions {
            config: BenchConfig {
                speed_factor,
                ..Default::default()
            },
            short_circuit_limit: None,
            track_steps: false,
            compile_parallelism: 1,
        }
    }

    fn candidate(slot: usize, samples: impl IntoIterator<Item = u64>) -> CandidateTracker {
        let mut candidate = CandidateTracker::new(slot, alloc::format!("k{slot}"));
        for micros in samples {
            candidate.samples.keep(Duration::from_micros(micros));
        }
        candidate
    }

    fn live(candidates: &[CandidateTracker]) -> Vec<usize> {
        candidates
            .iter()
            .filter(|c| c.live)
            .map(|c| c.index)
            .collect()
    }

    #[test]
    fn a_round_that_may_stop_compiles_a_window_and_one_that_cannot_the_whole_batch() {
        let bounded = Scheduler::new(
            ScheduleOptions {
                short_circuit_limit: Some(Duration::from_micros(10)),
                compile_parallelism: 8,
                ..options(1.5)
            },
            None,
            None,
        );
        let patient = Scheduler::new(
            ScheduleOptions {
                compile_parallelism: 8,
                ..options(1.5)
            },
            None,
            None,
        );

        let windows = |first_count, later_count| CompileWindows {
            first_count,
            later_count,
        };

        assert_eq!(bounded.compile_windows(100, false), windows(8, 100));
        assert_eq!(bounded.compile_windows(100, true), windows(8, 8));
        assert_eq!(patient.compile_windows(100, true), windows(8, 8));
        assert_eq!(patient.compile_windows(100, false), windows(100, 100));
    }

    #[test]
    fn a_short_circuit_decides_the_confirmed_candidate() {
        let candidates = vec![candidate(0, [40]), candidate(1, [50, 50, 50])];

        let outcome = schedule(1.5).outcome(candidates, Some(1));

        assert_eq!(outcome.decided, Some(1));
        assert_eq!(outcome.short_circuit.as_deref(), Some("k1"));
    }

    #[test]
    fn a_candidate_compiled_but_never_measured_is_reported_unmeasured() {
        let mut prefetched = candidate(1, []);
        prefetched.skip();

        let outcome = schedule(1.5).outcome(vec![candidate(0, [10, 10, 10]), prefetched], None);

        assert_eq!(outcome.unmeasured, vec![1]);
        assert_eq!(outcome.decided, Some(0));
    }

    #[test]
    fn a_tracked_batch_reports_every_candidate_s_cost_in_order() {
        let tracked = Scheduler::new(
            ScheduleOptions {
                track_steps: true,
                ..options(1.5)
            },
            None,
            None,
        );
        let mut first = candidate(0, [10]);
        first.charge(Some(Duration::from_micros(3)));

        let outcome = tracked.outcome(vec![first, candidate(1, [20])], None);

        assert_eq!(
            outcome.steps,
            [
                ("k0".to_string(), Duration::from_micros(3)),
                ("k1".to_string(), Duration::ZERO)
            ]
        );
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

        assert_eq!(live(&candidates).len(), MIN_SURVIVOR_COUNT);
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
        let schedule = Scheduler::new(
            ScheduleOptions {
                short_circuit_limit: Some(Duration::from_micros(200)),
                ..options(1.5)
            },
            None,
            None,
        );

        let one = vec![candidate(0, [100, 900])];
        assert_eq!(schedule.short_circuit_hit(&one), None);

        let two = vec![candidate(0, [100, 900, 150])];
        assert_eq!(schedule.short_circuit_hit(&two), Some(0));
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
        candidate.fail(AutotuneError::NotMeasured {
            name: "k1".to_string(),
        });

        let result = candidate.into_result();

        assert!(matches!(
            result.outcome,
            Err(AutotuneError::NotMeasured { .. })
        ));
    }

    #[test]
    fn a_sample_that_measured_no_time_disqualifies() {
        let mut candidate = candidate(1, [10]);

        candidate.track_ticks(None);

        assert!(!candidate.live);
        assert!(matches!(
            candidate.into_result().outcome,
            Err(AutotuneError::NotMeasured { .. })
        ));
    }

    #[test]
    fn an_eliminated_candidate_never_wins_on_its_shorter_sample_set() {
        // Elimination froze candidate 1 at three tight samples while candidate 0 kept sampling
        // and picked up spread, so candidate 1 scores better on the arithmetic alone. It was
        // dropped on the evidence, and the evidence is what decides.
        let mut eliminated = candidate(1, [10, 10, 10]);
        eliminated.live = false;

        let survivor = candidate(0, [50, 900, 60, 800, 55]);
        assert!(
            eliminated.samples.computation(TimingMethod::System).score()
                < survivor.samples.computation(TimingMethod::System).score(),
            "the test is pointless unless the eliminated candidate scores better"
        );

        let outcome = schedule(1.5).outcome(vec![survivor, eliminated], None);

        assert_eq!(outcome.decided, Some(0));
        assert!(outcome.any_success);
    }

    #[test]
    fn the_fastest_survivor_is_the_one_declared() {
        let candidates = vec![
            candidate(0, [100, 100, 100]),
            candidate(1, [40, 40, 40]),
            candidate(2, [70, 70, 70]),
        ];

        assert_eq!(schedule(1.5).outcome(candidates, None).decided, Some(1));
    }

    #[test]
    fn no_survivor_leaves_the_decision_open() {
        // Every candidate failed, so there is nothing to declare and the caller falls back to
        // its own comparison rather than being handed an index that measured nothing.
        let mut failed = candidate(0, [10, 10, 10]);
        failed.fail(AutotuneError::NotMeasured {
            name: "k0".to_string(),
        });

        let outcome = schedule(1.5).outcome(vec![failed, candidate(1, [])], None);

        assert_eq!(outcome.decided, None);
        assert!(!outcome.any_success);
    }

    #[test]
    fn a_sampled_candidate_reports_its_measurements() {
        let result = candidate(1, [50, 60, 70]).into_result();
        let outcome = result.outcome.expect("a sampled candidate should succeed");
        assert_eq!(outcome.index, 1);
        assert_eq!(outcome.computation.min, Duration::from_micros(60));
    }

    #[test]
    fn a_round_ends_once_every_live_candidate_converged() {
        let converged = candidate(0, [20, 20, 20, 20, 20]);
        let mut out = candidate(1, [10]);
        out.live = false;

        assert_eq!(
            RoundEnd::after_elimination(&[converged, out]),
            RoundEnd::Done
        );
        assert_eq!(
            RoundEnd::after_elimination(&[candidate(0, [20])]),
            RoundEnd::Continue
        );
    }

    #[test]
    fn a_failed_hold_fails_every_candidate_of_the_batch() {
        let first = TuneFn::<(), ()>::new("first".to_string(), Box::new(|_| Ok(())));
        let reason = ServerError::Generic {
            reason: "the device thread is gone".into(),
            backtrace: Default::default(),
        };

        let outcome = BatchOutcome::hold_failed(&[0], &[&first], &reason);

        assert!(!outcome.any_success);
        assert!(matches!(
            outcome.results[..],
            [(
                0,
                AutotuneResult {
                    outcome: Err(AutotuneError::HoldFailed { .. })
                }
            )]
        ));
    }
}
