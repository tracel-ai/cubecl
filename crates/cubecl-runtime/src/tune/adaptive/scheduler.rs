use alloc::boxed::Box;
use alloc::collections::BinaryHeap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::time::Duration;

use cubecl_common::profile::{Instant, ProfileDuration};

use crate::config::autotune::BenchConfig;
use crate::server::ServerError;
use crate::tune::adaptive::candidate_tracker::{CandidateState, CandidateTracker};
use crate::tune::adaptive::first_pass_planner::{CompileWindows, FirstPassPlanner};
use crate::tune::hook::hook_binder::{BoundEvictor, BoundStageEnterer, enter_stage};
use crate::tune::measure::candidate_sampler::CandidateSampler;
use crate::tune::measure::outcome::best_scored;
use crate::tune::measure::tune_fn::TuneFn;
use crate::tune::plan::batch::Batch;
use crate::tune::plan::planner::TunePlanner;
use crate::tune::{AutotuneError, AutotuneOutput, AutotuneResult, TuneInputs, TuneStage};

/// The outcome of benchmarking one batch of the plan.
#[derive(Debug)]
pub struct BatchOutcome {
    /// Each candidate's result with its tunable index, in batch order (Arrival Usage).
    pub results: Vec<(usize, AutotuneResult)>,
    /// What each candidate of the batch cost, in batch order (Arrival Usage), unreached ones at
    /// zero.
    pub steps: Vec<(String, Duration)>,
    pub short_circuit: Option<String>,
    pub any_success: bool,
    /// The index the round robin picked, when it produced one. See [`Scheduler::conclude_batch`] for why
    /// the caller must not re-derive it by comparing the results. `None` where every survivor
    /// failed after elimination: the commit then falls back on the best-scored result, an
    /// eliminated candidate, as the base commit did.
    pub winner: Option<usize>,
    /// The candidates skipped by their groups' patience or not reached before a short circuit, in
    /// batch order (Arrival Usage).
    pub unmeasured: Vec<usize>,
}

impl BatchOutcome {
    /// The outcome of a batch of `indices` that could not be measured, the device not held.
    fn hold_failed<F: TuneInputs, Out: AutotuneOutput>(
        indices: &[usize],
        autotunables: &[&TuneFn<F, Out>],
        reason: &ServerError,
    ) -> Self {
        Self {
            results: indices
                .iter()
                .map(|&index| {
                    let error = AutotuneError::hold_failed(
                        operation_at(autotunables, index).name(),
                        reason,
                    );
                    (index, AutotuneResult::from_error(error))
                })
                .collect(),
            steps: Vec::new(),
            short_circuit: None,
            any_success: false,
            winner: None,
            unmeasured: Vec::new(),
        }
    }
}

/// What walking the plan produced: everything the tuner has to report or commit.
#[derive(Debug)]
pub struct PlanOutcome {
    /// What each candidate of every batch that ran cost, batch after batch (Arrival Usage).
    pub steps: Vec<(String, Duration)>,
    pub short_circuit: Option<String>,
    pub winner: Option<usize>,
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
    /// How many kernels compile at once ([`CompilationConfig::parallelism`]), at least one: the
    /// window a round that may stop early compiles ahead of measuring.
    ///
    /// [`CompilationConfig::parallelism`]: crate::config::compilation::CompilationConfig::parallelism
    pub compile_parallelism_count: usize,
}

/// Schedules a tune's candidates: round robin benchmarking with early elimination.
///
/// A [first pass](FirstPassPlanner) measures each candidate once, each with one warmup. After
/// that, every live candidate gets one sample per round and the whole round is resolved
/// at once: a profile only submits its readback when first polled, so resolving per sample would
/// serialize a device round trip per measurement, which for short kernels costs more than the
/// samples it saves.
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
    /// The candidates a round may eliminate, slowest first and the earliest slot among equals
    /// (Ordered and Scratch Usages).
    eliminable: BinaryHeap<(Duration, Reverse<usize>)>,
    /// Whether each candidate of a finished batch may win, by slot (Position and Scratch Usages).
    ranked: Vec<bool>,
    /// The indices of the batch being held, for its outcome should the hold fail (Scratch Usage).
    held: Vec<usize>,
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
            eliminable: BinaryHeap::new(),
            ranked: Vec::new(),
            held: Vec::new(),
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
        Out: AutotuneOutput,
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
                    winner: None,
                    exhausted: true,
                };
            }

            let outcome = self.schedule_batch(batch, autotunables, inputs.clone(), sampler);
            for (index, result) in outcome.results {
                *results
                    .get_mut(index)
                    .expect("an index should be one of the set's candidates") = result;
            }
            steps.extend(outcome.steps);

            if outcome.any_success {
                return PlanOutcome {
                    steps,
                    short_circuit: outcome.short_circuit,
                    winner: outcome.winner,
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
    fn schedule_batch<'a, F: TuneInputs, Out: AutotuneOutput>(
        &mut self,
        batch: Batch,
        autotunables: &[&TuneFn<F, Out>],
        inputs: <F as TuneInputs>::At<'a>,
        sampler: CandidateSampler<'_>,
    ) -> BatchOutcome
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        self.held.clear();
        self.held.extend(batch.indices());
        let held = sampler.hold(|| self.measure(batch, autotunables, inputs, sampler));
        held.unwrap_or_else(|err| BatchOutcome::hold_failed(&self.held, autotunables, &err))
    }

    /// Measure `batch` in a measuring stage, its stream executing what it launches.
    fn measure<'a, F: TuneInputs, Out: AutotuneOutput>(
        &mut self,
        batch: Batch,
        autotunables: &[&TuneFn<F, Out>],
        inputs: <F as TuneInputs>::At<'a>,
        sampler: CandidateSampler<'_>,
    ) -> BatchOutcome
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        let measuring = sampler.open_measuring_stage(self.stage_enterer.as_deref_mut());
        let outcome =
            cubecl_environment::future::block_on(self.drive(batch, autotunables, inputs, sampler));
        core::mem::drop(measuring);
        outcome
    }

    async fn drive<'a, F: TuneInputs, Out: AutotuneOutput>(
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
                CandidateTracker::new(
                    entry.index,
                    operation_at(autotunables, entry.index).name().to_string(),
                )
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
            .filter(|candidate| candidate.state().summary.count == 0)
            .for_each(CandidateTracker::skip);

        let confirmed = match first_pass {
            Some(slot) => Some(slot),
            None => {
                self.sample_rounds(&mut candidates, autotunables, &inputs, sampler)
                    .await
            }
        };

        self.conclude_batch(candidates, confirmed)
    }

    /// The [first pass](FirstPassPlanner), its samples resolved inline so the batch can stop
    /// before the remaining kernels compile: the one place where paying a device round trip per
    /// sample is worth it. Answers the slot of the candidate a short circuit confirmed, if one did.
    async fn first_pass_all<'a, F: TuneInputs, Out: AutotuneOutput>(
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
            let candidate = candidates
                .get_mut(slot)
                .expect("a slot should be one of the batch's candidates");
            self.first_pass(autotunables, inputs, candidate, sampler)
                .await;
            candidate.track_cost_since(launched);

            if let Some(best) = candidate.state().live_best() {
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
    fn compile_ahead<'a, F: TuneInputs, Out: AutotuneOutput>(
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
    fn conclude_batch(
        &mut self,
        candidates: Vec<CandidateTracker>,
        confirmed: Option<usize>,
    ) -> BatchOutcome {
        let short_circuit = confirmed.map(|slot| {
            let candidate = candidates
                .get(slot)
                .expect("a slot should be one of the batch's candidates");
            candidate.state().name.to_string()
        });
        let steps = candidates
            .iter()
            .filter(|_| self.options.track_steps)
            .map(|candidate| {
                (
                    candidate.state().name.to_string(),
                    candidate.state().elapsed,
                )
            })
            .collect();
        let unmeasured = candidates
            .iter()
            .map(CandidateTracker::state)
            .filter(|state| state.unmeasured)
            .map(|state| state.index)
            .collect();
        self.ranked.clear();
        self.ranked
            .extend(candidates.iter().enumerate().map(|(slot, candidate)| {
                candidate.state().live && confirmed.is_none_or(|hit| hit == slot)
            }));
        let results: Vec<(usize, AutotuneResult)> = candidates
            .into_iter()
            .map(|candidate| (candidate.state().index, candidate.into_result()))
            .collect();
        let winner = best_scored(
            results
                .iter()
                .zip(&self.ranked)
                .filter(|(_, ranked)| **ranked)
                .map(|((_, result), _)| result),
        );

        BatchOutcome {
            any_success: results.iter().any(|(_, result)| result.outcome.is_ok()),
            results,
            steps,
            short_circuit,
            winner,
            unmeasured,
        }
    }

    /// How many of a batch of `len` candidates the first pass compiles together when the next
    /// one to measure is not compiled yet. Where the round may stop early, `patient` or bounded by
    /// a time limit, its first window holds as many as compile at once. A `patient` batch, whose
    /// groups may skip the rest of their members, keeps to that window. Elsewhere the rest of the
    /// batch compiles at once: a bounded round its first window did not end rarely ends on the
    /// next, and window after window would wait on each window's slowest kernel, then on its
    /// measurements, with the compiler idle.
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
            return self.options.compile_parallelism_count;
        }
        len
    }

    /// Warm up a candidate and take its first sample. A candidate that rejected the problem while
    /// compiling is out of the round already.
    async fn first_pass<'a, F: TuneInputs, Out: AutotuneOutput>(
        &mut self,
        autotunables: &[&TuneFn<F, Out>],
        inputs: &<F as TuneInputs>::At<'a>,
        candidate: &mut CandidateTracker,
        sampler: CandidateSampler<'_>,
    ) where
        <F as TuneInputs>::At<'a>: Clone,
    {
        if !candidate.state().live {
            return;
        }
        let operation = operation_at(autotunables, candidate.state().index);
        if let Err(err) = sampler.warmup_once(operation, inputs.clone()) {
            return candidate.fail(err);
        }
        self.take_sample(operation, inputs, candidate, sampler)
            .await;
    }

    /// The slot of the first of `candidates`, in priority order, confirmed under the time limit,
    /// each one that has a sample under it first taking the confirmation samples it lacks. `None`
    /// without a limit or with the short circuit off.
    async fn confirm_under_limit<'a, F: TuneInputs, Out: AutotuneOutput>(
        &mut self,
        autotunables: &[&TuneFn<F, Out>],
        inputs: &<F as TuneInputs>::At<'a>,
        candidates: &mut [CandidateTracker],
        sampler: CandidateSampler<'_>,
    ) -> Option<usize>
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        let (limit, required) = self.confirmation_bar()?;

        for (slot, candidate) in candidates
            .iter_mut()
            .enumerate()
            .filter(|(_, candidate)| candidate.state().live && candidate.any_under(limit))
        {
            let operation = operation_at(autotunables, candidate.state().index);
            let launched = self.options.track_steps.then(Instant::now);
            while candidate.state().summary.count < required
                && self
                    .take_sample(operation, inputs, candidate, sampler)
                    .await
            {}
            candidate.track_cost_since(launched);
            if candidate.confirmed_under(limit, required) {
                return Some(slot);
            }
        }

        None
    }

    /// Queue one sample and resolve it immediately. Returns whether the candidate survived.
    async fn take_sample<'a, F: TuneInputs, Out: AutotuneOutput>(
        &mut self,
        operation: &TuneFn<F, Out>,
        inputs: &<F as TuneInputs>::At<'a>,
        candidate: &mut CandidateTracker,
        sampler: CandidateSampler<'_>,
    ) -> bool
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        let sampled = self.sample(operation, inputs, sampler);
        match sampled.map(|profile| candidate.track_queued(profile)) {
            Ok(profile) => candidate.track_ticks(profile.resolve().await),
            Err(err) => candidate.fail(err),
        }
        candidate.state().live
    }

    /// Queue one measured sample of `operation`, the eviction first.
    fn sample<'a, F: TuneInputs, Out: AutotuneOutput>(
        &mut self,
        operation: &TuneFn<F, Out>,
        inputs: &<F as TuneInputs>::At<'a>,
        sampler: CandidateSampler<'_>,
    ) -> Result<ProfileDuration, AutotuneError>
    where
        <F as TuneInputs>::At<'a>: Clone,
    {
        sampler.sample_once(operation, inputs.clone(), self.evictor.as_deref_mut())
    }

    /// The later rounds, each resolved whole, answering as
    /// [`first_pass_all`](Self::first_pass_all) does.
    async fn sample_rounds<'a, F: TuneInputs, Out: AutotuneOutput>(
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
    fn queue_round<'a, F: TuneInputs, Out: AutotuneOutput>(
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
            candidate.state().live && candidate.state().summary.count < max_sample_count
        }) {
            let launched = self.options.track_steps.then(Instant::now);
            let operation = operation_at(autotunables, candidate.state().index);
            match self.sample(operation, inputs, sampler) {
                Ok(profile) => self.pending.push((slot, candidate.track_queued(profile))),
                Err(err) => candidate.fail(err),
            }
            candidate.track_cost_since(launched);
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

    /// Resolve every sample the round queued, together, each timed on its own: they resolve
    /// concurrently, so the round's wall clock cannot be split across candidates, but the wait for
    /// one candidate's own readback can.
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
            let candidate = candidates
                .get_mut(slot)
                .expect("a slot should be one of the batch's candidates");
            candidate.track_ticks(ticks);
            candidate.track_cost(track_steps.then_some(waited));
        }
    }

    /// The time limit a short circuit needs a candidate under, and how many of its samples, where
    /// the short circuit is on.
    fn confirmation_bar(&self) -> Option<(Duration, usize)> {
        let limit = self.options.short_circuit_limit?;
        Some((
            limit,
            self.options.config.clamped_short_circuit_sample_count(),
        ))
    }

    /// The slot of the first candidate confirmed under the limit, which stops everything.
    ///
    /// The decision is persisted and reused on later runs, so it needs more than one sample:
    /// a single lucky measurement would pin a suboptimal kernel permanently.
    fn short_circuit_hit(&self, candidates: &[CandidateTracker]) -> Option<usize> {
        let (limit, required) = self.confirmation_bar()?;
        candidates
            .iter()
            .position(|candidate| candidate.confirmed_under(limit, required))
    }

    /// Drop candidates whose best time is past the speed factor, the slowest first, keeping
    /// [`MIN_SURVIVOR_COUNT`].
    ///
    /// Comparing minima is already outlier resistant: a slow sample cannot lower a candidate's
    /// best, so nothing is eliminated on the strength of one bad measurement.
    fn eliminate(&mut self, candidates: &mut [CandidateTracker], min_sample_count: usize) {
        let Some(best) = candidates
            .iter()
            .filter_map(|candidate| candidate.state().live_best())
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
                .filter(|(_, candidate)| candidate.state().summary.count >= min_sample_count)
                .filter_map(|(slot, candidate)| {
                    candidate
                        .state()
                        .live_best()
                        .map(|best| (best, Reverse(slot)))
                })
                .filter(|(best, _)| *best > threshold),
        );

        for _ in 0..budget {
            let Some((_, Reverse(slot))) = self.eliminable.pop() else {
                return;
            };
            candidates
                .get_mut(slot)
                .expect("a slot should be one of the batch's candidates")
                .eliminate();
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
        let mut live = live(candidates).peekable();
        let converged = live.peek().is_some() && live.all(|state| state.summary.converged);
        if converged {
            Self::Done
        } else {
            Self::Continue
        }
    }
}

/// What every candidate of `candidates` still in the round knows of itself.
fn live(candidates: &[CandidateTracker]) -> impl Iterator<Item = CandidateState<'_>> {
    candidates
        .iter()
        .map(CandidateTracker::state)
        .filter(|state| state.live)
}

/// The function of the candidate at `index` of `autotunables`, by tunable index.
fn operation_at<'f, F: TuneInputs, Out>(
    autotunables: &[&'f TuneFn<F, Out>],
    index: usize,
) -> &'f TuneFn<F, Out> {
    autotunables
        .get(index)
        .copied()
        .expect("an index should be one of the set's candidates")
}

/// How many of `candidates` are still in the round.
fn live_count(candidates: &[CandidateTracker]) -> usize {
    live(candidates).count()
}

/// Candidates kept alive no matter how far behind they are, so a batch never narrows to a
/// single kernel that was never compared against anything.
const MIN_SURVIVOR_COUNT: usize = 2;

/// Compile the kernels of `candidates` in one batch, on as many threads as compile at once,
/// rather than each inside its own warmup, one after another.
///
/// Each runs once on this stream with its launches discarded, which queues the kernels they reach
/// ([`LaunchMode::Queue`](crate::execution::LaunchMode::Queue)); the queue then compiles together.
/// Only this stream discards, so another thread's launches still run. A candidate that rejects
/// the problem or panics keeps its error, as its warmup would have reported it: it is never a
/// kernel to fall back on. One whose kernel fails to compile reports it at its warmup.
fn compile_together<'a, 'c, F: TuneInputs, Out: AutotuneOutput>(
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
    let gathering = crate::tune::measure::candidate_gathering::CandidateGathering::open();
    for candidate in candidates {
        if let Err(unusable @ (AutotuneError::Rejected { .. } | AutotuneError::Panicked { .. })) =
            sampler.contain_race(
                operation_at(autotunables, candidate.state().index),
                inputs.clone(),
            )
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
            compile_parallelism_count: 1,
        }
    }

    fn candidate(slot: usize, samples: impl IntoIterator<Item = u64>) -> CandidateTracker {
        let mut candidate = CandidateTracker::new(slot, alloc::format!("k{slot}"));
        for micros in samples {
            candidate.track_duration(Duration::from_micros(micros));
        }
        candidate
    }

    fn live(candidates: &[CandidateTracker]) -> Vec<usize> {
        candidates
            .iter()
            .map(CandidateTracker::state)
            .filter(|state| state.live)
            .map(|state| state.index)
            .collect()
    }

    fn windowed(short_circuit_limit: Option<Duration>) -> Scheduler<'static> {
        Scheduler::new(
            ScheduleOptions {
                short_circuit_limit,
                compile_parallelism_count: 8,
                ..options(1.5)
            },
            None,
            None,
        )
    }

    fn windows(first_count: usize, later_count: usize) -> CompileWindows {
        CompileWindows {
            first_count,
            later_count,
        }
    }

    #[test]
    fn a_bounded_round_compiles_its_first_window_then_the_rest() {
        let bounded = windowed(Some(Duration::from_micros(10)));

        assert_eq!(bounded.compile_windows(100, false), windows(8, 100));
    }

    #[test]
    fn a_patient_round_compiles_a_window_at_a_time() {
        assert_eq!(windowed(None).compile_windows(100, true), windows(8, 8));
    }

    #[test]
    fn a_round_that_cannot_stop_compiles_the_whole_batch() {
        assert_eq!(
            windowed(None).compile_windows(100, false),
            windows(100, 100)
        );
    }

    #[test]
    fn a_short_circuit_decides_the_confirmed_candidate() {
        let candidates = vec![candidate(0, [40]), candidate(1, [50, 50, 50])];

        let outcome = schedule(1.5).conclude_batch(candidates, Some(1));

        assert_eq!(outcome.winner, Some(1));
    }

    #[test]
    fn a_short_circuit_names_the_confirmed_candidate() {
        let candidates = vec![candidate(0, [40]), candidate(1, [50, 50, 50])];

        let outcome = schedule(1.5).conclude_batch(candidates, Some(1));

        assert_eq!(outcome.short_circuit.as_deref(), Some("k1"));
    }

    #[test]
    fn a_candidate_compiled_but_never_measured_is_reported_unmeasured() {
        let mut prefetched = candidate(1, []);
        prefetched.skip();

        let outcome =
            schedule(1.5).conclude_batch(vec![candidate(0, [10, 10, 10]), prefetched], None);

        assert_eq!(outcome.unmeasured, vec![1]);
    }

    #[test]
    fn a_tracked_batch_reports_every_candidate_s_cost_in_order() {
        let mut tracked = Scheduler::new(
            ScheduleOptions {
                track_steps: true,
                ..options(1.5)
            },
            None,
            None,
        );
        let mut first = candidate(0, [10]);
        first.track_cost(Some(Duration::from_micros(3)));

        let outcome = tracked.conclude_batch(vec![first, candidate(1, [20])], None);

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
    fn the_slowest_goes_first_when_the_survivor_floor_forces_a_choice() {
        // Every candidate but the leader deserves elimination on the numbers alone.
        let mut candidates = vec![
            candidate(0, [100, 100, 100]),
            candidate(1, [900, 900, 900]),
            candidate(2, [950, 950, 950]),
        ];

        schedule(1.5).eliminate(&mut candidates, 3);

        assert_eq!(live(&candidates), vec![0, 1]);
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

    fn short_circuiting() -> Scheduler<'static> {
        Scheduler::new(
            ScheduleOptions {
                short_circuit_limit: Some(Duration::from_micros(200)),
                ..options(1.5)
            },
            None,
            None,
        )
    }

    #[test]
    fn one_sample_under_the_limit_is_no_short_circuit() {
        let one = vec![candidate(0, [100, 900])];

        assert_eq!(short_circuiting().short_circuit_hit(&one), None);
    }

    #[test]
    fn the_configured_confirmations_short_circuit() {
        let two = vec![candidate(0, [100, 900, 150])];

        assert_eq!(short_circuiting().short_circuit_hit(&two), Some(0));
    }

    #[test]
    fn short_circuit_stays_off_without_a_limit() {
        let candidates = vec![candidate(0, [1, 1, 1])];
        assert_eq!(schedule(1.5).short_circuit_hit(&candidates), None);
    }

    #[test]
    fn an_eliminated_candidate_never_wins_on_its_shorter_sample_set() {
        // Elimination froze candidate 1 at three tight samples while candidate 0 kept sampling
        // and picked up spread, so candidate 1 scores better on the arithmetic alone. It was
        // dropped on the evidence, and the evidence is what decides.
        let mut eliminated = candidate(1, [10, 10, 10]);
        eliminated.eliminate();
        let survivor = candidate(0, [50, 900, 60, 800, 55]);

        let outcome = schedule(1.5).conclude_batch(vec![survivor, eliminated], None);

        assert_eq!(outcome.winner, Some(0));
    }

    #[test]
    fn an_eliminated_candidate_scores_better_on_the_arithmetic() {
        // What makes the test above meaningful.
        let eliminated = candidate(1, [10, 10, 10]).into_result();
        let survivor = candidate(0, [50, 900, 60, 800, 55]).into_result();

        assert!(eliminated.score() < survivor.score());
    }

    #[test]
    fn the_fastest_survivor_is_the_one_declared() {
        let candidates = vec![
            candidate(0, [100, 100, 100]),
            candidate(1, [40, 40, 40]),
            candidate(2, [70, 70, 70]),
        ];

        assert_eq!(
            schedule(1.5).conclude_batch(candidates, None).winner,
            Some(1)
        );
    }

    #[test]
    fn no_survivor_leaves_the_decision_open() {
        // Every candidate failed, so there is nothing to declare and the caller falls back to
        // its own comparison rather than being handed an index that measured nothing.
        let mut failed = candidate(0, [10, 10, 10]);
        failed.fail(AutotuneError::NotMeasured {
            name: "k0".to_string(),
        });

        let outcome = schedule(1.5).conclude_batch(vec![failed, candidate(1, [])], None);

        assert_eq!(outcome.winner, None);
    }

    #[test]
    fn a_round_ends_once_every_live_candidate_converged() {
        let converged = candidate(0, [20, 20, 20, 20, 20]);
        let mut out = candidate(1, [10]);
        out.eliminate();

        assert_eq!(
            RoundEnd::after_elimination(&[converged, out]),
            RoundEnd::Done
        );
    }

    #[test]
    fn a_round_continues_while_a_live_candidate_improves() {
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
