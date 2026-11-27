use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::time::Duration;

use super::binding::{BoundEvictor, BoundStageEnterer};
use super::candidate_sampler::CandidateSampler;
use super::committer::{LaunchedBench, QueuedBench, ResolvedBench, TuneCommitter, unmeasured_pick};
#[cfg(not(target_family = "wasm"))]
use super::plan::Batch;
use super::plan::TunePlanner;
use super::tune_benchmark::tune_benchmark;
use super::tune_cache::TuneCacheResult;
use super::tune_fn::TuneFn;
use super::{
    AutotuneCheckable, AutotuneKeyable, AutotuneResult, TunableSet, TuneDeviceAccess, TuneInputs,
};
use crate::client::Client;
use crate::config::RuntimeConfig;
#[cfg(not(target_family = "wasm"))]
use crate::server::ServerError;

/// How a [`TuneDriver`] measures.
#[derive(Debug, Clone, Copy)]
pub struct DriveOptions {
    /// The slowest median close enough to the peak, when the set has bounds.
    pub limit: Option<Duration>,
    /// Whether each candidate's cost is logged.
    pub track_steps: bool,
}

/// Drives one tune of a key: benchmarks its candidates batch by batch, the adaptive way or a
/// fixed sample count each, then hands what they measured to its [`TuneCommitter`]. `'t` borrows
/// the tunable set, `'i` the benchmark inputs.
pub struct TuneDriver<'t, 'i, K: AutotuneKeyable, F: TuneInputs, Out> {
    committer: TuneCommitter<K>,
    key: K,
    /// Every candidate's function, by tunable index (Position Usage).
    autotunables: Vec<&'t TuneFn<F, Out>>,
    test_inputs: <F as TuneInputs>::At<'i>,
    /// What runs before every measured sample, bound to the key and the reference inputs.
    evictor: Option<Box<BoundEvictor<'i>>>,
    stage_enterer: Option<Box<BoundStageEnterer<'i>>>,
    access: TuneDeviceAccess,
    plan: TunePlanner,
    /// The limit a candidate is close enough under, where the short circuit is on. Only used on
    /// native, where a benchmark can be resolved inline to exit early.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    short_circuit_limit: Option<Duration>,
    track_steps: bool,
}

impl<'t, 'i, K: AutotuneKeyable, F: TuneInputs, Out: AutotuneCheckable>
    TuneDriver<'t, 'i, K, F, Out>
where
    <F as TuneInputs>::At<'i>: Clone + Send,
{
    /// A driver of the tune of `key` over `tunables`, measured on `test_inputs`, its hooks bound
    /// to the reference `inputs`, walking `plan` and committing through `committer`.
    pub fn new(
        committer: TuneCommitter<K>,
        key: K,
        tunables: &'t TunableSet<K, F, Out, impl Sized>,
        inputs: &<F as TuneInputs>::At<'i>,
        test_inputs: <F as TuneInputs>::At<'i>,
        plan: TunePlanner,
        options: DriveOptions,
    ) -> Self {
        let short_circuit = tunables.is_short_circuit_enabled()
            && !crate::config::CubeClRuntimeConfig::get()
                .autotune
                .disable_short_circuit;
        Self {
            evictor: tunables.evictor(&key, inputs),
            stage_enterer: tunables.bound_stage_enterer(&key, inputs),
            committer,
            key,
            autotunables: tunables.autotunables().collect(),
            test_inputs,
            access: tunables.device_access(),
            plan,
            short_circuit_limit: options.limit.filter(|_| short_circuit),
            track_steps: options.track_steps,
        }
    }

    /// Tune the key on `client`'s device, as the configuration asks: adaptively, natively, unless
    /// it turned that off, and with a fixed sample count otherwise.
    pub fn drive(self, client: &Client) -> TuneCacheResult {
        #[cfg(not(target_family = "wasm"))]
        if crate::config::CubeClRuntimeConfig::get()
            .autotune
            .bench
            .adaptive
        {
            return self.drive_adaptive(client);
        }

        self.drive_fixed(client)
    }
}

impl<'t, 'i, K: AutotuneKeyable, F: TuneInputs, Out: AutotuneCheckable>
    TuneDriver<'t, 'i, K, F, Out>
where
    <F as TuneInputs>::At<'i>: Clone + Send,
{
    /// Round robin the candidates, eliminating them as the evidence allows. Native only: the
    /// driver has to resolve samples between rounds, which it cannot do on the browser event loop.
    #[cfg(not(target_family = "wasm"))]
    fn drive_adaptive(mut self, client: &Client) -> TuneCacheResult {
        let config = crate::config::CubeClRuntimeConfig::get();
        let mut scheduler = super::schedule::Scheduler::new(
            super::schedule::ScheduleOptions {
                config: config.autotune.bench.clone(),
                short_circuit_limit: self.short_circuit_limit,
                track_steps: self.track_steps,
                compile_parallelism: config.compilation.parallelism(),
            },
            self.evictor.take(),
            self.stage_enterer.take(),
        );
        let outcome = scheduler.schedule_plan(
            &mut self.plan,
            &self.autotunables,
            &self.test_inputs,
            CandidateSampler::new(client, self.access),
            self.committer.results_mut(),
        );
        core::mem::drop(scheduler);

        for (name, duration) in outcome.steps {
            self.committer.log_step(name, duration);
        }
        if outcome.exhausted {
            return self.exhausted(client);
        }
        if let Some(name) = outcome.short_circuit {
            self.committer.log_short_circuit(name);
        }
        self.committer.commit(Vec::new(), outcome.decided)
    }

    /// Benchmark every candidate with a fixed sample count, batch by batch until one measures a
    /// candidate, resolving each batch's samples before the next. This is the strategy the
    /// configuration may ask for natively.
    #[cfg(not(target_family = "wasm"))]
    fn drive_fixed(mut self, client: &Client) -> TuneCacheResult {
        let sampler = CandidateSampler::new(client, self.access);
        loop {
            let batch = self.plan.plan_next();
            if batch.is_empty() {
                return self.exhausted(client);
            }
            if self.bench_batch(&batch, sampler) {
                return self.committer.commit(Vec::new(), None);
            }
        }
    }

    /// Benchmark every candidate with a fixed sample count. This is the only strategy available
    /// on wasm, where nothing can be awaited inline: the first batch that queued a sample is
    /// resolved on the browser event loop, and the later ones are never reached.
    #[cfg(target_family = "wasm")]
    fn drive_fixed(mut self, client: &Client) -> TuneCacheResult {
        let sampler = CandidateSampler::new(client, self.access);
        loop {
            let batch = self.plan.plan_next();
            if batch.is_empty() {
                return self.exhausted(client);
            }
            let launched: Vec<LaunchedBench> = batch
                .indices()
                .map(|index| self.launch(index, sampler))
                .collect();
            if launched.iter().any(LaunchedBench::is_queued) {
                return self.committer.commit(launched, None);
            }
            launched
                .into_iter()
                .filter_map(LaunchedBench::into_failed)
                .for_each(|bench| self.committer.record_bench(bench));
        }
    }

    /// Benchmark `batch`, resolving its samples, and answer whether a candidate measured.
    #[cfg(not(target_family = "wasm"))]
    fn bench_batch(&mut self, batch: &Batch, sampler: CandidateSampler<'_>) -> bool {
        match self.short_circuit_limit {
            Some(limit) => self.bench_until_under(batch, sampler, limit),
            None => self.bench_all(batch, sampler),
        }
    }

    /// Launch every candidate of `batch`, then resolve their samples together.
    #[cfg(not(target_family = "wasm"))]
    fn bench_all(&mut self, batch: &Batch, sampler: CandidateSampler<'_>) -> bool {
        let launched: Vec<LaunchedBench> = batch
            .indices()
            .map(|index| self.launch(index, sampler))
            .collect();
        let resolved =
            cubecl_environment::future::block_on(super::committer::resolve_benches(launched));
        resolved
            .into_iter()
            .for_each(|bench| self.committer.record_bench(bench));
        self.batch_measured(batch)
    }

    /// Launch and resolve the candidates of `batch` one at a time, until one is under `limit`.
    #[cfg(not(target_family = "wasm"))]
    fn bench_until_under(
        &mut self,
        batch: &Batch,
        sampler: CandidateSampler<'_>,
        limit: Duration,
    ) -> bool {
        for index in batch.indices() {
            let launched = self.launch(index, sampler);
            let resolved =
                cubecl_environment::future::block_on(super::committer::resolve_bench(launched));
            let close_enough = resolved
                .result
                .outcome
                .as_ref()
                .is_ok_and(|outcome| outcome.computation.median <= limit);
            self.committer.record_bench(resolved);
            if close_enough {
                self.committer
                    .log_short_circuit(self.autotunables[index].name().to_string());
                return true;
            }
        }
        self.batch_measured(batch)
    }

    #[cfg(not(target_family = "wasm"))]
    /// Whether a candidate of `batch` measured.
    fn batch_measured(&self, batch: &Batch) -> bool {
        batch
            .indices()
            .any(|index| self.committer.results()[index].outcome.is_ok())
    }

    /// Launch the candidate at `index`: its samples queued, or the error that stopped it.
    fn launch(&mut self, index: usize, sampler: CandidateSampler<'_>) -> LaunchedBench {
        let operation = self.autotunables[index];
        let started = self.track_steps.then(cubecl_common::profile::Instant::now);
        let profiles = tune_benchmark(
            operation,
            self.test_inputs.clone(),
            sampler,
            self.evictor.as_deref_mut(),
            self.stage_enterer.as_deref_mut(),
        );
        let name = operation.name().to_string();
        let launch = started.map(|started| started.elapsed());
        match profiles {
            Ok(profiles) => LaunchedBench::Queued(QueuedBench {
                index,
                name,
                profiles,
                launch,
            }),
            Err(err) => LaunchedBench::Failed(ResolvedBench {
                index,
                step: launch.map(|launch| (name, launch)),
                result: AutotuneResult::error(err),
            }),
        }
    }

    /// End a tune whose every batch ran without a candidate measured: stop it on a lost card, or
    /// decide a candidate that executed unmeasured.
    fn exhausted(self, client: &Client) -> TuneCacheResult {
        #[cfg(not(target_family = "wasm"))]
        if let Some(lost) = lost_device(client) {
            log::error!(
                "Autotune of {} stopped, the device is lost: {lost}",
                self.key
            );
            return self.committer.stop_on_lost_card(client.service_id());
        }
        #[cfg(target_family = "wasm")]
        let _ = client;
        let decided = unmeasured_pick(&self.key, self.committer.results());
        self.committer.commit(Vec::new(), Some(decided))
    }
}

/// The fault a sync of `client` reports, where it poisoned the device.
#[cfg(not(target_family = "wasm"))]
fn lost_device(client: &Client) -> Option<ServerError> {
    cubecl_environment::future::block_on(client.sync())
        .err()
        .filter(ServerError::is_device_poisoned)
}
