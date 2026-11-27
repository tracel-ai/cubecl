use alloc::sync::Arc;
use alloc::vec::Vec;
#[cfg(persistence)]
use alloc::{format, string::String};
use cubecl_environment::sync::Mutex;

use crate::client::Client;
use crate::config::Logger;
#[cfg(persistence)]
use crate::config::autotune::AutotuneLogLevel;
use crate::config::{CubeClRuntimeConfig, RuntimeConfig};
use crate::device::HardwareIdentity;
use crate::execution::{ProcessMode, TuneRegistration};
use crate::tune::cache::tracker::{TuneCacheResult, TuneTracker};
use crate::tune::cache::tune_logger::TuneLogger;
use crate::tune::drive::candidate_sampler::CandidateSampler;
use crate::tune::drive::committer::TuneCommitter;
use crate::tune::drive::driver::{DriveOptions, TuneDriver};
use crate::tune::drive::gathering::CandidateGatherer;
use crate::tune::drive::pending_unmarker::PendingUnmarker;
use crate::tune::drive::settler::TuneSettler;
use crate::tune::{AutotuneCheckable, AutotuneKeyable, CheckResult, TunableSet, TuneInputs};

/// Tunes the keys of one candidate list on the devices of one hardware, and keeps their answers.
///
/// On wasm, [`tune`](Self::tune) spawns its work on the browser event loop; elsewhere it blocks
/// inline. Either way the benchmarking itself is synchronous; only the per-sample profile
/// resolution is awaited.
///
/// # Assumptions
///
/// - Every tune of one tuner runs the same candidate list: the answers are keyed by the key alone.
/// - `name` holds no `/`: the table is named `hardware/name`.
#[derive(Debug)]
pub struct Tuner<K: AutotuneKeyable> {
    cache: Arc<Mutex<TuneTracker<K>>>,
    logger: Arc<Mutex<Logger>>,
}

/// What a key a tune opened holds until it settles: the unmarker of its pending mark and where
/// its tune was registered, or nothing where it only gathers kernels.
type Opened<K> = Option<(PendingUnmarker<K>, Option<TuneRegistration>)>;

impl<K: AutotuneKeyable> Tuner<K> {
    /// A tuner whose unsettled keys are read from the persistent cache when persistence is
    /// available (feature `persistence`: a file natively, OPFS in the browser), in the table every
    /// device of `hardware` shares.
    pub fn new(name: &str, hardware: &HardwareIdentity) -> Self {
        Self {
            cache: Arc::new(Mutex::new(TuneTracker::new(name, hardware))),
            logger: Arc::new(Mutex::new(Logger::new())),
        }
    }

    /// Drop the answers of an environment the process switched away from, so the next
    /// [`fastest`](Self::fastest) answers the current one's.
    pub fn reset_if_environment_switched(&self) {
        #[cfg(persistence)]
        self.cache.lock().reset_if_environment_switched();
    }

    /// What memory holds for `key`: a fast-path probe that never reads the persistent cache, so
    /// a miss here falls through to [`tune`](Self::tune), which reads the store and resolves the
    /// real state.
    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        self.cache.lock().fastest(key)
    }

    /// Answer `key` on `client`'s device: from memory, from the store when memory holds no
    /// answer, and otherwise by tuning `tunables` on `inputs`. `checks` cross-checks the
    /// candidates, collecting a mismatch when asked to: only a tune that drives reports them, so
    /// every other answer runs them uncollected.
    ///
    /// # Assumptions
    ///
    /// - `key` is the key `tunables` generates for `inputs`.
    /// - `tunables` holds a candidate, and some candidate serves `inputs`.
    ///
    /// # Guarantees
    ///
    /// - Never answers [`Miss`](TuneCacheResult::Miss).
    /// - One tune of a key runs at a time within an environment: a concurrent call answers
    ///   [`Pending`](TuneCacheResult::Pending). A switch drops the mark, so a tune may run once
    ///   more for the new environment while the old one's finishes.
    /// - A tune that unwinds, or stops on a lost card, leaves the key to the next call.
    /// - An answer picked with nothing measured is kept in memory, never in the store.
    pub fn tune<'a, F: TuneInputs, Out: AutotuneCheckable, Id: Sync>(
        &self,
        key: &K,
        inputs: &F::At<'a>,
        tunables: &TunableSet<K, F, Out, Id>,
        client: &Client,
        checks: impl FnOnce(bool) -> Option<Vec<CheckResult>>,
    ) -> TuneCacheResult
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        // A key reached by candidates that only gather kernels would measure on the outputs of
        // discarded launches.
        let compiling =
            ProcessMode::current() == ProcessMode::CompileOnly || CandidateGatherer::is_gathering();
        let opened = match self.open(key, tunables, client, compiling) {
            Ok(opened) => opened,
            Err(answer) => return answered(answer, checks),
        };

        log::info!("Tuning {key}");
        // A `CompileOnly` override queues the kernels of the top of the plan and measures
        // nothing; the key stays untuned for the pass that profiles it.
        let Some((unmarker, gathered)) = opened else {
            return answered(self.compile_plan(key, inputs, tunables, client), checks);
        };
        // Sure to measure now: a tune registered nowhere yet registers with the override open.
        let settling = (unmarker, gathered.or_else(TuneRegistration::register));
        let decisions = self.logger.lock().config().autotune.decisions_enabled();
        self.driver(key, inputs, tunables, client, settling, checks(decisions))
            .drive(client)
    }
}

impl<K: AutotuneKeyable> Tuner<K> {
    /// Open the tune of `key` under the lock, or answer it without one: from memory, from the
    /// store, or at once for a set of one candidate.
    ///
    /// Marked before the candidates run, so one that panics leaves the key to the pass that
    /// profiles it rather than pending for good. Each card queues the candidates on itself, since
    /// a tune compiles only what its own card queued.
    fn open<F: TuneInputs, Out: 'static, Id>(
        &self,
        key: &K,
        tunables: &TunableSet<K, F, Out, Id>,
        client: &Client,
        compiling: bool,
    ) -> Result<Opened<K>, TuneCacheResult> {
        // Scoped: the rest of the tune re-locks `self.cache`, and the mutex is non-reentrant.
        let mut cache = self.cache.lock();
        #[cfg(persistence)]
        cache.reset_if_environment_switched();
        if let Some(answer) = cache.answer(key, client.service_id(), compiling) {
            return Err(answer);
        }
        // A key with one candidate is answered, not tuned: nothing to read or to store.
        if tunables.len() <= 1 {
            return Err(cache.track_answer(key.clone(), 0));
        }
        #[cfg(persistence)]
        if let hit @ TuneCacheResult::Hit { .. } = self.track_stored(&mut cache, key, tunables) {
            return Err(hit);
        }
        // Its tune is registered with the open override only when no other key's candidates
        // reached it: the pass that profiles stops a plan at its first close-enough candidate, and
        // may never run the one that reaches a key gathered inside it. Such a key is registered if
        // it is ever measured, as an ungathered one is.
        if compiling {
            return Ok(track_compiled(&mut cache, key, client));
        }
        let gathered = cache.track_pending(key.clone());
        // Made last under the lock: no gap follows the mark, and the unmarker, whose drop takes
        // this lock, can never drop while it is held.
        let unmarker = PendingUnmarker::new(self.cache.clone(), &cache, key.clone());
        Ok(Some((unmarker, gathered)))
    }

    /// Track the store's answer for `key`, read under the lock `cache` holds and logged at the
    /// full level, and answer it.
    #[cfg(persistence)]
    fn track_stored<F: TuneInputs, Out: 'static, Id>(
        &self,
        cache: &mut TuneTracker<K>,
        key: &K,
        tunables: &TunableSet<K, F, Out, Id>,
    ) -> TuneCacheResult {
        let checksum = tunables.compute_checksum();
        let stored = cache.track_stored(key, &checksum, tunables.len());
        let mut log = self.logger.lock();
        if let AutotuneLogLevel::Full = log.config().autotune.logger.level {
            log.log_autotune(&format!(
                "look up key={key}, checksum={checksum}: {stored:?}"
            ));
        }
        stored
    }

    /// The driver of the tune of `key`, prepared to measure: its inputs generated, the kernels a
    /// `CompileOnly` override queued compiled, its plan, bounds and record begun. `settling` is
    /// the unmarker of the key's pending mark and where its tune is registered.
    fn driver<'t, 'a, F: TuneInputs, Out: AutotuneCheckable, Id>(
        &self,
        key: &K,
        inputs: &F::At<'a>,
        tunables: &'t TunableSet<K, F, Out, Id>,
        client: &Client,
        settling: (PendingUnmarker<K>, Option<TuneRegistration>),
        checks: Option<Vec<CheckResult>>,
    ) -> TuneDriver<'t, 'a, K, F, Out>
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        #[cfg(persistence)]
        let checksum: String = tunables.compute_checksum();
        #[cfg(persistence)]
        let mut recorder = crate::tune::cache::record::TuneRecorder::new(
            self.cache.lock().table(),
            key,
            &checksum,
        );
        #[cfg(persistence)]
        let recorded = recorder.is_open();
        #[cfg(not(persistence))]
        let recorded = false;
        let mut logger = TuneLogger::new(&self.logger.lock(), recorded);
        #[cfg(feature = "autotune-checks")]
        checks
            .into_iter()
            .for_each(|checks| logger.log_checks(checks));
        #[cfg(not(feature = "autotune-checks"))]
        let _ = checks;

        let test_inputs = tunables.generate_inputs(key, inputs);
        // Kernels a `CompileOnly` override queued compile here, together, before anything is
        // timed: left to the next launch, they would all compile inside the first candidate's
        // warmup, and its measurement would carry every candidate's compilation.
        client.compile_queued();
        let plan = tunables.plan(key);
        let autotunables: Vec<_> = tunables.autotunables().collect();
        #[cfg(persistence)]
        recorder.record_plan(&plan, &autotunables);
        let bounds = tunables.bounds(key, inputs);
        let limit = bounds.as_ref().and_then(|bounds| bounds.time_limit());
        logger.log_bounds(bounds.clone(), limit);
        let options = DriveOptions {
            limit,
            track_steps: logger.context().is_some(),
        };

        let (unmarker, registration) = settling;
        let settler = TuneSettler::new(
            self.cache.clone(),
            self.logger.clone(),
            registration,
            unmarker,
            #[cfg(persistence)]
            crate::tune::drive::persister::TunePersister::new(checksum, limit, bounds, recorder),
        );
        let names = autotunables.iter().map(|tune| tune.name());
        let committer = TuneCommitter::new(key.clone(), names, logger, settler);
        TuneDriver::new(
            committer,
            key.clone(),
            tunables,
            inputs,
            test_inputs,
            plan,
            options,
        )
    }

    /// Runs the top of the plan once each, on the inputs a tune generates, so their launches
    /// queue their kernels under a `CompileOnly` override, and answers
    /// [`Compiled`](TuneCacheResult::Compiled): the key is left untuned.
    ///
    /// Runs [`TunePlanner::plan_top`](crate::tune::plan::planner::TunePlanner::plan_top) with the
    /// configured compilation parallelism; the tune compiles the rest at their turn. The pass
    /// holds the device as the set's access does, and contains a candidate that panics as the
    /// measuring pass does: it serves nothing. A device that cannot be held queues nothing.
    fn compile_plan<'a, F: TuneInputs, Out: AutotuneCheckable, Id: Sync>(
        &self,
        key: &K,
        inputs: &F::At<'a>,
        tunables: &TunableSet<K, F, Out, Id>,
        client: &Client,
    ) -> TuneCacheResult
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        let test_inputs = tunables.generate_inputs(key, inputs);
        let mut plan = tunables.plan(key);
        let gathering = tunables
            .bound_stage_enterer(key, inputs)
            .map(|mut enter| enter(crate::tune::TuneStage::Gathering));
        let parallelism = CubeClRuntimeConfig::get().compilation.parallelism();
        let sampler = CandidateSampler::new(client, tunables.device_access());
        let held = sampler.hold(move || {
            let gatherer = CandidateGatherer::gather();
            plan.plan_top(parallelism, |index| {
                sampler
                    .contain_race(tunables.function_at(index), test_inputs.clone())
                    .is_ok()
            });
            core::mem::drop(gatherer);
        });
        if let Err(err) = held {
            log::warn!(
                "The candidates of {key} queued nothing: the device could not be held. {err}"
            );
        }
        core::mem::drop(gathering);
        TuneCacheResult::Compiled
    }
}

/// `answer`, a key answered without a tune, its candidates cross-checked uncollected: nothing
/// reports them.
fn answered(
    answer: TuneCacheResult,
    checks: impl FnOnce(bool) -> Option<Vec<CheckResult>>,
) -> TuneCacheResult {
    checks(false);
    answer
}

/// Track `key` as one whose candidates' kernels `client`'s card queues, in place of a tune, and
/// open nothing to settle.
fn track_compiled<K: AutotuneKeyable>(
    cache: &mut TuneTracker<K>,
    key: &K,
    client: &Client,
) -> Opened<K> {
    cache.track_compiled(key.clone(), client.service_id(), register_ungathered);
    None
}

/// The registration of a key's tune, unless another key's candidates reached it.
fn register_ungathered() -> Option<TuneRegistration> {
    (!CandidateGatherer::is_gathering())
        .then(TuneRegistration::register)
        .flatten()
}
