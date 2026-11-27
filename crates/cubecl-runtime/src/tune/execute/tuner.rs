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
use crate::tune::cache::tracker::{TuneAnswer, TuneTracker};
use crate::tune::drive::committer::TuneCommitter;
use crate::tune::drive::driver::{DriveOptions, TuneDriver};
use crate::tune::drive::pending_unmarker::PendingUnmarker;
use crate::tune::drive::settler::TuneSettler;
use crate::tune::hook::hook_binder::enter_stage;
use crate::tune::log::tune_logger::TuneLogger;
use crate::tune::measure::candidate_gathering::CandidateGathering;
use crate::tune::measure::candidate_sampler::CandidateSampler;
use crate::tune::{
    AutotuneCheckable, AutotuneKeyable, CheckResult, TunableSet, TuneInputs, TuneStage,
};

/// Tunes the keys of one candidate list on the devices of one hardware, and keeps their tunes.
///
/// # Assumptions
///
/// - Every tune of one tuner runs the same candidate list: the tunes are keyed by the key alone.
#[derive(Debug)]
pub struct Tuner<K: AutotuneKeyable> {
    tracker: Arc<Mutex<TuneTracker<K>>>,
    logger: Arc<Mutex<Logger>>,
}

/// What a key a tune opened holds until it settles: the unmarker of its pending mark and where
/// its tune was registered, or nothing where it only gathers kernels.
type Opened<K> = Option<(PendingUnmarker<K>, Option<TuneRegistration>)>;

impl<K: AutotuneKeyable> Tuner<K> {
    /// A tuner whose unsettled keys are read from the persistent cache when persistence is
    /// available (feature `persistence`: a file natively, OPFS in the browser), in the table every
    /// device of `hardware` shares.
    ///
    /// # Assumptions
    ///
    /// - `name` holds no `/`: the table is named `hardware/name`.
    pub fn new(name: &str, hardware: &HardwareIdentity) -> Self {
        Self {
            tracker: Arc::new(Mutex::new(TuneTracker::new(name, hardware))),
            logger: Arc::new(Mutex::new(Logger::new())),
        }
    }

    /// Drop the tunes of an environment the process switched away from, then answer what memory
    /// holds for `key`, both under one lock of the tracker: the probe of a call's fast path.
    pub fn reset_and_answer(&self, key: &K) -> TuneAnswer {
        #[cfg_attr(not(persistence), allow(unused_mut))]
        let mut tracker = self.tracker.lock();
        #[cfg(persistence)]
        tracker.reset_if_environment_switched();
        tracker.decided(key)
    }

    /// What memory holds for `key`, reading only: it never reads the persistent cache and never
    /// drops an earlier environment's tunes, which it reports as a miss.
    pub fn decided(&self, key: &K) -> TuneAnswer {
        self.tracker.lock().decided(key)
    }

    /// Answer `key` on `client`'s device: from memory, from the store when memory holds no
    /// tune, and otherwise by tuning `tunables` on `inputs`. `checks` cross-checks the
    /// candidates, collecting a mismatch when asked to: only a tune that drives reports them, so
    /// every other answer runs them uncollected.
    ///
    /// # Assumptions
    ///
    /// - `key` is the key `tunables` generates for `inputs`.
    /// - `tunables` holds a candidate, and some candidate serves `inputs`.
    /// - The plan's candidates end as [`LocalTuner::execute`](crate::tune::LocalTuner::execute)'s
    ///   third Assumption says.
    ///
    /// # Guarantees
    ///
    /// - Never answers [`Miss`](TuneAnswer::Miss).
    /// - One tune of a key runs at a time within an environment: a concurrent call answers
    ///   [`Pending`](TuneAnswer::Pending). A switch drops the mark, so a tune may run once
    ///   more for the new environment while the old one's finishes.
    /// - A tune that unwinds leaves the key to the next call; one that stops on a lost card leaves
    ///   it to another card.
    /// - A tune picked with nothing measured is kept in memory, never in the store.
    pub fn tune<'a, F: TuneInputs, Out: AutotuneCheckable, Id: Sync>(
        &self,
        key: &K,
        inputs: &F::At<'a>,
        tunables: &TunableSet<K, F, Out, Id>,
        client: &Client,
        checks: impl FnOnce(bool) -> Option<Vec<CheckResult>>,
    ) -> TuneAnswer
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        // A key reached by candidates that only gather kernels would measure on the outputs of
        // discarded launches.
        let compiling =
            ProcessMode::current() == ProcessMode::CompileOnly || CandidateGathering::is_open();
        let opened = match self.open(key, tunables, client, compiling) {
            Ok(opened) => opened,
            Err(answer) => return check_then_answer(answer, checks),
        };

        log::info!("Tuning {key}");
        // A `CompileOnly` override queues the kernels of the top of the plan and measures
        // nothing; the key stays untuned for the pass that profiles it.
        let Some((unmarker, gathered)) = opened else {
            return check_then_answer(self.compile_plan(key, inputs, tunables, client), checks);
        };
        // Sure to measure now: a tune registered nowhere yet registers with the override open.
        let settling = (unmarker, gathered.or_else(TuneRegistration::register));
        let logged = self.logger.lock().config().autotune.tunes_logged();
        self.prepare_driver(key, inputs, tunables, client, settling, checks(logged))
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
    ) -> Result<Opened<K>, TuneAnswer> {
        // Scoped: the rest of the tune re-locks `self.tracker`, and the mutex is non-reentrant.
        let mut tracker = self.tracker.lock();
        #[cfg(persistence)]
        tracker.reset_if_environment_switched();
        if let Some(answer) = tracker.answer(key, client.service_id(), compiling) {
            return Err(answer);
        }
        // A key with one candidate is answered, not tuned: nothing to read or to store.
        if tunables.len() <= 1 {
            return Err(tracker.track_tune(key.clone(), 0));
        }
        #[cfg(persistence)]
        if let hit @ TuneAnswer::Hit { .. } = self.track_stored(&mut tracker, key, tunables) {
            return Err(hit);
        }
        // Tracked compiled once its candidates queued, by the compile pass.
        if compiling {
            return Ok(None);
        }
        let gathered = tracker.track_pending(key.clone());
        // Made last under the lock: no gap follows the mark, and the unmarker, whose drop takes
        // this lock, can never drop while it is held.
        let unmarker = PendingUnmarker::new(self.tracker.clone(), &tracker, key.clone());
        Ok(Some((unmarker, gathered)))
    }

    /// Track the store's answer for `key`, read under the lock `tracker` is guarded by and logged
    /// at the full level, and answer it.
    #[cfg(persistence)]
    fn track_stored<F: TuneInputs, Out: 'static, Id>(
        &self,
        tracker: &mut TuneTracker<K>,
        key: &K,
        tunables: &TunableSet<K, F, Out, Id>,
    ) -> TuneAnswer {
        let checksum = tunables.compute_checksum();
        let stored = tracker.track_stored(key, &checksum, tunables.len());
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
    fn prepare_driver<'t, 'a, F: TuneInputs, Out: AutotuneCheckable, Id>(
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
            self.tracker.lock().table(),
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
        #[cfg(persistence)]
        recorder.record_plan(|| plan.account(|index| tunables.function_at(index).name()));
        let bounds = tunables.bounds(key, inputs);
        let limit = bounds.as_ref().and_then(|bounds| bounds.time_limit());
        logger.log_bounds(bounds.clone(), limit);
        let options = DriveOptions {
            limit,
            track_steps: logger.as_context().is_some(),
        };

        let (unmarker, registration) = settling;
        let settler = TuneSettler::new(
            self.tracker.clone(),
            self.logger.clone(),
            registration,
            unmarker,
            #[cfg(persistence)]
            crate::tune::drive::persister::TunePersister::new(checksum, limit, bounds, recorder),
        );
        let names = tunables.autotunables().map(|tune| tune.name());
        let committer = TuneCommitter::new(key.clone(), names, logger, settler);
        TuneDriver::new(committer, key, tunables, inputs, test_inputs, plan, options)
    }

    /// Runs the top of the plan once each, on the inputs a tune generates, so their launches
    /// queue their kernels under a `CompileOnly` override, and answers
    /// [`Compiled`](TuneAnswer::Compiled): the key is left untuned.
    ///
    /// Runs [`TunePlanner::plan_top`](crate::tune::plan::planner::TunePlanner::plan_top) with the
    /// configured compilation parallelism; the tune compiles the rest at their turn. The pass
    /// holds the device as the set's access does, enters the gathering stage inside the hold, and
    /// contains a candidate that panics as the measuring pass does: it serves nothing. The key is
    /// tracked compiled on the card once the pass ran; a device that cannot be held queues
    /// nothing, and leaves the key to the next call.
    fn compile_plan<'a, F: TuneInputs, Out: AutotuneCheckable, Id: Sync>(
        &self,
        key: &K,
        inputs: &F::At<'a>,
        tunables: &TunableSet<K, F, Out, Id>,
        client: &Client,
    ) -> TuneAnswer
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        let test_inputs = tunables.generate_inputs(key, inputs);
        let mut plan = tunables.plan(key);
        let mut enterer = tunables.bound_stage_enterer(key, inputs);
        let parallelism = CubeClRuntimeConfig::get().compilation.parallelism();
        let sampler = CandidateSampler::new(client, tunables.device_access());
        let enterer = enterer.as_deref_mut();
        let held = sampler.hold(move || {
            let stage = enter_stage(enterer, TuneStage::Gathering);
            let gathering = CandidateGathering::open();
            plan.plan_top(parallelism, |index| {
                sampler
                    .contain_race(tunables.function_at(index), test_inputs.clone())
                    .is_ok()
            });
            core::mem::drop(gathering);
            core::mem::drop(stage);
        });
        match held {
            Ok(()) => self.tracker.lock().track_compiled(
                key.clone(),
                client.service_id(),
                register_ungathered,
            ),
            Err(err) => log::warn!(
                "The candidates of {key} queued nothing: the device could not be held. {err}"
            ),
        }
        TuneAnswer::Compiled
    }
}

/// `answer`, a key answered without a tune, its candidates cross-checked uncollected: nothing
/// reports them.
fn check_then_answer(
    answer: TuneAnswer,
    checks: impl FnOnce(bool) -> Option<Vec<CheckResult>>,
) -> TuneAnswer {
    checks(false);
    answer
}

/// The registration of a key's tune, unless another key's candidates reached it: the pass that
/// profiles stops a plan at its first close-enough candidate, and may never run the one that
/// reaches a key gathered inside it. Such a key is registered if it is ever measured, as an
/// ungathered one is.
fn register_ungathered() -> Option<TuneRegistration> {
    (!CandidateGathering::is_open())
        .then(TuneRegistration::register)
        .flatten()
}
