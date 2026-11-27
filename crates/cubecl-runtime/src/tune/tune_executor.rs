use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::TypeId;
use core::hash::Hash;
use cubecl_environment::sync::{Mutex, RwLock};

use super::set_initializer::SetInitializer;
use super::tune_cache::TuneCacheResult;
use super::tuner_opener::TunerOpener;
use super::{
    AutotuneCheckable, AutotuneError, AutotuneKeyable, CheckResult, TunableSet, TuneInputs,
};
use crate::client::Client;

/// Executes what a [`LocalTuner`](super::LocalTuner) is asked to: builds its sets, opens its
/// tuners, and runs the candidate a key's tune elects.
pub struct TuneExecutor<AK: AutotuneKeyable, ID> {
    name: &'static str,
    tuners: Mutex<TunerOpener<AK, ID>>,
    sets: RwLock<SetInitializer<ID>>,
}

impl<AK: AutotuneKeyable, ID: Hash + Eq + Clone> TuneExecutor<AK, ID> {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            tuners: Mutex::new(TunerOpener::new()),
            sets: RwLock::new(SetInitializer::new()),
        }
    }

    /// The set `init_set` builds for `id`, built once per initializer type and id until
    /// [`clear`](Self::clear).
    pub fn init<I, Out, Id, F>(&self, id: &ID, init_set: F) -> Arc<TunableSet<AK, I, Out, Id>>
    where
        F: Fn() -> TunableSet<AK, I, Out, Id> + 'static + Send + Sync,
        Id: Send + Sync + 'static,
        I: TuneInputs,
        Out: AutotuneCheckable,
    {
        let key = (TypeId::of::<F>(), id.clone());
        if let Some(set) = self.sets.read().cached(&key) {
            return set;
        }
        self.sets.write().initialize(key, init_set)
    }

    /// What the candidate settled for `key` of `operations` on `id` is identified by.
    pub fn fastest_identity<I, Out, Id>(
        &self,
        id: &ID,
        operations: &TunableSet<AK, I, Out, Id>,
        key: &AK,
    ) -> Option<Id>
    where
        I: TuneInputs,
        Id: Clone,
    {
        let tuner = self
            .tuners
            .lock()
            .opened(id, operations.checksum())?
            .clone();
        let fastest_index = tuner.settled(key)?;
        Some(operations.identity_at(fastest_index).clone())
    }

    /// Forget every set built and every tuner opened.
    pub fn clear(&self) {
        // Dropped after unlocking: a captured value's drop may call back into this tuner.
        let closed = {
            let mut sets = self.sets.write();
            (self.tuners.lock().close(), sets.close())
        };
        core::mem::drop(closed);
    }

    /// Run the candidate `operations` elects for the key of `inputs` on `id`, tuning the key on
    /// its first miss.
    pub fn execute<'a, I: TuneInputs, Out, Id>(
        &self,
        id: &ID,
        client: &Client,
        operations: Arc<TunableSet<AK, I, Out, Id>>,
        inputs: <I as TuneInputs>::At<'a>,
    ) -> Out
    where
        <I as TuneInputs>::At<'a>: Clone + Send,
        Out: AutotuneCheckable,
        Id: Send + Sync,
    {
        let key = operations.generate_key(&inputs);
        let tuner = self.tuners.lock().open(id, self.name, client, &operations);

        // Fast path: a cached hit skips straight to the fastest operation. `fastest` also resets
        // the tuner cache if the environment switched, so a miss here falls through to `tune`,
        // which reads the store.
        if let TuneCacheResult::Hit { fastest_index } = tuner.fastest(&key) {
            // `autotune-checks` cross-checks the candidates of every launch, a hit's included:
            // nothing reports a hit's checks, so a mismatch there panics.
            check_candidates(&operations, &inputs, false);
            return execute_elected(&operations, &key, fastest_index, inputs);
        }

        let checks = check_candidates(&operations, &inputs, decisions_enabled());
        match tuner.tune::<I, Out, Id>(&key, &inputs, &operations, client, checks) {
            TuneCacheResult::Hit { fastest_index } => {
                execute_elected(&operations, &key, fastest_index, inputs)
            }
            TuneCacheResult::Miss => unreachable!("a tune should never answer a miss"),
            // Still waiting (e.g. on wasm), stopped on a lost device, or its candidates' kernels
            // queued under a `CompileOnly` override, where an operation only queues its kernels:
            // the candidates run in the plan's order, the one the measurement follows, so the
            // best ranked candidate that launches runs rather than the first declared.
            TuneCacheResult::Pending | TuneCacheResult::Compiled => {
                run_in_plan_order(&operations, &key, inputs)
            }
        }
    }
}

/// Whether the cross-checks of a tune are collected, rather than fatal: they are reported only
/// where decisions are written somewhere.
fn decisions_enabled() -> bool {
    use crate::config::RuntimeConfig;

    crate::config::CubeClRuntimeConfig::get()
        .autotune
        .decisions_enabled()
}

/// Under the `autotune-checks` feature, the cross-check of every candidate of `operations` on
/// `inputs`, collected where `collect`; `None` without the feature.
#[cfg_attr(
    not(feature = "autotune-checks"),
    allow(unused_variables, clippy::unnecessary_wraps)
)]
fn check_candidates<'a, AK: AutotuneKeyable, I: TuneInputs, Out: AutotuneCheckable, Id>(
    operations: &TunableSet<AK, I, Out, Id>,
    inputs: &<I as TuneInputs>::At<'a>,
    collect: bool,
) -> Option<Vec<CheckResult>>
where
    <I as TuneInputs>::At<'a>: Clone,
{
    #[cfg(feature = "autotune-checks")]
    {
        use alloc::string::ToString;

        let outputs = operations
            .autotunables()
            .map(|operation| (operation.name().to_string(), operation.race(inputs.clone())))
            .collect();
        Some(super::checks::check_outputs(outputs, collect))
    }
    #[cfg(not(feature = "autotune-checks"))]
    {
        None
    }
}

/// Runs the candidates of `key` in its run order, the first that launches answering.
///
/// # Panics
///
/// When no candidate launches: some candidate serves the problem.
fn run_in_plan_order<'a, AK: AutotuneKeyable, I: TuneInputs, Out: 'static, Id>(
    operations: &TunableSet<AK, I, Out, Id>,
    key: &AK,
    inputs: <I as TuneInputs>::At<'a>,
) -> Out
where
    <I as TuneInputs>::At<'a>: Clone,
{
    execute_in_plan_order(operations, &operations.run_order(key), inputs).unwrap_or_else(
        |failures| panic!("No autotune candidate for {key} launches.\n{failures:?}"),
    )
}

/// Runs the candidates in `order`, the key's run order, and hands back the first output, or every
/// candidate's error when none launches. A candidate that fails to launch is skipped, as the
/// measurement skips it.
fn execute_in_plan_order<'a, AK: AutotuneKeyable, I: TuneInputs, Out: 'static, Id>(
    operations: &TunableSet<AK, I, Out, Id>,
    order: &[usize],
    inputs: <I as TuneInputs>::At<'a>,
) -> Result<Out, Vec<AutotuneError>>
where
    <I as TuneInputs>::At<'a>: Clone,
{
    first_served(order.iter().copied(), |index| {
        // Run as the work, but before the tune decided: a candidate whose kernel fails to compile
        // is the tune's to drop, not the work's failure.
        crate::execution::run_provisionally(|| {
            operations.function_at(index).execute(inputs.clone())
        })
    })
}

/// The first output `run` gives over `order`, or every candidate's error when none gives one.
fn first_served<Out>(
    order: impl Iterator<Item = usize>,
    run: impl Fn(usize) -> Result<Out, AutotuneError>,
) -> Result<Out, Vec<AutotuneError>> {
    let mut failures = Vec::new();
    for index in order {
        match run(index) {
            Ok(output) => return Ok(output),
            Err(err) => failures.push(err),
        }
    }
    Err(failures)
}

/// Runs `elected`, the candidate the tune elected for `key`, or the next one in the key's run
/// order that serves `inputs` when it rejects them: a key groups calls, and the elected candidate
/// may serve only some of them.
fn execute_elected<'a, AK: AutotuneKeyable, I: TuneInputs, Out: 'static, Id>(
    operations: &TunableSet<AK, I, Out, Id>,
    key: &AK,
    elected: usize,
    inputs: <I as TuneInputs>::At<'a>,
) -> Out {
    let spare = I::spare(&inputs);
    let rejection = match operations.function_at(elected).execute(inputs) {
        Ok(output) => return output,
        Err(rejection) => rejection,
    };
    log::debug!("The candidate elected for {key} rejected a call of it: {rejection}");

    let spare = spare.unwrap_or_else(|| {
        panic!("The candidate elected for {key} should serve every call of it.\n{rejection}")
    });
    let order = operations.run_order(key);
    let others = order.iter().copied().filter(|&index| index != elected);
    first_served(others, |index| {
        operations.function_at(index).execute(spare.clone())
    })
    .unwrap_or_else(|failures| {
        panic!("No autotune candidate for {key} serves this call.\n{rejection}\n{failures:?}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::{Tunable, TuneGroup};
    use alloc::string::String;
    use core::fmt::Display;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize, Debug)]
    struct FakeAutotuneKey;

    impl Display for FakeAutotuneKey {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("FakeAutotuneKey")
        }
    }

    impl AutotuneKeyable for FakeAutotuneKey {}

    fn set() -> TunableSet<FakeAutotuneKey, (), usize> {
        TunableSet::new(|_: &()| FakeAutotuneKey, |_: &FakeAutotuneKey, _: &()| ())
    }

    fn answers(name: &str, index: usize) -> Tunable<FakeAutotuneKey, (), usize> {
        Tunable::new(name, move |_: ()| Ok::<_, String>(index))
    }

    fn fails(name: &str) -> Tunable<FakeAutotuneKey, (), usize> {
        Tunable::new(name, |_: ()| Err::<usize, _>("does not launch here"))
    }

    #[test]
    fn the_pending_fallback_runs_the_plan_s_first_candidate_not_the_first_declared() {
        let group = TuneGroup::<FakeAutotuneKey>::new("group", |_| 1);
        let operations = set()
            .with(answers("declared_first", 0))
            .with(answers("planned_first", 1).group(&group, |_| 1));

        assert_eq!(
            execute_in_plan_order(&operations, &operations.run_order(&FakeAutotuneKey), ()).ok(),
            Some(1)
        );
    }

    #[test]
    fn the_pending_fallback_moves_down_the_plan_when_a_candidate_does_not_launch() {
        let group = TuneGroup::<FakeAutotuneKey>::new("group", |_| 1);
        let operations = set()
            .with(answers("planned_last", 0).group(&group, |_| 0))
            .with(fails("planned_first").group(&group, |_| 2))
            .with(answers("planned_second", 2).group(&group, |_| 1));

        assert_eq!(
            execute_in_plan_order(&operations, &operations.run_order(&FakeAutotuneKey), ()).ok(),
            Some(2)
        );
    }

    #[test]
    fn the_pending_fallback_tries_what_the_plan_left_out_last() {
        let never = TuneGroup::<FakeAutotuneKey>::new("never", |_| -1);
        let operations = set()
            .with(fails("declared_first"))
            .with(answers("left_out", 1).group(&never, |_| 1));

        assert_eq!(
            execute_in_plan_order(&operations, &operations.run_order(&FakeAutotuneKey), ()).ok(),
            Some(1)
        );
    }

    #[test]
    fn a_pending_fallback_nothing_launches_hands_back_every_failure() {
        let nothing = set().with(fails("fails"));

        let failures = execute_in_plan_order(&nothing, &nothing.run_order(&FakeAutotuneKey), ());

        assert!(failures.is_err_and(|failures| failures.len() == 1));
    }

    #[test]
    fn a_rejected_call_is_served_by_the_next_candidate_in_the_plan() {
        let group = TuneGroup::<FakeAutotuneKey>::new("group", |_| 1);
        let operations = set()
            .with(answers("planned_last", 0).group(&group, |_| 0))
            .with(fails("elected").group(&group, |_| 2))
            .with(answers("planned_second", 2).group(&group, |_| 1));

        assert_eq!(execute_elected(&operations, &FakeAutotuneKey, 1, ()), 2);
    }

    #[test]
    fn an_elected_candidate_that_serves_the_call_runs_alone() {
        let operations = set().with(answers("other", 0)).with(answers("elected", 1));

        assert_eq!(execute_elected(&operations, &FakeAutotuneKey, 1, ()), 1);
    }

    #[test]
    #[should_panic(expected = "serves this call")]
    fn a_call_no_candidate_serves_panics() {
        let operations = set().with(fails("elected")).with(fails("other"));

        execute_elected(&operations, &FakeAutotuneKey, 0, ());
    }

    #[test]
    #[should_panic(expected = "should serve every call")]
    fn a_rejected_call_with_no_spare_panics() {
        /// Inputs that implement the trait by hand, and so have no spare.
        struct Borrowed;

        impl TuneInputs for Borrowed {
            type At<'a> = ();
        }

        let operations =
            TunableSet::<FakeAutotuneKey, Borrowed, usize>::new_cloning_inputs(|_: &()| {
                FakeAutotuneKey
            })
            .with(Tunable::<FakeAutotuneKey, Borrowed, usize>::new(
                "elected",
                |_| Err::<usize, _>("does not launch here"),
            ))
            .with(Tunable::<FakeAutotuneKey, Borrowed, usize>::new(
                "other",
                |_| Ok::<_, String>(1),
            ));

        execute_elected(&operations, &FakeAutotuneKey, 0, ());
    }

    #[test]
    fn an_id_that_executed_nothing_has_no_fastest_identity() {
        let executor = TuneExecutor::<FakeAutotuneKey, u8>::new("test");

        assert_eq!(
            executor.fastest_identity(&0, &set(), &FakeAutotuneKey),
            None
        );
    }

    #[test]
    fn an_initializer_runs_once_per_id() {
        use core::sync::atomic::{AtomicUsize, Ordering};

        static BUILT: AtomicUsize = AtomicUsize::new(0);
        let build = || {
            BUILT.fetch_add(1, Ordering::Relaxed);
            TunableSet::<FakeAutotuneKey, (), ()>::new_cloning_inputs(|_: &()| FakeAutotuneKey)
        };
        let executor = TuneExecutor::<FakeAutotuneKey, u8>::new("test");

        let first = executor.init(&0, build);
        let again = executor.init(&0, build);
        executor.init(&1, build);

        assert!(Arc::ptr_eq(&first, &again));
        assert_eq!(BUILT.load(Ordering::Relaxed), 2);
    }
}
