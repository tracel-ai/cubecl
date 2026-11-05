use super::{AutotuneError, AutotuneKey, AutotuneOutput, TunableSet, TuneInputs, Tuner};
#[cfg(feature = "autotune-checks")]
use crate::tune::AutotuneLoggerExt;
use crate::{client::Client, device::HardwareIdentity, tune::TuneCacheResult};
#[cfg(feature = "autotune-checks")]
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::{
    any::{Any, TypeId},
    hash::Hash,
};
use cubecl_environment::collections::HashMap;
use cubecl_environment::sync::{Mutex, RwLock};

/// The tunable sets a [`LocalTuner`] has built, keyed by device as well as by
/// initializer: a set is built from the device it will run on — its client, its
/// hardware properties — so one device's set cannot answer for another's. See
/// [`LocalTuner::init`].
type Sets<ID> = RwLock<Option<HashMap<(TypeId, ID), Arc<dyn Any + Send + Sync>>>>;

/// A local tuner allows to create a tuner for a specific key that can be different from the server
/// key.
pub struct LocalTuner<AK: AutotuneKey, ID> {
    state: Mutex<Option<Tuners<AK, ID>>>,
    name: &'static str,
    sets: Sets<ID>,
}

/// The tuners a [`LocalTuner`] opened. An id keeps a tuner per candidate list it executed, shared
/// with the ids of its hardware, so ids of the same hardware running one list tune a key once for
/// all of them.
struct Tuners<AK: AutotuneKey, ID> {
    /// An id's tuners, by the checksum of the list each runs.
    by_id: HashMap<ID, HashMap<u64, Arc<Tuner<AK>>>>,
    opened: HashMap<TunerIdentity, Arc<Tuner<AK>>>,
}

/// Which tuner a device's executes reach: one per hardware and candidate list.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TunerIdentity {
    hardware: HardwareIdentity,
    /// The low half of the candidate list's [`compute_checksum`](TunableSet::compute_checksum): each
    /// device builds its own list, and one of the same hardware may build another.
    checksum: u64,
}

impl<AK: AutotuneKey, ID: Hash + Eq + Clone> Tuners<AK, ID> {
    fn new() -> Self {
        Self {
            by_id: HashMap::new(),
            opened: HashMap::new(),
        }
    }

    /// The tuner `id` executes `operations` on, opened by the first id of its hardware to execute
    /// the same candidate list.
    fn get_or_open<I: TuneInputs, Out, Id>(
        &mut self,
        id: &ID,
        name: &str,
        client: &Client,
        operations: &TunableSet<AK, I, Out, Id>,
    ) -> Arc<Tuner<AK>> {
        let checksum = operations.checksum();
        if let Some(tuner) = self.by_id.get(id).and_then(|lists| lists.get(&checksum)) {
            return tuner.clone();
        }
        let identity = TunerIdentity {
            hardware: client.hardware_identity(),
            checksum,
        };
        let tuner = self
            .opened
            .entry(identity)
            .or_insert_with_key(|identity| {
                Arc::new(Tuner::new(&name.replace("::", "-"), &identity.hardware))
            })
            .clone();
        self.by_id
            .entry(id.clone())
            .or_default()
            .insert(checksum, tuner.clone());
        tuner
    }
}

/// Create a local tuner with the provided name.
#[macro_export]
macro_rules! local_tuner {
    ($name:expr) => {
        LocalTuner::new(concat!(module_path!(), "-", $name))
    };
    () => {
        LocalTuner::new(module_path!())
    };
}

pub use local_tuner;

impl<AK, ID> LocalTuner<AK, ID>
where
    AK: AutotuneKey + 'static,
    ID: Hash + PartialEq + Eq + Clone,
{
    /// Create a new local tuner.
    pub const fn new(name: &'static str) -> Self {
        Self {
            state: Mutex::new(None),
            name,
            sets: RwLock::new(None),
        }
    }

    /// Get or initialize the [`TunableSet`] for `id`.
    ///
    /// Returns a cached `Arc<TunableSet>` keyed by the `TypeId` of `init_set`
    /// *and* by `id`, so the initializer runs once per device rather than once
    /// per process.
    ///
    /// The device is part of the key because a set is routinely built from the
    /// device it will run on: a closure captures that device's
    /// [`Client`] to ask what it supports, or reads its hardware
    /// properties to decide which tunables are worth offering at all. Keyed by
    /// the initializer alone, whichever device tuned first would answer those
    /// questions for every device that followed — promoting kernels onto
    /// hardware that cannot run them, or withholding kernels from hardware
    /// that can.
    pub fn init<I, Out, Id, F>(&self, id: &ID, init_set: F) -> Arc<TunableSet<AK, I, Out, Id>>
    where
        F: Fn() -> TunableSet<AK, I, Out, Id> + 'static + Send + Sync,
        Id: Send + Sync + 'static,
        I: TuneInputs,
        Out: AutotuneOutput,
    {
        let key = (TypeId::of::<F>(), id.clone());
        let sets = self.sets.read();

        static DOWNCAST_ERROR: &str = "Local tuner only support one set of tunable that must work on the same input and output declared with the init function.";

        if let Some(sets) = sets.as_ref()
            && let Some(set) = sets.get(&key)
        {
            return set.clone().downcast().expect(DOWNCAST_ERROR);
        };

        core::mem::drop(sets);

        let mut sets = self.sets.write();

        if let Some(sets) = sets.as_ref()
            && let Some(set) = sets.get(&key)
        {
            return set.clone().downcast().expect(DOWNCAST_ERROR);
        };

        let content = Arc::new(init_set());

        if let Some(sets) = sets.as_mut() {
            sets.insert(key, content.clone());
        } else {
            let mut map = HashMap::<(TypeId, ID), Arc<dyn Any + Send + Sync>>::new();
            map.insert(key, content.clone());
            *sets = Some(map);
        };

        content
    }

    /// What the fastest tunable in `operations` for `key` on `id` is
    /// [identified](super::Tunable::identified) by: `None` until a result for the key is
    /// settled, tuned by a round in this process or read back from disk by an
    /// [`execute`](Self::execute).
    ///
    /// Results are shared by every `id` of the same [hardware](crate::device::HardwareIdentity)
    /// executing the same candidates, from the first [`execute`](Self::execute) of each: an `id`
    /// that has not executed `operations` sees none.
    ///
    /// It reads results and never produces one: it never starts a round, never waits on one in
    /// flight, never reads the persistent cache, and never resets the tuner's cache after an
    /// environment switch. It reports nothing settled there until the next `execute`.
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
            .state
            .lock()
            .as_ref()?
            .by_id
            .get(id)?
            .get(&operations.checksum())?
            .clone();
        let fastest_index = tuner.settled(key)?;
        Some(operations.identity(fastest_index).clone())
    }

    /// Clear the autotune state.
    pub fn clear(&self) {
        self.state.lock().take();
    }

    #[cfg(feature = "autotune-checks")]
    fn checks<'a, I: TuneInputs, Out: AutotuneOutput, Id>(
        &self,
        operations: &TunableSet<AK, I, Out, Id>,
        inputs: &<I as TuneInputs>::At<'a>,
    ) -> alloc::vec::Vec<crate::tune::log::CheckResult>
    where
        <I as TuneInputs>::At<'a>: Clone + Send,
    {
        let mut checks_outputs = Vec::new();
        for i in 0..operations.len() {
            let op = operations.fastest(i);
            let result = op.execute(inputs.clone());
            checks_outputs.push((op.name.to_string(), result));
        }
        super::check_autotune_outputs(checks_outputs)
    }

    /// Execute the fastest operation in a [`TunableSet`], triggering a tuning pass on
    /// the first call for a given key.
    pub fn execute<'a, I: TuneInputs, Out, Id>(
        &self,
        id: &ID,
        client: &Client,
        operations: Arc<TunableSet<AK, I, Out, Id>>,
        inputs: <I as TuneInputs>::At<'a>,
    ) -> Out
    where
        <I as TuneInputs>::At<'a>: Clone + Send,
        Out: AutotuneOutput,
        Id: Send + Sync,
    {
        let key = operations.generate_key(&inputs);

        let tuner = self
            .state
            .lock()
            .get_or_insert_with(Tuners::new)
            .get_or_open(id, self.name, client, &operations);

        // Fast path: a cached hit skips straight to the fastest operation.
        // `fastest` also resets the tuner cache if the environment switched, so
        // a miss here falls through to `check_tune`, which reads the store.
        if let TuneCacheResult::Hit { fastest_index } = tuner.fastest(&key) {
            // `autotune-checks` cross-checks the candidates of every launch, a hit's included.
            // Bound first, so the logger guard drops before the checks run every candidate.
            #[cfg(feature = "autotune-checks")]
            {
                let mut log_context =
                    crate::tune::AutotuneLogContext::new(&mut tuner.logger().lock());
                log_context.set_checks(|| self.checks::<I, Out, Id>(&operations, &inputs));
            }
            return operations
                .fastest(fastest_index)
                .execute(inputs)
                .expect("Should run when selected by autotune.");
        }

        // Built after the fast path: a hit logs nothing, so it takes no logger lock.
        #[allow(unused_mut)]
        let mut log_context = crate::tune::AutotuneLogContext::new(&mut tuner.logger().lock());

        #[cfg(feature = "autotune-checks")]
        log_context.set_checks(|| self.checks::<I, Out, Id>(&operations, &inputs));

        let fastest =
            tuner.check_tune::<I, Out, Id>(&key, &inputs, &operations, client, log_context);

        // Run the execution depending on the cache state.
        match fastest {
            TuneCacheResult::Hit { fastest_index } => operations
                .fastest(fastest_index)
                .execute(inputs)
                .expect("Should run when selected by autotune."),
            TuneCacheResult::Miss => {
                panic!(
                    "Somehow we STILL didn't check a tuning checksum or start tuning, something has gone wrong."
                )
            }
            // Still waiting (e.g. on wasm), stopped on a lost device, or its candidates' kernels
            // queued under a `CompileOnly` override, where an operation only queues its kernels:
            // the candidates run in the plan's order, the one the measurement follows, so the
            // best ranked candidate that launches runs rather than the first declared.
            TuneCacheResult::Pending | TuneCacheResult::Compiled => {
                match execute_in_plan_order(&operations, &key, inputs) {
                    Ok(output) => output,
                    Err(failures) => panic!(
                        "All autotune operations failed, no viable operation found.\n{failures:?}"
                    ),
                }
            }
        }
    }
}

/// Runs the candidates in the plan's [order](super::TunePlan::order) and hands back the
/// first output, or every candidate's error when none launches. A candidate that fails to
/// launch is skipped, as the measurement skips it.
fn execute_in_plan_order<'a, AK: AutotuneKey, I: TuneInputs, Out: 'static, Id>(
    operations: &TunableSet<AK, I, Out, Id>,
    key: &AK,
    inputs: <I as TuneInputs>::At<'a>,
) -> Result<Out, Vec<AutotuneError>>
where
    <I as TuneInputs>::At<'a>: Clone,
{
    let mut failures = Vec::new();
    for (index, _) in operations.plan(key).order(operations.len()) {
        match operations.fastest(index).execute(inputs.clone()) {
            Ok(output) => return Ok(output),
            Err(err) => failures.push(err),
        }
    }
    Err(failures)
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

    impl AutotuneKey for FakeAutotuneKey {}

    fn set() -> TunableSet<FakeAutotuneKey, (), usize> {
        TunableSet::new(|_: &()| FakeAutotuneKey, |_: &FakeAutotuneKey, _: &()| ())
    }

    fn answers(index: usize) -> impl Fn(()) -> Result<usize, String> {
        move |_| Ok(index)
    }

    fn fails(_: ()) -> Result<usize, String> {
        Err("does not launch here".into())
    }

    #[test]
    fn the_pending_fallback_runs_the_plan_s_first_candidate_not_the_first_declared() {
        let group = TuneGroup::<FakeAutotuneKey>::new("group", |_| 1);
        let operations = set()
            .with(Tunable::<FakeAutotuneKey, (), usize>::new(
                "declared_first",
                answers(0),
            ))
            .with(
                Tunable::<FakeAutotuneKey, (), usize>::new("planned_first", answers(1))
                    .group(&group, |_| 1),
            );

        assert_eq!(
            execute_in_plan_order(&operations, &FakeAutotuneKey, ()).ok(),
            Some(1)
        );
    }

    #[test]
    fn the_pending_fallback_moves_down_the_plan_when_a_candidate_does_not_launch() {
        let group = TuneGroup::<FakeAutotuneKey>::new("group", |_| 1);
        let operations = set()
            .with(
                Tunable::<FakeAutotuneKey, (), usize>::new("planned_last", answers(0))
                    .group(&group, |_| 0),
            )
            .with(
                Tunable::<FakeAutotuneKey, (), usize>::new("planned_first", fails)
                    .group(&group, |_| 2),
            )
            .with(
                Tunable::<FakeAutotuneKey, (), usize>::new("planned_second", answers(2))
                    .group(&group, |_| 1),
            );

        assert_eq!(
            execute_in_plan_order(&operations, &FakeAutotuneKey, ()).ok(),
            Some(2)
        );
    }

    #[test]
    fn the_pending_fallback_tries_what_the_plan_left_out_last() {
        let never = TuneGroup::<FakeAutotuneKey>::new("never", |_| -1);
        let operations = set()
            .with(Tunable::<FakeAutotuneKey, (), usize>::new(
                "declared_first",
                fails,
            ))
            .with(
                Tunable::<FakeAutotuneKey, (), usize>::new("left_out", answers(1))
                    .group(&never, |_| 1),
            );

        assert_eq!(
            execute_in_plan_order(&operations, &FakeAutotuneKey, ()).ok(),
            Some(1)
        );
        let nothing = set().with(Tunable::<FakeAutotuneKey, (), usize>::new("fails", fails));
        assert!(execute_in_plan_order(&nothing, &FakeAutotuneKey, ()).is_err());
    }
}
