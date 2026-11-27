use crate::client::Client;
use crate::tune::execute::tune_executor::TuneExecutor;
use crate::tune::{AutotuneKey, AutotuneOutput, TunableSet, TuneInputs};
use alloc::sync::Arc;
use core::hash::Hash;

/// The autotune of one call site: the [`TunableSet`] each device builds there, and the candidate
/// a tune of each key decides, run on every call of the key.
///
/// # Assumptions
///
/// - An `ID` names one device.
///
/// # Guarantees
///
/// - An answer is shared by every id of the same [hardware](crate::device::HardwareIdentity)
///   executing the same candidates, from the first [`execute`](Self::execute) of each.
///
/// # Examples
///
/// ```ignore
/// static TUNER: LocalTuner<String, DeviceId> = local_tuner!();
///
/// let tunables = TUNER.init(&device, || {
///     TunableSet::new_cloning_inputs(|inputs: &Inputs| inputs.key())
///         .with(Tunable::new("k1", kernel_1))
///         .with(Tunable::new("k2", kernel_2))
/// });
/// let output = TUNER.execute(&device, &client, tunables, inputs);
/// ```
pub struct LocalTuner<AK: AutotuneKey, ID> {
    executor: TuneExecutor<AK, ID>,
}

/// A [`LocalTuner`] named after the module that declares it, and `$name` when given: `$name`
/// holds no `/`, as [`LocalTuner::new`] assumes.
#[macro_export]
macro_rules! local_tuner {
    ($name:expr) => {
        LocalTuner::new(concat!(module_path!(), "-", $name))
    };
    () => {
        LocalTuner::new(module_path!())
    };
}

impl<AK, ID> LocalTuner<AK, ID>
where
    AK: AutotuneKey,
    ID: Hash + PartialEq + Eq + Clone,
{
    /// A tuner whose answers are stored in the table `name` names, for each hardware.
    ///
    /// # Assumptions
    ///
    /// - `name` holds no `/`: the table is named `hardware/name`, and a `/` would store it under
    ///   another namespace than readers of the table look in.
    pub const fn new(name: &'static str) -> Self {
        Self {
            executor: TuneExecutor::new(name),
        }
    }

    /// The [`TunableSet`] `init_set` builds for `id`.
    ///
    /// # Assumptions
    ///
    /// - `init_set` answers from the device alone, and never calls back into this tuner: it runs
    ///   under the lock of the sets.
    ///
    /// # Guarantees
    ///
    /// - `init_set` runs once per initializer type and `id`, until a [`clear`](Self::clear), and
    ///   every later call answers the set it built.
    pub fn init<I, Out, Id, F>(&self, id: &ID, init_set: F) -> Arc<TunableSet<AK, I, Out, Id>>
    where
        F: Fn() -> TunableSet<AK, I, Out, Id> + 'static + Send + Sync,
        Id: Send + Sync + 'static,
        I: TuneInputs,
        Out: AutotuneOutput,
    {
        self.executor.init(id, init_set)
    }

    /// What the candidate decided for `key` of `operations` on `id` is
    /// [identified](crate::tune::Tunable::identified) by.
    ///
    /// # Guarantees
    ///
    /// - `None` until the key is settled, tuned in this process or read back from the persistent
    ///   cache by an [`execute`](Self::execute).
    /// - It never starts a round, never waits for one in flight to measure, never reads the
    ///   persistent cache, and never resets the tuner's cache after an environment switch: it
    ///   answers `None` there until the next `execute`. It may wait for this tuner's locks while
    ///   an `execute` opens a tuner, which may purge the tables of an earlier layout, or while a
    ///   tune reads or writes the store.
    /// - An `id` that has not executed `operations` sees none.
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
        self.executor.fastest_identity(id, operations, key)
    }

    /// Forget every tune and every set [`init`](Self::init) built.
    ///
    /// # Guarantees
    ///
    /// - The next [`init`](Self::init) builds its set again, which plans every key afresh: a set
    ///   keeps the run orders it planned.
    pub fn clear(&self) {
        self.executor.clear();
    }

    /// Execute the candidate `operations` decides for the key of `inputs`, tuning the key on the
    /// first call that misses it.
    ///
    /// # Assumptions
    ///
    /// - `client` is the client of the device `id` names, every call: the tuner of an id is opened
    ///   on the hardware of the first client it is called with.
    /// - `operations` holds a candidate, and some candidate serves `inputs`.
    /// - Some candidate the plan schedules for the key, at a non-negative priority, ends measured,
    ///   or with only its profile failed ([`Unknown`](crate::tune::AutotuneError::Unknown)): a
    ///   device that cannot be held ([`HoldFailed`](crate::tune::AutotuneError::HoldFailed)) or a
    ///   profile that cannot be opened ([`Unopened`](crate::tune::AutotuneError::Unopened)) for
    ///   every candidate breaks it. On
    ///   wasm, that candidate is in the first batch of the plan that queues a sample: the
    ///   fixed-count pass there resolves its samples on the browser event loop, and commits that
    ///   batch without reaching a later one.
    /// - The decided candidate serves `inputs`, or `inputs` have a [spare](TuneInputs::spare).
    /// - Under `autotune-checks`, every candidate that serves `inputs` matches the others' output.
    ///
    /// # Guarantees
    ///
    /// - A call the decided candidate rejects runs the next one that serves it.
    /// - A tune that stopped on a lost card decides nothing: the call runs the candidates in plan
    ///   order.
    ///
    /// # Panics
    ///
    /// When no candidate serves `inputs`; when a tune measured none and none it scheduled ended
    /// [`Unknown`](crate::tune::AutotuneError::Unknown); when the decided candidate rejects `inputs` and
    /// they have no spare; and under `autotune-checks`, when a candidate's output does not match
    /// and nothing reports it: the assumptions above.
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
        self.executor.execute(id, client, operations, inputs)
    }
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

    #[test]
    fn a_cleared_tuner_plans_its_keys_afresh() {
        use core::sync::atomic::{AtomicBool, Ordering};

        let second_leads = Arc::new(AtomicBool::new(false));
        let read = second_leads.clone();
        let build = move || {
            let read = read.clone();
            let group = TuneGroup::<FakeAutotuneKey>::new("group", |_| 1);
            TunableSet::new(|_: &()| FakeAutotuneKey, |_: &FakeAutotuneKey, _: &()| ())
                .with(
                    Tunable::<FakeAutotuneKey, (), ()>::new("first", |_| Ok::<_, String>(()))
                        .group(&group, |_| 1),
                )
                .with(
                    Tunable::<FakeAutotuneKey, (), ()>::new("second", |_| Ok::<_, String>(()))
                        .group(
                            &group,
                            move |_| if read.load(Ordering::Relaxed) { 2 } else { 0 },
                        ),
                )
        };
        let tuner = LocalTuner::<FakeAutotuneKey, u8>::new("test");
        let before = tuner
            .init(&0, build.clone())
            .order_candidates(&FakeAutotuneKey);

        second_leads.store(true, Ordering::Relaxed);
        tuner.clear();

        let after = tuner.init(&0, build).order_candidates(&FakeAutotuneKey);
        assert_eq!((&*before, &*after), (&[0, 1][..], &[1, 0][..]));
    }
}
