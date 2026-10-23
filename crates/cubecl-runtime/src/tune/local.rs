use super::append_only::AppendOnlyMap;
use super::{AutotuneKey, AutotuneOutput, TunableSet, TuneInputs, Tuner};
#[cfg(feature = "autotune-checks")]
use crate::tune::AutotuneLoggerExt;
use crate::{client::Client, device::HardwareIdentity, tune::TuneCacheResult};
#[cfg(feature = "autotune-checks")]
use alloc::string::ToString;
use alloc::{sync::Arc, vec::Vec};
use core::any::{Any, TypeId};
use crossbeam_utils::CachePadded;
use cubecl_environment::{collections::HashMap, environment, sync::Mutex};

/// A local tuner allows to create a tuner for a specific key that can be different from the server
/// key.
///
/// A hit touches only its own device's state, so devices driven from their own threads never wait
/// on one another.
pub struct LocalTuner<AK: AutotuneKey, ID> {
    devices: AppendOnlyMap<ID, DeviceTuning<AK>>,
    opened: Mutex<Option<HashMap<TunerIdentity, Arc<Tuner<AK>>>>>,
    name: &'static str,
}

/// Which tuner a device's executes reach: one per hardware and candidate list.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TunerIdentity {
    hardware: HardwareIdentity,
    /// The low half of the candidate list's [`compute_checksum`](TunableSet::compute_checksum): each
    /// device builds its own list, and one of the same hardware may build another.
    checksum: u64,
}

/// What a [`LocalTuner`] holds for one device id.
struct DeviceTuning<AK: AutotuneKey> {
    sets: AppendOnlyMap<TypeId, Arc<dyn Any + Send + Sync>>,
    /// By the low half of each list's checksum: a pick is an index into the list it was tuned for.
    lists: AppendOnlyMap<u64, ListTuning<AK>>,
}

/// What a device holds for one candidate list.
struct ListTuning<AK: AutotuneKey> {
    /// Padded: every launch on this device writes the lock, and the allocation beside it may be
    /// another device's.
    picks: CachePadded<Mutex<Picks<AK>>>,
}

/// The picks a device read from its tuner, served without the tuner's lock, which every device of
/// the hardware shares.
struct Picks<AK: AutotuneKey> {
    /// `None` until the list's first execute, and again after a [`clear`](LocalTuner::clear).
    tuner: Option<Arc<Tuner<AK>>>,
    by_key: HashMap<AK, usize>,
    /// The environment [generation](environment::generation) `by_key` was read under.
    generation: u32,
}

impl<AK: AutotuneKey> DeviceTuning<AK> {
    fn new() -> Self {
        Self {
            sets: AppendOnlyMap::new(),
            lists: AppendOnlyMap::new(),
        }
    }

    fn list(&self, checksum: u64) -> &ListTuning<AK> {
        self.lists.get_or_insert_with(&checksum, ListTuning::new)
    }

    fn clear(&self) {
        for (_, list) in self.lists.iter() {
            list.clear();
        }
    }
}

impl<AK: AutotuneKey> ListTuning<AK> {
    fn new() -> Self {
        Self {
            picks: CachePadded::new(Mutex::new(Picks {
                tuner: None,
                by_key: HashMap::new(),
                generation: environment::generation(),
            })),
        }
    }

    fn pick(&self, key: &AK) -> Option<usize> {
        let mut picks = self.picks.lock();
        let generation = environment::generation();
        if picks.generation != generation {
            picks.generation = generation;
            picks.by_key.clear();
        }
        picks.by_key.get(key).copied()
    }

    /// Keep the pick `tuner` answered for `key` under `generation`, unless a switch or a
    /// [`clear`](LocalTuner::clear) has since made it another environment's or another tuner's.
    fn keep_pick(&self, tuner: &Arc<Tuner<AK>>, generation: u32, key: AK, fastest_index: usize) {
        let mut picks = self.picks.lock();
        let same_tuner = picks
            .tuner
            .as_ref()
            .is_some_and(|kept| Arc::ptr_eq(kept, tuner));
        if same_tuner && picks.generation == generation && environment::generation() == generation {
            picks.by_key.insert(key, fastest_index);
        }
    }

    /// The tuner this list executes on, opened by `open` on its first execute.
    fn tuner(&self, open: impl FnOnce() -> Arc<Tuner<AK>>) -> Arc<Tuner<AK>> {
        self.picks.lock().tuner.get_or_insert_with(open).clone()
    }

    fn clear(&self) {
        let mut picks = self.picks.lock();
        picks.tuner = None;
        picks.by_key.clear();
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
    ID: PartialEq + Eq + Clone,
{
    /// Create a new local tuner.
    pub const fn new(name: &'static str) -> Self {
        Self {
            devices: AppendOnlyMap::new(),
            opened: Mutex::new(None),
            name,
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
        static DOWNCAST_ERROR: &str = "Local tuner only support one set of tunable that must work on the same input and output declared with the init function.";

        let set: Arc<dyn Any + Send + Sync> = self
            .device(id)
            .sets
            .get_or_insert_with(&TypeId::of::<F>(), || Arc::new(init_set()))
            .clone();
        set.downcast().expect(DOWNCAST_ERROR)
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
            .devices
            .get(id)?
            .lists
            .get(&operations.checksum())?
            .picks
            .lock()
            .tuner
            .clone()?;
        let fastest_index = tuner.settled(key)?;
        Some(operations.identity(fastest_index).clone())
    }

    /// Clear the autotune state.
    pub fn clear(&self) {
        self.opened.lock().take();
        for (_, device) in self.devices.iter() {
            device.clear();
        }
    }

    fn device(&self, id: &ID) -> &DeviceTuning<AK> {
        self.devices.get_or_insert_with(id, DeviceTuning::new)
    }

    /// The tuner of `client`'s hardware for `operations`, opened by the first device of that
    /// hardware to execute the same candidate list.
    fn open<I: TuneInputs, Out, Id>(
        &self,
        client: &Client,
        operations: &TunableSet<AK, I, Out, Id>,
    ) -> Arc<Tuner<AK>> {
        let identity = TunerIdentity {
            hardware: client.hardware_identity(),
            checksum: operations.checksum(),
        };
        self.opened
            .lock()
            .get_or_insert_with(HashMap::new)
            .entry(identity)
            .or_insert_with_key(|identity| {
                Arc::new(Tuner::new(
                    &self.name.replace("::", "-"),
                    &identity.hardware,
                ))
            })
            .clone()
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
        let list = self.device(id).list(operations.checksum());

        if let Some(fastest_index) = list.pick(&key) {
            // `autotune-checks` cross-checks the candidates of every launch, a hit's included.
            // Bound first, so the logger guard drops before the checks run every candidate.
            #[cfg(feature = "autotune-checks")]
            {
                let tuner = list.tuner(|| self.open(client, &operations));
                let mut log_context =
                    crate::tune::AutotuneLogContext::new(&mut tuner.logger().lock());
                log_context.set_checks(|| self.checks::<I, Out, Id>(&operations, &inputs));
            }
            return operations
                .fastest(fastest_index)
                .execute(inputs)
                .expect("Should run when selected by autotune.");
        }

        let tuner = list.tuner(|| self.open(client, &operations));
        let generation = environment::generation();

        #[allow(unused_mut)]
        let mut log_context = crate::tune::AutotuneLogContext::new(&mut tuner.logger().lock());

        #[cfg(feature = "autotune-checks")]
        log_context.set_checks(|| self.checks::<I, Out, Id>(&operations, &inputs));

        let fastest =
            tuner.check_tune::<I, Out, Id>(&key, &inputs, &operations, client, log_context);

        // Run the execution depending on the cache state.
        match fastest {
            TuneCacheResult::Hit { fastest_index } => {
                list.keep_pick(&tuner, generation, key, fastest_index);
                operations
                    .fastest(fastest_index)
                    .execute(inputs)
                    .expect("Should run when selected by autotune.")
            }
            TuneCacheResult::Miss => {
                panic!(
                    "Somehow we STILL didn't check a tuning checksum or start tuning, something has gone wrong."
                )
            }
            // Still waiting (e.g. on wasm), stopped on a lost device, or its candidates' kernels
            // queued under a `CompileOnly` override: run the first operation that serves the
            // problem, which there only queues its kernels.
            TuneCacheResult::Pending | TuneCacheResult::Compiled => {
                let mut failures = Vec::new();
                for i in 0..operations.len() {
                    match operations.fastest(i).execute(inputs.clone()) {
                        Ok(output) => return output,
                        Err(err) => failures.push(err),
                    }
                }
                panic!("All autotune operations failed, no viable operation found.\n{failures:?}");
            }
        }
    }
}
