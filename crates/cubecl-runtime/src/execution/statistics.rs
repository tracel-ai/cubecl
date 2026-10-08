//! What the overrides opened with a [`StatisticsCollector`] triggered: the
//! kernels obtained and the autotune keys tuned, counted to it by the code
//! doing the work.

use core::marker::PhantomData;
use cubecl_environment::sync::{Arc, AtomicUsize, Ordering};

/// A snapshot of what a collector's overrides triggered. Every count only
/// grows, over every override and on every device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExecutionStatistics {
    /// The kernels obtained.
    pub compilation: CompilationStatistics,
    /// The autotune keys tuned.
    pub autotune: AutotuneStatistics,
}

/// The kernels the launches under a collector's overrides obtained.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompilationStatistics {
    /// Queued, or asked for by a launch that found them neither loaded nor
    /// queued. A kernel tried again after failing registers again.
    pub registered: usize,
    /// Compiled by the backend's compiler.
    pub compiled: usize,
    /// Read from the compilation store instead.
    pub loaded: usize,
    /// Failed to compile.
    pub failed: usize,
    /// Dropped before it settled: the batch compiling it panicked, or the
    /// server holding it went away.
    pub abandoned: usize,
    /// Compiled or read from the store, then refused by the device when it
    /// was loaded: counted among [`compiled`](Self::compiled) or
    /// [`loaded`](Self::loaded) too, since those are counted as the batch
    /// obtains them, before anything is loaded.
    pub refused: usize,
    /// Kept in the compilation store once loaded: what the environment grew
    /// by. Read from the store under its own key, or refused, a kernel is not
    /// stored again; one taken from the store by its source is, under its
    /// own key.
    pub stored: usize,
}

impl CompilationStatistics {
    /// The kernels done, however they ended: never more than
    /// [`registered`](Self::registered).
    pub fn settled(&self) -> usize {
        self.compiled + self.loaded + self.failed + self.abandoned
    }
}

/// The autotune keys tuned under a collector's overrides. A key with one
/// candidate is answered, not tuned, and counts nowhere.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AutotuneStatistics {
    /// Gathered by a [`CompileOnly`](super::ProcessMode::CompileOnly)
    /// pass, or reached ungathered by a tune that measures. A key reached
    /// inside another key's candidates is not gathered: it registers if it is
    /// ever measured.
    pub registered: usize,
    /// Tuned by measuring their candidates.
    pub measured: usize,
    /// Every candidate failed: the pick was made unmeasured.
    pub failed: usize,
    /// Dropped before it settled: a candidate panicked, the device was lost
    /// while it measured, or an environment switch dropped the key it was
    /// gathered for.
    pub abandoned: usize,
    /// Measured, and kept in the persistent autotune cache: what the
    /// environment grew by. An unmeasured pick is never kept.
    pub persisted: usize,
}

impl AutotuneStatistics {
    /// The keys done, however they ended: never more than
    /// [`registered`](Self::registered).
    pub fn settled(&self) -> usize {
        self.measured + self.failed + self.abandoned
    }
}

/// Collects what the [overrides](super::ProcessModeOverride) opened with it
/// triggered.
///
/// Not `Clone`: whoever holds it opens overrides that count into it. A
/// reader on another thread holds a [`StatisticsReader`].
#[derive(Debug, Default)]
pub struct StatisticsCollector {
    tallies: Arc<CollectorTallies>,
}

/// Reads a [`StatisticsCollector`]'s statistics from any thread, and opens
/// nothing.
#[derive(Debug, Clone)]
pub struct StatisticsReader {
    tallies: Arc<CollectorTallies>,
}

/// Where the work under an override is tallied: what a [`Registration`]
/// holds, and the open override hands out. A clone tallies into the same
/// collector, which is how a kernel queued under one override is counted to
/// it when its batch compiles.
#[derive(Debug, Clone)]
pub(crate) struct StatisticsRecorder {
    tallies: Arc<CollectorTallies>,
}

impl StatisticsCollector {
    /// A collector nothing has counted into yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// What reads it from another thread.
    pub fn reader(&self) -> StatisticsReader {
        StatisticsReader {
            tallies: self.tallies.clone(),
        }
    }

    /// What its overrides triggered so far.
    pub fn statistics(&self) -> ExecutionStatistics {
        self.tallies.statistics()
    }

    /// Where an override opened with it tallies.
    pub(crate) fn recorder(&self) -> StatisticsRecorder {
        StatisticsRecorder {
            tallies: self.tallies.clone(),
        }
    }
}

impl StatisticsReader {
    /// What the collector's overrides triggered so far.
    pub fn statistics(&self) -> ExecutionStatistics {
        self.tallies.statistics()
    }
}

impl StatisticsRecorder {
    /// Whether it tallies into `collector`.
    pub(crate) fn tallies_into(&self, collector: &StatisticsCollector) -> bool {
        Arc::ptr_eq(&self.tallies, &collector.tallies)
    }
}

/// How a registered kernel ended.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelOutcome {
    /// Compiled by the backend's compiler.
    Compiled,
    /// Read from the compilation store.
    Loaded,
    /// Failed to compile.
    Failed,
    /// Dropped before it settled.
    Abandoned,
}

/// What the server did with a settled kernel: the device refused it, or the
/// compilation store kept it.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelHandling {
    /// The device refused it.
    Refused,
    /// The compilation store kept it.
    Stored,
}

/// How a registered autotune key ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TuneOutcome {
    /// Tuned by measuring its candidates.
    Measured,
    /// Every candidate failed: the pick was made unmeasured.
    Failed,
    /// Dropped before it settled.
    Abandoned,
}

/// What became of a settled key's pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TunePick {
    /// The persistent autotune cache kept it.
    Persisted,
}

/// One kernel registered with the collector of the override open when it
/// was.
#[doc(hidden)]
pub type KernelRegistration = Registration<KernelOutcome>;
/// A registered kernel once its outcome is counted.
#[doc(hidden)]
pub type SettledKernel = Settled<KernelOutcome>;
/// One autotune key registered with the collector of the override open when
/// it was.
pub(crate) type TuneRegistration = Registration<TuneOutcome>;

/// One item of work registered with the collector of the override open when
/// it was, which it is counted to once it [settles](Self::settle) —
/// wherever and whenever that is, whether or not that override is still
/// open. Dropped before it settled — its batch or its tune panicked, or the
/// key it holds was dropped — it settles as abandoned.
///
/// What holds it decides when that is: a kernel queued and never loaded, or
/// a key gathered and never measured again, stays registered until a later
/// pass reaches it, its environment switches or its server goes away.
///
/// Public for the kernel loader, which lives in `cubecl-server`; counting is
/// not for callers, which read their own [`StatisticsCollector`].
#[doc(hidden)]
#[must_use]
pub struct Registration<O: Outcome> {
    /// Taken as it settles, so a drop knows whether it did.
    recorder: Option<StatisticsRecorder>,
    outcome: PhantomData<O>,
}

/// A [`Registration`] whose outcome is counted: what alone can count what
/// happens to the work afterwards, each thing once.
#[doc(hidden)]
#[derive(Debug)]
pub struct Settled<O: Outcome> {
    recorder: StatisticsRecorder,
    /// The afterwards already counted, one bit each.
    recorded: u8,
    outcome: PhantomData<O>,
}

impl<O: Outcome> Registration<O> {
    /// Register one item with the collector of the override open where the
    /// launch running now was issued — or, outside a launch, of the one open
    /// now — if one is.
    pub fn register() -> Option<Self> {
        let recorder = super::IssuedRecorder::running()
            .unwrap_or_else(super::ProcessModeOverride::active_recorder)?;
        O::tally(&recorder.tallies).register();
        Some(Self {
            recorder: Some(recorder),
            outcome: PhantomData,
        })
    }

    /// Count how it ended.
    pub fn settle(mut self, outcome: O) -> Settled<O> {
        let recorder = self.recorder.take().expect("settled once, by value");
        O::tally(&recorder.tallies).settle(outcome.index());
        Settled {
            recorder,
            recorded: 0,
            outcome: PhantomData,
        }
    }
}

impl<O: Outcome> Drop for Registration<O> {
    fn drop(&mut self) {
        if let Some(recorder) = self.recorder.take() {
            O::tally(&recorder.tallies).settle(O::ABANDONED.index());
        }
    }
}

impl<O: Outcome> core::fmt::Debug for Registration<O> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Registration")
            .field("settled", &self.recorder.is_none())
            .finish()
    }
}

impl<O: Outcome> Settled<O> {
    /// Count `later`, once however often it is told.
    pub fn record(&mut self, later: O::Later) {
        let index = O::later_index(later);
        let bit = 1 << index;
        if self.recorded & bit == 0 {
            self.recorded |= bit;
            O::tally(&self.recorder.tallies).record(index);
        }
    }
}

/// A collector's tallies, shared by the collector, its readers and its
/// recorders.
#[derive(Debug, Default)]
pub struct CollectorTallies {
    kernels: Tally,
    tunes: Tally,
}

impl CollectorTallies {
    fn statistics(&self) -> ExecutionStatistics {
        let kernels = self.kernels.read();
        let tunes = self.tunes.read();
        ExecutionStatistics {
            compilation: CompilationStatistics {
                registered: kernels.registered,
                compiled: kernels.settled[KernelOutcome::Compiled.index()],
                loaded: kernels.settled[KernelOutcome::Loaded.index()],
                failed: kernels.settled[KernelOutcome::Failed.index()],
                abandoned: kernels.settled[KernelOutcome::Abandoned.index()],
                refused: kernels.later[KernelOutcome::later_index(KernelHandling::Refused)],
                stored: kernels.later[KernelOutcome::later_index(KernelHandling::Stored)],
            },
            autotune: AutotuneStatistics {
                registered: tunes.registered,
                measured: tunes.settled[TuneOutcome::Measured.index()],
                failed: tunes.settled[TuneOutcome::Failed.index()],
                abandoned: tunes.settled[TuneOutcome::Abandoned.index()],
                persisted: tunes.later[TuneOutcome::later_index(TunePick::Persisted)],
            },
        }
    }
}

/// Room for every outcome a kind of work has.
const OUTCOMES: usize = 4;
/// Room for everything that can happen to a kind of work once it settled.
const LATER: usize = 2;

// Every kind of work fits its tally: a variant added past the room fails
// here rather than out of bounds on a count.
const _: () = assert!(KernelOutcome::OUTCOMES <= OUTCOMES && KernelOutcome::LATER <= LATER);
const _: () = assert!(TuneOutcome::OUTCOMES <= OUTCOMES && TuneOutcome::LATER <= LATER);

/// Items registered, how many ended in each outcome, and how many of those
/// something happened to afterwards: the one counting mechanism every kind
/// of work shares.
///
/// An item is registered before its outcome, and its outcome before
/// anything after it; a read loads them the other way round, so a read
/// never sees more settled than registered, nor more afterwards than
/// settled.
#[derive(Debug, Default)]
pub struct Tally {
    registered: AtomicUsize,
    settled: [AtomicUsize; OUTCOMES],
    later: [AtomicUsize; LATER],
}

/// A [`Tally`] as read at one moment.
struct TallyRead {
    registered: usize,
    settled: [usize; OUTCOMES],
    later: [usize; LATER],
}

impl Tally {
    fn register(&self) {
        self.registered.fetch_add(1, Ordering::Release);
    }

    fn settle(&self, outcome: usize) {
        self.settled[outcome].fetch_add(1, Ordering::Release);
    }

    fn record(&self, later: usize) {
        self.later[later].fetch_add(1, Ordering::Release);
    }

    fn read(&self) -> TallyRead {
        let later = core::array::from_fn(|index| self.later[index].load(Ordering::Acquire));
        let settled = core::array::from_fn(|index| self.settled[index].load(Ordering::Acquire));
        TallyRead {
            registered: self.registered.load(Ordering::Acquire),
            settled,
            later,
        }
    }
}

/// A kind of work a collector counts: how one item of it can end, and what
/// can happen to it after.
///
/// Sealed: it, [`CollectorTallies`] and [`Tally`] are `pub` only so the
/// public [`Registration`] can name them, in a module nothing outside the
/// crate reaches, so no kind of work is added from outside — the tallies it
/// picks are the collector's own.
#[doc(hidden)]
pub trait Outcome: Copy + 'static {
    /// What can happen to an item once it settled.
    type Later: Copy;
    /// How many outcomes it has, which a [`Tally`] has room for.
    const OUTCOMES: usize;
    /// How many things can happen to it once settled.
    const LATER: usize;
    /// The outcome of an item dropped before it settled.
    const ABANDONED: Self;

    /// Its slot in a [`Tally`]'s outcomes.
    fn index(self) -> usize;
    /// `later`'s slot in a [`Tally`]'s afterwards.
    fn later_index(later: Self::Later) -> usize;
    /// The tally that counts this kind of work.
    fn tally(tallies: &CollectorTallies) -> &Tally;
}

impl Outcome for KernelOutcome {
    type Later = KernelHandling;
    const OUTCOMES: usize = 4;
    const LATER: usize = 2;
    const ABANDONED: Self = Self::Abandoned;

    fn index(self) -> usize {
        match self {
            Self::Compiled => 0,
            Self::Loaded => 1,
            Self::Failed => 2,
            Self::Abandoned => 3,
        }
    }

    fn later_index(later: KernelHandling) -> usize {
        match later {
            KernelHandling::Refused => 0,
            KernelHandling::Stored => 1,
        }
    }

    fn tally(tallies: &CollectorTallies) -> &Tally {
        &tallies.kernels
    }
}

impl Outcome for TuneOutcome {
    type Later = TunePick;
    const OUTCOMES: usize = 3;
    const LATER: usize = 1;
    const ABANDONED: Self = Self::Abandoned;

    fn index(self) -> usize {
        match self {
            Self::Measured => 0,
            Self::Failed => 1,
            Self::Abandoned => 2,
        }
    }

    fn later_index(later: TunePick) -> usize {
        match later {
            TunePick::Persisted => 0,
        }
    }

    fn tally(tallies: &CollectorTallies) -> &Tally {
        &tallies.tunes
    }
}
