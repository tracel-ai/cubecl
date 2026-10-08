//! What the overrides opened with a [`StatisticsCollector`] triggered: the
//! kernels obtained and the autotune keys tuned, counted to it by the code
//! doing the work.

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
    /// Compiled or read from the store, then refused by the device when it
    /// was loaded: counted among [`compiled`](Self::compiled) or
    /// [`loaded`](Self::loaded) too, since those are counted as the batch
    /// obtains them, before anything is loaded.
    pub refused: usize,
}

impl CompilationStatistics {
    /// The kernels done, however they ended: never more than
    /// [`registered`](Self::registered).
    pub fn settled(&self) -> usize {
        self.compiled + self.loaded + self.failed
    }
}

/// The autotune keys tuned under a collector's overrides. A key with one
/// candidate is answered, not tuned, and counts nowhere.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AutotuneStatistics {
    /// Gathered by a [`CompileOnly`](super::ExecutionPolicy::CompileOnly)
    /// pass, or reached ungathered by a tune that measures. A key reached
    /// inside another key's candidates is not gathered: it registers if it is
    /// ever measured.
    pub registered: usize,
    /// Tuned by measuring their candidates.
    pub measured: usize,
    /// Every candidate failed: the pick was made unmeasured.
    pub failed: usize,
}

impl AutotuneStatistics {
    /// The keys done, however they ended: never more than
    /// [`registered`](Self::registered).
    pub fn settled(&self) -> usize {
        self.measured + self.failed
    }
}

/// Collects what the [overrides](super::ExecutionOverride) opened with it
/// triggered.
///
/// Not `Clone`: whoever holds it opens overrides that count into it. A
/// reader on another thread holds a [`StatisticsReader`].
#[derive(Debug)]
pub struct StatisticsCollector {
    counts: Arc<CollectorCounts>,
}

/// Reads a [`StatisticsCollector`]'s statistics from any thread, and opens
/// nothing.
#[derive(Debug, Clone)]
pub struct StatisticsReader {
    counts: Arc<CollectorCounts>,
}

/// Where the work under an override is counted: what a
/// [`KernelRegistration`] or a [`TuneRegistration`] holds. A clone counts to
/// the same collector, which is how a kernel queued under one override is
/// counted to it when its batch compiles.
#[derive(Debug, Clone)]
pub(crate) struct StatisticsRecorder {
    counts: Arc<CollectorCounts>,
}

/// One collector's counts, shared by the collector, its readers and its
/// recorders.
#[derive(Debug)]
struct CollectorCounts {
    id: usize,
    compilation: CompilationRecorder,
    autotune: AutotuneRecorder,
}

/// The id the next collector takes.
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

impl StatisticsCollector {
    /// A collector nothing has counted into yet.
    pub fn new() -> Self {
        Self {
            counts: Arc::new(CollectorCounts {
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                compilation: CompilationRecorder {
                    outcomes: Outcomes::new(),
                    refused: AtomicUsize::new(0),
                },
                autotune: AutotuneRecorder {
                    outcomes: Outcomes::new(),
                },
            }),
        }
    }

    /// What reads it from another thread.
    pub fn reader(&self) -> StatisticsReader {
        StatisticsReader {
            counts: self.counts.clone(),
        }
    }

    /// What its overrides triggered so far.
    pub fn statistics(&self) -> ExecutionStatistics {
        self.counts.statistics()
    }

    /// Where an override opened with it counts.
    pub(crate) fn recorder(&self) -> StatisticsRecorder {
        StatisticsRecorder {
            counts: self.counts.clone(),
        }
    }
}

impl Default for StatisticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticsReader {
    /// What the collector's overrides triggered so far.
    pub fn statistics(&self) -> ExecutionStatistics {
        self.counts.statistics()
    }
}

impl StatisticsRecorder {
    /// Whether it counts into `collector`.
    pub(crate) fn counts_into(&self, collector: &StatisticsCollector) -> bool {
        self.counts.id == collector.counts.id
    }
}

impl CollectorCounts {
    fn statistics(&self) -> ExecutionStatistics {
        ExecutionStatistics {
            compilation: self.compilation.read(),
            autotune: self.autotune.read(),
        }
    }
}

/// One kernel registered with the collector of the override open when it
/// was, which it settles to once: as it is compiled, read from the store or
/// fails, wherever and whenever its batch runs. Dropped before it settled —
/// its batch panicked — it settles as failed, so no registration is left
/// open for good.
///
/// Public for the kernel loader, which lives in `cubecl-server`; counting is
/// not for callers, which read their own [`StatisticsCollector`].
#[doc(hidden)]
#[derive(Debug)]
pub struct KernelRegistration {
    recorder: StatisticsRecorder,
    settled: bool,
}

impl KernelRegistration {
    /// Register one kernel with the collector of the override open now, if
    /// one is.
    pub fn register() -> Option<Self> {
        let recorder = super::recorder()?;
        recorder.counts.compilation.outcomes.register();
        Some(Self {
            recorder,
            settled: false,
        })
    }

    /// The kernel was compiled. Only the first outcome counts.
    pub fn compiled(&mut self) {
        self.settle(CompilationRecorder::COMPILED);
    }

    /// The kernel was read from the compilation store. Only the first
    /// outcome counts.
    pub fn loaded(&mut self) {
        self.settle(CompilationRecorder::LOADED);
    }

    /// The kernel failed to compile. Only the first outcome counts.
    pub fn failed(&mut self) {
        self.settle(CompilationRecorder::FAILED);
    }

    /// The device refused the kernel once it was compiled or read from the
    /// store.
    pub fn refused(&mut self) {
        let compilation = &self.recorder.counts.compilation;
        compilation.refused.fetch_add(1, Ordering::Release);
    }

    fn settle(&mut self, outcome: usize) {
        if !core::mem::replace(&mut self.settled, true) {
            self.recorder.counts.compilation.outcomes.settle(outcome);
        }
    }
}

impl Drop for KernelRegistration {
    fn drop(&mut self) {
        self.failed();
    }
}

/// One autotune key registered with the collector of the override open when
/// it was, which it settles to once its pick commits, whether or not that
/// override is still open. Dropped before it settled — a candidate panicked,
/// the plan was empty, an environment switch dropped the key — it settles as
/// failed, so no registration is left open for good.
#[derive(Debug)]
pub(crate) struct TuneRegistration {
    recorder: StatisticsRecorder,
    settled: bool,
}

impl TuneRegistration {
    /// Register one key with the collector of the override open now, if one
    /// is.
    pub(crate) fn register() -> Option<Self> {
        let recorder = super::recorder()?;
        recorder.counts.autotune.outcomes.register();
        Some(Self {
            recorder,
            settled: false,
        })
    }

    /// The key was tuned by measuring its candidates.
    pub(crate) fn measured(mut self) {
        self.settle(AutotuneRecorder::MEASURED);
    }

    /// Every candidate of the key failed.
    pub(crate) fn failed(mut self) {
        self.settle(AutotuneRecorder::FAILED);
    }

    fn settle(&mut self, outcome: usize) {
        if !core::mem::replace(&mut self.settled, true) {
            self.recorder.counts.autotune.outcomes.settle(outcome);
        }
    }
}

impl Drop for TuneRegistration {
    fn drop(&mut self) {
        self.settle(AutotuneRecorder::FAILED);
    }
}

/// Where a collector counts kernels.
#[derive(Debug)]
struct CompilationRecorder {
    outcomes: Outcomes<3>,
    /// Refusals are not outcomes: a refused kernel was already settled.
    refused: AtomicUsize,
}

impl CompilationRecorder {
    const COMPILED: usize = 0;
    const LOADED: usize = 1;
    const FAILED: usize = 2;

    fn read(&self) -> CompilationStatistics {
        // Before the outcomes, so a read never sees more refused than
        // compiled and loaded.
        let refused = self.refused.load(Ordering::Acquire);
        let (registered, settled) = self.outcomes.read();
        CompilationStatistics {
            registered,
            compiled: settled[Self::COMPILED],
            loaded: settled[Self::LOADED],
            failed: settled[Self::FAILED],
            refused,
        }
    }
}

/// Where a collector counts autotune keys.
#[derive(Debug)]
struct AutotuneRecorder {
    outcomes: Outcomes<2>,
}

impl AutotuneRecorder {
    const MEASURED: usize = 0;
    const FAILED: usize = 1;

    fn read(&self) -> AutotuneStatistics {
        let (registered, settled) = self.outcomes.read();
        AutotuneStatistics {
            registered,
            measured: settled[Self::MEASURED],
            failed: settled[Self::FAILED],
        }
    }
}

/// Items registered, and how many ended in each of `N` outcomes: the one
/// counting mechanism both recorders share.
///
/// An item is registered before its outcome, and a read loads the outcomes
/// before the registrations, so a read never sees more settled than
/// registered.
#[derive(Debug)]
struct Outcomes<const N: usize> {
    registered: AtomicUsize,
    settled: [AtomicUsize; N],
}

impl<const N: usize> Outcomes<N> {
    fn new() -> Self {
        Self {
            registered: AtomicUsize::new(0),
            settled: core::array::from_fn(|_| AtomicUsize::new(0)),
        }
    }

    fn register(&self) {
        self.registered.fetch_add(1, Ordering::Release);
    }

    fn settle(&self, outcome: usize) {
        self.settled[outcome].fetch_add(1, Ordering::Release);
    }

    fn read(&self) -> (usize, [usize; N]) {
        let settled = core::array::from_fn(|outcome| self.settled[outcome].load(Ordering::Acquire));
        (self.registered.load(Ordering::Acquire), settled)
    }
}
