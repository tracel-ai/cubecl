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
    collected: Arc<Collected>,
}

/// Reads a [`StatisticsCollector`]'s statistics from any thread, and opens
/// nothing.
#[derive(Debug, Clone)]
pub struct StatisticsReader {
    collected: Arc<Collected>,
}

/// Where the code doing the work counts it: the kernel loader and autotune,
/// which reach the collector of the override open now through
/// [`recorder`](super::recorder). A clone keeps counting to the same
/// collector, which is how a kernel queued under one override is counted to
/// it when its batch compiles.
#[derive(Debug, Clone)]
pub struct StatisticsRecorder {
    collected: Arc<Collected>,
}

/// One collector's counts, shared by the collector, its readers and its
/// recorders.
#[derive(Debug)]
struct Collected {
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
            collected: Arc::new(Collected {
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                compilation: CompilationRecorder {
                    outcomes: Outcomes::new(),
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
            collected: self.collected.clone(),
        }
    }

    /// What its overrides triggered so far.
    pub fn statistics(&self) -> ExecutionStatistics {
        self.collected.statistics()
    }

    /// Where an override opened with it counts.
    pub(crate) fn recorder(&self) -> StatisticsRecorder {
        StatisticsRecorder {
            collected: self.collected.clone(),
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
        self.collected.statistics()
    }
}

impl StatisticsRecorder {
    /// Where kernels are counted.
    pub fn compilation(&self) -> &CompilationRecorder {
        &self.collected.compilation
    }

    /// Where autotune keys are counted.
    pub fn autotune(&self) -> &AutotuneRecorder {
        &self.collected.autotune
    }

    /// Whether it counts into `collector`.
    pub(crate) fn counts_into(&self, collector: &StatisticsCollector) -> bool {
        self.collected.id == collector.collected.id
    }
}

impl Collected {
    fn statistics(&self) -> ExecutionStatistics {
        ExecutionStatistics {
            compilation: self.compilation.read(),
            autotune: self.autotune.read(),
        }
    }
}

/// Where a collector counts kernels.
#[derive(Debug)]
pub struct CompilationRecorder {
    outcomes: Outcomes<3>,
}

impl CompilationRecorder {
    const COMPILED: usize = 0;
    const LOADED: usize = 1;
    const FAILED: usize = 2;

    /// One more kernel set out to be obtained.
    pub fn register(&self) {
        self.outcomes.register();
    }

    /// A registered kernel was compiled.
    pub fn compiled(&self) {
        self.outcomes.settle(Self::COMPILED);
    }

    /// A registered kernel was read from the compilation store.
    pub fn loaded(&self) {
        self.outcomes.settle(Self::LOADED);
    }

    /// A registered kernel failed to compile.
    pub fn failed(&self) {
        self.outcomes.settle(Self::FAILED);
    }

    fn read(&self) -> CompilationStatistics {
        let (registered, settled) = self.outcomes.read();
        CompilationStatistics {
            registered,
            compiled: settled[Self::COMPILED],
            loaded: settled[Self::LOADED],
            failed: settled[Self::FAILED],
        }
    }
}

/// Where a collector counts autotune keys.
#[derive(Debug)]
pub struct AutotuneRecorder {
    outcomes: Outcomes<2>,
}

impl AutotuneRecorder {
    const MEASURED: usize = 0;
    const FAILED: usize = 1;

    /// One more key set out to be tuned.
    pub fn register(&self) {
        self.outcomes.register();
    }

    /// A registered key was tuned by measuring its candidates.
    pub fn measured(&self) {
        self.outcomes.settle(Self::MEASURED);
    }

    /// Every candidate of a registered key failed.
    pub fn failed(&self) {
        self.outcomes.settle(Self::FAILED);
    }

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
