use super::{StatisticsCollector, StatisticsRecorder, StreamMode};
use cubecl_environment::sync::{AtomicUsize, Mutex, Ordering};

/// What the process does with the work it is asked to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessMode {
    /// Launches run. What the process does when no override is open.
    Execute,
    /// Launches and tune candidates queue their kernels and are dropped. The
    /// queue compiles in one batch, at the next launch that loads a kernel. A
    /// tune measures and decides nothing.
    CompileOnly,
    /// Launches compile their kernels and are dropped. A tune measures its
    /// candidates for real.
    CompileAndAutotune,
}

impl ProcessMode {
    /// The process mode now.
    pub fn current() -> Self {
        OpenState::load().mode()
    }

    /// The mode every stream has under it, unless a
    /// [`StreamModeOverride`](super::StreamModeOverride) sets its own.
    pub fn stream_mode(self) -> StreamMode {
        match self {
            Self::Execute => StreamMode::Execute,
            Self::CompileOnly | Self::CompileAndAutotune => StreamMode::Discard,
        }
    }
}

/// Applies a process mode to the whole process while it lives, counting what it
/// triggers into a collector, on every thread and every device.
///
/// A build that compiles everything and then tunes with it compiled opens
/// two, one after the other, with one collector:
///
/// ```no_run
/// # fn warm_up() {}
/// use cubecl_runtime::execution::{ProcessModeOverride, ProcessMode, StatisticsCollector};
///
/// let collector = StatisticsCollector::new();
///
/// // Gather every kernel the warm-up reaches, and compile them together...
/// let compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
/// warm_up();
/// drop(compile);
///
/// // ...then tune with them compiled.
/// let _tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
/// warm_up();
///
/// let statistics = collector.statistics();
/// assert_eq!(statistics.compilation.settled(), statistics.compilation.registered);
/// ```
///
/// Overlapping overrides of one process mode and one collector compose, so one
/// opened while another is still open leaves the process mode until the last of
/// them drops. One process mode and one collector at a time: the process mode is
/// process-wide, and the work under it counts to one collector.
///
/// The process mode is read where a launch is issued, with relaxed ordering, so a
/// launch another thread had already begun issuing may still execute. What
/// is guaranteed is the launches issued by the thread that opened it, and
/// every launch issued after other threads observe it.
///
/// There is deliberately no configuration file or environment variable for
/// it. An override left open by accident turns the rest of the process into
/// launches that quietly do nothing and read back uninitialized memory, so
/// its lifetime belongs to a scope in the code that wants it.
#[derive(Debug)]
pub struct ProcessModeOverride {
    mode: ProcessMode,
}

impl ProcessModeOverride {
    /// # Panics
    ///
    /// If an override of another process mode, or of another collector, is open.
    pub fn new(mode: ProcessMode, collector: &StatisticsCollector) -> Self {
        let mut active = ACTIVE.lock();
        let open = OpenState::load();
        match Opening::of(open, active.as_ref(), mode, collector) {
            Opening::First => {
                *active = Some(collector.recorder());
                OpenState::first(mode).store();
            }
            Opening::Joins => open.joined().store(),
            Opening::OtherCollector => {
                drop(active);
                panic!("an override cannot open while one of another collector is")
            }
            Opening::OtherMode => {
                drop(active);
                panic!(
                    "an override of {mode:?} cannot open while one of {:?} is",
                    open.mode()
                )
            }
        }
        Self { mode }
    }

    /// The process mode it applies.
    pub fn mode(&self) -> ProcessMode {
        self.mode
    }

    /// Where the work the open override triggers is tallied, or `None` when
    /// none is open: what a [`Registration`](super::Registration) holds.
    ///
    /// Read without a lock first, then under [`ACTIVE`]'s: work begun as one
    /// override closes while another collector's opens on another thread
    /// counts to the one opening. Overrides follow each other under a
    /// caller's lease, so nothing starts work in that gap.
    pub(crate) fn active_recorder() -> Option<StatisticsRecorder> {
        if !OpenState::load().is_open() {
            return None;
        }
        ACTIVE.lock().clone()
    }
}

impl Drop for ProcessModeOverride {
    fn drop(&mut self) {
        // The last guard out closes the override, its process mode and its
        // collector with it.
        let mut active = ACTIVE.lock();
        let left = OpenState::load().left();
        if !left.is_open() {
            *active = None;
        }
        left.store();
    }
}

/// What opening an override meets.
enum Opening {
    /// Nothing open: it opens.
    First,
    /// One of its process mode and its collector: it joins.
    Joins,
    /// One of another collector.
    OtherCollector,
    /// One of its collector, of another process mode.
    OtherMode,
}

impl Opening {
    fn of(
        open: OpenState,
        active: Option<&StatisticsRecorder>,
        mode: ProcessMode,
        collector: &StatisticsCollector,
    ) -> Self {
        let Some(active) = active.filter(|_| open.is_open()) else {
            return Self::First;
        };
        if !active.tallies_into(collector) {
            Self::OtherCollector
        } else if open.mode() != mode {
            Self::OtherMode
        } else {
            Self::Joins
        }
    }
}

/// The open override as the process holds it: its process mode's level in the low
/// [`LEVEL_BITS`](Self::LEVEL_BITS), and above them how many guards hold it
/// open. Written only under [`ACTIVE`]'s lock, and read without it on every
/// launch.
#[derive(Debug, Clone, Copy)]
struct OpenState(usize);

/// [`OpenState`] as the process holds it.
static OPEN: AtomicUsize = AtomicUsize::new(0);
/// Where the open override tallies, while one is.
static ACTIVE: Mutex<Option<StatisticsRecorder>> = Mutex::new(None);

impl OpenState {
    /// The bits that hold the process mode's level.
    const LEVEL_BITS: u32 = 2;
    /// Masks the state down to the process mode's level.
    const LEVEL_MASK: usize = (1 << Self::LEVEL_BITS) - 1;
    /// One guard, in the state's count.
    const GUARD: usize = 1 << Self::LEVEL_BITS;

    fn load() -> Self {
        Self(OPEN.load(Ordering::Relaxed))
    }

    fn store(self) {
        OPEN.store(self.0, Ordering::Relaxed);
    }

    /// One guard holding `mode` open. Every process mode has a level of its own,
    /// [`Execute`](ProcessMode::Execute) included, so an override of it
    /// is told apart from none: it tallies what a real run compiles and
    /// tunes.
    fn first(mode: ProcessMode) -> Self {
        let level = match mode {
            ProcessMode::CompileOnly => 1,
            ProcessMode::CompileAndAutotune => 2,
            ProcessMode::Execute => 3,
        };
        Self(Self::GUARD | level)
    }

    fn is_open(self) -> bool {
        self.0 != 0
    }

    /// The open process mode; none open executes.
    fn mode(self) -> ProcessMode {
        match self.0 & Self::LEVEL_MASK {
            1 => ProcessMode::CompileOnly,
            2 => ProcessMode::CompileAndAutotune,
            _ => ProcessMode::Execute,
        }
    }

    /// One more guard.
    fn joined(self) -> Self {
        Self(self.0 + Self::GUARD)
    }

    /// One guard fewer, closed with the last.
    fn left(self) -> Self {
        match self.0 - Self::GUARD {
            rest if rest < Self::GUARD => Self(0),
            rest => Self(rest),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::statistics::{KernelLoad, KernelOutcome, TuneOutcome, TunePick};
    use super::super::{CompilationStatistics, KernelRegistration, TuneRegistration};
    use super::*;
    // `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
    // bring in itself.
    use alloc::vec;
    use alloc::vec::Vec;

    /// One process mode is open at a time: an override of another refuses to open
    /// beside it.
    #[test]
    #[serial_test::serial]
    #[should_panic(
        expected = "an override of CompileOnly cannot open while one of CompileAndAutotune is"
    )]
    fn policies_do_not_overlap() {
        let collector = StatisticsCollector::new();
        let _tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
        let _compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
    }

    /// The work under an override counts to one collector: another's
    /// refuses to open beside it, whatever its process mode.
    #[test]
    #[serial_test::serial]
    #[should_panic(expected = "cannot open while one of another collector is")]
    fn two_collectors_do_not_overlap() {
        let (first, second) = (StatisticsCollector::new(), StatisticsCollector::new());
        let _first = ProcessModeOverride::new(ProcessMode::CompileOnly, &first);
        let _second = ProcessModeOverride::new(ProcessMode::CompileOnly, &second);
    }

    /// A refused open leaves the open override as it was.
    #[test]
    #[serial_test::serial]
    fn a_refused_open_leaves_the_open_process_mode() {
        let collector = StatisticsCollector::new();
        let tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
        let refused = std::panic::catch_unwind(|| {
            ProcessModeOverride::new(ProcessMode::CompileOnly, &collector)
        });
        assert!(refused.is_err());
        assert_eq!(ProcessMode::current(), ProcessMode::CompileAndAutotune);
        drop(tune);
        assert_eq!(ProcessMode::current(), ProcessMode::Execute);
    }

    /// Overlapping overrides of one process mode and collector compose: the last
    /// out closes it.
    #[test]
    #[serial_test::serial]
    fn overrides_nest() {
        let collector = StatisticsCollector::new();
        {
            let _outer = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
            {
                let _inner = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
            }
            assert_eq!(
                ProcessMode::current(),
                ProcessMode::CompileOnly,
                "the outer is in force"
            );
        }
        assert_eq!(ProcessMode::current(), ProcessMode::Execute);
        assert!(ProcessModeOverride::active_recorder().is_none());
    }

    /// What an override triggers counts to its collector, over every
    /// override of it — an `Execute` one included — and a registration made
    /// under one settles there after it closes; what happens to settled work
    /// counts once; a reader reads the same counts.
    #[test]
    #[serial_test::serial]
    fn work_counts_to_the_collector_of_the_open_override() {
        assert!(
            KernelRegistration::register().is_none(),
            "nothing counts outside an override"
        );
        let collector = StatisticsCollector::new();
        let reader = collector.reader();

        let compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
        let queued: Vec<KernelRegistration> = (0..4)
            .filter_map(|_| KernelRegistration::register())
            .collect();
        let gathered = TuneRegistration::register().expect("an override is open");
        drop(compile);

        // Kernels queued in one override compile in the next.
        let _tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
        let mut queued = queued.into_iter();
        let mut compiled = queued.next().unwrap().settle(KernelOutcome::Compiled);
        compiled.record(KernelLoad::Refused);
        compiled.record(KernelLoad::Refused);
        let mut loaded = queued.next().unwrap().settle(KernelOutcome::Loaded);
        loaded.record(KernelLoad::Stored);
        let _failed = queued.next().unwrap().settle(KernelOutcome::Failed);
        let mut measured = gathered.settle(TuneOutcome::Measured);
        measured.record(TunePick::Persisted);

        let statistics = collector.statistics();
        assert_eq!(
            statistics.compilation,
            CompilationStatistics {
                registered: 4,
                compiled: 1,
                loaded: 1,
                failed: 1,
                refused: 1,
                stored: 1,
            },
            "a refusal told twice counts once"
        );
        assert_eq!(
            statistics.autotune.settled(),
            statistics.autotune.registered
        );
        assert_eq!(statistics.autotune.persisted, 1);
        assert_eq!(reader.statistics(), statistics);
        drop(_tune);

        let execute = ProcessModeOverride::new(ProcessMode::Execute, &collector);
        let _measured = TuneRegistration::register()
            .expect("an execute override counts too")
            .settle(TuneOutcome::Measured);
        drop(execute);
        assert_eq!(collector.statistics().autotune.measured, 2);
        // The fourth kernel, never settled, fails as it drops.
        drop(queued);
        assert_eq!(collector.statistics().compilation.failed, 2);
    }

    /// A registration dropped before it settled — its batch or its tune
    /// panicked — settles as failed: none stays open.
    #[test]
    #[serial_test::serial]
    fn a_dropped_registration_settles_as_failed() {
        let collector = StatisticsCollector::new();
        let _compile = ProcessModeOverride::new(ProcessMode::CompileOnly, &collector);
        drop(KernelRegistration::register());
        drop(TuneRegistration::register());
        let statistics = collector.statistics();
        assert_eq!(
            (
                statistics.compilation.failed,
                statistics.compilation.settled()
            ),
            (1, 1)
        );
        assert_eq!(
            (statistics.autotune.failed, statistics.autotune.settled()),
            (1, 1)
        );
    }
}
