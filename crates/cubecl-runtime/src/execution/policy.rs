use super::{StatisticsCollector, StatisticsRecorder, StreamMode};
use cubecl_environment::sync::{AtomicUsize, Mutex, Ordering};

/// What the process does with the work it is asked to run.
///
/// Each policy is a level, the number the process's open override holds in
/// its low bits; zero is none, which executes. [`Execute`](Self::Execute) is
/// a level of its own rather than zero so that an override of it is told
/// apart from none: it counts what a real run compiles and tunes into its
/// collector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum ExecutionPolicy {
    /// Launches run. What the process does when no override is open.
    Execute = 3,
    /// Launches and tune candidates queue their kernels and are dropped. The
    /// queue compiles in one batch, at the next launch that loads a kernel. A
    /// tune measures and decides nothing.
    CompileOnly = 1,
    /// Launches compile their kernels and are dropped. A tune measures its
    /// candidates for real.
    CompileAndAutotune = 2,
}

impl ExecutionPolicy {
    /// The policy at `level`: none open executes.
    fn at(level: usize) -> Self {
        match level {
            1 => Self::CompileOnly,
            2 => Self::CompileAndAutotune,
            _ => Self::Execute,
        }
    }

    /// The mode a stream has under it, unless a [`StreamModeOverride`] says
    /// otherwise.
    pub(super) fn stream_default(self) -> StreamMode {
        match self {
            Self::Execute => StreamMode::Execute,
            Self::CompileOnly | Self::CompileAndAutotune => StreamMode::Compile,
        }
    }

    /// Whether launches on a stream it sets are dropped.
    pub fn drops_launches(self) -> bool {
        self.stream_default() == StreamMode::Compile
    }
}

/// The open override: its policy's level in the low [`LEVEL_BITS`], and
/// above them how many guards hold it open. Written only under [`ACTIVE`]'s
/// lock, and read without it on every launch.
static OPEN: AtomicUsize = AtomicUsize::new(0);
/// The bits of [`OPEN`] that hold the policy's level.
const LEVEL_BITS: u32 = 2;
/// Masks [`OPEN`] down to the policy's level.
const LEVEL_MASK: usize = (1 << LEVEL_BITS) - 1;
/// One guard, in [`OPEN`]'s count.
const GUARD: usize = 1 << LEVEL_BITS;

/// Where the open override counts, while one is.
static ACTIVE: Mutex<Option<StatisticsRecorder>> = Mutex::new(None);

/// The process's policy now.
pub fn policy() -> ExecutionPolicy {
    ExecutionPolicy::at(OPEN.load(Ordering::Relaxed) & LEVEL_MASK)
}

/// Where the work the open override triggers is counted, or `None` when
/// none is open: what a [`KernelRegistration`] or a [`TuneRegistration`]
/// holds.
///
/// Read without a lock first, then under [`ACTIVE`]'s: work begun as one
/// override closes while another collector's opens on another thread counts
/// to the one opening. Overrides follow each other under a caller's lease,
/// so nothing starts work in that gap.
pub(crate) fn recorder() -> Option<StatisticsRecorder> {
    if OPEN.load(Ordering::Relaxed) == 0 {
        return None;
    }
    ACTIVE.lock().clone()
}

/// Applies a policy to the whole process while it lives, counting what it
/// triggers into a collector, on every thread and every device.
///
/// A build that compiles everything and then tunes with it compiled opens
/// two, one after the other, with one collector:
///
/// ```no_run
/// # fn warm_up() {}
/// use cubecl_runtime::execution::{ExecutionOverride, ExecutionPolicy, StatisticsCollector};
///
/// let collector = StatisticsCollector::new();
///
/// // Gather every kernel the warm-up reaches, and compile them together...
/// let compile = ExecutionOverride::new(ExecutionPolicy::CompileOnly, &collector);
/// warm_up();
/// drop(compile);
///
/// // ...then tune with them compiled.
/// let _tune = ExecutionOverride::new(ExecutionPolicy::CompileAndAutotune, &collector);
/// warm_up();
///
/// let statistics = collector.statistics();
/// assert_eq!(statistics.compilation.settled(), statistics.compilation.registered);
/// ```
///
/// Overlapping overrides of one policy and one collector compose, so one
/// opened while another is still open leaves the policy until the last of
/// them drops. One policy and one collector at a time: the policy is
/// process-wide, and the work under it counts to one collector.
///
/// The policy is read where a launch is issued, with relaxed ordering, so a
/// launch another thread had already begun issuing may still execute. What
/// is guaranteed is the launches issued by the thread that opened it, and
/// every launch issued after other threads observe it.
///
/// There is deliberately no configuration file or environment variable for
/// it. An override left open by accident turns the rest of the process into
/// launches that quietly do nothing and read back uninitialized memory, so
/// its lifetime belongs to a scope in the code that wants it.
#[derive(Debug)]
pub struct ExecutionOverride {
    policy: ExecutionPolicy,
}

impl ExecutionOverride {
    /// # Panics
    ///
    /// If an override of another policy, or of another collector, is open.
    pub fn new(policy: ExecutionPolicy, collector: &StatisticsCollector) -> Self {
        let level = policy as usize;
        let mut active = ACTIVE.lock();
        let open = OPEN.load(Ordering::Relaxed);
        let open_level = open & LEVEL_MASK;
        if open_level == 0 {
            *active = Some(collector.recorder());
            OPEN.store(GUARD | level, Ordering::Relaxed);
            return Self { policy };
        }
        let same_collector = active
            .as_ref()
            .is_some_and(|recorder| recorder.counts_into(collector));
        let open_policy = ExecutionPolicy::at(open_level);
        drop(active);
        match (open_level == level, same_collector) {
            (true, true) => {
                OPEN.fetch_add(GUARD, Ordering::Relaxed);
                Self { policy }
            }
            (_, false) => panic!("an override cannot open while one of another collector is"),
            (false, true) => {
                let (policy, open_policy) = (Article(policy), Article(open_policy));
                panic!("{policy} override cannot open while {open_policy} one is")
            }
        }
    }

    /// The policy it applies.
    pub fn policy(&self) -> ExecutionPolicy {
        self.policy
    }
}

impl Drop for ExecutionOverride {
    fn drop(&mut self) {
        // The last guard out closes the override, its policy and its
        // collector with it.
        let mut active = ACTIVE.lock();
        let open = OPEN.load(Ordering::Relaxed) - GUARD;
        if open < GUARD {
            OPEN.store(0, Ordering::Relaxed);
            *active = None;
        } else {
            OPEN.store(open, Ordering::Relaxed);
        }
    }
}

/// A policy with its article, for a panic message.
struct Article(ExecutionPolicy);

impl core::fmt::Display for Article {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            ExecutionPolicy::Execute => write!(f, "an {:?}", self.0),
            policy => write!(f, "a {policy:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{CompilationStatistics, KernelRegistration, TuneRegistration};
    use super::*;
    // `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
    // bring in itself.
    use alloc::vec;
    use alloc::vec::Vec;

    /// One policy is open at a time: an override of another refuses to open
    /// beside it.
    #[test]
    #[serial_test::serial]
    #[should_panic(
        expected = "a CompileOnly override cannot open while a CompileAndAutotune one is"
    )]
    fn policies_do_not_overlap() {
        let collector = StatisticsCollector::new();
        let _tune = ExecutionOverride::new(ExecutionPolicy::CompileAndAutotune, &collector);
        let _compile = ExecutionOverride::new(ExecutionPolicy::CompileOnly, &collector);
    }

    /// The work under an override counts to one collector: another's
    /// refuses to open beside it, whatever its policy.
    #[test]
    #[serial_test::serial]
    #[should_panic(expected = "cannot open while one of another collector is")]
    fn two_collectors_do_not_overlap() {
        let (first, second) = (StatisticsCollector::new(), StatisticsCollector::new());
        let _first = ExecutionOverride::new(ExecutionPolicy::CompileOnly, &first);
        let _second = ExecutionOverride::new(ExecutionPolicy::CompileOnly, &second);
    }

    /// A refused open leaves the open override as it was.
    #[test]
    #[serial_test::serial]
    fn a_refused_open_leaves_the_open_policy() {
        let collector = StatisticsCollector::new();
        let tune = ExecutionOverride::new(ExecutionPolicy::CompileAndAutotune, &collector);
        let refused = std::panic::catch_unwind(|| {
            ExecutionOverride::new(ExecutionPolicy::CompileOnly, &collector)
        });
        assert!(refused.is_err());
        assert_eq!(policy(), ExecutionPolicy::CompileAndAutotune);
        drop(tune);
        assert_eq!(policy(), ExecutionPolicy::Execute);
    }

    /// Overlapping overrides of one policy and collector compose: the last
    /// out closes it.
    #[test]
    #[serial_test::serial]
    fn overrides_nest() {
        let collector = StatisticsCollector::new();
        {
            let _outer = ExecutionOverride::new(ExecutionPolicy::CompileOnly, &collector);
            {
                let _inner = ExecutionOverride::new(ExecutionPolicy::CompileOnly, &collector);
            }
            assert_eq!(
                policy(),
                ExecutionPolicy::CompileOnly,
                "the outer is in force"
            );
        }
        assert_eq!(policy(), ExecutionPolicy::Execute);
        assert!(recorder().is_none());
    }

    /// What an override triggers counts to its collector, over every
    /// override of it — an `Execute` one included — and a registration made
    /// under one settles there after it closes; a reader reads the same
    /// counts.
    #[test]
    #[serial_test::serial]
    fn work_counts_to_the_collector_of_the_open_override() {
        assert!(
            KernelRegistration::register().is_none(),
            "nothing counts outside an override"
        );
        let collector = StatisticsCollector::new();
        let reader = collector.reader();

        let compile = ExecutionOverride::new(ExecutionPolicy::CompileOnly, &collector);
        let mut queued: Vec<KernelRegistration> = (0..4)
            .filter_map(|_| KernelRegistration::register())
            .collect();
        let gathered = TuneRegistration::register().expect("an override is open");
        drop(compile);

        // Kernels queued in one override compile in the next.
        let _tune = ExecutionOverride::new(ExecutionPolicy::CompileAndAutotune, &collector);
        queued[0].compiled();
        queued[0].refused();
        queued[1].stored();
        queued[1].loaded();
        queued[2].failed();
        queued[2].compiled();
        let mut gathered = gathered;
        gathered.measured();
        gathered.persisted();

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
            "only a kernel's first outcome counts"
        );
        assert_eq!(
            statistics.autotune.settled(),
            statistics.autotune.registered
        );
        assert_eq!(reader.statistics(), statistics);
        drop(_tune);

        let execute = ExecutionOverride::new(ExecutionPolicy::Execute, &collector);
        TuneRegistration::register()
            .expect("an execute override counts too")
            .measured();
        assert_eq!(collector.statistics().autotune.persisted, 1);
        drop(execute);
        assert_eq!(collector.statistics().autotune.measured, 2);
    }

    /// A registration dropped before it settled — its batch or its tune
    /// panicked — settles as failed: none stays open.
    #[test]
    #[serial_test::serial]
    fn a_dropped_registration_settles_as_failed() {
        let collector = StatisticsCollector::new();
        let _compile = ExecutionOverride::new(ExecutionPolicy::CompileOnly, &collector);
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
