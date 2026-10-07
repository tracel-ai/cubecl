//! What the process does with the work it is asked to run, and what that
//! work triggered.
//!
//! Two levels decide what a launch does. The [`ExecutionPolicy`] is the whole
//! process's, on every device: it is where the state it governs lives — each
//! server's compile queue, the tune cache keyed by tune key, measurements
//! any other work on a device would skew, pages shared across streams. A
//! stream only knows whether its launches run, its [`StreamMode`]: the policy
//! sets every stream's default, and a measurement — autotune's candidates, a
//! throughput probe — switches the stream it measures on to
//! [`StreamMode::Execute`] while it measures. The stream never knows why.
//!
//! Under [`ExecutionPolicy::CompileOnly`] every launch and every tune's
//! candidates only queue their kernels, so a pass gathers everything a
//! workload and its tuning reach; nothing compiles until a later launch loads
//! a kernel — the first of a [`CompileAndAutotune`](ExecutionPolicy::CompileAndAutotune)
//! pass, or the first tune before it measures — which compiles the whole
//! queue in one batch. Under `CompileAndAutotune` every launch compiles its
//! kernel and is dropped, and the tunes measure. A workload then pays for
//! compilation and tuning without paying for the work that provoked them,
//! which is what makes producing a shippable environment affordable.
//!
//! An [`ExecutionOverride`] counts what it triggers into a
//! [`StatisticsCollector`], which a [`StatisticsReader`] reads from any
//! thread: the kernels obtained and the tunes measured, counted to it by the
//! code doing the work, on every device.
//!
//! **Buffers are left as they were** under either policy that drops
//! launches, so anything read back is meaningless. It only suits a pass
//! driven by the *shapes* it produces, which is what keys the caches, and
//! never one that branches on a computed value.
//!
//! The verdict is resolved where the launch is issued, on the stream it goes
//! out on, and handed to the server as a [`LaunchAction`].

use crate::client::Client;
use alloc::vec::Vec;
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::{AtomicUsize, Mutex, Ordering};

mod statistics;

pub use statistics::{
    AutotuneRecorder, AutotuneStatistics, CompilationRecorder, CompilationStatistics,
    ExecutionStatistics, StatisticsCollector, StatisticsReader, StatisticsRecorder,
};

/// What the process does with the work it is asked to run.
///
/// Each policy is a level, the number the process's open override holds in
/// its low bits; zero is none, which executes.
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
    fn stream_default(self) -> StreamMode {
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

/// Whether a stream's launches run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// They run.
    Execute,
    /// Their kernels compile — now under
    /// [`CompileAndAutotune`](ExecutionPolicy::CompileAndAutotune), queued for
    /// a batch otherwise — and they are dropped.
    Compile,
}

/// What a server does with one launch: the verdict its stream's mode and the
/// process's policy resolve to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchAction {
    /// Compile if needed, then run it. The normal case.
    Execute,
    /// Compile if needed, cache the artifact, and drop the launch.
    ///
    /// A server honoring this must still do everything a first launch does
    /// short of dispatching — expand, compile, validate, populate its caches —
    /// or the pass buys nothing.
    Compile,
    /// Queue the kernel to be compiled with others, and drop the launch.
    ///
    /// A server honoring this compiles the queue when it next loads a kernel
    /// for a launch, and only then: flushing or syncing compiles nothing, so
    /// a pass that only queues gathers everything it reaches into one batch.
    /// A kernel that fails to compile there reports it when it is launched.
    Queue,
}

impl LaunchAction {
    /// Whether the launch is dropped rather than run.
    pub fn drops_launch(self) -> bool {
        matches!(self, Self::Compile | Self::Queue)
    }
}

/// What a launch issued on `stream` does now.
pub fn launch_action(stream: StreamId) -> LaunchAction {
    let policy = policy();
    let mode = stream_mode(stream).unwrap_or(policy.stream_default());
    match (mode, policy) {
        (StreamMode::Execute, _) => LaunchAction::Execute,
        // A stream dropping its launches while the process executes queues
        // them: they compile with the next launch that loads a kernel.
        (StreamMode::Compile, ExecutionPolicy::CompileOnly | ExecutionPolicy::Execute) => {
            LaunchAction::Queue
        }
        (StreamMode::Compile, ExecutionPolicy::CompileAndAutotune) => LaunchAction::Compile,
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
/// none is open.
///
/// For the code doing that work — the kernel loader registering and
/// compiling, autotune gathering and measuring — not for a caller, which
/// reads its own [`StatisticsCollector`].
pub fn recorder() -> Option<StatisticsRecorder> {
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
                panic!("a {policy:?} override cannot open while a {open_policy:?} one is")
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

/// Sets the mode of one client's stream while it lives, restored on drop:
/// what a measurement opens, so its launches run whatever the policy drops.
///
/// Keyed on the stream the client's launches go out on — a client bound to
/// a stream of its own does not follow the thread's — so it holds wherever
/// those launches are issued from, a task resumed on another thread
/// included. Overrides of one stream nest, the newest deciding, and may
/// drop in any order.
#[derive(Debug)]
pub struct StreamModeOverride {
    id: usize,
}

/// One live [`StreamModeOverride`].
#[derive(Debug)]
struct StreamModeEntry {
    id: usize,
    stream: StreamId,
    mode: StreamMode,
}

/// How many [`StreamModeOverride`]s are live: a launch looks them up only
/// when some are.
static LIVE_STREAM_MODES: AtomicUsize = AtomicUsize::new(0);
/// The live overrides, oldest first.
static STREAM_MODES: Mutex<Vec<StreamModeEntry>> = Mutex::new(Vec::new());
/// The id the next override takes.
static NEXT_STREAM_MODE: AtomicUsize = AtomicUsize::new(0);

impl StreamModeOverride {
    /// Put `client`'s stream in `mode` until the guard drops.
    pub fn new(mode: StreamMode, client: &Client) -> Self {
        Self::on(mode, client.stream_id())
    }

    fn on(mode: StreamMode, stream: StreamId) -> Self {
        let id = NEXT_STREAM_MODE.fetch_add(1, Ordering::Relaxed);
        let mut modes = STREAM_MODES.lock();
        modes.push(StreamModeEntry { id, stream, mode });
        LIVE_STREAM_MODES.store(modes.len(), Ordering::Release);
        Self { id }
    }
}

impl Drop for StreamModeOverride {
    fn drop(&mut self) {
        let mut modes = STREAM_MODES.lock();
        modes.retain(|entry| entry.id != self.id);
        LIVE_STREAM_MODES.store(modes.len(), Ordering::Release);
    }
}

/// The mode the newest live override sets for `stream`, if one does.
fn stream_mode(stream: StreamId) -> Option<StreamMode> {
    if LIVE_STREAM_MODES.load(Ordering::Acquire) == 0 {
        return None;
    }
    STREAM_MODES
        .lock()
        .iter()
        .rev()
        .find(|entry| entry.stream == stream)
        .map(|entry| entry.mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    // `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
    // bring in itself.
    use alloc::vec;

    fn stream(value: u64) -> StreamId {
        StreamId { value }
    }

    /// The verdict is a table of a stream's mode and the policy.
    #[test]
    #[serial_test::serial]
    fn a_launch_follows_its_stream_and_the_policy() {
        let collector = StatisticsCollector::new();
        assert_eq!(launch_action(stream(7)), LaunchAction::Execute);
        {
            let _compile = ExecutionOverride::new(ExecutionPolicy::CompileOnly, &collector);
            assert_eq!(launch_action(stream(7)), LaunchAction::Queue);
        }
        {
            let _tune = ExecutionOverride::new(ExecutionPolicy::CompileAndAutotune, &collector);
            assert_eq!(launch_action(stream(7)), LaunchAction::Compile);
        }
        let _execute = ExecutionOverride::new(ExecutionPolicy::Execute, &collector);
        assert_eq!(launch_action(stream(7)), LaunchAction::Execute);
    }

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
    /// override of it, and a recorder handed out under one keeps counting
    /// there after it closes; a reader reads the same counts.
    #[test]
    #[serial_test::serial]
    fn work_counts_to_the_collector_of_the_open_override() {
        assert!(recorder().is_none(), "nothing counts outside an override");
        let collector = StatisticsCollector::new();
        let reader = collector.reader();

        let compile = ExecutionOverride::new(ExecutionPolicy::CompileOnly, &collector);
        let queued = recorder().expect("an override is open");
        (0..3).for_each(|_| queued.compilation().register());
        queued.autotune().register();
        drop(compile);

        // Kernels queued in one override compile in the next.
        let _tune = ExecutionOverride::new(ExecutionPolicy::CompileAndAutotune, &collector);
        queued.compilation().compiled();
        queued.compilation().loaded();
        queued.compilation().failed();
        recorder()
            .expect("an override is open")
            .autotune()
            .measured();

        let statistics = collector.statistics();
        assert_eq!(
            statistics.compilation,
            CompilationStatistics {
                registered: 3,
                compiled: 1,
                loaded: 1,
                failed: 1,
            }
        );
        assert_eq!(statistics.compilation.settled(), 3);
        assert_eq!(
            statistics.autotune.settled(),
            statistics.autotune.registered
        );
        assert_eq!(reader.statistics(), statistics);
        assert_eq!(
            StatisticsCollector::new().statistics(),
            ExecutionStatistics::default()
        );
    }

    /// A stream's mode overrides the policy's default for that stream alone,
    /// the newest override deciding, and overrides drop in any order.
    #[test]
    #[serial_test::serial]
    fn a_stream_mode_holds_for_its_stream() {
        let collector = StatisticsCollector::new();
        let _tune = ExecutionOverride::new(ExecutionPolicy::CompileAndAutotune, &collector);
        let measuring = StreamModeOverride::on(StreamMode::Execute, stream(1));
        assert_eq!(launch_action(stream(1)), LaunchAction::Execute);
        assert_eq!(
            launch_action(stream(2)),
            LaunchAction::Compile,
            "another stream"
        );

        let nested = StreamModeOverride::on(StreamMode::Compile, stream(1));
        assert_eq!(
            launch_action(stream(1)),
            LaunchAction::Compile,
            "the newest decides"
        );
        drop(measuring);
        assert_eq!(launch_action(stream(1)), LaunchAction::Compile);
        drop(nested);
        assert_eq!(
            launch_action(stream(1)),
            LaunchAction::Compile,
            "the policy's again"
        );
    }
}
