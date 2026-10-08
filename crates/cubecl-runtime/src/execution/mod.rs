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
//! out on, and handed to the server as a [`LaunchMode`].

mod base;
mod process_mode;
mod statistics;
mod stream_mode;

pub use base::LaunchMode;
pub use process_mode::{ExecutionOverride, ExecutionPolicy};
pub use statistics::{
    AutotuneStatistics, CompilationStatistics, ExecutionStatistics, KernelLoad, KernelOutcome,
    KernelRegistration, Registration, SettledKernel, StatisticsCollector, StatisticsReader,
};
pub(crate) use statistics::{StatisticsRecorder, TuneOutcome, TuneRegistration};
// Only a persisted pick names it.
#[cfg_attr(not(persistence), allow(unused_imports))]
pub(crate) use statistics::TunePick;
pub(crate) use stream_mode::ServiceStream;
pub use stream_mode::{StreamMode, StreamModeOverride};
