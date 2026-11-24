use super::{Duration, ProfileDuration, TimingMethod};

/// What an observer asks be done with each launch's measurement.
///
/// Answered by [`timing`](super::ExecutionObserver::timing) as each execution
/// is issued: the launch path has to know where a measurement will go *before*
/// it takes one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimingRequest {
    /// Don't time launches. The default, because timing is not free.
    #[default]
    None,
    /// Time each launch and read its measurement back, delivering it as
    /// [`Measurement::Read`].
    ///
    /// Reading blocks the issuing thread until the kernel has run, so kernels
    /// run one at a time and their sum is not the pass's device time. Right for
    /// each kernel's own cost, wrong for measuring a pipeline.
    Resolved,
    /// Time each launch and hand its measurement over **unread**, as
    /// [`Measurement::Unread`], for the executions [`TimedExecutions`] names.
    ///
    /// Nothing waits, so the kernels around it keep running back to back and
    /// the observer reads them once the pass is over. Right for measuring where
    /// a pass spends its time.
    ///
    /// **Only while the profiling logger is off.** A measurement is read once,
    /// and past [`ExecutionOnly`](super::ProfileLevel::ExecutionOnly) the logger
    /// needs it too, so the launch path reads it there and delivers
    /// [`Measurement::Read`] instead, which runs the kernels one at a time. An
    /// observer whose figures only mean something on a pipelined pass should
    /// say so when timings arrive that way.
    Deferred(TimedExecutions),
}

impl TimingRequest {
    /// Whether launches are timed.
    pub fn times_launches(self) -> bool {
        self != Self::None
    }

    /// Whether communications are timed too, and so their waits issued.
    pub fn times_communications(self) -> bool {
        self == Self::Deferred(TimedExecutions::KernelsAndCommunications)
    }
}

/// Which executions a [`Deferred`](TimingRequest::Deferred) request times.
///
/// Only a deferred request can time a communication: reading a part back blocks until its peers
/// have run theirs, so a thread that queues several would wait forever on the first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimedExecutions {
    /// Kernels only.
    Kernels,
    /// Kernels, and each communication: its parts, and the compute streams' waits on them,
    /// issued as [`Wait`](super::ExecutionKind::Wait)s at this level only.
    ///
    /// A part's or a wait's measurement arrives unread right after it is
    /// issued, and is read on its device's clock. A part runs from when its
    /// inputs are written to when it ends, the time it spends waiting for its
    /// peers included. A wait runs from when its compute stream reaches it to
    /// when what it awaits ends, and is zero when that had already ended. A
    /// host-staged transfer blocks the issuing thread, so it is measured on the
    /// host, as [`TimingMethod::System`].
    ///
    /// A part costs its device one timing event more than an untimed one: the
    /// events that order it against the compute stream are its marks. One
    /// inside a graph capture, or refused before it ran, reads back empty.
    KernelsAndCommunications,
}

/// A launch's measurement, as an observer asked for it.
pub enum Measurement {
    /// Not yet read back, under [`TimingRequest::Deferred`]. Keep it, and read
    /// it once the work being measured is over.
    ///
    /// Where the backend times on the device without waiting (CUDA, HIP, and
    /// wgpu's timestamp queries) the measurement is two events in the stream
    /// that nothing has waited for, and the resolved [`ProfileTicks`](super::ProfileTicks) carry the
    /// window's start and end on one clock, so an observer can also say where
    /// the device sat idle between kernels. Metal waits for the window and
    /// places it at the moment it was read: its lengths are device time, its
    /// starts and ends do not line up, and keeping the measurement saves
    /// nothing there.
    ///
    /// Reading it, [`ProfileDuration::resolve`], blocks until the device has
    /// reached both events: doing that inside
    /// [`observe_measurement`](super::ExecutionObserver::observe_measurement) would hold the observer's lock
    /// for the length of the kernel, which is the thing
    /// [`Deferred`](TimingRequest::Deferred) exists to avoid.
    Unread(ProfileDuration),
    /// A kernel's, read back by the launch path: under
    /// [`TimingRequest::Resolved`], and under [`TimingRequest::Deferred`]
    /// whenever the profiling logger reads them as well.
    Read {
        /// How long the kernel took.
        duration: Duration,
        /// How it was measured, which is not a detail.
        ///
        /// A backend falls back to [`System`](TimingMethod::System) where it
        /// cannot get a device timestamp (wgpu does exactly that once the
        /// timestamp-query budget is spent) and a system timing is host wall
        /// around a blocking submit, which includes submission, sync, and the
        /// kernel's compilation on its first launch, rather than the kernel.
        /// The two are not the same measurement, and an observer reporting them
        /// as one shows a number that moves several-fold between runs.
        method: TimingMethod,
    },
}
