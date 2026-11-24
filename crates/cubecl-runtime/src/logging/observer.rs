use super::{Execution, Measurement, TimingRequest};

/// Notified of every execution the process issues, on the thread that issued it.
///
/// One trait for every kind of [`Execution`]: an implementer matches the kinds
/// it cares about.
///
/// Implementations must be cheap, must not launch, and must not install or drop
/// an [`ExecutionObservation`] or open an [`ExecutionScope`]: this runs inside
/// the launch path, before the work reaches the server, and holds the lock that
/// guards the installed observer.
pub trait ExecutionObserver: Send + Sync {
    /// An execution was issued.
    fn issued(&self, execution: &Execution);

    /// Whether each launch should be timed, and how its measurement reaches
    /// [`measured`](Self::measured).
    ///
    /// **[`None`](TimingRequest::None) by default, because timing is not free.**
    /// Bracketing a launch with profile markers costs the issuing thread a round
    /// trip to the server per kernel, and [`Resolved`](TimingRequest::Resolved)
    /// also blocks until the kernel has run, removing the overlap between
    /// kernels. An observer that only wants to know *which* kernels ran should
    /// leave this alone; one measuring where a pass spends its time wants
    /// [`Deferred`](TimingRequest::Deferred).
    ///
    /// Communications are issued at every level, and timed only under
    /// [`Timed::KernelsAndCommunications`].
    ///
    /// Two situations refuse the measurement without refusing the launch:
    ///
    /// * A profile the server cannot take: a graph capture window refuses
    ///   them on the spot. The kernel is still launched, still reported to
    ///   [`issued`](Self::issued), and the measurement is skipped for it, with
    ///   a warning in the log.
    /// * Don't ask for [`Resolved`](TimingRequest::Resolved) around **collective**
    ///   kernels. Reading blocks until the kernel completes, and a collective
    ///   completes only when its peers launch: a thread that issues more than
    ///   one side of a collective deadlocks waiting for the first.
    fn timing(&self) -> TimingRequest {
        TimingRequest::None
    }

    /// An execution reported to [`issued`](Self::issued) was measured.
    ///
    /// Called on the thread that issued it, after the launch, or right after
    /// [`issued`](Self::issued) for a communication part or wait, under the
    /// same lock as every other method here, so keeping the measurement must be
    /// cheap. **An observer that asked to keep measurements unread still
    /// receives [`Measurement::Read`]** whenever the profiling logger is on.
    ///
    /// It arrives *after* the launch rather than before it, so an observer
    /// pairing executions with its own state should do that in
    /// [`issued`](Self::issued) and match the [`ExecutionId`] here. A
    /// measurement goes to the observer the execution was reported to, and only
    /// while it is still installed: an observation that ends mid-read is not
    /// told.
    ///
    /// The default drops it: an observer asking for timing owes an
    /// implementation.
    #[allow(unused_variables)]
    fn measured(&self, execution: &Execution, measurement: Measurement) {}
}
