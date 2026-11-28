use super::{Execution, Measurement, TimingRequest};

/// Notified of every execution the process issues, on the thread that issued it.
///
/// One trait for every kind of [`Execution`]: an implementer matches the kinds
/// it cares about. Every method runs inside the launch path, before the work
/// reaches the server, under the lock that guards the installed observer, so
/// each must be cheap.
///
/// # Assumptions
///
/// No method launches, installs or drops an
/// [`ExecutionObservation`](super::ExecutionObservation), or opens an
/// [`ExecutionScope`](super::ExecutionScope).
pub trait ExecutionObserver: Send + Sync {
    /// Observes an execution as it is issued, before it is submitted: the
    /// moment to pair it with whatever the issuing thread is doing.
    fn observe(&self, execution: &Execution);

    /// Whether each launch should be timed, and how its measurement reaches
    /// [`observe_measurement`](Self::observe_measurement). Asked as each
    /// execution is issued, so an observer may answer differently from one
    /// execution to the next.
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
    /// [`TimedExecutions::KernelsAndCommunications`](super::TimedExecutions::KernelsAndCommunications).
    ///
    /// A profile the server cannot take, as a graph capture window refuses
    /// them, skips the measurement and not the launch: the kernel is still
    /// observed, and the log says so.
    ///
    /// Don't ask for [`Resolved`](TimingRequest::Resolved) around
    /// **collective** kernels. Reading blocks until the kernel completes, and a
    /// collective completes only when its peers launch: a thread that issues
    /// more than one side of a collective deadlocks waiting for the first.
    fn timing(&self) -> TimingRequest {
        TimingRequest::None
    }

    /// Observes the measurement of an execution [observed](Self::observe)
    /// before, delivered as the answer [`timing`](Self::timing) gave when it
    /// was issued.
    ///
    /// Called on the thread that issued it, after the launch, or right after
    /// [`observe`](Self::observe) for a communication part or wait.
    ///
    /// It arrives *after* the launch rather than before it, so an observer
    /// pairing executions with its own state should do that in
    /// [`observe`](Self::observe) and match the
    /// [`ExecutionId`](super::ExecutionId) here. An observation that ends
    /// mid-read is not told.
    ///
    /// The default drops it: an observer asking for timing owes an
    /// implementation.
    #[allow(unused_variables)]
    fn observe_measurement(&self, execution: &Execution, measurement: Measurement) {}
}
