//! Watching what the process asks its devices to run, from the threads that ask.
//!
//! The profiling logger ([`ServerLogger`](super::ServerLogger)) already knows
//! every kernel that runs, but it formats them into a sink: durations are
//! aggregated by name into a private table and written out through a detached
//! task. That is the right shape for reading a log and the wrong one for a
//! caller that wants to *attribute* the launches: by the time a line is
//! written, the context that issued it is gone, and nothing about the ordering
//! is guaranteed against the caller's own state.
//!
//! An observer is the other half: a hook called **synchronously, on the thread
//! that issued an [`Execution`](super::Execution), before it is submitted**. That is the only
//! point where host-side context still exists, so a caller that keeps a stack
//! of what it is currently doing can pair a kernel, a collective or a transfer
//! with it.
//!
//! ```
//! use std::collections::HashMap;
//! use std::sync::{Arc, Mutex};
//!
//! use cubecl_runtime::logging::{Execution, ExecutionKind, ExecutionObservation, ExecutionObserver};
//!
//! #[derive(Default)]
//! struct CountThem(Mutex<HashMap<&'static str, usize>>);
//!
//! impl ExecutionObserver for CountThem {
//!     fn issued(&self, execution: &Execution) {
//!         if let ExecutionKind::Kernel { name, .. } = execution.kind {
//!             *self.0.lock().unwrap().entry(name).or_default() += 1;
//!         }
//!     }
//! }
//!
//! let counts = Arc::new(CountThem::default());
//! let watching = ExecutionObservation::new(counts.clone());
//! the_pass_to_attribute();
//! drop(watching);
//!
//! for (kernel, count) in counts.0.lock().unwrap().iter() {
//!     println!("{count} × {kernel}");
//! }
//! # fn the_pass_to_attribute() {}
//! ```
//!
//! # Cost
//!
//! One relaxed atomic load per execution when nothing is installed, which is
//! every ordinary run: no [`ExecutionId`](super::ExecutionId) is minted and nothing is allocated. An observer that asks for timing is the
//! expensive case, and pays per launch → [`timing`](ExecutionObserver::timing).
//!
//! # What it reports
//!
//! An execution that was **issued**, not one that finished: [`issued`] arrives
//! before it reaches the server, and a measurement, when one was asked for,
//! arrives afterwards through [`measured`]. Kernels are measured, and so are
//! communication parts and the waits on them when the observer
//! [times them too](super::Timed::KernelsAndCommunications).
//!
//! Issued is not the same as executed. A launch whose
//! [mode](crate::execution::LaunchMode) discards it is still compiled and
//! still reported here, and is then discarded instead of reaching the device; a
//! duration measured over one is the compile and the submit, with no kernel
//! under it. Which launches are discarded is decided per stream, where they are
//! issued, and is not visible here.
//!
//! A replayed [`Graph`](crate::client::Graph) is the other direction: its
//! kernels were observed once, when the capture window recorded them, and a
//! replay re-executes the whole graph without issuing them again, so an
//! observed benchmark of a graph-replayed pass reports the capture run and
//! nothing per replay.
//!
//! [`issued`]: ExecutionObserver::issued
//! [`measured`]: ExecutionObserver::measured

use alloc::sync::Arc;

use super::{ExecutionObserver, OBSERVERS};

/// Watches every execution the process issues for as long as it lives, then
/// puts back whatever it replaced.
///
/// Process-wide rather than per client: a caller attributing executions wants
/// every one its work causes, and work reaches several clients on several
/// streams. Filtering is the observer's to do, since only it knows what it is
/// attributing to.
///
/// A guard rather than an install/stop pair so the scope being attributed is
/// the scope the observer is installed for, with no restore step a caller can
/// skip on an early return. There is one slot, so a second observation replaces
/// the first for its lifetime.
///
/// # Assumptions
///
/// Observations drop in the reverse order they were taken.
///
/// # Guarantees
///
/// Once it drops, its observer is told of nothing more, and the observer it
/// replaced is installed again.
#[must_use = "an observation stops as soon as it is dropped"]
pub struct ExecutionObservation {
    previous: Option<Arc<dyn ExecutionObserver>>,
}

impl ExecutionObservation {
    /// Installs `observer` until the guard drops.
    pub fn new(observer: Arc<dyn ExecutionObserver>) -> Self {
        Self {
            previous: OBSERVERS.install(observer),
        }
    }
}

impl Drop for ExecutionObservation {
    fn drop(&mut self) {
        OBSERVERS.restore(self.previous.take());
    }
}

impl core::fmt::Debug for ExecutionObservation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExecutionObservation")
            .finish_non_exhaustive()
    }
}
