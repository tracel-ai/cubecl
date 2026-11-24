mod execution;
mod installer;
mod observation;
mod observer;
mod profiling;
mod reported;
mod scope;
mod server;
mod timing;

/// Re-exported because [`Measurement`] names them: an implementor that reached
/// this trait through `cubecl` cannot otherwise spell its own arguments, and
/// `Duration` is not `core::time::Duration` on every target.
pub use cubecl_common::profile::{Duration, ProfileDuration, ProfileTicks, TimingMethod};
pub use execution::{
    CollectivePath, Communication, Execution, ExecutionId, ExecutionKind, ExecutionPlace,
    StreamRole, TransferRoute, WaitTarget,
};
pub use observation::ExecutionObservation;
pub use observer::ExecutionObserver;
pub use profiling::ProfileLevel;
pub use scope::ExecutionScope;
pub use server::ServerLogger;
pub use timing::{Measurement, Timed, TimingRequest};

pub(crate) use installer::{issue, times_communications, timing_requested};

pub(crate) use reported::ReportedExecution;

use installer::OBSERVERS;
use profiling::ProfileAggregator;
