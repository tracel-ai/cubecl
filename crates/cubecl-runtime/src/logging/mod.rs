mod execution;
pub use execution::*;

mod observation;
pub use observation::*;

mod observer;
pub use observer::*;

mod timing;
pub use timing::*;

mod profiling;
pub use profiling::*;

mod scope;
pub use scope::*;

mod server;

pub use server::*;

/// Re-exported because [`Measurement`] names them: an implementor that reached
/// this trait through `cubecl` cannot otherwise spell its own arguments, and
/// `Duration` is not `core::time::Duration` on every target.
pub use cubecl_common::profile::{Duration, ProfileDuration, ProfileTicks, TimingMethod};
