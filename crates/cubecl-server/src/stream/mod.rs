/// Scheduler based multi-stream support.
pub mod scheduler;

mod allocation_planner;
pub mod base;
mod capture;
mod execute_scope;
mod failures;
mod lazy_reads;
mod stream_fence;

#[cfg(multi_threading)]
mod event;

pub use allocation_planner::{Divergence, Placement, PlannedAllocation};
pub use base::*;
pub use capture::*;
pub use execute_scope::*;
pub use failures::*;
pub use stream_fence::*;

#[cfg(multi_threading)]
pub use event::*;
