/// Scheduler based multi-stream support.
pub mod scheduler;

pub mod base;
mod capture;
mod failures;
mod write_scope;

#[cfg(multi_threading)]
mod event;

pub use base::*;
pub use capture::*;
pub use failures::*;
pub use write_scope::*;

#[cfg(multi_threading)]
pub use event::*;
