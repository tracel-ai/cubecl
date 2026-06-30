/// Scheduler based multi-stream support.
pub mod scheduler;

mod base;
mod handle;

#[cfg(multi_threading)]
mod event;

pub use base::*;
pub use handle::Stream;

#[cfg(multi_threading)]
pub use event::*;
